import re
from pathlib import Path
from litellm import acompletion
from .base_provider import BaseProvider

# Matches <think>...</think> blocks (DeepSeek-R1 style, also used as fallback)
THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Qwen3-Thinking emits only the closing </think> (opening tag is in chat template)
THINK_CLOSE_ONLY_RE = re.compile(r"^(.*?)</think>\s*(.*)$", re.DOTALL)
# Gemma 4 uses an asymmetric channel marker instead of <think>: <|channel>thought ... <channel|>
GEMMA_CHANNEL_RE = re.compile(r"<\|channel>thought\s*(.*?)\s*<channel\|>", re.DOTALL)


class GptOssProvider(BaseProvider):
    """GPT-OSS provider via liteLLM proxy (vLLM-hosted models).

    Routes requests through a local OpenAI-compatible API server (e.g. vLLM).
    Model names are prefixed with ``hosted_vllm/`` for liteLLM routing.
    Reasoning traces are extracted via the ``reasoning_content`` field
    (OpenAI reasoning API) or ``<think>`` tags as fallback.

    Provider config (config/providers.yaml):
      - ``api_base``: proxy URL, default ``http://localhost:19743/v1/``
      - ``api_key``: optional, defaults to ``"no-key"`` (local proxy usually needs no auth)
    """

    def __init__(self, config):
        super().__init__(config)
        self.api_base = config.get("api_base", "http://localhost:19743/v1/")
        self.api_key = config.get("api_key", "no-key")

    @staticmethod
    def _extract_reasoning(message):
        """Extract reasoning trace and final answer from a model response message.

        Tries multiple strategies in order:
          1. ``reasoning_content`` field (OpenAI reasoning API)
          2. ``<think>...</think>`` tags in content (DeepSeek-R1 style)
          2b. Closing-only ``</think>`` (Qwen3-Thinking style)
          2c. ``<|channel>thought ... <channel|>`` markers (Gemma 4 style)
          3. Treat entire content as the answer (no reasoning detected)
        """
        raw_content = message.content or ""

        # Strategy 1: reasoning_content field (OpenAI reasoning API)
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            return {
                "reasoning": reasoning_content,
                "generation": raw_content,
                "raw_generation": raw_content,
            }

        # Strategy 2: <think>...</think> tags
        think_match = THINK_TAG_RE.search(raw_content)
        if think_match:
            reasoning = think_match.group(1).strip()
            generation = THINK_TAG_RE.sub("", raw_content).strip()
            return {
                "reasoning": reasoning,
                "generation": generation,
                "raw_generation": raw_content,
            }

        # Strategy 2b: only closing </think> (Qwen3-Thinking)
        if "</think>" in raw_content:
            close_match = THINK_CLOSE_ONLY_RE.match(raw_content)
            if close_match:
                return {
                    "reasoning": close_match.group(1).strip(),
                    "generation": close_match.group(2).strip(),
                    "raw_generation": raw_content,
                }

        # Strategy 2c: Gemma 4 channel markers
        channel_match = GEMMA_CHANNEL_RE.search(raw_content)
        if channel_match:
            reasoning = channel_match.group(1).strip()
            generation = GEMMA_CHANNEL_RE.sub("", raw_content).strip()
            return {
                "reasoning": reasoning,
                "generation": generation,
                "raw_generation": raw_content,
            }

        # Strategy 3: no reasoning detected
        return {
            "reasoning": None,
            "generation": raw_content,
            "raw_generation": raw_content,
        }

    @staticmethod
    def _model_system_content(path):
        # System prompt a model card requires (config/models.yaml: system_prompt_file).
        # Ministral reasoning only reasons with its card's prompt, whose
        # [THINK]...[/THINK] part has to be sent as a thinking chunk.
        text = Path(path).read_text(encoding="utf-8")
        start, end = text.find("[THINK]"), text.find("[/THINK]")
        if start < 0 or end < 0:
            return [{"type": "text", "text": text}]
        return [
            {"type": "text", "text": text[:start]},
            {"type": "thinking", "thinking": text[start + len("[THINK]"):end], "closed": True},
            {"type": "text", "text": text[end + len("[/THINK]"):]},
        ]

    async def generate(self, model_id, prompt, params, system_prompt=None, reasoning_effort=None, thinking_budget=None):
        """Generate completion via liteLLM proxy (async)."""

        async def _generate():
            messages = []
            if params.get("system_prompt_file"):
                content = self._model_system_content(params["system_prompt_file"])
                if system_prompt:
                    content.append({"type": "text", "text": "\n\n" + system_prompt})
                messages.append({"role": "system", "content": content})
            elif system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            kwargs = dict(
                model=f"hosted_vllm/{model_id}",
                messages=messages,
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=params.get("temperature", 0),
                max_tokens=params.get("max_tokens", 4096),
                top_p=params.get("top_p", 1),
                num_retries=self.max_retries,
                timeout=self.timeout,
            )
            # enable_thinking is sent only to models whose chat template has that switch
            # (thinking_switch in config/models.yaml): Mistral tokenizers answer any
            # chat_template_kwargs with HTTP 400.
            if params.get("thinking_switch"):
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(params.get("reasoning", False))}}

            response = await acompletion(**kwargs)
            output = self._extract_reasoning(response.choices[0].message)
            # finish_reason "length" marks an output cut off at max_tokens, which was
            # invisible in the saved results; usage gives the prompt and output lengths.
            output["finish_reason"] = response.choices[0].finish_reason
            output["usage"] = {"prompt_tokens": response.usage.prompt_tokens,
                               "completion_tokens": response.usage.completion_tokens}
            return output

        return await self._retry_with_backoff(_generate)
