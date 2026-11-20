# Second Look: LLMs as a Quality Control Tool to Detect Missannotated Data Points

A framework for evaluating language models on different tasks.

## Requirements

- Python 3.x
- API keys for the model providers you intend to use

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sara-Rajaee/lingo_reason
   cd your-repo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your API keys as environment variables:
   ```bash
   export TOGETHER_API_KEY=your_key_here
   export GEMINI_API_KEY=your_key_here
   export OPENAI_API_KEY=your_key_here
   ```
   You only need to set the keys for the model providers you plan to use.

   For GPT-OSS models via a local vLLM proxy:
   ```bash
   export GPT_OSS_API_BASE=http://localhost:19743/v1/   # proxy URL (default)
   export GPT_OSS_API_KEY=no-key                        # optional, proxy usually has no auth
   ```
4. (optional) register polymath repo as a module (needed for polymath evaluation)
   ```bash
   git submodule add https://github.com/QwenLM/PolyMath.git third_party/PolyMath
   git submodule update --init --recursive
   ```
## Serving and Evaluating GPT-OSS Models on Snellius

GPT-OSS (and other HuggingFace) models are served locally via [vLLM](https://github.com/vllm-project/vllm) and evaluated in a single SLURM job using `scripts/snellius_eval.sh`. The script starts a vLLM server and runs `run.py` against it on the same node, so no manual server/eval handoff is needed.

**Prerequisites:** A `second-look` conda environment with vLLM installed, and GPU(s) available on the allocated node.

```bash
sbatch scripts/snellius_eval.sh <vllm_model_id_or_path> <eval_model_name> <task> [extra run.py args...]

# Example: serve gpt-oss-120b and evaluate it on polymath
sbatch scripts/snellius_eval.sh openai/gpt-oss-120b gpt-oss-120b polymath
```

- `<vllm_model_id_or_path>` — the model to serve (HuggingFace ID or local checkpoint path)
- `<eval_model_name>` — the `--model` name as registered in `config/models.yaml`
- `<task>` — the `--task` name as registered in `config/tasks.yaml`
- any extra arguments are forwarded to `run.py`

The vLLM server exposes an OpenAI-compatible API on `localhost` (port `19743` by default, override with `PORT`) and tensor-parallel size auto-detects from the GPUs allocated to the job. The server is shut down automatically once the evaluation finishes.

## Usage

Run an evaluation by specifying a model and a task:

```bash
# API models
python run.py --model gemini-2.5-flash --task polymath

# Local models (require a running vLLM server)
python run.py --model qwen3.5-4b --task polymath
python run.py --model qwen3.5-4b --task polymath --no-reasoning   # thinking switched off
python run.py --model gpt-oss-20b --task mmmlu
```

On the ILPS cluster each local model has a submission script that serves it with vLLM
and runs `run.py` against it in one job (see `scripts/run_model/_serve_and_eval.sh`):

```bash
sbatch scripts/run_model/qwen3.5-4b.sh mmmlu,polymath [--no-reasoning]
```

To see all available models and tasks:

```bash
python run.py --list
```

To collect high-temperature samples for distillation:

```bash
python run.py --model gpt-oss-120b --task mgsm --distillation --distillation-samples 32
```

Distillation runs write to `distilled_results/<model_name>/<task>/<subset>/` with `sampled_outputs.json` for every sampled output and `gold_outputs.json` for one gold-matching output per example. The gold file keeps the original task columns plus the selected reasoning trace.

For distillation sampling params, add `distillation_temperature` and `distillation_top_p` under the task's `defaults` in `config/tasks.yaml`. If they are not set, distillation falls back to the task's normal `temperature` and `top_p`.

## Configuration

Task configurations (e.g. languages, number of evaluation samples) can be modified in the config files located in the `config/` directory.
