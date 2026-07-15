#!/usr/bin/env python3
"""
Estimate the cost of a TRANSLATION-QA run over multilingual benchmarks.

Mode: for every non-English item we build a reasoning-model input tuple:

    (src, tgt, gold)                      if the gold label is language-invariant
    (src, tgt, gold_src, gold_tgt)        if the gold material was translated too

where
    src      = English prompt material (question + options + ...)
    tgt      = target-language prompt material (aligned by id)
    gold     = answer key / final answer / CoT chain / constraint spec

Per-dataset gold handling:
    global_mmlu / mmmlu     : answer KEY should be invariant, but dataset defects
                              exist -> include both source and target keys so the
                              QA report can detect a changed keyed answer.
    mmlu_prox               : same as above, plus translated CoT chains
                              -> (answer, cot_src, cot_tgt). Strongest check.
    polymath                : final answer is language-invariant -> (src, tgt, gold),
                              aggregated across all 4 difficulty splits.
    global_piqa_parallel    : keyed solution + distractors are translated
                              -> both sides of all candidate solutions included.
    m_arenahard_v2          : EXCLUDED (outlier: no gold -> pure (src, tgt)).
    maxife                  : constraint specs are translated -> both sides of
                              the instruction/constraint text included as "gold"
                              (the checkable artifact), base question has no gold.

Outputs per dataset and per English-to-target language pair for three explicit
reasoning budgets (1k, 5k, and 20k), each with a 1k visible-report allowance.
The Markdown export retains unavailable pairs instead of silently dropping them.

Usage:
    pip install datasets tiktoken pandas tqdm
    python scripts/cost_estimation/estimate_cost.py --help

    # example
    python scripts/cost_estimation/estimate_cost.py --quick --allow-partial-languages \
        --price-in 3 --price-out 15 \
        --reasoning-budgets 1000 5000 20000 --visible-output-tokens 1000 \
        --sample-n 300 --sample-seed 42 \
        --tokenizer o200k_base \
        --summary-markdown outputs/costs.md \
        --datasets polymath global_piqa_parallel

--quick takes a deterministic random sample per language and extrapolates its mean.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Any

import pandas as pd
import tiktoken
from datasets import load_dataset, load_dataset_builder, get_dataset_config_names
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Run configuration: all knobs are argparse params (see main / --help).
# ----------------------------------------------------------------------------
@dataclass
class RunConfig:
    price_in: float             # $/1M input tokens
    price_out: float            # $/1M output tokens (reasoning bills as output)
    reasoning_budgets: tuple[int, ...]
    visible_output_tokens: int
    sample_n: int               # rows sampled per language in --quick mode
    sample_seed: int
    instruction: str
    model_id: str
    tokenizer_name: str
    enc: object                 # tiktoken encoding instance

    def ntok(self, x) -> int:
        if x is None:
            return 0
        if isinstance(x, (list, tuple)):
            return sum(self.ntok(i) for i in x)
        return len(self.enc.encode(str(x), disallowed_special=()))


_ENGLISH_INDEX_CACHE = {}


def build_run_config(
    price_in: float,
    price_out: float,
    reasoning_budgets: list[int] | tuple[int, ...],
    visible_output_tokens: int,
    sample_n: int,
    sample_seed: int,
    instruction: str,
    model_id: str,
    tokenizer: str,
) -> RunConfig:
    """Construct a RunConfig from serializable CLI parameters."""
    return RunConfig(
        price_in=price_in,
        price_out=price_out,
        reasoning_budgets=tuple(reasoning_budgets),
        visible_output_tokens=visible_output_tokens,
        sample_n=sample_n,
        sample_seed=sample_seed,
        instruction=instruction,
        model_id=model_id,
        tokenizer_name=tokenizer,
        enc=tiktoken.get_encoding(tokenizer),
    )


def run_config_to_kwargs(cfg: RunConfig, tokenizer: str) -> dict:
    """Convert a RunConfig back into serializable constructor kwargs."""
    return {
        "price_in": cfg.price_in,
        "price_out": cfg.price_out,
        "reasoning_budgets": list(cfg.reasoning_budgets),
        "visible_output_tokens": cfg.visible_output_tokens,
        "sample_n": cfg.sample_n,
        "sample_seed": cfg.sample_seed,
        "instruction": cfg.instruction,
        "model_id": cfg.model_id,
        "tokenizer": tokenizer,
    }


# This is the exact instruction included in token accounting. Keep it in sync
# with the eventual production request rather than estimating a fixed overhead.
DEFAULT_QA_INSTRUCTION = (
    "Compare the English source and translated target. Check meaning, answer "
    "choices, and supplied gold material. Return a concise Markdown error "
    "report with severity and evidence; return 'No issues' when appropriate."
)

SHARED_LANGUAGE_SET = ("ar", "zh", "en", "fr", "de", "id", "it", "ja", "ko", "pt", "ru", "es", "vi")
TARGET_LANGUAGE_SET = tuple(code for code in SHARED_LANGUAGE_SET if code != "en")


# ----------------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    key: str
    hf_path: str
    revision: str
    split: str | list[str]
    id_field: str                     # join key for src<->tgt alignment
    english_config: str               # config name of the English pivot
    key_normalizer: Callable[[str, dict, int], Any] | None = None
    source_row_normalizer: Callable[[dict], dict] | None = None
    target_embedded_source_builder: Callable[[dict], dict] | None = None
    # fields forming the PROMPT side (translated): question + options etc.
    prompt_fields: list = field(default_factory=list)
    prompt_list_fields: list = field(default_factory=list)
    # fields forming the GOLD side:
    gold_invariant_fields: list = field(default_factory=list)   # counted once
    gold_translated_fields: list = field(default_factory=list)  # counted on BOTH sides
    gold_translated_list_fields: list = field(default_factory=list)
    configs: list = None              # None -> auto-discover
    language_configs: dict[str, str] = field(default_factory=dict)
    # Some datasets have no English config inside the repo (MMMLU): the English
    # source must come from an external dataset.
    external_english: dict = None     # includes its own immutable revision
    notes: str = ""


def polymath_key_normalizer(split: str, row: dict, i: int):
    """Normalize PolyMath ids so parallel language rows share one join key."""
    raw_key = row.get("id", i)
    if not isinstance(raw_key, str):
        raise KeyError(
            f"polymath/{split}: expected string id, got {type(raw_key).__name__}"
        )
    match = re.match(r"^(top|high|medium|low)-[^-]+-(\d+)$", raw_key)
    if not match:
        raise KeyError(
            f"polymath/{split}: unexpected PolyMath id format {raw_key!r}"
        )
    return (match.group(1), int(match.group(2)))


def mmmlu_key_normalizer(split: str, row: dict, i: int):
    """Use global row order as the canonical MMMLU join key."""
    return (split, i)


def normalize_mmmlu_english_row(row: dict) -> dict:
    """Map external cais/mmlu fields onto MMMLU's translated schema."""
    choices = row.get("choices")
    if not isinstance(choices, (list, tuple)) or len(choices) != 4:
        raise ValueError("mmmlu English pivot: expected exactly four choices")
    answer = row.get("answer")
    if answer not in range(4):
        raise ValueError(f"mmmlu English pivot: invalid answer index {answer!r}")
    # MMMLU uses letters and title-cased fields; normalizing here prevents the
    # English prompt and gold from being silently counted as zero tokens.
    return {
        "Question": row.get("question"),
        "A": choices[0],
        "B": choices[1],
        "C": choices[2],
        "D": choices[3],
        "Answer": "ABCD"[answer],
        "Subject": row.get("subject"),
    }


def global_piqa_parallel_key_normalizer(split: str, row: dict, i: int):
    """Normalize Global PIQA example ids across languages."""
    raw_key = row.get("example_id", i)
    if not isinstance(raw_key, str):
        raise KeyError(
            "global_piqa_parallel/"
            f"{split}: expected string example_id, got {type(raw_key).__name__}"
        )
    language = row.get("language")
    if isinstance(language, str) and raw_key.endswith(f"_{language}"):
        return (split, raw_key[:-(len(language) + 1)])
    return (split, raw_key)


def global_piqa_parallel_source_row(row: dict):
    """Build the English source side from embedded fields in a target row."""
    return {
        "prompt": row.get("eng_prompt"),
        "solution0": row.get("eng_solution0"),
        "solution1": row.get("eng_solution1"),
        "solution2": row.get("eng_solution2"),
        "solution3": row.get("eng_solution3"),
        "label": row.get("label"),
        "example_id": row.get("example_id"),
        "language": "eng_latn",
    }


SPECS = {
    "global_mmlu": DatasetSpec(
        key="global_mmlu",
        hf_path="CohereLabs/Global-MMLU",
        revision="0e619dbeb34206cd48705a1a0ea7fb21cae09993",
        split="test",
        id_field="sample_id",
        english_config="en",
        prompt_fields=["question", "option_a", "option_b", "option_c", "option_d"],
        gold_invariant_fields=["answer"],   # letter key A-D, invariant
        language_configs={code: code for code in TARGET_LANGUAGE_SET},
        # option texts already included on both sides via prompt_fields,
        # so answer-key validity is checkable from the tuple as-is.
        notes="41 target langs x ~14k. Tuple = (srcQ+opts, tgtQ+opts, key). "
              "Provenance prior: 16 MT langs = highest expected yield.",
    ),
    "mmmlu": DatasetSpec(
        key="mmmlu",
        hf_path="openai/MMMLU",
        revision="325a01dc3e173cac1578df94120499aaca2e2504",
        split="test",
        id_field="Unnamed: 0",
        english_config=None,    # repo has no English config
        # MMMLU uses locale-style HF config names, and `Unnamed: 0` is only a
        # per-subject counter. We therefore join against `cais/mmlu` by the
        # full test-set row order, which we verified matches on subject,
        # options, and answer index after A/B/C/D -> 0/1/2/3 conversion.
        key_normalizer=mmmlu_key_normalizer,
        source_row_normalizer=normalize_mmmlu_english_row,
        configs=["AR_XY", "BN_BD", "DE_DE", "ES_LA", "FR_FR", "HI_IN",
                 "ID_ID", "IT_IT", "JA_JP", "KO_KR", "PT_BR", "SW_KE",
                 "YO_NG", "ZH_CN"],
        prompt_fields=["Question", "A", "B", "C", "D"],
        gold_invariant_fields=["Answer"],
        language_configs={
            "ar": "AR_XY", "zh": "ZH_CN", "fr": "FR_FR", "de": "DE_DE",
            "id": "ID_ID", "it": "IT_IT", "ja": "JA_JP", "ko": "KO_KR",
            "pt": "PT_BR", "es": "ES_LA",
        },
        external_english={
            "hf_path": "cais/mmlu", "config": "all", "split": "test",
            "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        },
        notes="14 target langs x 14,042. English source pulled from cais/mmlu "
              "and joined by verified global row order.",
    ),
    "mmlu_prox": DatasetSpec(
        key="mmlu_prox",
        hf_path="li-lab/MMLU-ProX",
        revision="8e6106a6c6ce1c5027e66cc338143cf997b2aa09",
        split="test",
        id_field="question_id",
        english_config="en",
        prompt_fields=["question", "option_0", "option_1", "option_2", "option_3",
                       "option_4", "option_5", "option_6", "option_7", "option_8",
                       "option_9"],
        gold_invariant_fields=["answer", "answer_index"],
        # The public test split has empty cot_content, so test-run costing must
        # not claim that translated CoT is present. Validation is a separate run.
        gold_translated_fields=[],
        language_configs={code: code for code in TARGET_LANGUAGE_SET},
        notes="28 target langs x 11,759 test rows. Test CoT is unavailable; "
              "costing covers questions, options, and answer keys only.",
    ),
    "mmlu_prox_lite": DatasetSpec(
        key="mmlu_prox_lite",
        hf_path="li-lab/MMLU-ProX-Lite",
        revision="e82aafb9460529687d3c7e51b401d8dd1dd309dd",
        split="test",
        id_field="question_id",
        english_config="en",
        prompt_fields=["question", "option_0", "option_1", "option_2", "option_3",
                       "option_4", "option_5", "option_6", "option_7", "option_8",
                       "option_9"],
        gold_invariant_fields=["answer", "answer_index"],
        gold_translated_fields=[],
        language_configs={code: code for code in TARGET_LANGUAGE_SET},
        notes="Lite: 28 target langs x 588 test rows. Test CoT is unavailable.",
    ),
    "polymath": DatasetSpec(
        key="polymath",
        hf_path="Qwen/PolyMath",
        revision="71e0db902e0ece05e208dfd8b1695bd3d95cf130",
        split=["top", "high", "medium", "low"],
        id_field="id",
        english_config="en",
        key_normalizer=polymath_key_normalizer,
        prompt_fields=["question"],
        gold_invariant_fields=["answer"],  # language-invariant number/expression
        language_configs={code: code for code in TARGET_LANGUAGE_SET},
        notes="17 target langs x 500 across 4 difficulty splits. Tuple = "
              "(srcQ, tgtQ, gold). Free extra check: gold must be "
              "byte-identical across languages per id.",
    ),
    # NOTE: m-ArenaHard-v2 intentionally excluded: it's the outlier of the set
    # (no gold of any kind -> pure (src, tgt) mode, different QA rubric).
    # Re-add later as a gold-less spec if needed.
    "global_piqa_parallel": DatasetSpec(
        key="global_piqa_parallel",
        hf_path="mrlbenchmarks/global-piqa-parallel",
        revision="b0b18516a8bc2cb1106bce3dd4db32848ca715ea",
        split="test",
        id_field="example_id",
        english_config="eng_latn",
        key_normalizer=global_piqa_parallel_key_normalizer,
        target_embedded_source_builder=global_piqa_parallel_source_row,
        prompt_fields=["prompt", "solution0", "solution1", "solution2", "solution3"],
        gold_invariant_fields=["label"],   # index of correct solution
        language_configs={
            "ar": "arb_arab", "zh": "cmn_hans", "fr": "fra_latn_fran",
            "de": "deu_latn", "id": "ind_latn", "it": "ita_latn",
            "ja": "jpn_jpan", "ko": "kor_hang", "pt": "por_latn_braz",
            "ru": "rus_cyrl", "es": "spa_latn_spai", "vi": "vie_latn",
        },
        notes="130 target varieties x ~103. Source-side English prompt and "
              "solutions come from each target row's embedded eng_* fields, "
              "which preserve candidate ordering across languages.",
    ),
    # NOTE: MaXIFE is intentionally excluded for now. The current HF path does
    # not resolve via `datasets`, and the public repository exposes the data
    # through GitHub files rather than a verified Hub dataset entry. Re-add it
    # once we implement a repository-backed loader or confirm an official HF ID.
}


# ----------------------------------------------------------------------------
# Validation and token accounting for one aligned (src_row, tgt_row) pair
# ----------------------------------------------------------------------------
def require_fields(row: dict, fields: list[str], context: str) -> None:
    """Reject schema drift instead of turning missing fields into zero tokens."""
    missing = [name for name in fields if name not in row]
    if missing:
        raise KeyError(f"{context}: missing required fields {missing}")


def serialized_input(
    src_row: dict,
    tgt_row: dict,
    spec: DatasetSpec,
    cfg: RunConfig,
    target_iso: str,
) -> str:
    """Return the exact JSON request body whose input tokens are estimated."""
    prompt_fields = spec.prompt_fields + spec.prompt_list_fields
    require_fields(src_row, prompt_fields, f"{spec.key}/en")
    require_fields(tgt_row, prompt_fields, f"{spec.key}/{target_iso}")
    require_fields(src_row, spec.gold_invariant_fields, f"{spec.key}/en gold")
    require_fields(tgt_row, spec.gold_invariant_fields, f"{spec.key}/{target_iso} gold")
    translated_gold = spec.gold_translated_fields + spec.gold_translated_list_fields
    require_fields(src_row, translated_gold, f"{spec.key}/en gold")
    require_fields(tgt_row, translated_gold, f"{spec.key}/{target_iso} gold")

    # Include both values even though these fields should be invariant. Real
    # translation datasets contain key defects; hiding the target key or aborting
    # would prevent the downstream QA task from reporting exactly that problem.
    gold = {
        name: {"source": src_row[name], "target": tgt_row[name]}
        for name in spec.gold_invariant_fields
    }
    for name in spec.gold_translated_fields + spec.gold_translated_list_fields:
        gold[f"{name}_source"] = src_row[name]
        gold[f"{name}_target"] = tgt_row[name]

    payload = {
        "instruction": cfg.instruction,
        "source_language": "en",
        "target_language": target_iso,
        "source": {name: src_row[name] for name in prompt_fields},
        "target": {name: tgt_row[name] for name in prompt_fields},
        "gold": gold,
    }
    # Compact, stable JSON makes the accounting reproducible across machines.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def tuple_tokens(src_row, tgt_row, spec, cfg: RunConfig, target_iso: str) -> int:
    return cfg.ntok(serialized_input(src_row, tgt_row, spec, cfg, target_iso))


def has_invariant_gold_mismatch(src_row: dict, tgt_row: dict, spec: DatasetSpec) -> bool:
    """Flag changed answer keys/labels without treating them as join failures."""
    return any(src_row.get(name) != tgt_row.get(name) for name in spec.gold_invariant_fields)


def mismatch_example_id(spec: DatasetSpec, row: dict, key: Any) -> str:
    """Prefer the dataset's human-readable ID, falling back to the join key."""
    raw_id = row.get(spec.id_field)
    # MMMLU's displayed counter is not globally unique, so its canonical row
    # position is safer for locating a reported mismatch.
    if raw_id is not None and spec.key != "mmmlu":
        return str(raw_id)
    return str(key)


def cot_content_tokens(src_row, tgt_row, spec, cfg: RunConfig) -> int:
    """Return the token contribution from cot_content fields alone."""
    if "cot_content" not in spec.gold_translated_fields:
        return 0
    return cfg.ntok(src_row.get("cot_content")) + cfg.ntok(tgt_row.get("cot_content"))


def iter_splits(spec: DatasetSpec) -> list[str]:
    """Return the dataset splits as a concrete list."""
    if isinstance(spec.split, str):
        return [spec.split]
    return list(spec.split)


def canonical_row_key(spec: DatasetSpec, split: str, row: dict, i: int):
    """Build the canonical alignment key for one dataset row."""
    if spec.key_normalizer is not None:
        return spec.key_normalizer(split, row, i)

    raw_key = row.get(spec.id_field, i)
    return (split, raw_key)


# ----------------------------------------------------------------------------
# Loading / alignment
# ----------------------------------------------------------------------------
def load_english_index(spec) -> dict:
    """Return {canonical_alignment_key: row} for the English pivot."""
    idx = {}
    splits = iter_splits(spec)

    if spec.english_config is not None:
        for split in splits:
            ds = load_dataset(
                spec.hf_path, spec.english_config, split=split, revision=spec.revision
            )
            for i, row in enumerate(ds):
                key = canonical_row_key(spec, split, row, i)
                if key in idx:
                    raise ValueError(f"{spec.key}/{split}: duplicate English key {key!r}")
                idx[key] = row
    elif spec.external_english is not None:
        if len(splits) != 1:
            raise ValueError(
                f"{spec.key}: external English pivot does not support multiple splits"
            )
        ee = spec.external_english
        ds = load_dataset(
            ee["hf_path"], ee["config"], split=ee["split"], revision=ee["revision"]
        )
        for i, row in enumerate(ds):
            key = canonical_row_key(spec, splits[0], row, i)
            if key in idx:
                raise ValueError(f"{spec.key}/{splits[0]}: duplicate English key {key!r}")
            normalized = spec.source_row_normalizer(row) if spec.source_row_normalizer else row
            idx[key] = normalized
    else:
        raise ValueError(f"{spec.key}: no English pivot is configured")

    if not idx:
        raise ValueError(f"{spec.key}: English pivot loaded zero rows")
    return idx


def get_cached_english_index(spec: DatasetSpec) -> dict:
    """Load and cache the English pivot once per process."""
    if spec.target_embedded_source_builder is not None:
        return {}
    if spec.key not in _ENGLISH_INDEX_CACHE:
        _ENGLISH_INDEX_CACHE[spec.key] = load_english_index(spec)
    return _ENGLISH_INDEX_CACHE[spec.key]


def source_row_for_target(spec: DatasetSpec, split: str, row: dict, i: int, en_idx: dict):
    """Return the source-side row aligned to a target-side row."""
    if spec.target_embedded_source_builder is not None:
        return spec.target_embedded_source_builder(row)

    key = canonical_row_key(spec, split, row, i)
    src = en_idx.get(key)
    if src is None:
        raise KeyError(
            f"{spec.key}/{row.get('language', 'target')}/{split}: "
            f"missing English pivot for key {key!r}"
        )
    return src


def deterministic_sample_positions(total: int, sample_n: int, seed_text: str) -> set[int]:
    """Choose stable random row positions without disturbing alignment order."""
    if sample_n >= total:
        return set(range(total))
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    import random
    return set(random.Random(seed).sample(range(total), sample_n))


def deterministic_reservoir_sample(rows, sample_n: int, seed_text: str):
    """Sample a stream with unknown length while preserving original indices."""
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    import random
    rng = random.Random(seed)
    reservoir = []
    total = 0
    for i, row in enumerate(rows):
        total = i + 1
        if len(reservoir) < sample_n:
            reservoir.append((i, row))
            continue
        replacement = rng.randint(0, i)
        if replacement < sample_n:
            reservoir[replacement] = (i, row)
    return sorted(reservoir, key=lambda item: item[0]), total


def measure_language(
    spec: DatasetSpec,
    iso_language: str,
    dataset_config: str,
    quick: bool,
    cfg: RunConfig,
    en_idx: dict,
):
    """Measure one target language/config for a dataset."""
    splits = iter_splits(spec)
    lang_tokens, lang_cot_tokens, n_pairs = 0, 0, 0
    observed_pairs, observed_gold_mismatches = 0, 0
    gold_mismatch_examples = []

    for split in splits:
        if quick:
            ds = load_dataset(
                spec.hf_path, dataset_config, split=split,
                streaming=True, revision=spec.revision,
            )
            toks, cot_toks, sampled_rows = [], [], 0
            total_rows = get_split_size(
                spec.hf_path, dataset_config, split, spec.revision
            )
            if total_rows is None:
                # Some HF configs (notably MMMLU) omit split-size metadata.
                # Reservoir sampling scans the stream but retains only sample_n
                # rows and their original indices, which MMMLU needs for joining.
                selected_rows, total_rows = deterministic_reservoir_sample(
                    ds, cfg.sample_n,
                    f"{cfg.sample_seed}:{spec.key}:{dataset_config}:{split}",
                )
            else:
                positions = deterministic_sample_positions(
                    total_rows, cfg.sample_n,
                    f"{cfg.sample_seed}:{spec.key}:{dataset_config}:{split}",
                )
                selected_rows = []
                for i, row in enumerate(ds):
                    if i in positions:
                        selected_rows.append((i, row))
                    if len(selected_rows) == len(positions):
                        break
                if len(selected_rows) != len(positions):
                    raise ValueError(
                        f"{spec.key}/{dataset_config}/{split}: metadata says "
                        f"{total_rows} rows but only {len(selected_rows)}/"
                        f"{len(positions)} samples were found"
                    )

            for i, row in selected_rows:
                src = source_row_for_target(spec, split, row, i, en_idx)
                toks.append(tuple_tokens(src, row, spec, cfg, iso_language))
                cot_toks.append(cot_content_tokens(src, row, spec, cfg))
                mismatch = has_invariant_gold_mismatch(src, row, spec)
                observed_gold_mismatches += mismatch
                if mismatch and len(gold_mismatch_examples) < 5:
                    key = canonical_row_key(spec, split, row, i)
                    gold_mismatch_examples.append(mismatch_example_id(spec, row, key))
                observed_pairs += 1
                sampled_rows += 1
            mean_tok = sum(toks) / max(len(toks), 1)
            mean_cot_tok = sum(cot_toks) / max(len(cot_toks), 1)
            lang_tokens += int(mean_tok * total_rows)
            lang_cot_tokens += int(mean_cot_tok * total_rows)
            n_pairs += total_rows
        else:
            ds = load_dataset(
                spec.hf_path, dataset_config, split=split, revision=spec.revision
            )
            seen_keys = set()
            for i, row in enumerate(ds):
                key = canonical_row_key(spec, split, row, i)
                if key in seen_keys:
                    raise ValueError(
                        f"{spec.key}/{dataset_config}/{split}: duplicate target key {key!r}"
                    )
                seen_keys.add(key)
                src = source_row_for_target(spec, split, row, i, en_idx)
                lang_tokens += tuple_tokens(src, row, spec, cfg, iso_language)
                lang_cot_tokens += cot_content_tokens(src, row, spec, cfg)
                mismatch = has_invariant_gold_mismatch(src, row, spec)
                observed_gold_mismatches += mismatch
                if mismatch and len(gold_mismatch_examples) < 5:
                    gold_mismatch_examples.append(mismatch_example_id(spec, row, key))
                observed_pairs += 1
                n_pairs += 1
            if spec.target_embedded_source_builder is None:
                english_keys = {key for key in en_idx if key[0] == split}
                if seen_keys != english_keys:
                    # A one-sided join would undercount a language even if every
                    # target row happened to find an English source row.
                    raise ValueError(
                        f"{spec.key}/{dataset_config}/{split}: alignment cardinality "
                        f"differs (target={len(seen_keys)}, English={len(english_keys)})"
                    )

    return {
        "iso_language": iso_language,
        "dataset_config": dataset_config,
        "language_pair": f"en-{iso_language}",
        "pairs": n_pairs,
        "in_tokens": lang_tokens,
        "cot_tokens": lang_cot_tokens,
        "avg_in_tok": round(lang_tokens / max(n_pairs, 1)),
        "gold_mismatches_observed": observed_gold_mismatches,
        "rows_observed": observed_pairs,
        "gold_mismatch_examples": gold_mismatch_examples,
        "gold_mismatch_rate": round(
            observed_gold_mismatches / max(observed_pairs, 1), 6
        ),
    }


def get_split_size(
    hf_path: str, config_name: str, split: str, revision: str
) -> int:
    """Return the split size from metadata when available, else None."""
    builder = load_dataset_builder(hf_path, config_name, revision=revision)
    if builder.info.splits is None:
        return None
    split_info = builder.info.splits.get(split)
    if split_info is None or split_info.num_examples is None:
        return None
    return split_info.num_examples


def resolve_language_configs(
    spec: DatasetSpec,
    requested_isos: list[str] | tuple[str, ...],
    available_configs: list[str] | tuple[str, ...],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Resolve public ISO codes to verified dataset configs without guessing."""
    available = set(available_configs)
    resolved, missing = [], []
    for iso in requested_isos:
        config = spec.language_configs.get(iso)
        if config is None or config not in available:
            missing.append(iso)
        else:
            resolved.append((iso, config))
    return resolved, missing


def measure_dataset(
    spec: DatasetSpec,
    targets: list[tuple[str, str]],
    quick: bool,
    cfg: RunConfig,
    lang_jobs: int = 1,
):
    if not targets:
        return targets, 0, 0, 0, []
    if lang_jobs == 1:
        en_idx = get_cached_english_index(spec)
        per_lang = [
            measure_language(spec, iso, config, quick, cfg, en_idx)
            for iso, config in tqdm(targets, desc=spec.key)
        ]
    else:
        en_idx = get_cached_english_index(spec)
        max_workers = min(lang_jobs, len(targets), os.cpu_count() or 1)
        per_lang = []
        # Use threads here: HF dataset config loading is mostly I/O-bound, and
        # sharing one in-process datasets cache avoids per-process config-cache
        # mismatches like MMMLU loading AR_XY and JA_JP concurrently.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    measure_language, spec, iso, config, quick, cfg, en_idx
                ): iso
                for iso, config in targets
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=spec.key):
                per_lang.append(future.result())
        per_lang.sort(key=lambda row: row["iso_language"])

    total_pairs = sum(row["pairs"] for row in per_lang)
    total_in = sum(row["in_tokens"] for row in per_lang)
    total_cot = sum(row["cot_tokens"] for row in per_lang)
    return targets, total_pairs, total_in, total_cot, per_lang


def add_pair_costs(per_lang: list[dict], cfg: RunConfig) -> list[dict]:
    """Compute costs from each measured pair row, never aggregate division."""
    for row in per_lang:
        for budget in cfg.reasoning_budgets:
            total_output = budget + cfg.visible_output_tokens
            cost = (
                row["in_tokens"] * cfg.price_in
                + row["pairs"] * total_output * cfg.price_out
            ) / 1e6
            row[f"cost_reasoning_{budget}_usd"] = round(cost, 2)
    return per_lang


def cost_rows(spec, per_lang, in_tokens, cot_tokens, cfg: RunConfig):
    out = []
    n_pairs = sum(row["pairs"] for row in per_lang)
    for budget in cfg.reasoning_budgets:
        out_tok = budget + cfg.visible_output_tokens
        total = (in_tokens * cfg.price_in
                 + n_pairs * out_tok * cfg.price_out) / 1e6
        cot_input_cost = (cot_tokens * cfg.price_in) / 1e6
        out.append({
            "dataset": spec.key,
            "target_languages": len(per_lang),
            "pairs_total": n_pairs,
            "input_tokens_M": round(in_tokens / 1e6, 2),
            "cot_input_tokens_M": round(cot_tokens / 1e6, 2),
            "avg_in_tok/pair": round(in_tokens / max(n_pairs, 1)),
            "scenario": f"reasoning_{budget}",
            "reasoning_tok/pair": budget,
            "visible_tok/pair": cfg.visible_output_tokens,
            "model_id": cfg.model_id,
            "tokenizer_proxy": cfg.tokenizer_name,
            "input_$_per_MTok": cfg.price_in,
            "output_$_per_MTok": cfg.price_out,
            "sample_n": cfg.sample_n,
            "sample_seed": cfg.sample_seed,
            "dataset_revision": spec.revision,
            "english_source_revision": (
                spec.external_english["revision"] if spec.external_english else spec.revision
            ),
            "out_tok/pair": out_tok,
            "TOTAL_$": round(total, 2),
            "COT_INPUT_$": round(cot_input_cost, 2),
            "TOTAL_$_WITHOUT_COT": round(total - cot_input_cost, 2),
            "notes": spec.notes,
        })
    return out


def process_dataset(
    dataset_key: str,
    targets: list[tuple[str, str]],
    missing_isos: list[str],
    quick: bool,
    cfg_kwargs: dict,
    lang_jobs: int,
):
    """Measure one dataset and return its summary rows plus per-language details."""
    cfg = build_run_config(**cfg_kwargs)
    spec = SPECS[dataset_key]
    _, n_pairs, in_tokens, cot_tokens, per_lang = measure_dataset(
        spec, targets, quick, cfg, lang_jobs
    )
    per_lang = add_pair_costs(per_lang, cfg)
    if n_pairs == 0:
        return {
            "dataset": dataset_key,
            "records": [],
            "per_lang": per_lang,
            "missing_isos": missing_isos,
            "skipped": True,
        }
    records = cost_rows(spec, per_lang, in_tokens, cot_tokens, cfg)
    for record in records:
        record["sampling"] = (
            f"deterministic-random-n={cfg.sample_n}" if quick else "full-split"
        )
    return {
        "dataset": dataset_key,
        "records": records,
        "per_lang": per_lang,
        "missing_isos": missing_isos,
        "skipped": False,
    }


def render_markdown(
    results: list[dict],
    cfg: RunConfig,
    args: argparse.Namespace,
) -> str:
    """Render auditable per-pair costs, including unavailable language pairs."""
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    budget_headers = [f"{budget // 1000}k reasoning" for budget in cfg.reasoning_budgets]
    lines = [
        "# Translation-QA cost per language pair",
        "",
        f"Generated: {generated}",
        f"Model/pricing label: `{args.model_id}`; input ${cfg.price_in}/MTok; output ${cfg.price_out}/MTok.",
        f"Tokenizer proxy: `{args.tokenizer}`. Sampling: "
        + (f"{cfg.sample_n} deterministic random rows/language (seed {cfg.sample_seed})." if args.quick else "full splits."),
        "Reasoning budgets are project sensitivity scenarios, not measured percentiles. "
        f"Every scenario reserves {cfg.visible_output_tokens:,} additional visible-output tokens.",
        "Dataset revisions are pinned to immutable Hugging Face commit hashes recorded in the summary metadata.",
    ]
    lines.extend(["", "Pinned dataset revisions:", ""])
    for result in results:
        spec = SPECS[result["dataset"]]
        revision_text = f"`{spec.hf_path}@{spec.revision}`"
        if spec.external_english:
            ee = spec.external_english
            revision_text += f"; English source `{ee['hf_path']}@{ee['revision']}`"
        lines.append(f"- {result['dataset']}: {revision_text}")
    lines.extend([
        "",
        "| Dataset | Language pair | Dataset config | Items | "
        + " | ".join(budget_headers)
        + " | Key mismatches observed | Coverage |",
        "|---|---|---|---:|" + "---:|" * len(budget_headers) + "---:|---|",
    ])
    for result in results:
        dataset = result["dataset"]
        by_iso = {row["iso_language"]: row for row in result["per_lang"]}
        for iso in args.languages:
            row = by_iso.get(iso)
            if row is None:
                cells = [dataset, f"en–{iso}", "—", "—"]
                cells.extend("—" for _ in cfg.reasoning_budgets)
                cells.append("—")
                cells.append("unavailable")
            else:
                cells = [
                    dataset,
                    f"en–{iso}",
                    f"`{row['dataset_config']}`",
                    f"{row['pairs']:,}",
                ]
                cells.extend(
                    f"${row[f'cost_reasoning_{budget}_usd']:,.2f}"
                    for budget in cfg.reasoning_budgets
                )
                cells.append(
                    f"{row['gold_mismatches_observed']}/{row['rows_observed']}"
                )
                cells.append("complete")
            lines.append("| " + " | ".join(cells) + " |")
    potential_issues = [
        (result["dataset"], row)
        for result in results
        for row in result["per_lang"]
        if row["gold_mismatches_observed"]
    ]
    if potential_issues:
        lines.extend(["", "## Potential dataset mistakes", ""])
        for dataset, row in potential_issues:
            examples = ", ".join(f"`{key}`" for key in row["gold_mismatch_examples"])
            lines.append(
                f"- {dataset} en–{row['iso_language']}: "
                f"{row['gold_mismatches_observed']}/{row['rows_observed']} observed rows "
                f"have different source/target gold keys. Example IDs: {examples}. "
                "Review the paired content before classifying the cause."
            )
    lines.extend([
        "",
        "## Limitations",
        "",
        "Input counts use the serialized request implemented in `estimate_cost.py`; production prompt changes require a recount. "
        "Quick mode extrapolates sampled input lengths. MMMLU has no Russian or Vietnamese config. "
        "MMLU-ProX test splits do not provide usable translated CoT, so CoT is excluded rather than estimated. "
        "MMLU-ProX-Lite overlaps the full dataset and should be treated as an alternative workload, not added to it. "
        "Key-mismatch counts cover sampled rows in quick mode and all rows in a full run; they are potential dataset mistakes requiring review, not join failures or automatically confirmed errors.",
        "",
    ])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Translation-QA cost estimator over multilingual benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(SPECS),
                    choices=list(SPECS), metavar="DATASET",
                    help=f"subset of: {', '.join(SPECS)}")
    ap.add_argument("--quick", action="store_true",
                    help="sample-based token estimate (stream + extrapolate)")
    ap.add_argument("--price-in", type=float, default=3.00,
                    help="$ per 1M input tokens of the reasoning model")
    ap.add_argument("--price-out", type=float, default=15.00,
                    help="$ per 1M output tokens (reasoning bills as output)")
    ap.add_argument("--reasoning-budgets", type=int, nargs="+", default=[1000, 5000, 20000],
                    help="reasoning-token sensitivity scenarios")
    ap.add_argument("--visible-output-tokens", type=int, default=1000,
                    help="visible-report allowance added to every reasoning budget")
    ap.add_argument("--sample-n", type=int, default=300,
                    help="rows sampled per language in --quick mode")
    ap.add_argument("--sample-seed", type=int, default=42,
                    help="seed for deterministic quick-mode sampling")
    ap.add_argument("--tokenizer", type=str, default="o200k_base",
                    help="tiktoken encoding name used as token-count proxy "
                         "(e.g. o200k_base, cl100k_base)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="number of datasets to process in parallel")
    ap.add_argument("--lang-jobs", type=int, default=30,
                    help="number of languages/configs to process in parallel within each dataset")
    ap.add_argument("--summary-json", type=str, default=None,
                    help="optional path for JSON summary output")
    ap.add_argument("--summary-markdown", type=str, default=None,
                    help="optional per-language-pair Markdown report path")
    ap.add_argument("--languages", nargs="+", default=list(TARGET_LANGUAGE_SET),
                    help="ISO 639-1 target languages; English is always the source")
    ap.add_argument("--allow-partial-languages", action="store_true",
                    help="export unavailable pairs instead of failing coverage validation")
    ap.add_argument("--model-id", default="unspecified",
                    help="model/pricing label recorded in Markdown metadata")
    args = ap.parse_args()

    if not args.datasets:
        ap.error("select at least one dataset")
    if args.jobs <= 0 or args.lang_jobs <= 0:
        ap.error("--jobs and --lang-jobs must be positive")
    unknown = sorted(set(args.languages) - set(TARGET_LANGUAGE_SET))
    if unknown:
        ap.error(f"unsupported language code(s): {', '.join(unknown)}")
    if len(set(args.languages)) != len(args.languages):
        ap.error("--languages contains duplicates")
    if any(value <= 0 for value in args.reasoning_budgets):
        ap.error("--reasoning-budgets must contain positive integers")
    if len(set(args.reasoning_budgets)) != len(args.reasoning_budgets):
        ap.error("--reasoning-budgets contains duplicates")
    if args.visible_output_tokens < 0 or args.sample_n <= 0:
        ap.error("visible output must be non-negative and sample-n must be positive")

    selections = {}
    coverage_errors = []
    for key in args.datasets:
        spec = SPECS[key]
        available = spec.configs or get_dataset_config_names(
            spec.hf_path, revision=spec.revision
        )
        resolved, missing = resolve_language_configs(spec, args.languages, available)
        selections[key] = (resolved, missing)
        if missing:
            coverage_errors.append(f"{key}: {', '.join(missing)}")
    # Coverage is checked before loading rows, so a strict run cannot produce a
    # deceptively partial report after spending time on other datasets.
    if coverage_errors and not args.allow_partial_languages:
        ap.error(
            "requested languages are unavailable (pass --allow-partial-languages "
            "to retain them as unavailable rows): " + "; ".join(coverage_errors)
        )

    cfg = build_run_config(
        price_in=args.price_in,
        price_out=args.price_out,
        reasoning_budgets=args.reasoning_budgets,
        visible_output_tokens=args.visible_output_tokens,
        sample_n=args.sample_n,
        sample_seed=args.sample_seed,
        instruction=DEFAULT_QA_INSTRUCTION,
        model_id=args.model_id,
        tokenizer=args.tokenizer,
    )
    cfg_kwargs = {
        "price_in": args.price_in,
        "price_out": args.price_out,
        "reasoning_budgets": args.reasoning_budgets,
        "visible_output_tokens": args.visible_output_tokens,
        "sample_n": args.sample_n,
        "sample_seed": args.sample_seed,
        "instruction": DEFAULT_QA_INSTRUCTION,
        "model_id": args.model_id,
        "tokenizer": args.tokenizer,
    }

    if args.summary_json is None:
        summary_csv_path = "translation_qa_cost_summary.csv"
        per_language_csv_path = lambda dataset_key: f"per_language_{dataset_key}.csv"
    else:
        summary_stem, _ = os.path.splitext(args.summary_json)
        summary_csv_path = f"{summary_stem}.csv"
        per_language_csv_path = (
            lambda dataset_key: f"{summary_stem}__per_language_{dataset_key}.csv"
        )

    records = []
    if args.jobs == 1:
        results = [
            process_dataset(
                key, selections[key][0], selections[key][1],
                args.quick, cfg_kwargs, args.lang_jobs,
            )
            for key in args.datasets
        ]
    else:
        max_workers = min(args.jobs, len(args.datasets), os.cpu_count() or 1)
        # Parallelize at the dataset level so each worker can load/tokenize independently.
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    process_dataset,
                    args.datasets,
                    [selections[key][0] for key in args.datasets],
                    [selections[key][1] for key in args.datasets],
                    [args.quick] * len(args.datasets),
                    [cfg_kwargs] * len(args.datasets),
                    [args.lang_jobs] * len(args.datasets),
                )
            )

    for result in results:
        key = result["dataset"]
        if result["skipped"]:
            print(f"[skip] {key}: nothing aligned (check hf_path / id_field / "
                  f"english_config / field names).")
            continue
        records.extend(result["records"])
        pd.DataFrame(result["per_lang"]).to_csv(per_language_csv_path(key), index=False)

    df = pd.DataFrame(records)
    df.to_csv(summary_csv_path, index=False)
    if args.summary_json is not None:
        df.to_json(args.summary_json, orient="records", indent=2)
    if args.summary_markdown is not None:
        markdown_dir = os.path.dirname(args.summary_markdown)
        if markdown_dir:
            os.makedirs(markdown_dir, exist_ok=True)
        with open(args.summary_markdown, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(results, cfg, args))
    print("\n" + df.to_string(index=False))
    if args.summary_json is None:
        print("\nSaved: translation_qa_cost_summary.csv + per_language_<dataset>.csv")
    else:
        print(
            f"\nSaved: {summary_csv_path} + {args.summary_json} + "
            f"{summary_stem}__per_language_<dataset>.csv"
        )
    if args.summary_markdown is not None:
        print(f"Saved: {args.summary_markdown}")


if __name__ == "__main__":
    main()
