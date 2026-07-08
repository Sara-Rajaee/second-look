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
    global_mmlu / mmmlu     : answer KEY (letter) is invariant, but the option
                              TEXTS it points to are translated -> we include
                              (answer_letter, option_text_src, option_text_tgt).
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

Outputs per dataset (two scenarios: typical / upper-bound error report length):
    TOTAL_$          full run over the union of all non-English languages
    PER_LANGUAGE_$   TOTAL / n_target_languages  (uniform-coverage assumption)
plus per_language_<dataset>.csv with measured token counts per language, so the
uniformity assumption can be checked against reality.

Usage:
    pip install datasets tiktoken pandas tqdm
    python scripts/cost_estimation/estimate_cost.py --help

    # example
    python scripts/cost_estimation/estimate_cost.py --quick \
        --price-in 3 --price-out 15 \
        --out-typical 800 --out-upper 4000 \
        --qa-instruction-tokens 600 --sample-n 300 \
        --tokenizer o200k_base \
        --datasets polymath global_piqa_parallel

--quick streams ~sample-n rows per language and extrapolates mean token counts.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
    qa_instruction_tokens: int  # rubric/system/output-schema overhead per call
    out_typical: int            # error-report tokens, typical scenario
    out_upper: int              # error-report tokens, upper-bound scenario
    sample_n: int               # rows sampled per language in --quick mode
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
    qa_instruction_tokens: int,
    out_typical: int,
    out_upper: int,
    sample_n: int,
    tokenizer: str,
) -> RunConfig:
    """Construct a RunConfig from serializable CLI parameters."""
    return RunConfig(
        price_in=price_in,
        price_out=price_out,
        qa_instruction_tokens=qa_instruction_tokens,
        out_typical=out_typical,
        out_upper=out_upper,
        sample_n=sample_n,
        enc=tiktoken.get_encoding(tokenizer),
    )


def run_config_to_kwargs(cfg: RunConfig, tokenizer: str) -> dict:
    """Convert a RunConfig back into serializable constructor kwargs."""
    return {
        "price_in": cfg.price_in,
        "price_out": cfg.price_out,
        "qa_instruction_tokens": cfg.qa_instruction_tokens,
        "out_typical": cfg.out_typical,
        "out_upper": cfg.out_upper,
        "sample_n": cfg.sample_n,
        "tokenizer": tokenizer,
    }


# ----------------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    key: str
    hf_path: str
    split: str | list[str]
    id_field: str                     # join key for src<->tgt alignment
    english_config: str               # config name of the English pivot
    key_normalizer: Callable[[str, dict, int], Any] | None = None
    target_embedded_source_builder: Callable[[dict], dict] | None = None
    # fields forming the PROMPT side (translated): question + options etc.
    prompt_fields: list = field(default_factory=list)
    prompt_list_fields: list = field(default_factory=list)
    # fields forming the GOLD side:
    gold_invariant_fields: list = field(default_factory=list)   # counted once
    gold_translated_fields: list = field(default_factory=list)  # counted on BOTH sides
    gold_translated_list_fields: list = field(default_factory=list)
    configs: list = None              # None -> auto-discover
    # Some datasets have no English config inside the repo (MMMLU): the English
    # source must come from an external dataset.
    external_english: dict = None     # {"hf_path":..., "config":..., "split":...}
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
        split="test",
        id_field="sample_id",
        english_config="en",
        prompt_fields=["question", "option_a", "option_b", "option_c", "option_d"],
        gold_invariant_fields=["answer"],   # letter key A-D, invariant
        # option texts already included on both sides via prompt_fields,
        # so answer-key validity is checkable from the tuple as-is.
        notes="41 target langs x ~14k. Tuple = (srcQ+opts, tgtQ+opts, key). "
              "Provenance prior: 16 MT langs = highest expected yield.",
    ),
    "mmmlu": DatasetSpec(
        key="mmmlu",
        hf_path="openai/MMMLU",
        split="test",
        id_field="Unnamed: 0",
        english_config=None,    # repo has no English config
        # MMMLU uses locale-style HF config names, and `Unnamed: 0` is only a
        # per-subject counter. We therefore join against `cais/mmlu` by the
        # full test-set row order, which we verified matches on subject,
        # options, and answer index after A/B/C/D -> 0/1/2/3 conversion.
        key_normalizer=mmmlu_key_normalizer,
        configs=["AR_XY", "BN_BD", "DE_DE", "ES_LA", "FR_FR", "HI_IN",
                 "ID_ID", "IT_IT", "JA_JP", "KO_KR", "PT_BR", "SW_KE",
                 "YO_NG", "ZH_CN"],
        prompt_fields=["Question", "A", "B", "C", "D"],
        gold_invariant_fields=["Answer"],
        external_english={"hf_path": "cais/mmlu", "config": "all", "split": "test"},
        notes="14 target langs x 14,042. English source pulled from cais/mmlu "
              "and joined by verified global row order.",
    ),
    "mmlu_prox": DatasetSpec(
        key="mmlu_prox",
        hf_path="li-lab/MMLU-ProX",
        split="test",
        id_field="question_id",
        english_config="en",
        prompt_fields=["question", "option_0", "option_1", "option_2", "option_3",
                       "option_4", "option_5", "option_6", "option_7", "option_8",
                       "option_9"],
        gold_invariant_fields=["answer", "answer_index"],
        gold_translated_fields=["cot_content"],   # translated CoT: both sides
        notes="28 target langs x 11,829. Richest tuple: includes CoT chains on "
              "both sides -> model can verify translated reasoning still yields "
              "the keyed answer. Consider li-lab/MMLU-ProX-Lite (658/lang) first.",
    ),
    "mmlu_prox_lite": DatasetSpec(
        key="mmlu_prox_lite",
        hf_path="li-lab/MMLU-ProX-Lite",
        split="test",
        id_field="question_id",
        english_config="en",
        prompt_fields=["question", "option_0", "option_1", "option_2", "option_3",
                       "option_4", "option_5", "option_6", "option_7", "option_8",
                       "option_9"],
        gold_invariant_fields=["answer", "answer_index"],
        gold_translated_fields=["cot_content"],
        notes="Lite: 28 target langs x 658.",
    ),
    "polymath": DatasetSpec(
        key="polymath",
        hf_path="Qwen/PolyMath",
        split=["top", "high", "medium", "low"],
        id_field="id",
        english_config="en",
        key_normalizer=polymath_key_normalizer,
        prompt_fields=["question"],
        gold_invariant_fields=["answer"],  # language-invariant number/expression
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
        split="test",
        id_field="example_id",
        english_config="eng_latn",
        key_normalizer=global_piqa_parallel_key_normalizer,
        target_embedded_source_builder=global_piqa_parallel_source_row,
        prompt_fields=["prompt", "solution0", "solution1", "solution2", "solution3"],
        gold_invariant_fields=["label"],   # index of correct solution
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
# Token accounting for one aligned (src_row, tgt_row) pair
# ----------------------------------------------------------------------------
def side_prompt_tokens(row, spec, cfg: RunConfig):
    t = sum(cfg.ntok(row.get(f)) for f in spec.prompt_fields)
    t += sum(cfg.ntok(row.get(f)) for f in spec.prompt_list_fields)
    return t


def tuple_tokens(src_row, tgt_row, spec, cfg: RunConfig) -> int:
    t = cfg.qa_instruction_tokens
    t += side_prompt_tokens(src_row, spec, cfg)     # src
    t += side_prompt_tokens(tgt_row, spec, cfg)     # tgt
    for f in spec.gold_invariant_fields:            # gold (once)
        t += cfg.ntok(src_row.get(f) if src_row else tgt_row.get(f))
    for f in spec.gold_translated_fields:           # gold_src + gold_tgt
        t += cfg.ntok(src_row.get(f)) + cfg.ntok(tgt_row.get(f))
    for f in spec.gold_translated_list_fields:
        t += cfg.ntok(src_row.get(f)) + cfg.ntok(tgt_row.get(f))
    return t


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
            ds = load_dataset(spec.hf_path, spec.english_config, split=split)
            for i, row in enumerate(ds):
                idx[canonical_row_key(spec, split, row, i)] = row
    elif spec.external_english is not None:
        if len(splits) != 1:
            raise ValueError(
                f"{spec.key}: external English pivot does not support multiple splits"
            )
        ee = spec.external_english
        ds = load_dataset(ee["hf_path"], ee["config"], split=ee["split"])
        for i, row in enumerate(ds):
            idx[canonical_row_key(spec, splits[0], row, i)] = row
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


def measure_language(spec: DatasetSpec, lang: str, quick: bool, cfg: RunConfig, en_idx: dict):
    """Measure one target language/config for a dataset."""
    splits = iter_splits(spec)
    lang_tokens, lang_cot_tokens, n_pairs = 0, 0, 0

    for split in splits:
        if quick:
            ds = load_dataset(spec.hf_path, lang, split=split, streaming=True)
            toks, cot_toks, sampled_rows = [], [], 0
            total_rows = get_split_size(spec.hf_path, lang, split)
            has_metadata_size = total_rows is not None
            for i, row in enumerate(ds):
                if not has_metadata_size:
                    total_rows = i + 1
                if sampled_rows >= cfg.sample_n:
                    if has_metadata_size:
                        break
                    continue
                src = source_row_for_target(spec, split, row, i, en_idx)
                toks.append(tuple_tokens(src, row, spec, cfg))
                cot_toks.append(cot_content_tokens(src, row, spec, cfg))
                sampled_rows += 1
            mean_tok = sum(toks) / max(len(toks), 1)
            mean_cot_tok = sum(cot_toks) / max(len(cot_toks), 1)
            lang_tokens += int(mean_tok * total_rows)
            lang_cot_tokens += int(mean_cot_tok * total_rows)
            n_pairs += total_rows
        else:
            ds = load_dataset(spec.hf_path, lang, split=split)
            for i, row in enumerate(ds):
                src = source_row_for_target(spec, split, row, i, en_idx)
                lang_tokens += tuple_tokens(src, row, spec, cfg)
                lang_cot_tokens += cot_content_tokens(src, row, spec, cfg)
                n_pairs += 1

    return {
        "lang": lang,
        "pairs": n_pairs,
        "in_tokens": lang_tokens,
        "cot_tokens": lang_cot_tokens,
        "avg_in_tok": round(lang_tokens / max(n_pairs, 1)),
    }


def get_split_size(hf_path: str, config_name: str, split: str) -> int:
    """Return the split size from metadata when available, else None."""
    builder = load_dataset_builder(hf_path, config_name)
    if builder.info.splits is None:
        return None
    split_info = builder.info.splits.get(split)
    if split_info is None or split_info.num_examples is None:
        return None
    return split_info.num_examples


def measure_dataset(
    spec: DatasetSpec,
    quick: bool,
    cfg: RunConfig,
    tokenizer: str,
    lang_jobs: int = 1,
):
    configs = spec.configs or get_dataset_config_names(spec.hf_path)
    # exclude English pivot from targets
    targets = [c for c in configs if c != spec.english_config]

    if lang_jobs == 1:
        en_idx = get_cached_english_index(spec)
        per_lang = [
            measure_language(spec, lang, quick, cfg, en_idx)
            for lang in tqdm(targets, desc=spec.key)
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
                executor.submit(measure_language, spec, lang, quick, cfg, en_idx): lang
                for lang in targets
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=spec.key):
                per_lang.append(future.result())
        per_lang.sort(key=lambda row: row["lang"])

    total_pairs = sum(row["pairs"] for row in per_lang)
    total_in = sum(row["in_tokens"] for row in per_lang)
    total_cot = sum(row["cot_tokens"] for row in per_lang)
    return targets, total_pairs, total_in, total_cot, per_lang


def cost_rows(spec, n_langs, n_pairs, in_tokens, cot_tokens, cfg: RunConfig):
    out = []
    for label, out_tok in [("typical", cfg.out_typical),
                           ("upper_bound", cfg.out_upper)]:
        total = (in_tokens * cfg.price_in
                 + n_pairs * out_tok * cfg.price_out) / 1e6
        cot_input_cost = (cot_tokens * cfg.price_in) / 1e6
        out.append({
            "dataset": spec.key,
            "target_languages": n_langs,
            "pairs_total": n_pairs,
            "input_tokens_M": round(in_tokens / 1e6, 2),
            "cot_input_tokens_M": round(cot_tokens / 1e6, 2),
            "avg_in_tok/pair": round(in_tokens / max(n_pairs, 1)),
            "scenario": label,
            "out_tok/pair": out_tok,
            "TOTAL_$": round(total, 2),
            "COT_INPUT_$": round(cot_input_cost, 2),
            "TOTAL_$_WITHOUT_COT": round(total - cot_input_cost, 2),
            "PER_LANGUAGE_$": round(total / max(n_langs, 1), 2),
            "notes": spec.notes,
        })
    return out


def process_dataset(dataset_key: str, quick: bool, cfg_kwargs: dict, lang_jobs: int):
    """Measure one dataset and return its summary rows plus per-language details."""
    cfg = build_run_config(**cfg_kwargs)
    spec = SPECS[dataset_key]
    targets, n_pairs, in_tokens, cot_tokens, per_lang = measure_dataset(
        spec, quick, cfg, cfg_kwargs["tokenizer"], lang_jobs
    )
    if n_pairs == 0:
        return {
            "dataset": dataset_key,
            "records": [],
            "per_lang": per_lang,
            "skipped": True,
        }
    return {
        "dataset": dataset_key,
        "records": cost_rows(spec, len(per_lang), n_pairs, in_tokens, cot_tokens, cfg),
        "per_lang": per_lang,
        "skipped": False,
    }


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
    ap.add_argument("--qa-instruction-tokens", type=int, default=600,
                    help="fixed per-call overhead: QA rubric / system prompt / "
                         "output schema (error spans, severity, key-validity)")
    ap.add_argument("--out-typical", type=int, default=800,
                    help="assumed error-report output tokens, typical scenario")
    ap.add_argument("--out-upper", type=int, default=4000,
                    help="assumed error-report output tokens, upper bound "
                         "(reasoning budget cap)")
    ap.add_argument("--sample-n", type=int, default=300,
                    help="rows sampled per language in --quick mode")
    ap.add_argument("--tokenizer", type=str, default="o200k_base",
                    help="tiktoken encoding name used as token-count proxy "
                         "(e.g. o200k_base, cl100k_base)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="number of datasets to process in parallel")
    ap.add_argument("--lang-jobs", type=int, default=30,
                    help="number of languages/configs to process in parallel within each dataset")
    ap.add_argument("--summary-json", type=str, default=None,
                    help="optional path for JSON summary output")
    args = ap.parse_args()

    cfg = build_run_config(
        price_in=args.price_in,
        price_out=args.price_out,
        qa_instruction_tokens=args.qa_instruction_tokens,
        out_typical=args.out_typical,
        out_upper=args.out_upper,
        sample_n=args.sample_n,
        tokenizer=args.tokenizer,
    )
    cfg_kwargs = {
        "price_in": args.price_in,
        "price_out": args.price_out,
        "qa_instruction_tokens": args.qa_instruction_tokens,
        "out_typical": args.out_typical,
        "out_upper": args.out_upper,
        "sample_n": args.sample_n,
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
            process_dataset(key, args.quick, cfg_kwargs, args.lang_jobs)
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
    print("\n" + df.to_string(index=False))
    if args.summary_json is None:
        print("\nSaved: translation_qa_cost_summary.csv + per_language_<dataset>.csv")
    else:
        print(
            f"\nSaved: {summary_csv_path} + {args.summary_json} + "
            f"{summary_stem}__per_language_<dataset>.csv"
        )


if __name__ == "__main__":
    main()
