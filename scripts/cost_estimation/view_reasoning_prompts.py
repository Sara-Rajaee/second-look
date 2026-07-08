#!/usr/bin/env python3
"""Preview reasoning-model input tuples for the supported evaluation datasets."""

from __future__ import annotations

import argparse
from functools import lru_cache

from datasets import get_dataset_config_names, load_dataset

import estimate_cost as ec


def global_piqa_parallel_key_normalizer(split: str, row: dict, i: int):
    """Normalize Global PIQA example ids across languages."""
    raw_key = row.get("example_id", i)
    if not isinstance(raw_key, str):
        raise KeyError(
            f"global_piqa_parallel/{split}: expected string example_id, "
            f"got {type(raw_key).__name__}"
        )
    return (split, raw_key.rsplit("_", 2)[0])


@lru_cache(maxsize=None)
def load_split(hf_path: str, config_name: str, split: str):
    """Load and cache a dataset split."""
    return load_dataset(hf_path, config_name, split=split)


@lru_cache(maxsize=None)
def dataset_configs(hf_path: str):
    """Load and cache the available config names for a dataset."""
    return tuple(get_dataset_config_names(hf_path))


def english_config_for_spec(spec: ec.DatasetSpec) -> str | None:
    """Return the English config name used for source-side previews."""
    if spec.key == "global_piqa_parallel":
        return "eng_latn"
    return spec.english_config


def target_configs_for_spec(spec: ec.DatasetSpec) -> list[str]:
    """Return the non-English configs that can be previewed for a dataset."""
    configs = list(spec.configs or dataset_configs(spec.hf_path))
    english_config = english_config_for_spec(spec)
    return [cfg for cfg in configs if cfg != english_config]


def normalize_mmmlu_english_row(row: dict) -> dict:
    """Map cais/mmlu English rows onto the MMMLU field names."""
    choices = row["choices"]
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    return {
        "Question": row["question"],
        "A": choices[0],
        "B": choices[1],
        "C": choices[2],
        "D": choices[3],
        "Answer": answer_map[row["answer"]],
        "Subject": row["subject"],
    }


def embedded_global_piqa_source_row(row: dict) -> dict:
    """Build the English source side from embedded fields in a target row."""
    return {
        "prompt": row["eng_prompt"],
        "solution0": row["eng_solution0"],
        "solution1": row["eng_solution1"],
        "solution2": row["eng_solution2"],
        "solution3": row["eng_solution3"],
        "label": row["label"],
        "example_id": row["example_id"],
    }


def build_prompt_parts(spec: ec.DatasetSpec, row: dict) -> list[tuple[str, object]]:
    """Extract the prompt-side fields for a row in display order."""
    if spec.key == "global_piqa_parallel":
        return [
            ("prompt", row["prompt"]),
            ("solution0", row["solution0"]),
            ("solution1", row["solution1"]),
            ("solution2", row["solution2"]),
            ("solution3", row["solution3"]),
        ]
    if spec.key in {"mmlu_prox", "mmlu_prox_lite"}:
        parts = [("question", row["question"])]
        options = []
        for i in range(10):
            value = row.get(f"option_{i}")
            if value is None:
                continue
            options.append(value)
        parts.append(("options", options))
        return parts

    parts = [(field, row.get(field)) for field in spec.prompt_fields]
    parts.extend((field, row.get(field)) for field in spec.prompt_list_fields)
    return parts


def build_gold_parts(spec: ec.DatasetSpec, src_row: dict, tgt_row: dict) -> list[tuple[str, object]]:
    """Extract the gold-side tuple components for display."""
    parts = []
    for field in spec.gold_invariant_fields:
        parts.append((field, src_row.get(field, tgt_row.get(field))))
    for field in spec.gold_translated_fields:
        parts.append((f"{field}_src", src_row.get(field)))
        parts.append((f"{field}_tgt", tgt_row.get(field)))
    for field in spec.gold_translated_list_fields:
        parts.append((f"{field}_src", src_row.get(field)))
        parts.append((f"{field}_tgt", tgt_row.get(field)))
    return parts


def load_source_index(spec: ec.DatasetSpec, split: str) -> dict:
    """Load a canonical-key index for source-side rows for one split."""
    if spec.key == "global_piqa_parallel":
        return {}

    if spec.external_english is not None:
        ee = spec.external_english
        ds = load_split(ee["hf_path"], ee["config"], ee["split"])
        if spec.key == "mmmlu":
            return {
                ec.canonical_row_key(spec, split, row, i): normalize_mmmlu_english_row(row)
                for i, row in enumerate(ds)
            }
        return {
            ec.canonical_row_key(spec, split, row, i): row
            for i, row in enumerate(ds)
        }

    english_config = english_config_for_spec(spec)
    if english_config is None:
        raise ValueError(f"{spec.key}: no English source configuration is available")

    ds = load_split(spec.hf_path, english_config, split)
    return {
        ec.canonical_row_key(spec, split, row, i): row
        for i, row in enumerate(ds)
    }


def load_aligned_rows(spec: ec.DatasetSpec, lang: str, split: str, index: int) -> tuple[dict, dict]:
    """Load one aligned source/target example pair for preview."""
    tgt_ds = load_split(spec.hf_path, lang, split)
    if index < 0 or index >= len(tgt_ds):
        raise IndexError(
            f"{spec.key}/{lang}/{split}: index {index} is out of range for {len(tgt_ds)} rows"
        )
    tgt_row = tgt_ds[index]

    if spec.key == "global_piqa_parallel":
        return embedded_global_piqa_source_row(tgt_row), tgt_row

    source_index = load_source_index(spec, split)
    key = ec.canonical_row_key(spec, split, tgt_row, index)
    src_row = source_index.get(key)
    if src_row is None:
        raise KeyError(
            f"{spec.key}/{lang}/{split}: missing source row for canonical key {key!r}"
        )
    return src_row, tgt_row


def render_reasoning_tuple(spec: ec.DatasetSpec, lang: str, split: str, index: int) -> str:
    """Render one reasoning-model tuple preview for a dataset example."""
    src_row, tgt_row = load_aligned_rows(spec, lang, split, index)
    src_parts = build_prompt_parts(spec, src_row)
    tgt_parts = build_prompt_parts(spec, tgt_row)
    gold_parts = build_gold_parts(spec, src_row, tgt_row)
    return format_reasoning_tuple(src_parts, tgt_parts, gold_parts)


def format_part_lines(parts: list[tuple[str, object]]) -> list[str]:
    """Format one tuple section into readable text lines."""
    lines = []
    for name, value in parts:
        if isinstance(value, list):
            lines.append(f"{name}:")
            for i, item in enumerate(value):
                lines.append(f"  [{i}] {item}")
        else:
            lines.append(f"{name}: {value}")
    return lines


def format_reasoning_tuple(
    src_parts: list[tuple[str, object]],
    tgt_parts: list[tuple[str, object]],
    gold_parts: list[tuple[str, object]],
) -> str:
    """Render the tuple as readable SRC/TGT/GOLD sections."""
    lines = ["SRC:"]
    lines.extend(f"  {line}" for line in format_part_lines(src_parts))
    lines.append("TGT:")
    lines.extend(f"  {line}" for line in format_part_lines(tgt_parts))
    lines.append("GOLD:")
    lines.extend(f"  {line}" for line in format_part_lines(gold_parts))
    return "\n".join(lines)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preview reasoning-model tuples for the evaluation datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(ec.SPECS),
        choices=list(ec.SPECS),
        metavar="DATASET",
        help=f"subset of: {', '.join(ec.SPECS)}",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="dataset config/language to inspect; defaults to the first non-English config",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="split to inspect; defaults to the first split for each dataset",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="example index within the selected split/config",
    )
    return parser.parse_args()


def main():
    """Print one formatted reasoning tuple preview per requested dataset."""
    args = parse_args()

    for dataset_key in args.datasets:
        spec = ec.SPECS[dataset_key]
        splits = ec.iter_splits(spec)
        split = args.split or splits[0]
        if split not in splits:
            raise ValueError(f"{dataset_key}: split {split!r} is not in {splits}")

        targets = target_configs_for_spec(spec)
        if not targets:
            raise ValueError(f"{dataset_key}: no non-English target configs are available")
        lang = args.lang or targets[0]
        if lang not in targets:
            raise ValueError(f"{dataset_key}: language/config {lang!r} is not in {targets}")

        print(f"=== {dataset_key} ===")
        print(f"lang={lang} split={split} index={args.index}")
        print(render_reasoning_tuple(spec, lang, split, args.index))
        print()


if __name__ == "__main__":
    main()
