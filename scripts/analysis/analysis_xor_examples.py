"""Dump the actual examples behind cross-language XOR disagreements: cases where a
model solves an item in one language but not in another.

Examples are aligned across languages via their trailing index (id = f"{lang}_{index}"),
same as analysis_heat_map.py. For each task, and for each pair of languages, this writes
one jsonl per model plus one "all_models" jsonl restricted to XOR cases that agree across
every model tested for that task (a "strong" XOR, not just one model's noise).

Usage:
    python scripts/analysis/analysis_xor_examples.py [--results-dir DIR] [--out-dir DIR]
                                                       [--tasks TASK ...] [--models MODEL ...]

Outputs, per task:
    {out_dir}/{task}/all_models/{lang_a}_{lang_b}/output.jsonl
    {out_dir}/{task}/{safe_model}/{lang_a}_{lang_b}/output.jsonl
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_results import DEFAULT_RESULTS_DIR, load_examples, local_id

DEFAULT_OUT_DIR = Path("/fnwi_fs/ivi/irlab/personal/stroshi/second_look/outputs/xor_examples")

RECORD_COLS = ["source", "prompt", "generation", "target_text", "score_accuracy"]


def per_model_score_wide(model_df: pd.DataFrame) -> pd.DataFrame:
    return model_df.pivot_table(
        index="local_id", columns="lang", values="score_accuracy", aggfunc="first"
    )


def per_model_records(model_df: pd.DataFrame, lang: str) -> dict:
    lang_df = model_df[model_df["lang"] == lang].set_index("local_id")
    return lang_df[RECORD_COLS].to_dict(orient="index")


def write_per_model_xor(out_dir: Path, task: str, lang_a: str, lang_b: str,
                         task_models: list[str], score_wide: dict, records: dict) -> None:
    for model in task_models:
        wide = score_wide[model]
        if lang_a not in wide.columns or lang_b not in wide.columns:
            continue
        pair = wide[[lang_a, lang_b]].dropna()
        a, b = pair[lang_a], pair[lang_b]
        a_solved = pair[(a == 1) & (b == 0)].index.tolist()
        b_solved = pair[(a == 0) & (b == 1)].index.tolist()
        rows = sorted(
            [(i, lang_a) for i in a_solved] + [(i, lang_b) for i in b_solved]
        )
        if not rows:
            continue

        safe_model = model.replace("/", "_")
        pair_dir = out_dir / task / safe_model / f"{lang_a}_{lang_b}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        with open(pair_dir / "output.jsonl", "w") as f:
            for id_, solved_lang in rows:
                rec = {
                    "id": id_,
                    "solved_lang": solved_lang,
                    lang_a: records[(model, lang_a)].get(id_),
                    lang_b: records[(model, lang_b)].get(id_),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_all_models_xor(out_dir: Path, task: str, lang_a: str, lang_b: str,
                          task_models: list[str], score_wide: dict, records: dict) -> None:
    eligible = [m for m in task_models
                if lang_a in score_wide[m].columns and lang_b in score_wide[m].columns]
    if len(eligible) < 2:
        return

    common_ids = None
    for m in eligible:
        ids = set(score_wide[m][[lang_a, lang_b]].dropna().index)
        common_ids = ids if common_ids is None else common_ids & ids
    if not common_ids:
        return

    rows = []
    for id_ in sorted(common_ids):
        directions = set()
        agrees = True
        for m in eligible:
            va = score_wide[m].loc[id_, lang_a]
            vb = score_wide[m].loc[id_, lang_b]
            if va == 1 and vb == 0:
                directions.add(lang_a)
            elif va == 0 and vb == 1:
                directions.add(lang_b)
            else:
                agrees = False
                break
        if agrees and len(directions) == 1:
            rows.append((id_, next(iter(directions))))
    if not rows:
        return

    pair_dir = out_dir / task / "all_models" / f"{lang_a}_{lang_b}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    with open(pair_dir / "output.jsonl", "w") as f:
        for id_, solved_lang in rows:
            rec = {"id": id_, "solved_lang": solved_lang, "models": eligible}
            for lang in (lang_a, lang_b):
                shared = None
                per_model = {}
                for m in eligible:
                    full = records[(m, lang)].get(id_)
                    if full is None:
                        continue
                    if shared is None:
                        shared = {k: full[k] for k in ("source", "prompt", "target_text")}
                    per_model[m] = {"generation": full["generation"],
                                    "score_accuracy": full["score_accuracy"]}
                rec[lang] = {**(shared or {}), "per_model": per_model}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_xor_examples(examples: pd.DataFrame, out_dir: Path,
                        tasks: list[str] | None = None,
                        models: list[str] | None = None) -> None:
    examples = examples.copy()
    examples["local_id"] = examples.apply(local_id, axis=1)

    tasks = tasks if tasks is not None else sorted(examples["task"].unique())

    for task in tasks:
        task_df = examples[examples["task"] == task]
        if task_df.empty:
            print(f"[{task}] no data found, skipping")
            continue

        task_models = sorted(task_df["model"].unique())
        if models is not None:
            task_models = [m for m in task_models if m in models]
        if not task_models:
            print(f"[{task}] none of the requested models have data, skipping")
            continue

        langs = sorted(task_df["lang"].unique())
        if len(langs) < 2:
            print(f"[{task}] fewer than 2 languages, skipping")
            continue

        score_wide = {}
        records = {}
        for model in task_models:
            model_df = task_df[task_df["model"] == model]
            score_wide[model] = per_model_score_wide(model_df)
            for lang in langs:
                records[(model, lang)] = per_model_records(model_df, lang)

        for lang_a, lang_b in itertools.combinations(langs, 2):
            write_per_model_xor(out_dir, task, lang_a, lang_b, task_models, score_wide, records)
            write_all_models_xor(out_dir, task, lang_a, lang_b, task_models, score_wide, records)

        print(f"[{task}] wrote XOR examples for models {task_models} across {len(langs)} languages")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tasks", nargs="*", default=None,
                         help="Tasks to process (default: all found in results-dir)")
    parser.add_argument("--models", nargs="*", default=None,
                         help="Models to consider (default: all found in results-dir)")
    args = parser.parse_args()

    examples = load_examples(args.results_dir)
    build_xor_examples(examples, Path(args.out_dir), tasks=args.tasks, models=args.models)


if __name__ == "__main__":
    main()
