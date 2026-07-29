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
    {out_dir}/{task}/by_losing_language/{lang}/strong_xor_losses.jsonl
    {out_dir}/{task}/{safe_model}/{lang_a}__{lang_b}/examples.jsonl

The strong-XOR results are grouped by the language that *lost*: one folder per
language, written only when that language actually lost at least one item, each
holding every such item regardless of which language won it. That matches the
n_losing_strong_xor column of analysis_xor_stats.py, so the counts there can be
audited by counting lines here.

The per-model pair dirs join the language pair on a double underscore because the
codes themselves contain single underscores and vary in segment count (arb_arab,
fra_latn_fran), so a single-underscore join could not be parsed back apart.

Language payloads are nested under a "langs" key rather than spread across the top
level, so a language code can never collide with a metadata key.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_results import DEFAULT_RESULTS_DIR, load_examples, local_id

XOR_OUT_ROOT = Path("/fnwi_fs/ivi/irlab/personal/stroshi/second_look/outputs/xor")
DEFAULT_OUT_DIR = XOR_OUT_ROOT / "examples"

RECORD_COLS = ["source", "prompt", "generation", "target_text", "score_accuracy"]


def per_model_score_wide(model_df: pd.DataFrame) -> pd.DataFrame:
    return model_df.pivot_table(
        index="local_id", columns="lang", values="score_accuracy", aggfunc="first"
    )


def per_model_records(model_df: pd.DataFrame, lang: str) -> dict:
    lang_df = model_df[model_df["lang"] == lang].set_index("local_id")
    return lang_df[RECORD_COLS].to_dict(orient="index")


def pair_jsonl_path(out_dir: Path, task: str, who: str, lang_a: str, lang_b: str) -> Path:
    """{out_dir}/{task}/{who}/{lang_a}__{lang_b}/examples.jsonl, dir created."""
    pair_dir = out_dir / task / who.replace("/", "_") / f"{lang_a}__{lang_b}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    return pair_dir / "examples.jsonl"


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

        with open(pair_jsonl_path(out_dir, task, model, lang_a, lang_b), "w") as f:
            for id_, solved_lang in rows:
                rec = {
                    "id": id_,
                    "solved_lang": solved_lang,
                    # Nested rather than spread at the top level: language codes and
                    # metadata keys would otherwise share a namespace, and Indonesian's
                    # code ("id") silently overwrote the item id.
                    "langs": {
                        lang_a: records[(model, lang_a)].get(id_),
                        lang_b: records[(model, lang_b)].get(id_),
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def lang_payload(lang: str, id_: str, eligible: list[str], records: dict) -> dict:
    """The shared item text for `lang` plus each model's generation and score."""
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
    return {**(shared or {}), "per_model": per_model}


def collect_strong_xor_losses(task: str, lang_a: str, lang_b: str, task_models: list[str],
                               score_wide: dict, records: dict) -> dict[str, list[dict]]:
    """Strong XOR for one language pair, grouped by the language that lost.

    A "strong" XOR is an item that every eligible model (at least two) solves in one
    language and fails in the other, with all of them agreeing on which language wins.
    Returns {losing_lang: [record, ...]}; empty when the pair yields nothing.
    """
    eligible = [m for m in task_models
                if lang_a in score_wide[m].columns and lang_b in score_wide[m].columns]
    if len(eligible) < 2:
        return {}

    common_ids = None
    for m in eligible:
        ids = set(score_wide[m][[lang_a, lang_b]].dropna().index)
        common_ids = ids if common_ids is None else common_ids & ids
    if not common_ids:
        return {}

    losses: dict[str, list[dict]] = {}
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
        if not (agrees and len(directions) == 1):
            continue

        winner = next(iter(directions))
        loser = lang_b if winner == lang_a else lang_a
        losses.setdefault(loser, []).append({
            "id": id_,
            "task": task,
            "losing_lang": loser,
            "winning_lang": winner,
            "models": eligible,
            "langs": {lang: lang_payload(lang, id_, eligible, records)
                      for lang in (loser, winner)},
        })
    return losses


def write_losses_by_language(out_dir: Path, task: str,
                              losses_by_lang: dict[str, list[dict]]) -> list[str]:
    """One folder per losing language, written only for languages that actually lost.

    {out_dir}/{task}/by_losing_language/{lang}/strong_xor_losses.jsonl

    One line per distinct item, not per language pair: an item that several languages
    beat is merged into a single record listing all of them under `winning_langs`.
    Line count therefore equals n_losing_strong_xor in analysis_xor_stats.py's CSV,
    which also counts distinct items.
    """
    written = []
    for lang in sorted(losses_by_lang):
        merged: dict[str, dict] = {}
        for rec in losses_by_lang[lang]:
            existing = merged.get(rec["id"])
            if existing is None:
                merged[rec["id"]] = {**rec, "winning_langs": [rec["winning_lang"]]}
                merged[rec["id"]].pop("winning_lang")
            else:
                existing["winning_langs"].append(rec["winning_lang"])
                existing["langs"].update(rec["langs"])
        if not merged:
            continue

        lang_dir = out_dir / task / "by_losing_language" / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        with open(lang_dir / "strong_xor_losses.jsonl", "w") as f:
            # Sorted so the file is stable across runs and diffable.
            for id_ in sorted(merged):
                rec = merged[id_]
                rec["winning_langs"] = sorted(set(rec["winning_langs"]))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written.append(f"{lang}({len(merged)})")
    return written


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

        # Strong-XOR losses are accumulated across every pair first, so each language
        # ends up with a single file listing every item it lost, whoever beat it.
        losses_by_lang: dict[str, list[dict]] = {}
        for lang_a, lang_b in itertools.combinations(langs, 2):
            write_per_model_xor(out_dir, task, lang_a, lang_b, task_models, score_wide, records)
            for lang, rows in collect_strong_xor_losses(
                    task, lang_a, lang_b, task_models, score_wide, records).items():
                losses_by_lang.setdefault(lang, []).extend(rows)

        written = write_losses_by_language(out_dir, task, losses_by_lang)
        print(f"[{task}] models {task_models} across {len(langs)} languages")
        print(f"[{task}] strong-XOR losses by language: {' '.join(written) or 'none'}")


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
