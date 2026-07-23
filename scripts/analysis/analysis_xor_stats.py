"""Per (task, language) summary: how many examples exist, and what fraction show up as
the "losing" side of at least one all-models "strong XOR" language pair — i.e. this
language got accuracy 0 while some other language got accuracy 1, agreed by every
model tested for that task (same "strong XOR" definition as
analysis_xor_examples.py's all_models output).

Usage:
    python scripts/analysis/analysis_xor_stats.py [--results-dir DIR] [--out-csv PATH]
                                                    [--tasks TASK ...] [--models MODEL ...]
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_results import DEFAULT_RESULTS_DIR, load_examples, local_id

DEFAULT_OUT_CSV = Path("/fnwi_fs/ivi/irlab/personal/stroshi/second_look/outputs/xor_loser_stats.csv")


def compute_loser_stats(examples: pd.DataFrame,
                         tasks: list[str] | None = None,
                         models: list[str] | None = None) -> pd.DataFrame:
    examples = examples.copy()
    examples["local_id"] = examples.apply(local_id, axis=1)

    tasks = tasks if tasks is not None else sorted(examples["task"].unique())

    rows = []
    for task in tasks:
        task_df = examples[examples["task"] == task]
        if task_df.empty:
            continue

        task_models = sorted(task_df["model"].unique())
        if models is not None:
            task_models = [m for m in task_models if m in models]

        langs = sorted(task_df["lang"].unique())
        total_examples = {
            lang: task_df.loc[task_df["lang"] == lang, "local_id"].nunique() for lang in langs
        }
        loser_ids = {lang: set() for lang in langs}

        if len(task_models) >= 2:
            score_wide = {
                m: task_df[task_df["model"] == m].pivot_table(
                    index="local_id", columns="lang", values="score_accuracy", aggfunc="first"
                )
                for m in task_models
            }

            for lang_a, lang_b in itertools.combinations(langs, 2):
                eligible = [m for m in task_models
                            if lang_a in score_wide[m].columns and lang_b in score_wide[m].columns]
                if len(eligible) < 2:
                    continue

                common_ids = None
                for m in eligible:
                    ids = set(score_wide[m][[lang_a, lang_b]].dropna().index)
                    common_ids = ids if common_ids is None else common_ids & ids
                if not common_ids:
                    continue

                for id_ in common_ids:
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
                        winner = next(iter(directions))
                        loser = lang_b if winner == lang_a else lang_a
                        loser_ids[loser].add(id_)

        for lang in langs:
            n_total = total_examples[lang]
            n_losing = len(loser_ids[lang])
            pct = 100 * n_losing / n_total if n_total else 0.0
            rows.append({
                "task": task,
                "lang": lang,
                "n_total": n_total,
                "n_losing_strong_xor": n_losing,
                "pct_losing_strong_xor": round(pct, 2),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--tasks", nargs="*", default=None,
                         help="Tasks to process (default: all found in results-dir)")
    parser.add_argument("--models", nargs="*", default=None,
                         help="Models to consider (default: all found in results-dir)")
    args = parser.parse_args()

    examples = load_examples(args.results_dir)
    stats = compute_loser_stats(examples, tasks=args.tasks, models=args.models)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_csv, index=False)
    print(stats.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
