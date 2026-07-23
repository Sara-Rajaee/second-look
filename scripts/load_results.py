"""Load results_analysis/{task}/{model}/{lang}/{metrics,metadata,raw_outputs}.json into dataframes.

Usage:
    python scripts/load_results.py [results_dir]

Or import:
    from scripts.load_results import load_runs, load_examples
"""

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_RESULTS_DIR = "/fnwi_fs/ivi/irlab/personal/srajaee/second-look/results_analysis"


def local_id(row) -> str:
    """Strip the `{lang}_` prefix from an example id so it aligns across languages
    (e.g. "en_42" and "de_42" both become "42", the same underlying item)."""
    prefix = f"{row['lang']}_"
    return row["id"][len(prefix):] if row["id"].startswith(prefix) else row["id"]


def _find_runs(results_dir: Path):
    """Yield (task, model, lang, run_dir) for every leaf dir with a metrics.json."""
    for metrics_path in sorted(results_dir.glob("*/*/*/metrics.json")):
        run_dir = metrics_path.parent
        lang = run_dir.name
        model = run_dir.parent.name
        task = run_dir.parent.parent.name
        yield task, model, lang, run_dir


def load_runs(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """One row per (task, model, lang) run: metrics + metadata."""
    results_dir = Path(results_dir)
    rows = []
    for task, model, lang, run_dir in _find_runs(results_dir):
        metrics = json.loads((run_dir / "metrics.json").read_text())
        metadata = json.loads((run_dir / "metadata.json").read_text())
        gen_params = metadata.pop("generation_params", {})
        rows.append(
            {
                "task": task,
                "model": model,
                "lang": lang,
                **metrics,
                **metadata,
                **{f"gen_{k}": v for k, v in gen_params.items()},
                "run_dir": str(run_dir),
            }
        )
    return pd.DataFrame(rows)


def load_examples(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """One row per example: raw_outputs.json entries, tagged with task/model/lang."""
    results_dir = Path(results_dir)
    rows = []
    for task, model, lang, run_dir in _find_runs(results_dir):
        raw_path = run_dir / "raw_outputs.json"
        if not raw_path.exists():
            continue
        examples = json.loads(raw_path.read_text())
        for ex in examples:
            scores = ex.pop("scores", {}) or {}
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "lang": lang,
                    **ex,
                    **{f"score_{k}": v for k, v in scores.items()},
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    print(f"Runs: {len(runs)} rows, {runs.shape[1]} cols")
    print(runs.head())

    examples = load_examples(args.results_dir)
    print(f"\nExamples: {len(examples)} rows, {examples.shape[1]} cols")
    print(examples.head())

    counts = examples.groupby(["task", "model", "lang"]).size().reset_index(name="n")
    print(counts)


if __name__ == "__main__":
    main()
