"""Scratch space for exploring cross-language / cross-model consistency, per task.

Loads all examples into one big dataframe (task, model, lang, local_id, score_accuracy,
...) and leaves the actual analysis/plotting as TODOs to fill in.

Usage:
    python scripts/analysis/check_consistency.py [--results-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_results import DEFAULT_RESULTS_DIR, load_examples, local_id
DEFAULT_OUT_DIR = Path("outputs/consistency/")

def load_all(results_dir: str | Path = DEFAULT_RESULTS_DIR,
             cache_path: str | Path | None = None) -> pd.DataFrame:
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_pickle(cache_path)

    examples = load_examples(results_dir)
    examples["local_id"] = examples.apply(local_id, axis=1)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        examples.to_pickle(cache_path)

    return examples


def _plot_consistency_bars(consistency: pd.DataFrame, group_col: str, title: str,
                            out_path: Path) -> None:
    """Bar chart of mean score_accuracy per `group_col` value (e.g. per model or per
    language), averaged over items, with error bars showing the mean per-item std
    (how much that item's score varies across the group being averaged over)."""
    means = consistency["score_accuracy"]["mean"].groupby(level=group_col).mean()
    stds = consistency["score_accuracy"]["std"].groupby(level=group_col).mean()
    means, stds = means.align(stds)
    order = means.sort_values().index
    means, stds = means.loc[order], stds.loc[order]

    labels = [str(v).rsplit("/", 1)[-1] for v in means.index]

    fig, ax = plt.subplots(figsize=(0.6 * len(means) + 2, 4))
    ax.bar(labels, means.to_numpy(), yerr=stds.to_numpy(), color="#5598e7", capsize=4)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_consistency_over_langauges(consistency_over_languages: pd.DataFrame, task: str,
                                     out_dir: Path = DEFAULT_OUT_DIR) -> None:
    """Per model, how much score_accuracy varies across languages for the same item."""
    _plot_consistency_bars(
        consistency_over_languages, "model",
        f"{task}-cons. over lang (per model)",
        Path(out_dir) / f"{task}__consistency_over_languages.png",
    )


def plot_consistency_over_models(consistency_over_models: pd.DataFrame, task: str,
                                  out_dir: Path = DEFAULT_OUT_DIR) -> None:
    """Per language, how much score_accuracy varies across models for the same item."""
    _plot_consistency_bars(
        consistency_over_models, "lang",
        f"{task}-cons. over models (per language)",
        Path(out_dir) / f"{task}__consistency_over_models.png",
    )


def plot_n_suspicious_by_k(n_suspicious_by_k: dict[int, int], task: str,
                            out_dir: Path = DEFAULT_OUT_DIR) -> None:
    """Bar chart of how many items are "suspicious" (all models correct in >= k
    languages, but all models wrong in >= 1 language) as k varies."""
    ks = sorted(n_suspicious_by_k)
    counts = [n_suspicious_by_k[k] for k in ks]

    fig, ax = plt.subplots(figsize=(0.6 * len(ks) + 2, 4))
    ax.bar([str(k) for k in ks], counts, color="#5598e7")
    ax.set_xlabel("k")
    ax.set_ylabel("# suspicious items")
    ax.set_title(f"{task}-suspicious items by k")

    fig.tight_layout()
    out_path = Path(out_dir) / f"{task}__n_suspicious_by_k.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--outputs-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache_dir", default="/tmp/cached_pandas_frame.pkl")
    args = parser.parse_args()

    df = load_all(args.results_dir, cache_path=args.cache_dir)
    print(f"Loaded {len(df)} rows, {df.shape[1]} cols")
    print(df.groupby(["task", "model", "lang"]).size())

    N_MODELS = 2
    MAX_LANG = 10

    for task in df.task.unique():
        print(task)
        df_task = df[df.task == task]
        # Index(['task', 'model', 'lang', 'id', 'source', 'prompt', 'reasoning',
        #    'generation', 'raw_generation', 'finish_reason', 'target_text',
        #    'eval_type', 'points', 'score_accuracy', 'local_id'],
        #   dtype='str')

        # model consistency for each language
        assert not df_task.duplicated(["model", "lang", "local_id"]).any()  # assert that we group correctly (1 result per model land and id)
        consistency_over_languages = df_task.groupby(["model", "local_id"]).agg({"score_accuracy": ["mean", "std"]})   # over lang
        plot_consistency_over_langauges(consistency_over_languages, task, args.outputs_dir)

        # consistency across languages for each model;
        consistency_over_models = df_task.groupby(["lang", "local_id"]).agg({"score_accuracy": ["mean", "std"]})   # over model
        plot_consistency_over_models(consistency_over_models, task, args.outputs_dir)

        # for a given n_languages to be wrong, how many local_ids are considered suspicious at least for one language;
        out_acc_by_lang = df_task.groupby(["lang", "local_id"]).agg(total=("score_accuracy", "sum"))
        # how many models scored correctly on this lang and local_id
        per_id = out_acc_by_lang.assign(
            hi=out_acc_by_lang["total"] == N_MODELS, lo=out_acc_by_lang["total"] == 0
        ).groupby(level="local_id").agg(n_hi=("hi", "sum"), n_lo=("lo", "sum"))
        n_suspicious_by_k = {}
        for k in range(1, MAX_LANG + 1):
            suspicious = per_id[(per_id["n_hi"] >= k) & (per_id["n_lo"] >= 1)]
            n_suspicious_by_k[k] = len(suspicious)
        print(n_suspicious_by_k)
        plot_n_suspicious_by_k(n_suspicious_by_k, task, args.outputs_dir)

if __name__ == "__main__":
    main()
