"""Per-problem difficulty by language: a (problem x language) failure-rate heatmap
for each task, with problems sorted by how much that failure rate varies across
languages.

For problem p and language L:

    failure_rate(p, L) = 1 - mean over models of the task metric
                       = (# models that answered wrong) / (# models tested)
    variance(p)        = variance of failure_rate(p, ·) across languages

Absolute failure rates are plotted, not values relative to a reference language —
English is just another column. Sorting by variance puts the language-sensitive
problems on top: the ones solved in some languages and failed in others. Problems
that are uniformly easy, or uniformly hard, have near-zero variance and sink to
the bottom regardless of how hard they are in absolute terms.

Cells covered by fewer than --min-models models are dropped, since a 1-model
failure rate is a coarser measurement than a 2-model one and mixing them distorts
the variance. This currently drops polymath_high's seven gemma-less languages.

Usage:
    python scripts/analysis/analysis_difficulty_heatmap.py [--results-dir DIR]
        [--out-dir DIR] [--min-models INT]

Outputs, per task:
    {out_dir}/{task}/failure_rate_heatmap.png
    {out_dir}/{task}/failure_by_problem.csv    variance, mean failure, per-language rate
Plus one summary across tasks:
    {out_dir}/failure_summary.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_results import DEFAULT_RESULTS_DIR, load_examples

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "difficulty"

# One hue, light->dark, red so "more red = more failure" reads straight off the
# plot. Sequential rather than diverging: a failure rate has no meaningful
# midpoint for a two-pole ramp to sit on.
SEQUENTIAL_FAIL = ["#fdf0f0", "#f3aeae", "#e87b7b", "#c33a3a", "#7a2020"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"

# Vertical inches per problem row when problem ids are labelled on the y axis.
# Much below this the labels overlap and stop being readable.
LABEL_ROW_HEIGHT = 0.13


def failure_by_language(task_df: pd.DataFrame, min_models: int) -> pd.DataFrame | None:
    """problem x language matrix of failure rate, or None if nothing qualifies."""
    cell = task_df.groupby(["lang", "local_id"]).agg(
        n_failed=("failed", "sum"),
        n_models=("model", "nunique"),
    )
    cell = cell[cell["n_models"] >= min_models]
    if cell.empty:
        return None
    return (cell["n_failed"] / cell["n_models"]).unstack("lang")


def plot_failure_heatmap(failure: pd.DataFrame, variance: pd.Series, task: str,
                          out_path: Path, max_labelled: int = 200) -> None:
    """Failure-rate heatmap with rows in `variance` order, beside the variance
    profile that defines that order. Both panels share the row order."""
    failure = failure.loc[variance.index]
    # Languages ordered by mean failure so the hardest sit on the left.
    lang_order = failure.mean(axis=0).sort_values(ascending=False).index
    data = failure[lang_order].to_numpy(dtype=float)

    # The figure grows with the row count so every problem id has room to be read.
    # Past `max_labelled` rows that would be an absurdly tall image, so only every
    # k-th id is labelled and the height is capped there.
    n_rows = len(variance)
    step = max(1, -(-n_rows // max_labelled))
    height = max(6.0, LABEL_ROW_HEIGHT * min(n_rows, max_labelled) + 2.0)

    cmap = LinearSegmentedColormap.from_list("seq_fail", SEQUENTIAL_FAIL)
    cmap.set_bad(GRIDLINE)

    fig, (ax, ax_profile) = plt.subplots(
        1, 2, figsize=(0.42 * len(lang_order) + 6.5, height),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.08},
        layout="constrained",  # colorbar-compatible; tight_layout warns on these axes
    )
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, ax_profile):
        a.set_facecolor(SURFACE)
        for spine in a.spines.values():
            spine.set_visible(False)

    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=Normalize(0.0, 1.0),
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(lang_order)))
    ax.set_xticklabels(lang_order, rotation=45, ha="right",
                       color=INK_SECONDARY, fontsize=9)
    ylabel = f"problem id, sorted by cross-language variance  (n={n_rows}"
    ylabel += f", every {step}th shown)" if step > 1 else ")"
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_yticks(range(0, n_rows, step))
    ax.set_yticklabels(variance.index[::step], color=INK_SECONDARY, fontsize=6)
    ax.tick_params(length=0)
    fig.suptitle(f"{task} — failure rate by language, most language-sensitive first",
                 color=INK_PRIMARY, fontsize=11, x=0.01, ha="left")

    # Anchored to the heatmap, not the profile: the profile's x-axis is a different
    # quantity entirely, and a colorbar beside it invites reading the colors off it.
    cbar = fig.colorbar(im, ax=ax, location="top", shrink=0.45, pad=0.02, aspect=40)
    cbar.set_label("failure rate across models  (more red = more failure)",
                   color=INK_MUTED, fontsize=8)
    cbar.ax.tick_params(labelsize=8, color=INK_MUTED, labelcolor=INK_MUTED)
    cbar.outline.set_visible(False)

    # Same row order as the heatmap, so the two panels read across.
    ax_profile.plot(variance.to_numpy(), np.arange(n_rows),
                    color=INK_SECONDARY, linewidth=2)
    ax_profile.set_ylim(n_rows - 0.5, -0.5)
    ax_profile.set_yticks([])
    ax_profile.tick_params(axis="x", labelsize=8, colors=INK_MUTED, length=0)
    ax_profile.grid(axis="x", color=GRIDLINE, linewidth=1)
    ax_profile.set_axisbelow(True)
    ax_profile.set_title("variance across languages", color=INK_SECONDARY,
                         fontsize=9, pad=10, loc="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_failure_heatmaps(examples: pd.DataFrame, out_dir: Path,
                            min_models: int) -> pd.DataFrame:
    examples = examples.copy()
    # Row-wise .apply over ~20k rows is the slowest step in the whole pipeline;
    # zip is the same logic without the per-row Series construction.
    examples["local_id"] = [
        id_[len(lang) + 1:] if id_.startswith(f"{lang}_") else id_
        for id_, lang in zip(examples["id"], examples["lang"])
    ]
    examples["failed"] = (examples["score_accuracy"] == 0).astype(int)

    summary_rows = []
    for task, task_df in examples.groupby("task"):
        failure = failure_by_language(task_df, min_models)
        if failure is None:
            print(f"[{task}] skipped: nothing meets --min-models {min_models}")
            continue

        # ddof=0: this describes the languages actually tested, it is not an
        # estimate over some wider population of languages.
        variance = failure.var(axis=1, ddof=0).dropna().sort_values(ascending=False)
        failure = failure.loc[variance.index]

        task_dir = out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        out = failure.copy()
        out.insert(0, "lang_variance", variance)
        out.insert(1, "mean_failure", failure.mean(axis=1))
        out.index.name = "local_id"
        out.to_csv(task_dir / "failure_by_problem.csv")

        plot_failure_heatmap(failure, variance, task,
                             task_dir / "failure_rate_heatmap.png")

        print(f"[{task}] {len(variance)} problems x {failure.shape[1]} languages; "
              f"variance median {variance.median():.4f}, max {variance.max():.4f}")
        summary_rows.append({
            "task": task,
            "n_problems": len(variance),
            "n_langs": failure.shape[1],
            "mean_failure": round(float(failure.to_numpy().mean()), 4),
            "variance_median": round(float(variance.median()), 4),
            "variance_max": round(float(variance.max()), 4),
            "pct_zero_variance": round(100 * float((variance == 0).mean()), 2),
        })

    summary = pd.DataFrame(summary_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "failure_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-models", type=int, default=2,
                        help="Drop (language, problem) cells with fewer models "
                             "than this (default 2)")
    args = parser.parse_args()

    examples = load_examples(args.results_dir)
    summary = build_failure_heatmaps(examples, Path(args.out_dir), args.min_models)
    print()
    print(summary.to_string(index=False))
    print(f"\nWrote {{task}}/failure_rate_heatmap.png + .csv to {args.out_dir}")


if __name__ == "__main__":
    main()
