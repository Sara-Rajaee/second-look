"""Grouped bar charts of quality (accuracy) by language, one subplot per task.

Within each task's subplot, languages are the x-axis groups and models are the
grouped bars, so cross-language and cross-model comparisons are both visible
at a glance.

Usage:
    python scripts/analysis/analysis_quality_bars.py [--results-dir DIR] [--out-dir DIR]

Outputs:
    {out_dir}/quality_by_language.png   one figure, one subplot per task
    {out_dir}/quality_by_language.csv   long-format table backing the figure
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_results import DEFAULT_RESULTS_DIR, load_runs

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "quality_bars"

# Categorical palette, fixed hue order, per dataviz skill references/palette.md.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def plot_quality_by_language(runs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = sorted(runs["task"].unique())
    models = sorted(runs["model"].unique())
    if len(models) > len(CATEGORICAL):
        raise ValueError(f"{len(models)} models but only {len(CATEGORICAL)} palette slots")
    model_color = dict(zip(models, CATEGORICAL))

    ncols = min(2, len(tasks))
    nrows = -(-len(tasks) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.2 * nrows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor(SURFACE)

        task_df = runs[runs["task"] == task]
        langs = sorted(task_df["lang"].unique())
        table = task_df.pivot_table(index="lang", columns="model", values="accuracy").reindex(langs)

        n_models = len(models)
        width = 0.8 / n_models
        x = np.arange(len(langs))
        for i, model in enumerate(models):
            values = table[model].to_numpy() if model in table.columns else np.full(len(langs), np.nan)
            offset = (i - (n_models - 1) / 2) * width
            ax.bar(x + offset, values, width=width * 0.9, color=model_color[model],
                   label=model, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(langs, rotation=45, ha="right", color=INK_SECONDARY, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("accuracy", color=INK_SECONDARY, fontsize=9)
        ax.set_title(task, color=INK_PRIMARY, fontsize=11, loc="left")

        ax.tick_params(axis="y", labelsize=8, colors=INK_MUTED, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for spine_name, spine in ax.spines.items():
            spine.set_visible(spine_name == "bottom")
            if spine_name == "bottom":
                spine.set_color(BASELINE)

    # Hide any unused trailing axes when tasks doesn't fill the grid.
    for idx in range(len(tasks), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(models), 4),
               frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
               bbox_to_anchor=(0.5, -0.02 / nrows))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "quality_by_language.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    long = runs[["task", "model", "lang", "accuracy"]].sort_values(["task", "lang", "model"])
    long.to_csv(out_dir / "quality_by_language.csv", index=False)
    return long


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    table = plot_quality_by_language(runs, Path(args.out_dir))
    print(table.to_string(index=False))
    print(f"\nWrote quality_by_language.png/.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
