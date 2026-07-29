"""Pairwise cross-language agreement heatmaps: for each (task, model), how many
examples does the model answer correctly in both languages vs. only one?

Examples are aligned across languages via their trailing index (id = f"{lang}_{index}"),
since each language subset is a parallel/translated version of the same underlying items.

Usage:
    python scripts/analysis/analysis_heat_map.py [--results-dir DIR] [--out-dir DIR]

Outputs, per (task, model), one CSV matrix and one matching PNG heatmap for
each of four quadrants (languages x languages):
    {out_dir}/{task}/{model}/both_correct.csv / .png     both langs correct
    {out_dir}/{task}/{model}/both_wrong.csv / .png       both langs wrong
    {out_dir}/{task}/{model}/only_a_correct.csv / .png   row lang correct, col lang wrong
    {out_dir}/{task}/{model}/only_b_correct.csv / .png   col lang correct, row lang wrong
Plus one long-format summary across all pairs:
    {out_dir}/pairwise_summary.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_results import DEFAULT_RESULTS_DIR, load_examples, local_id

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "heatmaps"

# One-hue light->dark sequential ramps, one per quadrant, each taking the next
# categorical slot's hue per dataviz skill references/palette.md (blue, orange,
# aqua, yellow — slots 1-4). Blue and orange are the palette's own stops;
# aqua/yellow are generated with the same white/black mix ratios as orange's.
QUADRANT_RAMPS = {
    "both_correct": ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"],
    "both_wrong": ["#fbe4d8", "#f6c1a1", "#ef9a63", "#c96a2e", "#7a3f16"],
    "only_a_correct": ["#d4f0e6", "#9bdcc4", "#59c59e", "#189a6b", "#0e5d41"],
    "only_b_correct": ["#fcedcf", "#f7d68f", "#f2ba45", "#d18e00", "#7e5500"],
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"


def pairwise_counts(wide: pd.DataFrame, lang_a: str, lang_b: str) -> dict:
    pair = wide[[lang_a, lang_b]].dropna()
    a, b = pair[lang_a], pair[lang_b]
    return {
        "n": len(pair),
        "both_correct": int(((a == 1) & (b == 1)).sum()),
        "both_wrong": int(((a == 0) & (b == 0)).sum()),
        "only_a_correct": int(((a == 1) & (b == 0)).sum()),
        "only_b_correct": int(((a == 0) & (b == 1)).sum()),
    }


def plot_heatmap(matrix: pd.DataFrame, title: str, ramp: list[str], out_path: Path) -> None:
    """Render a languages x languages count matrix as a PNG heatmap.

    The diagonal is masked: it holds a different quantity (single-language
    correct count) than the off-diagonal pairwise counts, so including it
    would skew the color scale.
    """
    langs = list(matrix.columns)
    data = matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(data, np.nan)
    masked = np.ma.masked_invalid(data)

    cmap = LinearSegmentedColormap.from_list("seq", ramp)
    cmap.set_bad(SURFACE)

    fig, ax = plt.subplots(figsize=(0.55 * len(langs) + 2, 0.55 * len(langs) + 1.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(masked, cmap=cmap, aspect="equal")

    ax.set_xticks(range(len(langs)))
    ax.set_yticks(range(len(langs)))
    ax.set_xticklabels(langs, rotation=45, ha="right", color=INK_SECONDARY, fontsize=9)
    ax.set_yticklabels(langs, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(langs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(langs), 1), minor=True)
    ax.grid(which="minor", color=GRIDLINE, linewidth=1)
    ax.tick_params(which="minor", length=0)

    vmax = np.nanmax(data) if np.isfinite(data).any() else 1
    for i in range(len(langs)):
        for j in range(len(langs)):
            if i == j:
                continue
            value = data[i, j]
            label_ink = SURFACE if value > 0.6 * vmax else INK_PRIMARY
            ax.text(j, i, f"{int(value)}", ha="center", va="center",
                     color=label_ink, fontsize=8)

    ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=12, loc="left")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8, color=INK_MUTED, labelcolor=INK_MUTED)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_heatmaps(examples: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = examples.copy()
    examples["local_id"] = examples.apply(local_id, axis=1)

    summary_rows = []
    for (task, model), group in examples.groupby(["task", "model"]):
        wide = group.pivot_table(
            index="local_id", columns="lang", values="score_accuracy", aggfunc="first"
        )
        langs = sorted(wide.columns)
        if len(langs) < 2:
            continue

        quadrants = {
            name: pd.DataFrame(0, index=langs, columns=langs, dtype=int)
            for name in QUADRANT_RAMPS
        }

        for lang_a in langs:
            for lang_b in langs:
                if lang_a == lang_b:
                    valid = wide[lang_a].dropna()
                    quadrants["both_correct"].loc[lang_a, lang_b] = int((valid == 1).sum())
                    quadrants["both_wrong"].loc[lang_a, lang_b] = int((valid == 0).sum())
                    continue
                c = pairwise_counts(wide, lang_a, lang_b)
                quadrants["both_correct"].loc[lang_a, lang_b] = c["both_correct"]
                quadrants["both_wrong"].loc[lang_a, lang_b] = c["both_wrong"]
                quadrants["only_a_correct"].loc[lang_a, lang_b] = c["only_a_correct"]
                quadrants["only_b_correct"].loc[lang_a, lang_b] = c["only_b_correct"]
                if lang_a < lang_b:
                    summary_rows.append({"task": task, "model": model,
                                          "lang_a": lang_a, "lang_b": lang_b, **c})

        titles = {
            "both_correct": "both correct",
            "both_wrong": "both wrong",
            "only_a_correct": "row language correct, column wrong",
            "only_b_correct": "column language correct, row wrong",
        }
        pair_dir = out_dir / task / model.replace("/", "_")
        pair_dir.mkdir(parents=True, exist_ok=True)
        for name, matrix in quadrants.items():
            matrix.to_csv(pair_dir / f"{name}.csv")
            plot_heatmap(matrix, f"{task} / {model} — {titles[name]}",
                         QUADRANT_RAMPS[name], pair_dir / f"{name}.png")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "pairwise_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    examples = load_examples(args.results_dir)
    summary = build_heatmaps(examples, Path(args.out_dir))
    print(summary.to_string(index=False))
    print(f"\nWrote {{task}}/{{model}}/ matrices and pairwise_summary.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
