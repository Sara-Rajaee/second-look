#!/usr/bin/env bash

# Runs every analysis in scripts/analysis/ sequentially against one results dir,
# one output subfolder per script. Plots and summaries go to OUT_ROOT in the repo;
# the bulkier XOR artifacts go to XOR_OUT_ROOT on the group filesystem.
#
# Usage:
#   scripts/run_analysis.sh [OUT_ROOT] [RESULTS_DIR]
#   XOR_OUT_ROOT=/some/other/path scripts/run_analysis.sh
#
# Example:
#   scripts/run_analysis.sh outputs/run_2026_07_29

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

OUT_ROOT="${1:-outputs}"
RESULTS_DIR="${2:-/fnwi_fs/ivi/irlab/personal/srajaee/second-look/results_analysis}"
PY=.venv/bin/python

export MPLBACKEND=Agg  # headless: write PNGs without needing a display

# The XOR artifacts are large and shared, so they live on the group filesystem
# rather than in the repo checkout. Override with XOR_OUT_ROOT=... if needed.
XOR_OUT_ROOT="${XOR_OUT_ROOT:-/fnwi_fs/ivi/irlab/personal/stroshi/second_look/outputs/xor}"

# script name -> output dir; every script takes --results-dir and --out-dir.
ANALYSES=(
    "analysis_quality_bars:$OUT_ROOT/quality_bars"
    "analysis_heat_map:$OUT_ROOT/heatmaps"
    "analysis_difficulty_heatmap:$OUT_ROOT/difficulty"
    "analysis_xor_stats:$XOR_OUT_ROOT/stats"
    "analysis_xor_examples:$XOR_OUT_ROOT/examples"
    "check_consistency:$OUT_ROOT/consistency"
)

echo "results dir: $RESULTS_DIR"
echo "output root: $OUT_ROOT"
echo "xor outputs: $XOR_OUT_ROOT"

for entry in "${ANALYSES[@]}"; do
    script="${entry%%:*}"
    dest="${entry#*:}"

    echo
    echo "──────── $script ────────"

    $PY "scripts/analysis/$script.py" \
        --results-dir "$RESULTS_DIR" \
        --out-dir "$dest"
done

mkdir -p "$OUT_ROOT"
cat > "$OUT_ROOT/run_metadata.json" <<EOF
{
  "results_dir": "$RESULTS_DIR",
  "generated_at": "$(date -Is)",
  "git_rev": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "host": "$(hostname -f)"
}
EOF

echo
echo "──────── outputs ────────"
for entry in "${ANALYSES[@]}"; do
    script="${entry%%:*}"
    dest="${entry#*:}"
    printf '%-22s %s (%s files)\n' \
        "$script" "$dest" "$(find "$dest" -type f | wc -l)"
done
echo "run_metadata.json -> $OUT_ROOT/run_metadata.json"
