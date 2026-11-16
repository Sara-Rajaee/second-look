#!/bin/bash
#SBATCH --job-name=sl-qwen3.5-4b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_model_%x_%j.out
# qwen3.5-4b on one RTX A6000 (ILPS). Submit from the repo root:
#   sbatch scripts/run_model/qwen3.5-4b.sh TASK[,TASK...] [run.py args]
# KV cache holds ~1.0M tokens on a 48 GB GPU, so 64 long generations fit.
# --language-model-only: text only, the vision tower is not loaded.
MODEL=qwen3.5-4b
SERVE_ARGS=(--reasoning-parser qwen3 --language-model-only)
MAX_MODEL_LEN=40960
CONCURRENCY=64
source "${SLURM_SUBMIT_DIR:-.}/scripts/run_model/_serve_and_eval.sh"
