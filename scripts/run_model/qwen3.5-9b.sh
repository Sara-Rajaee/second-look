#!/bin/bash
#SBATCH --job-name=sl-qwen3.5-9b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/run_model_%x_%j.out
# qwen3.5-9b on one RTX A6000 (ILPS). Submit from the repo root:
#   sbatch scripts/run_model/qwen3.5-9b.sh TASK[,TASK...] [run.py args]
# KV cache holds ~730k tokens on a 48 GB GPU.
# --language-model-only: text only, the vision tower is not loaded.
MODEL=qwen3.5-9b
SERVE_ARGS=(--reasoning-parser qwen3 --language-model-only)
MAX_MODEL_LEN=40960
CONCURRENCY=48
source "${SLURM_SUBMIT_DIR:-.}/scripts/run_model/_serve_and_eval.sh"
