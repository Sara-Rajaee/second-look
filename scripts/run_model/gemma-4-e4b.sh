#!/bin/bash
#SBATCH --job-name=sl-gemma-4-e4b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_model_%x_%j.out
# gemma-4-e4b on one RTX A6000 (ILPS). Submit from the repo root:
#   sbatch scripts/run_model/gemma-4-e4b.sh TASK[,TASK...] [run.py args]
# KV cache holds ~1.4M tokens on a 48 GB GPU.
MODEL=gemma-4-e4b
SERVE_ARGS=(--reasoning-parser gemma4)
MAX_MODEL_LEN=40960
CONCURRENCY=64
source "${SLURM_SUBMIT_DIR:-.}/scripts/run_model/_serve_and_eval.sh"
