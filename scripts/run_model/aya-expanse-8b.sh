#!/bin/bash
#SBATCH --job-name=sl-aya-expanse-8b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/run_model_%x_%j.out
# aya-expanse-8b on one RTX A6000 (ILPS). Submit from the repo root:
#   sbatch scripts/run_model/aya-expanse-8b.sh TASK[,TASK...] [run.py args]
# 8K context in total (the model's limit).
MODEL=aya-expanse-8b
SERVE_ARGS=()
MAX_MODEL_LEN=8192
CONCURRENCY=64
source "${SLURM_SUBMIT_DIR:-.}/scripts/run_model/_serve_and_eval.sh"
