#!/bin/bash
#SBATCH --job-name=sl-eurollm-9b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_model_%x_%j.out
# eurollm-9b on one RTX A6000 (ILPS). Submit from the repo root:
#   sbatch scripts/run_model/eurollm-9b.sh TASK[,TASK...] [run.py args]
# 32K context (the model's limit). KV cache holds only ~150k tokens on a 48 GB GPU.
MODEL=eurollm-9b
SERVE_ARGS=()
MAX_MODEL_LEN=32768
CONCURRENCY=12
source "${SLURM_SUBMIT_DIR:-.}/scripts/run_model/_serve_and_eval.sh"
