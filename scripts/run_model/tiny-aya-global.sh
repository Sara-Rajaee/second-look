#!/bin/bash
#SBATCH --job-name=sl-tiny-aya-global
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/run_model_%x_%j.out
# tiny-aya-global on one RTX A6000 (ILPS). Submit from the repo root:
#   sbatch scripts/run_model/tiny-aya-global.sh TASK[,TASK...] [run.py args]
# 8K context in total (the model's limit; vLLM refuses to serve more).
MODEL=tiny-aya-global
SERVE_ARGS=()
MAX_MODEL_LEN=8192
CONCURRENCY=64
source "${SLURM_SUBMIT_DIR:-.}/scripts/run_model/_serve_and_eval.sh"
