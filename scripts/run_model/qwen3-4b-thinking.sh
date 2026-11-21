#!/bin/bash
#SBATCH --job-name=sl-qwen3-4b-thinking
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/run_model_%x_%j.out
# qwen3-4b-thinking on one RTX A6000 (ILPS). Submit from the repo root:
#   sbatch scripts/run_model/qwen3-4b-thinking.sh TASK[,TASK...] [run.py args]
# Always thinks. KV cache holds only ~250k tokens on a 48 GB GPU. Measured on PolyMath high:
# 326 tok/s at 16 in flight, 292 at 22, 240 at 64 (the server evicts and recomputes).
MODEL=qwen3-4b-thinking
SERVE_ARGS=(--reasoning-parser qwen3)
MAX_MODEL_LEN=40960
CONCURRENCY=16
source "${SLURM_SUBMIT_DIR:-.}/scripts/run_model/_serve_and_eval.sh"
