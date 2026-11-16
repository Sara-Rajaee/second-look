#!/bin/bash
#SBATCH --job-name=sl-gemma-4-12b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/run_model_%x_%j.out
# gemma-4-12b on one RTX A6000 (ILPS). Submit from the repo root:
#   sbatch scripts/run_model/gemma-4-12b.sh TASK[,TASK...] [run.py args]
# KV cache is 18 GiB at the default memory fraction, 20 GiB at 0.95. Measured on PolyMath high
# with thinking: 462 tok/s at 0.95 with 38 in flight, 440 at 0.9 with 40 (evicting), 391 with 28.
MODEL=gemma-4-12b
SERVE_ARGS=(--reasoning-parser gemma4)
MAX_MODEL_LEN=40960
CONCURRENCY=38
GPU_MEMORY_UTILIZATION=0.95
source "${SLURM_SUBMIT_DIR:-.}/scripts/run_model/_serve_and_eval.sh"
