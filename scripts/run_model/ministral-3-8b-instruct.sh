#!/bin/bash
#SBATCH --job-name=sl-ministral-3-8b-instruct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_model_%x_%j.out
# ministral-3-8b-instruct on one RTX A6000 (ILPS). Submit from the repo root:
#   sbatch scripts/run_model/ministral-3-8b-instruct.sh TASK[,TASK...] [run.py args]
# The BF16 weights: the default FP8 repo generates garbage on an A6000 (no native FP8).
# Mistral-format weights and tokenizer. --language-model-only: with the vision tower vLLM 0.23
# fails at startup. KV cache ~245k tokens: 24 answers of up to 8,192 tokens fit (estimated).
MODEL=ministral-3-8b-instruct
SERVE_ARGS=(--tokenizer_mode mistral --config_format mistral --load_format mistral --language-model-only)
MAX_MODEL_LEN=40960
CONCURRENCY=24
source "${SLURM_SUBMIT_DIR:-.}/scripts/run_model/_serve_and_eval.sh"
