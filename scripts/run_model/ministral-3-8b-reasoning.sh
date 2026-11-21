#!/bin/bash
#SBATCH --job-name=sl-ministral-3-8b-reasoning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/run_model_%x_%j.out
# ministral-3-8b-reasoning on one RTX A6000 (ILPS). Submit from the repo root:
#   sbatch scripts/run_model/ministral-3-8b-reasoning.sh TASK[,TASK...] [run.py args]
# Always reasons, given its system prompt (config/models.yaml: system_prompt_file).
# Mistral-format weights and tokenizer. --language-model-only: with the vision tower vLLM 0.23
# fails at startup. KV cache holds only ~190k tokens, about 8 long generations.
MODEL=ministral-3-8b-reasoning
SERVE_ARGS=(--tokenizer_mode mistral --config_format mistral --load_format mistral --reasoning-parser mistral --language-model-only)
MAX_MODEL_LEN=40960
CONCURRENCY=10
source "${SLURM_SUBMIT_DIR:-.}/scripts/run_model/_serve_and_eval.sh"
