#!/bin/bash
#SBATCH --job-name=seclook-eval
#SBATCH --partition=gpu_h100
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Serves a model with vLLM and runs the second-look evaluation against it,
# both inside a single sbatch job/node (no .server_endpoint handoff needed).
#
# Usage:
#   sbatch [--gpus-per-task=N] scripts/snellius_eval.sh <vllm_model_id_or_path> <eval_model_name> <task> [extra run.py args...]
#
# Example:
#   sbatch scripts/snellius_eval.sh \
#       /scratch-shared/srajaee/second-look/models/train_runs/sft/qwen_..._20260616_093807 \
#       qwen3-sft-v2 polymath
set -eo pipefail
mkdir -p logs

VLLM_MODEL="${1:?Usage: sbatch snellius_eval.sh <vllm_model_id_or_path> <eval_model_name> <task> [extra run.py args...]}"
EVAL_MODEL="${2:?Missing <eval_model_name> (the --model name as registered in config/models.yaml)}"
TASK="${3:?Missing <task> (the --task name as registered in config/tasks.yaml)}"
shift 3
EXTRA_RUN_ARGS=("$@")

PORT="${PORT:-19743}"

echo "════════════════════════════════════════════════════════════"
echo "  Second-Look serve + eval"
echo "════════════════════════════════════════════════════════════"
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $(hostname -f)"
echo "  vLLM model:  $VLLM_MODEL"
echo "  Eval model:  $EVAL_MODEL"
echo "  Task:        $TASK"
echo "  Port:        $PORT"
echo "════════════════════════════════════════════════════════════"
nvidia-smi

export PATH="/home/srajaee/.conda/envs/second-look/bin:$PATH"
source activate second-look

export HF_HOME="/scratch-shared/srajaee/HF/cache/hub/"
export HF_HUB_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR="${HOME}/.cache/tiktoken-rs-cache"
export TIKTOKEN_ENCODINGS_BASE="${HOME}/.cache/tiktoken-rs-cache/"
export http_proxy="" HTTP_PROXY="" https_proxy="" HTTPS_PROXY=""

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"

TP=$(nvidia-smi -L | wc -l)

echo "Starting vLLM server (TP=${TP}) on port ${PORT}..."
vllm serve "${VLLM_MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --dtype auto \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    > "logs/eval_${SLURM_JOB_ID}_vllm.log" 2>&1 &
VLLM_PID=$!
trap '[ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null' EXIT

echo "Waiting for vLLM server to come up..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM server process died, see logs/eval_${SLURM_JOB_ID}_vllm.log"
        exit 1
    fi
    sleep 5
done
echo "vLLM server ready."

export GPT_OSS_API_BASE="http://localhost:${PORT}/v1/"

echo "Running evaluation: --model ${EVAL_MODEL} --task ${TASK} ${EXTRA_RUN_ARGS[*]}"
python run.py --model "$EVAL_MODEL" --task "$TASK" "${EXTRA_RUN_ARGS[@]}"

kill "$VLLM_PID" 2>/dev/null
