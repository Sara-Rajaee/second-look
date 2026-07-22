#!/bin/bash
#SBATCH --job-name=seclook-eval
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --exclude=ilps-cn[101-115],ilps-cn119

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

export PATH="/home/srajaee/.conda/envs/sec-look/bin:$PATH"
source /home/srajaee/.bashrc
conda activate sec-look

export CUDA_HOME="/usr/local/cuda-12.3"
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/srajaee/anaconda3/lib/
# Pick a free port so concurrent jobs on the same node never collide (unless
# the caller pins one explicitly via PORT=... in the environment).
if [ -z "${PORT}" ]; then
    PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
fi

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

export HF_HOME="/fnwi_fs/ivi/irlab/personal/srajaee/hf_cache"
export TIKTOKEN_RS_CACHE_DIR="${HOME}/.cache/tiktoken-rs-cache"
export TIKTOKEN_ENCODINGS_BASE="${HOME}/.cache/tiktoken-rs-cache/"
export http_proxy="" HTTP_PROXY="" https_proxy="" HTTPS_PROXY=""

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"

TP=$(nvidia-smi -L | wc -l)

REASONING_PARSER_ARGS=()
if [[ "${VLLM_MODEL,,}" == *gemma* ]]; then
    # Gemma 4 wraps reasoning in <|channel>thought ... <channel|> special tokens that
    # get stripped from plain-text output; use vLLM's built-in parser so it lands in
    # the response's reasoning_content field instead.
    REASONING_PARSER_ARGS=(--reasoning-parser gemma4)
fi

echo "Starting vLLM server (TP=${TP}) on port ${PORT}..."
vllm serve "${VLLM_MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --dtype auto \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    "${REASONING_PARSER_ARGS[@]}" \
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
