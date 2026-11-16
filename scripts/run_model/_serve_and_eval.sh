# Shared part of scripts/run_model/<model>.sh: serve one model with vLLM on the job's
# GPU and run run.py against it. Sourced by the model scripts, which set
#   MODEL          name in config/models.yaml
#   SERVE_ARGS     model-specific `vllm serve` flags
#   MAX_MODEL_LEN  context length to serve
#   CONCURRENCY    requests in flight (sized to the model's KV cache on a 48 GB GPU)
# and are submitted from the repo root:
#   sbatch scripts/run_model/<model>.sh TASK[,TASK...] [run.py args, e.g. --no-reasoning]
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
TASKS="${1:?usage: sbatch scripts/run_model/<model>.sh TASK[,TASK...] [run.py args]}"
shift
RUN_ARGS=("$@")

# --- ILPS cluster settings (override from the environment) ----------------------
PYTHON="${PYTHON:-.venv/bin/python}"                     # environment with the repo's requirements
VLLM_SIF="${VLLM_SIF:-/fnwi_fs/ivi/irlab/personal/stroshi/docker_folder/vllm.sif}"   # vLLM 0.23 image
HF_HUB_DIRS="${HF_HUB_DIRS:-/fnwi_fs/ivi/irlab/personal/stroshi/base_cache/hf_cache/hub /fnwi_fs/ivi/irlab/personal/srajaee/hf_cache/hub}"
LOCAL_CACHE="${LOCAL_CACHE:-/nvmestore/$USER/second_look_cache}"   # node-local disk; "" to skip
export GPT_OSS_TIMEOUT="${GPT_OSS_TIMEOUT:-7200}"        # seconds per request (config/providers.yaml)
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-2400}"

# --- locate the weights: the model's snapshot in one of the Hugging Face caches ---
MODEL_ID=$("$PYTHON" -c "import sys, yaml; print(next(m['model_id'] for m in yaml.safe_load(open('config/models.yaml'))['models'] if m['name'] == sys.argv[1]))" "$MODEL")
MODEL_PATH=""
for hub in $HF_HUB_DIRS; do
    repo="$hub/models--${MODEL_ID//\//--}"
    if [ -f "$repo/refs/main" ]; then MODEL_PATH="$repo/snapshots/$(cat "$repo/refs/main")"; break; fi
done
[ -n "$MODEL_PATH" ] || { echo "weights of $MODEL_ID not found in: $HF_HUB_DIRS"; exit 1; }

# --- copy image and weights to node-local disk, once per node ---------------------
# Group storage reads at ~70 MB/s on the GPU nodes (5-10 min for a 20 GB model);
# later jobs on the same node start from local disk in about 2 minutes.
cache_copy() {   # cache_copy SRC DEST: prints the path to use
    local src="$1" dest="$2" sig
    sig=$(find -L "$src" -type f -printf '%P %s\n' | sort | md5sum | cut -c1-32)
    (
        flock 9
        # recopy when the source changed; an empty marker (older copies) counts as current
        if [ ! -e "$dest.complete" ] || { [ -s "$dest.complete" ] && [ "$(cat "$dest.complete")" != "$sig" ]; }; then
            rm -rf "$dest" "$dest.complete"
            cp -rL "$src" "$dest" && echo "$sig" > "$dest.complete"
        fi
    ) 9> "$dest.lock"
    if [ -e "$dest.complete" ]; then echo "$dest"; else echo "$src"; fi
}
if [ -n "$LOCAL_CACHE" ] && mkdir -p "$LOCAL_CACHE/models" 2>/dev/null; then
    VLLM_SIF=$(cache_copy "$VLLM_SIF" "$LOCAL_CACHE/$(basename "$VLLM_SIF")")
    MODEL_PATH=$(cache_copy "$MODEL_PATH" "$LOCAL_CACHE/models/$MODEL")
fi

# --- serve ----------------------------------------------------------------------
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
LOG_PREFIX="logs/run_model_${MODEL}_${SLURM_JOB_ID:-local}"
mkdir -p logs
echo "job ${SLURM_JOB_ID:-local} on $(hostname -s): $MODEL ($MODEL_ID) from $MODEL_PATH, port $PORT"
export APPTAINER_CACHEDIR="/tmp/$USER/apptainer/cache" APPTAINER_TMPDIR="/tmp/$USER/apptainer/tmp"
export TRITON_CACHE_DIR="/tmp/$USER/triton" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
# run.py asks for the model by its Hugging Face id, so the server answers to that name.
HF_HUB_OFFLINE=1 apptainer exec --nv --bind /fnwi_fs --bind "$(dirname "$MODEL_PATH")" "$VLLM_SIF" \
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL_ID" --host 127.0.0.1 --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.9 --dtype auto \
    "${SERVE_ARGS[@]}" > "${LOG_PREFIX}_vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" ${METRICS_PID:-} 2>/dev/null' EXIT
T0=$(date +%s)
until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died during startup:"; tail -30 "${LOG_PREFIX}_vllm.log"; exit 1; }
    (( $(date +%s) - T0 < STARTUP_TIMEOUT )) || { echo "vLLM startup timed out"; exit 1; }
    sleep 3
done
echo "vLLM ready after $(( $(date +%s) - T0 )) s"

# --- record server load every 30 s: sequences running/waiting, KV-cache use, tokens ---
(
    while sleep 30; do
        echo "$(date +%T) $(curl -s "http://127.0.0.1:${PORT}/metrics" | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total|generation_tokens_total)\{' | sed -E 's/^vllm:([a-z_]+)\{[^}]*\}/\1/' | tr '\n' ' ')"
    done > "${LOG_PREFIX}_metrics.log"
) &
METRICS_PID=$!

# --- evaluate -------------------------------------------------------------------
export GPT_OSS_API_BASE="http://127.0.0.1:${PORT}/v1/"
STATUS=0
for task in ${TASKS//,/ }; do
    # RUN_PY=scripts/run_model/check_run.py runs the same pipeline on a few examples
    "$PYTHON" "${RUN_PY:-run.py}" --model "$MODEL" --task "$task" --concurrency "$CONCURRENCY" "${RUN_ARGS[@]}" || STATUS=1
done
exit $STATUS
