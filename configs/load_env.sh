# configs/load_env.sh
# Generic entry point: figures out the current node name and sources that
# node's generated artifact, configs/.node_envs/<node_name>/load_env.sh
# (see configs/derive_paths.sh, which writes it). Does NOT read paths.env,
# run uv sync, or touch any lock -- so it's safe to source inside any
# interactive session; there's nothing here that should ever fail once the
# artifact exists.
#
# Artifacts are keyed by node name (not $SHARED) because $SHARED itself can
# resolve differently per node, so run `bash configs/derive_paths.sh` once
# on EACH node you use to create that node's artifact.

load_env_main() {
  local -
  set -euo pipefail

  local node_dir="configs/.node_envs/$(hostname -s)"
  local env_artifact="$node_dir/load_env.sh"
  if [[ ! -f "$env_artifact" ]]; then
    echo "ERROR: $env_artifact not found." >&2
    echo "Run this once on this node first: bash configs/derive_paths.sh" >&2
    return 1
  fi

  source "$env_artifact"
  echo "Loaded env from $env_artifact"
}

if ! load_env_main; then
  unset -f load_env_main
  return 1 2>/dev/null || exit 1
fi
unset -f load_env_main
