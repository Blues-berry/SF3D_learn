#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
cd "${REPO_ROOT}"

printf '[%s] DEPRECATED: scripts/run_material_refine_trainV5_material_priority_supervisor.sh delegates to datasetscrip/trainv5_dataset.py supervisor\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2

CONFIG="${CONFIG:-datasetscrip/trainv5_dataset_config.json}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
SUPERVISOR_ARGS="${SUPERVISOR_ARGS:---with-ingest --with-queue --download-objaverse}"

# Compatibility entrypoint for old tmux/guard sessions. With no arguments,
# only ensure the new datasetscrip supervisor is alive; this prevents legacy
# outer 30-second loops from repeatedly running full ingest/queue cycles.
if [[ "$#" -eq 0 ]]; then
  exec bash datasetscrip/trainv5_supervisor.sh
fi

# Explicit args are treated as a one-cycle debugging/compatibility invocation.
# shellcheck disable=SC2206
DEFAULT_ARGS=(${SUPERVISOR_ARGS})
exec python datasetscrip/trainv5_dataset.py \
  --config "${CONFIG}" \
  supervisor \
  --once \
  --interval-seconds "${INTERVAL_SECONDS}" \
  "${DEFAULT_ARGS[@]}" \
  "$@"
