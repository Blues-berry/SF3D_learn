#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
cd "${REPO_ROOT}"

LOCK_FILE="${LOCK_FILE:-output/material_refine_dataset_factory/supervisor_7day/ensure_data_stack_alive.lock}"
LOG_DIR="${LOG_DIR:-output/material_refine_dataset_factory/supervisor_7day}"
MIGRATION_HOLD_FILE="${MIGRATION_HOLD_FILE:-output/material_refine_rebake_v2/LEGACY_DATA_STACK_DISABLED.flag}"
mkdir -p "$(dirname "${LOCK_FILE}")" "${LOG_DIR}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

if [[ -f "${MIGRATION_HOLD_FILE}" ]]; then
  log "legacy data stack hold active at ${MIGRATION_HOLD_FILE}; not restarting old factory/supervisor"
  exit 0
fi

ensure_tmux_session() {
  local session="$1"
  local command="$2"
  if tmux has-session -t "${session}" >/dev/null 2>&1; then
    log "alive ${session}"
    return 0
  fi
  tmux new-session -d -s "${session}" "${command}"
  log "started ${session}"
}

ensure_tmux_session \
  "sf3d_material_refine_dataset_factory_gpu0" \
  "bash output/material_refine_dataset_factory/factory_gpu0_loop.run.sh"

ensure_tmux_session \
  "sf3d_material_refine_dataset_7day_supervisor" \
  "bash output/material_refine_dataset_factory/supervisor_7day/sf3d_material_refine_dataset_7day_supervisor.run.sh"

log "data stack guard pass complete"
