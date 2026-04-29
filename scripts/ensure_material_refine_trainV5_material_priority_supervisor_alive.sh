#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
cd "${REPO_ROOT}"

AUTO_ROOT="${AUTO_ROOT:-output/material_refine_trainV5_auto}"
LOCK_FILE="${LOCK_FILE:-${AUTO_ROOT}/ensure_material_priority_supervisor_alive.lock}"
LOG_DIR="${LOG_DIR:-${AUTO_ROOT}/logs}"
STATUS_DIR="${STATUS_DIR:-${AUTO_ROOT}/status}"
LAST_CYCLE_JSON="${LAST_CYCLE_JSON:-${AUTO_ROOT}/last_cycle_state.json}"

mkdir -p "$(dirname "${LOCK_FILE}")" "${LOG_DIR}" "${STATUS_DIR}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

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
  "trainv5_material_priority_supervisor" \
  "bash scripts/run_material_refine_trainV5_material_priority_supervisor.sh"

python scripts/build_material_refine_trainV5_auto_status.py \
  --output-dir "${STATUS_DIR}" \
  --last-cycle-state "${LAST_CYCLE_JSON}" \
  >/dev/null 2>&1 || true

log "material-priority supervisor guard pass complete"
