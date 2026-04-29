#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
cd "${REPO_ROOT}"

AUTO_ROOT="${AUTO_ROOT:-output/material_refine_trainV5_auto}"
LOG_DIR="${LOG_DIR:-${AUTO_ROOT}/logs}"
STATUS_DIR="${STATUS_DIR:-${AUTO_ROOT}/status}"
LAST_CYCLE_JSON="${LAST_CYCLE_JSON:-${AUTO_ROOT}/last_cycle_state.json}"
GUARD_INTERVAL_SECONDS="${GUARD_INTERVAL_SECONDS:-300}"
GUARD_LOG="${GUARD_LOG:-${LOG_DIR}/material_priority_guard.log}"

mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

bash scripts/ensure_material_refine_trainV5_material_priority_supervisor_alive.sh

if tmux has-session -t "trainv5_material_priority_guard" >/dev/null 2>&1; then
  log "guard session already alive"
else
  tmux new-session -d -s "trainv5_material_priority_guard" \
    "bash -lc 'cd \"${REPO_ROOT}\" && while true; do bash scripts/ensure_material_refine_trainV5_material_priority_supervisor_alive.sh >> \"${GUARD_LOG}\" 2>&1; sleep ${GUARD_INTERVAL_SECONDS}; done'"
  log "started trainv5_material_priority_guard"
fi

python scripts/build_material_refine_trainV5_auto_status.py \
  --output-dir "${STATUS_DIR}" \
  --last-cycle-state "${LAST_CYCLE_JSON}" \
  >/dev/null 2>&1 || true

log "trainV5 material-priority automation ready"
