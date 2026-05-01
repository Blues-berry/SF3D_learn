#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
cd "${REPO_ROOT}"

AUTO_ROOT="${AUTO_ROOT:-output/material_refine_trainV5_auto}"
LOG_DIR="${LOG_DIR:-${AUTO_ROOT}/logs}"
STATUS_DIR="${STATUS_DIR:-${AUTO_ROOT}/status}"
LAST_CYCLE_JSON="${LAST_CYCLE_JSON:-${AUTO_ROOT}/last_cycle_state.json}"

mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

printf '[%s] DEPRECATED: material-priority guard delegates to datasetscrip/trainv5_supervisor.sh\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2

bash datasetscrip/trainv5_supervisor.sh

python scripts/build_material_refine_trainV5_auto_status.py \
  --output-dir "${STATUS_DIR}" \
  --last-cycle-state "${LAST_CYCLE_JSON}" \
  >/dev/null 2>&1 || true
