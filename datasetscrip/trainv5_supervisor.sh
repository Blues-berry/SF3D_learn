#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-datasetscrip/trainv5_dataset_config.json}"
SESSION="${SESSION:-trainv5_dataset_supervisor}"
LOG_DIR="${LOG_DIR:-output/material_refine_trainV5_auto/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/trainv5_dataset_supervisor.log}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
SUPERVISOR_ARGS="${SUPERVISOR_ARGS:---with-ingest --with-queue --download-objaverse}"

mkdir -p "${LOG_DIR}"

if [[ "${1:-}" == "--once" ]]; then
  shift
  # shellcheck disable=SC2206
  DEFAULT_ARGS=(${SUPERVISOR_ARGS})
  python datasetscrip/trainv5_dataset.py --config "${CONFIG}" supervisor --once "${DEFAULT_ARGS[@]}" "$@"
  exit 0
fi

if tmux has-session -t "${SESSION}" >/dev/null 2>&1; then
  printf '[%s] alive %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "bash -lc 'cd \"${REPO_ROOT}\" && while true; do bash datasetscrip/trainv5_supervisor.sh --once >> \"${LOG_PATH}\" 2>&1 || true; sleep ${INTERVAL_SECONDS}; done'"

printf '[%s] started %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SESSION}"
