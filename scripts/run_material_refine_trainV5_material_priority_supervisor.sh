#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
cd "${REPO_ROOT}"

AUTO_ROOT="${AUTO_ROOT:-output/material_refine_trainV5_auto}"
LOG_DIR="${LOG_DIR:-${AUTO_ROOT}/logs}"
STATUS_DIR="${STATUS_DIR:-${AUTO_ROOT}/status}"
LOCK_FILE="${LOCK_FILE:-${AUTO_ROOT}/material_priority_cycle.lock}"
LAST_CYCLE_JSON="${LAST_CYCLE_JSON:-${AUTO_ROOT}/last_cycle_state.json}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
DOWNLOAD_FLAG="${DOWNLOAD_FLAG:---download-objaverse}"

mkdir -p "${LOG_DIR}" "${STATUS_DIR}" "$(dirname "${LOCK_FILE}")"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

write_cycle_state() {
  local cycle_started="$1"
  local cycle_finished="$2"
  local cycle_log="$3"
  local outcome="$4"
  local stage_rc="$5"
  local batch64_rc="$6"
  local batch256_rc="$7"
  local batch512_rc="$8"
  local batch1000_rc="$9"
  python - <<'PY' "${LAST_CYCLE_JSON}" "${cycle_started}" "${cycle_finished}" "${cycle_log}" "${outcome}" "${stage_rc}" "${batch64_rc}" "${batch256_rc}" "${batch512_rc}" "${batch1000_rc}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "cycle_started_utc": sys.argv[2],
    "cycle_finished_utc": sys.argv[3],
    "cycle_log": sys.argv[4],
    "outcome": sys.argv[5],
    "stage_returncode": int(sys.argv[6]),
    "batch_0_64_preflight_returncode": int(sys.argv[7]),
    "batch_1_256_preflight_returncode": int(sys.argv[8]),
    "batch_1_512_preflight_returncode": int(sys.argv[9]),
    "batch_2_1000_preflight_returncode": int(sys.argv[10]),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
}

run_status_snapshot() {
  python scripts/build_material_refine_trainV5_auto_status.py \
    --output-dir "${STATUS_DIR}" \
    --last-cycle-state "${LAST_CYCLE_JSON}" \
    >/dev/null 2>&1 || true
}

stage_process_running() {
  pgrep -af "python scripts/stage_material_refine_material_priority_sources.py" >/dev/null 2>&1
}

while true; do
  cycle_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cycle_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  cycle_log="${LOG_DIR}/material_priority_cycle_${cycle_stamp}.log"
  outcome="success"
  stage_rc=0
  batch64_rc=0
  batch256_rc=0
  batch512_rc=0
  batch1000_rc=0

  if stage_process_running; then
    outcome="skipped_existing_stage_process"
    log "skip cycle: existing material-priority stage process is already running"
    write_cycle_state "${cycle_started}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${cycle_log}" "${outcome}" 0 0 0 0 0
    run_status_snapshot
    sleep "${INTERVAL_SECONDS}"
    continue
  fi

  log "starting material-priority cycle -> ${cycle_log}"
  set +e
  (
    flock -n 9 || exit 90
    {
      echo "[cycle ${cycle_stamp}] stage_material_refine_material_priority_sources.py start"
      python scripts/stage_material_refine_material_priority_sources.py ${DOWNLOAD_FLAG}
      echo "[cycle ${cycle_stamp}] stage_material_refine_material_priority_sources.py done"
    } >>"${cycle_log}" 2>&1
  ) 9>"${LOCK_FILE}"
  stage_rc=$?
  set -e

  if [[ "${stage_rc}" -eq 90 ]]; then
    outcome="skipped_lock_busy"
    log "skip cycle: lock busy"
    write_cycle_state "${cycle_started}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${cycle_log}" "${outcome}" "${stage_rc}" 0 0 0 0
    run_status_snapshot
    sleep "${INTERVAL_SECONDS}"
    continue
  fi

  if [[ "${stage_rc}" -ne 0 ]]; then
    outcome="stage_failed"
    log "stage cycle failed rc=${stage_rc}"
  else
    log "stage cycle finished, refreshing B-track preflight drafts"
    set +e
    python scripts/build_material_refine_trainV5_abc.py \
      --run-b-preflight \
      --b-queue output/material_refine_trainV5/expansion_second_pass/batch_0_64_material_first.json \
      --b-batch-name batch_0_64_material_first \
      --b-batch-size 64 \
      --b-expected-record-count 64 \
      >>"${cycle_log}" 2>&1
    batch64_rc=$?

    python scripts/build_material_refine_trainV5_abc.py \
      --run-b-preflight \
      --b-queue output/material_refine_trainV5/expansion_second_pass/batch_1_256_material_first.json \
      --b-batch-name batch_1_256_material_first \
      --b-batch-size 256 \
      --b-expected-record-count 256 \
      >>"${cycle_log}" 2>&1
    batch256_rc=$?

    python scripts/build_material_refine_trainV5_abc.py \
      --run-b-preflight \
      --b-queue output/material_refine_trainV5/expansion_second_pass/batch_1_512_material_first.json \
      --b-batch-name batch_1_512_material_first \
      --b-batch-size 512 \
      --b-expected-record-count 512 \
      >>"${cycle_log}" 2>&1
    batch512_rc=$?

    python scripts/build_material_refine_trainV5_abc.py \
      --run-b-preflight \
      --b-queue output/material_refine_trainV5/expansion_second_pass/batch_2_1000_material_first.json \
      --b-batch-name batch_2_1000_material_first \
      --b-batch-size 1000 \
      --b-expected-record-count 1000 \
      >>"${cycle_log}" 2>&1
    batch1000_rc=$?
    set -e

    if [[ "${batch64_rc}" -ne 0 || "${batch256_rc}" -ne 0 || "${batch512_rc}" -ne 0 || "${batch1000_rc}" -ne 0 ]]; then
      outcome="preflight_failed"
      log "preflight refresh failed rc64=${batch64_rc} rc256=${batch256_rc} rc512=${batch512_rc} rc1000=${batch1000_rc}"
    else
      log "preflight refresh done"
    fi
  fi

  write_cycle_state "${cycle_started}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${cycle_log}" "${outcome}" "${stage_rc}" "${batch64_rc}" "${batch256_rc}" "${batch512_rc}" "${batch1000_rc}"
  run_status_snapshot
  sleep "${INTERVAL_SECONDS}"
done
