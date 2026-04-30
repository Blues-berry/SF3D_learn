#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
cd "${REPO_ROOT}"

AUTO_ROOT="${AUTO_ROOT:-output/material_refine_trainV5_auto}"
LOG_DIR="${LOG_DIR:-${AUTO_ROOT}/logs}"
STATUS_DIR="${STATUS_DIR:-${AUTO_ROOT}/status}"
LOCK_FILE="${LOCK_FILE:-${AUTO_ROOT}/material_priority_cycle.lock}"
LAST_CYCLE_JSON="${LAST_CYCLE_JSON:-${AUTO_ROOT}/last_cycle_state.json}"
STAGE_SUMMARY_JSON="${STAGE_SUMMARY_JSON:-output/material_refine_expansion_candidates/material_priority_stage/material_priority_stage_summary.json}"
OBJAVERSE_INCREMENT_MANIFEST="${OBJAVERSE_INCREMENT_MANIFEST:-output/highlight_pool_a_8k/objaverse_cached_increment_material_priority/objaverse_cached_increment_manifest.json}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
DOWNLOAD_PROBE_SIZE="${DOWNLOAD_PROBE_SIZE:-100}"
MIN_DOWNLOAD_SUCCESS_RATE="${MIN_DOWNLOAD_SUCCESS_RATE:-0.20}"
B_PARALLEL_WORKERS_DRAFT="${B_PARALLEL_WORKERS_DRAFT:-1}"
B_RENDER_RESOLUTION_DRAFT="${B_RENDER_RESOLUTION_DRAFT:-320}"
B_CYCLES_SAMPLES_DRAFT="${B_CYCLES_SAMPLES_DRAFT:-8}"
B_VIEW_LIGHT_PROTOCOL_DRAFT="${B_VIEW_LIGHT_PROTOCOL_DRAFT:-production_32}"

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
  local download_mode="${10}"
  python - <<'PY' "${LAST_CYCLE_JSON}" "${cycle_started}" "${cycle_finished}" "${cycle_log}" "${outcome}" "${stage_rc}" "${batch64_rc}" "${batch256_rc}" "${batch512_rc}" "${batch1000_rc}" "${download_mode}"
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
    "download_mode": sys.argv[11],
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

next_download_mode() {
  python - <<'PY' "${STAGE_SUMMARY_JSON}" "${MIN_DOWNLOAD_SUCCESS_RATE}" "${OBJAVERSE_INCREMENT_MANIFEST}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
threshold = float(sys.argv[2])
increment_path = Path(sys.argv[3])
def load_payload(candidate: Path):
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}
if not path.exists():
    payload = {}
else:
    payload = load_payload(path)
summary = payload.get("summary", payload) if isinstance(payload, dict) else {}
mode = str(summary.get("download_mode") or "")
attempted = int(summary.get("download_attempted") or 0)
success_rate = float(summary.get("download_success_rate") or 0.0)
reason = str(summary.get("download_failure_reason") or "").lower()
if not mode:
    increment = load_payload(increment_path)
    mode = "direct"
    attempted = int(increment.get("selected_count") or attempted or 0)
    downloaded = int(increment.get("downloaded_count") or 0)
    if attempted > 0:
        success_rate = float(downloaded) / float(attempted)
    reason = str(increment.get("download_error") or reason).lower()
network_tokens = ("ssl", "max retries", "connection", "timeout", "proxy", "tls", "http")
network_like = any(token in reason for token in network_tokens)
if attempted > 0 and success_rate >= threshold:
    print("direct")
elif network_like and attempted > 0 and success_rate < threshold:
    if mode == "direct":
        print("proxy-probe")
    elif mode == "proxy-probe":
        print("mirror-probe")
    else:
        print("off")
else:
    print("direct")
PY
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
  serial_launch_rc=0
  download_mode="$(next_download_mode)"

  if stage_process_running; then
    outcome="skipped_existing_stage_process"
    log "skip cycle: existing material-priority stage process is already running"
    write_cycle_state "${cycle_started}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${cycle_log}" "${outcome}" 0 0 0 0 0 "${download_mode}"
    run_status_snapshot
    sleep "${INTERVAL_SECONDS}"
    continue
  fi

  log "starting material-priority cycle mode=${download_mode} -> ${cycle_log}"
  set +e
  (
    flock -n 9 || exit 90
    {
      echo "[cycle ${cycle_stamp}] stage_material_refine_material_priority_sources.py start mode=${download_mode}"
      if [[ "${download_mode}" == "off" ]]; then
        python scripts/stage_material_refine_material_priority_sources.py \
          --download-mode "${download_mode}" \
          --download-probe-size "${DOWNLOAD_PROBE_SIZE}" \
          --min-download-success-rate "${MIN_DOWNLOAD_SUCCESS_RATE}"
      else
        python scripts/stage_material_refine_material_priority_sources.py \
          --download-objaverse \
          --download-mode "${download_mode}" \
          --download-probe-size "${DOWNLOAD_PROBE_SIZE}" \
          --min-download-success-rate "${MIN_DOWNLOAD_SUCCESS_RATE}"
      fi
      echo "[cycle ${cycle_stamp}] stage_material_refine_material_priority_sources.py done"
    } >>"${cycle_log}" 2>&1
  ) 9>"${LOCK_FILE}"
  stage_rc=$?
  set -e

  if [[ "${stage_rc}" -eq 90 ]]; then
    outcome="skipped_lock_busy"
    log "skip cycle: lock busy"
    write_cycle_state "${cycle_started}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${cycle_log}" "${outcome}" "${stage_rc}" 0 0 0 0 "${download_mode}"
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
      --b-parallel-workers "${B_PARALLEL_WORKERS_DRAFT}" \
      --b-render-resolution "${B_RENDER_RESOLUTION_DRAFT}" \
      --b-cycles-samples "${B_CYCLES_SAMPLES_DRAFT}" \
      --b-view-light-protocol "${B_VIEW_LIGHT_PROTOCOL_DRAFT}" \
      >>"${cycle_log}" 2>&1
    batch64_rc=$?

    python scripts/build_material_refine_trainV5_abc.py \
      --run-b-preflight \
      --b-queue output/material_refine_trainV5/expansion_second_pass/batch_1_256_material_first.json \
      --b-batch-name batch_1_256_material_first \
      --b-batch-size 256 \
      --b-expected-record-count 256 \
      --b-parallel-workers "${B_PARALLEL_WORKERS_DRAFT}" \
      --b-render-resolution "${B_RENDER_RESOLUTION_DRAFT}" \
      --b-cycles-samples "${B_CYCLES_SAMPLES_DRAFT}" \
      --b-view-light-protocol "${B_VIEW_LIGHT_PROTOCOL_DRAFT}" \
      >>"${cycle_log}" 2>&1
    batch256_rc=$?

    python scripts/build_material_refine_trainV5_abc.py \
      --run-b-preflight \
      --b-queue output/material_refine_trainV5/expansion_second_pass/batch_1_512_material_first.json \
      --b-batch-name batch_1_512_material_first \
      --b-batch-size 512 \
      --b-expected-record-count 512 \
      --b-parallel-workers "${B_PARALLEL_WORKERS_DRAFT}" \
      --b-render-resolution "${B_RENDER_RESOLUTION_DRAFT}" \
      --b-cycles-samples "${B_CYCLES_SAMPLES_DRAFT}" \
      --b-view-light-protocol "${B_VIEW_LIGHT_PROTOCOL_DRAFT}" \
      >>"${cycle_log}" 2>&1
    batch512_rc=$?

    python scripts/build_material_refine_trainV5_abc.py \
      --run-b-preflight \
      --b-queue output/material_refine_trainV5/expansion_second_pass/batch_2_1000_material_first.json \
      --b-batch-name batch_2_1000_material_first \
      --b-batch-size 1000 \
      --b-expected-record-count 1000 \
      --b-parallel-workers "${B_PARALLEL_WORKERS_DRAFT}" \
      --b-render-resolution "${B_RENDER_RESOLUTION_DRAFT}" \
      --b-cycles-samples "${B_CYCLES_SAMPLES_DRAFT}" \
      --b-view-light-protocol "${B_VIEW_LIGHT_PROTOCOL_DRAFT}" \
      >>"${cycle_log}" 2>&1
    batch1000_rc=$?
    set -e

    if [[ "${batch64_rc}" -ne 0 || "${batch256_rc}" -ne 0 || "${batch512_rc}" -ne 0 || "${batch1000_rc}" -ne 0 ]]; then
      outcome="preflight_failed"
      log "preflight refresh failed rc64=${batch64_rc} rc256=${batch256_rc} rc512=${batch512_rc} rc1000=${batch1000_rc}"
    else
      set +e
      python scripts/maybe_launch_objaverse_1200_serial_after_b1.py >>"${cycle_log}" 2>&1
      serial_launch_rc=$?
      set -e
      if [[ "${serial_launch_rc}" -ne 0 ]]; then
        outcome="serial_launch_gate_failed"
        log "serial launch gate failed rc=${serial_launch_rc}"
      fi
      log "preflight refresh done"
    fi
  fi

  write_cycle_state "${cycle_started}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${cycle_log}" "${outcome}" "${stage_rc}" "${batch64_rc}" "${batch256_rc}" "${batch512_rc}" "${batch1000_rc}" "${download_mode}"
  run_status_snapshot
  sleep "${INTERVAL_SECONDS}"
done
