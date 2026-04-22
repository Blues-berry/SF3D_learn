#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
SESSION="${SESSION:-sf3d_material_refine_paper_stage1}"
PIPELINE_OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT:-output/material_refine_paper/paper_stage1_pipeline}"
WAIT_FOR_READY="${WAIT_FOR_READY:-true}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_POLLS="${MAX_POLLS:-0}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-true}"
GPU_INDEX="${GPU_INDEX:-0}"
MAX_GPU_USED_MB="${MAX_GPU_USED_MB:-4096}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
MANIFEST_GLOB_1="${MANIFEST_GLOB_1:-output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest*.json}"
MANIFEST_GLOB_2="${MANIFEST_GLOB_2:-output/highlight_pool_a_8k/**/canonical_manifest*.json}"
MANIFEST_GLOB_3="${MANIFEST_GLOB_3:-output/material_refine_longrun_stress24_hdri900_*/**/canonical_manifest*.json}"
MANIFEST_GLOB_4="${MANIFEST_GLOB_4:-output/material_refine_dataset_factory/**/canonical_manifest*.json}"

TMUX_BIN="${TMUX_BIN:-}"
if [[ -z "${TMUX_BIN}" ]]; then
  if command -v tunx >/dev/null 2>&1; then
    TMUX_BIN="$(command -v tunx)"
  else
    TMUX_BIN="$(command -v tmux)"
  fi
fi

cd "${REPO_ROOT}"
mkdir -p "${PIPELINE_OUTPUT_ROOT}/logs"
RUN_SCRIPT="${PIPELINE_OUTPUT_ROOT}/logs/${SESSION}.run.sh"
STATUS_PATH="${PIPELINE_OUTPUT_ROOT}/logs/${SESSION}.status.json"

cat > "${RUN_SCRIPT}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_ROOT}"
mkdir -p "${PIPELINE_OUTPUT_ROOT}/logs"

write_status() {
  local status="\$1"
  local exit_code="\$2"
  printf '{"session":"%s","status":"%s","exit_code":%s,"updated_at":"%s","output_root":"%s"}\n' \
    "${SESSION}" "\${status}" "\${exit_code}" "\$(date -Iseconds)" "${PIPELINE_OUTPUT_ROOT}" > "${STATUS_PATH}"
}

on_exit() {
  local exit_code="\$?"
  if [[ "\${exit_code}" -eq 0 ]]; then
    write_status "completed" "\${exit_code}"
  else
    write_status "failed" "\${exit_code}"
  fi
  exit "\${exit_code}"
}
trap on_exit EXIT

write_status "running" 0
if [[ "${WAIT_FOR_GPU}" == "true" ]]; then
  echo "[paper-stage:gpu-wait] waiting for gpu ${GPU_INDEX} used memory <= ${MAX_GPU_USED_MB} MB"
  "${PYTHON_BIN}" - "${GPU_INDEX}" "${MAX_GPU_USED_MB}" "${GPU_POLL_SECONDS}" <<'PY'
import sys
import time

import pynvml

gpu_index = int(sys.argv[1])
max_used_mb = int(sys.argv[2])
poll_seconds = int(sys.argv[3])

pynvml.nvmlInit()
try:
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    while True:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = int(info.used / 1024 / 1024)
        total_mb = int(info.total / 1024 / 1024)
        print(
            f"[paper-stage:gpu-wait] gpu={gpu_index} used_mb={used_mb} total_mb={total_mb} threshold_mb={max_used_mb}",
            flush=True,
        )
        if used_mb <= max_used_mb:
            break
        time.sleep(poll_seconds)
finally:
    pynvml.nvmlShutdown()
PY
fi
if [[ -n "${MANIFEST_PATH}" ]]; then
  "${PYTHON_BIN}" scripts/run_material_refine_paper_stage1_pipeline.py \
    --output-root "${PIPELINE_OUTPUT_ROOT}" \
    --wait-for-ready "${WAIT_FOR_READY}" \
    --poll-seconds "${POLL_SECONDS}" \
    --max-polls "${MAX_POLLS}" \
    --manifest "${MANIFEST_PATH}" 2>&1 | tee "${PIPELINE_OUTPUT_ROOT}/logs/pipeline.log"
else
  "${PYTHON_BIN}" scripts/run_material_refine_paper_stage1_pipeline.py \
    --output-root "${PIPELINE_OUTPUT_ROOT}" \
    --wait-for-ready "${WAIT_FOR_READY}" \
    --poll-seconds "${POLL_SECONDS}" \
    --max-polls "${MAX_POLLS}" \
    --manifest-glob "${MANIFEST_GLOB_1}" \
    --manifest-glob "${MANIFEST_GLOB_2}" \
    --manifest-glob "${MANIFEST_GLOB_3}" \
    --manifest-glob "${MANIFEST_GLOB_4}" 2>&1 | tee "${PIPELINE_OUTPUT_ROOT}/logs/pipeline.log"
fi
SCRIPT

chmod +x "${RUN_SCRIPT}"

if "${TMUX_BIN}" has-session -t "${SESSION}" >/dev/null 2>&1; then
  echo "tmux session already exists: ${SESSION}"
  echo "attach: ${TMUX_BIN} attach -t ${SESSION}"
  echo "status: ${STATUS_PATH}"
  exit 0
fi

"${TMUX_BIN}" new-session -d -s "${SESSION}" "${RUN_SCRIPT}"
echo "started ${SESSION}"
echo "attach: ${TMUX_BIN} attach -t ${SESSION}"
echo "logs: ${PIPELINE_OUTPUT_ROOT}/logs"
echo "status: ${STATUS_PATH}"
