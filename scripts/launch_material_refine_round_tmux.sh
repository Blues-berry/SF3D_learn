#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
PYTHON="${PYTHON:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
SESSION="${SESSION:-sf3d_material_refine_round2_gpu0}"
GPU_INDEX="${GPU_INDEX:-0}"
MAX_GPU_USED_MB="${MAX_GPU_USED_MB:-4096}"
POLL_SECONDS="${POLL_SECONDS:-60}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/material_refine_train_round2_gpu0_gated.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/material_refine_eval_round2_gpu0.yaml}"
MANIFEST="${MANIFEST:-output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_full.json}"
TRAIN_DIR="${TRAIN_DIR:-output/material_refine_pipeline_20260418T091559Z/train_round2_gpu0_gated}"
EVAL_DIR="${EVAL_DIR:-output/material_refine_pipeline_20260418T091559Z/eval_round2_gpu0_test}"
AUDIT_DIR="${AUDIT_DIR:-output/material_refine_pipeline_20260418T091559Z/manifest_audit_round2}"
ROUND_ANALYSIS_DIR="${ROUND_ANALYSIS_DIR:-output/material_refine_pipeline_20260418T091559Z/round2_analysis}"
PANEL_DIR="${PANEL_DIR:-${EVAL_DIR}/validation_comparison_panels}"

TMUX_BIN="${TMUX_BIN:-}"
if [[ -z "${TMUX_BIN}" ]]; then
  if command -v tunx >/dev/null 2>&1; then
    TMUX_BIN="$(command -v tunx)"
  else
    TMUX_BIN="$(command -v tmux)"
  fi
fi

cd "${REPO_ROOT}"
mkdir -p "${TRAIN_DIR}/logs"
RUN_SCRIPT="${TRAIN_DIR}/logs/${SESSION}.run.sh"
STATUS_PATH="${TRAIN_DIR}/logs/${SESSION}.status.json"

cat > "${RUN_SCRIPT}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_ROOT}"
mkdir -p "${TRAIN_DIR}/logs" "${EVAL_DIR}" "${AUDIT_DIR}" "${ROUND_ANALYSIS_DIR}" "${PANEL_DIR}"

write_status() {
  local status="\$1"
  local exit_code="\$2"
  printf '{"session":"%s","status":"%s","exit_code":%s,"updated_at":"%s","train_config":"%s","eval_config":"%s"}\n' \
    "${SESSION}" "\${status}" "\${exit_code}" "\$(date -Iseconds)" "${TRAIN_CONFIG}" "${EVAL_CONFIG}" > "${STATUS_PATH}"
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
echo "[round] session=${SESSION} repo=${REPO_ROOT}"
echo "[round] train_config=${TRAIN_CONFIG} eval_config=${EVAL_CONFIG}"

echo "[round:preflight] checking manifest and training contract"
"${PYTHON}" scripts/train_material_refiner.py --config "${TRAIN_CONFIG}" --preflight-only 2>&1 | tee "${TRAIN_DIR}/logs/preflight.log"

echo "[round:gpu-wait] waiting for gpu ${GPU_INDEX} used memory <= ${MAX_GPU_USED_MB} MB"
"${PYTHON}" - "${GPU_INDEX}" "${MAX_GPU_USED_MB}" "${POLL_SECONDS}" <<'PY'
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
            f"[gpu-wait] gpu={gpu_index} used_mb={used_mb} total_mb={total_mb} threshold_mb={max_used_mb}",
            flush=True,
        )
        if used_mb <= max_used_mb:
            break
        time.sleep(poll_seconds)
finally:
    pynvml.nvmlShutdown()
PY

echo "[round:train] launching material refiner training"
"${PYTHON}" scripts/train_material_refiner.py --config "${TRAIN_CONFIG}" 2>&1 | tee "${TRAIN_DIR}/logs/train.log"

CHECKPOINT="${TRAIN_DIR}/best.pt"
if [[ ! -e "\${CHECKPOINT}" ]]; then
  CHECKPOINT="${TRAIN_DIR}/latest.pt"
fi
if [[ ! -e "\${CHECKPOINT}" ]]; then
  echo "[round:error] checkpoint not found in ${TRAIN_DIR}" >&2
  exit 2
fi

echo "[round:eval] checkpoint=\${CHECKPOINT}"
"${PYTHON}" scripts/eval_material_refiner.py --config "${EVAL_CONFIG}" --checkpoint "\${CHECKPOINT}" 2>&1 | tee "${TRAIN_DIR}/logs/eval.log"

echo "[round:attributes] exporting SF3D baseline vs refined attribute metrics"
"${PYTHON}" scripts/export_material_attribute_comparison.py \
  --metrics-json "${EVAL_DIR}/metrics.json" \
  --output-dir "${EVAL_DIR}" 2>&1 | tee "${TRAIN_DIR}/logs/attribute_comparison.log"

echo "[round:panels] exporting validation inference comparison panels"
"${PYTHON}" scripts/export_material_validation_comparison_panels.py \
  --manifest "${MANIFEST}" \
  --metrics "${EVAL_DIR}/metrics.json" \
  --output-dir "${PANEL_DIR}" \
  --max-panels 64 \
  --report-to wandb \
  --tracker-run-name "material-refine-round2-validation-panels" \
  --tracker-group "material-refine-round2" \
  --tracker-tags "material-refine,sf3d,round2,validation-panels" \
  --wandb-mode online 2>&1 | tee "${TRAIN_DIR}/logs/validation_panels.log"

echo "[round:audit] auditing manifest"
"${PYTHON}" scripts/audit_material_refine_manifest.py \
  --manifest "${MANIFEST}" \
  --output-dir "${AUDIT_DIR}" \
  --max-records -1 2>&1 | tee "${TRAIN_DIR}/logs/manifest_audit.log"

echo "[round:analysis] exporting round analysis"
"${PYTHON}" scripts/export_material_refine_round_analysis.py \
  --train-dir "${TRAIN_DIR}" \
  --eval-dir "${EVAL_DIR}" \
  --audit-dir "${AUDIT_DIR}" \
  --output-dir "${ROUND_ANALYSIS_DIR}" 2>&1 | tee "${TRAIN_DIR}/logs/round_analysis.log"

echo "[round:wandb-summary] logging final round summary"
"${PYTHON}" scripts/log_material_refine_round_to_wandb.py \
  --train-dir "${TRAIN_DIR}" \
  --eval-dir "${EVAL_DIR}" \
  --audit-dir "${AUDIT_DIR}" \
  --round-analysis-dir "${ROUND_ANALYSIS_DIR}" \
  --panel-dir "${PANEL_DIR}" \
  --tracker-run-name "material-refine-round2-summary" \
  --tracker-group "material-refine-round2" \
  --tracker-tags "material-refine,sf3d,round2,summary" \
  --wandb-mode online 2>&1 | tee "${TRAIN_DIR}/logs/wandb_round_summary.log"

echo "[round:done] outputs: ${TRAIN_DIR} ${EVAL_DIR} ${PANEL_DIR} ${ROUND_ANALYSIS_DIR}"
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
echo "logs: ${TRAIN_DIR}/logs"
echo "status: ${STATUS_PATH}"
