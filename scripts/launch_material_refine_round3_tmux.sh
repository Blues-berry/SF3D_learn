#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-sf3d_material_refine_round3_gpu0_20260421}"
CONFIG="${CONFIG:-configs/material_refine_train_paper_stage1_round3.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/material_refine_paper/paper_stage1_round3_20260421}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_stage1_round3}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_ROOT}/logs"

tmux new-session -d -s "${SESSION}" "
cd /home/ubuntu/ssd_work/projects/stable-fast-3d
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +e
${PYTHON_BIN} scripts/train_material_refiner.py \
  --config ${CONFIG} \
  --output-dir ${TRAIN_OUTPUT_DIR} \
  ${EXTRA_ARGS} \
  2>&1 | tee ${OUTPUT_ROOT}/logs/train.log
status=\${PIPESTATUS[0]}
if [ \${status} -eq 0 ]; then state=completed; else state=failed; fi
printf '{\"session\":\"%s\",\"status\":\"%s\",\"exit_code\":%s,\"output_dir\":\"%s\"}\n' \
  '${SESSION}' \"\${state}\" \"\${status}\" '${TRAIN_OUTPUT_DIR}' \
  > ${OUTPUT_ROOT}/logs/status.json
exit \${status}
"

echo "session=${SESSION}"
echo "train_log=${OUTPUT_ROOT}/logs/train.log"
echo "status_json=${OUTPUT_ROOT}/logs/status.json"
