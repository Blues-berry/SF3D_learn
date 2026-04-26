#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
RUN_ROOT="output/material_refine_paper/stage1_v3_round13_view_render_boundary_guard"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "${LOG_DIR}"
export PYTHONUNBUFFERED=1

echo "[round13] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[round13] python=${PYTHON_BIN}"
echo "[round13] train_config=configs/material_refine_train_stage1_v3_round13_view_render_boundary_guard.yaml"

"${PYTHON_BIN}" scripts/train_material_refiner.py \
  --config configs/material_refine_train_stage1_v3_round13_view_render_boundary_guard.yaml \
  2>&1 | tee "${LOG_DIR}/train.log"

echo "[round13] train finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"

for config in \
  configs/material_refine_eval_stage1_v3_round13_balanced_test.yaml \
  configs/material_refine_eval_stage1_v3_round13_locked346_regression.yaml \
  configs/material_refine_eval_stage1_v3_round13_ood.yaml
do
  name="$(basename "${config}" .yaml)"
  echo "[round13] eval ${name} start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${PYTHON_BIN}" scripts/eval_material_refiner.py \
    --config "${config}" \
    2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[round13] eval ${name} finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[round13] all done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
