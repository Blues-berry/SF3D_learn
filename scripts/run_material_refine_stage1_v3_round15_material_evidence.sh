#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
RUN_ROOT="output/material_refine_paper/stage1_v3_round15_material_evidence_calibration"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "${LOG_DIR}"
export PYTHONUNBUFFERED=1

echo "[round15] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[round15] python=${PYTHON_BIN}"
echo "[round15] base_config=configs/material_refine_train_stage1_v3_round14_backbone_topology_render.yaml"
echo "[round15] method_override=configs/material_refine_train_stage1_v3_round15_material_evidence_calibration.yaml"

"${PYTHON_BIN}" scripts/train_material_refiner.py \
  --config configs/material_refine_train_stage1_v3_round14_backbone_topology_render.yaml \
  --config configs/material_refine_train_stage1_v3_round15_material_evidence_calibration.yaml \
  2>&1 | tee "${LOG_DIR}/train.log"

echo "[round15] train finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"

BEST_CKPT="${RUN_ROOT}/best.pt"
if [[ ! -e "${BEST_CKPT}" ]]; then
  echo "[round15][error] missing best checkpoint: ${BEST_CKPT}" >&2
  exit 2
fi

declare -a EVAL_CONFIGS=(
  "configs/material_refine_eval_stage1_v3_round15_balanced_test.yaml"
  "configs/material_refine_eval_stage1_v3_round15_locked346_regression.yaml"
  "configs/material_refine_eval_stage1_v3_round15_ood.yaml"
)

for eval_config in "${EVAL_CONFIGS[@]}"; do
  eval_name="$(basename "${eval_config}" .yaml)"
  echo "[round15] eval ${eval_name} start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${PYTHON_BIN}" scripts/eval_material_refiner.py \
    --config "${eval_config}" \
    2>&1 | tee "${LOG_DIR}/${eval_name}.log"
  echo "[round15] eval ${eval_name} finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[round15] all done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
