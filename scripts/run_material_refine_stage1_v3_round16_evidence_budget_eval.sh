#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
RUN_ROOT="output/material_refine_paper/stage1_v3_round16_evidence_update_budget"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "${LOG_DIR}"
export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"

declare -a EVAL_CONFIGS=(
  "configs/material_refine_eval_stage1_v3_round16_evidence_budget_balanced_test.yaml"
  "configs/material_refine_eval_stage1_v3_round16_evidence_budget_locked346_regression.yaml"
  "configs/material_refine_eval_stage1_v3_round16_evidence_budget_ood.yaml"
)

echo "[round16] eval-only evidence budget start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for eval_config in "${EVAL_CONFIGS[@]}"; do
  eval_name="$(basename "${eval_config}" .yaml)"
  echo "[round16] eval ${eval_name} start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${PYTHON_BIN}" scripts/eval_material_refiner.py \
    --config "${eval_config}" \
    2>&1 | tee "${LOG_DIR}/${eval_name}.log"
  echo "[round16] eval ${eval_name} finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done
echo "[round16] eval-only evidence budget done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
