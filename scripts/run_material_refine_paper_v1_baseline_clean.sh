#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/material_refine_train_paper_v1_baseline_clean.yaml}"
SESSION_NAME="${SESSION_NAME:-sf3d_paper_v1_baseline_clean_v5_gpu1}"
OUTPUT_DIR="${OUTPUT_DIR:-output/material_refine_paper/paper_v1_baseline_clean_v5_20260426}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

RUN_SCRIPT="${LOG_DIR}/${SESSION_NAME}.run.sh"
cat > "${RUN_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT_DIR}"
export WANDB_PROJECT=stable-fast-3d-material-refine-paper-v1-clean
export WANDB_DIR="${ROOT_DIR}/wandb"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="\${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="\${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="\${NUMEXPR_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="\${TORCH_NUM_THREADS:-4}"
export TORCH_NUM_INTEROP_THREADS="\${TORCH_NUM_INTEROP_THREADS:-2}"
echo "[bootstrap] $(date -u +%Y-%m-%dT%H:%M:%SZ) config=${CONFIG_PATH} output=${OUTPUT_DIR} python=${PYTHON_BIN}" | tee "${LOG_DIR}/train.log"
"${PYTHON_BIN}" -u scripts/train_material_refiner.py \\
  --config "${CONFIG_PATH}" \\
  2>&1 | tee -a "${LOG_DIR}/train.log"
EOF
chmod +x "${RUN_SCRIPT}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux kill-session -t "${SESSION_NAME}"
fi
tmux new-session -d -s "${SESSION_NAME}" "${RUN_SCRIPT}"

echo "[launch] session=${SESSION_NAME}"
echo "[launch] log=${LOG_DIR}/train.log"
echo "[launch] attach=tmux attach -t ${SESSION_NAME}"
