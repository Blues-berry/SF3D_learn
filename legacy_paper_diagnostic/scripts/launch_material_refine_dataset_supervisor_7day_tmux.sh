#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
CONFIG="${CONFIG:-configs/material_refine_dataset_supervisor_7day_gpu0.json}"
SESSION="${SESSION:-sf3d_material_refine_dataset_7day_supervisor}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/material_refine_dataset_factory/supervisor_7day}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

run_script="${OUTPUT_ROOT}/${SESSION}.run.sh"
log_path="${OUTPUT_ROOT}/${SESSION}.log"
cat > "${run_script}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
"${PYTHON_BIN}" scripts/run_material_refine_dataset_supervisor_7day.py \\
  --config "${CONFIG}" \\
  > "${log_path}" 2>&1
SCRIPT
chmod +x "${run_script}"

if tmux has-session -t "${SESSION}" >/dev/null 2>&1; then
  echo "tmux session already exists: ${SESSION}"
else
  tmux new-session -d -s "${SESSION}" "${run_script}"
  echo "started ${SESSION}"
fi

echo "log: ${log_path}"
echo "status: ${OUTPUT_ROOT}/status.json"
echo "attach: tmux attach -t ${SESSION}"
