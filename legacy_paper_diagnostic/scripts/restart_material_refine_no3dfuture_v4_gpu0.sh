#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
FACTORY_CONFIG="${FACTORY_CONFIG:-configs/material_refine_dataset_factory_gpu0.json}"
SUPERVISOR_CONFIG="${SUPERVISOR_CONFIG:-configs/material_refine_dataset_supervisor_7day_gpu0.json}"
SUPERVISOR_SESSION="${SUPERVISOR_SESSION:-sf3d_material_refine_dataset_7day_supervisor}"
PRELAUNCH_ONCE="${PRELAUNCH_ONCE:-0}"

cd "${REPO_ROOT}"

echo "==== stop old data sessions $(date -Iseconds) ===="
if tmux ls >/dev/null 2>&1; then
  while IFS= read -r session; do
    [ -n "${session}" ] || continue
    case "${session}" in
      sf3d_factory_*|sf3d_longrun_material_refine_*|sf3d_material_refine_dataset_*)
        tmux kill-session -t "${session}" || true
        echo "killed tmux ${session}"
        ;;
    esac
  done < <(tmux ls 2>/dev/null | awk -F: '{print $1}')
fi

echo "==== stop orphan data processes $(date -Iseconds) ===="
patterns=(
  "scripts/stage_objaverse_cached_increment.py"
  "scripts/stage_polyhaven_material_bank.py"
  "scripts/stage_highlight_aux_sources.py"
  "scripts/run_material_refine_dataset_factory.py"
  "scripts/run_material_refine_dataset_supervisor_7day.py"
  "scripts/build_material_refine_longrun_manifest.py"
  "scripts/prepare_material_refine_dataset.py"
  "scripts/monitor_material_refine_merged_manifest.sh"
)
for pattern in "${patterns[@]}"; do
  while IFS= read -r pid; do
    [ -n "${pid}" ] || continue
    [ "${pid}" != "$$" ] || continue
    kill "${pid}" 2>/dev/null || true
    echo "sent TERM to ${pid} (${pattern})"
  done < <(ps -eo pid=,args= | awk -v pat="${pattern}" '$0 ~ pat && $0 !~ /awk -v pat/ {print $1}')
done
sleep 3

for pattern in "${patterns[@]}"; do
  while IFS= read -r pid; do
    [ -n "${pid}" ] || continue
    [ "${pid}" != "$$" ] || continue
    kill -9 "${pid}" 2>/dev/null || true
    echo "sent KILL to ${pid} (${pattern})"
  done < <(ps -eo pid=,args= | awk -v pat="${pattern}" '$0 ~ pat && $0 !~ /awk -v pat/ {print $1}')
done

echo "==== validate configs/scripts $(date -Iseconds) ===="
"${PYTHON_BIN}" -m json.tool "${FACTORY_CONFIG}" >/dev/null
"${PYTHON_BIN}" -m json.tool "${SUPERVISOR_CONFIG}" >/dev/null
"${PYTHON_BIN}" -m py_compile \
  scripts/run_material_refine_dataset_factory.py \
  scripts/build_material_refine_stage1_v3_subsets.py \
  scripts/promote_material_refine_targets.py \
  scripts/quarantine_material_refine_rejected_assets.py

if [[ "${PRELAUNCH_ONCE}" == "1" || "${PRELAUNCH_ONCE}" == "true" ]]; then
  echo "==== optional one-shot promotion/audit pass before longrun $(date -Iseconds) ===="
  "${PYTHON_BIN}" scripts/run_material_refine_dataset_factory.py \
    --config "${FACTORY_CONFIG}" \
    --state-json output/material_refine_dataset_factory/factory_state_prelaunch_v4_no3dfuture.json \
    --once \
    --start-downloads \
    --audit || true
else
  echo "==== skip prelaunch once pass; supervisor owns promotion/audit loop $(date -Iseconds) ===="
fi

echo "==== launch 7-day GPU0 data supervisor $(date -Iseconds) ===="
CONFIG="${SUPERVISOR_CONFIG}" \
SESSION="${SUPERVISOR_SESSION}" \
PYTHON_BIN="${PYTHON_BIN}" \
  bash scripts/launch_material_refine_dataset_supervisor_7day_tmux.sh

echo "==== status $(date -Iseconds) ===="
tmux ls || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
echo "factory state: output/material_refine_dataset_factory/factory_state.json"
echo "supervisor status: output/material_refine_dataset_factory/supervisor_7day/status.json"
