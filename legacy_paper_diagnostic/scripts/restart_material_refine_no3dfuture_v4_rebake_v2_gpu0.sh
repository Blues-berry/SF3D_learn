#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
CONFIG="${CONFIG:-configs/material_refine_dataset_factory_gpu0_rebake_v2.json}"
SESSION="${SESSION:-sf3d_material_refine_rebake_v2_gpu0_factory}"
LOG_ROOT="${LOG_ROOT:-output/material_refine_rebake_v2/no3dfuture_longrun_latest/logs}"
STATE_JSON="${STATE_JSON:-output/material_refine_rebake_v2/no3dfuture_longrun_latest/factory_state_rebake_v2.json}"
HDRI_BANK="${HDRI_BANK:-output/highlight_pool_a_8k/aux_sources/polyhaven_hdri_bank.json}"
MIN_4T_FREE_GB="${MIN_4T_FREE_GB:-200}"

cd "${REPO_ROOT}"
mkdir -p "${LOG_ROOT}" "$(dirname "${STATE_JSON}")"

echo "==== stop legacy data sessions $(date -Iseconds) ===="
if tmux ls >/dev/null 2>&1; then
  while IFS= read -r session; do
    [ -n "${session}" ] || continue
    case "${session}" in
      sf3d_longrun_material_refine_*|sf3d_material_refine_dataset_*|sf3d_factory_*|sf3d_rebake_v2_no3dfuture_*|sf3d_material_refine_rebake_v2_gpu0_factory)
        tmux kill-session -t "${session}" || true
        echo "killed tmux ${session}"
        ;;
    esac
  done < <(tmux ls 2>/dev/null | awk -F: '{print $1}')
fi

echo "==== validate rebake_v2 config $(date -Iseconds) ===="
"${PYTHON_BIN}" -m json.tool "${CONFIG}" >/dev/null
"${PYTHON_BIN}" -m py_compile \
  scripts/run_material_refine_dataset_factory.py \
  scripts/audit_material_refine_rebake_v2_contract.py \
  scripts/build_material_refine_rebake_v2_pilot.py \
  scripts/prepare_material_refine_dataset.py

echo "==== validate HDRI bank $(date -Iseconds) ===="
test -f "${HDRI_BANK}"
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
p=Path("${HDRI_BANK}")
d=json.loads(p.read_text())
records=d.get("records") or d.get("hdri_assets") or d.get("assets") or []
if len(records) < 900:
    raise SystemExit(f"hdri_bank_below_900:{len(records)}")
print(f"hdri_bank_records={len(records)}")
PY

echo "==== validate /4T free space $(date -Iseconds) ===="
free_gb=$(df -BG /4T | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if [ "${free_gb}" -lt "${MIN_4T_FREE_GB}" ]; then
  echo "not enough /4T free space: ${free_gb}GB < ${MIN_4T_FREE_GB}GB" >&2
  exit 2
fi
echo "free_4T_gb=${free_gb}"

echo "==== run/reuse pilot_64 gate $(date -Iseconds) ===="
"${PYTHON_BIN}" scripts/run_material_refine_dataset_factory.py \
  --config "${CONFIG}" \
  --state-json "${STATE_JSON%.json}.preflight.json" \
  --once \
  --start-downloads \
  --audit

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
p=Path("output/material_refine_rebake_v2/pilot_64_no3dfuture/pilot_64_decision.json")
if not p.exists():
    raise SystemExit("pilot_64_decision_missing")
d=json.loads(p.read_text())
if not d.get("pilot_64_rebake_v2_pass"):
    print("pilot_64_rebake_v2_pass=false; not launching 7-day rebake_v2 longrun")
    raise SystemExit(10)
print("pilot_64_rebake_v2_pass=true")
PY

echo "==== launch rebake_v2 GPU0 factory $(date -Iseconds) ===="
run_script="${LOG_ROOT}/${SESSION}.run.sh"
log_path="${LOG_ROOT}/${SESSION}.log"
cat > "${run_script}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
"${PYTHON_BIN}" scripts/run_material_refine_dataset_factory.py \\
  --config "${CONFIG}" \\
  --state-json "${STATE_JSON}" \\
  --loop \\
  --start-downloads \\
  --start-render \\
  --audit \\
  > "${log_path}" 2>&1
SCRIPT
chmod +x "${run_script}"
tmux new-session -d -s "${SESSION}" "${run_script}"
echo "started ${SESSION}"
echo "log: ${log_path}"
echo "state: ${STATE_JSON}"
