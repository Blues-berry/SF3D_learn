#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
SESSION_PREFIX="${SESSION_PREFIX:-sf3d_longrun_material_refine}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/material_refine_longrun_$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_RECORDS="${MAX_RECORDS:-1200}"
SHARDS="${SHARDS:-2}"
VIEW_LIGHT_PROTOCOL="${VIEW_LIGHT_PROTOCOL:-stress_24}"
HDRI_BANK_JSON="${HDRI_BANK_JSON:-output/highlight_pool_a_8k/aux_sources/polyhaven_hdri_bank.json}"
MIN_HDRI_COUNT="${MIN_HDRI_COUNT:-900}"
ATLAS_RESOLUTION="${ATLAS_RESOLUTION:-768}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-256}"
CYCLES_SAMPLES="${CYCLES_SAMPLES:-8}"
MAX_HDRI_LIGHTS="${MAX_HDRI_LIGHTS:-0}"
HDRI_SELECTION_OFFSET="${HDRI_SELECTION_OFFSET:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
REFRESH_PARTIAL_EVERY="${REFRESH_PARTIAL_EVERY:-5}"
GPU_LIST="${GPU_LIST:-0,1}"
REBAKE_VERSION="${REBAKE_VERSION:-legacy}"
DISABLE_RENDER_CACHE="${DISABLE_RENDER_CACHE:-0}"
DISALLOW_PRIOR_COPY_FALLBACK="${DISALLOW_PRIOR_COPY_FALLBACK:-0}"
TARGET_VIEW_ALIGNMENT_MEAN_THRESHOLD="${TARGET_VIEW_ALIGNMENT_MEAN_THRESHOLD:-0.03}"
TARGET_VIEW_ALIGNMENT_P95_THRESHOLD="${TARGET_VIEW_ALIGNMENT_P95_THRESHOLD:-0.08}"
LONGRUN_INPUT_MANIFESTS="${LONGRUN_INPUT_MANIFESTS:-output/material_refine_pipeline_20260418T091559Z/material_refine_manifest_v1.json,output/highlight_pool_a_8k/objaverse_github_lfs_increment_manifest/material_refine_manifest_objaverse_increment.json}"
PAPER_MAIN_SOURCES="${PAPER_MAIN_SOURCES:-ABO_locked_core}"
AUXILIARY_SOURCES="${AUXILIARY_SOURCES:-}"
PRIORITY_MATERIAL_FAMILIES="${PRIORITY_MATERIAL_FAMILIES:-}"
TARGET_MATERIAL_FAMILY_RATIOS="${TARGET_MATERIAL_FAMILY_RATIOS:-}"
INTERLEAVE_SELECTION_KEYS="${INTERLEAVE_SELECTION_KEYS:-}"
PAPER_FRONTLOAD_RECORDS="${PAPER_FRONTLOAD_RECORDS:-0}"
PREFER_PAPER_MAIN_FIRST="${PREFER_PAPER_MAIN_FIRST:-true}"
START_MERGED_MONITOR="${START_MERGED_MONITOR:-true}"
MERGED_MONITOR_POLL_SECONDS="${MERGED_MONITOR_POLL_SECONDS:-120}"
MERGED_MONITOR_SESSION="${MERGED_MONITOR_SESSION:-${SESSION_PREFIX}_merged_monitor}"
MERGED_MONITOR_OUTPUT="${MERGED_MONITOR_OUTPUT:-${OUTPUT_ROOT}/canonical_manifest_monitor_merged.json}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/shards"

build_cmd=("${PYTHON_BIN}" scripts/build_material_refine_longrun_manifest.py)
IFS=',' read -r -a INPUT_MANIFEST_ARRAY <<< "${LONGRUN_INPUT_MANIFESTS}"
for input_manifest in "${INPUT_MANIFEST_ARRAY[@]}"; do
  build_cmd+=(--input-manifest "${input_manifest}")
done
build_cmd+=(
  --output-manifest "${OUTPUT_ROOT}/longrun_input_manifest.json"
  --output-shard-dir "${OUTPUT_ROOT}/shards"
  --shards "${SHARDS}"
  --max-records "${MAX_RECORDS}"
  --paper-main-sources "${PAPER_MAIN_SOURCES}"
  --auxiliary-sources "${AUXILIARY_SOURCES}"
)
if [[ "${PREFER_PAPER_MAIN_FIRST}" == "true" || "${PREFER_PAPER_MAIN_FIRST}" == "1" ]]; then
  build_cmd+=(--prefer-paper-main-first)
fi
if [[ -n "${PRIORITY_MATERIAL_FAMILIES}" ]]; then
  build_cmd+=(--priority-material-families "${PRIORITY_MATERIAL_FAMILIES}")
fi
if [[ -n "${TARGET_MATERIAL_FAMILY_RATIOS}" ]]; then
  build_cmd+=(--target-material-family-ratios "${TARGET_MATERIAL_FAMILY_RATIOS}")
fi
if [[ "${PAPER_FRONTLOAD_RECORDS}" != "0" ]]; then
  build_cmd+=(--paper-frontload-records "${PAPER_FRONTLOAD_RECORDS}")
fi
if [[ -n "${INTERLEAVE_SELECTION_KEYS}" ]]; then
  build_cmd+=(--interleave-selection-keys "${INTERLEAVE_SELECTION_KEYS}")
fi
"${build_cmd[@]}" | tee "${OUTPUT_ROOT}/logs/build_longrun_manifest.log"

prepare_extra_args=(
  --rebake-version "${REBAKE_VERSION}"
  --target-view-alignment-mean-threshold "${TARGET_VIEW_ALIGNMENT_MEAN_THRESHOLD}"
  --target-view-alignment-p95-threshold "${TARGET_VIEW_ALIGNMENT_P95_THRESHOLD}"
)
if [[ "${DISABLE_RENDER_CACHE}" == "1" || "${DISABLE_RENDER_CACHE}" == "true" ]]; then
  prepare_extra_args+=(--disable-render-cache)
fi
if [[ "${DISALLOW_PRIOR_COPY_FALLBACK}" == "1" || "${DISALLOW_PRIOR_COPY_FALLBACK}" == "true" ]]; then
  prepare_extra_args+=(--disallow-prior-copy-fallback)
fi
printf '%q ' "${prepare_extra_args[@]}" > "${OUTPUT_ROOT}/logs/prepare_extra_args.txt"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
for shard_idx in $(seq 0 $((SHARDS - 1))); do
  gpu="${GPUS[$((shard_idx % ${#GPUS[@]}))]}"
  shard_name="$(printf 'longrun_input_shard_%02d.json' "${shard_idx}")"
  session="${SESSION_PREFIX}_shard${shard_idx}_gpu${gpu}"
  shard_manifest="${OUTPUT_ROOT}/shards/${shard_name}"
  shard_output="${OUTPUT_ROOT}/prepared_shard_${shard_idx}"
  run_script="${OUTPUT_ROOT}/logs/${session}.run.sh"
  cat > "${run_script}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/prepare_material_refine_dataset.py \\
  --input-manifest "${shard_manifest}" \\
  --output-root "${shard_output}" \\
  --split full \\
  --atlas-resolution "${ATLAS_RESOLUTION}" \\
  --render-resolution "${RENDER_RESOLUTION}" \\
  --cycles-samples "${CYCLES_SAMPLES}" \\
  --view-light-protocol "${VIEW_LIGHT_PROTOCOL}" \\
  --hdri-bank-json "${HDRI_BANK_JSON}" \\
  --min-hdri-count "${MIN_HDRI_COUNT}" \\
  --max-hdri-lights "${MAX_HDRI_LIGHTS}" \\
  --hdri-selection-offset "${HDRI_SELECTION_OFFSET}" \\
  --cuda-device-index "${gpu}" \\
  --parallel-workers "${WORKERS_PER_GPU}" \\
  --refresh-partial-every "${REFRESH_PARTIAL_EVERY}" \\
  \$(cat "${OUTPUT_ROOT}/logs/prepare_extra_args.txt") \\
  > "${OUTPUT_ROOT}/logs/${session}.log" 2>&1
SCRIPT
  chmod +x "${run_script}"
  if tmux has-session -t "${session}" >/dev/null 2>&1; then
    echo "tmux session already exists: ${session}"
  else
    tmux new-session -d -s "${session}" "${run_script}"
    echo "started ${session}"
  fi
done

if [[ "${START_MERGED_MONITOR}" == "true" || "${START_MERGED_MONITOR}" == "1" ]]; then
  monitor_script="${OUTPUT_ROOT}/logs/${MERGED_MONITOR_SESSION}.run.sh"
  cat > "${monitor_script}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}" \\
PYTHON_BIN="${PYTHON_BIN}" \\
POLL_SECONDS="${MERGED_MONITOR_POLL_SECONDS}" \\
OUTPUT_MANIFEST="${MERGED_MONITOR_OUTPUT}" \\
  bash scripts/monitor_material_refine_merged_manifest.sh \\
  > "${OUTPUT_ROOT}/logs/${MERGED_MONITOR_SESSION}.log" 2>&1
SCRIPT
  chmod +x "${monitor_script}"
  if tmux has-session -t "${MERGED_MONITOR_SESSION}" >/dev/null 2>&1; then
    echo "tmux session already exists: ${MERGED_MONITOR_SESSION}"
  else
    tmux new-session -d -s "${MERGED_MONITOR_SESSION}" "${monitor_script}"
    echo "started ${MERGED_MONITOR_SESSION}"
  fi
fi

cat > "${OUTPUT_ROOT}/logs/quality_check_command.txt" <<EOF
${PYTHON_BIN} scripts/evaluate_material_refine_dataset_quality.py --manifest <prepared_manifest.json> --output-dir <quality_output_dir>
EOF

echo "output_root: ${OUTPUT_ROOT}"
echo "logs: ${OUTPUT_ROOT}/logs"
echo "merged_monitor: ${MERGED_MONITOR_OUTPUT}"
tmux ls
