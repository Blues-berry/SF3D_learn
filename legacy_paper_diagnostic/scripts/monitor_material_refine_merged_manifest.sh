#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
POLL_SECONDS="${POLL_SECONDS:-120}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-${OUTPUT_ROOT}/canonical_manifest_monitor_merged.json}"

cd "${REPO_ROOT}"

while true; do
  mapfile -t manifests < <(
    find "${OUTPUT_ROOT}" \
      \( -path '*/full/canonical_manifest_monitor_partial.json' -o -path '*/full/canonical_manifest_partial.json' \) \
      -type f | sort
  )
  if [[ "${#manifests[@]}" -gt 0 ]]; then
    cmd=("${PYTHON_BIN}" scripts/merge_material_refine_partial_manifests.py --output-manifest "${OUTPUT_MANIFEST}")
    for manifest in "${manifests[@]}"; do
      cmd+=(--manifest "${manifest}")
    done
    "${cmd[@]}" || true
  else
    echo "$(date -Iseconds) no partial manifests found under ${OUTPUT_ROOT}"
  fi
  sleep "${POLL_SECONDS}"
done
