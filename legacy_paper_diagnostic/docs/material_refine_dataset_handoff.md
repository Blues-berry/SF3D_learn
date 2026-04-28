# Material Refine Dataset Handoff

## Purpose

This document separates the dataset-side work from the model-side pipeline so it can be handed to other subprocesses cleanly.

Dataset subprocess scope:

- build or refresh the `CanonicalAssetRecordV1` source manifest
- prepare canonical bundles under `prepared/full/canonical_bundle`
- render canonical view buffers on `GPU0`
- bake `uv_target_roughness`, `uv_target_metallic`, and `uv_target_confidence`
- refresh a `canonical_manifest_partial.json` snapshot while the full run is still in progress
- finish with a valid `canonical_manifest_full.json`

Out of scope for dataset subprocesses:

- `sf3d/material_refine/*`
- `scripts/train_material_refiner.py`
- `scripts/eval_material_refiner.py`
- `scripts/export_refined_material_report.py`
- `run.py`
- `gradio_app.py`
- any `GPU1` training or inference job

## Current Snapshot

Timestamp: `2026-04-18 UTC`

Project root:

- `/home/ubuntu/ssd_work/projects/stable-fast-3d`

Current pipeline root:

- `/home/ubuntu/ssd_work/projects/stable-fast-3d/output/material_refine_pipeline_20260418T091559Z`

Current source manifest status:

- `material_refine_manifest_v1.json` exists
- total source records: `560`
- locally available source assets: `530`
- unavailable source assets: `30`
- source mix: `500 ABO`, `30 3D-FUTURE`, `30 Objaverse-XL candidate`
- prior mix: `500 with prior`, `60 without prior`

Current prepared-data status:

- bundle directories currently present: `248`
- source mix among bundle directories: `218 ABO`, `30 3D-FUTURE`
- validated partial manifest currently present: `245` prepared records
- full manifest is not finished yet: `prepared/full/canonical_manifest_full.json` does not exist

Current model-side consumer state:

- partial training already completed through epoch `10`
- checkpoint: `/home/ubuntu/ssd_work/projects/stable-fast-3d/output/material_refine_pipeline_20260418T091559Z/train_full/latest.pt`
- refreshed partial manifest:
  `/home/ubuntu/ssd_work/projects/stable-fast-3d/output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_partial.json`
- the model side should keep consuming `canonical_manifest_partial.json` until `canonical_manifest_full.json` is ready

Current runtime state:

- no active dataset prepare process
- no active train or eval process
- `GPU0` and `GPU1` are idle

## Inputs

Primary source manifest input:

- [material_refine_manifest_v1.json](/home/ubuntu/ssd_work/projects/stable-fast-3d/output/material_refine_pipeline_20260418T091559Z/material_refine_manifest_v1.json)

Optional rebuild inputs if source registry changed:

- [mini_v1_manifest.json](/home/ubuntu/ssd_work/projects/stable-fast-3d/docs/neural_gaffer_dataset_audit/mini_v1_manifest.json)
- [pool_A_pilot_3dfuture_30.json](/home/ubuntu/ssd_work/projects/stable-fast-3d/docs/neural_gaffer_dataset_audit/pool_A_pilot_3dfuture_30.json)
- [pool_A_pilot_objaverse_30.csv](/home/ubuntu/ssd_work/projects/stable-fast-3d/docs/neural_gaffer_dataset_audit/pool_A_pilot_objaverse_30.csv)

Reference audit docs:

- [source_decision_v2.md](/home/ubuntu/ssd_work/projects/stable-fast-3d/docs/neural_gaffer_dataset_audit/source_decision_v2.md)
- [source_registry_v2.csv](/home/ubuntu/ssd_work/projects/stable-fast-3d/docs/neural_gaffer_dataset_audit/source_registry_v2.csv)
- [material_refine_manifest_v1_summary.md](/home/ubuntu/ssd_work/projects/stable-fast-3d/output/material_refine_pipeline_20260418T091559Z/material_refine_manifest_v1_summary.md)

Smoke reference:

- [canonical_manifest_smoke.json](/home/ubuntu/ssd_work/projects/stable-fast-3d/output/material_refine_pipeline_20260418T091559Z/prepared/smoke/canonical_manifest_smoke.json)

## Required Deliverables

The dataset subprocesses must hand back these artifacts:

1. Source manifest, if rebuilt:
   - `output/material_refine_pipeline_20260418T091559Z/material_refine_manifest_v1.json`
   - matching `.csv`
   - matching `_summary.md`
2. Full canonical bundle tree:
   - `output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_bundle/<object_id>/...`
3. Final full manifest:
   - `output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_full.json`
4. Rolling partial manifest snapshots for model-side warm starts:
   - `output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_partial.json`
5. A short run summary with:
   - prepared count
   - skipped count
   - skipped reasons
   - source mix
   - with-prior and without-prior counts

## Canonical Record Contract

Each prepared record that is handed to `GPU1` must contain usable values for these fields:

- `object_id`
- `generator_id`
- `license_bucket`
- `has_material_prior`
- `prior_mode`
- `canonical_mesh_path`
- `uv_albedo_path`
- `uv_normal_path`
- `uv_prior_roughness_path`
- `uv_prior_metallic_path`
- `uv_target_roughness_path`
- `uv_target_metallic_path`
- `uv_target_confidence_path`
- `canonical_views_json`
- `canonical_buffer_root`
- `view_rgba_paths`
- `supervision_tier`
- `default_split`

Notes:

- `canonical_glb_path` may be empty in the current implementation.
- `uv_prior_*` paths must still exist for `has_material_prior=false`; they are placeholder priors for mode B.
- `default_split` must be preserved so `train/val/test` filtering on `GPU1` keeps working.

## Execution Rules

Resource rules:

- dataset-side rendering and baking uses `GPU0`
- model-side training, eval, inference, and demo use `GPU1`
- do not hardcode `CUDA_VISIBLE_DEVICES=1`
- prefer the explicit `--cuda-device-index` flag

Parallelism rules:

- do not launch multiple top-level `prepare_material_refine_dataset.py` jobs that write to the same `output-root`
- use a single top-level prepare process and increase `--parallel-workers` inside that process
- start with `--parallel-workers 6`
- if `GPU0` memory stays below roughly `24 GiB` and Blender is stable, you can try `7` or `8`
- if Blender crashes, output files become inconsistent, or GPU memory spikes too high, drop back to `4`

Important:

- the previous attempt launched two top-level prepare jobs against the same output root and caused duplicate work
- if you truly want multiple subprocesses, give each subprocess a disjoint manifest shard and a disjoint output root, then merge later
- the repo does not yet ship a dedicated merge script, so the preferred path is still one prepare process with internal workers

Reuse rules:

- keep reusing the existing `canonical_bundle` tree
- keep reusing the ABO render cache:
  `/home/ubuntu/ssd_work/projects/stable-fast-3d/output/abo_rm_mini/renders`
- do not delete `canonical_manifest_partial.json`; refresh it in place

## Commands

Run everything from:

```bash
cd /home/ubuntu/ssd_work/projects/stable-fast-3d
```

Use this interpreter:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python
```

### 1. Rebuild Source Manifest Only If Needed

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/build_material_refine_manifest.py \
  --mini-manifest-json docs/neural_gaffer_dataset_audit/mini_v1_manifest.json \
  --three-future-json docs/neural_gaffer_dataset_audit/pool_A_pilot_3dfuture_30.json \
  --objaverse-csv docs/neural_gaffer_dataset_audit/pool_A_pilot_objaverse_30.csv \
  --output-json output/material_refine_pipeline_20260418T091559Z/material_refine_manifest_v1.json \
  --output-csv output/material_refine_pipeline_20260418T091559Z/material_refine_manifest_v1.csv \
  --output-md output/material_refine_pipeline_20260418T091559Z/material_refine_manifest_v1_summary.md
```

### 2. Prepare Full Dataset on GPU0

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/prepare_material_refine_dataset.py \
  --input-manifest output/material_refine_pipeline_20260418T091559Z/material_refine_manifest_v1.json \
  --output-root output/material_refine_pipeline_20260418T091559Z/prepared \
  --output-manifest output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_full.json \
  --split full \
  --cuda-device-index 0 \
  --render-resolution 256 \
  --cycles-samples 8 \
  --parallel-workers 6
```

### 3. Refresh Partial Manifest While Full Run Is Still Running

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/refresh_material_refine_partial_manifest.py \
  --input-manifest output/material_refine_pipeline_20260418T091559Z/material_refine_manifest_v1.json \
  --canonical-bundle-root output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_bundle \
  --output-manifest output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_partial.json \
  --split full
```

This command is safe to re-run. It only collects records whose bundles are already complete enough for the model-side loader.

## Monitoring

Rough progress by bundle directory count:

```bash
find output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_bundle \
  -mindepth 1 -maxdepth 1 -type d | wc -l
```

Rough source mix:

```bash
find output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_bundle \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | cut -d_ -f1 | sort | uniq -c
```

Refresh the validated partial snapshot:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/refresh_material_refine_partial_manifest.py \
  --input-manifest output/material_refine_pipeline_20260418T091559Z/material_refine_manifest_v1.json \
  --canonical-bundle-root output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_bundle \
  --output-manifest output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_partial.json \
  --split full
```

Check live processes:

```bash
ps -ef | rg 'prepare_material_refine_dataset.py|abo_material_passes_blender.py|refresh_material_refine_partial_manifest.py'
```

Check GPU usage:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

## Acceptance Criteria

The dataset handoff is complete when all of these are true:

1. `prepared/full/canonical_manifest_full.json` exists and loads.
2. Full manifest records are ready for `GPU1` without further editing.
3. Every full-manifest record has existing paths for:
   `uv_albedo_path`, `uv_normal_path`, `uv_prior_roughness_path`, `uv_prior_metallic_path`,
   `uv_target_roughness_path`, `uv_target_metallic_path`, `uv_target_confidence_path`,
   `canonical_views_json`, and `canonical_buffer_root`.
4. The expected local ceiling is currently `530` prepared records.
   `500 ABO + 30 3D-FUTURE` are locally available today.
   The `30 Objaverse-XL` candidate records are currently unavailable and should not block completion unless new assets are staged locally.
5. `canonical_manifest_partial.json` can be refreshed at any time and used by `GPU1` for continued training.

## Recommended Handoff Message

If you are assigning this to another subprocess, this is the shortest safe brief:

> Own only the dataset side under `scripts/build_material_refine_manifest.py`,
> `scripts/prepare_material_refine_dataset.py`, and
> `scripts/refresh_material_refine_partial_manifest.py`.
> Use `GPU0` only.
> Reuse the existing pipeline root at
> `output/material_refine_pipeline_20260418T091559Z`.
> Do not start multiple top-level prepare jobs against the same output root.
> Keep refreshing `prepared/full/canonical_manifest_partial.json` while the full run is in progress.
> Finish when `prepared/full/canonical_manifest_full.json` is valid and the locally available prepared count reaches the current ceiling.
