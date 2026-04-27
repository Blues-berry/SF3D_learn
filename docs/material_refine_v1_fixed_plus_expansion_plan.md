# Material Refine v1_fixed + Expansion Plan

## Direction

This is a two-track data-only plan.

- Track A has priority: rebuild derived view/target supervision for an existing v1/stage1/locked manifest and release `stage1_v1_fixed_trainable`.
- Track B continues download/source staging and writes expansion candidates only. It must not block Track A and must not feed the v1_fixed trainable release directly.

No R training, Phase4, or paper-stage rehearsal is launched by these scripts.

## Track A

Base selection priority:

- `output/material_refine_paper/stage1_locked346/stage1_locked346_manifest.json`
- `output/material_refine_paper/latest_dataset_check_20260421/stage1_subset_merged490/paper_stage1_subset_manifest.json`
- `output/material_refine_r_v2_dayrun/paper_stage_rehearsal_210/eval_all/manifest_snapshot.json`
- Stage1-v3/v4 manifests only if the above are missing

The selected base is written to:

- `output/material_refine_v1_fixed/base_manifest_v1.json`
- `output/material_refine_v1_fixed/base_selection.md`

Legacy view/target fields are invalidated and preserved only as `legacy_*` fields. Fresh rebake writes:

- `rebake_version = "v1_fixed_rebake"`
- `target_view_contract_version = "v1_fixed"`
- fresh view `roughness/metallic/uv/visibility/mask`
- fresh `uv_target_roughness`, `uv_target_metallic`, `uv_target_confidence`
- `prior_as_pred_pass`, `target_as_pred_pass`, `target_view_alignment_mean`, `target_view_alignment_p95`

Trainable release is engineering/ablation data, not a paper claim.

The active rebake launcher is GPU0-only (`CUDA_VISIBLE_DEVICES=0`) and
defaults to 6 Blender workers. It writes an explicit progress manifest at:

- `output/material_refine_v1_fixed/releases/stage1_v1_fixed_rebaked_partial_manifest.json`

When the full rebake completes, the launcher automatically runs the release
builder and writes `stage1_v1_fixed_trainable_manifest.json`,
`stage1_v1_fixed_paper_candidate_manifest.json`, diagnostic/reject manifests,
sampler config, gate audit, and `stage1_v1_fixed_decision.md`.

## Track B

Expansion sources continue as candidate-only:

- PolyHaven HDRI bank: lighting only
- PolyHaven material bank: material auxiliary only
- Objaverse no-3D-FUTURE scarce/cached sources
- Smithsonian, Thingiverse, Sketchfab scarce
- local object sources
- legacy no-3D-FUTURE v4 candidate-only pool

All expansion records are marked:

- `dataset_role = "expansion_candidate"`
- `target_view_contract_version = "not_rebaked_yet"`
- `stored_view_target_valid_for_paper = false`
- `paper_stage_eligible = false`
- `candidate_pool_only = true`

They are merged at:

- `output/material_refine_expansion_candidates/merged_expansion_candidate_manifest.json`
- `output/material_refine_expansion_candidates/expansion_status.md`

## Entry Point

Use:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/run_material_refine_v1_fixed_and_expansion_manager.py \
  --select-base \
  --start-expansion-downloads \
  --build-expansion-candidates \
  --start-a-rebake \
  --a-parallel-workers 6
```

This starts only data processing and download/source staging. It does not start training.
