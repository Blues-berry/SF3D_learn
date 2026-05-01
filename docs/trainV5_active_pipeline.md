# TrainV5 Active Pipeline

TrainV5 is the only active engineering mainline for material refinement data.
Everything under `legacy_paper_diagnostic/` is retained for paper diagnostics,
historical explanation, or audit traceability, but active scripts must not
import or call it.

The only recommended operator entrypoint for dataset processing is:

```bash
python datasetscrip/trainv5_dataset.py <status|ingest|queue|launch|finalize|supervisor>
```

For continuous automation, use:

```bash
bash datasetscrip/trainv5_supervisor.sh
```

Older `scripts/` files remain as internal compatibility helpers unless this
document explicitly names them as active operator commands.

## Active Flow

1. Stage or select object candidates and source metadata.
2. Run CPU-only screening/repair to produce a screened object queue.
3. Rebuild target/view supervision with the active data preparation path.
4. Apply `trainv5_target_truth_gate` to verify target truth, view alignment,
   view supervision readiness, and required path availability.
5. Build target bundles, prior variants, training pairs, sampler config, and
   readiness reports under `train/trainV5_*`.
6. Train or evaluate only from TrainV5 training-pair manifests after manual
   confirmation.

The current `1155` B-track queue is a screened object queue, not a prior queue.
Fields such as `has_material_prior`, `prior_mode`, and
`expected_prior_variant_types` are only planning hints at screening time.
Actual prior variants are configured later in
`scripts/finalize_material_refine_trainV5_b_track.py` after target bundles pass
the active truth gate.

## Active Gate

The only engineering gate is `sf3d/material_refine/trainv5_target_gate.py`.
It may block records for missing target paths, failed target/view alignment,
missing view supervision, or unresolved assets.

Target admission and prior construction are separate:

- screening/repair decides whether an object is worth rebake
- target gate decides whether a rebaked supervision bundle is usable
- finalize decides which prior variants to attach for training pairs

The following fields are diagnostic metadata only and must not block TrainV5
engineering readiness:

- `paper_split`
- `paper_stage_eligible`
- `paper_pseudo`
- `paper_strong`
- source or material quota
- `target_prior_identity`
- `target_is_prior_copy`
- paper-only license buckets

## Active Files

The operator-facing TrainV5 dataset entrypoint is in `datasetscrip/`. Core
TrainV5 implementation helpers remain in `scripts/`, including dataset
preparation, TrainV5 initial/A/B builders, B-track finalize/monitor helpers,
and TrainV5 pair auditing. Runtime training and evaluation entrypoints remain
in place.

Core runtime modules remain in `sf3d/material_refine/`, including dataset,
I/O, experiment utilities, model code, manifest quality helpers required by
legacy-compatible readers, and the TrainV5 target gate.

Active TrainV5 outputs are kept in:

- `train/trainV5_initial`
- `train/trainV5_plus_a_track`
- `train/trainV5_plus_full`
- `train/trainV5_merged_ab`
- `output/material_refine_trainV5_abc`

## Deprecated Direct Script Calls

Use `datasetscrip/trainv5_dataset.py` instead of calling these flows directly:

- source staging/download: `python datasetscrip/trainv5_dataset.py ingest`
- second-pass/repair queue generation: `python datasetscrip/trainv5_dataset.py queue`
- next batch preflight/launch: `python datasetscrip/trainv5_dataset.py launch`
- completed batch cumulative update: `python datasetscrip/trainv5_dataset.py finalize --batch <batch_name>`

Do not add new `build_*_iterN.py`, `cutover_*`, or one-batch launcher scripts.
Add new behavior through `datasetscrip/trainv5_dataset_config.json` or CLI
parameters.

## Legacy Boundary

Paper-stage promotion, old stage1 subset builders, v1-fixed release scripts,
round logs, legacy ablations, old no-3D-FUTURE supervisors, paper configs, and
Neural Gaffer audit records were moved to `legacy_paper_diagnostic/`.

The legacy tree is read-only by convention for active engineering. If a useful
idea from legacy is needed again, copy the behavior into a TrainV5 script or
document it explicitly rather than importing the legacy path.

## Cleanup Reports

The cleanup dry run and execution reports are stored under:

- `output/material_refine_trainV5_cleanup/dry_run_report.md`
- `output/material_refine_trainV5_cleanup/cleanup_final_report.md`

The old legacy migration manifest is no longer an active TrainV5 dependency.
Use the cleanup reports above for current provenance checks.

No active B-track data process was stopped during cleanup, and no TrainV5
manifest-pointed assets were deleted.
