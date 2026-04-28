# TrainV5 Active Pipeline

TrainV5 is the only active engineering mainline for material refinement data.
Everything under `legacy_paper_diagnostic/` is retained for paper diagnostics,
historical explanation, or audit traceability, but active scripts must not
import or call it.

## Active Flow

1. Stage or select object candidates and source metadata.
2. Rebuild target/view supervision with the active data preparation path.
3. Apply `trainv5_target_truth_gate` to verify target truth, view alignment,
   view supervision readiness, and required path availability.
4. Build target bundles, prior variants, training pairs, sampler config, and
   readiness reports under `train/trainV5_*`.
5. Train or evaluate only from TrainV5 training-pair manifests after manual
   confirmation.

## Active Gate

The only engineering gate is `sf3d/material_refine/trainv5_target_gate.py`.
It may block records for missing target paths, failed target/view alignment,
missing view supervision, or unresolved assets.

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

Core TrainV5 scripts remain in `scripts/`, including dataset preparation,
TrainV5 initial/A/B builders, B-track finalize/monitor helpers, and TrainV5
pair auditing. Runtime training and evaluation entrypoints remain in place.

Core runtime modules remain in `sf3d/material_refine/`, including dataset,
I/O, experiment utilities, model code, manifest quality helpers required by
legacy-compatible readers, and the TrainV5 target gate.

Active TrainV5 outputs are kept in:

- `train/trainV5_initial`
- `train/trainV5_plus_a_track`
- `train/trainV5_plus_full`
- `train/trainV5_merged_ab`
- `output/material_refine_trainV5_abc`

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
- `legacy_paper_diagnostic/MIGRATION_MANIFEST.json`

No active B-track data process was stopped during cleanup, and no TrainV5
manifest-pointed assets were deleted.
