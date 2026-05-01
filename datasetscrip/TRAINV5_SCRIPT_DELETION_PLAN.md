# TrainV5 Script Deletion Plan

## Purpose

`datasetscrip/trainv5_dataset.py` is the operator-facing TrainV5 dataset entrypoint.
The old `scripts/` files are now split into four groups so deletion can proceed
without interrupting running rebake or finalize sessions.

## Current Migration State

- First one-off script deletion batch is complete.
- `datasetscrip/internal/` now owns the TrainV5 dataset command orchestration
  layer used by `datasetscrip/trainv5_dataset.py`.
- `datasetscrip/internal/` has been split by responsibility:
  `common.py`, `ingest.py`, `queue.py`, `status.py`, `launch.py`, and
  `finalize.py`.
- The remaining `scripts/` files in `keep_active` are internal implementation
  helpers or live rebake/finalize dependencies. Delete them only after their
  behavior has moved into `datasetscrip/internal/` and one rolling batch has
  completed through the new path.

## keep_active

These files are still used by `datasetscrip`, the running B-track rebake, monitor,
or finalize watcher. Do not delete them until their behavior has been migrated
and the current running batch has completed.

| file | reason |
|---|---|
| `scripts/prepare_material_refine_dataset.py` | Current rebake command calls it. |
| `scripts/build_material_refine_trainV5_abc.py` | `datasetscrip launch` uses it for B-track preflight. |
| `scripts/finalize_material_refine_trainV5_b_track.py` | Current finalize watcher and `datasetscrip finalize` call it. |
| `scripts/monitor_material_refine_trainV5_b_rebake.py` | Current truth monitor uses it. |
| `scripts/build_material_refine_trainV5_auto_status.py` | `datasetscrip status` refreshes `pipeline_brief`. |
| `scripts/stage_material_refine_material_priority_sources.py` | `datasetscrip ingest` internal helper. |
| `scripts/build_material_refine_trainV5_expansion_second_pass.py` | `datasetscrip queue` internal helper. |
| `scripts/build_material_refine_trainV5_repair_and_expansion_plan.py` | `datasetscrip queue` internal helper. |
| `scripts/resolve_next_trainv5_b_batch.py` | `datasetscrip launch` internal helper. |
| `scripts/maybe_launch_next_trainv5_b_batch.py` | `datasetscrip launch` internal helper. |
| `scripts/launch_trainv5_b_rebake_batch.py` | Rolling launcher used by `maybe_launch_next_trainv5_b_batch.py`. |

## migrate_first

These files should be generalized or moved behind `datasetscrip` before deleting
their original `scripts/` versions.

| file | migration target |
|---|---|
| `scripts/build_trainv5_objaverse_1200_serial_artifacts.py` | Generic rolling batch artifact helper. |
| `scripts/stage_objaverse_cached_increment.py` | Internal source staging function/module. |
| `scripts/build_objaverse_increment_manifest.py` | Internal source manifest conversion helper. |
| `scripts/merge_material_refine_expansion_candidates.py` | Internal ingest helper. |

## next_delete_candidates_after_migration

These should become wrappers first, then be deleted after dry-run and one
supervisor cycle pass.

| file | prerequisite |
|---|---|
| `scripts/stage_material_refine_material_priority_sources.py` | Move source staging implementation into `datasetscrip/internal/ingest.py`. |
| `scripts/build_material_refine_trainV5_expansion_second_pass.py` | Move second-pass implementation into `datasetscrip/internal/queue.py`. |
| `scripts/build_material_refine_trainV5_repair_and_expansion_plan.py` | Move repair/ready-deferred implementation into `datasetscrip/internal/queue.py`. |
| `scripts/resolve_next_trainv5_b_batch.py` | Move next-batch decision into `datasetscrip/internal/launch.py`. |

## deleted_first_batch

These one-off scripts have been removed from active code. Their former behavior
is now covered by `datasetscrip/trainv5_dataset.py`, the parameterized staging
helper, or the rolling launch path.

| removed file | replacement |
|---|---|
| `scripts/manage_trainv5_objaverse_1200_serial_plan.py` | `python datasetscrip/trainv5_dataset.py ingest` |
| `scripts/maybe_launch_objaverse_1200_serial_after_b1.py` | `python datasetscrip/trainv5_dataset.py launch` |
| `scripts/cutover_trainv5_b1_rebake.py` | `python datasetscrip/trainv5_dataset.py launch` |
| `scripts/build_material_refine_trainV5_plus_a_withprior_iter1.py` | `python datasetscrip/trainv5_dataset.py finalize --batch <batch>` and TrainV5 cumulative outputs |

## legacy_only

These categories are not active TrainV5 dataset entrypoints. They may be moved
to a legacy diagnostic area or deleted in a later cleanup after separate review.

- Paper-stage or old round-specific builders.
- Old audit reports and one-off command drafts.
- Historical stage1/stage3/stage4 helpers that are no longer imported by
  `datasetscrip`.
- Training scripts are out of scope for this deletion pass.

## Deletion Rule

Before deleting any additional `scripts/` file, run:

```bash
rg "<script_name>" datasetscrip scripts docs output/material_refine_trainV5_auto
```

The file can only be deleted if references are limited to this deletion plan,
pipeline documentation, or historical output snapshots.
