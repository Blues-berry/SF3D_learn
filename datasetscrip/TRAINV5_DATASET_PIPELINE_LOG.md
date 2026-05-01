# TrainV5 Dataset Pipeline Log

## Purpose

`datasetscrip/` is the operator-facing entry layer for TrainV5 dataset processing.
The older files under `scripts/` remain as internal compatibility helpers, but
new data-source, batch, queue, and launch behavior should be expressed through
`datasetscrip/trainv5_dataset.py` parameters or `trainv5_dataset_config.json`.

## Current Active Chain

The active TrainV5 data chain is:

`ingest -> queue -> launch -> prepare/rebake -> target truth gate -> finalize -> cumulative TrainV5`

The finalize step keeps the existing order:

`target truth gate -> target bundles -> 5 prior variants -> training pairs`

The cumulative outputs are:

- `train/trainV5_plus_full`
- `train/trainV5_merged_ab`

Batch-local outputs remain under each B-track batch directory in
`output/material_refine_trainV5_abc/B_track/`.

## Recommended Commands

Show the short live status:

```bash
python datasetscrip/trainv5_dataset.py status
```

Refresh CPU-side queues and rolling next-batch decision:

```bash
python datasetscrip/trainv5_dataset.py queue
```

Resolve/preflight/launch the next rolling B batch:

```bash
python datasetscrip/trainv5_dataset.py launch
```

Run a dry-run launch check while GPU0 is busy:

```bash
python datasetscrip/trainv5_dataset.py launch --dry-run
```

Finalize a completed B batch:

```bash
python datasetscrip/trainv5_dataset.py finalize --batch <batch_name>
```

Dry-run finalize on a running batch may use the partial manifest for estimation.
Those partial manifests can contain unfinished records as skipped placeholders;
only the completed final manifest should be used for real cumulative updates.

Run one supervisor cycle without changing the long-running tmux setup:

```bash
bash datasetscrip/trainv5_supervisor.sh --once
```

Start the lightweight supervisor wrapper:

```bash
bash datasetscrip/trainv5_supervisor.sh
```

## Replacement Map

Use these operator commands instead of calling old `scripts/` files directly:

| old direct entry | replacement |
|---|---|
| `scripts/start_material_refine_trainV5_material_priority_supervisor.sh` | `bash datasetscrip/trainv5_supervisor.sh` |
| `scripts/run_material_refine_trainV5_material_priority_supervisor.sh` | `python datasetscrip/trainv5_dataset.py supervisor` |
| `scripts/stage_material_refine_material_priority_sources.py` | `python datasetscrip/trainv5_dataset.py ingest` |
| `scripts/build_material_refine_trainV5_expansion_second_pass.py` + repair | `python datasetscrip/trainv5_dataset.py queue` |
| `scripts/resolve_next_trainv5_b_batch.py` / `maybe_launch_next_trainv5_b_batch.py` | `python datasetscrip/trainv5_dataset.py launch` |
| `scripts/finalize_material_refine_trainV5_b_track.py` | `python datasetscrip/trainv5_dataset.py finalize --batch <batch_name>` |

The old supervisor scripts now delegate to `datasetscrip/` and should not be
edited with independent business logic again.

## Phase 2 Replacement Status

- Operator-facing replacement is complete: use `datasetscrip/trainv5_dataset.py`
  and `datasetscrip/trainv5_supervisor.sh`.
- Active ingest no longer calls
  `scripts/manage_trainv5_objaverse_1200_serial_plan.py`.
- Objaverse material-priority selection, retry, topup, and status writing are
  now parameterized through `scripts/stage_material_refine_material_priority_sources.py`
  as an internal helper called by `datasetscrip/trainv5_dataset.py ingest`.
- The Objaverse 1200 file names are still emitted for compatibility with the
  current status/brief tools, but the active path is no longer tied to the old
  batch-specific manager script.

## Phase 3 Replacement Status

- `datasetscrip/internal/` is now the internal orchestration layer behind the
  public `datasetscrip/trainv5_dataset.py` CLI.
- The public CLI no longer embeds long command-construction logic directly.
- Internal orchestration is split into responsibility modules:
  `common`, `ingest`, `queue`, `status`, `launch`, and `finalize`.
- Remaining `scripts/` calls are internal compatibility helpers and live
  rebake/finalize dependencies. They must be migrated behind `datasetscrip/internal/`
  before deletion.

## Calling Principles

- Prefer `datasetscrip/trainv5_dataset.py` over adding new one-off scripts.
- Add new source behavior through config or CLI flags, not `build_*_iterN.py`.
- Keep operator-facing behavior in `datasetscrip/`; keep reusable heavy
  implementation in existing internal `scripts/` helpers until a full batch has
  run successfully through the new entrypoint.
- Keep `scripts/prepare_material_refine_dataset.py`,
  `scripts/finalize_material_refine_trainV5_b_track.py`,
  `scripts/build_material_refine_trainV5_abc.py`, and
  `scripts/monitor_material_refine_trainV5_b_rebake.py` as internal helpers.
- Do not inject records into a running rebake batch. Freeze a manifest, launch it,
  and send failures/deferred records back to the rolling queues.
- Failed or unknown records should enter deferred/problem outputs. They should
  not block the main chain unless a batch-level fail rate exceeds the configured
  engineering threshold.

## Current Defaults

The default config is `datasetscrip/trainv5_dataset_config.json`.

- GPU: `0`
- render protocol: `production_32`
- render resolution: `320`
- cycles samples: `8`
- parallel workers: `2`
- min launch records: `256`
- max fail rate: `0.30`

## Deprecated Direction

Do not add more batch-specific launcher scripts such as:

- `maybe_launch_<specific_batch>.py`
- `manage_<specific_source>_<specific_count>.py`
- `cutover_<specific_batch>.py`
- `build_*_iterN.py`

If the behavior is still useful, expose it as a parameter or config option on
`datasetscrip/trainv5_dataset.py`.

The first batch of deprecated one-off scripts has been removed from active code:

- `scripts/maybe_launch_objaverse_1200_serial_after_b1.py`
- `scripts/manage_trainv5_objaverse_1200_serial_plan.py`
- `scripts/cutover_trainv5_b1_rebake.py`
- `scripts/build_material_refine_trainV5_plus_a_withprior_iter1.py`

Use `datasetscrip/TRAINV5_SCRIPT_DELETION_PLAN.md` for the current
keep/migrate/delete classification before removing additional `scripts/` files.
