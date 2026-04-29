# Material Refine Continuous Dataset Factory

This document records the continuous SF3D material-refine dataset expansion setup. It is data-only and does not launch training.

## Source Plan

- Pool-A paper seed: `ABO_locked_core`, from `output/highlight_pool_a_8k/material_refine_manifest_pool_a_8k.json`.
- Pool-A auxiliary upgrade: `3D-FUTURE_highlight_local_8k`, from the same Pool-A 8k manifest, kept in the noncommercial/research license bucket until promotion gates pass.
- Pool-A auxiliary upgrade: `Objaverse-XL_cached_strict_increment`, staged from cached Objaverse parquet/downloads, then converted to canonical object manifests.
- Pool-D lighting: `PolyHaven_HDRI`, target at least `900`, currently configured to extend toward `1200`.
- Pool-E material prior: `PolyHaven_Materials`, target `1000` CC0 PBR materials.
- Pool-B/Pool-F real benchmark sidecars: OpenIllumination and Stanford-ORB manifests remain benchmark/calibration only and are not mixed into Pool-A training manifests.

## Status

The old continuous dataset factory runner was removed from the active TrainV5 path.
Use the TrainV5 builders and audit scripts instead:

- `scripts/build_material_refine_trainV5_initial.py`
- `scripts/build_material_refine_trainV5_plus_a_track.py`
- `scripts/audit_material_refine_trainV5_pairs.py`

The active runtime manifest is:

```text
train/trainV5_plus_a_track/trainV5_training_pairs.json
```

## Active Sessions

- `sf3d_longrun_material_refine_shard*_gpu0`: GPU0 preprocessing/render shards.
- `sf3d_longrun_material_refine_merged_monitor`: merged manifest monitor for the active longrun root.
- `sf3d_factory_polyhaven_hdri`: PolyHaven HDRI refresh/download retry loop.
- `sf3d_factory_objaverse_cached_increment`: cached Objaverse strict increment selector/downloader.
- `sf3d_factory_objaverse_strict_full_retry`: full Objaverse strict downloader retry loop.
- `sf3d_factory_polyhaven_material_bank`: PolyHaven PBR material bank downloader.

## Outputs

Factory longrun roots are written under:

```text
output/material_refine_dataset_factory/longrun_gpu0_<timestamp>/
```

The active merged manifest path is:

```text
output/material_refine_dataset_factory/longrun_gpu0_<timestamp>/canonical_manifest_monitor_merged.json
```

Quality audits are written under:

```text
output/material_refine_dataset_factory/quality/<timestamp>/
```

## Notes

- A lone `sf3d_longrun_material_refine_merged_monitor` no longer blocks new rendering; only active shard sessions block a new longrun.
- Download tasks are retry loops. Network failures such as `RemoteDisconnected` do not permanently stop the source expansion path.
- Downloaded Objaverse raw manifests are converted through `scripts/build_objaverse_increment_manifest.py` before becoming longrun input manifests.
- Paper promotion is still controlled by manifest quality gates; auxiliary/no-prior objects are not automatically treated as paper-main records.
