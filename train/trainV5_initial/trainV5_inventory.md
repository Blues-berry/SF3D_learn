# TrainV5 Initial Inventory

- generated_at_utc: `2026-04-28T01:48:23Z`
- target_bundles: `322`
- prior_variants: `322`
- training_pairs: `322`
- manifest_role: `source_intermediate`
- runtime_training_ready: `false`
- split: `{"test": 63, "train": 221, "val": 38}`
- material_family: `{"glossy_non_metal": 322}`
- prior_quality_bin: `{"near_gt": 322}`
- prior_spatiality: `{"scalar_broadcast": 322}`
- sanity_ready: `true`

This directory is an intermediate TrainV5 source handoff, not a runtime training manifest. Use `train/trainV5_plus_a_track/trainV5_training_pairs.json` for `train_material_refiner.py`.

Large view/UV/buffer images are not copied into `train/trainV5_initial`; manifests keep logical/physical paths to `output` or `/4T`.
