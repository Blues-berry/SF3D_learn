# Semantic Validation Summary

- subset_csv: /4T/CXY/Neural_Gaffer/md/asset_supervision_semantic_validation_subset_24.csv
- metrics_json: /home/ubuntu/ssd_work/projects/stable-fast-3d/output/abo_material_probe_semantic24/metrics.json
- output: /4T/CXY/Neural_Gaffer/md/semantic_validation_summary.md

## Scope

- objects: 24
- views: 72
- inferred_views: 0

## Subset Breakdown

- dual_material: 7
- single_material_texture_only: 8
- single_material_factor_texture: 9

## Error Snapshot

- inference_mode: gt_render_only
- mean_abs_roughness_error: n/a (skip_inference)
- mean_abs_metallic_error: n/a (skip_inference)
- objects_with_any_failure_tag: n/a (skip_inference)

## GT Signal Checks

- constant_like_roughness_objects: 4/24
- constant_like_metallic_objects: 11/24
- constant_like_metallic_near_zero: 11
- constant_like_metallic_near_one: 0
- low_semantic_variation_objects: 5/24
- note: constant-like metallic near 0 is often a valid non-metal signal and should be treated as a review flag, not an automatic reject.

## Constant-like Roughness Candidates

- abo_1e8379c254c91802bf04 | B074VKWJWP | dual_material | stats=(0.028228302175800007, 0.02875816822052002)
- abo_0384757ab6aa9e75220b | B07K6RH9H9 | single_material_texture_only | stats=(0.022677466894189518, 0.003921568393707275)
- abo_0b8bbb4402e19999dc0e | B07SSSBKPQ | single_material_texture_only | stats=(0.02445267140865326, 0.003921568393707275)
- abo_16d3dcd0cb0ad3bdd363 | B075X33SBS | single_material_texture_only | stats=(0.025919952740271885, 0.006535947322845459)

## Constant-like Metallic Candidates

- abo_1e8379c254c91802bf04 | B074VKWJWP | dual_material | stats=(0.0032963089955349765, 0.009150327183306217)
- abo_0054874c26e5aece25cb | B075QGD5HS | single_material_texture_only | stats=(0.0019297955635314186, 0.003921568859368563)
- abo_0384757ab6aa9e75220b | B07K6RH9H9 | single_material_texture_only | stats=(0.001931005894827346, 0.003921568859368563)
- abo_0b8bbb4402e19999dc0e | B07SSSBKPQ | single_material_texture_only | stats=(0.0019307219578574102, 0.003921568859368563)
- abo_1afd3907ebff2408cf38 | B07DYGBNC8 | single_material_texture_only | stats=(0.001930795144289732, 0.003921568859368563)
- abo_255ef8855cae79be53b0 | B075QLF8X7 | single_material_texture_only | stats=(0.0019256686403726537, 0.003921568859368563)
- abo_000996dae516e9f3a4a7 | B07NYSH2S4 | single_material_factor_texture | stats=(0.002092799792687098, 0.003921568393707275)
- abo_09853ea5df5d371c8186 | B073P648NJ | single_material_factor_texture | stats=(0.0031162274535745382, 0.007843137563516697)

## Low Semantic Variation Candidates

- abo_1e8379c254c91802bf04 | B074VKWJWP | dual_material | stats=(0.028228302175800007, 0.0032963089955349765)
- abo_0384757ab6aa9e75220b | B07K6RH9H9 | single_material_texture_only | stats=(0.022677466894189518, 0.001931005894827346)
- abo_0b8bbb4402e19999dc0e | B07SSSBKPQ | single_material_texture_only | stats=(0.02445267140865326, 0.0019307219578574102)
- abo_000996dae516e9f3a4a7 | B07NYSH2S4 | single_material_factor_texture | stats=(0.0341296698898077, 0.002092799792687098)
- abo_234724e7175795b85707 | B071SHLFLX | single_material_factor_texture | stats=(0.0348372341444095, 0.0019302077901860077)

## Decision Read

- This summary is a semantic calibration layer below the structural A_ready gate.
- If failures cluster in a narrow mode, tighten rules for that mode instead of replacing the dataset.
- If GT-only checks stay clean, the current ABO/ecommerce pool is strong enough to keep scaling before introducing new sources.
