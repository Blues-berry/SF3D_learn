# mini_v1 Manifest Summary

- manifest_version: mini_v1_abo_ecommerce_v0.1
- pool_name: mini_v1_candidate_pool
- generated_from: /4T/CXY/Neural_Gaffer/md/asset_supervision_miniv1_candidate_pool_500.csv
- semantic_subset: /4T/CXY/Neural_Gaffer/md/asset_supervision_semantic_validation_subset_24.csv
- semantic_metrics: /home/ubuntu/ssd_work/projects/stable-fast-3d/output/abo_material_probe_semantic24/metrics.json
- output_csv: /4T/CXY/Neural_Gaffer/md/mini_v1_manifest.csv
- output_json: /4T/CXY/Neural_Gaffer/md/mini_v1_manifest.json

## Decision Lock

- source_pool: ABO/ecommerce only
- structural_gate: 500/500 A_ready
- source_gate: 500/500 abo_selected
- policy: keep the 500-object pool intact, add soft review flags, avoid automatic demotion in this stage

## Split Counts

- train: 450
- val: 25
- test: 25
- main_supervision_total: 500
- review_flagged_total: 12
- review_flagged_train: 12
- review_high: 6
- review_medium: 6

## Review Flag Counts

- semantic_probe_needed: 12
- constant_like_roughness_review: 4
- metallic_near_zero_review: 11
- low_semantic_variation: 5

## Split Stratification

- train material_slot_count: {1: 444, 2: 6}
- val material_slot_count: {1: 24, 2: 1}
- test material_slot_count: {1: 24, 2: 1}

## Semantic Review Objects

| object_id | sku | semantic_stratum | review_priority | review_flags |
| --- | --- | --- | --- | --- |
| abo_000996dae516e9f3a4a7 | B07NYSH2S4 | single_material_factor_texture | high | semantic_probe_needed, metallic_near_zero_review, low_semantic_variation |
| abo_0054874c26e5aece25cb | B075QGD5HS | single_material_texture_only | medium | semantic_probe_needed, metallic_near_zero_review |
| abo_0384757ab6aa9e75220b | B07K6RH9H9 | single_material_texture_only | high | semantic_probe_needed, constant_like_roughness_review, metallic_near_zero_review, low_semantic_variation |
| abo_09853ea5df5d371c8186 | B073P648NJ | single_material_factor_texture | medium | semantic_probe_needed, metallic_near_zero_review |
| abo_0b8bbb4402e19999dc0e | B07SSSBKPQ | single_material_texture_only | high | semantic_probe_needed, constant_like_roughness_review, metallic_near_zero_review, low_semantic_variation |
| abo_16d3dcd0cb0ad3bdd363 | B075X33SBS | single_material_texture_only | high | semantic_probe_needed, constant_like_roughness_review |
| abo_18e93966bf91d8129a9a | B07QCQ1J7M | single_material_factor_texture | medium | semantic_probe_needed, metallic_near_zero_review |
| abo_1afd3907ebff2408cf38 | B07DYGBNC8 | single_material_texture_only | medium | semantic_probe_needed, metallic_near_zero_review |
| abo_1e8379c254c91802bf04 | B074VKWJWP | dual_material | high | semantic_probe_needed, constant_like_roughness_review, metallic_near_zero_review, low_semantic_variation |
| abo_234724e7175795b85707 | B071SHLFLX | single_material_factor_texture | high | semantic_probe_needed, metallic_near_zero_review, low_semantic_variation |
| abo_255ef8855cae79be53b0 | B075QLF8X7 | single_material_texture_only | medium | semantic_probe_needed, metallic_near_zero_review |
| abo_26fb929af5a8ce3175eb | B07B4Z7XVD | single_material_factor_texture | medium | semantic_probe_needed, metallic_near_zero_review |

## Notes

- flagged objects stay inside the main supervision manifest and keep review labels instead of being split out.
- `val` and `test` are deterministic holdouts built from the non-flagged pool so existing holdouts stay stable.
- `metallic_near_zero_review` remains a review-only soft flag, not a rejection or demotion signal.
