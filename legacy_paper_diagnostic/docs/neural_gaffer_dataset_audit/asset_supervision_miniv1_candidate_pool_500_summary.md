# Mini-v1 Candidate Pool 500

- source_csv: /4T/CXY/Neural_Gaffer/md/asset_supervision_audit_second_pass_ecommerce_glb_priority_500_probed.csv
- semantic_subset_csv: /4T/CXY/Neural_Gaffer/md/asset_supervision_semantic_validation_subset_24.csv
- semantic_metrics_json: /home/ubuntu/ssd_work/projects/stable-fast-3d/output/abo_material_probe_semantic24/metrics.json
- output_csv: /4T/CXY/Neural_Gaffer/md/asset_supervision_miniv1_candidate_pool_500.csv
- output_json: /4T/CXY/Neural_Gaffer/md/asset_supervision_miniv1_candidate_pool_500.json

## Decision

- fix the current 500 ABO/ecommerce objects as the mini-v1 primary candidate supervision pool
- keep the whole pool in place; do not reshuffle, replace, or downgrade the pool itself
- attach review labels only to the small set surfaced by semantic calibration

## Pool Counts

- total_objects: 500
- primary_candidates: 500
- keep_in_pool: 500

## Semantic Validation Status

- checked_clean: 12
- flagged: 12
- not_sampled: 476

## Review Counts

- needs_review: 12
- review_high: 6
- review_medium: 6

## Review Labels

- review_constant_metallic_near_zero: 11
- review_low_semantic_variation: 5
- review_constant_roughness: 4

## Review Objects

- abo_000996dae516e9f3a4a7 | B07NYSH2S4 | high | review_constant_metallic_near_zero;review_low_semantic_variation
- abo_0054874c26e5aece25cb | B075QGD5HS | medium | review_constant_metallic_near_zero
- abo_0384757ab6aa9e75220b | B07K6RH9H9 | high | review_constant_roughness;review_constant_metallic_near_zero;review_low_semantic_variation
- abo_09853ea5df5d371c8186 | B073P648NJ | medium | review_constant_metallic_near_zero
- abo_0b8bbb4402e19999dc0e | B07SSSBKPQ | high | review_constant_roughness;review_constant_metallic_near_zero;review_low_semantic_variation
- abo_16d3dcd0cb0ad3bdd363 | B075X33SBS | high | review_constant_roughness
- abo_18e93966bf91d8129a9a | B07QCQ1J7M | medium | review_constant_metallic_near_zero
- abo_1afd3907ebff2408cf38 | B07DYGBNC8 | medium | review_constant_metallic_near_zero
- abo_1e8379c254c91802bf04 | B074VKWJWP | high | review_constant_roughness;review_constant_metallic_near_zero;review_low_semantic_variation
- abo_234724e7175795b85707 | B071SHLFLX | high | review_constant_metallic_near_zero;review_low_semantic_variation
- abo_255ef8855cae79be53b0 | B075QLF8X7 | medium | review_constant_metallic_near_zero
- abo_26fb929af5a8ce3175eb | B07B4Z7XVD | medium | review_constant_metallic_near_zero
