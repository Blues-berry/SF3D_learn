# Asset Supervision Audit Summary

- audit_time: 2026-04-16T03:12:40Z
- auditor: first_pass_auto_v1
- main_supervision_pool: ecommerce + landscape
- legacy_reference_pool: official_2000

## Per-Subset Totals

| subset | total_objects |
| --- | ---: |
| ecommerce | 2000 |
| landscape | 1000 |
| official_2000 | 2000 |

## Status Breakdown

| subset | A_ready | B_fixable | C_render_only | D_drop |
| --- | --- | --- | --- | --- |
| ecommerce | 0 (0.0%) | 2000 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| landscape | 0 (0.0%) | 1000 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| official_2000 | 0 (0.0%) | 0 (0.0%) | 2000 (100.0%) | 0 (0.0%) |

## Main Supervision Pool

| status | count | ratio |
| --- | ---: | ---: |
| A_ready | 0 | 0.0% |
| B_fixable | 3000 | 100.0% |
| C_render_only | 0 | 0.0% |
| D_drop | 0 | 0.0% |

### Top Reject Reasons

| reason | count |
| --- | ---: |
| needs_material_probe | 3000 |
| needs_obj_to_glb | 392 |
| needs_fbx_to_glb | 235 |

### Next-Round Probe Count

- 3000

## Legacy / Reference Pool

| status | count | ratio |
| --- | ---: | ---: |
| A_ready | 0 | 0.0% |
| B_fixable | 0 | 0.0% |
| C_render_only | 2000 | 100.0% |
| D_drop | 0 | 0.0% |

### Top Reject Reasons

| reason | count |
| --- | ---: |
| source_asset_not_local | 2000 |

### Next-Round Probe Count

- 0

## Notes

- first-pass 只填最稳定、最便宜、最不容易误判的列。
- `has_uv` / `has_albedo` / `has_normal` / `has_roughness` / `has_metallic` / `roughness_valid` / `metallic_valid` 在这一轮允许保留 `unknown`。
- `official_2000` 被单列为 legacy/reference，不参与主监督池 ready rate。
