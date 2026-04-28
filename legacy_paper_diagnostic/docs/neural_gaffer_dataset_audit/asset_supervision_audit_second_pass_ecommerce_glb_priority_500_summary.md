# asset supervision audit second pass ecommerce glb priority 500

- audit_time: 2026-04-16T04:23:52Z
- auditor: second_pass_glb_json_probe_v1
- input: /4T/CXY/Neural_Gaffer/md/asset_supervision_audit_second_pass_ecommerce_glb_priority_500.csv
- output: /4T/CXY/Neural_Gaffer/md/asset_supervision_audit_second_pass_ecommerce_glb_priority_500_probed.csv

## Scope Note

- This pass is a GLB JSON/material-structure audit.
- It confirms mesh, UV, and declared PBR channel presence from local source assets.
- It does not yet prove that roughness/metallic maps are semantically strong supervision signals or non-constant textures.

## Source Breakdown

| source | count | ratio |
| --- | ---: | ---: |
| abo_selected | 500 | 100.0% |

## Status Breakdown

| status | count | ratio |
| --- | ---: | ---: |
| A_ready | 500 | 100.0% |
| B_fixable | 0 | 0.0% |
| C_render_only | 0 | 0.0% |
| D_drop | 0 | 0.0% |

## Top Reject Reasons

| reason | count |
| --- | ---: |

## Channel Coverage

- has_uv: 500/500 (100.0%)
- has_albedo: 500/500 (100.0%)
- has_normal: 500/500 (100.0%)
- has_roughness: 500/500 (100.0%)
- has_metallic: 500/500 (100.0%)
- roughness_valid: 500/500 (100.0%)
- metallic_valid: 500/500 (100.0%)

## Decision Hint

- 如果 `A_ready` 已经够形成 mini-v1，就继续扩到 500。
- 如果 `A_ready` 偏少，但 `B_fixable` 主要集中在材质字段缺失，就优先补 material bake / channel extraction。
- 如果大面积掉到 `C_render_only`，再考虑引入更干净的新源做增强池。
