# Asset Supervision Second-Pass Plan

## Route

- 主线: `ecommerce`
- 后置 OOD: `landscape`
- reference only: `official_2000`

## Pool Sizes

- ecommerce total: 2000
- landscape total: 1000
- official_2000 total: 2000

## Priority Tiers

- Tier 1: ecommerce native `glb/gltf`, `needs_conversion = false` -> 1980 objects
- Tier 2: ecommerce conversion pool (`obj/fbx`) -> 20 objects
- Tier 3: landscape OOD pool -> 1000 objects

### Tier 1 Source Breakdown

| source | count |
| --- | ---: |
| abo_selected | 1980 |

### Tier 2 Format Breakdown

| format | count |
| --- | ---: |
| obj | 20 |

### Tier 3 Format Breakdown

| format | count |
| --- | ---: |
| fbx | 235 |
| glb | 393 |
| obj | 372 |

## Generated Files

- [asset_supervision_audit_second_pass_ecommerce_glb_priority_300.csv](/4T/CXY/Neural_Gaffer/md/asset_supervision_audit_second_pass_ecommerce_glb_priority_300.csv)
- [asset_supervision_audit_second_pass_ecommerce_glb_priority_500.csv](/4T/CXY/Neural_Gaffer/md/asset_supervision_audit_second_pass_ecommerce_glb_priority_500.csv)
- [asset_supervision_audit_second_pass_ecommerce_conversion_pool.csv](/4T/CXY/Neural_Gaffer/md/asset_supervision_audit_second_pass_ecommerce_conversion_pool.csv)
- [asset_supervision_audit_second_pass_landscape_ood_pool.csv](/4T/CXY/Neural_Gaffer/md/asset_supervision_audit_second_pass_landscape_ood_pool.csv)

## Second-Pass Scope

第二轮只建议优先补这些列:

- `has_uv`
- `has_albedo`
- `has_roughness`
- `has_metallic`
- `has_normal`
- `roughness_valid`
- `metallic_valid`
- `needs_material_bake`
- `reject_reason`
- `notes`

## Upgrade Rules

### Promote to `A_ready`

- `has_uv = true`
- `has_albedo = true`
- `has_roughness = true`
- `has_metallic = true`
- `roughness_valid = true`
- `metallic_valid = true`
- `needs_conversion = false`
- `needs_material_bake = false`

### Keep as `B_fixable`

- asset structure is good
- but still needs conversion / relink / bake / node extraction

### Downgrade to `C_render_only`

- mesh is readable, but RM supervision is not成立
- e.g. no UV, base color only, or roughness / metallic are constant placeholders

### Mark as `D_drop`

- import failure
- corrupt file
- irrecoverable material
- repair cost is not worth it

## Recommended Next Action

- 先对 `ecommerce_glb_priority_300` 做 second-pass mini-batch probe
- 如果 `A_ready` 比例足够，再扩到 `500`
- `ecommerce_conversion_pool` 先只做格式统一，不跟材质 probe 混跑
- `landscape_ood_pool` 暂时只保留为后置泛化池
