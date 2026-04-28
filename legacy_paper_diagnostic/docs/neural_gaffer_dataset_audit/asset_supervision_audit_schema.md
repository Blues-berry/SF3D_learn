# Asset Supervision Audit Schema

这份表用于判断现有 Neural Gaffer 数据对象，是否能进入“资产监督”训练池。

目标不是只看“能不能继续渲染图片”，而是判断对象是否保留了足够的 3D / PBR 监督信息，可用于后续的:

- mesh supervision
- UV-aware material supervision
- roughness / metallic / albedo / normal GT
- novel-light / novel-view 评测

## 最小表结构

建议按对象一行，最少包含以下列:

| column | type | meaning |
| --- | --- | --- |
| `object_id` | string | 对象唯一 ID，建议直接用训练/manifest 使用的 UID |
| `subset` | enum | `ecommerce` / `landscape` / `official_2000` / `official_subset` |
| `source` | string | 数据来源，如 `abo_selected` / `gso_selected` / `kenney_selected` / `smithsonian_selected` / `objaverse` |
| `source_model_path` | string | 原始模型文件绝对路径 |
| `format` | enum | `glb` / `gltf` / `fbx` / `obj` |
| `has_mesh` | bool | 是否能成功解析出 mesh |
| `has_uv` | bool | 是否存在可用 UV |
| `has_albedo` | bool | 是否存在 albedo/basecolor/diffuse 贴图或可直接读取的 base color |
| `has_normal` | bool | 是否存在 normal 贴图，或可稳定导出 normal GT |
| `has_roughness` | bool | 是否存在 roughness 通道或 roughness 贴图 |
| `has_metallic` | bool | 是否存在 metallic 通道或 metallic 贴图 |
| `roughness_valid` | bool | roughness 是否是可训练的真实材质信号，而不是常数占位/空贴图/全白全黑坏值 |
| `metallic_valid` | bool | metallic 是否是可训练的真实材质信号，而不是常数占位/空贴图/全白全黑坏值 |
| `material_slot_count` | int | 材质槽数量 |
| `needs_conversion` | bool | 是否需要把 `obj/fbx` 转成统一格式，如 `glb` |
| `needs_material_bake` | bool | 是否需要从材质节点、MTL、分散贴图中重新整理/烘焙出监督贴图 |
| `audit_status` | enum | `A_ready` / `B_fixable` / `C_render_only` / `D_drop` |
| `reject_reason` | string | 不可用原因或主要问题摘要 |

## 推荐追加列

这几列不是必须，但很建议一起存:

| column | type | meaning |
| --- | --- | --- |
| `source_uid` | string | 原始源库中的对象 ID 或文件 ID |
| `texture_root` | string | 贴图目录根路径 |
| `license` | string | 数据许可，后面论文和开源整理会用到 |
| `notes` | string | 人工备注 |
| `audit_time` | string | 审计时间，ISO 8601 |
| `auditor` | string | 审计人/脚本版本 |

## 状态定义

### `A_ready`

可直接进入监督池。

最低建议标准:

- `has_mesh = true`
- `has_uv = true`
- `has_albedo = true`
- `has_roughness = true`
- `has_metallic = true`
- `roughness_valid = true`
- `metallic_valid = true`
- `needs_conversion = false`
- `needs_material_bake = false`

典型对象:

- 带完整 PBR 材质的 `glb/gltf`
- `obj + mtl + texture` 且能稳定转成统一材质格式的对象，如果已经完成统一转换，也可记为 `A_ready`

### `B_fixable`

当前不能直接进监督池，但通过一次确定性的预处理后可用。

常见情况:

- `obj/fbx` 需要统一转换成 `glb`
- 有 mesh 和 UV，但贴图命名混乱，需要重映射
- 有 base color，但 roughness / metallic 需要从节点或材质参数抽取
- 有多材质槽，需要整理成统一输出

典型规则:

- `has_mesh = true`
- 至少存在稳定的源模型
- 问题是“工程修复型”，不是“数据本体缺失型”

### `C_render_only`

只能继续做图像渲染、质化展示、输入测试、泛化测试，不适合做 RM GT 监督。

常见情况:

- 只有 mesh，没有可靠 PBR 通道
- 只有 albedo，没有 roughness / metallic
- roughness / metallic 只是常数占位，没有真实监督意义
- 没有 UV，但还能作为几何体继续做渲染

这是“还能继续用”，但不是“能做资产监督”。

### `D_drop`

直接丢弃，不进入后续资产监督链，也不建议继续投入修复时间。

常见情况:

- 文件损坏
- Blender / importer 无法稳定读取
- mesh 拓扑严重异常
- 材质完全缺失且无法补
- 重复对象、错误对象、无效对象
- 许可或来源存在问题

## `reject_reason` 建议口径

为方便统计，建议优先使用短标签，再补一句自然语言备注。可选标签例如:

- `missing_mesh`
- `missing_uv`
- `missing_albedo`
- `missing_roughness`
- `missing_metallic`
- `roughness_constant`
- `metallic_constant`
- `material_parse_failed`
- `blender_import_failed`
- `needs_obj_to_glb`
- `needs_texture_relink`
- `needs_material_bake`
- `corrupt_asset`
- `duplicate_asset`
- `license_risk`

## 判定建议

建议把判定拆成两层:

1. 结构可用性
   - mesh
   - UV
   - 材质槽
   - 源格式是否可解析

2. 材质监督可用性
   - albedo
   - normal
   - roughness
   - metallic
   - 通道是否真实有效

最终状态可按下面的经验规则打:

- 结构和材质都齐: `A_ready`
- 结构齐，但材质需要工程整理: `B_fixable`
- 结构可渲染，但材质监督不成立: `C_render_only`
- 结构本身不可用或成本不值得: `D_drop`

## 建议输出格式

建议同时维护两份:

1. `asset_supervision_audit.csv`
   - 主审计表
   - 方便脚本统计

2. `asset_supervision_audit_summary.md`
   - 汇总每个 subset 的对象数、状态占比、主要 reject reason

## First-Pass Defaults

第一轮自动审计建议只填最稳定、最便宜、最不容易误判的列:

- `object_id`
- `subset`
- `source`
- `source_uid`
- `source_model_path`
- `format`
- `has_mesh`
- `material_slot_count`
- `needs_conversion`
- `audit_status`
- `reject_reason`
- `audit_time`
- `auditor`

第一轮允许保留 `unknown` 的列:

- `has_uv`
- `has_albedo`
- `has_normal`
- `has_roughness`
- `has_metallic`
- `roughness_valid`
- `metallic_valid`

第一轮的目标不是深挖每个材质节点，而是先把对象分成:

- 能不能读
- 要不要转
- 值不值得继续 probe

## Pool Split

统计时建议固定拆成两池:

- 主监督池: `ecommerce + landscape`
- legacy/reference: `official_2000`

`official_2000` 当前更适合作为 reference 支线，不应污染主监督池的 ready rate。

## 当前项目上的直接用法

按我们现在已确认的现状，最值得优先审计的是:

- `ecommerce`
- `landscape`

其中 `official_2000` 目前更像“有渲染体系和 UID 映射，但未确认本地完整保留 Objaverse 原始资产库”，所以应单独记为一条审计支线，不要和外部子集混在一张 ready pool 统计里。
