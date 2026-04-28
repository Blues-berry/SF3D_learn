# Material Refine 数据集现状分析与扩充规划 2026-04-23

## 总体结论

当前数据集已经从早期 smoke 阶段进入“可做单源 paper-stage 训练，但还不足以支撑顶会级泛化结论”的状态。

最稳的 paper-stage 主训练入口仍然是 locked ABO 346 subset。它的优点是路径完整、view supervision 完整、target/prior 非同源、confidence 高；缺点也非常明确：全部来自 `ABO_locked_core`，全部是 `glossy_non_metal`，全部有上游 material prior，没有 no-prior、多材质、多 source、多 generator 覆盖。

新增池子已经显著变大，尤其 `factory_promoted_2060` 有 2060 条，其中 862 条标记为 paper eligible，且包含大量 `3D-FUTURE_highlight_local_8k`、no-prior、多材质和 `gt_render_baked` target。它是下一轮扩充的核心来源，但不能直接全量并入主训练，因为其中仍有 `smoke_only`、`auxiliary_upgrade_queue`、`copied_from_prior`、license 隔离和 confidence 波动问题。

因此下一步不应简单“扩大训练集”，而应构建 `Stage1-v3 strict paper subset`：以当前 346 ABO 为稳定 anchor，从 promoted pool 中经过二次验证筛出多材质、no-prior、gt-render-backed 样本，按 license bucket 和 object-level split 固定进入训练。

## 当前数据集数量

盘点文件：

`output/material_refine_paper/dataset_inventory_20260423.json`

| Dataset | Records | Paper Eligible | Main Use | 主要问题 |
| --- | ---: | ---: | --- | --- |
| `current_main_346` | 346 | 346 | 当前主训练 | 单 source / 单材质 / with-prior only |
| `current_enriched_490` | 490 | 346 | 历史扩展池 | 144 条仍为 smoke/auxiliary，87 条 identity-like |
| `stage1_v2_strict_362` | 362 | 362 | 候选 strict paper | 只比 346 多 16 条，分布没有变宽 |
| `stage1_v2_diagnostic_512` | 512 | 19 | 诊断 / OOD / 可视化 | 493 条 smoke_only，不进论文主训练 |
| `stage1_v2_ood_256` | 256 | 0 | OOD eval-only | 全部 smoke_only，全部 no-prior |
| `longrun_subset_211` | 211 | 211 | ABO pending reconcile | 数量小，仍是 glossy with-prior |
| `factory_promoted_2060` | 2060 | 862 | 下一轮扩充核心池 | license、quality、copied prior、confidence 需二次验证 |

## 当前数据质量判断

### 1. current main 346

数量与质量：

- records：346
- paper eligible：346
- paper split：train 240，val 40，test_iid 31，test_material_holdout 35
- source：`ABO_locked_core: 346`
- generator：`abo_locked_core: 346`
- material family：`glossy_non_metal: 346`
- license：`cc_by_nc_4_0: 346`
- target quality：`paper_pseudo: 346`
- target source：`pseudo_from_multiview: 346`
- has material prior：true 346
- view supervision ready：true 346
- identity-like count >= 0.999：0
- nontrivial target count：346
- target/prior distance mean：0.11096
- target confidence mean：0.94987
- path completeness：关键路径无缺失

判断：

这套数据适合作为稳定 paper-stage Stage1 主训练入口，但只能支持“ABO with-prior glossy non-metal 场景有效”的结论。它不能证明 no-prior、多 generator、metal/glass/boundary-heavy 材质泛化。

### 2. stage1_v2 strict 362

数量与质量：

- records：362
- paper eligible：362
- source：`ABO_locked_core: 362`
- material family：`glossy_non_metal: 362`
- target quality：`paper_pseudo: 362`
- view supervision ready：true 362
- nontrivial target count：362
- confidence mean：0.94987
- path completeness：关键路径无缺失

判断：

它比当前 346 多 16 条，但仍然没有引入 non-glossy material、no-prior 或新 generator。现阶段不值得替换主训练集，只能作为补充候选。替换条件应至少满足：strict records >= 384，并且每个新增材质族至少 16 条 paper eligible；更稳妥的论文目标是每个主要材质族 >= 64 条。

### 3. stage1_v2 diagnostic 512

数量与质量：

- records：512
- paper eligible：19
- source：`3D-FUTURE_highlight_local_8k: 492`，`ABO_locked_core: 19`，`Objaverse-XL_strict_filtered_increment: 1`
- material family：`metal_dominant: 369`，`glossy_non_metal: 48`，`ceramic_glazed_lacquer: 46`，`mixed_thin_boundary: 25`，`glass_metal: 24`
- target source：`gt_render_baked: 492`，`pseudo_from_multiview: 20`
- target quality：`smoke_only: 493`，`paper_pseudo: 19`
- no-prior：493
- confidence mean：0.64737
- path completeness：关键路径无缺失

判断：

这是目前最有价值的诊断集，因为它覆盖 metal、glass、ceramic、thin-boundary 和 no-prior。它暂时不能进 paper-stage 主训练，但应该立刻进入 eval-only、failure taxonomy、OOD 分析和二次验证队列。

### 4. stage1_v2 OOD 256

数量与质量：

- records：256
- paper eligible：0
- source：`3D-FUTURE_highlight_local_8k: 256`
- material family：`metal_dominant: 146`，`ceramic_glazed_lacquer: 32`，`glossy_non_metal: 29`，`mixed_thin_boundary: 25`，`glass_metal: 24`
- has material prior：false 256
- target quality：`smoke_only: 256`
- target source：`gt_render_baked: 256`
- confidence mean：0.62095
- path completeness：关键路径无缺失

判断：

它适合固定为 no-prior / 3D-FUTURE / material-diverse OOD test，不适合训练。后续报告中必须标注为 diagnostic-only 或 eval-only，不能写入论文主表的训练结论。

### 5. factory promoted 2060

数量与质量：

- records：2060
- paper eligible：862
- source：`3D-FUTURE_highlight_local_8k: 1739`，`ABO_locked_core: 256`，`Objaverse-XL_strict_filtered_increment: 65`
- material family：`metal_dominant: 1558`，`glossy_non_metal: 348`，`ceramic_glazed_lacquer: 66`，`mixed_thin_boundary: 47`，`glass_metal: 41`
- target quality：`smoke_only: 1198`，`paper_strong: 635`，`paper_pseudo: 227`
- target source：`gt_render_baked: 1739`，`pseudo_from_multiview: 308`，`copied_from_prior: 13`
- no-prior：1804
- view supervision ready：2047 true，13 false
- identity-like count >= 0.999：13
- confidence mean：0.68272
- path completeness：关键路径无缺失

判断：

这是下一轮扩充主池。它已经具备论文需要的多材质、多 source、no-prior 和 gt render target，但当前不能无筛选并入：13 条 copied/prior-like 必须剔除，13 条 view supervision 不完整必须剔除或降级，`smoke_only` 必须经过二次验证提升质量等级，license bucket 必须隔离。

## 主要结构性问题

1. Paper 主训练分布过窄。

当前 346/362 strict paper 都是 ABO glossy with-prior。即使模型指标提升，也只能说明在这个窄域内有效。

2. 多材质主要停留在 diagnostic。

metal/glass/ceramic/thin-boundary 的样本已经存在，但大部分是 `smoke_only` 或 `auxiliary_upgrade_queue`。这会造成训练和论文结论断层：eval 能看到复杂样本，但训练没有合法使用它们。

3. no-prior 覆盖不足。

paper main 中 no-prior 为 0，而系统方法里有 Prior-bootstrap path。这个模块如果没有训练样本，会变成“接口存在但贡献未验证”。

4. Generator/source 泛化还没有进入主训练。

当前主训练 generator_id 只有 `abo_locked_core`。`3D-FUTURE` 和 `Objaverse-XL` 已在池子中，但需要明确是 paper train、diagnostic-only 还是 OOD eval。

5. Target quality 命名还需要严格执行。

`factory_promoted_2060` 同时存在 `paper_strong`、`paper_pseudo`、`smoke_only`。后续训练脚本必须默认只接受 `paper_strong/paper_pseudo`，并显式拒绝 `smoke_only`，除非配置声明为 diagnostic 或 curriculum pretrain。

6. License bucket 必须贯穿训练、checkpoint 和报告。

3D-FUTURE 样本主要是 `custom_tianchi_research_noncommercial_no_redistribution`。可用于内部研究训练时，checkpoint metadata、report、demo export 必须记录 license 构成，不能和可公开展示桶混淆。

## 扩充目标

### Stage1-v3 strict paper subset

建议目标：

- total records：800-1000
- paper eligible：>= 800
- view supervision ready：>= 99%
- copied_from_prior：0
- identity-like rate：<= 0.30，且最好 <= 0.10
- target confidence mean：>= 0.75
- 每个主要 material family 至少 64 条，理想 >= 100 条
- no-prior records：15%-25%
- secondary source/generator records：>= 100
- object-level split 固定，不允许隐式重切

推荐材质比例：

- `metal_dominant`：30%
- `ceramic_glazed_lacquer`：20%
- `glass_metal`：15%
- `mixed_thin_boundary`：20%
- `glossy_non_metal`：15%

说明：

当前 promoted pool 中 `metal_dominant` 过多，`glass_metal` 和 `mixed_thin_boundary` 偏少。扩充时不应按自然分布采样，而应按材质族 quota 重采样，避免模型进一步偏向 metal。

### Stage1-v3 diagnostic subset

建议目标：

- records：512-1024
- 允许 `smoke_only`，但必须标记 `diagnostic_only`
- 覆盖 no-prior、OOD source、hard boundary、highlight-heavy、metal/non-metal confusion
- 每个 material family 至少 32 条
- 不进入论文主训练，不参与主表训练结论

### Stage1-v3 OOD eval subset

建议目标：

- records：256-512
- source/generator 与训练集尽量不同
- no-prior 和 cross-generator 分开统计
- 固定 object-level split
- 用于 OOD robustness 和 generator-aware report

## 扩充执行计划

### P0：二次验证 promoted pool

输入：

`output/material_refine_paper/reworked_candidates/factory_promoted/latest/canonical_manifest_promoted.json`

规则：

- 剔除 `target_source_type in {copied_from_prior, unknown}`
- 剔除 `view_supervision_ready != true`
- 剔除 `target_is_prior_copy == true`
- 剔除 identity-like records，阈值建议 `target_prior_identity >= 0.999`
- paper train 只允许 `target_quality_tier in {paper_strong, paper_pseudo}`
- `smoke_only` 进入二次验证队列，不直接训练
- license bucket 必须保留，不合并成一个训练桶
- 对 `gt_render_baked` 样本检查 confidence、view coverage、UV target non-empty、mask coverage

输出：

- `stage1_v3_strict_paper_candidates.json`
- `stage1_v3_auxiliary_upgrade_queue.json`
- `stage1_v3_rejects.json`
- `stage1_v3_dataset_audit.json`

### P1：按材质族和来源构建 balanced subset

优先从 `factory_promoted_2060` 中挑选：

- 保留当前 ABO 346 作为 glossy anchor。
- 从 3D-FUTURE 中选择通过二次验证的 `paper_strong gt_render_baked`。
- 从 Objaverse-XL 中只选择 license clear、路径完整、target 非同源的样本。
- 如果 Neural Gaffer 数据能提供可靠 reference render，可优先作为 render-consistency eval 和 lighting stress test；只有补齐 canonical UV RM target 后才进入训练。

建议第一版 Stage1-v3：

- ABO glossy anchor：250-346
- 3D-FUTURE metal dominant：200-260
- ceramic/glazed/lacquer：120-160
- mixed thin boundary：120-160
- glass+metal：80-120
- Objaverse / other source：64-128

### P2：固定 split protocol

要求：

- object-level split，不按 render row 随机。
- train/val/test 分布报告必须包含 source、generator_id、material_family、has_material_prior、target_quality_tier、target_source_type、license_bucket。
- test 至少拆成 IID、material holdout、OOD source 三类。
- no-prior 样本在 train/val/test 都要有覆盖，否则 Prior-bootstrap path 只能作为 diagnostic。

建议比例：

- train：70%
- val：10%-15%
- IID test：10%
- material holdout test：5%-10%
- OOD eval：单独固定，不混进 IID test

### P3：接入 Round9/Round10 后续训练

短期策略：

- 当前 Round9/Round10/Round11 仍使用 locked 346 作为 paper-safe baseline。
- Stage1-v2 diagnostic 和 OOD 只用于 eval-only，观察 metal/non-metal、boundary、render proxy 是否真实改善。
- Stage1-v3 strict paper 未通过 audit 前，不启动 full paper training。

Stage1-v3 通过后：

- 新建 `configs/material_refine_train_paper_stage1_v3_balanced.yaml`
- 默认启用 balanced-object sampler：按 `material_family x source_name x has_material_prior` 采样。
- validation 每 1/40 progress milestone 触发，但 val set 使用 balanced val，不再只看 glossy ABO。
- W&B 主表同步分层指标，不上传大量 case，只上传少量 per-object panels。

## 建议新增门禁

训练 preflight 需要新增或强化：

- `min_paper_eligible_records >= 800`
- `min_material_family_records >= 64`
- `max_material_family_ratio <= 0.40`
- `min_no_prior_records >= 100`
- `min_secondary_generator_records >= 100`
- `max_identity_like_rate <= 0.10`，硬上限仍为 0.30
- `min_confidence_mean >= 0.75`
- `min_view_supervision_ready_rate >= 0.99`
- `allow_smoke_only_for_training = false`
- `require_license_bucket_metadata = true`

报告中必须明确区分：

- `paper-stage train`
- `paper-stage eval`
- `diagnostic-only`
- `OOD eval-only`
- `engineering smoke`

## 当前可立即执行的下一步

1. 用 `factory_promoted_2060` 生成二次验证候选清单，先不训练。
2. 对 862 条 paper eligible 逐项输出 material/source/license split audit。
3. 从 635 条 `paper_strong gt_render_baked` 中按材质族 quota 抽取 Stage1-v3 候选。
4. 对 1198 条 `smoke_only` 做 upgrade audit，能升为 `paper_pseudo` 的进入候选，不能升的保留 diagnostic。
5. 固定 `Stage1-v3 strict / diagnostic / OOD` 三份 manifest。
6. 在新数据通过门禁前，继续把现有 Round9/Round11 结果表述为 single-source paper-safe baseline，而不是完整泛化结果。

## 对论文实验的意义

扩充后的数据集应支撑三类结论：

- In-domain：ABO / glossy / with-prior 下，R 模块能相对 SF3D prior 改善 UV RM 与 render proxy。
- Material-sensitive：metal、glass、ceramic、thin-boundary 上，边界感知和材质拓扑模块能减少 boundary bleed 与 metal/non-metal confusion。
- Generator/source-aware：不同 source/generator 的 prior reliability 不同，Dual-Path Prior Initialization 和 source-aware calibration 能提升跨源鲁棒性。

在 Stage1-v3 之前，不建议把当前结果写成“通用材质精修方法已验证”。更准确的表述是：工程链路、单源训练和复杂诊断集已经就绪；下一步是通过 promoted pool 二次验证，把多材质/no-prior/cross-source 样本提升到 paper-stage 主训练。
