# Round9 数据集问题分析与适配记录 2026-04-23

## 结论

当前数据集的主要问题不是总量不足，而是 paper-stage 可训练样本的结构仍然单一。

- paper-stage 主训练仍应使用当前 locked 346 ABO subset。
- Stage1-v2 strict paper 目前是 362 条，但仍全部是 `ABO_locked_core / glossy_non_metal / with_prior`，不足以替换当前主训练入口。
- 多材质和 no-prior 覆盖主要来自 Stage1-v2 diagnostic/OOD，绝大多数是 `smoke_only` / `auxiliary_upgrade_queue`，只能用于诊断、OOD、模型选择，不能进入论文主训练。
- Objaverse cached 当前不适合进入训练：`target_prior_identity_rate=0.984375`，paper eligible 为 0，并且部分 view buffer 不完整。

因此 Round9 采用：

- 主训练：`current locked 346`
- full-test：current locked test split
- diagnostic-only：Stage1-v2 diverse diagnostic
- OOD eval-only：Stage1-v2 OOD eval

## 最新数据源统计

数据 supervisor 更新时间：

`2026-04-23T01:41:34Z`

来源统计：

- `locked346`：346 records，346 eligible，全部 `paper_pseudo / pseudo_from_multiview / ABO_locked_core / glossy_non_metal / with_prior`
- `longrun`：1496 records，211 eligible；其中 1255 条 3D-FUTURE metal_dominant 为 `smoke_only / no_prior`
- `paper_unlock`：253 records，223 eligible；仍然是 ABO glossy_non_metal with-prior
- `scarce`：153 records，3 eligible；多材质覆盖存在，但 150 条仍是 `smoke_only / no_prior`
- `objaverse_cached`：64 records，0 eligible；`target_prior_identity_rate=0.984375`

## Stage1-v2 2026-04-23 子集

生成命令输出：

`output/material_refine_paper/stage1_v2_dataset_20260423`

生成结果：

- unique records：1834
- strict paper：362
- diverse diagnostic：512
- OOD eval：256

strict paper 分布：

- source：`ABO_locked_core: 362`
- material：`glossy_non_metal: 362`
- prior：`with_prior: 362`
- license：`cc_by_nc_4_0: 346`，`cc_by_nc_4_0_pending_reconcile: 16`
- target：`paper_pseudo / pseudo_from_multiview: 362`

diagnostic 分布：

- source：`3D-FUTURE_highlight_local_8k: 492`，`ABO_locked_core: 19`，`Objaverse-XL_strict_filtered_increment: 1`
- material：`metal_dominant: 369`，`glossy_non_metal: 48`，`ceramic_glazed_lacquer: 46`，`mixed_thin_boundary: 25`，`glass_metal: 24`
- prior：`without_prior: 493`，`with_prior: 19`
- role：大部分为 `auxiliary_upgrade_queue`

OOD 分布：

- source：`3D-FUTURE_highlight_local_8k: 256`
- material：`metal_dominant: 146`，`ceramic_glazed_lacquer: 32`，`glossy_non_metal: 29`，`mixed_thin_boundary: 25`，`glass_metal: 24`
- prior：全部 `without_prior`
- role：全部 `auxiliary_upgrade_queue`

## 数据问题

1. paper-stage 样本仍是单材质族。

`strict_paper` 虽然从 346 增到 362，但没有引入 non-glossy paper-eligible 样本。它不能证明模型在 metal / glass / ceramic / thin boundary 上有效。

2. 多材质数据无法直接训练。

3D-FUTURE 和 scarce material 提供了 metal、ceramic、glass、mixed boundary，但多数是 `smoke_only`，不能进入论文主训练。

3. no-prior 泛化仍是诊断失败点。

现有 with-prior ABO 模型在 Stage1-v2 no-prior diagnostic/OOD 上明显退化，因此不能把 multi-generator/no-prior 当作已解决贡献。

4. Objaverse 目前质量门禁未过。

当前 Objaverse cached 的 target/prior identity 过高，且 paper license bucket 与完整性还未达到主训练要求。

## Round9 适配

新增 readiness 输出：

`output/material_refine_paper/round9_dataset_readiness_20260423/round9_dataset_readiness.json`

`output/material_refine_paper/round9_dataset_readiness_20260423/round9_dataset_readiness.md`

readiness 结论：

- recommendation：`KEEP_CURRENT_MAIN_AND_USE_STAGE1_V2_FOR_DIAGNOSTIC`
- strict replacement：`False`
- blocker：`strict_records=362<min_replace=384`
- blocker：`non_glossy_paper_material_count_below_16:{}`

新增配置：

- `configs/material_refine_train_paper_stage1_round9_conservative_boundary.yaml`
- `configs/material_refine_eval_paper_stage1_round9_conservative_boundary.yaml`
- `configs/material_refine_eval_stage1_v2_diagnostic_20260423.yaml`
- `configs/material_refine_eval_stage1_v2_ood_20260423.yaml`

Round9 conservative boundary 设计：

- 初始化：Round7 best checkpoint
- 主训练数据：current locked 346 中的 `paper_train`
- 验证数据：current locked 346 中的 `paper_val_iid`
- boundary loss：`weight=0.05`，`kernel=9`
- R-v2 新分支：本主线关闭；R-v2 继续通过 component ablation 评估
- validation：40 个 progress milestone，每 10 个 milestone 额外记录 render-proxy
- eval：full-test / diagnostic-only / OOD 分开执行

## 已启动任务

tmux：

`sf3d_round9_conservative_boundary_gpu1_20260423`

训练日志：

`output/material_refine_paper/stage1_round9_conservative_boundary/logs/train.log`

后续自动 eval 日志：

- `output/material_refine_paper/stage1_round9_conservative_boundary/logs/eval_full_test.log`
- `output/material_refine_paper/stage1_round9_conservative_boundary/logs/eval_diagnostic_v2.log`
- `output/material_refine_paper/stage1_round9_conservative_boundary/logs/eval_ood_v2.log`

当前启动检查：

- preflight 通过
- paper eligible：346 / 346
- target_prior_identity：0.0000
- effective_view_rate：1.0000
- checkpoint init：`missing=0`，`unexpected=0`
- GPU：`cuda:1`
- `CUDA_VISIBLE_DEVICES=None`，没有硬编码绑卡
- 首个 progress validation：`progress_001_of_040`，UV total `0.061786`，baseline `0.114024`，residual regression rate `0.2349`

观察命令：

```bash
tail -f output/material_refine_paper/stage1_round9_conservative_boundary/logs/train.log | grep -E '\\[train\\]|\\[val\\]|\\[epoch\\]|epoch '
```

## 后续门禁

Round9 conservative boundary 只有在以下条件满足时，才能进入论文主结论讨论：

- full-test UV RM MAE 优于或接近 Round7
- boundary_bleed_score 不劣于 Round7
- proxy PSNR / SSIM / LPIPS 不明显退化
- residual safety 不低于 Round6/Round7
- diagnostic/OOD 明确标为 diagnostic-only，不作为 paper-stage 主表

Stage1-v2 strict 只有在满足以下条件时，才允许替换 current locked 346：

- strict paper records 至少 384
- 至少一个 non-glossy material family 有 16 条以上 paper-eligible 样本
- target/prior identity rate 不超过 0.30
- view supervision ready rate 接近 1.0
- license bucket 可用于当前研究训练
