# Stage1-v3 Round12 迭代记录 2026-04-23

## 数据状态

本轮使用新生成的 `stage1_v3_dataset_latest`。它比 Stage1-v2 明显更接近真实训练需求，但仍不能直接作为 paper-stage full training。

关键文件：

- `output/material_refine_paper/stage1_v3_dataset_latest/stage1_v3_dataset_audit.json`
- `output/material_refine_paper/stage1_v3_dataset_latest/stage1_v3_balanced_paper_manifest.json`
- `output/material_refine_paper/stage1_v3_dataset_latest/stage1_v3_diagnostic_manifest.json`
- `output/material_refine_paper/stage1_v3_dataset_latest/stage1_v3_ood_eval_manifest.json`

当前审计结论：

- recommendation：`KEEP_AS_DATA_CANDIDATE_ONLY`
- stage1_v3_ready：`False`
- strict candidates：868
- balanced paper：454
- diagnostic：1024
- OOD eval：512

主要 blocker：

- `balanced_paper_records=454 below 800`
- `ceramic_glazed_lacquer=20 below 64`
- `glass_metal=8 below 64`
- `mixed_thin_boundary=21 below 64`
- `max_material_family_ratio=0.595 above 0.400`
- `strict_candidates_max_material_family_ratio=0.665 above 0.400`

## Balanced Paper 子集质量

`stage1_v3_balanced_paper_manifest.json` 当前可用于 data-adaptation / diagnostic training。

数量：

- records：454
- train：315
- val：38
- test_iid：46
- test_material_holdout：38
- test_ood_object：1

分布：

- source：`3D-FUTURE_highlight_local_8k: 334`，`ABO_locked_core: 119`，`Objaverse-XL_strict_filtered_increment: 1`
- material：`metal_dominant: 270`，`glossy_non_metal: 135`，`mixed_thin_boundary: 21`，`ceramic_glazed_lacquer: 20`，`glass_metal: 8`
- target quality：`paper_strong: 334`，`paper_pseudo: 120`
- target source：`gt_render_baked: 334`，`pseudo_from_multiview: 120`
- prior：without_prior 335，with_prior 119
- view supervision：454/454 ready
- target prior copy：0
- target identity-like：0

判断：

这套数据可以用于验证 no-prior、多 source、多材质训练是否改善模型，但不能作为最终 paper full training。尤其 `metal_dominant` 占比仍偏高，`glass_metal` 只有 8 条，不能支撑稳定论文结论。

## Round12 策略

本轮命名为 `data_adaptation`，不是 paper full。

训练配置：

`configs/material_refine_train_stage1_v3_round12_balanced_data_adaptation.yaml`

评测配置：

- `configs/material_refine_eval_stage1_v3_round12_balanced_test.yaml`
- `configs/material_refine_eval_stage1_v3_round12_locked346_regression.yaml`
- `configs/material_refine_eval_stage1_v3_round12_ood.yaml`

初始化：

`output/material_refine_paper/stage1_round10_r8_full/best.pt`

原因：

Round10 是当前 R-8 全模块结构的稳定 checkpoint。Round11b 边界更保守，但 UV/PSNR 收益明显弱于 Round10，因此 Round12 先用 Round10 做 warm start，再通过 v3 多材质/no-prior 数据纠正数据分布问题。

## Smoke 结果

已运行 1-step smoke：

`output/material_refine_paper/stage1_v3_round12_balanced_data_adaptation_smoke`

已确认：

- preflight 通过
- 454/454 paper eligible
- target prior identity rate：0.0000
- effective view supervision rate：1.0000
- checkpoint load：missing 0，unexpected 0，skipped shape 0
- train split：315 records
- val split：38 records
- train 分布含 no-prior 231、with-prior 84
- train 分布含 5 个 material family
- forward/backward/optimizer step 已跑通

暴露问题：

- smoke 使用 `num_workers=0` 时 I/O 很慢，1 个 optimizer step 约 127 秒。
- 主要瓶颈来自 512 atlas、多 view buffer 读取和 startup probe。
- 正式训练必须使用 `num_workers=4`，并减少单样本 view 数量。

已调整：

- train views：12 -> 10
- val views：24 -> 16
- startup probe batches：2 -> 1

## 正式 Round12 启动状态

tmux：

`sf3d_stage1_v3_round12_balanced_gpu1_20260423`

训练日志：

`output/material_refine_paper/stage1_v3_round12_balanced_data_adaptation/logs/train.log`

W&B：

`https://wandb.ai/chexueyuan2027-southeastern-university/stable-fast-3d-material-refine/runs/5adhqrn2`

启动检查：

- preflight：通过
- train records：315
- val records：38
- checkpoint：Round10 R-8 full，shape compatible，missing 0，unexpected 0
- GPU：`cuda:1`
- `CUDA_VISIBLE_DEVICES=None`
- train distribution：3D-FUTURE 231，ABO 84
- train prior distribution：no-prior 231，with-prior 84
- train material distribution：metal 189，glossy 94，mixed boundary 14，ceramic 13，glass 5
- val material distribution：glossy 16，metal 16，mixed boundary 3，ceramic 2，glass 1
- data path check：train/val 关键 UV 和 buffer 路径全部存在

早期 validation：

- baseline UV total：0.281590
- progress 1 refined UV total：0.589976
- progress 2 refined UV total：0.576493
- progress 3 refined UV total：0.528047
- progress 4 refined UV total：0.449044
- progress 5 refined UV total：0.403067
- progress 6 refined UV total：0.340792

早期判断：

模型正在快速适应 v3 多源/no-prior 数据，但 progress 6 仍未打过当时的 input-prior baseline（主要来自 SF3D prior）。progress 5 的 proxy view delta 为负，说明 render proxy 也还没有恢复。当前继续训练到中段；如果 progress 10-15 仍无法接近 baseline，则下一轮不应继续加学习率，而应改成更强 residual gate / prior-bootstrap safety 版本。

## 验收逻辑

Round12 训练后必须同时看三类结果：

1. `balanced_test`

验证 v3 新数据内部是否真正提升，重点看 no-prior、metal、boundary-heavy 组。

2. `locked346_regression`

检查是否遗忘原始 ABO with-prior glossy 场景。若 locked346 明显退化，则不能替代 Round10/Round11。

3. `ood`

只作为 diagnostic-only，看 no-prior / 3D-FUTURE OOD 是否方向正确。包含 smoke_only 的结果不能进入论文主表。

## 下一步数据扩充要求

Stage1-v3 进入 paper full 前仍需补齐：

- balanced paper records >= 800
- `ceramic_glazed_lacquer >= 64`
- `glass_metal >= 64`
- `mixed_thin_boundary >= 64`
- 最大 material family 占比 <= 0.40
- Objaverse / 第二上游来源不应只有 1 条
- license bucket 必须继续隔离记录

当前最优先扩充方向不是继续加 metal，而是补 glass、ceramic、mixed thin-boundary，以及第二 source/generator。
