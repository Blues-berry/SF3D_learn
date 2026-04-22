# Material Refine 迭代日志 2026-04-22

## 当前结论

Round7 不能被表述为“整体视觉质量已经稳定提升”。它在 UV RM、view-space RM、LPIPS proxy 和 residual safety 上有真实进展，但 proxy render 的 PSNR/SSIM 退化，boundary bleed 仍明显差于 SF3D baseline。因此下一轮优化必须围绕边界扩散和渲染代理指标，而不是继续单纯压 UV MAE。

## NG 风格终端输出整改

已在 `scripts/train_material_refiner.py` 中加入 tqdm 进度条，paper-stage 配置默认启用：

- `progress_bar: true`
- `train_line_logs: false`
- `terminal_json_logs: false`
- `validation_progress_milestones: 40`

终端现在以单 epoch 进度条为主，只显示关键字段：

- `loss`: 当前训练总 loss。
- `lr`: 当前学习率。
- `ref`: UV RM refinement L1。
- `bnd`: boundary-band loss，Round8 开启后用于观察材质边界约束。
- `gn`: gradient norm。
- `v`: batch 内有效 view supervision 比例。
- `sps`: samples/second。
- `mem`: GPU max allocated memory。

保留了启动阶段检查信息，包括系统、设备、GPU、manifest preflight、dataset distribution、首批 tensor shape/device/数值范围检查。这样训练时不再刷大量 JSON，但仍能定位数据和设备问题。

## 数据同步状态

数据端没有停滞。7-day supervisor 在 `2026-04-22T07:08:39Z` 仍在更新：

- tmux sessions: `14`
- GPU0: `15552/32607 MB`, util `14%`
- GPU1: `15/32607 MB`, util `0%`
- longrun manifest: `846` records，其中 `211` paper-stage eligible，`635` smoke_only。
- paper_unlock manifest: `253` records，其中 `223` paper-stage eligible，`30` smoke_only。
- 最新质量报告仍显示 paper_unlock `paper_stage_ready=True`、`target_prior_identity_rate=0.1186`、`view_supervision_ready_rate=1.0000`。

当前最佳 paper-stage 训练入口仍是：

`output/material_refine_paper/latest_dataset_check_20260421/stage1_subset_merged490/paper_stage1_subset_manifest.json`

原因是它有 `346` 条 paper-stage eligible 样本，且 `target_prior_identity_rate=0.0`。新 longrun 虽然有更多 `metal_dominant` 和 `3D-FUTURE` 样本，但这些主要仍是 `smoke_only` 或 `auxiliary_upgrade_queue`，不能直接混入 paper-stage 主训练。后续应把它们用于 diagnostic/OOD 或等待数据端提升质量标签后再并入。

## Round7 Eval 结果

评测输出：

`output/material_refine_paper/stage1_round7_gradient_guard_eval/summary.json`

主要结果：

| 指标 | Baseline | Refined | Delta | 判断 |
| --- | ---: | ---: | ---: | --- |
| UV RM MAE total | 0.11596 | 0.05459 | +0.06137 | 真提升 |
| View RM MAE total | 1.03209 | 0.98106 | +0.05103 | 真提升 |
| Proxy PSNR | 9.41600 | 9.29687 | -0.11914 | 退化 |
| Proxy SSIM | 0.92240 | 0.91723 | -0.00517 | 退化 |
| Proxy LPIPS | 0.10988 | 0.10784 | +0.00204 | 小幅提升 |
| Boundary bleed score | 0.00598 | 0.02281 | -0.01683 | 明显退化 |
| RM gradient preservation | 0.59488 | 0.59706 | +0.00218 | 小幅提升 |
| Residual safety score | n/a | 0.50719 | n/a | 好于 Round6 |

对比 Round6：

- Round6 UV delta: `0.05407`，Round7 UV delta: `0.06137`，Round7 更好。
- Round6 view delta: `0.06361`，Round7 view delta: `0.05103`，Round7 略差。
- Round6 PSNR delta: `+0.60860`，Round7 PSNR delta: `-0.11914`，Round7 明显退化。
- Round6 boundary delta: `-0.01919`，Round7 boundary delta: `-0.01683`，Round7 边界略改善但仍不可接受。
- Round6 safety score: `0.48378`，Round7 safety score: `0.50719`，Round7 更安全。

## Round8 改动原则

新增一个默认关闭的训练项：`boundary_bleed_loss`。它与现有 `edge_aware_l1_loss` 不同：

- `edge_aware_l1_loss` 是目标边缘附近加权 L1。
- `boundary_bleed_loss` 会生成材质边界膨胀带，直接惩罚边界带误差，并额外惩罚“边界误差高于内部误差”的情况。

Round8 配置：

`configs/material_refine_train_paper_stage1_round8_boundary_band.yaml`

Round8 全 test split 评测配置：

`configs/material_refine_eval_paper_stage1_round8_boundary_band.yaml`

核心调整：

- 开启 `boundary_bleed_weight: 0.10`。
- 使用 `boundary_band_kernel: 7`。
- 降低 `smoothness_weight: 0.001`，避免过度平滑扩大边界 bleed。
- 维持 residual safety，避免为了修边界而无意义改动 prior。
- 略提高 `view_consistency_weight: 0.34`，尝试恢复 proxy render 指标。

## 风险

- 当前 paper-stage 训练集仍几乎全是 `glossy_non_metal`，材质族覆盖不足，metal/non-metal 结论只能作为诊断，不能作为主论文结论。
- Boundary-band loss 是针对当前 eval failure 的直接优化，可能牺牲少量 UV MAE，需要用 PSNR/SSIM/LPIPS 和 boundary score 共同裁决。
- 新 longrun 数据更丰富，但 paper-stage 标签还没完全解锁，暂不自动切换主训练 manifest。

## 下一步

先用 Round8 在当前 best paper-stage manifest 上训练，训练完成后用升级后的 eval 全 test split 重跑，并重点比较：

- `boundary_bleed_score`
- `proxy_render_psnr`
- `proxy_render_ssim`
- `proxy_render_lpips`
- `uv_rm_mae`
- `view_rm_mae`
- `prior_residual_safety`

只有当边界和渲染代理指标不再明显退化时，才把 Round8 作为下一轮有效实验候选。

## 后台任务

Round8 训练已设置为后台 tmux 任务：

`sf3d_round8_boundary_band_gpu1_20260422`

训练完成后，eval waiter 会自动启动全 test split 评测：

`sf3d_round8_boundary_band_eval_waiter_20260422`

主要日志路径：

- Train: `output/material_refine_paper/stage1_round8_boundary_band/logs/train.log`
- Eval: `output/material_refine_paper/stage1_round8_boundary_band_eval/logs/eval.log`
- Eval summary: `output/material_refine_paper/stage1_round8_boundary_band_eval/summary.json`
