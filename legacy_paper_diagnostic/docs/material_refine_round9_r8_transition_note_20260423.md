# Round9 -> R-8 Module Transition Note

日期：2026-04-23

## 当前结论

Round9 conservative boundary 训练已经完成 46/46 epoch，最佳验证点为 `epoch=41 / step=594`：

- best checkpoint：`output/material_refine_paper/stage1_round9_conservative_boundary/best.pt`
- latest checkpoint：`output/material_refine_paper/stage1_round9_conservative_boundary/latest.pt`
- W&B run：`material-refine-paper-stage1-round9-conservative-boundary-resume`

Round9 可以作为 **paper-test 上的工程有效 baseline**，但不能直接作为跨数据/跨材质泛化的最终论文方法，因为 diagnostic/OOD 退化明显。

## Round9 Full Eval 摘要

Paper test：

- `UV RM MAE delta = +0.069545`，refined 优于 SF3D/canonical prior。
- `View RM MAE delta = +0.079938`，view-space RM 有提升。
- `PSNR delta = +0.531967`，proxy render 像素误差改善。
- `SSIM delta = -0.005292`，结构相似度轻微退化。
- `LPIPS delta = +0.005176`，感知距离改善。
- `boundary_bleed_score delta = -0.016603`，边界 bleed 退化。
- `prior_residual_safety_score = 0.655679`，paper test 上残差安全性尚可。

Diagnostic subset：

- `UV RM MAE delta = -0.512379`。
- `View RM MAE delta = -1.002800`。
- `PSNR delta = -0.373139`。
- `SSIM delta = -0.010057`。
- `boundary_bleed_score delta = -0.061020`。
- `prior_residual_safety_score = -0.368785`。

OOD subset：

- `UV RM MAE delta = -0.604994`。
- `View RM MAE delta = -1.191771`。
- `PSNR delta = -0.586970`。
- `SSIM delta = -0.011327`。
- `boundary_bleed_score delta = -0.074023`。
- `prior_residual_safety_score = -0.533250`。

## 方法决策

不建议把 Round9 直接晋升为最终方法主结果。它说明当前 residual refiner 能在 locked Stage1 paper-test 上学习到有效 UV 修正，但同时暴露了三个方法短板：

- 边界区域缺少结构化安全门控，`boundary_bleed_score` 在 paper/diagnostic/OOD 都退化。
- 材质族泛化不足，diagnostic/OOD 上 residual update 过激，`prior_residual_safety_score` 变负。
- UV 指标和 render/structure 指标仍存在冲突，说明需要把 render consistency 从 eval 前移到 R 模块内部。

因此进入 R-8 module iteration：

- A：Dual-path prior initialization 解决 with-prior/no-prior 和 generator/source prior reliability。
- B-D：material-sensitive view encoder + hard-view routing + tri-branch fusion 解决多视图证据不分区的问题。
- E：boundary safety module 针对 boundary bleed 做结构化约束。
- F：material topology reasoning 把 texel-level 修正提升到 region-level 一致性。
- G：confidence-gated residual trunk 控制 change gate、uncertainty、boundary stability。
- H：inverse material-response check 把 render consistency 融入 update gate。

## 新增/更新入口

- 模型：`sf3d/material_refine/model.py`
- 训练入口：`scripts/train_material_refiner.py`
- Full R 配置：`configs/material_refine_train_paper_stage1_round10_r8_full.yaml`
- 方法文档：`docs/material_refine_r8_module_design.md`

## 验证结果

- `py_compile` 通过：`sf3d/material_refine/model.py`、`scripts/train_material_refiner.py`
- `diff --check` 通过。
- Round10/R8-full config preflight 通过：
  - `records=346`
  - `paper_eligible=346`
  - `target_prior_identity=0.0000`
  - `effective_view_rate=1.0000`
- 旧 Round9 checkpoint strict load 通过：
  - `missing=0`
  - `unexpected=0`
- 新 R-8 module 使用 Round9 checkpoint 做 compatible init 可运行，新增模块权重会正常随机初始化。
- 已补 warm-start 安全初始化：
  - A 模块有 prior 时初始 reliability bias 为高可信，`initial` 接近 prior。
  - G 模块新增 residual head 的输出层初始为 0，避免随机 residual 破坏旧 checkpoint。
  - E/H 的安全门控初始为保守通过，不在 step 0 制造大幅退化。
- CPU warm-start forward 检查通过：
  - `mean_abs_refined_minus_prior=0.000117`
  - `mean_abs_initial_minus_prior=0.000117`
  - `residual_delta_abs=0.0`
- CPU warm-start train smoke 通过：
  - output：`output/material_refine_paper/stage1_round10_r8_full_cpu_smoke_warmstart_20260423`
  - `baseline=0.080681`
  - `uv_total=0.094781`
  - `res_change=0.0000`
- GPU1 warm-start train smoke 通过：
  - output：`output/material_refine_paper/stage1_round10_r8_full_gpu1_smoke_warmstart_20260423`
  - `baseline=0.080695`
  - `uv_total=0.095114`
  - `proxy_view_delta=0.007917`
  - `res_change=0.0000`
  - peak memory from progress bar：约 `0.89GB`，说明小规模 R-8 full smoke 可在 GPU1 正常执行。
- GPU1 prior-locked warm-start train smoke 通过：
  - output：`output/material_refine_paper/stage1_round10_r8_full_gpu1_smoke_priorlocked_20260423`
  - `baseline=0.080681`
  - `uv_total=0.080680`
  - `proxy_view_delta=-0.000000`
  - `res_change=0.0000`
  - `res_reg=0.0000`
  - 这说明 Full R 初始状态已不会破坏 SF3D/canonical prior，后续训练提升来自可学习 residual，而不是随机扰动。

## 下一步

1. `configs/material_refine_train_paper_stage1_round10_r8_full.yaml` 已启动主训练。
2. 若 Round10 full 稳定，再跑 `A+B+D+G`、`+E`、`+F`、`+H`、`Full` ablation。
3. Full R 不以单一 UV MAE 作为晋升标准，必须同时检查 `boundary_bleed_score`、`PSNR/SSIM/LPIPS`、`prior_residual_safety` 和 diagnostic/OOD 退化是否收敛。

## Round10 运行状态

- tmux session：`sf3d_round10_r8_full_gpu1_20260423`
- log：`output/material_refine_paper/stage1_round10_r8_full/logs/train_round10_r8_full_20260423T_now.log`
- W&B run：`https://wandb.ai/chexueyuan2027-southeastern-university/stable-fast-3d-material-refine/runs/uvvbng5r`
- 启动状态：
  - `epoch=1/52`
  - `batches_per_epoch=80`
  - `optimizer_steps_per_epoch=20`
  - `total_optimizer_steps=1040`
  - GPU1 训练显存约 `12.28GB`

## Round10 恢复与收尾记录

### CUDA timeout 修复

Round10 主训练在 `epoch=50`、`progress_038/040` 后触发 CUDA timeout：

- 位置：`sf3d/material_refine/model.py` 的 `UVFeatureFusion._scatter_view_features`
- 原因：tri-branch fusion 每个 batch/view 都对 CUDA 标量调用 `.detach().item()`，长跑后容易形成频繁 host/device 同步并触发 launch timeout。
- 修复：将 `view_importance` 这个很小的 `B x V` routing 权重一次性 `detach().float().cpu().tolist()`，scatter 循环只读取 CPU float，不再逐 view 同步 CUDA scalar。

新增恢复配置：

- `configs/material_refine_train_paper_stage1_round10_r8_full_resume_latest.yaml`
- 清除 `init_from_checkpoint`
- 从 `output/material_refine_paper/stage1_round10_r8_full/latest.pt` 恢复 optimizer/scheduler/scaler
- 使用同一个 W&B run id：`uvvbng5r`

### 训练收尾结果

恢复训练从 `step_000988.pt` 继续，完成 `epoch=51-52`，没有复现 CUDA timeout。

关键 validation：

- `progress_038/040`，`epoch=50`，`step=988`
  - `baseline=0.114024`
  - `uv_total=0.040697`
  - `roughness=0.016157`
  - `metallic=0.024540`
  - `res_change=0.6102`
  - `res_reg=0.1373`
  - 该 checkpoint 是当前 best。
- `progress_040/040`，`epoch=52`，`step=1028`
  - `baseline=0.114024`
  - `uv_total=0.044643`
  - `roughness=0.017507`
  - `metallic=0.027136`
  - `proxy_view_delta=0.061581`
  - `res_change=0.6002`
  - `res_reg=0.1390`

结论：

- 当前 best checkpoint 保持为 `output/material_refine_paper/stage1_round10_r8_full/best.pt -> step_000988.pt`。
- 后两次 validation 没有超过 best，因此 `latest.pt` 也仍指向 `step_000988.pt`，这符合 `save_only_best_checkpoint=true` 的策略。
- Round10/R8-full 在 Stage1 单材质族数据上继续压低 UV RM error，但最后 proxy render delta 仍提示必须等待 full eval 判断渲染一致性。
- 当前 train/val 仍全部为 `glossy_non_metal`，不能把该结果表述为多材质 paper-stage 结论。

### Full-test eval

已新增并启动：

- `configs/material_refine_eval_paper_stage1_round10_r8_full.yaml`
- output：`output/material_refine_paper/stage1_round10_r8_full_eval`
- log：`output/material_refine_paper/stage1_round10_r8_full_eval/logs/eval_round10_r8_full_test_20260423T102740Z.log`
- W&B run：`material-refine-paper-stage1-round10-r8-full-test`

该 eval 会裁决：

- UV RM 是否真实优于 SF3D/canonical prior baseline。
- proxy render PSNR/SSIM/LPIPS 是否同步改善。
- `boundary_bleed_score`、`prior_residual_safety`、`metal_nonmetal_confusion` 等专项指标是否支持 R-8 结构晋升。
- 是否存在 UV-level 提升但 render/object-level 退化的 metric conflict。

### Full-test eval 结果与 Round11 决策

Round10 full-test eval 已完成：

- output：`output/material_refine_paper/stage1_round10_r8_full_eval`
- summary：`output/material_refine_paper/stage1_round10_r8_full_eval/summary.json`
- W&B run：`https://wandb.ai/chexueyuan2027-southeastern-university/stable-fast-3d-material-refine/runs/vb6ixt9m`
- objects：`66`
- rows/views：`1584`

与 Round7/Round8/Round9 对比：

| round | UV RM MAE ↓ | View RM MAE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Boundary Bleed ↓ | Safety ↑ | Gradient Preservation ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Round7 | 0.054585 | 0.981059 | 9.296866 | 0.917226 | 0.107844 | 0.022807 | 0.507188 | 0.597058 |
| Round8 | 0.058257 | 0.978749 | 9.699976 | 0.917556 | 0.105561 | 0.023946 | 0.477805 | 0.558619 |
| Round9 | 0.046414 | 0.952154 | 9.947972 | 0.917106 | 0.104706 | 0.022583 | 0.655679 | 0.578589 |
| Round10 | 0.041388 | 0.963984 | 9.977360 | 0.906225 | 0.112547 | 0.027378 | 0.793841 | 0.354237 |

判定：

- Round10/R8-full 不是最终可晋升模型。
- 可确认的有效点：
  - UV RM MAE 明显改善。
  - PSNR 小幅改善。
  - prior residual safety 明显改善。
- 主要退化：
  - view-space RM 比 Round9 变差。
  - SSIM/LPIPS 变差，说明感知/结构质量退化。
  - boundary bleed 变差。
  - RM gradient preservation 大幅下降，说明 full R 过度平滑/过度区域化。
- 结论表述：
  - 这是一轮 valid diagnostic experiment，不是 paper-stage best result。
  - R-8 full 的方向值得保留，但必须进入 boundary/render/gradient-safe 迭代。

### Round11 已启动

新增配置：

- `configs/material_refine_train_paper_stage1_round11_r8_boundary_render_guard.yaml`
- `configs/material_refine_eval_paper_stage1_round11_r8_boundary_render_guard.yaml`

新增安全机制：

- `max_residual_gate`：限制 residual gate 上限，控制 changed pixel rate。
- `boundary_residual_suppression_strength`：按 boundary cue 抑制边界区域更新。

Round11 设计：

- 从 Round10 best 初始化。
- 降低学习率到 `3e-5`。
- `max_residual_gate=0.42`。
- `boundary_safety_strength=0.65`。
- `boundary_residual_suppression_strength=0.65`。
- 启用 render gate：`render_gate_strength=0.35`。
- 提高 inverse check：`inverse_check_strength=0.45`。
- topology 改成轻量版：`feature_channels=8`、`layers=1`、`heads=1`。
- 强化 loss：`boundary_bleed_weight=0.22`、`gradient_preservation_weight=0.32`、`view_consistency_weight=0.42`、`residual_safety_weight=0.35`。

运行状态：

- tmux session：`sf3d_round11_r8_boundary_guard_gpu1_20260423`
- log：`output/material_refine_paper/stage1_round11_r8_boundary_render_guard/logs/train_round11_r8_boundary_render_guard_20260423T105117Z.log`
- W&B run：`https://wandb.ai/chexueyuan2027-southeastern-university/stable-fast-3d-material-refine/runs/osk8v4g8`
- 当前已进入 `epoch=1/16`。
- 预期 missing/shape mismatch：
  - render gate 是 Round11 新启用模块。
  - topology 从 16ch/2 layers 改成 8ch/1 layer。
  - 新增 residual cap/boundary suppression 没有参数，不影响加载。

### Round11 纠偏：改为 Round11b shape-compatible

Round11 初版已停止：

- session：`sf3d_round11_r8_boundary_guard_gpu1_20260423`
- 原因：为了降低 over-smoothing，把 topology 从 `16ch/2 layers/2 heads` 改成 `8ch/1 layer/1 head`，同时启用了新的 render gate。
- 结果：`refine_head.in_conv` 与 topology block 出现 shape mismatch，部分 trunk 重新初始化。
- 早期 validation 从 Round10 best 的 `0.040697` 退回到约 `0.094`，该结果不能用于方法判断。

Round11b 已启动，作为新的有效迭代入口：

- 配置：`configs/material_refine_train_paper_stage1_round11b_r8_boundary_gradient_guard_compatible.yaml`
- eval 配置：`configs/material_refine_eval_paper_stage1_round11b_r8_boundary_gradient_guard_compatible.yaml`
- output：`output/material_refine_paper/stage1_round11b_r8_boundary_gradient_guard_compatible`
- log：`output/material_refine_paper/stage1_round11b_r8_boundary_gradient_guard_compatible/logs/train_round11b_r8_boundary_gradient_guard_compatible_20260423T105603Z.log`
- W&B run：`https://wandb.ai/chexueyuan2027-southeastern-university/stable-fast-3d-material-refine/runs/4g5i808u`

Round11b 关键约束：

- 从 Round10 best 初始化。
- 保持 topology 形状为 `16ch/2 layers/2 heads`。
- 初始化日志确认：`missing=0`、`unexpected=0`、`skipped_shape=0`。
- 不启用随机初始化的 render gate，render consistency 先作为 validation/eval gate。
- 保留 inherited inverse check。
- 新增无参数安全阀：
  - `max_residual_gate=0.52`
  - `boundary_residual_suppression_strength=0.55`
- 目标：牺牲少量 UV，以换取 boundary bleed、SSIM/LPIPS、gradient preservation 回升。

Round11b 早期状态：

- `progress_001/040`：`uv_total=0.073190`，`res_change=0.4574`，`res_reg=0.1056`
- `progress_002/040`：`uv_total=0.071619`，`res_change=0.4823`，`res_reg=0.1081`
- `progress_003/040`：`uv_total=0.070159`，`res_change=0.5035`，`res_reg=0.1099`

解释：

- Round11b 比 Round10 best 更保守，UV 暂时明显变差。
- 这是 residual cap/boundary suppression 的直接代价，不是 checkpoint 兼容问题。
- 是否继续保留该方向，要等后续 proxy validation 和 full-test eval 判断 boundary/render/gradient 是否显著回升。
- 若最终 UV 无法回到 `<=0.055` 且 render/gradient 没有补偿提升，应改为中等强度配置，例如 `max_residual_gate=0.70`、`boundary_residual_suppression_strength=0.30`。

### Round11c 预置

已预置中等强度备选，不自动启动，等待 Round11b full eval 后决策：

- train config：`configs/material_refine_train_paper_stage1_round11c_r8_mid_guard_compatible.yaml`
- eval config：`configs/material_refine_eval_paper_stage1_round11c_r8_mid_guard_compatible.yaml`

Round11c 相比 Round11b：

- `max_residual_gate` 从 `0.52` 放宽到 `0.72`。
- `boundary_residual_suppression_strength` 从 `0.55` 降到 `0.28`。
- `boundary_safety_strength` 从 `0.58` 降到 `0.45`。
- `prior_confidence_gate_strength` 从 `0.68` 降到 `0.58`。
- loss 从强边界/强梯度约束改成中等约束。

启动条件：

- 若 Round11b 的 full eval 显示 boundary/LPIPS/gradient 明显回升，但 UV/view RM 不足，则启动 Round11c。
- 若 Round11b 没有带来 boundary/render/gradient 改善，则不启动 Round11c，应回退到 Round9/Round10 并重新检查 loss 与目标定义。
