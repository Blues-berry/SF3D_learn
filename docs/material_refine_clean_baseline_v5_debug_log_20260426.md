# Material Refine Clean Baseline v5 Debug Log

日期：2026-04-26

## 当前干净运行

- W&B project：`stable-fast-3d-material-refine-paper-v1-clean`
- W&B run：`paper-v1-baseline-clean-v5-gpu1`
- W&B run id：`i7pp9ju9`
- 本地输出：`output/material_refine_paper/paper_v1_baseline_clean_v5_20260426`
- tmux：`sf3d_paper_v1_baseline_clean_v5_gpu1`

## 已清理的坏运行

- `4ynwkaww`：v2，residual head 冷启动几乎不动，已从 W&B 删除。
- `nl2ntwur`：v3，仍存在数值/梯度问题，已从 W&B 删除。
- `101ysdzq`：v5 首次尝试，暴露 NaN 训练问题，已从 W&B 删除。

## 本轮修复

- 冷启动输出不再全零：bootstrap RM head 和 confidence-gated residual delta head 使用极小非零初始化，避免 Pred 长时间为常值图。
- BF16 不再启用 GradScaler：`gradient_scaler_enabled=False`，避免 scaler 掩盖非有限梯度和 optimizer step 是否有效的问题。
- 修复三分支 view uncertainty 的无穷梯度：`sqrt(var)` 改为 `sqrt(clamp(var, 1e-6))`。
- 加入非有限 loss/grad 硬门禁：一旦出现 NaN/Inf，训练直接失败，不继续污染 W&B。
- preflight audit 改为 fast audit：训练入口优先使用 manifest 已审计字段，避免每次训练启动都重复读取/哈希贴图。
- 启动日志增强：`[bootstrap]`、`[startup]`、`[preflight:audit:start/done]` 会明确显示当前阶段。
- CPU 线程上限：`torch_num_threads=4`、`torch_num_interop_threads=2`，同时限制 OMP/MKL/OpenBLAS/NumExpr，减少和 GPU0 数据脚本抢资源。

## 首个验证结果

- `eval/rm/uv_total_mae`：`0.349313`
- baseline UV total MAE：`0.213853`
- `eval/rm/view_total_mae`：`0.209948`
- baseline view RM MAE：`0.195196`
- `eval/main/psnr`：`11.165720`
- `eval/main/ssim`：`0.296473`
- `eval/main/lpips`：`0.150405`
- `eval/special/boundary_bleed_score`：`0.021562`
- `eval/special/metal_confusion_rate`：`0.214646`
- `eval/special/highlight_localization_error`：`0.219337`
- `eval/special/rm_gradient_preservation`：`0.325784`
- `eval/special/prior_residual_safety`：`0.306294`
- `eval/improvement_rate`：`0.208333`
- `eval/regression_rate`：`0.791667`

说明：这是 cold-start 第一个 milestone，不是方法有效性结论。当前结果仍弱于 SF3D/canonical prior baseline，但训练链路已经恢复为可诊断状态。

## W&B 读取方式

主质量：

- `eval/rm/uv_total_mae` 越低越好。
- `eval/rm/view_total_mae` 越低越好。
- `eval/main/psnr` 越高越好。
- `eval/main/ssim` 越高越好。
- `eval/main/lpips` 越低越好。

专项问题：

- `eval/special/boundary_bleed_score` 越低越好。
- `eval/special/metal_confusion_rate` 越低越好。
- `eval/special/highlight_localization_error` 越低越好。
- `eval/special/rm_gradient_preservation` 越高越好。

安全稳定：

- `eval/special/prior_residual_safety` 越高越好。
- `eval/improvement_rate` 越高越好。
- `eval/regression_rate` 越低越好。
- `eval/object_level/avg_improvement_total` 越高越好。

## 后续观察点

- 如果 milestone 5-8 后仍长期弱于 baseline，需要优先调结构或初始化策略，而不是先改学习率。
- 重点观察 `prior_residual_safety` 和 `regression_rate`，避免模型通过大面积扰动换取局部 UV 改善。
- 当前训练集仍有 `canonical_buffer_root` 缺失记录，effective view supervision 约 `0.6168`，后续数据增量应优先补 view buffers。
