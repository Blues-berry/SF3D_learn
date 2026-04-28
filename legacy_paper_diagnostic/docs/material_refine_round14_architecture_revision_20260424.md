# Round14 八模块与主干升级说明

## 目标

Round13 已经说明，仅靠更强的保守门控还不够。当前真正的瓶颈是：

1. `B/C/D` 产生了材质/边界/高光诊断信息，但这些信息没有真正进入 `G` 主干。
2. `H` 的 render consistency 过去更像后验约束，而不是对当前 provisional refined RM 的在线校正。
3. 主干 `SmallUNet` 多尺度上下文不足，容易出现 UV MAE 降低但 view/render 退化。
4. progress validation 会让早停过早触发，弱化了新模块学习空间。

## 模块级调整

### A. Dual-Path Prior Initialization

- 保留原有 with-prior / no-prior 统一结构。
- 不再额外加重此模块复杂度，避免把错误 prior 过度放大。
- 重点保持 domain-aware prior calibration 与后续 trunk 的兼容性。

### B. Material-sensitive Multi-view Encoder

- 保留材质分类、高光响应、边界响应三类 side heads。
- 新增 `material_entropy` 输出，后续直接投到 UV，避免材质不确定性只停留在 view-space。

### C. Hard-view Routing

- 继续使用 global / boundary / highlight 三路权重。
- routing 统计现在显式考虑 `learned_boundary_score + material_entropy`，弱化纯 coverage 驱动的平均化融合。

### D. Tri-branch UV Fusion

- 保留三路 UV fusion。
- 新增 `branch_confidence_uv`，让 trunk 能感知三路证据的可靠性，而不是只看到融合后的 feature。

### E. Boundary Safety Module

- 保留原有 boundary gate / safe update mask / bleed risk。
- boundary cue 现在额外融合 `view_boundary_response_uv` 和 `material_transition_uv`，减少只靠 albedo/normal 边缘的误触发。

### F. Material Topology Reasoning

- 保留 patch-token Transformer。
- 新增 `topology_residual_suppression_strength`，把 region consistency 真正接到 residual gate，而不是只作为旁路特征。
- 新增 local-detail reinjection，避免 topology 分支只会平滑。

### G. Confidence-Gated Residual Refinement Trunk

- trunk 输入新增三类 view diagnostic UV maps：
  - `view_boundary_response_uv`
  - `view_highlight_response_uv`
  - `material_transition_uv`
- 若启用 tri-branch fusion，trunk 额外接收 `branch_confidence_uv`。
- 主干 `SmallUNet` 的 bottleneck 升级为 `MultiScaleBridge2d`，在不破坏旧 checkpoint 的前提下补上多尺度上下文。

### H. Render-consistency / Inverse-check

- render gate 不再围绕 `initial RM`，而是基于 `provisional refined RM`。
- render feature 额外融合高光响应和材质转移提示，减少“UV 更好但 render 更差”的错觉提升。
- inverse check 继续在 render gate 之后执行，形成更紧的正反向闭环。

## 主干网络升级

- `ConvBlock` 增加更稳的 residual path。
- `SmallUNet` bottleneck 增加 `MultiScaleBridge2d`。
- `FeatureAdapter` / `MultiScaleBridge2d` 都使用零扰动初始化，允许从 Round13 checkpoint 热启动。
- GroupNorm 改成自动寻找可整除 group 数，避免通道数变化时直接报错。

## 训练侧同步修正

- 新增 `validation_selection_metric=uv_render_guarded` 的稳定入口。
- 新增 `topology_residual_suppression_strength` 配置项。
- 修复 progress validation 逐次累加 stale counter 的问题，默认改为 `early_stopping_scope=epoch`。
- `best/selection_metric` 与 `val/selection/*` 进入日志，避免继续把 composite selection 当成 `best/val_uv_total_mae`。

## 预期收益

1. 更强的边界感知和材质感知信息真正进入 refinement trunk。
2. residual update 会同时受 view uncertainty / bleed risk / topology consistency 约束。
3. render consistency 从“后验判断”变成“在线门控”。
4. 主干对高频细节和跨尺度上下文更友好。
5. 新结构仍可复用已有 Round13 权重，不需要从零训练。
