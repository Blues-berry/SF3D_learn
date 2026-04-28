# Material Refine R-8 Module Design

本文档记录当前 `SF3D + UV RM Refiner` 的新一轮方法结构。实现目标不是替换 `G -> C -> R` 协议，而是在 `R` 内部把原先轻量 residual refiner 升级成边界感知、材质感知、渲染一致性感知的兼容式模块。

## 兼容原则

- 默认关闭所有新增 A-H 模块，旧 Round6/7/8/9 配置和 checkpoint 仍按原结构加载。
- 新模块只消费 `CanonicalAssetRecordV1` 已经提供或可推导的字段：UV albedo/normal/prior RM/confidence、多视图 `RGB+mask+depth+normal+position+uv`、`generator_id`、`source_name`。
- 上游生成器不限定 SF3D。`generator_id/source_name` 通过稳定 hash bucket 转成 embedding，因此 SPAR3D、Hunyuan3D、Objaverse/3D-FUTURE 数据都能接入同一接口。
- 新模块通过 YAML 显式启用，便于 ablation、回滚和论文实验复现。

## A. Dual-Path Prior Initialization

配置：

```yaml
enable_dual_path_prior_init: true
enable_domain_prior_calibration: true
domain_feature_channels: 8
prior_feature_channels: 16
```

作用：

- 有 prior 时走 `Prior-adaptation path`，预测 per-texel prior reliability。
- 无 prior 或 prior dropout 后走 `Prior-bootstrap path`，从 `uv_albedo + uv_normal + mask + domain embedding` 预测 coarse RM。
- 输出 `rm_init_uv`、`init_confidence_uv`、`prior_feat_uv`、`domain_feat_uv`。

论文点：`Generator-/Source-aware Prior Calibration`，显式建模不同上游和不同数据域的 prior 可靠性差异。

## B. Material-Sensitive Multi-View Encoder

配置：

```yaml
enable_material_sensitive_view_encoder: true
```

作用：

- 对每个 view 编码 `RGB+mask+depth+normal+position`。
- 从 RGB 动态估计 local highlight cue，从 mask 提取 silhouette/boundary cue，从 normal 估计 grazing cue。
- 输出 `view_feat_k`、`material_logit_k`、`highlight_response_k`、`view_quality_k`。

论文点：不是普通多视图 encoder，而是把 metal/non-metal、highlight misread、边界线索作为显式中间监督/诊断对象。

## C. Hard-View Routing

配置：

```yaml
enable_hard_view_routing: true
```

作用：

- 根据 view feature、mask coverage、view quality、highlight、grazing/boundary cue 预测三类权重。
- 输出 `global_weight_k`、`boundary_weight_k`、`highlight_weight_k`。

论文点：`Region-specific Hard-view Routing`，普通区域、边界区域、高光区域不共享同一套视图权重。

## D. Tri-Branch UV Fusion

配置：

```yaml
enable_tri_branch_fusion: true
```

作用：

- 使用 C 的三组路由权重分别从 view-space scatter/fuse 到 UV atlas。
- 输出 `global_uv_feat`、`boundary_uv_feat`、`highlight_uv_feat`、`view_coverage_uv`、`view_uncertainty_uv`。

论文点：`Tri-branch UV Evidence Fusion`，把多视图证据拆成全局、边界、高光三条信息流。

## E. Mask-Aware Boundary Safety Module

配置：

```yaml
enable_boundary_safety_module: true
boundary_safety_strength: 0.35
```

作用：

- 从 `uv_mask`、boundary band、distance-to-boundary、boundary UV feature、normal/albedo/RM init 预测边界风险。
- 输出 `boundary_gate_uv`、`safe_update_mask_uv`、`bleed_risk_uv`。
- 主干 residual update 会被边界安全门控抑制，减少 thin-boundary mixed-material bleed。

论文点：边界不是只靠 loss 惩罚，而是作为结构化安全模块参与更新决策。

## F. Material Topology Reasoning

配置：

```yaml
enable_material_topology_reasoning: true
topology_feature_channels: 16
topology_patch_size: 16
topology_layers: 2
topology_heads: 2
```

作用：

- 将 UV atlas 切成 patch tokens。
- 使用轻量 Transformer 做 region-level material consistency reasoning。
- 输出 `topology_feat_uv`、`material_region_logits`、`region_consistency_score`。

论文点：`Material Topology Reasoning`，从 texel-level refinement 升级到 region-level 材质一致性建模。

## G. Confidence-Gated Residual Refinement Trunk

配置：

```yaml
enable_confidence_gated_trunk: true
uncertainty_gate_strength: 0.25
```

作用：

- 主干输入拼接 A-F 的全部 UV/domain/view/boundary/topology 信息。
- 输出 `delta_roughness`、`delta_metallic`、`change_gate`、`uncertainty_uv`、`material_type_logits_uv`、`boundary_stability_uv`。
- 更新形式为 `RM_refined = RM_init + change_gate * delta_RM`，同时受 prior confidence、boundary safety、uncertainty、inverse check 约束。

论文点：不是直接预测 RM，而是带置信度和安全约束的 residual refinement。

## H. Render-Consistency and Inverse-Check Head

配置：

```yaml
enable_inverse_material_check: true
inverse_check_strength: 0.25
```

作用：

- 当前实现为轻量 render-response proxy head，使用 fused view feature、highlight feature、view coverage、RM proxy 和 boundary cue 估计 inverse material consistency。
- 输出 `render_support_gate`、`inverse_material_gate_uv`、`inverse_highlight_alignment_uv`、`inverse_material_response_uv`、`inverse_inconsistency_uv`。
- 训练中用 gate 抑制 view evidence 不支持或材质响应反常的 residual update。

论文点：`Inverse Material-response Consistency Check`，将“UV 指标下降但 render 退化”的问题前移到方法结构。

## 当前落地文件

- 模型实现：`sf3d/material_refine/model.py`
- 训练参数接入：`scripts/train_material_refiner.py`
- Full R 配置：`configs/material_refine_train_paper_stage1_round10_r8_full.yaml`

## 建议实验顺序

1. 先用 Round9 conservative checkpoint 完成 full eval，确认边界和 render proxy 是否优于 Round7/8。
2. 使用 `material_refine_train_paper_stage1_round10_r8_full.yaml` 做 1 epoch smoke/preflight，不直接覆盖 paper-stage 主结论。
3. 跑模块 ablation：`A+B+D+G`、`+E`、`+F`、`+H`、`Full`。
4. 只有当 `UV RM MAE`、`boundary_bleed_score`、`PSNR/SSIM/LPIPS` 和 `prior_residual_safety` 不冲突时，再把 Full R 写成 paper-stage result。

## 风险点

- Stage1 当前仍以 `glossy_non_metal` 为主，多材质结论需要等 Stage1-v2 或 diagnostic/OOD 数据补齐。
- `TriBranchUVFusion` 仍复用现有 scatter 烘焙逻辑，视图权重参与区域路由，但 view-to-UV scatter 本身不是完全可微的高性能 CUDA kernel。
- H 模块当前是 render proxy/inverse response gate，不是完整 differentiable renderer；真正 paper-stage contribution 需要在 eval 中用 PSNR/SSIM/LPIPS 和 view-space RM 验证。
