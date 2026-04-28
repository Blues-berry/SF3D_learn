# Material Refine Round13 迭代日志：View/Render/Boundary Guard

## 当前判断

Round12 证明了 Stage1-v3 增量数据对 UV-space RM 学习是有效的，但不能直接作为论文结论。原因是 balanced heldout 上出现了指标冲突：

- UV RM MAE 明显下降，说明模型确实学到了 target atlas。
- view-space RM MAE、PSNR、SSIM、LPIPS 和 boundary_bleed_score 出现退化，说明更新没有稳定转化到可渲染视角。
- prior_residual_safety 中 changed_pixel_rate 偏高，模型改动范围过大，尤其在低视图证据与边界风险区域容易过冲。

所以 Round13 的目标不是继续压 UV MAE，而是让 R-8 模块更稳：在证据不足、边界风险高、渲染一致性弱的区域少动，在高置信目标区域精修。

## 已落实的模型修改

新增两个残差安全门控，均兼容现有 CanonicalAssetRecordV1 和旧 checkpoint warm-start：

- `view_uncertainty_residual_suppression_strength`：使用 Tri-branch UV fusion 输出的 `view_uncertainty_uv` 抑制低视图一致性区域的 residual update。
- `bleed_risk_residual_suppression_strength`：使用 Boundary Safety Module 输出的 `bleed_risk_uv` 抑制高边界泄漏风险区域的 residual update。

这两个门控位于 residual gate 后、material/render gate 前，属于“更新幅度安全层”，不会改变 roughness/metallic 输出格式。

## Round13 配置原则

训练配置：`configs/material_refine_train_stage1_v3_round13_view_render_boundary_guard.yaml`

- 从 `output/material_refine_paper/stage1_v3_round12_balanced_data_adaptation/best.pt` 初始化。
- 学习率降到 `1.2e-5`，只做 12 epoch 的短程安全修正。
- `max_residual_gate=0.38`、`residual_gate_bias=-1.85`，避免大面积重写。
- 提高 `view_consistency_weight`、`boundary_bleed_weight`、`gradient_preservation_weight`、`residual_safety_weight`。
- 启用 `render_consistency_gate` 和 `inverse_material_check`，让 UV 改动必须通过渲染代理证据约束。
- 使用 Stage1-v3 strict paper candidates 作为训练入口，当前选择器检查为 877 条、eligible=877、identity=0、view ready=1.0；balanced manifest 保留为验证/heldout 诊断入口。
- 继续通过 `reload_manifest_every=1` 感知数据端增量，但新增数据仍必须先通过 strict paper 门禁。

## 数据快照

`round13_manifest_selection_check.json` 当前选择 strict paper candidates：

- 总量：877 records，paper eligible 877，target/prior identity 0.0，view supervision ready 1.0。
- source：3D-FUTURE 650，ABO 226，Objaverse-XL strict increment 1。
- prior：no-prior 651，with scalar RM prior 226。
- target：`gt_render_baked` 650，`pseudo_from_multiview` 227。
- 材质族：`metal_dominant` 583，`glossy_non_metal` 242，`mixed_thin_boundary` 21，`ceramic_glazed_lacquer` 20，`glass_metal` 11。

主要不足仍然是材质不均衡：glass/ceramic/mixed-boundary 还偏少，所以 Round13 使用 balanced sampler，但论文主表仍需要后续数据端补齐这些族。

## 新增 ablation

配置：`configs/material_refine_round13_view_render_boundary_guard_ablation_matrix.yaml`

用于确认真正有效的是哪一个安全门控：

- `guard_full`
- `no_view_uncertainty_gate`
- `no_bleed_risk_gate`
- `render_gate_light`
- `residual_extra_safe`

选择标准不再只看 UV：优先要求 view RM、PSNR、SSIM、LPIPS、boundary bleed 不退化。

## W&B 与报告阅读方式

Round13 的 W&B 只应该被解读为 diagnostic / method-iteration，不能提前写成 paper-stage result。重点看：

- `eval/main/uv_rm_mae`：UV 参数误差是否保持改善。
- `eval/main/view_rm_mae`：UV 改动是否能投影到视图空间。
- `eval/main/psnr`、`eval/main/ssim`、`eval/main/lpips`：渲染代理质量是否不再被 UV 优化拖累。
- `eval/special/boundary_bleed_score`：边界泄漏是否下降。
- `eval/special/prior_residual_safety`：无必要改动和 regression rate 是否下降。
- `eval/diagnostics/metric_disagreement`：如果为 1，说明 UV 与 view/render/object-level 结论冲突，不能作为论文主结论。

本地 HTML 继续保存 top improved / top regressed / top boundary failures；W&B 不上传大量 case，只上传总体指标和少量清晰 panel。

## 风险与下一步

- 如果 Round13 UV 略退但 view/render/boundary 明显改善，可以作为 Round14 的稳定初始化。
- 如果 Round13 仍出现 metric disagreement，说明仅靠 gate 还不够，需要把 render proxy 从 validation gate 提升为训练 loss。
- 如果 Stage1-v3 strict/diagnostic 数据继续增长，应优先补足 `glass_metal`、`ceramic_glazed_lacquer`、`mixed_thin_boundary`，而不是继续堆 `metal_dominant`。
