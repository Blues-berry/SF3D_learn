# Material Refine W&B 指标白名单说明

更新时间：2026-04-28

当前 W&B 策略已从“完整工程日志”改成“论文主看板白名单”。详细 case、全量 panel、dataset/source/license 分布、baseline 对照表和 top failure 明细继续保存在本地 `summary.json`、`metrics.jsonl`、HTML report 和 `dataset_state/*.json`，默认不再上传到 W&B。

## 一、论文主表指标

这些指标用于主表和跨 run 对比。

| W&B key | 含义 | 趋势 |
| --- | --- | --- |
| `eval/rm/uv_total_mae` | refined roughness/metallic 在 UV-space 的总 MAE。 | 越低越好 |
| `eval/rm/view_total_mae` | refined RM 投影到可见视角后的 view-space MAE。 | 越低越好 |
| `eval/rgb_proxy/mse/refined` | eval `proxy_uv_shading` 的 refined RGB proxy MSE。 | 越低越好 |
| `eval/rgb_proxy/psnr/refined` | eval `proxy_uv_shading` 的 refined RGB proxy PSNR。 | 越高越好 |
| `eval/rgb_proxy/ssim/refined` | eval `proxy_uv_shading` 的 refined RGB proxy SSIM。 | 越高越好 |
| `eval/rgb_proxy/lpips/refined` | eval `proxy_uv_shading` 的 refined RGB proxy LPIPS。 | 越低越好 |
| `eval/special/boundary_bleed_score` | 材质边界带误差相对 interior 的增量。 | 越低越好 |
| `eval/special/metal_confusion_rate` | metallic/non-metallic 二分类混淆率。 | 越低越好 |
| `eval/special/highlight_localization_error` | 高光区域与材质响应不一致的误差。 | 越低越好 |
| `eval/special/prior_residual_safety` | residual 更新是否集中在应该修改区域的安全分数。 | 越高越好 |

说明：SF3D Original、SF3D Canonical Prior、Prior Smoothing、Scalar Broadcast、Ours w/o View、Ours w/o Residual、Ours Full 的完整对照表保存在本地 eval summary/report；W&B 默认只记录当前 eval variant 的聚合标量，避免主看板被 baseline 表格刷屏。

## 二、Checkpoint 与验证决策

这些指标用于观察训练步数对应的验证表现。

| W&B key | 含义 | 趋势 |
| --- | --- | --- |
| `best/selection_metric` | 当前 run 内验证选择指标；默认 `gain_render_guarded`，越低代表 UV gain 更高且 render/residual penalty 更小。 | 越低越好 |
| `val/gain_total` | input prior UV total MAE 减 refined UV total MAE。 | 越高越好 |
| `val/rm_proxy/view_mae/delta` | view-projected RM proxy 中 baseline MAE 减 refined MAE。 | 越高越好 |
| `val/rm_proxy/view_mse/delta` | view-projected RM proxy 中 baseline MSE 减 refined MSE。 | 越高越好 |
| `val/rm_proxy/view_psnr/delta` | view-projected RM proxy 中 refined PSNR 减 baseline PSNR。 | 越高越好 |
| `val/improvement_uv_total_mae` | baseline UV total MAE 减 refined UV total MAE。 | 越高越好 |
| `val/improvement_rate` | 验证样本中 refined 优于 baseline 的比例。 | 越高越好 |
| `val/regression_rate` | 验证样本中 refined 差于 baseline 的比例。 | 越低越好 |
| `eval/effective_view_supervision_rate` | eval 样本具备有效 view supervision 的比例。 | 越高越可信 |
| `eval/object_level/avg_improvement_total` | object-level 聚合后的平均提升。 | 越高越好 |
| `val/case_level/regression_rate` | case-level 聚合后 refined 差于 baseline 的比例。 | 越低越好 |

训练期间还保留少量横轴和效率信息：`trainer/global_step`、`trainer/epoch`、`optim/lr`、`throughput/samples_per_second`、`throughput/seconds_per_batch`。验证里程碑会记录 `val/rm/uv_total_mae`、`val/rm_proxy/view_mae/*`、`val/rm_proxy/view_mse/*`、`val/rm_proxy/view_psnr/*`。`val/main/*` 仍作为 refined-only 兼容别名保留，但不作为推荐主看板指标。训练 loss 只保留总 loss 和少数与当前方法直接相关的关键 loss。

## 三、辅助分析

这些指标用于解释提升来源和失败模式，不作为唯一论文结论。

| W&B key | 含义 | 趋势 |
| --- | --- | --- |
| `eval/special/rm_gradient_preservation` | RM 边缘/梯度保持能力，衡量是否过平滑。 | 越高越好 |
| `eval/failure_tag_reduction/metal_nonmetal_confusion/reduction` | metal/non-metal 错误数量下降量。 | 越高越好 |
| `eval/failure_tag_reduction/boundary_bleed/reduction` | boundary bleed 错误数量下降量。 | 越高越好 |
| `eval/failure_tag_reduction/local_highlight_misread/reduction` | 局部高光误读错误数量下降量。 | 越高越好 |
| `eval/failure_tag_reduction/over_smoothing/reduction` | 过平滑错误数量下降量。 | 越高越好 |
| `eval/runtime/avg_object_seconds` | eval 平均每个 object 的耗时。 | 越低越好 |
| `eval/memory/peak_allocated_gb` | eval 峰值显存分配。 | 越低越省资源 |

## 四、默认不再上传的内容

以下内容仍在本地可审计，但默认不进入 W&B 主看板。

| 类别 | 处理方式 | 原因 |
| --- | --- | --- |
| 训练/验证 loss 全家桶 | 只保留 `train/total`、`train/refine_l1`、`train/boundary_bleed`、`train/gradient_preservation`、`train/residual_safety`、`val/loss/total`。 | 大量辅助 loss 会遮住真正的验证指标。 |
| dataset/source/license/generator 常量统计 | 写入 `dataset_state/*.json` 和 audit report。 | 这些是数据审计信息，不适合作为训练曲线。 |
| 分组指标爆炸 | 默认不传；需要时只允许 `material_family` 和 `prior_label` 两轴。 | 避免 `source_name/generator/license/view` 产生海量 charts。 |
| `validation_panels/table` 与全部 top cases | 本地 HTML/JSON 保存；W&B 只上传少量清晰 sample panel。 | 每个物体都上传会污染 run summary 且加载很慢。 |
| baseline 全量标量展开 | 本地 paper table 保存。 | 主看板只看当前方法曲线，baseline 用专门对比表展示。 |

## 五、读取建议

训练是否真的在进步，优先看 `trainer/global_step` 横轴下的 `val/gain_total`、`val/rm_proxy/view_psnr/delta`、`val/rm_proxy/view_mse/delta`、`val/rm_proxy/view_mae/delta`、`val/case_level/regression_rate`、`val/by_variant/<variant>/gain_total`。`RM proxy` 是 UV RM map 通过 `view_uvs` 投影后的 roughness/metallic 指标；`RGB proxy` 是 eval 里的 `proxy_uv_shading`；真实 renderer/Blender 重渲染指标还未接入。论文是否站得住，必须看 eval 阶段的 `UV RM MAE + View RM MAE + RGB proxy + boundary/highlight/residual`，不能只看 UV-space 单项。
