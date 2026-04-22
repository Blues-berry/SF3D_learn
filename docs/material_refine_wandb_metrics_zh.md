# Material Refine W&B 指标说明日志

更新时间：2026-04-22

本文说明当前 `stable-fast-3d-material-refine` 项目在 W&B 中保留的指标含义。当前策略是：终端保留详细训练进度，W&B 只保留主指标、专项指标和少量诊断信息；大量中间进度、重复大图和 top-case 明细默认不上传。

## 终端训练日志

这些内容主要写入 `train.log`，默认不再全部上传 W&B，目的是让远程终端和 tmux/tunx 后台日志也能像 NG 一样快速定位问题。

| 终端前缀 | 含义 | 读数建议 |
| --- | --- | --- |
| `[preflight]` | 训练前硬门禁结果，包含设备、W&B 登录、manifest target/prior identity 和 view supervision 审计。 | `status=ok` 才能继续；paper-stage 不允许绕过 identity / non-trivial target 门禁。 |
| `[run]` | run 名称、seed、输出目录和当前工作目录。 | 用于确认没有写错 output dir 或复用了错误 seed。 |
| `[system]` | Python、PyTorch、CUDA/cuDNN、CPU、系统 load、磁盘剩余空间。 | load 很高或磁盘不足时，训练慢/中断通常不是模型问题。 |
| `[device]` / `[device:gpu]` | 当前选择设备、AMP、TF32、所有 GPU 的显存占用和空闲量。 | `*cuda:N` 是本次训练设备；若 selected GPU 已被占满，应先换卡或降低 batch。 |
| `[data:train]` / `[data:val]` | train/val 样本数、split 分布。 | 确认 train/val/test 没隐式重切。 |
| `[data:*:dist]` | source、generator、material_family、with/without prior 分布。 | 用于检查是否真的多材质、多上游、多 prior 状态。 |
| `[data:*:quality]` | supervision tier/role、target_quality、license、view supervision 分布。 | 判断数据是否可进入 paper-stage。 |
| `[data:*:paths]` | 关键贴图和 buffer 路径存在性检查。 | `miss>0` 要优先排查 canonical bundle。 |
| `[data:*:probe]` | 首批 batch 读取速度、H2D 传输速度、prior/view supervision 比例、target-prior 差异、confidence 均值。 | `read_samples_s` 很低说明 dataloader/PNG 解码是瓶颈；`target_prior_delta` 接近 0 说明 target 可能过于接近 prior。 |
| `[data:*:tensor]` | 首批 batch 的 tensor shape、dtype、范围、均值和 finite rate。 | 主要排查 NaN、全 0、维度不匹配、UV/view target 缺失。 |
| `[model]` | 参数量和核心结构开关。 | 复现实验时确认 residual/view/prior 开关是否正确。 |
| `[optimizer]` | Adam/AdamW、LR、warmup、scheduler、grad clip。 | 用于确认优化器没有被 config 覆盖错。 |
| `[schedule]` | epoch、batch、optimizer step、validation 计划。 | 当前 paper-stage 推荐 `validation_progress_milestones=40`。 |
| `[loader]` | batch size、num_workers、pin memory、sampler。 | sampler 应匹配 generator/prior 或 source/prior balance 设计。 |
| `[epoch:start]` | 每个 epoch 开始时的数据规模和计划 step。 | manifest 动态增加时可看到 reload 后规模变化。 |
| `epoch N/M: ... loss=... lr=...` | tqdm 训练进度条。 | 默认只显示关键训练状态：`loss`、`lr`、`ref`、`edge`、`grad`、`v`、`sps`、`mem`。 |
| `[train]` | epoch/step/batch/global step、总进度、elapsed、ETA、LR、loss 分项、速度、显存。 | 这是主要训练仪表盘；重点看 `eta_total`、`samples_s`、`grad`、`max_mem_gb`。 |
| `[val]` | 验证触发点、baseline/refined UV MAE、improvement、roughness/metallic 分项、best。 | 只在进度里程碑或配置指定周期触发，不再每个 step/epoch 无脑验证。 |
| `[epoch]` | epoch 汇总、是否保存 checkpoint。 | 用于快速定位 best epoch 和 checkpoint。 |

如果需要恢复旧的机器 JSON 控制台输出，可在 config 或命令行设置 `terminal_json_logs: true`。如果需要恢复每个 log interval 的长 `[train]` 文本行，可设置 `train_line_logs: true`。当前 paper 配置默认 `progress_bar: true`、`train_line_logs: false`，避免终端和 W&B console 被大段 JSON/长行淹没。

## 训练基础轴

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `epoch` | 当前训练 epoch 编号。 | 只作为横轴/阶段定位，不代表模型优劣。 |
| `optimizer_step` | 优化器实际更新次数。 | 比 batch 更适合做训练曲线横轴，因为包含梯度累积。 |
| `global_batch_step` | 已读取的 batch 数。 | 用于排查 dataloader/梯度累积是否符合预期。 |
| `lr` | 当前学习率。 | warmup、plateau 降学习率是否生效主要看它。 |

## 训练损失

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `train/total` | 总训练损失，包含所有启用 loss 的加权和。 | 只能和同一配置下的 run 比较，不适合跨配置直接比较。 |
| `train/refine_l1` | refined RM UV map 与 target RM 的置信加权 L1。 | 核心收敛指标之一，下降说明预测 RM 更接近 target。 |
| `train/coarse_l1` | no-prior/coarse head 的 RM 初始化误差。 | 主要约束无先验模式和粗估计能力。 |
| `train/prior_consistency` | refined 与 prior 的一致性约束。 | 太低可能模型几乎不改 prior；太高可能过度束缚 refinement。 |
| `train/smoothness` | UV RM 平滑正则。 | 防止噪声，但过高会导致 `over_smoothing`。 |
| `train/view_consistency` | view-to-UV 监督下的多视角一致性损失。 | 只有 `view_sup_rate` 有效时才是有效监督。 |
| `train/edge_aware` | 材质边界/梯度敏感损失。 | 用于压低 `boundary_bleed_score`，避免边界被抹开。 |
| `train/boundary_bleed` | 边界膨胀带 RM 误差和边界-内部误差差值约束。 | Round8 新增，用于更直接压低 `eval/special/boundary_bleed_score`。 |
| `train/gradient_preservation` | RM 梯度保持损失。 | Round7 新增，直接约束 Pred 与 GT 的 roughness/metallic 梯度一致，目标是减少过平滑并保留薄边界。 |
| `train/metallic_classification` | metallic 二值分类辅助损失。 | 用于改善 metal/non-metal 判别。 |
| `train/residual_safety` | residual 安全约束。 | 惩罚在 target/prior 差异小或高置信区域的不必要改动。 |

## 训练诊断

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `train/prior_dropout_probability` | 当前 prior dropout 概率。 | 越往后通常越高，用来增强无/弱先验鲁棒性。 |
| `train/view_consistency_enabled` | 当前配置是否启用 view consistency loss。 | 只是配置开关，是否真正有效还要看 `train/effective_view_supervision_rate`。 |
| `train/effective_view_supervision_rate` | batch 内具备有效 view-to-UV 监督样本比例。 | 接近 1 表示 view consistency 是真监督；接近 0 时相关结论不能成立。 |
| `train/effective_view_supervision_samples` | 当前统计窗口内有效 view supervision 样本数。 | 用于排查 batch 是否覆盖有效视角监督。 |
| `train/prior_dropout_samples` | 当前统计窗口内被 dropout prior 的样本数。 | 用于确认 prior dropout curriculum 是否真的发生。 |
| `train/residual_gate_mean` | residual gate 平均值。 | 越大表示模型更愿意改动 prior；过大可能导致无谓扰动。 |
| `train/residual_delta_abs` | refined 相对 prior 的平均绝对变化。 | 与 residual safety 一起看，判断改动幅度是否合理。 |
| `train/grad_norm` | 梯度裁剪前/后的梯度范数。 | 突然暴涨通常意味着训练不稳定或数据异常。 |
| `train/samples_per_second` | 训练吞吐量。 | 用于比较配置效率，不是质量指标。 |
| `train/seconds_per_batch` | batch 平均耗时。 | 用于定位 dataloader、显存或模型计算瓶颈。 |
| `train/epoch_total` | 触发验证时对应 epoch 的训练总损失。 | 只在验证日志附近出现，用于关联 train/val。 |

## 验证指标

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `best/val_uv_total_mae` | 当前 run 最优验证 UV RM MAE。 | 越低越好，是 checkpoint 选择主指标。 |
| `best/epoch` | 最优验证指标出现的 epoch。 | 若远早于最后 epoch，说明后期可能过拟合或震荡。 |
| `val/uv_total_mae` | refined roughness + metallic 的验证 UV MAE。 | 越低越好，不能单独作为论文证据。 |
| `val/uv_roughness_mae` | refined roughness UV MAE。 | 越低越好，用于判断粗糙度是否改善。 |
| `val/uv_metallic_mae` | refined metallic UV MAE。 | 越低越好，用于判断金属度是否改善。 |
| `val/baseline_uv_total_mae` | SF3D/canonical prior 的 UV RM MAE。 | 作为 baseline，对比 refined 是否真实超过原始结果。 |
| `val/baseline_uv_roughness_mae` | baseline roughness UV MAE。 | refined roughness 必须低于它才算 roughness 改进。 |
| `val/baseline_uv_metallic_mae` | baseline metallic UV MAE。 | refined metallic 必须低于它才算 metallic 改进。 |
| `val/improvement_uv_total_mae` | baseline total MAE 减 refined total MAE。 | 大于 0 表示 refined 优于 SF3D baseline。 |
| `val/improvement_uv_roughness_mae` | roughness 方向的改进量。 | 大于 0 表示 roughness 有提升。 |
| `val/improvement_uv_metallic_mae` | metallic 方向的改进量。 | 大于 0 表示 metallic 有提升。 |
| `val/improvement_rate` | 验证样本中 refined 优于 baseline 的比例。 | 接近 1 更稳；若均值提升但比例低，说明少数样本拉高均值。 |
| `val/regression_rate` | 验证样本中 refined 差于 baseline 的比例。 | 越低越好，是 residual safety 的重要观察指标。 |
| `val/effective_view_supervision_rate` | 验证样本有效 view supervision 比例。 | 用来判断 view consistency 相关验证是否可信。 |
| `val/view_consistency_enabled` | 验证时是否启用 view consistency。 | 和上一个指标一起看。 |

## 验证损失

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `val/loss/total` | 验证总损失。 | 和训练 loss 同口径，只能同配置比较。 |
| `val/loss/refine_l1` | 验证 refined L1。 | 通常应与 `val/uv_total_mae` 同方向。 |
| `val/loss/coarse_l1` | 验证 coarse head L1。 | 用于跟踪 no-prior/coarse 初始化能力。 |
| `val/loss/prior_consistency` | 验证 prior consistency。 | 用来判断模型是否过度偏离 prior。 |
| `val/loss/smoothness` | 验证平滑正则。 | 与 `over_smoothing` 风险相关。 |
| `val/loss/view_consistency` | 验证 view consistency loss。 | 仅在 view supervision 有效时解释。 |
| `val/loss/edge_aware` | 验证边界敏感损失。 | 与边界保真相关。 |
| `val/loss/gradient_preservation` | 验证 RM 梯度保持损失。 | 与 `rm_gradient_preservation` 专项指标同方向观察。 |
| `val/loss/metallic_classification` | 验证 metallic 分类辅助损失。 | 与 metal/non-metal 指标一起看。 |
| `val/loss/residual_safety` | 验证 residual 安全损失。 | 与 regression/unnecessary change 一起看。 |
| `val/loss/residual_gate_mean` | 验证 residual gate 均值。 | 高值说明模型更激进地改 prior。 |
| `val/loss/residual_delta_abs` | 验证 refined-prior 平均改变量。 | 辅助判断是否出现过度精修。 |

## 数据覆盖与验证集均衡

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `dataset/train_records` | 当前训练记录数。 | 用于确认 manifest/filter 是否符合预期。 |
| `dataset/val_records` | 当前验证记录数。 | 当前 Round6 为 40。 |
| `dataset/train_batches` | 每个 epoch 训练 batch 数。 | 与 batch size 和 sampler 有关。 |
| `dataset/val_batches` | 每次验证 batch 数。 | Round6 为 10。 |
| `dataset/val_balance_group_count` | 验证集中 balance key 的实际组数。 | 当前为 1，说明只有一个 material family。 |
| `dataset/val_balance_warning_count` | 验证均衡警告数量。 | 大于 0 表示当前验证集无法满足预设均衡目标。 |
| `dataset/val_material_family/<family>/count` | 每个材质族的验证样本数。 | 当前只有 `glossy_non_metal/count=40`，不能支撑多材质论文结论。 |

## 显存与效率

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `memory/gpu_allocated_gb` | 当前已分配显存。 | 用于排查显存波动。 |
| `memory/gpu_reserved_gb` | PyTorch cache 预留显存。 | 通常高于 allocated。 |
| `memory/gpu_max_allocated_gb` | run 到目前为止峰值分配显存。 | 用于规划 batch size 和 GPU 资源。 |

## 评测主指标

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `eval/baseline_total_mae` | baseline UV RM 总 MAE。 | SF3D/canonical prior 的整体材料误差。 |
| `eval/refined_total_mae` | refined UV RM 总 MAE。 | 应低于 baseline。 |
| `eval/avg_improvement_total` | baseline 减 refined 的 UV RM MAE 改进。 | 大于 0 表示整体提升。 |
| `eval/main/psnr` | refined proxy render PSNR。 | 越高越好，衡量渲染像素误差。 |
| `eval/main/baseline_psnr` | baseline proxy render PSNR。 | 用作 SF3D 对照。 |
| `eval/main/ssim` | refined proxy render SSIM。 | 越高越好，衡量结构相似度。 |
| `eval/main/baseline_ssim` | baseline proxy render SSIM。 | 用作 SF3D 对照。 |
| `eval/main/lpips` | refined proxy render LPIPS。 | 越低越好，衡量感知距离。 |
| `eval/main/baseline_lpips` | baseline proxy render LPIPS。 | 用作 SF3D 对照。 |
| `eval/rm/uv_total_mae` | refined UV-space RM MAE。 | 越低越好。 |
| `eval/rm/baseline_uv_total_mae` | baseline UV-space RM MAE。 | 用作 SF3D 对照。 |
| `eval/rm/view_total_mae` | refined view-space RM MAE。 | 越低越好，比 UV MAE 更接近可见视角表现。 |
| `eval/rm/baseline_view_total_mae` | baseline view-space RM MAE。 | 用作 SF3D 对照。 |

## 评测专项材质指标

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `eval/special/boundary_bleed_score` | refined 边界带误差相对 interior 的增量。 | 越低越好；升高说明材质边界可能被抹开。 |
| `eval/special/baseline_boundary_bleed_score` | baseline 边界 bleed 分数。 | 用作 SF3D 对照。 |
| `eval/special/metal_confusion_rate` | refined metal/non-metal 混淆率。 | 越低越好。 |
| `eval/special/baseline_metal_confusion_rate` | baseline metal/non-metal 混淆率。 | 用作 SF3D 对照。 |
| `eval/special/highlight_localization_error` | refined 高光定位误差。 | 越低越好，用于判断是否误把亮斑当材质属性。 |
| `eval/special/baseline_highlight_localization_error` | baseline 高光定位误差。 | 用作 SF3D 对照。 |
| `eval/special/rm_gradient_preservation` | refined RM 梯度保持度。 | 越高越好，避免过平滑。 |
| `eval/special/baseline_rm_gradient_preservation` | baseline RM 梯度保持度。 | 用作 SF3D 对照。 |
| `eval/special/prior_residual_safety` | residual 安全分数。 | 越高越好，表示改动更集中在应该改的区域。 |
| `eval/special/prior_residual_regression_rate` | residual 导致退化的比例。 | 越低越好。 |

## 评测 metal/non-metal 指标

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `eval/metal_nonmetal/baseline_f1` | baseline metallic 二分类 F1。 | 类别单一时可能虚高，必须结合 AUROC 和 warning。 |
| `eval/metal_nonmetal/refined_f1` | refined metallic 二分类 F1。 | 越高越好，但不能单独解释。 |
| `eval/metal_nonmetal/baseline_auroc` | baseline metallic AUROC。 | 越高越好；若 F1=1 但 AUROC=0，会触发退化/单类 warning。 |
| `eval/metal_nonmetal/refined_auroc` | refined metallic AUROC。 | 越高越好。 |

## 评测诊断与可靠性

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `eval/effective_view_supervision_rate` | eval 样本有效 view supervision 比例。 | 判断 view-space 指标可信度。 |
| `eval/object_level/refined_total_mae` | object-level 聚合后的 refined MAE。 | 防止 view 重复计数掩盖对象级退化。 |
| `eval/object_level/avg_improvement_total` | object-level 平均改进。 | 若与 view/UV 方向冲突，不能下论文结论。 |
| `eval/diagnostics/metric_disagreement` | UV/view/object 指标是否方向冲突。 | 1 表示存在冲突，需要人工分析。 |
| `eval/diagnostics/warning_count` | eval warning 数量。 | 越少越好；高值通常说明数据覆盖或指标可解释性不足。 |
| `eval/failure_tag_reduction/<tag>/baseline` | baseline 某 failure tag 数量。 | 本地 diagnosis 的聚合入口。 |
| `eval/failure_tag_reduction/<tag>/refined` | refined 某 failure tag 数量。 | 应低于 baseline。 |
| `eval/failure_tag_reduction/<tag>/reduction` | failure tag 减少量。 | 大于 0 表示该类错误减少。 |
| `eval/failure_tag_reduction/<tag>/relative_reduction` | failure tag 相对减少比例。 | 样本数足够时更适合论文表述。 |

## 评测 runtime 与显存

| W&B key | 含义 | 读数建议 |
| --- | --- | --- |
| `eval/runtime/avg_batch_seconds` | eval 平均 batch 耗时。 | 越低越好。 |
| `eval/runtime/avg_object_seconds` | eval 平均每对象耗时。 | 论文 runtime 表建议使用它。 |
| `eval/runtime/seconds_per_batch` | 同 `avg_batch_seconds`，保留清晰命名。 | 用于 chart。 |
| `eval/runtime/seconds_per_object` | 同 `avg_object_seconds`，保留清晰命名。 | 用于 chart。 |
| `eval/runtime/ms_per_object` | 每对象毫秒耗时。 | 更直观，适合报告。 |
| `eval/runtime/objects_per_second` | 每秒处理对象数。 | 越高越好。 |
| `eval/memory/peak_allocated_gb` | eval 峰值分配显存。 | 用于资源报告。 |
| `eval/memory/peak_reserved_gb` | eval 峰值预留显存。 | 用于资源报告。 |

## 评测分组指标

| W&B key 模式 | 含义 | 读数建议 |
| --- | --- | --- |
| `eval/group/<axis>/group_count` | 某个分组轴有多少组。 | 例如 material_family 有几类。 |
| `eval/group/<axis>/<group>/count` | 该组样本/视图数量。 | 小于 16 的组只建议 diagnostic-only。 |
| `eval/group/<axis>/<group>/baseline_total_mae` | 该组 baseline MAE。 | 分组对照。 |
| `eval/group/<axis>/<group>/refined_total_mae` | 该组 refined MAE。 | 应低于 baseline。 |
| `eval/group/<axis>/<group>/improvement_total` | 该组改进量。 | 大于 0 表示提升。 |
| `eval/group/<axis>/<group>/baseline_psnr` | 该组 baseline PSNR。 | 渲染质量对照。 |
| `eval/group/<axis>/<group>/refined_psnr` | 该组 refined PSNR。 | 应高于 baseline。 |
| `eval/group/<axis>/<group>/baseline_ssim` | 该组 baseline SSIM。 | 渲染结构对照。 |
| `eval/group/<axis>/<group>/refined_ssim` | 该组 refined SSIM。 | 应高于 baseline。 |
| `eval/group/<axis>/<group>/baseline_lpips` | 该组 baseline LPIPS。 | 感知质量对照。 |
| `eval/group/<axis>/<group>/refined_lpips` | 该组 refined LPIPS。 | 应低于 baseline。 |
| `eval/group/<axis>/<group>/baseline_boundary_bleed_score` | 该组 baseline 边界 bleed。 | 对照项。 |
| `eval/group/<axis>/<group>/refined_boundary_bleed_score` | 该组 refined 边界 bleed。 | 应低于 baseline。 |
| `eval/group/<axis>/<group>/baseline_metal_confusion_rate` | 该组 baseline 金属混淆率。 | 对照项。 |
| `eval/group/<axis>/<group>/refined_metal_confusion_rate` | 该组 refined 金属混淆率。 | 应低于 baseline。 |
| `eval/group/<axis>/<group>/prior_residual_safety_score` | 该组 residual safety。 | 越高越好。 |
| `eval/group/<axis>/<group>/prior_residual_regression_rate` | 该组 residual regression。 | 越低越好。 |

当前 compact group axes 包括：`material_family`、`paper_split`、`target_quality_tier`、`generator_id`、`source_name`、`prior_label`。

## 图片与 artifact

| W&B key / artifact | 含义 | 读数建议 |
| --- | --- | --- |
| `val/comparison_panels` | 小块 validation 对比图。包含 SF3D、GT、Pred、Error，并展示 roughness、metallic、normal、albedo、confidence。 | 用于快速看模型是否真的改善材质图，不再上传大 grid。 |
| checkpoint artifact | best/final checkpoint 与训练状态。 | 用于复现实验。 |
| eval artifact | `summary.json`、HTML report、metric disagreement report、diagnostic cases 路径。 | top cases 明细默认本地保存，不再上传 W&B 大表。 |

## 已下线或默认不上传的旧内容

| 旧内容 | 当前处理 | 原因 |
| --- | --- | --- |
| `progress/*` | 只在终端和 `train.log` 保留，不上传 W&B。 | W&B charts 噪声太大。 |
| `val/preview_grid` | 默认不上传。 | 单张大合成图看不清。 |
| `val/preview_00` 到 `val/preview_XX` | 默认不上传。 | 造成重复图片和无意义 chart。 |
| `eval/top_cases` | 默认不上传，只本地保存 `diagnostic_cases.json`。 | 表格太重，W&B 只保留总体均值和诊断聚合。 |
| 完整 `dataset/*` 展开 | 改为 compact dataset 指标。 | 避免 Charts 面板被常量统计污染。 |
| eval 完整 `summary.json` 控制台打印 | 默认不打印，只输出 compact eval summary，并把完整 JSON 存本地。 | 避免 W&B console 出现几万行日志。 |

## 当前 Round6 已知限制

- 当前 Stage-1 manifest 只有 `glossy_non_metal`，因此 `dataset/val_balance_group_count=1` 是真实限制，不是脚本错误。
- 当前结论只能说明 `ABO_locked_core + with_prior + glossy_non_metal` 子集上的表现，不能扩展成多材质、多上游生成器论文结论。
- 需要等待数据端补齐 `metal_dominant`、`ceramic_glossy`、`glass_metal`、`mixed_thin_boundary` 等材质族后，再使用同一套 W&B 分组指标输出 paper-stage 主结论。
