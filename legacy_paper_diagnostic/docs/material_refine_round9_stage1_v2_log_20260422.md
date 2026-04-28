# Material Refine Round9 / Stage1-v2 更新日志 2026-04-22

## 当前状态

- Round8 已完成 full test eval 和 promotion decision，但没有通过 Round9 初始化门禁，因此 Round9 full training 仍不能直接启动。
- 当前主训练数据仍保持 346 条 locked Stage1 subset，不自动切换到 longrun/paper_unlock。
- 数据侧仍在持续更新；新增多样本优先进入 diagnostic/OOD，不绕过 paper-stage 门禁。

## Round8 收尾门禁

新增脚本：

`scripts/decide_material_refine_round8_promotion.py`

该脚本读取 Round6、Round7、Round8 的 `summary.json`，输出：

- `promote_to_round9_init`
- `recommended_checkpoint`
- 每个 promotion gate 的通过/失败原因

默认门禁：

- Round8 UV RM MAE 需优于或接近 Round7。
- Round8 boundary bleed 需明显低于 Round7。
- Round8 proxy PSNR/SSIM 不允许明显退化。
- Round8 LPIPS 不允许退化。
- Round8 residual safety 不低于 Round6。

当前已完成决策：

`output/material_refine_paper/round8_promotion_decision_20260422/round8_promotion_decision.json`

结论：

- `promote_to_round9_init=false`
- `recommended_checkpoint=output/material_refine_paper/stage1_round7_gradient_guard/best.pt`
- Round8 没有通过 UV close、boundary better、SSIM not degraded、safety not below Round6 四项门禁。

关键数值：

- Round7 UV refined `0.054585`，Round8 UV refined `0.058257`。
- Round7 boundary refined `0.022807`，Round8 boundary refined `0.023946`。
- Round8 PSNR delta `+0.283971`，LPIPS delta `+0.004322`，但 SSIM delta `-0.004843`。
- Round6 safety `0.483779`，Round8 safety `0.477805`。

因此 Round8 只作为方法探索结果保留，不作为 Round9 初始化 checkpoint，也不能写成 paper-stage improvement。

## Round9 方法接口

训练脚本新增：

- `train_balance_mode=material_x_source_x_prior`
- `train_balance_mode=material_x_generator_x_prior`
- `init_from_checkpoint`
- `render_proxy_validation_milestone_interval`
- `render_proxy_validation_max_batches`
- validation 端 `render_proxy_validation`
- validation 端 `residual_gate_diagnostics`

这些改动用于让 Round9 不再只看 UV MAE：

- 每 10 个 progress milestone 可计算一次轻量 view-projected RM proxy。
- 每次 validation 都会记录 residual gate 的改动率、不必要改动率、退化率、安全提升率。
- per-case residual 诊断只保存在本地 validation JSON，不向 W&B 上传大量 case。
- `init_from_checkpoint` 只加载模型权重，不恢复旧 optimizer、scheduler、epoch 和 step；用于 Round9 从 Round7 稳定权重重新开始，而不是误把 Round7 当作 resume 继续训练。

Round9 草案配置：

`configs/material_refine_train_paper_stage1_round9_boundary_render_gate.yaml`

默认初始化：

`init_from_checkpoint=output/material_refine_paper/stage1_round7_gradient_guard/best.pt`

Round9 boundary ablation matrix：

`configs/material_refine_round9_boundary_ablation_matrix.yaml`

Round9 boundary ablation launcher：

`scripts/run_material_refine_round9_boundary_ablation.py`

默认矩阵覆盖：

- `boundary_bleed_weight`: `0.05 / 0.10 / 0.20`
- `boundary_band_kernel`: `5 / 7 / 9`

launcher 设计：

- 每个 variant 会生成独立 `resolved_round9_ablation_config.yaml`。
- 支持 `--dry-run true`、`--only`、`--max-variants`、`--device`、`--cuda-device-index`、`--report-to`、`--wandb-mode`。
- 默认顺序执行，避免多个 ablation 同时抢 GPU。
- 失败默认停止；如需继续收集失败信息，可加 `--continue-on-error true`。

## R-v2 模块升级

本轮已把原来的轻量 UV refiner 扩展成可选的边界感知、材质感知、渲染一致性感知 R 模块。默认配置保持关闭，因此 Round6/Round7/Round8 checkpoint 仍可按旧结构严格加载；Round9/R-v2 配置显式开启新分支。

改动文件：

- `sf3d/material_refine/model.py`
- `scripts/train_material_refiner.py`
- `configs/material_refine_train_paper_stage1_round9_boundary_render_gate.yaml`
- `configs/material_refine_round9_rv2_component_ablation_matrix.yaml`

模块设计：

- 边界感知：从 UV albedo、normal、prior RM 和 view-fusion validity 中提取 differentiable boundary cues，并用 `boundary_gate_head` 调制 residual gate，目标是减少 mixed-material boundary 被抹开。
- 材质感知：新增 `material_context_head`，从 UV 输入、initial RM、boundary cue、prior confidence 推断粗材质上下文，并输出小幅 `material_delta`；训练侧新增 `material_context_loss`，用 target RM 自动构造 rough dielectric / glossy dielectric / metal 三类弱监督。
- 渲染一致性感知：新增 `render_gate_head`，使用 fused multi-view features、validity、initial RM 和 boundary cue 预测 render-support gate；无可靠 view 支持的位置会抑制 residual，避免只在 UV 上变好但 view/render 退化。
- 兼容性：`init_from_checkpoint` 改成 compatible loader，会跳过 shape mismatch 和新增层，适合从 Round7 稳定权重初始化 R-v2；普通 resume 仍保持严格加载，避免误恢复到不一致结构。

新增配置：

- `enable_boundary_context`
- `boundary_context_strength`
- `enable_material_context`
- `material_context_classes`
- `material_delta_scale`
- `enable_render_consistency_gate`
- `render_gate_strength`
- `material_context_weight`

新增 R-v2 组件消融矩阵：

`configs/material_refine_round9_rv2_component_ablation_matrix.yaml`

覆盖：

- `rv2_full`
- `rv2_no_boundary_context`
- `rv2_no_material_context`
- `rv2_no_render_gate`
- `rv2_safe_low_delta`

该矩阵复用 `scripts/run_material_refine_round9_boundary_ablation.py`，用于在 full training 前隔离三类新分支的贡献。

## Stage1-v2 数据更新接口

新增脚本：

`scripts/build_material_refine_stage1_v2_subsets.py`

输出三类 manifest：

- `stage1_v2_strict_paper_manifest.json`
- `stage1_v2_diverse_diagnostic_manifest.json`
- `stage1_v2_ood_eval_manifest.json`

三类 subset 的用途严格区分：

- strict paper：只用于 paper-stage 训练候选。
- diverse diagnostic：只用于诊断、可视化、loader/eval smoke。
- OOD eval：只用于跨 source/generator 的 eval-only 检查。

当前默认建议：

- strict paper 只有在 eligible 数超过当前 346 且材质族覆盖超过 1 类时，才允许替换当前主训练 manifest。
- smoke_only / auxiliary_upgrade_queue 不自动提升为 paper-stage。

本轮 Stage1-v2 最新生成结果：

- unique records：`1240`
- strict paper：`362`，全部来自 `ABO_locked_core`，全部为 `glossy_non_metal`，全部 `with_prior`，因此不替换当前 346 主训练入口。
- diverse diagnostic：`512`，其中 `metal_dominant=420`、`glossy_non_metal=48`、`ceramic_glazed_lacquer=44`，仅用于 diagnostic-only / eval-only。
- OOD eval：`256`，其中 `metal_dominant=224`、`ceramic_glazed_lacquer=32`，仅作 OOD eval。
- 当前 blocker：strict subset 虽然数量超过 346，但材质族覆盖仍为 1，不能作为更强 paper-stage 主训练数据。
- selection policy：diagnostic 每个材质族保底 `48`，OOD 每个材质族保底 `32`，剩余配额优先给 3D-FUTURE metal-dominant。

## W&B 和报告策略

- W&B 继续只记录 aggregate 指标、少量 comparison panels、runtime 和关键专项指标。
- heavy top cases 保存在本地 HTML/JSON。
- 新一轮报告必须显式标注 `paper-stage` / `diagnostic-only` / `engineering smoke`。

## 下一步

1. 先跑 Round9 boundary ablation smoke，使用 Round7 model-only init。
2. 不启动 Round9 full training，直到 boundary ablation 同时改善 UV、boundary、render-proxy 和 residual safety。
3. Stage1-v2 strict 继续等待多材质 paper-eligible 数据补齐；当前 `glossy_non_metal` 单族 strict 只能作为 paper-stage ABO 主线的增量候选。
4. Stage1-v2 diagnostic/OOD 可以跑 eval-only，用来暴露跨 source、no-prior、metal_dominant 的失败模式，但不能进入论文主表。

## 2026-04-22 验证结果

### Round9 init/render-proxy smoke

命令输出目录：

`output/material_refine_paper/debug_round9_init_render_proxy_smoke`

验收结果：

- `init_from_checkpoint` 成功从 Round7 `step_000462.pt` 加载模型，`missing=0`，`unexpected=0`。
- 新实验从 `epoch 1/1`、`optimizer_step=0` 开始，不继承旧训练状态。
- progress validation 成功触发 `progress_001_of_002` 和 `progress_002_of_002`。
- validation JSON 中写入 `render_proxy_validation.enabled=true` 和 `residual_gate_diagnostics`。

关键观察：

- 小样本 UV refined 约 `0.0448`，相对 baseline `0.09327` 明显改善。
- 同时 `proxy_view_delta` 约 `-0.009`，说明 view-proxy 退化；这证明 Round9 不能再只按 UV MAE 选模型。
- residual changed pixel rate 约 `0.565`，regression rate 约 `0.215`，需要在 ablation 里继续压低不安全改动。

### Stage1-v2 diagnostic-only eval smoke

命令输出目录：

`output/material_refine_paper/stage1_v2_dataset_20260422/diagnostic_eval_smoke_cpu_round7`

验收结果：

- 3D-FUTURE / `metal_dominant` / no-prior diagnostic records 能被 eval loader 正常读取。
- `summary.json` 正常生成。
- 该结果明确标记为 diagnostic-only，不进入 paper-stage 主结论。

关键观察：

- 2 个对象 smoke 上 baseline UV total MAE `0.2500`，refined UV total MAE `1.3296`，明显退化。
- view total MAE delta `-0.9754`，proxy PSNR delta `-0.2518`，proxy SSIM delta `-0.0050`。
- refined metal confusion rate 为 `1.0`，说明当前 ABO with-prior 模型迁移到 no-prior metal_dominant 诊断集会失败。

结论：

- diagnostic-only 数据通路可用。
- 当前模型没有 no-prior metal_dominant 泛化能力；后续需要 no-prior init 分支、metal-aware sampler 或第二上游/材质族训练数据补齐后再判断。

### Stage1-v2 balanced diagnostic eval smoke

命令输出目录：

`output/material_refine_paper/stage1_v2_dataset_20260422/diagnostic_eval_smoke_cpu_balanced_round7`

验收结果：

- balanced diagnostic manifest 能让新增 `ceramic_glazed_lacquer` no-prior smoke records 被 eval loader 读取。
- `summary.json` 正常生成，仍标记为 diagnostic-only。

关键观察：

- 2 个对象 smoke 上 baseline UV total MAE `0.4092`，refined UV total MAE `1.0631`，明显退化。
- view total MAE delta `-1.1576`，proxy PSNR delta `-0.9763`，proxy SSIM delta `-0.0144`。
- refined metal confusion rate 仍为 `1.0`。

结论：

- 新增 ceramic diagnostic 数据通路可用，但当前 ABO with-prior 模型在 no-prior scarce material 上退化明显。
- 这些样本适合作为 Round9/Stage2 的 failure discovery，不适合直接混入 paper-stage 主训练。

### Round9 ablation launcher dry-run

命令输出目录：

`output/material_refine_paper/debug_round9_ablation_launcher_dryrun`

验收结果：

- `scripts/run_material_refine_round9_boundary_ablation.py` 通过 `py_compile`。
- `--dry-run true --max-variants 2` 成功解析矩阵并生成 launcher summary。
- 解析到的前两个 variant 为 `boundary_w005_k5`、`boundary_w005_k7`。

正式小规模 ablation 示例：

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/run_material_refine_round9_boundary_ablation.py \
  --cuda-device-index 1 \
  --report-to wandb \
  --wandb-mode online
```

只跑单个 variant 示例：

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/run_material_refine_round9_boundary_ablation.py \
  --only boundary_w005_k7 \
  --cuda-device-index 1 \
  --report-to wandb \
  --wandb-mode online
```

### Round9 boundary ablation smoke 已启动

tmux session：

`sf3d_round9_boundary_ablation_smoke_gpu1_20260422`

输出根目录：

`output/material_refine_paper/round9_boundary_ablation`

启动方式：

- 9 个 variant 顺序执行，避免抢 GPU。
- 使用 GPU1：`--cuda-device-index 1`。
- W&B online，group 继承 `paper-stage1-round9-boundary-ablation`。
- 每个 variant 使用 `init_from_checkpoint=output/material_refine_paper/stage1_round7_gradient_guard/best.pt`。

当前已确认：

- 第一个 variant `boundary_w005_k5` 已通过 preflight。
- 训练数据：96 ABO locked records；验证数据：40 ABO locked records。
- 设备：`cuda:1`，日志中 `CUDA_VISIBLE_DEVICES=None`，没有硬编码绑卡。
- 训练进度条已输出 loss、lr、boundary loss、view supervision rate、samples/s 和 GPU memory。
- 已写出 `progress_001_of_040` 到至少 `progress_005_of_040` 的 validation JSON。
- `progress_005_of_040` 当前 UV total `0.060075`，baseline `0.114024`；residual changed pixel rate `0.5715`，regression rate `0.2261`。

### Round9/R-v2 render-proxy smoke

命令输出目录：

`output/material_refine_paper/debug_round9_rv2_render_proxy_smoke_val`

验收结果：

- Round7 checkpoint 能作为 model-only init 加载到 R-v2，新增层 `missing=70`，`unexpected=0`，`skipped_shape=0`。
- R-v2 参数量约 `1.90M`。
- progress validation 成功触发 `progress_001_of_001`。
- validation JSON 写入 `render_proxy_validation.enabled=true`、`residual_gate_diagnostics` 和 `loss.material_context`。

关键 smoke 数值：

- baseline UV total MAE `0.093270`，R-v2 refined UV total MAE `0.049377`。
- view-proxy delta `-0.004867`，说明该极小 smoke 仍存在 view/render 退化风险。
- residual changed pixel rate `0.4644`，regression rate `0.1749`。
- `material_context` loss 已进入训练/验证 loss 字段，progress bar 中显示为 `mat=...`。

结论：

- R-v2 工程链路可用，但 smoke 不能作为论文有效性结论。
- Round9 full training 前应先跑 R-v2 组件消融，优先选择同时不过度退化 render-proxy 和 residual safety 的配置。

### Stage1-v2 diagnostic eval smoke 更新

命令输出目录：

`output/material_refine_paper/stage1_v2_dataset_20260422/diagnostic_eval_smoke_cpu`

验收结果：

- diagnostic-only manifest 可被 eval loader 正常读取。
- `summary.json` 正常生成，rows `4`，objects `2`。

关键观察：

- baseline UV total MAE `0.4092`，refined UV total MAE `1.0631`。
- view total MAE delta `-1.1576`，proxy PSNR delta `-0.9763`，proxy SSIM delta `-0.0144`。
- refined metal confusion rate `1.0`，prior residual safety score `-0.5192`。

结论：

- Stage1-v2 diagnostic/OOD 通路可用。
- 当前 with-prior ABO 模型在 no-prior scarce material diagnostic 子集上明显退化；这些数据只能用于 failure discovery/OOD 诊断，不能进入 paper-stage 主训练。

### R-v2 ablation launcher dry-run

命令：

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/run_material_refine_round9_boundary_ablation.py \
  --matrix-config configs/material_refine_round9_rv2_component_ablation_matrix.yaml \
  --dry-run true \
  --max-variants 2 \
  --report-to none \
  --wandb-mode disabled \
  --device cpu
```

验收结果：

- 成功解析 `rv2_full` 与 `rv2_no_boundary_context`。
- launcher 默认 summary root 已按 matrix 分离，R-v2 dry-run 写入 `output/material_refine_paper/rv2_component_ablation_matrix/round9_ablation_launcher_summary.json`，不会再覆盖 boundary ablation summary。
- 后续可用同一 launcher 顺序执行完整 R-v2 component ablation。

### R-v2 component ablation smoke 已启动

tmux session：

`sf3d_round9_rv2_component_ablation_smoke_gpu1_20260422`

launcher 日志：

`output/material_refine_paper/rv2_component_ablation_matrix/launcher.log`

variant 日志目录：

`output/material_refine_paper/rv2_component_ablation_matrix/logs`

当前已确认：

- 第一个 variant `rv2_full` 已通过 preflight。
- 设备为 `cuda:1`，`CUDA_VISIBLE_DEVICES=None`，没有硬编码绑卡。
- Round7 checkpoint 以 model-only init 方式加载到 R-v2，新增层 `missing=70`，`unexpected=0`，`skipped_shape=0`。
- 训练数据为 96 条 ABO locked records，验证数据为 40 条 ABO locked records；当前仍只有 `glossy_non_metal`，因此只能作为 R-v2 工程/消融 smoke，不是多材质 paper-stage 结论。

观察命令：

```bash
tail -f output/material_refine_paper/round9_boundary_ablation/logs/boundary_w005_k5.log | grep -E '\\[train\\]|\\[val\\]|\\[epoch\\]|epoch '
```

launcher 总日志：

```bash
tail -f output/material_refine_paper/round9_boundary_ablation/launcher.log
```

R-v2 component ablation 观察命令：

```bash
tail -f output/material_refine_paper/rv2_component_ablation_matrix/logs/rv2_full.log | grep -E '\\[train\\]|\\[val\\]|\\[epoch\\]|epoch '
```

## 2026-04-23 数据同步与 Round9 适配

详细中文报告：

`docs/material_refine_round9_dataset_analysis_20260423.md`

最新 Stage1-v2 输出：

`output/material_refine_paper/stage1_v2_dataset_20260423`

关键结论：

- unique records 已到 `1834`。
- strict paper 仍只有 `362`，且全部是 `ABO_locked_core / glossy_non_metal / with_prior`。
- diverse diagnostic 有 `512`，覆盖 `metal_dominant`、`ceramic_glazed_lacquer`、`glass_metal`、`mixed_thin_boundary`、`glossy_non_metal`。
- OOD eval 有 `256`，全部是 3D-FUTURE no-prior diagnostic/OOD。
- current main 仍保留 locked 346，不替换为 Stage1-v2 strict。

新增 readiness 输出：

`output/material_refine_paper/round9_dataset_readiness_20260423/round9_dataset_readiness.json`

`output/material_refine_paper/round9_dataset_readiness_20260423/round9_dataset_readiness.md`

readiness 结论：

- `KEEP_CURRENT_MAIN_AND_USE_STAGE1_V2_FOR_DIAGNOSTIC`
- blocker：`strict_records=362<min_replace=384`
- blocker：`non_glossy_paper_material_count_below_16:{}`

新增 Round9 配置：

- `configs/material_refine_train_paper_stage1_round9_conservative_boundary.yaml`
- `configs/material_refine_eval_paper_stage1_round9_conservative_boundary.yaml`
- `configs/material_refine_eval_stage1_v2_diagnostic_20260423.yaml`
- `configs/material_refine_eval_stage1_v2_ood_20260423.yaml`

已启动 Round9 conservative boundary：

tmux session：

`sf3d_round9_conservative_boundary_gpu1_20260423`

训练日志：

`output/material_refine_paper/stage1_round9_conservative_boundary/logs/train.log`

启动检查：

- preflight 通过。
- paper eligible：`346/346`。
- target_prior_identity：`0.0000`。
- effective_view_rate：`1.0000`。
- Round7 model-only init：`missing=0`，`unexpected=0`。
- 训练数据：`240`，验证数据：`40`，没有混入 Stage1-v2 smoke_only。
- 首个 validation：`progress_001_of_040`，UV total `0.061786`，baseline `0.114024`，residual regression rate `0.2349`。
