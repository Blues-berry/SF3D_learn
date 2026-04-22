# Material Refine Round9 / Stage1-v2 更新日志 2026-04-22

## 当前状态

- Round8 仍在训练，full test eval 尚未生成，因此 Round9 full training 不能启动。
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

观察命令：

```bash
tail -f output/material_refine_paper/round9_boundary_ablation/logs/boundary_w005_k5.log | grep -E '\\[train\\]|\\[val\\]|\\[epoch\\]|epoch '
```

launcher 总日志：

```bash
tail -f output/material_refine_paper/round9_boundary_ablation/launcher.log
```
