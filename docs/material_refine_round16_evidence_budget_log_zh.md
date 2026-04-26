# Material Refine Round16 Evidence Update Budget 迭代日志

更新时间：2026-04-24 UTC

## 背景

Round15 的 material evidence calibration 明显缓解了 Round14 的 render 退化，但 residual safety 仍不理想：

- balanced 的 `changed_pixel_rate` 约 0.852，`regression_rate` 约 0.203。
- OOD 的 `changed_pixel_rate` 接近 0.972，说明 no-prior 场景仍在大面积修改 UV atlas。
- boundary/SSIM/LPIPS 仍有轻微退化，说明模型还缺少“只在证据充分区域更新”的约束。

## 新增模块

新增 `Evidence Update Budget`，它不是重新训练的分支，而是一个结构性 residual 安全门：

- 输入：prior confidence、view uncertainty、branch confidence、boundary safe mask、bleed risk、topology consistency、material confidence、metallic evidence。
- 输出：roughness/metallic 两个通道的 `evidence_update_budget_uv`。
- 作用：在 no-prior、边界高风险、视角证据弱、拓扑不一致区域压低 residual gate。
- 设计原则：优先提高稳定性和可解释性，允许少量牺牲 UV MAE，换取更低的 unnecessary/regression update。

## 当前已完成

- 已在 `sf3d/material_refine/model.py` 接入 `enable_evidence_update_budget`。
- 已在 `pipeline.from_checkpoint` 支持 eval-only `model_cfg_overrides`，不用重新保存 checkpoint 即可测试无参数结构门。
- 已在 `eval_material_refiner.py` 中保存 `manifest_snapshot.json`，避免后续 `dataset_latest` 漂移后无法审计。
- 已创建 round16 三套 eval 配置：balanced、locked346、OOD。
- 已启动 tmux：`sf3d_round16_evidence_budget_eval`。

## 已出结果

balanced test：

- Round15 `avg_improvement_total`: 0.04383
- Round16 `avg_improvement_total`: 0.04284
- Round15 `PSNR delta`: 0.06426
- Round16 `PSNR delta`: 0.05546
- Round15 `boundary delta`: -0.02801
- Round16 `boundary delta`: -0.02543
- Round15 `changed_pixel_rate`: 0.85225
- Round16 `changed_pixel_rate`: 0.84491
- Round15 `regression_rate`: 0.20252
- Round16 `regression_rate`: 0.19771

locked346：

- Round15 `avg_improvement_total`: 0.01584
- Round16 `avg_improvement_total`: 0.01511
- Round15 `PSNR delta`: 0.88539
- Round16 `PSNR delta`: 0.84427
- Round15 `SSIM delta`: -0.00646
- Round16 `SSIM delta`: -0.00596
- Round15 `LPIPS delta`: 0.00063
- Round16 `LPIPS delta`: 0.00080
- Round15 `boundary delta`: -0.01209
- Round16 `boundary delta`: -0.01134
- Round15 `changed_pixel_rate`: 0.42421
- Round16 `changed_pixel_rate`: 0.38850
- Round15 `regression_rate`: 0.11202
- Round16 `regression_rate`: 0.10762

## 临时判断

Round16 不是 UV 提升型迭代，而是 safety/stability 迭代：

- 优点：changed pixel、regression、boundary 退化、SSIM/LPIPS 在 locked346 上都有小幅改善。
- 代价：UV improvement 与 PSNR 略降。
- 风险：balanced 的 metal confusion 略差，说明 metallic budget 可能过于保守或证据定义仍偏弱。

## 下一步

- 等 OOD 完成后决定是否将 `Evidence Update Budget` 作为默认 conservative gate。
- 若 OOD safety 明显改善且 render 不再明显退化，Round17 应进入训练型迭代：把 budget 作为可学习但受约束的辅助头，并加入 budget calibration loss。
- 若 OOD UV/render 掉得太多，则保留该模块为 eval/export safety option，不纳入主训练默认路径。
