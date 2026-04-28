# Material Refine Round15 启动与清理日志

更新时间：2026-04-24 UTC

## Round15 当前状态

- 已启动 tmux 会话：`sf3d_round15_material_evidence`
- 训练入口：`scripts/run_material_refine_stage1_v3_round15_material_evidence.sh`
- 统一实验入口：`scripts/run_material_refine_experiment.py`
- 实验配置：`configs/material_refine_experiment_round15.yaml`
- 输出目录：`output/material_refine_paper/stage1_v3_round15_material_evidence_calibration`
- 初始化 checkpoint：`output/material_refine_paper/stage1_v3_round14_backbone_topology_render/best.pt`
- GPU 策略：训练/eval 使用 `cuda_device_index: 1`，避免抢占 GPU0 上的数据处理任务。

## Round15 数据与评测入口

- train：`stage1_v3_strict_paper_candidates.json`，当前训练过滤后 786 条。
- val：`stage1_v3_balanced_paper_manifest.json`，当前验证过滤后 57 条，按 `material_family` 平衡。
- balanced test：112 条，包含 3D-FUTURE 与 ABO、5 类材质族。
- locked346 regression：从旧协议恢复完整 346 条 manifest，固定评测 66 条 heldout。
- OOD：509 条 3D-FUTURE OOD object，全部为 `paper_test_ood_object`。

## 关键修复

- 修复 `locked346` manifest 缺失：恢复到 `output/material_refine_paper/stage1_locked346/stage1_locked346_manifest.json`。
- 修复 `locked346` split：使用历史协议 `paper_test_iid,paper_test_material_holdout`，不再使用不存在的 `paper_test_regression`。
- 修复 OOD split：使用真实字段 `paper_test_ood_object`，不再使用旧的 `ood_eval`。
- Round15 脚本训练完成后自动串联 balanced、locked346、OOD 三套 eval。

## 清理策略

- 已执行清理：先将 53 个旧 smoke / round1-11 / stage1-v2 输出目录移入时间戳 trash 目录并生成清单，随后按本轮“删除无关旧实验”的要求物理删除该 trash 目录。
- 清理候选报告：`output/material_refine_cleanup_candidates_round15_executed.json`
- 移动执行报告：`output/material_refine_cleanup_candidates_round15_executed_executed.json`
- 释放空间：约 5.6GB。
- 已归档过期配置与旧一次性启动脚本：`output/material_refine_config_script_archive_20260424.md`
- 未清理内容：当前 round15、stage1-v3 dataset latest、locked346、round12/13/14/15 主线对照、活跃数据处理脚本与输出。

## 新增顶会级统一启动器

后续推荐使用：

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/run_material_refine_experiment.py \
  --spec configs/material_refine_experiment_round15.yaml \
  --launch-tmux
```

该入口会做：

- 依赖检查：`torch`、`wandb`、`tqdm`、`rich`、`psutil`、`orjson`、`lpips`、`torchmetrics`。
- CUDA 检查与 GPU 信息输出。
- train/val/eval manifest 路径检查。
- 训练与多 benchmark eval 串联。
- 统一日志写入 `run_root/logs`。

## 依赖状态

- 已确认存在：`orjson`、`rich`、`psutil`、`tqdm`、`einops`、`safetensors`、`lpips`、`skimage`、`pytest`、`pynvml`。
- 已补装：`torchmetrics`、`lightning-utilities`。

## 注意事项

- 当前 Round15 是工程迭代与方法诊断，不应在 balanced/locked346/OOD 三套 eval 完成前表述为 paper-stage 有效结论。
- 若 Round15 仍出现 UV 提升但 render / boundary / metal confusion 退化，应优先回看 material evidence gate 与 metallic cap 的强度，而不是继续盲目增加训练 epoch。
