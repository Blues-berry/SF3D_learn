# Material Refine Smoke 清理与增量数据适配日志 2026-04-23

## 结论

目前不建议直接删除历史 smoke 输出或旧 round 配置，因为它们仍然承担复现实验、回退 checkpoint 和 ablation 对照的作用。更安全的清理方式是：

- 默认入口只保留 Round9 active/latest。
- 旧 smoke、旧 round、旧 Stage1-v2 20260422 配置统一视为 archive / reproducibility-only。
- GPU0 新增数据通过自动同步脚本进入 `strict_paper / diverse_diagnostic / ood_eval` 三类子集，而不是手工改训练配置。

最新自动同步输出：

- latest index：`output/material_refine_paper/round9_dataset_latest.json`
- latest markdown：`output/material_refine_paper/round9_dataset_latest.md`
- 自动 Stage1-v2 数据：`output/material_refine_paper/stage1_v2_dataset_auto_20260423T021048Z`
- readiness：`output/material_refine_paper/round9_dataset_readiness_auto_20260423T021048Z`

同步命令：

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/sync_material_refine_round9_datasets.py
```

## 当前可用数据统计

本轮同步自动发现并合并以下数据源：

- `output/material_refine_dataset_factory/longrun_gpu0_20260421T122652Z/canonical_manifest_supervisor_merged.json`
- `output/material_refine_dataset_factory/objaverse_cached_gpu0_20260422T124111Z/canonical_manifest_supervisor_merged.json`
- `output/material_refine_dataset_factory/paper_unlock_gpu0_20260421T123946Z/canonical_manifest_supervisor_merged.json`
- `output/material_refine_dataset_factory/scarce_material_gpu0_20260422T073856Z/canonical_manifest_supervisor_merged.json`
- `output/material_refine_paper/latest_dataset_check_20260421/stage1_subset_merged490/paper_stage1_subset_manifest.json`

Stage1-v2 latest 结果：

- strict paper：362 条，全部 `ABO_locked_core / glossy_non_metal / with_prior / paper_pseudo`
- diverse diagnostic：512 条，覆盖 `metal_dominant / ceramic_glazed_lacquer / glass_metal / mixed_thin_boundary / glossy_non_metal`
- OOD eval：256 条，全部来自 `3D-FUTURE_highlight_local_8k`，覆盖 5 类 material family

readiness 结论：

- `KEEP_CURRENT_MAIN_AND_USE_STAGE1_V2_FOR_DIAGNOSTIC`
- Round9 主训练继续使用 locked 346。
- Stage1-v2 diagnostic/OOD 只进入验证、OOD、可视化和失败分析。

## 仍不适配新数据增量的地方

1. strict paper 的材质族仍然单一。

虽然 strict paper 从 346 增到 362，但仍全部是 `glossy_non_metal`。这意味着新数据尚不能替换主训练集，也不能支撑多材质 paper-stage 主结论。

2. 多材质样本大多还没有完成二次验证。

`3D-FUTURE` 和 scarce material 提供了 metal、glass、ceramic、thin boundary，但大多还是 `smoke_only` 或 `auxiliary_upgrade_queue`。这些数据可以用于诊断，不应直接训练。

3. 二次验证后的字段必须和训练过滤器一致。

如果新处理数据想进入 Round9/后续主训练，需要满足：

- `target_quality_tier` 为 `paper_pseudo` 或 `paper_strong`
- `target_source_type` 不是 `copied_from_prior` 或 `unknown`
- `target_is_prior_copy=false`
- `supervision_role=paper_main`
- `paper_split` 使用 `paper_train / paper_val_iid / paper_test_iid / paper_test_material_holdout`
- `view_supervision_ready=true`
- `license_bucket` 在 paper license allowlist 内

4. license bucket 仍需要归一化。

本轮已经把 Stage1-v2 paper 默认 allowlist 收紧，去掉了 `unknown`。Objaverse 这类来源即使质量过关，也必须先把 license bucket 映射到可审计的标准桶。

5. 当前 eval diagnostic/OOD 仍是抽样入口。

`configs/material_refine_eval_stage1_v2_diagnostic_20260423.yaml` 和 `configs/material_refine_eval_stage1_v2_ood_20260423.yaml` 当前设置 `max_samples: 128`，适合快速诊断。等 Round9 full test 结束后，论文阶段应基于 latest manifest 生成 full diagnostic/OOD eval，不再限制样本数。

6. Stage1-v2 build 还缺少 per-source audit cache。

自动同步已经能吃增量数据，但每次会重新审计所有源 manifest。随着 GPU0 数据继续增长，建议下一步给 `build_material_refine_stage1_v2_subsets.py` 加 per-source audit cache，避免 refresh latest 越来越慢。

## 活跃入口

当前只建议直接使用这些入口：

- train：`configs/material_refine_train_paper_stage1_round9_conservative_boundary.yaml`
- resume：`configs/material_refine_train_paper_stage1_round9_conservative_boundary_resume_latest.yaml`
- full-test eval：`configs/material_refine_eval_paper_stage1_round9_conservative_boundary.yaml`
- diagnostic eval：`configs/material_refine_eval_stage1_v2_diagnostic_20260423.yaml`
- OOD eval：`configs/material_refine_eval_stage1_v2_ood_20260423.yaml`
- dataset sync：`scripts/sync_material_refine_round9_datasets.py`

## 归档候选

以下类型不再作为默认入口：

- `configs/material_refine_eval_stage1_v2_diagnostic_smoke.yaml`
- Round1/Round2/Round3 训练与评估配置
- Round4/Round5/Round6/Round7/Round8 的历史配置
- Round9 boundary/R-v2 ablation smoke matrix
- `debug_*_smoke` 输出目录
- `stage1_v2_dataset_20260422/*diagnostic_eval_smoke*`

这些内容暂不删除，只在 `round9_dataset_latest.md` 中标记为历史/归档候选。等当前 Round9 和二次验证闭环完成后，再统一移动到 archive 目录更稳。

## 对 Round9 的适配

Round9 当前运行保持：

- 主训练：locked 346 的 `paper_train`
- 验证：locked 346 的 `paper_val_iid`
- full test：locked 346 的 test split
- 增量多材质：Stage1-v2 latest diagnostic/OOD eval-only

这样做的原因很简单：训练证据继续保持可审计，新增多材质数据先用于发现失败模式。等二次验证把多材质样本提升为 `paper_pseudo / paper_strong + paper_main`，再通过 `sync_material_refine_round9_datasets.py` 刷新 latest，若 replacement gate 通过，再切换主训练 manifest。
