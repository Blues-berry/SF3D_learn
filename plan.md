# SF3D 轻量材质精修三段式方案 v1

## 2026-04-20 Readiness Update

- `paper-stage` 数据门禁已被最新 merged manifest 真正通过，不再停留在 smoke 状态：
  - manifest: `output/material_refine_longrun_stress24_hdri900_20260419T134158Z/canonical_manifest_monitor_merged.json`
  - `records=237`
  - `paper_stage_eligible_records=210`
  - `target_prior_identity_rate=0.1139`
  - `effective_view_supervision_record_rate=1.0`
  - `readiness_blockers=[]`
- 固定 stage1 subset 已生成：
  - `output/material_refine_pipeline_20260418T091559Z/paper_stage1_pipeline_auto_select/readiness/stage1_subset/paper_stage1_subset_manifest.json`
  - 当前 `stage1_subset_records=210`，`paper_train=142`，`paper_val_iid=28`，`paper_test_iid=18`，`paper_test_material_holdout=22`。
- `train_material_refiner.py` 的 `preflight` 兼容性 bug 已修复：
  - 之前即使显式传入 `--train-manifest/--val-manifest`，仍会因为 config 里的旧 `--manifest` 别名路径不存在而误报失败。
  - 现在 `preflight` 只校验实际使用的 `train_manifest` 与 `val_manifest`，paper-stage subset 可以正常通过 preflight。
- `scripts/launch_material_refine_paper_stage1_tmux.sh` 默认候选范围已补齐到 HDRI900 longrun 输出：
  - 新增默认 glob：`output/material_refine_longrun_stress24_hdri900_*/**/canonical_manifest*.json`
  - 后续 paper-stage watcher 不再只盯旧 smoke/full manifest。
- `scripts/launch_material_refine_paper_stage1_tmux.sh` 现已支持 GPU idle wait：
  - 可通过 `WAIT_FOR_GPU=true`、`GPU_INDEX=0`、`MAX_GPU_USED_MB=4096`、`GPU_POLL_SECONDS=60` 在后台挂起，等 Blender 释放目标卡后再启动正式 pipeline。
- `paper-stage` launcher 默认等待卡已修正为 `GPU0`，与
  - `configs/material_refine_train_paper_stage1.yaml`
  - `configs/material_refine_eval_paper_benchmark.yaml`
  中的 `cuda_device_index: 0` 保持一致，避免出现“launcher 等 GPU1，但训练实际跑 GPU0”的错位。
- 当前剩余阻塞不再是数据有效性，而是资源占用：
  - 2026-04-20 检查时两张卡都被 Blender 数据任务占用，`GPU1` 已使用约 `24GB` 显存。
  - 因此当前策略是先保持 paper-stage 入口 ready，再等待数据侧释放 GPU 后启动正式训练，而不是与数据端抢卡。

## 2026-04-19 P0 Checkpoint

- 已落地 paper-stage 数据门禁闭环：manifest 新增 `target_source_type`、`target_is_prior_copy`、`target_quality_tier`、`target_confidence_summary`，训练 preflight 会把 target/prior 同源和 non-trivial target 数量一起作为硬门禁。
- 已落地严格 buffer 校验：当前 full manifest 的 `effective_view_supervision_record_rate=0.0`，因此 `view_consistency` 已明确下线到 `view_consistency_mode=disabled`、`view_consistency_weight=0.0`，不再把它当作已验证贡献。
- 已落地固定 split/stage1 subset 工具链：`scripts/build_material_refine_paper_stage1_subset.py` 会生成固定 split、split audit、stage1 subset manifest 和 readiness summary。
- 已落地 paper-stage 自动编排：`scripts/run_material_refine_paper_stage1_pipeline.py` 会循环执行 readiness 审计，只有在 P0 通过后才启动 train/eval/report/W&B 总链路；`scripts/launch_material_refine_paper_stage1_tmux.sh` 提供后台入口。
- 已落地多 manifest 自动择优：`scripts/select_material_refine_best_manifest.py` 会对现有 canonical manifests 做缓存化审计和排序，paper-stage watcher 会自动追踪当前最有希望的一份，而不是只盯固定旧 manifest。
- 已落地论文 ablation 入口：训练侧支持 `disable_prior_inputs / disable_view_fusion / disable_residual_head`，并新增 `scripts/run_material_refine_ablation_suite.py` 与对应 override configs。
- 已落地 failure-aware 训练接口与更完整评测维度：支持 tag/difficulty/quality-tier 采样重加权；eval 现在输出 metal/non-metal F1/AUROC、failure-tag reduction、runtime/memory、以及按 `license_bucket/category_bucket/target_quality_tier` 的分层汇总。
- 历史 full manifest 仍然 **blocked**：`target_prior_identity_rate=1.0000`、`paper_stage_eligible_records=0`、`stage1_subset_records=0`。这些结果只能表述为 engineering smoke，不可表述为方法已有效。
- 在数据侧产出 non-trivial RM target 之前，禁止启动 `configs/material_refine_train_paper_stage1.yaml` 的 full training。

## Production Dataset Run 2026-04-18

- 数据侧当前目标切到 `Pool-A 8,200+`，先用本地可立即处理资产长跑，不等待远端源头全部下载完成。
- 主 manifest 已生成：`output/highlight_pool_a_8k/material_refine_manifest_pool_a_8k.json`，共 `8,200` 条，`8,200` 条本地可访问，`0` 条 blocked。
- 主渲染已挂在 `GPU1`：`prepare_material_refine_dataset.py`，`10` 个 Blender worker，`256` 分辨率，`8` samples，输出目录 `output/highlight_pool_a_8k/prepared_gpu1/full`。
- `GPU0` 上已有训练进程，数据处理不触碰；所有新增渲染显式使用 `--cuda-device-index 1`。
- 当前主线组成：`ABO_locked_core 500` + `3D-FUTURE_highlight_local_8k 7,700`。
- 当前 split：`train 6,645`、`val 778`、`test 777`，按 object id deterministic hash 做对象级隔离。
- 当前高光类别计数：`metal_dominant 1,959`、`ceramic_glazed_lacquer 345`、`glass_metal 1,230`、`mixed_thin_boundary 1,566`、`glossy_non_metal 2,600`、`ABO pending 500`。
- 远端源头并行下载已启动：`Poly Haven HDRI 60` 已落盘，`Objaverse-XL strict-filtered 1,500` 正在抓 annotations/下载，输出目录 `output/highlight_pool_a_8k/aux_sources`。
- 3D-FUTURE 的 license bucket 固定为 `custom_tianchi_research_noncommercial_no_redistribution`，不和开放源混发。
- Objaverse-XL 只进严格筛选通道：先按 per-object license 排除 NC/ND/NoAI/unknown，再按高光关键词选入 ceramic/glass/metal/thin/glossy 增量候选。
- Poly Haven HDRI 作为 Pool-D 自然光 stress bank，优先覆盖 indoor high contrast、soft window、hard sun、overcast、night urban、studio product 六类光照。
- Pool-B/C/E/F 仍不直接混入主 RM CSV：OLATverse/OpenIllumination/ICTPolarReal 用于受控高光辅助，3DCoMPaT++ 用于 part-material 先验，OpenSVBRDF/MatSynth 用于材料先验，Stanford-ORB/Objects-With-Lighting 用于 OOD/benchmark。
- 本地 8.2k 先跑完 canonical bundles；远端下载完成后生成 `Pool-A remote increment` manifest，再单独追加渲染 shard 并合并 manifest。
- 2026-04-19 checkpoint：`Pool-A 8,200` 主准备已完成 `8,199/8,200`，唯一失败对象是 `3dfuture_41601bc0-d767-47b3-9a12-dc591cceef5f`，Blender `SIGSEGV`，后续单独 retry/替换。
- 2026-04-19 checkpoint：`Poly Haven HDRI 60/60` 已下载，并已进入两条 GPU 数据任务：`GPU1` 跑全量 `8,199 objects x 6 HDRI strata` stress render，`GPU0` 跑 `240 objects x 60 HDRIs` dense stress subset。
- 2026-04-19 checkpoint：Objaverse-XL 四个 annotations parquet 已缓存到 `output/highlight_pool_a_8k/aux_sources/objaverse_xl`；GitHub 下载因系统缺 `git-lfs` 不作为当前主通道，改用 cached `Sketchfab + Smithsonian` permissive-license rows 生成增量候选，再做 second-pass。
- 2026-04-19 checkpoint：系统 `git-lfs` 已安装并初始化，`git-lfs/2.9.2` 可用；Objaverse GitHub 源已从 blocked 通道恢复为可验证通道。
- 2026-04-19 checkpoint：`scripts/stage_objaverse_cached_increment.py` 已补 GitHub `save_repo_format=files`、selection pre-manifest、严格 permissive license 过滤，避免把 GPL/MPL 这类 copyleft license 因 “Public” 字符串误判为开放主池。
- 2026-04-19 checkpoint：Objaverse cached `Sketchfab + Smithsonian` 增量下载仍在跑，候选 `1,500` 条，其中 `Sketchfab 1,491`、`Smithsonian 9`；manifest 输出位于 `output/highlight_pool_a_8k/objaverse_cached_increment`。
- 2026-04-19 checkpoint：Objaverse GitHub LFS 增量已启动，目标 `180` 个 `glb/gltf`，严格限定明确许可 `MIT/Apache/CC0/BSD/Unlicense/CC-BY` 等，保存目录 `output/highlight_pool_a_8k/objaverse_github_lfs_increment`，底层 repo 文件写入 `output/highlight_pool_a_8k/aux_sources/github/repos`。
- 2026-04-19 checkpoint：Objaverse GitHub LFS 增量完成：`180` 选中、`176` 下载成功、`4` 个 repo/object missing；`git-lfs` 已确认不是失败原因。增量 manifest 已转换到 `output/highlight_pool_a_8k/objaverse_github_lfs_increment_manifest/material_refine_manifest_objaverse_increment.json`，`176/176` 本地可用，split 为 `train 141 / val 20 / test 15`。
- 2026-04-19 checkpoint：Objaverse GitHub LFS second-pass/prepare 已挂到 `GPU0` tmux session `sf3d_objaverse_github_prepare_gpu0`，`4` 个 Blender worker，输出 `output/highlight_pool_a_8k/objaverse_github_lfs_prepared_gpu0`，日志 `output/highlight_pool_a_8k/objaverse_github_lfs_prepare_gpu0_20260419T024437Z.log`。
- 2026-04-19 checkpoint：Thingiverse cached 源只找到 `8` 个许可合格 `.obj`，但下载端点全部 `download_missing`；保留审计记录，不作为有效增量源。
- 2026-04-19 checkpoint：新增 Pool-E `PolyHaven_CC0_PBR_material_bank`，脚本 `scripts/stage_polyhaven_material_bank.py` 已通过 3-material probe，正式目标 `120` 个高光相关 1K PBR material maps，输出目录 `output/highlight_pool_a_8k/aux_sources/polyhaven_materials`。
- 2026-04-19 checkpoint：训练/评测侧已把 `generator_id` 升级成一等实验轴，支持 `train_generator_ids` / `val_generator_ids` / `--generator-ids` 过滤、`generator_x_prior` 采样平衡、W&B 按上游分组统计和验证面板展示。
- 2026-04-19 checkpoint：新增论文级实验协议 `docs/material_refine_paper_protocol.md`，明确顶会提交所需 baselines、metrics、ablation、cross-generator generalization、figures/tables、W&B group 与 reproducibility gates。
- 2026-04-19 checkpoint：新增论文主实验配置 `configs/material_refine_train_paper_stage1.yaml` 与 benchmark 配置 `configs/material_refine_eval_paper_benchmark.yaml`；当前仍受 `target/prior identity=1.0` 数据门禁阻止，需等数据侧产出非平凡 RM target 后运行。

## Summary

- 方法固定为 `G -> C -> R` 三段：`G` 只接上游资产生成器输出，v1 先落地 `SF3D`；`C` 负责把不同上游结果统一成可重渲染、可训练、可评测的标准表示；`R` 是轻量化材质精修器，v1 输出 `UV roughness map + UV metallic map`。
- v1 不改 `SF3D` 主模型权重与几何生成逻辑；只在其后新增标准化、训练、推理、展示、评测链路。
- 训练主线采用 `Pool-A 强监督 + Pool-B/C/E 辅助 + Pool-D 光照增强 + Pool-F OOD 评测`。`ABO 500` 继续作为核心监督集，同时增补经审计后的 `3D-FUTURE` 与 `Objaverse-XL` 家具/家居子集。
- 展示拆成两层：保留当前 `Gradio` 交互演示做单例验证；复用现有 HTML report 骨架做 benchmark 与 before/after 对比页。

## Key Changes

- 定义统一中间表示 `CanonicalAssetRecordV1`，直接扩展现有 `mini_v1_manifest.json` 的 `records`，不另起一套 manifest。
- `CanonicalAssetRecordV1` 至少新增这些字段：
  - `generator_id`: `sf3d` / 未来 `spar3d` / `hunyuan3d`
  - `license_bucket`: 源数据许可证桶，训练与导出严格分桶
  - `has_material_prior`: 是否有上游材质先验
  - `prior_mode`: `scalar_rm` / `uv_rm` / `none`
  - `generator_bundle_root`: 上游输出根目录
  - `canonical_mesh_path`, `canonical_glb_path`
  - `uv_albedo_path`, `uv_normal_path`
  - `uv_prior_roughness_path`, `uv_prior_metallic_path`，若无则为空
  - `scalar_prior_roughness`, `scalar_prior_metallic`，若无则为空
  - `uv_target_roughness_path`, `uv_target_metallic_path`, `uv_target_confidence_path`
  - `canonical_views_json`, `canonical_buffer_root`
  - `supervision_tier`: `strong` / `aux_highlight` / `part_semantic` / `material_prior` / `eval_only`
- `C` 的标准输出同时保留两类表示：
  - UV 域：`mesh + uv + albedo + normal + prior rm + target rm + confidence`
  - 多视图域：固定相机下的 `rgba / mask / depth / normal / position / uv / visibility`
- v1 固定 canonical eval views 复用现有 3 视角：`front_studio`、`three_quarter_indoor`、`side_neon`；训练预处理可额外加少量抖动视角，但评测与报告始终回到这 3 个标准视角。
- `R` 采用轻量 UV 精修架构，而不是重新做一个大型 3D 网络：
  - 共享小型 view encoder 编码多视图 `RGB+mask+depth+normal+position`
  - 将多视图特征通过 UV baking/fusion 聚合到 atlas
  - 拼接 `uv_albedo + uv_normal + prior_rm + prior_confidence + fused_view_features`
  - 用 2D UNet/ResUNet 预测 `Δroughness_uv` 与 `Δmetallic_uv`
  - 最终输出 `roughness_uv`、`metallic_uv`
- 双模式统一到一个模型里：
  - 模式 A：有先验时，先验作为输入并预测残差
  - 模式 B：无先验时，启用 `init head` 先估计 coarse RM，再走同一个 refinement head
  - 训练时默认做 `prior dropout`，让同一模型兼容两种模式
- 对 `SF3D` 的特化处理固定如下：
  - `baseColorTexture`、`normalTexture` 直接作为可用输入
  - `roughnessFactor`、`metallicFactor` 视为弱先验，广播成 UV 常值图，并附低置信度 prior mask
  - 精修后导出的 GLB 改为显式写入 `roughnessTexture`、`metallicTexture`，对应 factor 设为 `1.0`
- geometry mismatch 不做在线可微监督，改走离线伪 GT 构建：
  - 先渲染 GT 资产多视图 `roughness/metallic`
  - 再把这些 view-space GT 按预测 mesh 的 `uv/visibility` 烘回预测 UV atlas
  - 生成 `uv_target_* + confidence map`
  - 低置信 texel 只做弱约束或回退 prior，避免几何偏差污染训练
- 多材质槽资产在 `C` 中统一 bake 成单 atlas；无 UV 资产进入 `B_fixable` 分支，先 unwrap/bake 再入训练。

## Data And Training

- `Pool-A` 是主训练池：
  - 保留 `ABO/ecommerce 500 A_ready` 为核心
  - 增加 `3D-FUTURE` 的高光敏感家具/家居子集，独立 license bucket
  - 增加 `Objaverse-XL` 严格精筛子集，先按 license 与材质可用性过滤，再做 second-pass
  - 采样配额固定为：`30% 金属主导`、`20% 陶瓷/釉面/漆面`、`15% 玻璃+金属`、`20% mixed-material thin-boundary`、`15% glossy non-metal`
- `Pool-B` 作为高光受控辅助监督池：`OLATverse`、`OpenIllumination`、`ICTPolarReal`
  - 不直接混进主 RM 强监督 CSV
  - 用于 highlight/specular 相关辅助 loss、校准与真实受控验证
- `Pool-C` 用 `3DCoMPaT++` 补 part-material 结构先验，不作为 v1 主 RM GT 源
- `Pool-D` 固定为 HDRI stress bank：`Poly Haven` 为立即可用主集，`Laval` 作为扩展室内 HDR 校准集
- `Pool-E` 固定为材料先验池：`OpenSVBRDF`、`MatSynth`
  - 用于预训练/正则化 `init head` 与 RM 局部纹理先验
- `Pool-F` 仅做真实 OOD holdout：先保留 `Stanford-ORB` 规划位，不进训练
- 训练分两阶段：
  - Stage 1：只训 `Pool-A`，目标是把 `SF3D scalar RM` 升级为稳定 `UV RM maps`
  - Stage 2：接入 `Pool-B/C/E` 的辅助任务与 `Pool-D` 光照增强，强化 highlight / thin-boundary / mixed-material 泛化
- 主损失固定为：
  - `UV confident L1/Huber` on roughness/metallic
  - `edge-aware smoothness` on UV maps
  - `prior consistency` on low-confidence texels
  - `view consistency loss` on canonical renders
  - Stage 2 再加 `highlight consistency` 与 `part/material auxiliary loss`
- 训练产线拆成 4 个脚本职责：
  - `build_material_refine_manifest`: 合并多池、写 split、写 license bucket
  - `prepare_material_refine_dataset`: 跑 `G -> C`、生成 canonical bundle 与 UV pseudo-GT
  - `train_material_refiner`: 训练 `R`
  - `eval_material_refiner`: 输出 before/after metrics 与 report JSON

## Demo And Evaluation

- `Gradio` 不替换原 demo，改成双标签或双阶段流程：
  - `Base Asset`: 现有 SF3D 生成
  - `Material Refine`: 调用 `R`，展示 before/after GLB、roughness/metallic atlas、材质球对比、HDRI 切换
- 单例展示页默认显示：
  - 输入图
  - SF3D 原始 GLB
  - refined GLB
  - `roughness/metallic` before/after atlas
  - 3 个固定 HDRI 下的材质球与 turntable 对比
  - 当前模式 `with prior / no prior / auto`
- HTML benchmark 页直接复用现有报告脚本思路，升级为 before/after 版本：
  - 保留当前 4 个 failure tags：`metal_nonmetal_confusion`、`boundary_bleed`、`local_highlight_misread`、`over_smoothing`
  - 每个 case 卡片增加 `baseline vs refined` 两列
  - 汇总统计增加 `error reduction` 与 `tag count reduction`
- 评测默认同时输出：
  - UV 域 `roughness/metallic MAE`
  - 3 视角 view-space `roughness/metallic MAE`
  - failure taxonomy 计数
  - 按 category、bucket、generator_id、source、license_bucket 分层统计

## Paper Protocol Addendum

- 论文核心表述从 “SF3D 材质修复” 升级为 “生成器无关的轻量后处理材质精修”，SF3D 作为首个 adapter 与主要实验上游，SPAR3D/Hunyuan3D 通过同一 `CanonicalAssetRecordV1` 接口接入。
- 主实验必须至少比较：`G raw`、`G canonical prior`、`scalar broadcast`、`prior smoothing`、`no-view refiner`、`no-prior refiner`、`no-residual refiner`、`Ours full`。
- 指标体系扩展为：UV RM MAE、view-space RM MAE、rendered PSNR/SSIM/LPIPS、highlight localization、boundary bleed、metal/non-metal F1/AUROC、failure-tag reduction、GLB validity、runtime/memory。
- Cross-generator protocol 固定为：SF3D train/test、SF3D-to-SPAR3D/Hunyuan3D、mixed-generator train/test、leave-one-generator-out。
- Ablation protocol 固定为：去掉 material prior、prior dropout、multi-view buffers、UV fusion、residual head、confidence weighting、edge-aware loss、generator conditioning。
- W&B group 固定为：`paper-stage1`、`paper-ablation`、`paper-cross-generator`、`paper-benchmark`、`paper-figures`。

## Test Plan

- Smoke 预处理：
  - 用 8 个 `ABO` 资产完成 `G -> C`，确认能生成 canonical bundle、UV pseudo-GT、confidence map
  - 用 4 个 `3D-FUTURE` 和 4 个 `Objaverse-XL` 样本验证 `B_fixable/A_ready` 分流与 license bucket 写入
- Smoke 训练：
  - 在 16 个 `Pool-A` 样本上跑通 1 个 epoch，确认 `with prior` 和 `no prior` 两种 batch 都能前向、反向、保存 checkpoint
- Smoke 推理：
  - 从单张图完成 `SF3D -> R -> refined GLB`，输出中必须含 `roughnessTexture` 与 `metallicTexture`
- Smoke 展示：
  - `Gradio` 能加载 baseline 与 refined 结果
  - HTML report 能从新 `metrics.json` 生成 before/after 对比页
- 回归评测：
  - 在当前 `ABO RM mini` 的 200-object 基准上复跑
  - 重点检查 `metal_nonmetal_confusion` 与 `boundary_bleed` 是否显著下降
  - 如果某源 bucket 退化明显，能按 `source` 和 `license_bucket` 快速定位

## Assumptions And Defaults

- v1 先只实现 `SF3D adapter`，但所有接口按多生成器扩展设计；`SPAR3D/Hunyuan3D` 暂不做联调实现。
- v1 目标是 `RM 贴图精修`，不做全 PBR 全量精修；`albedo/normal` 先复用上游结果。
- 现有 `mini_v1_manifest.json` 的 `records` 继续作为 source of truth，新增字段扩展，不迁移为新格式。
- 训练默认以研究型内部流程为前提；不同数据源严格按 `license_bucket` 分开组织、训练记录与导出。
- 若资产缺失可靠 UV 或 PBR 通道，不强行并入强监督主池，而是进入 `B_fixable` 或辅助池。


可以，下面这版我按 **P0 / P1 / P2** 收成可直接执行的整改清单。

## P0：不解决就不要开 paper-stage 主训练

### 1. 先把“非平凡 RM target”做出来

现在最大的阻塞仍然是 `target_prior_identity=1.0000`，也就是 target 和 SF3D prior 完全一致；在这种数据上继续 full training，只会学会复制 prior，不能证明材质精修有效。round1 里 baseline MAE 为 `0.0`、refined MAE 只是小扰动，也已经验证了这一点。

**要做什么**

* 新 manifest 必须区分 target 来源：真实/伪 GT/直接复制 prior。
* 训练时只允许高价值 target 进入 paper-stage 主训练。
* 把 `target==prior rate` 作为硬门禁保留，并写死阈值。

**出门标准**

* `target==prior rate < 30%`
* 可单独统计“真实 target 子集”的训练/验证/测试规模
* paper-stage 只在非同源 target 子集上启动

---

### 2. 让 `view_consistency` 变成真监督，或者暂时关闭

协议里已经把多视图 canonical buffer、UV fusion、view consistency 写成方法核心，但当前记录显示没有显式 `uv.npy` 或稳定 view-to-UV 映射，导致 `uv_view_rate = 0%`，这项监督目前基本无效。

**要做什么**

* 补 `uv.npy` / view-to-UV 映射链路，真正把 view render 约束到 UV atlas。
* 如果短期补不上，就把 `view_consistency_weight=0`，并从阶段性结论里去掉这项贡献表述。

**出门标准**

* 二选一明确落地，不能再处于“配置里有、训练里没真作用”的状态。

---

### 3. 固定 paper-stage 的 heldout 协议，避免 source / prior / difficulty 混淆

当前 with-prior 的 ABO 和 without-prior 的 3D-FUTURE 分支表现差异很大，这说明 source、prior 有无、样本难度可能缠在一起了。若不拆开，后面很难解释模型到底是在学 no-prior 初始化，还是只是在不同数据源上表现不同。

**要做什么**

* 固定 object-level split，不允许后续漂移。
* 让 with-prior / no-prior 在主要 source 内都出现。
* 固定 material-sensitive heldout 和 OOD holdout。

**出门标准**

* train / val / test 的 split 说明里能回答：
  “按 source、generator、prior、bucket 分布是否平衡？”

---

### 4. 先做一轮“可训练子集”审计，再开下一轮训练

现在不是数据越多越好，而是先确认哪些样本真的适合当前任务定义。协议里已经有强监督池、辅助池、stress bank、OOD bank，但主训练前仍需要一个“能证明方法有效”的子集。

**要做什么**

* 从 Pool-A 里先抽一版高质量、非同源、监督可信的 stage1 subset。
* 把低价值 pseudo-GT 标成低权重或排除。
* 先在这个 subset 上重跑 round2。

**出门标准**

* 新 round2 的 baseline MAE 不再是 `0.0`
* refined 相对 baseline 的变化能解释为真实提升，而不是复制 prior

---

## P1：paper 要站住，下一步优先补这些

### 5. 把评测从“工程闭环”升级为“论文裁决型”

现在训练曲线、HTML、validation panel、W&B、round analysis 都很完整，但论文还需要更硬的指标层。协议里已经列了 UV MAE、view-space MAE、rendered PSNR/SSIM/LPIPS、metal/non-metal F1/AUROC、failure-tag reduction、GLB validity、runtime/memory。

**优先补**

* `metal/non-metal F1 / AUROC`
* `failure-tag reduction`
* rendered 指标
* runtime / memory
* 按 `category / bucket / generator_id / source / license_bucket` 的分层统计

**目标**

* 不再只靠 UV MAE 说话，而是形成
  **数值 RM + 渲染效果 + failure taxonomy** 三层证据链。

---

### 6. 跑完整 ablation，而不只是 baseline vs refined

协议里的 ablation 设计已经够用了：`scalar broadcast`、`prior smoothing`、`no-view refiner`、`no-prior refiner`、`no-residual refiner`、`Ours full` 等。现在缺的是把它们真正跑成矩阵。

**优先顺序**

* `scalar broadcast`
* `no-prior refiner`
* `no-residual refiner`
* `no-view refiner`
* `Ours full`

**目标**

* 能回答“到底是哪一个模块带来了改进”。

---

### 7. 让 failure taxonomy 真正反哺采样与训练

现在你已经有很好的 failure tags 和 top failure 视图，但训练侧还没明显体现按这些 failure 做 hard mining 或 curriculum。协议和现有分析都说明，重点失败集中在 metal/nonmetal confusion、boundary bleed、local highlight misread。

**要做什么**

* 对这些 tag 做 oversampling 或 curriculum。
* 训练日志里单独看这些 failure bucket 的下降趋势。
* 验证集固定输出 tag-wise 改善。

**目标**

* 方法改进能对上你最在意的错误模式。

---

### 8. 至少接一个真实第二上游，验证 generator-aware 不是空架子

现在 `generator_id` 轴、过滤、采样平衡、分组指标都接上了，这很好；但严格说还只是 generator-aware infrastructure，距离“generator-agnostic 方法”还差一个真实第二 adapter 的实证。

**要做什么**

* 先选一个：SPAR3D 或 Hunyuan3D。
* 走通 `G -> C -> R -> eval` 最小闭环。
* 至少补一组 cross-generator 测试。

**目标**

* 让“多上游生成器可扩展”从协议变成结果。

---

## P2：不是当前阻塞，但后面要补齐

### 9. 补完整的 license-aware 发布链路

现在数据侧 `license_bucket` 管得已经比较细了，尤其是 3D-FUTURE、Objaverse permissive filter、Pool 划分都做得不错。下一步要补的是训练产物、checkpoint、报告、导出 demo 也继承这层隔离。

**建议**

* checkpoint 写入训练所含 `license_bucket`
* report 默认显示 bucket 构成
* demo/export 只允许可展示桶进入公开产物

---

### 10. 做一次真正的 refined GLB 导出审计

协议里已经写了 refined GLB 需要显式写入 `roughnessTexture`、`metallicTexture`，Gradio 也保留双阶段流程；但这部分还需要一次稳定的小样本全链路验证。

**要做什么**

* 抽 5–10 个对象
* 完整跑 `SF3D -> R -> refined GLB`
* 检查 viewer 兼容性、材质表现、before/after 一致性

---

### 11. 把 Stage 2 控制在最小必要增量

你现在的 Stage 2 设计很全：Pool-B/C/E/D、highlight consistency、part-material auxiliary、material bank、stress bank 都准备好了。问题不是“少”，而是“容易同时推进太多”。

**建议顺序**

* 先上和当前主 failure 最贴的两个：

  * highlight consistency
  * part/material auxiliary
* 其余后置，不要一口气全开

---

## 最后给你一个执行顺序

### 先做

**P0-1 → P0-2 → P0-3 → P0-4**

### 然后做

**P1-5 → P1-6 → P1-7 → P1-8**

### 最后补

**P2-9 → P2-10 → P2-11**

一句话概括：

> **P0 解决“能不能证明方法有效”，
> P1 解决“论文能不能站住”，
> P2 解决“系统能不能成熟交付”。**

---

## 2026-04-19 数据集监督治理续跑记录

### 已修正的顶会级风险点

* **旧 manifest 尺寸不一致会打挂 watcher**：`target_prior_identity_from_paths` 已统一把 roughness / metallic / prior / confidence resize 到同一 target grid，旧数据现在会被安全审计和降级，不会导致 auto-select 退出。
* **auto-select 缓存无审计版本**：manifest fingerprint 已加入 `AUDIT_SCHEMA_VERSION`，数据治理逻辑升级后不会继续相信旧缓存摘要。
* **单个坏候选会拖垮自动化**：`select_material_refine_best_manifest.py` 已改为 per-candidate 审计容错，坏 manifest 只记录 `audit_failed:*` blocker，不再中断整个选择器。
* **候选选择效率低**：auto-select 候选改为按 mtime 倒序，优先审正在产出的 reworked manifest，旧 8k manifest 只作为 fallback。
* **小样本 split 被 material holdout 吃空**：`build_material_refine_paper_stage1_subset.py` 已保护 train/val/test 基本分布，并把 `paper_train_records=0`、`paper_val_iid_records=0`、`paper_test_records=0` 纳入 readiness blocker。
* **watcher 无候选即退出**：`run_material_refine_paper_stage1_pipeline.py` 已改为无满足 min-records 候选时写 `readiness_state` 并继续等待下一轮。
* **半成品 render buffer 被误判完整**：`render_bundle_complete` 已从只检查 `rgba/roughness/metallic` 升级为检查 `rgba/mask/depth/normal/position/uv/visibility/roughness/metallic`。

### 当前正在跑的任务

* 非平凡候选数据重刷：
  * tmux: `sf3d_rework_abo160_gpu0`
  * input: `output/material_refine_paper/reworked_candidates/abo_160/abo_160_input_manifest.json`
  * output: `output/material_refine_paper/reworked_candidates/abo_160_prepared/full/canonical_manifest_partial.json`
  * 配置：160 个 local ABO、`atlas=512`、`render=128`、`cycles=2`、GPU0、`parallel_workers=3`
* auto-select watcher：
  * tmux: `sf3d_material_refine_paper_stage1_select`
  * status: `output/material_refine_pipeline_20260418T091559Z/paper_stage1_pipeline_auto_select/logs/sf3d_material_refine_paper_stage1_select.status.json`
  * 当前只盯 reworked candidate glob，等 `canonical_manifest_partial/full` 达到门槛后自动进入 readiness -> subset -> train -> eval。

### 顶会视角仍需补齐的不足

* 当前 `abo_160` 是快速解锁 non-trivial target 的候选，不应作为最终论文数据版本；最终版至少需要更高 render resolution / samples，并覆盖高光材质配额。
* Pool-B / Pool-F 真实光照验证还没有进入最终 benchmark，论文主结果必须补真实受控光照和自然 HDR OOD。
* 当前 fixed split 已阻止对象泄漏，但材质族配额、source/license 报表还需要在最终 manifest summary 中强制验收。
* Stage 1 继续保持 `view_consistency_weight=0.0` 是合理的；只有在 strict complete rate 稳定后再进入 Stage 2。

---

## 2026-04-19 View/Light 协议与两天级长跑数据生产

### 已升级的数据协议

* `prepare_material_refine_dataset.py` 已从固定 3-view 改成可配置 `view_light_protocol`：
  * `canonical_triplet`: 旧兼容协议，3 views。
  * `standard_12`: 6 camera poses × 2 HDRI。
  * `stress_24`: 8 camera poses × 3 HDRI。
  * `production_32`: 8 camera poses × 4 HDRI。
* 每个 view 会写入 `camera_label / lighting_asset_id / lighting_stratum / lighting_license_bucket / view_light_protocol`，对象 manifest 会写入：
  * `view_light_protocol`
  * `view_count`
  * `lighting_bank_id`
  * `hdri_asset_ids`
* 当前长跑使用 `stress_24`：每对象 24 个 view/light 条件，渲染 `rgba/mask/depth/normal/position/uv/visibility/roughness/metallic`，用于 object-level UV RM target bake。

### 新增自动化脚本

* `scripts/build_material_refine_longrun_manifest.py`
  * 从多个 manifest 构建长跑输入。
  * 固定 object-level SHA1 split。
  * 按 source 设置 `paper_main / auxiliary_upgrade_queue`。
  * 可自动写 shard manifest。
* `scripts/evaluate_material_refine_dataset_quality.py`
  * 汇总 paper readiness、target identity、view ready、strict complete、license/source/material/view-light/HDRI 覆盖。
  * 检查 object split leakage。
* `scripts/launch_material_refine_longrun_dataset_tmux.sh`
  * 自动构建 longrun manifest。
  * 按 GPU shard 启动多 tmux 预处理。
  * 默认 `stress_24 / 768 atlas / 256 render / 8 samples / 2 workers per GPU`。

### 当前已启动的长跑任务

* HDRI 扩容：
  * tmux: `sf3d_polyhaven_hdri_900`
  * 硬门槛：HDRI bank 至少 900 个，本轮不再接受 60/240 作为 production longrun 光照库。
  * 输出：`output/highlight_pool_a_8k/aux_sources/polyhaven_hdri_bank.json` 与 `polyhaven_hdri_bank_900.json`。
  * `stage_highlight_aux_sources.py` 已改为过采样候选、metadata retry、单资产失败跳过并记录 `metadata_failed`，避免 PolyHaven 单个 timeout 拖垮 900 bank。
  * 当前 bank 已落盘 964 条 PolyHaven CC0 HDRI 记录，其中 963 个本地 `.hdr` 已存在，满足至少 900 个 HDRI 的硬门槛。
* 长跑数据生产：
  * 旧 root: `output/material_refine_longrun_stress24_20260419T124126Z` 已暂停，不作为 HDRI900 版本。
  * 新 HDRI900 root: `output/material_refine_longrun_stress24_hdri900_20260419T134158Z`。
  * manifest: `longrun_input_manifest.json`
  * 目标总对象：至少 706 起步，后续继续扩到 8k+；其中通过 paper gate 的对象才进入论文主池。
  * paper_main: 500 ABO
  * auxiliary_upgrade_queue: 30 3D-FUTURE + 176 Objaverse GitHub/LFS increment
  * object split: train 627 / val 39 / test 40
  * tmux:
    * `sf3d_longrun_material_refine_shard0_gpu0`
    * `sf3d_longrun_material_refine_shard1_gpu1`
* auto-select watcher 已重启，glob 指向：
  * `output/material_refine_longrun_stress24_hdri900_*/**/canonical_manifest*.json`
  * `output/material_refine_paper/reworked_candidates/**/canonical_manifest*.json`

### 当前确认

* HDRI900 longrun 固定使用 `--min-hdri-count 900`，不满足时预处理脚本会直接拒绝启动，避免低光照覆盖版本混入最终数据。
* 新 views.json 会产生 24 条 view/light 记录，例如：
  * `front_mid__00_brown_photostudio_07`
  * `view_light_protocol=stress_24`
* 每个对象使用 object-level split；同一对象的所有 HDRI、视角、材质变体必须同 split，禁止 view/HDRI 级泄漏。

### 2026-04-19 HDRI900 长跑修正

* 旧 `manifest_selection_smoke/best_manifest_selection.json` 选中的 530 records manifest 仍为 `copied_from_prior`，只能作为 engineering smoke，不能进入 paper-stage 主训练。
* HDRI900 长跑初始 partial manifest 暴露出一个检查逻辑 bug：`refresh_material_refine_partial_manifest.py` 仍写死旧 3-view 名称，导致 `stress_24` bundle 被误判 `missing_bundle_files`。
* 已修正：
  * `refresh_material_refine_partial_manifest.py` 动态读取每个 bundle 的 `views.json`，按实际 view/light 协议检查 24 views。
  * `prepare_material_refine_dataset.py::render_bundle_complete` 只要求渲染器直接产出的基础 buffers，`mask/depth/visibility` 由 finalize 阶段派生。
* 已重启 HDRI900 两个 shard：
  * `sf3d_longrun_material_refine_shard0_gpu0`
  * `sf3d_longrun_material_refine_shard1_gpu1`
* 已新增独立 partial monitor：
  * tmux: `sf3d_longrun_hdri900_partial_monitor`
  * 输出：`canonical_manifest_monitor_partial.json`
  * 作用：每 120 秒刷新一次动态 partial manifest，让 auto-select watcher 能尽早接住第一批 non-trivial target。

### 2026-04-20 Paper-stage 门禁解锁记录

* 已新增 `scripts/merge_material_refine_partial_manifests.py`：
  * 合并多个 shard 的 `canonical_manifest_monitor_partial.json`。
  * 按 `canonical_object_id/object_id` 去重。
  * 对已完成对象做稳定 object-level `train/val/test` 重切分，避免单 shard 或完成顺序导致 `paper_val_iid_records=0`。
* 已新增 merged monitor：
  * tmux: `sf3d_longrun_hdri900_merged_monitor`
  * 输出：`output/material_refine_longrun_stress24_hdri900_20260419T134158Z/canonical_manifest_monitor_merged.json`
  * 每 120 秒刷新合并 manifest，持续接住两个 shard 后续增长。
* 当前合并 manifest 已通过 paper-stage preflight：
  * `paper_stage_ready=true`
  * `paper_stage_eligible_records=209+`，随 monitor 增长，最近一次 watcher 为 `210`
  * `stage1_subset_records=210`
  * `readiness_blockers=[]`
  * `target_prior_identity_rate≈0.11`
  * `effective_view_supervision_record_rate=1.0`
  * strict buffer validation：`rgba/mask/depth/normal/position/uv/visibility/roughness/metallic` 均为 `100%`
* 固定 stage1 subset 已生成：
  * `output/material_refine_pipeline_20260418T091559Z/paper_stage1_pipeline_auto_select/readiness/stage1_subset/paper_stage1_subset_manifest.json`
  * `output/material_refine_pipeline_20260418T091559Z/paper_stage1_pipeline_auto_select/readiness/stage1_subset/paper_stage1_split_v1.json`
  * `output/material_refine_pipeline_20260418T091559Z/paper_stage1_pipeline_auto_select/readiness/stage1_subset/paper_stage1_readiness_summary.json`
* 注意：当前解锁的是 paper-stage P0/P1 数据门禁，不代表最终顶会完整实验已完成；Pool-B/Pool-F 真实光照 benchmark 和更大 8k+ 生产语料仍需继续扩充。

### 2026-04-20 Pool-B / Pool-F 与 8k+ 生产扩展

* 已新增 `scripts/stage_material_refine_real_benchmarks.py`：
  * Pool-B：OpenIllumination 作为 `controlled_real_lighting_aux_eval`，license 标记 `cc_by_4_0`，只做辅助/验证，不混入 Pool-A 主训练。
  * Pool-F：Stanford-ORB 作为 `real_world_eval_holdout`，先下载 `ground_truth.tar.gz` + `blender_LDR.tar.gz`，保留 benchmark-only 标记。
  * OLATverse / ICTPolarReal 当前仅登记 `access_probe_only`，在 license/download endpoint 未确认前不进入训练或正式 benchmark manifest。
* 已启动真实 benchmark staging：
  * tmux: `sf3d_real_benchmark_openillumination`
  * tmux: `sf3d_real_benchmark_stanford_orb`
  * 输出根：`output/material_refine_real_benchmarks/`
* Stanford-ORB 下载已开始落盘：
  * `output/material_refine_real_benchmarks/stanford_orb_pool_f/stanford_orb/ground_truth.tar.gz`
  * 后续同进程继续 `blender_LDR.tar.gz`
* OpenIllumination 下载已切到 `curl` + proxy 路径，避免 `wget` 直连 HuggingFace 卡住；当前正在拉取选定 object 的 metadata/mask/thumbnail pilot 文件。
* 已增强 `scripts/launch_material_refine_longrun_dataset_tmux.sh`：
  * 支持 `LONGRUN_INPUT_MANIFESTS`、`PAPER_MAIN_SOURCES`、`AUXILIARY_SOURCES` 环境变量。
  * 现在可直接吃 `output/highlight_pool_a_8k/material_refine_manifest_pool_a_8k.json`。
* 已启动 8k+ 生产等待队列：
  * tmux: `sf3d_poola8k_expansion_waitlaunch`
  * 候选规模 dry-run：`9700` records = `500` ABO paper-main + `7700` 3D-FUTURE auxiliary + `1500` Objaverse auxiliary。
  * 队列策略：等待当前 HDRI900 seed longrun shard 完成后自动启动 `stress_24 / HDRI>=900 / atlas768 / render256 / cycles8 / 4 shards / 2 GPUs`，避免与当前 paper-stage 数据生成抢显存。

### 2026-04-20 继续推进检查记录

* 当前 paper-stage 数据门禁已确认从 smoke 阶段推进到 P0/P1 可用状态：
  * auto-select/readiness：`paper_stage_ready=true`，`stage1_subset_records=210`，`readiness_blockers=[]`。
  * direct pipeline 新一轮 subset：`271` 条 `paper_pseudo`，`paper_stage_eligible_records=271`，`target_prior_identity_rate=0.0`，`6504` 个 view/light buffer 全字段完整。
* direct train 链路失败原因已定位为资源竞争，不是数据 blocker：
  * GPU0 上 Blender 渲染进程继续占用约 `3.36G + 1.97G`，训练自身约 `25.8G`，最终 CUDA OOM。
  * 数据结论保持：manifest/preflight 可用；训练 retry 应在数据渲染空窗或更低 batch/memory 配置下执行。
* 已新增 Pool-F benchmark 索引脚本：
  * `scripts/index_stanford_orb_benchmark.py`
  * 功能：解包 `ground_truth.tar.gz` 与 `blender_LDR.tar.gz`，扫描 RGB/HDR/sidecar/environment 文件，生成 `benchmark_only / paper_test_real_lighting` manifest。
  * 输出：`output/material_refine_real_benchmarks/stanford_orb_pool_f/stanford_orb_benchmark_manifest.json`
* 已启动 Stanford-ORB 解包/索引后台进程：
  * tmux: `sf3d_stanford_orb_index_extract`
  * 当前状态：16GB tar 已下载完成，解包目录开始增长；完成后自动写 Pool-F benchmark manifest。
* 已追加 Stanford-ORB 解包后重索引进程：
  * tmux: `sf3d_stanford_orb_reindex_after_extract`
  * 原因：解包后发现 `mesh_blender/texture_*.png` 等材质贴图会被宽松扫描误当成 image candidate；已在索引脚本中改为 `material_sidecar`，重索引进程会等两个 archive marker 完成后覆盖最终 manifest。
* 已增强 8k+ longrun launcher：
  * `scripts/launch_material_refine_longrun_dataset_tmux.sh` 现在默认启动 merged monitor。
  * 8k+ 队列正式启动后会自动生成 `${OUTPUT_ROOT}/canonical_manifest_monitor_merged.json`，让 auto-select/readiness 不必等所有 shard 完成。

### 2026-04-20 Paper-stage round1 训练与评测修复记录

* `sf3d` 虚拟环境内已补齐 `rg` 可执行入口：
  * `/home/ubuntu/ssd_work/conda_envs/sf3d/bin/rg -> /home/ubuntu/.vscode-server/extensions/openai.chatgpt-26.415.20818-linux-x64/bin/linux-x86_64/rg`
  * 终端可直接使用 `tail -f ... | rg '\[train\]|\[val\]|\[epoch\]'` 查看训练进度。
* `scripts/eval_material_refiner.py` 已修复 `view_rgba_paths` 兼容问题：
  * 之前 manifest 中 `view_rgba_paths[view_name]` 为 `{"rgba": "/abs/path/rgba.png"}`，旧逻辑将整个 dict 传入 `Path(...)`，导致 test eval 崩溃。
  * 现已新增 `resolve_rgba_path()`，兼容 `str/pathlike/dict` 结构，paper-stage eval 可正常完成。
* `paper_stage1_pipeline_gpu0_live_step3_20260420` 当前关键产物已落地：
  * train: `output/material_refine_paper/paper_stage1_pipeline_gpu0_live_step3_20260420/train_stage1_main/`
  * eval: `output/material_refine_paper/paper_stage1_pipeline_gpu0_live_step3_20260420/eval_stage1_test/`
  * round analysis: `output/material_refine_paper/paper_stage1_pipeline_gpu0_live_step3_20260420/round_analysis/`
* round1 训练结果：
  * `best_epoch=57`
  * `best_val_uv_total=0.045929`
  * `last_val_uv_total=0.046004`
  * `epochs=60` 全部完成
  * best checkpoint: `train_stage1_main/best.pt`
* round1 test eval 已完成：
  * `rows=1176`
  * `effective_view_supervision_rate=1.0`
  * `baseline_total_mae=0.116709`
  * `refined_total_mae=0.042506`
  * `avg_improvement_total=0.074203`
* 当前 failure tag 改善情况：
  * `over_smoothing`: `21 -> 12`，reduction=`9`
  * `metal_nonmetal_confusion`: `921 -> 921`，无改善
  * `local_highlight_misread`: `170 -> 170`，无改善
  * `boundary_bleed`: `90 -> 90`，无改善
* 当前可视化/报告链路已恢复：
  * `material_attribute_comparison.html`
  * `material_attribute_summary.json`
  * `validation_comparison_panels/validation_comparison_index.html`
  * `round_analysis/round_analysis.html`
* W&B 离线 run 已生成并开始同步：
  * train main: `offline-run-20260420_083829-31l17p4t`
  * eval test: `offline-run-20260420_102119-qa5r05l1`
  * validation panels: `offline-run-20260420_102349-7tzg3nqq`
  * round summary: `offline-run-20260420_102419-6yyyvihc`

### 2026-04-20 数据分布与条件利用补强

* 已确认当前 paper-stage subset 的结构性风险：
  * `source_counts={"ABO_locked_core": 346}`
  * `prior_counts={"true": 346, "false": 0}`
  * `target_source_type_counts={"pseudo_from_multiview": 346}`
  * `target_quality_tier_counts={"paper_pseudo": 346}`
  * `material_family_counts={"glossy_non_metal": 346}`
  * 分布警告仍包括：缺 `metal_dominant / glass_metal / mixed_thin_boundary`、缺 `no_prior`、缺第二 generator、缺 `paper_strong`。
* 已修正审计命名/统计：
  * 保留兼容字段 `target_prior_identity_rate`，但新增更清晰的 `target_prior_similarity_mean` 和 `target_prior_distance_mean`。
  * `material_family` 不再从 noisy/pseudo metallic 图强行把旧 `glossy_non_metal` 改成 metal，避免“假补分布”。
  * `category_bucket` 会继续按 source + prior fallback，避免裸 `ABO_locked_core` 标签。
* 已改训练数据利用方式：
  * `CanonicalMaterialDataset` 支持 `max_views_per_sample / min_hard_views_per_sample / randomize_view_subset`。
  * paper-stage train 配置现为每样本随机取 `8` 个 view/light 条件，并保底 `3` 个 hard views。
  * hard views 包括 `grazing / thin_boundary / close / edge / highlight` 等标签；采样验证已看到 24 条件中随机抽 8 条且包含多个 hard views。
* 已改 view fusion：
  * `UVFeatureFusion` 不再把所有 view 等权平均；现在会读取 `view_importance`，对 grazing/thin-boundary/highlight 视角加权 scatter 到 UV。
  * 这先解决“hard cue 被普通视角冲淡”的主要问题，后续可继续做显式 camera/light embedding。
* 已启用 failure-aware 训练采样：
  * `train_failure_tag_metadata_key=failure_tags`
  * `metal_nonmetal_confusion=3.0`
  * `boundary_bleed=2.5`
  * `local_highlight_misread=2.5`
  * `paper_strong` 样本权重高于 `paper_pseudo`。
* 已补 object-level 评测口径：
  * `eval_material_refiner.py` 现在额外输出 `object_metrics.json`。
  * `summary.json` 新增 `object_level` 聚合，避免论文表只按 view rows 统计导致对象权重被 24 条件放大。
* 已增强 8k+ longrun 打标与排序：
  * `build_material_refine_longrun_manifest.py` 会从 notes/path 推断 `material_family / thin_boundary_flag / failure_tags / sampling_bucket`。
  * 8k+ 本地候选重建后分布为：
    * `metal_dominant=1460`
    * `glass_metal=31`
    * `mixed_thin_boundary=4730`
    * `glossy_non_metal=3469`
    * `ceramic_glazed_lacquer=10`
    * `no_prior=9200`
  * 8k+ 等待脚本已改成 `PREFER_PAPER_MAIN_FIRST=false`，优先按 material family 推进，不再早期只产 ABO。
* 已立即启动 GPU0 distribution-priority 长跑，不等 GPU1 seed shard 完成：
  * tmux: `sf3d_distribution_priority_material_refine_shard0_gpu0`
  * monitor: `sf3d_distribution_priority_material_refine_merged_monitor`
  * root: `output/material_refine_longrun_distribution_priority_hdri900_20260420T125737Z`
  * 规模：`2400` 条 3D-FUTURE no-prior 辅助晋升候选。
  * 分布：`metal_dominant=1460`、`mixed_thin_boundary=909`、`glass_metal=31`。
  * 渲染配置：`stress_24 / HDRI>=900 / atlas768 / render256 / cycles8 / GPU0 / workers=1`。
  * 该进程已确认进入 Blender 渲染，GPU0 显存开始占用。

### 2026-04-20 GPU0-only 七天级数据生产

* 已按“GPU0 长跑、GPU1 空载”重新收敛调度：
  * 已停止 `sf3d_longrun_material_refine_shard1_gpu1`，GPU1 当前约 `15MB / 0%`。
  * 已停止短量级 `sf3d_distribution_priority_material_refine_shard0_gpu0` 与旧 `sf3d_poola8k_expansion_waitlaunch`，避免多队列抢 GPU0 或后续误用 GPU1。
  * 旧 8k 等待脚本已改为 GPU0-only：`GPU_LIST=0`、`SHARDS=1`，并切到 `production_32`。
* 已增强 `scripts/launch_material_refine_longrun_dataset_tmux.sh`：
  * 新增 `MAX_HDRI_LIGHTS` 和 `HDRI_SELECTION_OFFSET` 环境变量。
  * prepare 阶段现在可显式传入 `--max-hdri-lights` 与 `--hdri-selection-offset`，用于构建不同 HDRI 组合 bank。
* 当前 HDRI bank 状态：
  * `output/highlight_pool_a_8k/aux_sources/polyhaven_hdri_bank.json`
  * `964` 条记录，`963` 个本地 `.hdr`，license 均为 `cc0`。
* 已启动 GPU0-only 七天级主生产任务：
  * tmux: `sf3d_gpu0_sevenday_material_refine_shard0_gpu0`
  * monitor: `sf3d_gpu0_sevenday_material_refine_merged_monitor`
  * root: `output/material_refine_longrun_gpu0_sevenday_hdri900_20260420T132250Z`
  * total records: `9700`
  * source mix: `7700` 3D-FUTURE + `1500` Objaverse cached strict increment + `500` ABO
  * no-prior records: `9200`
  * material mix:
    * `metal_dominant=1460`
    * `glass_metal=31`
    * `mixed_thin_boundary=4730`
    * `ceramic_glazed_lacquer=10`
    * `glossy_non_metal=3469`
  * render protocol: `production_32`
  * per object: `8` camera poses × `4` HDRIs = `32` view/light conditions
  * HDRI requirement: `--min-hdri-count 900`
  * HDRI combination controls: `--max-hdri-lights 4 --hdri-selection-offset 700`
  * render quality: `atlas=1024 / render=384 / cycles=16`
  * GPU: `cuda-device-index=0`
  * workers: `1`
  * partial refresh: every `5` records
* 已确认七天级任务进入 Blender 渲染：
  * 当前日志显示 `production_32` 下的 `misty_pines_1k.hdr` 等 HDRI 被加载并渲染。
  * 当前 GPU 状态：GPU0 有渲染占用，GPU1 保持空载。
* 已另起不占 GPU 的外部源下载/筛选进程：
  * tmux: `sf3d_aux_objaverse_strict_download`
  * root: `output/material_refine_aux_downloads/objaverse_strict_20260420T132336Z`
  * target: `3000` Objaverse-XL strict highlight candidates
  * allowed sources: `sketchfab,smithsonian`
  * 作用：持续下载/补源；不参与 GPU1 渲染，不混入 Pool-A，后续经 license/import/target gate 后再晋升。

### 2026-04-20 GPU0 稳定满载重启

* 因 `sf3d_gpu0_sevenday_*` 初始 workers 太低，GPU0 未充分利用，已改为 GPU0-only 高并发长跑。
* 18-worker 试探配置曾达到更激进并发，但 Blender/Cycles 出现 CUDA context/OOM，说明超过稳定点；已切到稳定满载配置，避免大量失败样本。
* 当前有效长跑任务：
  * tmux shards: `sf3d_longrun_material_refine_shard0_gpu0` 到 `sf3d_longrun_material_refine_shard4_gpu0`
  * root: `output/material_refine_longrun_gpu0_stablefill_hdri900_20260420T133139Z`
  * total records: `9700`
  * source mix: `7700` 3D-FUTURE + `1500` Objaverse cached strict increment + `500` ABO
  * render protocol: `production_32`
  * per object: `8` camera poses × `4` HDRIs = `32` view/light conditions
  * HDRI requirement: `--min-hdri-count 900`
  * HDRI combination controls: `--max-hdri-lights 4 --hdri-selection-offset 900`
  * render quality: `atlas=1024 / render=384 / cycles=16`
  * GPU: `cuda-device-index=0`
  * concurrency: `5` shards × `3` workers = `15` concurrent Blender workers
  * latest check: GPU0 about `31361MB / 32607MB`, GPU1 about `15MB / 32607MB`
  * latest log check: stablefill root has no current `Out of memory / CUDA context / RuntimeError` matches after restart.
* 当前下载/补源任务已重新挂起：
  * tmux: `sf3d_aux_objaverse_strict_download`
  * root: `output/material_refine_aux_downloads/objaverse_strict_20260420T133039Z`
  * target: `3000` Objaverse-XL strict highlight candidates
  * processes: `12`
  * 作用：CPU/network 后台下载，不占 GPU1；下载慢不阻塞当前 GPU0 使用，因为主渲染已优先消费本地 3D-FUTURE/ABO/Objaverse cached manifest。

### 2026-04-21 GPU0 数据生产恢复

* 检查时两张 GPU 均为空载，上一轮 `stablefill` shard 已结束，不是仍在后台工作。
* 上一轮失败/低产原因：
  * `15` concurrent Blender 在长时间运行中仍触发大量 CUDA OOM / CUDA context OOM。
  * `9700` selected records 中最终只准备出约 `129` 条完整记录，`skipped_reasons` 主要是 `CalledProcessError`。
  * 部分旧 prepared cache 会把输入 manifest 的 `material_family` 覆盖回 `glossy_non_metal`，导致 metal/mixed 分布在输出侧失真。
* 已修复 resume 时的材质分布标签覆盖：
  * 文件：`scripts/refresh_material_refine_partial_manifest.py`
  * 逻辑：读取 cached prepared record 后，会重新应用当前输入 manifest 的 `material_family / highlight_material_class / thin_boundary_flag / failure_tags / sampling_bucket / supervision_role / license_bucket / default_split`。
* 已重启 GPU0-only 稳定长跑：
  * root: `output/material_refine_longrun_gpu0_resume_hdri900_20260421T014732Z`
  * tmux shards: `sf3d_longrun_material_refine_shard0_gpu0` 到 `sf3d_longrun_material_refine_shard11_gpu0`
  * monitor: `sf3d_longrun_material_refine_merged_monitor`
  * total records: `9700`
  * source mix: `7700` 3D-FUTURE + `1500` Objaverse cached strict increment + `500` ABO
  * render protocol: `production_32`
  * per object: `8` camera poses × `4` HDRIs = `32` view/light conditions
  * HDRI requirement: `--min-hdri-count 900`
  * HDRI offset: `--hdri-selection-offset 1100`
  * render quality: `atlas=1024 / render=320 / cycles=8`
  * concurrency: `12` shards × `1` worker = `12` concurrent Blender workers
  * latest check: GPU0 about `22-26GB / 32GB`, GPU1 about `15MB / 32GB`
  * latest error scan: no current OOM / CUDA context / RuntimeError matches in the new resume logs.
* 第一批 bundle 已确认在写入：
  * 示例：`prepared_shard_0/full/canonical_bundle/3dfuture_61b03647-d3d0-4feb-a3e7-1a96570bbf17`
  * 当前早期状态约 `8/32` view directories，说明 manifest 计数暂时为 0 是因为对象尚未完成完整 `production_32`，不是进程卡死。
* Objaverse 下载已改为自动重试后台任务：
  * tmux: `sf3d_aux_objaverse_strict_download`
  * root: `output/material_refine_aux_downloads/objaverse_strict_retry_20260421T015037Z`
  * 失败后每 `300s` 重试一次，避免远端 `RemoteDisconnected` 直接让下载通道停掉。

### 2026-04-21 连续数据工厂配置

* 已把数据扩充从一次性命令升级为持续 factory：
  * config: `configs/material_refine_dataset_factory_gpu0.json`
  * script: `scripts/run_material_refine_dataset_factory.py`
  * docs: `docs/material_refine_continuous_dataset_factory.md`
  * state: `output/material_refine_dataset_factory/factory_state.json`
  * tmux: `sf3d_material_refine_dataset_factory_gpu0`
* 数据源规划已固化进 config：
  * Pool-A seed: `ABO_locked_core`
  * Pool-A auxiliary upgrade: `3D-FUTURE_highlight_local_8k`
  * Pool-A auxiliary upgrade: `Objaverse-XL_cached_strict_increment`
  * Pool-D: `PolyHaven_HDRI`，最低 `900`，扩展目标 `1200`
  * Pool-E: `PolyHaven_Materials`，目标 `1000`
  * Pool-B/Pool-F: OpenIllumination / Stanford-ORB，仅作 real benchmark / OOD sidecar
* factory 已支持：
  * 启动并维护下载 retry sessions。
  * 把 Objaverse raw increment manifest 转成 canonical manifest，再作为后续 longrun 输入。
  * 通过 manifest glob 自动吸收新下载/转换出的 Objaverse 增量。
  * 只有 active shard 会阻止新一轮 GPU0 longrun；孤立 merged monitor 不再阻塞调度。
  * 每轮自动运行 dataset quality audit，状态落盘到 `factory_state.json`。
* 当前已实际启动新一轮 GPU0-only factory longrun：
  * root: `output/material_refine_dataset_factory/longrun_gpu0_20260421T034417Z`
  * shards: `sf3d_longrun_material_refine_shard0_gpu0` 到 `sf3d_longrun_material_refine_shard11_gpu0`
  * monitor: `sf3d_longrun_material_refine_merged_monitor`
  * GPU policy: GPU0 only，GPU1 保持空载
  * latest check: GPU0 约 `20.9GB / 32GB`，GPU1 约 `15MB / 32GB`
* 当前下载/补源 sessions：
  * `sf3d_factory_polyhaven_hdri`
  * `sf3d_factory_objaverse_cached_increment`
  * `sf3d_factory_objaverse_strict_full_retry`
  * `sf3d_factory_polyhaven_material_bank`
  * `sf3d_aux_objaverse_strict_download`

### 2026-04-21 target/readiness 风险修复

* 已修复 `target_prior_identity_rate=1.000` 的关键代码根因：
  * `scripts/prepare_material_refine_dataset.py` 不再只把字面值 `supervision_tier == "strong"` 当强监督。
  * `strong_A_ready_locked` / `A_ready` / `paper_main` 现在都会生成 non-trivial degraded prior。
  * 对能成功 bake 出 `gt_render_baked` 或 `pseudo_from_multiview` 的辅助样本，也默认生成 non-trivial prior，避免辅助池继续污染 identity 统计。
* 已修复 factory audit 反复审旧 stablefill manifest 的问题：
  * `scripts/run_material_refine_dataset_factory.py` 现在只从 `output/material_refine_dataset_factory/...` 选择新 factory manifest。
  * audit 候选必须达到 `min_manifest_records_for_audit=8`，没有新记录时返回 `no_factory_manifest_with_min_records`，不再反复输出旧的 `target_prior_identity_rate=1.000 / eligible=0`。
* 已修复 merged monitor 看不到新 partial 的问题：
  * `scripts/monitor_material_refine_merged_manifest.sh` 同时识别 `canonical_manifest_partial.json` 和 `canonical_manifest_monitor_partial.json`。
* 已优化下载代理策略：
  * download retry wrapper 每次尝试前会在 `env` proxy 与 `direct` 之间测速。
  * PolyHaven 当前选择 `direct`，Objaverse cached 当前选择 `http://127.0.0.1:51081`。
  * Objaverse strict full retry 已继续推进到 annotation parquet 下载阶段。
* 已重建本地 Pool-A manifest：
  * `output/highlight_pool_a_8k/material_refine_manifest_pool_a_8k.json`
  * 总量从 `8200` 提到 `12000`，全部 local assets。
  * 当前可用材质：`glass_metal=1800`，`metal_dominant=1959`，`mixed_thin_boundary=1566`，`ceramic_glazed_lacquer=345`，`glossy_non_metal=5830`，ABO core `500`。
  * 结论：玻璃/金属已明显补足；陶瓷是本地 3D-FUTURE 明确短板，后续依赖 Objaverse 6000 strict increment 和材料先验增强继续补。
* 已重启 GPU0 数据进程：
  * production root: `output/material_refine_dataset_factory/longrun_gpu0_20260421T122652Z`
  * production: `8` shards，`production_32`，`atlas=1024 / render=320 / cycles=8`
  * paper unlock root: `output/material_refine_dataset_factory/paper_unlock_gpu0_20260421T123133Z`
  * paper unlock: `160` ABO paper_main，`4` shards，`standard_12`，`atlas=768 / render=256 / cycles=8`
  * 当前 GPU0-only 数据进程已启动；GPU1 上的占用来自已有 eval session，不是新数据工厂进程。

### 2026-04-21 7-day 数据工厂守护器

* 已新增并启动 7 天数据侧 supervisor，目标是避免下载、渲染、合并、审计、验证任一环节静默停止：
  * config: `configs/material_refine_dataset_supervisor_7day_gpu0.json`
  * runner: `scripts/run_material_refine_dataset_supervisor_7day.py`
  * launcher: `scripts/launch_material_refine_dataset_supervisor_7day_tmux.sh`
  * tmux: `sf3d_material_refine_dataset_7day_supervisor`
  * log: `output/material_refine_dataset_factory/supervisor_7day/sf3d_material_refine_dataset_7day_supervisor.log`
  * status: `output/material_refine_dataset_factory/supervisor_7day/status.json`
  * markdown status: `output/material_refine_dataset_factory/supervisor_7day/status.md`
* supervisor 职责：
  * 保持 `sf3d_material_refine_dataset_factory_gpu0` 存活；factory 继续负责下载重试、canonicalize、GPU0 longrun 调度和基础 audit。
  * 监控最新 production longrun root，并在 shard/monitor 掉线且 shard 未完成时从 `.run.sh` 自动重启。
  * 监控最新 paper unlock root；若 eligible 未达 `128` 且 shard 全部退出，会自动重拉新 paper unlock root。
  * 不依赖单个 monitor，额外合并 `canonical_manifest_supervisor_merged.json`，避免 partial 已写入但 readiness 看不到。
  * 每 `900s` 运行 dataset quality audit，每 `1800s` 运行 buffer validation，并扫描 OOM/CUDA/Traceback 等日志错误。
  * 显式检查 GPU1 上是否出现数据流程进程；当前策略仍是 GPU0-only，GPU1 保持给训练/用户任务。
* 首次 supervisor pass 已跑通：
  * production supervisor merge: `31` records，`27` paper-pseudo eligible，仍在继续渲染。
  * paper unlock supervisor merge: `61` records，`54` paper-pseudo eligible，仍在继续渲染。
  * 当前 blocker 只是数量尚未到 `128`，不是 prior-copy；recent error scan 为 `0`。
  * GPU 状态：GPU0 约 `17-24GB / 32GB`，GPU1 约 `15-18MB / 32GB`。

### 2026-04-22 GPU0 数据扩充与下载链路加固

* 已修复 PolyHaven 材料库下载的单点失败问题：
  * `scripts/stage_polyhaven_material_bank.py` 增加 metadata/API retry、贴图下载 retry、`.tmp` 原子写入、单资产/单贴图失败记录与跳过。
  * 单个 SSL EOF / 资产 API 失败不再导致整条材料库任务退出。
  * factory 配置显式传入 `--request-retries 7 --download-retries 5 --retry-delay 2.0`。
* 已优化下载代理候选：
  * `configs/material_refine_dataset_factory_gpu0.json` 的 proxy candidates 固化为 `env`、`http://127.0.0.1:51081`、`direct`。
  * 当前观测：PolyHaven 走 direct 更快；Objaverse cached 走 `http://127.0.0.1:51081`。
  * 下载 retry wrapper 已增加 `timeout --foreground`，避免 Objaverse/Sketchfab 卡在 `download_objects` 内部无限挂死。
  * `sf3d_factory_objaverse_cached_increment` 当前超时上限为 `5400s`，超时后自动重试。
  * 新增 `sf3d_factory_objaverse_github_lfs_increment`，走 GitHub/LFS permissive rows，目标 `600`，`4` processes，超时上限 `7200s`。
  * factory longrun 输入已重新纳入既有 GitHub/LFS 成功增量：`output/highlight_pool_a_8k/objaverse_github_lfs_increment_manifest/material_refine_manifest_objaverse_increment.json`。
  * supervisor recent-error 扫描改成 offset 增量扫描，首次见到日志从末尾建基线，之后只报新增 OOM/CUDA/Traceback；避免历史 OOM 和 Objaverse per-repo clone warning 反复污染状态。
* 已让 factory / supervisor 支持长跑期间热更新配置：
  * `scripts/run_material_refine_dataset_factory.py` 每轮 loop 重新读取 config。
  * `scripts/run_material_refine_dataset_supervisor_7day.py` 每轮 loop 重新读取 config。
  * supervisor 现在支持多个 production root group，可以同时监督主长跑和稀缺材质补齐队列。
* 已重启数据侧长期进程，吃到新配置：
  * supervisor: `sf3d_material_refine_dataset_7day_supervisor`
  * factory: `sf3d_material_refine_dataset_factory_gpu0`
  * Objaverse cached download: `sf3d_factory_objaverse_cached_increment`
  * Objaverse GitHub/LFS download: `sf3d_factory_objaverse_github_lfs_increment`
* Pool-D / Pool-E 当前状态：
  * PolyHaven HDRI 已完成 `965` 个，可满足 `MIN_HDRI_COUNT=900`。
  * PolyHaven material bank 已完成 `754` 个 CC0 PBR materials，`failed_asset_count=0`，`failed_map_count=0`。
  * PolyHaven material manifest 路径已修正为 `output/highlight_pool_a_8k/aux_sources/polyhaven_materials_factory/polyhaven_material_bank_manifest.json`。
* 已新增 GPU0 稀缺材质补齐队列：
  * root: `output/material_refine_dataset_factory/scarce_material_gpu0_20260422T073856Z`
  * sessions: `sf3d_scarce_material_refine_shard0_gpu0` 到 `sf3d_scarce_material_refine_shard2_gpu0`
  * monitor: `sf3d_scarce_material_refine_merged_monitor`
  * protocol: `production_32`，`MAX_HDRI_LIGHTS=6`，每对象最多 `8 cameras x 6 HDRI = 48` 条 view/light 条件。
  * input records: `2400`，其中 `ceramic_glazed_lacquer=350`，`glass_metal=1090`，`mixed_thin_boundary=576`，`metal_dominant=240`，`glossy_non_metal=144`。
  * `has_material_prior=false` 为 `2394`，主要用于补 no-prior / auxiliary upgrade 分布。
  * 因每个对象要产出 48 组 view/light full buffers，初期 partial manifest 可能先显示 `records=0`；日志已确认 Blender 正在连续写入 buffers，完成首个对象后 monitor/supervisor 会合并出记录。
* 曾短暂试探高质量 HQ top-up：
  * root: `output/material_refine_dataset_factory/scarce_material_hq_gpu0_20260422T074007Z`
  * 配置为 `render=384 / cycles=12 / 2 shards`。
  * 与已有 8 个主分片 + 3 个稀缺分片叠加后触发 GPU0 OOM，因此已停止并从 supervisor 监控组移除。
  * 当前长期配置保留更稳的 3 分片稀缺补齐队列，避免反复 OOM。
* 当前安全运行窗口：
  * GPU0 数据进程保持在约 `19-23GB / 32GB`，全部数据渲染队列绑定 GPU0。
  * GPU1 占用来自现有 Round8 训练/评估链路，不是数据工厂新进程。
  * supervisor 会继续合并、审计、验证，并在主分片或稀缺补齐分片掉线时按 `.run.sh` 自动恢复。

### 2026-04-22 follow-up: 分布轮询与 Objaverse cache 恢复

* 已修正下载 timeout 策略：
  * `scripts/run_material_refine_dataset_factory.py` 从 `timeout --foreground` 改为 `timeout -k 60s`。
  * 原因：Objaverse downloader 会创建 worker 子进程，`--foreground` 超时后可能留下 orphan worker，导致网络/内存被幽灵进程占用。
  * 已清理旧 GitHub/LFS orphan downloader，并重启 cached/GitHub 下载会话。
* 已增加 longrun manifest 的材质轮询能力：
  * `scripts/build_material_refine_longrun_manifest.py` 新增 `--interleave-selection-keys`。
  * `scripts/launch_material_refine_longrun_dataset_tmux.sh` 新增 `INTERLEAVE_SELECTION_KEYS` 透传。
  * factory 默认配置为 `material_family,source_name,has_material_prior`，后续新 longrun 不再按单一材质块连续喂入 shard。
* 已重排并重启稀缺材质补齐队列：
  * root 保持 `output/material_refine_dataset_factory/scarce_material_gpu0_20260422T073856Z`。
  * 已完成 bundle 会复用；新 shard 输入顺序改为 `glass_metal / mixed_thin_boundary / ceramic_glazed_lacquer / metal_dominant / glossy_non_metal` 交错。
  * 目标仍是 `2400` records，`production_32`，`MAX_HDRI_LIGHTS=6`，GPU0-only。
* 已修复 Objaverse cached “下载了但 canonical=0”的恢复问题：
  * `scripts/stage_objaverse_cached_increment.py` 增加本地 cache sha256 反查，网络异常后也能把已落盘 GLB 重新挂回 `local_path`。
  * staging stdout 改为只打印摘要，避免 6000 条 records 刷爆日志。
  * 当前 `output/material_refine_aux_downloads/objaverse_cached_factory/objaverse_cached_increment_manifest.json` 已恢复 `downloaded_count=1672`。
  * 当前 `output/material_refine_aux_downloads/objaverse_cached_factory_canonical/material_refine_manifest_objaverse_increment.json` 已恢复 `records=1672`，会进入后续 factory longrun 输入。
* 已避免 Objaverse unknown 被误归为 glossy：
  * `scripts/build_material_refine_longrun_manifest.py` 将 `unknown_pending_second_pass / pending_material_probe` 保留为 `unknown_pending_second_pass`。
  * 这样后续统计不会把 Objaverse 未分类对象算成 `glossy_non_metal`，避免重现“单一 glossy”假分布。
* 已新增第二上游 Objaverse cached GPU0 top-up：
  * root: `output/material_refine_dataset_factory/objaverse_cached_gpu0_20260422T124111Z`
  * sessions: `sf3d_objaverse_cached_refine_shard0_gpu0`、`sf3d_objaverse_cached_refine_merged_monitor`
  * input: `512` local Objaverse records from recovered cached canonical manifest。
  * material labels: `unknown_pending_second_pass=496`，`ceramic_glazed_lacquer=8`，`mixed_thin_boundary=6`，`metal_dominant=2`。
  * 该队列已加入 supervisor group `objaverse_cached_topup`，后续会被 7-day supervisor 合并/审计/重启。
