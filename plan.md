# SF3D 轻量材质精修三段式方案 v1

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


