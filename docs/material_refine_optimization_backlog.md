# Material Refine Optimization Backlog

## 2026-04-21 Round3 Live Training And Next Architecture Update

- Round3 upgraded full-test eval is now the reference cautionary result: UV RM MAE improved strongly, but view RM MAE and proxy PSNR/SSIM/LPIPS regressed. Treat it as `valid UV-fitting signal`, not as a paper-stage visual/material-quality win.
- Implemented: `sf3d/material_refine/eval_metrics.py` now owns the metric registry and helpers for PSNR, SSIM, LPIPS, boundary bleed, highlight localization, gradient preservation, residual safety, confidence calibration, metric availability, and metric-disagreement reports.
- Implemented: eval now supports batch LPIPS, `paper_metric_summary.html`, `metric_disagreement_report.{json,html}`, W&B main/special/diagnostic metrics, and NG-style `[eval] batch=x/y ... eta=...` terminal progress.
- Implemented: eval now supports `max_artifact_objects`, so full metrics can run over all heldout rows while limiting atlas/render image writes to a diagnostic subset. This avoids spending most evaluation time writing thousands of images.
- Implemented: W&B artifact upload is now policy-based. Training defaults to `wandb_artifact_policy: best_and_final`; eval defaults to `wandb_artifact_policy: summary`. Use `all`/`full` only for archival runs, not iterative debugging.
- Fixed: checkpoint pruning previously compared relative checkpoint paths against absolute symlink targets, so it could prune the real target behind `best.pt`. The pruning guard now resolves both sides before deletion. Round4 exposed this bug by leaving `best.pt -> epoch_024.pt` broken after epoch-24 was pruned.
- Operational note: because Round4 local best checkpoint target was already pruned before the fix, use `latest.pt` for immediate diagnostics unless `epoch_024.pt` is recovered from remote W&B artifacts. Future runs should not lose best checkpoints.
- Round4 latest full-test eval completed at `output/material_refine_paper/paper_stage1_round4_view_guard_20260421T101334Z/eval_stage1_test_latest_metric_upgrade_fast_20260421T123023Z`. Compared with the same baseline, it improved UV RM MAE (`0.1160 -> 0.0642`), view RM MAE (`1.0321 -> 0.9275`), proxy PSNR (`9.4160 -> 11.0893`), and LPIPS (`0.1099 -> 0.1013`), but SSIM slightly regressed (`0.9224 -> 0.9195`), boundary bleed worsened (`0.0060 -> 0.0300`), metal AUROC fell, and residual safety fell.
- Implemented for Round5: edge-aware RM loss, metallic classification loss, and residual safety loss were added to `scripts/train_material_refiner.py`, with terminal/W&B logging under `train/edge_aware`, `train/metallic_classification`, and `train/residual_safety`.
- Added config `configs/material_refine_train_paper_stage1_round5_boundary_safety.yaml`. It keeps the same dataset/split but intentionally optimizes the failure modes exposed by Round4 latest: boundary bleed, metallic AUROC, and unnecessary residual changes.
- Latest usable paper-stage subset has moved to `output/material_refine_paper/latest_dataset_check_20260421/stage1_subset_merged490/paper_stage1_subset_manifest.json`.
- Dataset gate status: `records=346`, `paper_stage_eligible_records=346`, `target_prior_identity_rate=0.0`, `effective_view_supervision_record_rate=1.0`, split counts `paper_train=240`, `paper_val_iid=40`, `paper_test_iid=31`, `paper_test_material_holdout=35`.
- Structural limitation remains: the eligible subset is still `ABO_locked_core + with_prior + glossy_non_metal + pseudo_from_multiview`. It is valid for Stage-1 with-prior proof, but not enough for final paper claims about no-prior, cross-generator, or broad material-family generalization.
- Round3 was launched from `configs/material_refine_train_paper_stage1_round3.yaml` with full 24-view validation, 40 validation samples, 10 validation batches, and 24 validation preview panels per epoch. GPU0 was not safe because active Blender/NG jobs occupied memory, so the live run was relaunched on GPU1 at `output/material_refine_paper/paper_stage1_round3_gpu1_20260421/train_stage1_round3`.
- Early validation is now a valid experiment signal, not smoke-only: the frozen validation SF3D prior baseline is approximately `0.1140` total UV RM MAE, while round3 reached `0.0605` by epoch 8. Preview regressions dropped from `3/24` at epoch 1 to `0/24` from epoch 2 onward.
- Implemented for the next run: training validation now reports aggregate `baseline_uv_mae`, `improvement_uv_mae`, improvement/regression rates, and grouped baseline/improvement metrics to JSON, terminal logs, training curves, and W&B.
- Implemented as a new model ablation/innovation: optional residual confidence gating. When enabled, the refiner predicts a learned per-texel residual gate and can additionally damp residual magnitude in high-prior-confidence regions. This keeps the v1 principle conservative: refine non-trivial errors without blindly overwriting good upstream material priors.
- Added config `configs/material_refine_train_paper_stage1_round4_residual_gate.yaml` as the next controlled experiment. It keeps round3 data/split/validation fixed and only enables `enable_residual_gate`, `residual_gate_bias`, `min_residual_gate`, and `prior_confidence_gate_strength`.
- Round4 online-W&B preflight passed with no blockers. It should be launched only after round3 finishes or a GPU is free, then compared directly against round3 using the same frozen split.

## 2026-04-20 Readiness Update

- Resolved: the latest merged HDRI900 manifest is now paper-stage ready instead of smoke-only. Audit summary:
  - `manifest=output/material_refine_longrun_stress24_hdri900_20260419T134158Z/canonical_manifest_monitor_merged.json`
  - `records=237`
  - `paper_stage_eligible_records=210`
  - `target_prior_identity_rate=0.1139`
  - `effective_view_supervision_record_rate=1.0`
  - `readiness_blockers=[]`
- Resolved: a fixed paper-stage subset has been materialized at `output/material_refine_pipeline_20260418T091559Z/paper_stage1_pipeline_auto_select/readiness/stage1_subset/paper_stage1_subset_manifest.json` with `210` records and non-empty `train/val/test/material_holdout` splits.
- Fixed: `scripts/train_material_refiner.py` preflight no longer fails on a stale config-level `--manifest` alias when `--train-manifest` and `--val-manifest` are explicitly overridden by the paper-stage pipeline.
- Fixed: `scripts/launch_material_refine_paper_stage1_tmux.sh` now includes the HDRI900 longrun manifest glob by default, so the watcher can discover the new ready dataset without manual patching.
- Fixed: `scripts/launch_material_refine_paper_stage1_tmux.sh` now supports GPU-idle waiting (`WAIT_FOR_GPU/GPU_INDEX/MAX_GPU_USED_MB/GPU_POLL_SECONDS`) so paper-stage training can sit in tmux and start automatically after Blender jobs release the target card.
- Operational note: the paper-stage launcher is now aligned to `GPU0`, which matches both `configs/material_refine_train_paper_stage1.yaml` and `configs/material_refine_eval_paper_benchmark.yaml`. A previous temporary launcher session was incorrectly waiting on `GPU1`; that mismatch has been corrected.

## 2026-04-19 Implementation Status

- Implemented: manifest quality schema now explicitly tracks `target_source_type`, `target_is_prior_copy`, `target_quality_tier`, and `target_confidence_summary` in `CanonicalAssetRecordV1`.
- Implemented: `scripts/audit_material_refine_manifest.py` now exports paper-stage readiness blockers, target-source/tier counts, target==prior rate, and paper-eligible sample counts as JSON/HTML/PNG.
- Implemented: `scripts/validate_material_refine_buffers.py` now acts as a strict buffer validator and reports `effective_view_supervision_record_rate`, `strict_complete_record_rate`, and per-buffer availability.
- Implemented: `scripts/build_material_refine_paper_stage1_subset.py` now builds a fixed paper-stage split, writes split audits, and emits a stage1 readiness summary plus round2 launch recommendation.
- Implemented: all new audit/subset scripts can now be launched directly from the repo root without requiring a manual `PYTHONPATH=.` prefix.
- Implemented: training preflight now persists `preflight_manifest_audit.json`, reports paper-eligible counts and effective view supervision rate, and hard-fails when paper-stage non-trivial target thresholds are not met.
- Implemented: `view_consistency` is now explicitly downgraded for the current stage by defaulting `view_consistency_mode: disabled` and `view_consistency_weight: 0.0` until real view-to-UV supervision exists.
- Implemented: model-level ablation switches now exist for `disable_view_fusion`, `disable_prior_inputs`, and `disable_residual_head`; these are checkpoint-config aware and intended for paper ablations.
- Implemented: training now supports failure-aware and difficulty-aware sampling hooks through `train_failure_tag_metadata_key`, `train_failure_tag_weights`, `train_difficulty_metadata_key`, `train_difficulty_weights`, and `train_target_quality_weights`.
- Implemented: evaluation now logs metal/non-metal F1/AUROC, failure-tag reduction, runtime, memory, and grouped summaries by `generator_id`, `source_name`, `license_bucket`, `category_bucket`, and `target_quality_tier`.
- Implemented: `scripts/run_material_refine_paper_stage1_pipeline.py` and `scripts/launch_material_refine_paper_stage1_tmux.sh` now provide a gated watcher that audits readiness first and only starts train/eval/report once paper-stage conditions are met.
- Implemented: `scripts/run_material_refine_ablation_suite.py` plus override configs now provide a fixed ablation entry point for `scalar_broadcast`, `prior_smoothing`, `ours_full`, and the trainable `no_prior/no_view/no_residual` suite.
- Implemented: `scripts/select_material_refine_best_manifest.py` now ranks multiple existing canonical manifests with cache-aware audits so the paper-stage watcher can follow the strongest currently available dataset instead of a single hard-coded manifest.
- Historical P0 status: the original audited full manifest remains blocked for valid experiments because `target_prior_identity_rate=1.0000`, `paper_stage_eligible_records=0`, `effective_view_supervision_record_rate=0.0`, and the generated stage1 subset is empty.
- Implemented: training preflight now audits target/prior identity and can hard-fail full training with `fail_on_target_prior_identity: true`.
- Implemented: round2 GPU0 config uses a growing-manifest contract with epoch-level train reload and frozen validation split creation.
- Implemented: optimizer and runtime knobs are explicit in config: AdamW/Adam, betas, epsilon, weight decay, gradient clipping, TF32, matmul precision, AMP dtype, warmup, ReduceLROnPlateau.
- Implemented: prior-dropout curriculum is configurable with `prior_dropout_start_prob`, `prior_dropout_end_prob`, and `prior_dropout_warmup_epochs`.
- Implemented: terminal logs now print preflight, train interval, validation, epoch, and final checkpoint state; W&B receives train/eval/group metrics and preview images.
- Implemented: environment acceleration/diagnostic packages are pinned: `nvidia-ml-py`, `tensorboard`, `orjson`, `tqdm`, `psutil`, and `rich`.
- Implemented: tmux/tunx-compatible background launcher `scripts/launch_material_refine_round_tmux.sh` runs preflight, GPU-memory wait, train, eval, attribute comparison, validation panels, manifest audit, round analysis, and final W&B summary.
- Implemented: validation comparison panels now compare SF3D original RGB/prior RM maps against refined RM maps and are exported locally plus uploaded to W&B.
- Implemented: training/eval data interfaces now expose `generator_id` as a first-class filter, sampler key, metric group, W&B axis, and validation-panel field.
- Implemented: paper-oriented configs were added in `configs/material_refine_train_paper_stage1.yaml` and `configs/material_refine_eval_paper_benchmark.yaml`.
- Implemented: a paper protocol was added in `docs/material_refine_paper_protocol.md` with baselines, metrics, ablations, cross-generator experiments, figures, tables, and W&B grouping.
- Data quality is no longer the immediate blocker for the latest HDRI900 merged subset. The next operational blocker is starting formal training without colliding with active Blender dataset jobs.
- Still future model work: visibility-weighted UV splatting, learned view weighting, material token conditioning, multi-scale UV refinement, and EMA checkpointing remain architectural upgrades after the data target issue is resolved.

## Paper-Ready Protocol

- Main contribution framing should be generator-agnostic: a lightweight post-refinement module `R` improves material roughness/metallic maps for any upstream generator `G` once `C` canonicalizes output assets.
- The main claim must be evaluated against `G raw`, `G canonical prior`, scalar broadcast, prior smoothing, no-view refiner, no-prior refiner, no-residual refiner, and full `R`.
- Cross-generator experiments are mandatory for a top-conference framing: train/test SF3D, SF3D-to-other, mixed-generator, and leave-one-generator-out once SPAR3D/Hunyuan3D canonical bundles are available.
- Metrics should include UV RM MAE, view-space RM MAE, rendered appearance PSNR/SSIM/LPIPS, highlight localization, boundary bleed, metal/non-metal confusion, failure-tag reduction, runtime, memory, and GLB validity.
- Ablations should remove one mechanism at a time: material prior, prior dropout curriculum, multi-view buffers, UV fusion, residual head, confidence weighting, edge-aware loss, and generator conditioning once added.
- Paper tables should be grouped by `generator_id`, `prior_label`, `source_name`, `license_bucket`, `supervision_tier`, and material category.
- Paper figures should include method overview, before/after GLB renders, roughness/metallic atlas comparison, cross-generator qualitative panels, failure taxonomy, and data reliability plots.
- W&B groups should be stable: `paper-stage1`, `paper-ablation`, `paper-cross-generator`, `paper-benchmark`, and `paper-figures`.
- Reproducibility gates should include manifest digest, target/prior identity rate, license-bucket counts, split leakage checks, git SHA, config artifact, checkpoint artifact, and round summary artifact.

## P0: Make Evaluation Trustworthy

- Build non-trivial RM targets. Current round1 targets are largely identical to SF3D/pseudo prior, so baseline MAE can be exactly zero. The next data pass should include physically rendered or manually validated roughness/metallic targets that differ from the upstream prior.
- Keep the manifest validator as a training gate. `scripts/audit_material_refine_manifest.py` now checks canonical mesh/glb, UV albedo/normal, prior RM, target RM, confidence, view buffers, target source/tier, and target/prior identity; `scripts/train_material_refiner.py` can now fail full training automatically when identity is above the configured threshold or when non-trivial target count is below the paper minimum.
- Treat view consistency honestly. Because current canonical bundles do not yet provide effective view-to-UV supervision, configs now default this term to disabled and the buffer validator reports `effective_view_supervision_record_rate` explicitly.
- Freeze the paper-stage split before training. Use `scripts/build_material_refine_paper_stage1_subset.py` to generate the fixed split file, split audit, and stage1 readiness report; do not let later full-data additions silently reshuffle heldout objects.
- Separate validation by supervision source. Keep `generator_id`, `with_prior`, `no_prior`, `strong_gt`, `pseudo_gt`, `license_bucket`, and `source_name` as first-class eval axes.
- Add material-sensitive heldout sets. Keep fixed small sets for metal/non-metal confusion, glossy ceramic, glass+metal, mixed thin boundaries, and glossy non-metal.
- Add rendered image metrics. UV MAE is useful but not enough; evaluate view-space roughness/metallic, shaded render difference, highlight localization, and edge bleeding.

## P1: Improve Training Stability And Signal

- Use prior dropout curriculum. Start low, e.g. `0.05`, increase to `0.25-0.4`, and separately report dropped-prior metrics.
- Add per-source and per-generator sampling quotas. `source_x_prior` and `generator_x_prior` balancing are available; full training should cap overrepresented easy prior samples and oversample hard no-prior/highlight cases.
- Add target confidence calibration. Confidence should downweight pseudo labels and ambiguous texels, not just serve as a mask.
- Add edge-aware losses. Use UV normal/albedo/segmentation edges to reduce material boundary bleed.
- Add optional no-prior auxiliary loss. Force coarse head to predict RM before prior fusion, especially for generators without material priors.
- Add EMA checkpointing. Track an exponential moving average model for evaluation and demo inference.

## P2: Improve Model Capacity Carefully

- Replace naive UV scatter fusion with visibility-weighted splatting. The current fusion is intentionally simple and can alias view features into the atlas.
- Add learned view weighting. Let the model choose which view contributes per texel based on mask, normal angle, and visibility.
- Add material token conditioning. Feed source/generator/prior mode/supervision tier through a small embedding instead of relying only on pixels; keep a zero-shot setting for held-out generators to preserve the generality claim.
- Add multi-scale UV refinement. Predict coarse global RM structure at low resolution and residual details at full atlas resolution.
- Add optional low-rank adapters. Keep the v1 model light while allowing source-specific adaptation.

## P3: Improve Runtime Efficiency

- Benchmark `torch.compile` on the refiner. The model is small enough that compile overhead may not pay off for short runs, but it may help full training.
- Add automatic batch-size probing. Previous GPU0 tests showed batch 8 is safe and 12+ OOM; make this a preflight option.
- Cache decoded atlas/buffer tensors. PNG decode and resize can become a bottleneck as full data grows.
- Add async preview export. Validation preview image saving should not block the training step on large validation sets.
- Add dataloader throughput diagnostics. Track examples/sec split into data time, forward time, backward time, and validation time.

## P4: Improve Experiment Management

- Make W&B online the default only after login is detected. Otherwise fail fast or switch to offline with a loud warning.
- Log final artifacts in a dedicated summary run. Include `best.pt`, `training_curves.png`, eval `report.html`, attribute comparison, and manifest snapshots.
- Add a `run_id` convention. Use `material-refine-{split}-{date}-{git_sha}` to avoid confusing repeated local runs.
- Export a single `round_summary.json`. It should include train config, manifest stats, best checkpoint, eval metrics, W&B URL, and known limitations.
- Add CI smoke. Run preflight, 1-step train, eval on 2 samples, and report generation.

## P5: Demo And Productization

- Add Gradio model checkpoint selector. Let the demo switch between baseline-only, latest, best, and a user-provided checkpoint.
- Add material attribute panels. Show roughness/metallic histograms and mean deltas next to before/after renders.
- Add failure tags in UI. Surface `metal_nonmetal_confusion`, `boundary_bleed`, `local_highlight_misread`, and `over_smoothing`.
- Add GLB inspection checks. Confirm exported GLB contains `metallicRoughnessTexture`, preserves base color/normal textures, and sets RM factors consistently.
- Add batch demo report mode. Given a manifest and checkpoint, generate an HTML gallery without launching Gradio.

## 2026-04-20 Round1 结果反推的新增优化项

- P0: fix eval/report schema assumptions early. `view_rgba_paths` 已经出现 `dict` 形态与 `str` 形态并存，后续所有 exporter/validator 都要统一走 path resolver，避免 manifest 细节变化再次打断评测链路。
- P1: current round1 only improves `over_smoothing`; `metal_nonmetal_confusion`、`local_highlight_misread`、`boundary_bleed` 几乎没有下降，后续采样和损失必须围绕这三个 failure tag 定向优化。
- P1: training/eval 当前仍是单源 `ABO_locked_core + with_prior + glossy_non_metal` 主导，论文主表前必须补齐 material-family、no-prior、cross-generator 三个方向的真实覆盖。
- P1: `view_consistency_weight=0.0` 仍是现状，短期内不要把它写成有效贡献；中期要么补真监督链路，要么在 ablation/main table 中明确排除。
- P1: metal/non-metal 相关指标当前 F1/AUROC 的可解释性仍弱，需增加更稳健的 object-level 与 rendered-view-level 二值判别统计，避免 view 级重复计数淹没真实变化。
- P2: W&B sync 目前依赖离线 run 再手动同步。建议后续增加 `wandb online preflight` 与 artifact 分级上传策略，避免训练大图/preview 过多时同步拖慢实验收尾。

## 2026-04-21 W&B 与验证链路整改

- Implemented: training validation can now run by total-progress milestones via `validation_progress_milestones`; the paper-stage preset uses 40 milestones, so validation fires every 1/40 of the planned run instead of blindly every epoch/step.
- Implemented: validation records can be frozen with a deterministic balance key (`val_balance_key=material_family`) and per-group cap. Current Stage-1 data still has only `glossy_non_metal`, so the training log now emits an explicit coverage limitation instead of pretending material-family balance exists.
- Implemented: validation previews were redesigned as compact per-object panels. Each panel shows input SF3D/albedo/normal/confidence plus roughness and metallic rows with simple `SF3D`, `GT`, `Pred`, and `Error` labels.
- Implemented: W&B training logging now uses a whitelist. Terminal logs keep detailed NG-style epoch/step/progress/ETA diagnostics, while W&B drops `progress/*`, large dataset expansions, `val/preview_grid`, and individual `val/preview_XX` spam.
- Implemented: W&B validation images are logged as small `val/comparison_panels` lists with a configurable cap (`wandb_val_preview_max`) instead of one large contact sheet and many redundant suffixed images.
- Implemented: evaluation W&B logging now keeps aggregate main metrics, material-specific metrics, diagnostic means, compact group summaries, and explicit runtime metrics (`seconds_per_object`, `ms_per_object`, `objects_per_second`). Heavy `eval/top_cases` media tables are disabled by default and remain local in `diagnostic_cases.json`.
- Risk: until the incoming data jobs add metal/glass/mixed-boundary families, material-balanced validation can only report the imbalance. The code path is ready, but the current manifest cannot satisfy that experimental requirement.

## 2026-04-22 Round6 结果与 Round7 改进

- Round6 full-test summary: UV RM MAE improved from `0.11596` to `0.06189`, view RM MAE improved from `1.03209` to `0.96848`, PSNR improved from `9.4160` to `10.0246`, and LPIPS improved from `0.10988` to `0.10429`.
- Round6 remaining problems: SSIM dropped from `0.92240` to `0.91765`, boundary bleed rose from `0.00598` to `0.02517`, metal AUROC dropped from `0.79386` to `0.77498`, and failure taxonomy still shows no reduction for `metal_nonmetal_confusion` / `boundary_bleed` / `local_highlight_misread`.
- Compared with Round4, Round6 is better on UV MAE, residual safety, regression rate, boundary bleed, and metal AUROC, but worse on view RM MAE, PSNR, SSIM, and LPIPS. The next target is to combine Round4's view/render behavior with Round6's safer residual behavior.
- Implemented: Round7 adds `gradient_preservation_loss`, logged as `train/gradient_preservation` and `val/loss/gradient_preservation`, to directly preserve roughness/metallic gradients and reduce over-smoothing / thin-boundary loss.
- Implemented: `configs/material_refine_train_paper_stage1_round7_gradient_guard.yaml` starts from Round6 but increases view recovery pressure, adds gradient preservation, and relaxes the residual gate slightly.
- Implemented: eval no longer prints the full `summary.json` to stdout by default. It prints a compact summary and keeps the full JSON locally, preventing W&B console from being flooded by tens of thousands of lines.
- Risk: Round7 is still limited by the same Stage-1 single-family data (`glossy_non_metal`) until the data pipeline contributes additional material families.
