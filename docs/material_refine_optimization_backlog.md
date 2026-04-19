# Material Refine Optimization Backlog

## 2026-04-19 Implementation Status

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
- Still blocked by data quality: non-trivial roughness/metallic targets are required before full training can claim improvement over SF3D, because the current audited manifest has target/prior identity near 100%.
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
- Keep the manifest validator as a training gate. `scripts/audit_material_refine_manifest.py` now checks canonical mesh/glb, UV albedo/normal, prior RM, target RM, confidence, view buffers, and target/prior identity; `scripts/train_material_refiner.py` can now fail full training automatically when identity is above the configured threshold.
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
