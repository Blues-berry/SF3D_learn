# Material Refine Training Architecture

## Design Goal

This training stack follows the useful parts of the Neural Gaffer training style:

- config-driven execution
- separated method/data configs
- step-based logging, validation, and checkpointing
- explicit W&B tracker setup
- stable validation snapshots
- resumable checkpoints
- rich validation visualizations

It does not copy the diffusion-specific parts of NG:

- no `accelerate` dependency is required for v1
- no diffusion scheduler or pipeline sampling
- no image-space relighting losses
- no multi-process distributed launcher assumption

The material-refine task is lighter and consumes `CanonicalAssetRecordV1` manifests, so the training loop stays PyTorch-native and uses an explicit `--cuda-device-index` instead of hard-coding a GPU.

## P0 Readiness Contract

Paper-stage full training is intentionally blocked until these conditions are all true:

- `target_prior_identity_rate <= 0.30`
- `paper_stage_eligible_records >= 128`
- the fixed stage1 subset is non-empty

The historical full manifest does not satisfy that contract. Its blocked state was:

- `target_prior_identity_rate = 1.0000`
- `paper_stage_eligible_records = 0`
- `effective_view_supervision_record_rate = 0.0`
- `stage1_subset_records = 0`

That older result remains engineering smoke only. It validates the pipeline, but it does not support a paper-stage effectiveness claim.

The latest longrun merged manifest does satisfy the contract:

- manifest: `output/material_refine_longrun_stress24_hdri900_20260419T134158Z/canonical_manifest_monitor_merged.json`
- `records = 490`
- `target_prior_identity_rate = 0.2939`
- `paper_stage_eligible_records = 346`
- `effective_view_supervision_record_rate = 1.0`
- `stage1_subset_records = 346`
- fixed split counts: `paper_train = 240`, `paper_val_iid = 40`, `paper_test_iid = 31`, `paper_test_material_holdout = 35`

The current Stage-1 subset is valid for a with-prior paper-stage run, but its scope is intentionally narrow:

- source/generator: `ABO_locked_core`
- prior mode: `with_prior`
- target type: `pseudo_from_multiview`
- material family: `glossy_non_metal`

Do not use this subset alone to claim no-prior generalization, cross-generator robustness, or broad material-family coverage.

One training-side compatibility bug was also fixed at this stage: `run_preflight_checks()` no longer treats the legacy optional `--manifest` alias as a separate required file once explicit `--train-manifest` and `--val-manifest` are provided. This matters for paper-stage launches because the config keeps a historical default `manifest` path while the pipeline injects the newly generated stage1 subset manifests at runtime.

## Module Principles

The v1 refiner follows a conservative residual-refinement principle:

- Treat the upstream generator output as a baseline, not as something to overwrite blindly.
- Preserve `baseColorTexture` and `normalTexture`; only refine roughness/metallic UV maps in v1.
- Use roughness/metallic priors when they exist, but train with `prior_dropout_prob` so the same model also learns the no-prior path.
- Predict a coarse RM estimate first, then predict a bounded residual delta around the prior/coarse initialization.
- Optionally predict a residual confidence gate for each texel. This is disabled in round3 for compatibility and enabled in the round4 residual-gate ablation to test conservative correction of upstream priors.
- Keep dataset preparation outside the training process; training consumes only canonical manifests and never audits raw assets.
- Log by generator, source, prior mode, supervision tier, and license bucket so quality changes are traceable to data provenance and upstream-generator generalization.

## Config Layout

Recommended split-config launch:

```bash
cd /home/ubuntu/ssd_work/projects/stable-fast-3d

/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/train_material_refiner.py \
  --method-config configs/material_refine_method_rm_refine.yaml \
  --data-config configs/material_refine_data_partial.yaml \
  --output-dir output/material_refine_train_ng_style \
  --wandb-mode online
```

Single-config launch is also supported:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/train_material_refiner.py \
  --config configs/material_refine_train_ng_style.yaml \
  --wandb-mode online
```

Round2 background launch is the recommended long-running path. It uses `tmux` by default and falls back to `tunx` if such a wrapper exists on the machine:

```bash
cd /home/ubuntu/ssd_work/projects/stable-fast-3d

SESSION=sf3d_material_refine_round2_gpu0 \
GPU_INDEX=0 \
MAX_GPU_USED_MB=4096 \
scripts/launch_material_refine_round_tmux.sh
```

The launcher runs the whole gated round in order:

- train preflight and manifest leakage gate
- GPU0 memory wait through `nvidia-ml-py`
- training
- evaluation
- SF3D baseline vs refined attribute comparison
- validation inference comparison panels
- manifest audit
- round analysis
- final W&B round-summary upload

Attach and inspect:

```bash
tmux attach -t sf3d_material_refine_round2_gpu0
tail -f output/material_refine_pipeline_20260418T091559Z/train_round2_gpu0_gated/logs/train.log
cat output/material_refine_pipeline_20260418T091559Z/train_round2_gpu0_gated/logs/sf3d_material_refine_round2_gpu0.status.json
```

Paper-stage background launch is now available as a gated watcher. It keeps polling the manifest until the readiness gate passes, then runs train/eval/report/W&B automatically:

```bash
cd /home/ubuntu/ssd_work/projects/stable-fast-3d

SESSION=sf3d_material_refine_paper_stage1 \
WAIT_FOR_READY=true \
POLL_SECONDS=300 \
WAIT_FOR_GPU=true \
GPU_INDEX=0 \
MAX_GPU_USED_MB=4096 \
scripts/launch_material_refine_paper_stage1_tmux.sh
```

By default the paper-stage watcher now scans multiple existing manifest lines, including the original paper pipeline bundle, the `highlight_pool_a_8k` outputs, and the newer HDRI900 longrun manifests, then selects the best current candidate with `scripts/select_material_refine_best_manifest.py`.

The launcher now also supports GPU-idle waiting before it hands control to the paper-stage pipeline, which is important when Blender dataset jobs are still consuming the same card.

Evaluation launch:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/eval_material_refiner.py \
  --config configs/material_refine_eval_ng_style.yaml \
  --checkpoint output/material_refine_train_ng_style/best.pt \
  --wandb-mode online
```

Paper-stage launch after non-trivial RM targets are available:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/train_material_refiner.py \
  --config configs/material_refine_train_paper_stage1.yaml

/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/eval_material_refiner.py \
  --config configs/material_refine_eval_paper_benchmark.yaml \
  --checkpoint output/material_refine_paper/stage1_main/best.pt
```

The current round3 live preset uses the 346-record subset and full 24-view validation:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/train_material_refiner.py \
  --config configs/material_refine_train_paper_stage1_round3.yaml \
  --cuda-device-index 1 \
  --output-dir output/material_refine_paper/paper_stage1_round3_gpu1_20260421/train_stage1_round3
```

The next controlled architecture ablation keeps the same split and enables residual gating:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/train_material_refiner.py \
  --config configs/material_refine_train_paper_stage1_round4_residual_gate.yaml \
  --cuda-device-index 1 \
  --output-dir output/material_refine_paper/stage1_round4_residual_gate
```

Per-generator paper eval can be produced by overriding only the filter and output name:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/eval_material_refiner.py \
  --config configs/material_refine_eval_paper_benchmark.yaml \
  --checkpoint output/material_refine_paper/stage1_main/best.pt \
  --generator-ids sf3d \
  --output-dir output/material_refine_paper/eval_sf3d_test \
  --tracker-run-name material-refine-paper-sf3d-test
```

## Data Policy

The training process does not own dataset preparation. It only consumes manifests produced by the dataset subprocess.

Current default data config:

- `train_manifest`: `output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_partial.json`
- `val_manifest`: `output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_partial.json`
- `freeze_val_manifest_to`: `output/material_refine_pipeline_20260418T091559Z/prepared/full/frozen_val_manifest.json`

Training can consume a growing partial manifest:

- `reload_manifest_every: 1` reloads the train loader at every epoch.
- `freeze_val_manifest_to` creates or reuses a stable validation manifest.
- `split_strategy: auto` uses manifest splits when `val/test` exist; otherwise it falls back to deterministic hash splits.
- `train_generator_ids` and `val_generator_ids` can restrict experiments to one upstream generator without changing the manifest.
- `train_balance_mode: generator_x_prior` keeps cross-generator paper runs from being dominated by SF3D or by with-prior samples.

When the dataset subprocess finishes the full manifest, switch only the data config:

```bash
--train-manifest output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_full.json
```

Keep the frozen validation manifest unchanged unless you intentionally want a new validation benchmark.

## Training Defaults

Recommended v1 defaults:

- `atlas_size: 512`
- `buffer_resolution: 256`
- `batch_size: 8` on RTX 5090 GPU0 when no competing jobs are active; use `batch_size: 4` for shared smoke runs
- `grad_accumulation_steps: 2`
- `optimizer: adamw`
- `learning_rate: 2.0e-4`
- `min_learning_rate: 2.0e-5`
- `adam_beta1: 0.9`
- `adam_beta2: 0.999`
- `adam_epsilon: 1.0e-8`
- `warmup_steps: 250`
- `lr_scheduler: plateau`
- `train_balance_mode: source_x_prior`
- `prior_dropout_prob: 0.25`
- `prior_dropout_start_prob: 0.05`
- `prior_dropout_end_prob: 0.35`
- `prior_dropout_warmup_epochs: 8`
- `refine_weight: 1.0`
- `coarse_weight: 0.35`
- `prior_consistency_weight: 0.10`
- `smoothness_weight: 0.02`
- `view_consistency_weight: 0.0`
- `view_consistency_mode: disabled`
- `matmul_precision: high`
- `allow_tf32: true`

Paper ablation switches are now first-class config knobs:

- `disable_prior_inputs: true` for `no_prior_refiner`
- `disable_view_fusion: true` for `no_view_refiner`
- `disable_residual_head: true` for `no_residual_refiner`

The training script supports `--optimizer adamw` and `--optimizer adam`. AdamW remains the default because it gives stable weight decay for the compact UNet heads, while plain Adam is useful for ablation or very small smoke runs.

Full training should set `fail_on_target_prior_identity: true`. The current round1 manifest is a useful pipeline smoke test, but its target roughness/metallic maps are identical to SF3D priors for the audited records, so the gate intentionally prevents treating it as a real quality-improvement training set.

The paper-stage thresholds are config-backed and should remain stable across runs:

- `max_target_prior_identity_rate_for_paper: 0.30`
- `min_nontrivial_target_count_for_paper: 128`

Debug logs include:

- runtime: torch version, CUDA device, AMP dtype, TF32, optimizer, parameter count
- throughput: samples/sec and seconds/batch
- optimization: learning rate, gradient norm, loss components
- memory: allocated, reserved, and max allocated GPU memory
- data mix: source, prior/no-prior, supervision tier, and license bucket counts
- generator mix: `generator_id` counts and validation metrics, needed for cross-generator tables
- supervision realism: paper-eligible non-trivial target counts and effective view-supervision rate
- failure emphasis: sampler-side difficulty/tag reweighting and eval-side failure-tag reduction

Terminal output is compact by default and mirrors the Neural Gaffer tqdm pattern. Full machine-readable JSON is kept in files/W&B; it is printed to the terminal only when `terminal_json_logs: true`.

- `[preflight]`: manifest/device/W&B/auth checks before training starts
- `[epoch:start]`: epoch, train/val record counts, planned samples, batch count, and optimizer-step count
- `epoch N/M`: tqdm step progress with key postfix values such as `loss`, `lr`, `ref`, `edge`, `grad`, `v`, `sps`, and `mem`
- `[train]`: optional verbose interval line, enabled with `train_line_logs: true` or used as the fallback when the progress bar is disabled
- `[val]`: validation label, UV MAE by channel, best metric, and whether the model improved
- `[epoch]`: compact end-of-epoch summary with checkpoint path
- `[final]`: final output directory, `latest.pt`, `best.pt`, and `history.json`

Validation preview logging now uses three layers instead of a single image list:

- `val/previews`: the full preview list for the step
- `val/preview_grid`: one contact-sheet image containing the whole step's preview set
- `val/preview_00`, `val/preview_01`, ...: fixed per-slot preview keys so W&B can show multiple sample trajectories side by side across epochs

Each preview panel now includes:

- representative SF3D input RGB views
- baseline / target / refined roughness
- baseline / target / refined metallic
- baseline-target and refined-target error heatmaps
- per-object baseline/refined MAE and improvement text in the header

Training visualizations are exported to the output directory when `export_training_curves: true`:

- `training_curves.png`
- `training_summary.html`

Existing runs can be visualized without retraining:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/export_material_refine_training_curves.py \
  --history-json output/material_refine_pipeline_20260418T091559Z/train_round1_gpu0/history.json
```

Manifest quality can be audited before trusting metrics:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/audit_material_refine_manifest.py \
  --manifest output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_full.json \
  --output-dir output/material_refine_pipeline_20260418T091559Z/manifest_audit_full

/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/validate_material_refine_buffers.py \
  --manifest output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_full.json \
  --output-dir output/material_refine_pipeline_20260418T091559Z/buffer_validation_full

/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/build_material_refine_paper_stage1_subset.py \
  --manifest output/material_refine_pipeline_20260418T091559Z/prepared/full/canonical_manifest_full.json \
  --output-root output/material_refine_pipeline_20260418T091559Z/paper_stage1_subset
```

Paper ablation suite entry point:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/run_material_refine_ablation_suite.py \
  --stage1-manifest output/material_refine_paper/paper_stage1_pipeline/readiness/stage1_subset/paper_stage1_subset_manifest.json \
  --reference-checkpoint output/material_refine_paper/paper_stage1_pipeline/train_stage1_main/best.pt \
  --output-root output/material_refine_paper/ablation_suite
```

Round-level analysis combines training curves, eval metrics, material attribute deltas, and manifest audit warnings:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/export_material_refine_round_analysis.py \
  --train-dir output/material_refine_pipeline_20260418T091559Z/train_round1_gpu0 \
  --eval-dir output/material_refine_pipeline_20260418T091559Z/eval_round1_gpu0_test \
  --audit-dir output/material_refine_pipeline_20260418T091559Z/manifest_audit_full \
  --output-dir output/material_refine_pipeline_20260418T091559Z/round1_analysis
```

For quick experiments, use step-based controls:

```bash
--max-train-steps 2000 \
--validation-steps 250 \
--checkpointing-steps 500
```

For longer runs, epoch-based validation is fine:

```bash
--validation-steps 0 \
--eval-every 1 \
--save-every 1
```

## W&B Contract

Training logs:

- `train/total`, `train/refine_l1`, `train/coarse_l1`
- `val/uv_total_mae`, `val/uv_roughness_mae`, `val/uv_metallic_mae`
- `val/groups/generator/*`
- `val/groups/source/*`
- `val/groups/prior/*`
- `val/groups/tier/*`
- `dataset/train/*`
- `dataset/val/*`
- validation preview panels
- checkpoint artifacts

Evaluation logs:

- final before/after MAE
- grouped metrics by generator, source, prior, supervision tier, and view
- top-case atlas table
- `metrics.json`, `summary.json`, `report.html`, and atlas artifacts

Validation panel logs:

- original SF3D RGB views
- SF3D baseline roughness/metallic atlas maps
- refined roughness/metallic atlas maps
- absolute delta maps
- target confidence

Round-summary logs:

- train/eval/audit scalar metrics
- training curves
- material attribute comparison
- manifest audit plot
- validation comparison panel table
- checkpoint/report/artifact bundle

If no `WANDB_API_KEY` is configured and `--wandb-mode auto` or `offline` is used, runs stay local. To push to `wandb.ai/home`, authenticate inside the SF3D environment and sync the offline runs:

```bash
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/wandb login
/home/ubuntu/ssd_work/conda_envs/sf3d/bin/wandb sync wandb/offline-run-*
```

For non-interactive launch, export `WANDB_API_KEY` first and run training/eval with `--wandb-mode online`.

## Checkpoints

Files written during training:

- `latest.pt`: most recent checkpoint
- `best.pt`: symlink to best validation checkpoint
- `epoch_*.pt`: epoch checkpoints
- `step_*.pt`: step checkpoints when `--checkpointing-steps > 0`
- `history.json`
- `train_state.json`
- `validation/*.json`
- `validation_previews/*/*.png`

Resume options:

```bash
--resume output/material_refine_train_ng_style/latest.pt
```

or:

```bash
--resume-from-checkpoint latest
```

## Task-Specific Deviations From NG

The material refiner uses RM-specific validation instead of image-relighting metrics:

- UV roughness MAE
- UV metallic MAE
- view-space consistency when UV buffers exist
- grouped source/prior/tier diagnostics
- grouped generator diagnostics
- before/after atlas panels
- SF3D-original-vs-refined validation comparison panels

The model-side process is intentionally decoupled from dataset preparation:

- Data subprocesses own canonicalization and baking.
- The training process owns training, validation, eval, reports, and W&B logging on the explicitly selected GPU.
- training can continue from partial manifests while the full dataset grows.
