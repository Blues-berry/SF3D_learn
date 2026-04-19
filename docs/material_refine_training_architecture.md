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

## Module Principles

The v1 refiner follows a conservative residual-refinement principle:

- Treat the upstream generator output as a baseline, not as something to overwrite blindly.
- Preserve `baseColorTexture` and `normalTexture`; only refine roughness/metallic UV maps in v1.
- Use roughness/metallic priors when they exist, but train with `prior_dropout_prob` so the same model also learns the no-prior path.
- Predict a coarse RM estimate first, then predict a bounded residual delta around the prior/coarse initialization.
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
- `view_consistency_weight: 0.15`
- `matmul_precision: high`
- `allow_tf32: true`

The training script supports `--optimizer adamw` and `--optimizer adam`. AdamW remains the default because it gives stable weight decay for the compact UNet heads, while plain Adam is useful for ablation or very small smoke runs.

Full training should set `fail_on_target_prior_identity: true`. The current round1 manifest is a useful pipeline smoke test, but its target roughness/metallic maps are identical to SF3D priors for the audited records, so the gate intentionally prevents treating it as a real quality-improvement training set.

Debug logs include:

- runtime: torch version, CUDA device, AMP dtype, TF32, optimizer, parameter count
- throughput: samples/sec and seconds/batch
- optimization: learning rate, gradient norm, loss components
- memory: allocated, reserved, and max allocated GPU memory
- data mix: source, prior/no-prior, supervision tier, and license bucket counts
- generator mix: `generator_id` counts and validation metrics, needed for cross-generator tables

Terminal output intentionally includes both machine-readable JSON and readable progress lines:

- `[preflight]`: manifest/device/W&B/auth checks before training starts
- `[train]`: epoch, optimizer step, lr, train loss, grad norm, throughput, and GPU memory
- `[val]`: validation label, UV MAE by channel, best metric, and whether the model improved
- `[epoch]`: compact end-of-epoch summary with checkpoint path
- `[final]`: final output directory, `latest.pt`, `best.pt`, and `history.json`

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
