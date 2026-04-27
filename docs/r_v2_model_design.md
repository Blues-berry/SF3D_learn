# R-v2 Model Design: PAUVMaterialRefiner / M3R-Net

Date: 2026-04-26

## Goal

R-v2 is a controllable strengthening of the current refiner, not a full rewrite of the eight-module research model.

Paper-facing name:

```text
M^3R-Net: Material-, Mask-, and Multi-view-aware UV Material Refinement Network
```

Code-facing role:

```text
Prior-Aware UV Material Refiner
```

## Main Blocks

### Prior Calibration

The refiner reads `input_prior_*` first, with legacy `uv_prior_*` fallback. It uses generator/source conditioning when enabled to estimate prior reliability and build `rm_init`.

### No-prior Bootstrap

When `prior_mode=none` or `input_prior_confidence≈0`, the dual-path initializer uses UV albedo/normal/mask and domain features to produce `bootstrap_rm_uv`. `rm_init` then comes from bootstrap instead of blindly trusting the placeholder prior.

### Canonical Multi-view Fusion

The view branch is now described as canonical view evidence, not SF3D view evidence. Existing view fusion, hard-view routing, and tri-branch fusion remain compatible.

### Boundary Safety

The boundary branch predicts `boundary_gate`, `safe_update_mask_uv`, and `bleed_risk_uv`, then suppresses unsafe residual updates near likely material/UV boundaries.

### Confidence-Gated Residual Trunk

The trunk predicts `delta_rm` and `change_gate`; final output is:

```text
refined_rm = clamp(rm_init + change_gate * delta_rm, 0, 1)
```

## Training Switches

Default train-side switches:

- `enable_prior_source_embedding=true`
- `enable_no_prior_bootstrap=true`
- `enable_boundary_safety=true`
- `enable_change_gate=true`
- `enable_material_aux_head=false`
- `enable_render_proxy_loss=false`

Ablation switches:

- `disable_prior_source_embedding`
- `disable_no_prior_bootstrap`
- `disable_boundary_safety`
- `disable_change_gate`
- `disable_prior_safe_loss`

## Loss Names

Training now exposes preferred R-v2 loss names:

- `train/loss_uv`
- `train/loss_prior_safe`
- `train/loss_boundary`
- `train/loss_gradient`
- `train/change_gate_mean`
- `train/mean_abs_delta`
- `train/prior_reliability_mean`

Legacy loss names remain for continuity.

## Smoke Results

See:

```text
output/material_refine_r_v2_smoke/r_v2_smoke_summary.json
```

The smoke verifies:

- with-prior and without-prior batches load
- `input_prior_*` aliases are present
- target keys are excluded from forward input
- no-prior bootstrap is active
- `change_gate` and `prior_reliability` are returned in diagnostics

