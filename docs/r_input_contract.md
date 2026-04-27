# R Input Contract

Date: 2026-04-26

## Canonical R Inputs

The R module consumes only input-side evidence:

- `uv_albedo`
- `uv_normal`
- `uv_mask` when available
- `input_prior_roughness`
- `input_prior_metallic`
- `input_prior_confidence`
- `view_features`
- `view_uvs`
- `view_masks`
- `view_importance`
- metadata such as `generator_id`, `source_name`, `prior_source_type`, `prior_generation_mode`, and `prior_mode`

For backward compatibility, datasets still provide:

- `uv_prior_roughness`
- `uv_prior_metallic`
- `uv_prior_confidence`

The dataset collator aliases them to:

- `input_prior_roughness`
- `input_prior_metallic`
- `input_prior_confidence`

## Target Non-Leakage

The following keys are supervision-only and must not be passed into model forward during inference or architecture tests:

- `uv_target_roughness`
- `uv_target_metallic`
- `uv_target_confidence`
- `view_targets`

Training and eval may read targets for loss and metrics after model forward.

## Prior Source Semantics

`prior_source_type` is resolved from:

1. explicit `prior_source_type`
2. `prior_generation_mode`
3. `prior_label`
4. `prior_mode`
5. `no_prior_placeholder` when `has_material_prior=false` or `prior_mode=none`

This keeps R compatible with SF3D today and future upstream generators later.

## Output Contract

Model forward now returns both legacy and R-v2 names:

- `baseline` and `input_prior`
- `initial`, `rm_init`, and `rm_init_uv`
- `refined` and `refined_rm`
- `residual_delta` and `delta_rm`
- `residual_gate` and `change_gate`
- `prior_reliability`
- `diagnostics`

Required diagnostics:

- `changed_pixel_rate`
- `mean_abs_delta`
- `boundary_delta_mean`
- `prior_reliability_mean`
- `change_gate_mean`
- `bootstrap_enabled`

