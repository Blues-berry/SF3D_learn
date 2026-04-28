# R-only Rename And Semantics Patch

Date: 2026-04-26

## Scope

This patch reframes the material refinement layer as:

```text
Input Prior RM + UV Albedo/Normal + Canonical Multi-view Features -> R -> Refined RM
```

SF3D is now treated as one possible `prior_source_type`, not as the hard-coded baseline identity.

## Naming Rules

- User-facing reports use `Input Prior`, `Baseline Prior`, `Refined / Pred`, and `GT / Target`.
- `baseline_total_mae` is retained only as a compatibility field for older metrics readers.
- New summaries and W&B logs also emit `input_prior_total_mae`, `refined_total_mae`, and `gain_total`.
- `Gain = Input Prior error - Refined error`.

## Prior Labels

`scripts/export_material_validation_comparison_panels.py` now resolves dynamic labels:

- `sf3d_rm_texture` -> `SF3D RM Texture`
- `sf3d_scalar_broadcast` -> `SF3D Scalar Prior`
- `external_asset_rm_texture` -> `External Asset Prior`
- `external_asset_scalar_broadcast` -> `External Scalar Prior`
- `synthetic_degraded_prior` / `synthetic_degraded_from_target` -> `Synthetic Prior`
- `fallback_default` -> `Fallback Prior`
- `no_prior_placeholder` -> `No-prior Default`
- `prior_mode=scalar_rm` -> `Input Scalar Prior`
- `prior_mode=uv_rm` -> `Input UV Prior`

Fallback, synthetic, external, and no-prior samples are no longer displayed as SF3D.

## Report Changes

- Validation panels now show input RGB views, input prior roughness/metallic, GT roughness/metallic, predicted roughness/metallic, and error maps.
- Panel titles include `generator_id`, `source_name`, `material_family`, `prior_source_type`, `prior_mode`, `target_source_type`, `target_prior_identity`, `input_prior_total_mae`, `refined_total_mae`, `gain_total`, and `regression_flag`.
- Attribute comparison HTML now explains that the baseline is the input prior atlas used by R, not necessarily SF3D.
- Eval summaries include `by_prior_source_type` and `by_prior_mode`.

## Compatibility

Legacy fields are not removed:

- `baseline_total_mae`
- `baseline_roughness_mae`
- `baseline_metallic_mae`
- `baseline_uv_mae`

New preferred aliases:

- `input_prior_total_mae`
- `input_prior_uv_mae`
- `gain_total`

