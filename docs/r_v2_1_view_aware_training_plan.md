# R-v2.1 View-Aware Training Plan

## Scope

R-v2.1 is a recovery run for the current R-v2 failure mode: UV-object metrics improve while view/render rows regress. It is not a paper-stage claim path until the identity, target-alignment, clean-subset, and view/render gates pass.

## Required Gates

- Day1 `prior_as_pred` identity must pass.
- Day1 `target_as_pred` identity must pass, or Day2 target/view alignment must be repaired.
- `target_prior_identity_mean <= 0.30` for any subset used in paper-facing training.
- View metric must not regress: `view_gain >= 0` or `view_regression_rate < 0.50`.
- UV metric must remain positive: `uv_gain > 0` and `uv_improvement_rate > uv_regression_rate`.

## Training Changes

- Use sampled-view RM supervision from UV maps:
  - Sample `Pred UV RM` through `view_uv`.
  - Sample `GT UV RM` through the same `view_uv`.
  - Compare in visible view pixels only.
  - Do not depend on stored view RM PNGs for the loss.
- Keep `enable_no_prior_bootstrap=true`, but evaluate no-prior rows explicitly for constant collapse.
- Keep `enable_change_gate=true` and select checkpoints with view/render penalties.
- Keep `enable_boundary_safety=true` and monitor boundary bleed / gradient preservation.
- Lower or exclude high-identity samples:
  - `target_prior_identity > 0.70` stays diagnostic-only.
  - `target_prior_identity <= 0.30` is preferred for clean training.

## Config

Primary config:

`configs/material_refine_train_r_v2_1_view_aware.yaml`

Important switches:

- `enable_sampled_view_rm_loss: true`
- `sampled_view_rm_loss_weight: 0.82`
- `view_consistency_mode: required`
- `validation_selection_metric: uv_render_guarded`
- `selection_view_rm_penalty: 0.90`
- `fail_on_target_prior_identity: true`

## Day4 Pass Criteria

- UV gain is positive.
- View gain is non-negative, or view regression rate is below 0.50.
- UV improvement rate is higher than UV regression rate.
- Without-prior outputs are not constant-collapse in panels and per-case metrics.
- Panels include both improved and regressed examples with view-space rows.

## Stop Conditions

Stop training and return to audit if any of these occur:

- `prior_as_pred` identity fails.
- `target_as_pred` identity fails.
- The clean subset target identity mean exceeds 0.30.
- View regression rate remains 1.0.
- Predicted RM maps become constant and only appear to improve UV through target bias.
- Validation panels cannot explain prior / GT / pred / view differences.
- Eval summary fields mix UV and view-level meanings again.
