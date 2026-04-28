# Material Refine Manifest Schema

## Scope

This document defines the paper-stage manifest extensions for `CanonicalAssetRecordV1`.

The goal is to distinguish three states clearly:

- `engineering smoke`: bundle exists and code can run
- `valid experiment`: target is non-trivial and supervision is auditably meaningful
- `paper-stage result`: dataset passes readiness gates and fixed split protocol

## Required Core Fields

Existing core fields remain unchanged:

- `object_id`
- `generator_id`
- `license_bucket`
- `has_material_prior`
- `prior_mode`
- `canonical_mesh_path`
- `canonical_glb_path`
- `uv_albedo_path`
- `uv_normal_path`
- `uv_prior_roughness_path`
- `uv_prior_metallic_path`
- `scalar_prior_roughness`
- `scalar_prior_metallic`
- `uv_target_roughness_path`
- `uv_target_metallic_path`
- `uv_target_confidence_path`
- `canonical_views_json`
- `canonical_buffer_root`
- `supervision_tier`
- `default_split`
- `bundle_root`

## Dataset Governance Fields

The following fields are required to separate paper-stage supervision from auxiliary or benchmark-only pools:

- `supervision_role`
  Allowed values:
  - `paper_main`
  - `auxiliary_upgrade_queue`
  - `auxiliary_highlight`
  - `material_prior`
  - `lighting_bank`
  - `benchmark_ood`
  - `unknown`
- `paper_split`
  Object-level split label. Typical values:
  - `paper_train`
  - `paper_val_iid`
  - `paper_test_iid`
  - `paper_test_material_holdout`
  - `paper_test_ood_object`
  - `paper_test_real_lighting`
  - `excluded`
- `material_family`
  Allowed values:
  - `metal_dominant`
  - `ceramic_glazed_lacquer`
  - `glass_metal`
  - `mixed_thin_boundary`
  - `glossy_non_metal`
  - `unknown`
- `thin_boundary_flag`
  Boolean.
- `lighting_bank_id`
  Stable identifier for the canonical lighting protocol or real-lighting bank used by the record.
- `view_supervision_ready`
  Boolean. `true` means the record has real view-to-UV supervision inputs; placeholder buffers must not set this to `true`.
- `valid_view_count`
  Number of effective views available for target construction or multi-view supervision.

## P0 Target-Quality Fields

The following fields are required for paper-stage readiness:

- `target_source_type`
  Allowed values:
  - `gt_render_baked`
  - `pseudo_from_multiview`
  - `pseudo_from_material_bank`
  - `copied_from_prior`
  - `unknown`
- `target_is_prior_copy`
  Boolean. `true` means the current target is effectively a copy of the prior and may only be used for smoke/debug. This is raised both for exact file matches and for high-identity targets whose `target_prior_identity >= 0.95`.
- `target_quality_tier`
  Allowed values:
  - `paper_strong`
  - `paper_pseudo`
  - `research_only`
  - `smoke_only`
  - `unknown`
- `target_confidence_summary`
  JSON object with at least:
  - `mean`
  - `std`
  - `min`
  - `p50`
  - `max`
  - `nonzero_rate`
  - `high_conf_rate`
- `target_prior_identity`
  Scalar in `[0, 1]` describing how close the current target remains to the prior, where `1.0` is identical.
- `target_confidence_mean`
- `target_confidence_nonzero_rate`
- `target_coverage`

## Tier Semantics

- `paper_strong`
  Reserved for strong, non-trivial targets such as trusted GT render baking with sufficient confidence coverage.
- `paper_pseudo`
  Non-trivial pseudo targets, typically multi-view baked targets, with adequate confidence coverage and no prior-copy leakage.
- `research_only`
  Useful for exploratory experiments or auxiliary tasks, but not strong enough to support the main paper claim.
- `smoke_only`
  Only valid for engineering smoke tests. `copied_from_prior` samples belong here by default.

## Paper-Stage Readiness Rules

Main training is blocked unless all of the following are true:

- `target_prior_identity_rate <= max_target_prior_identity_rate_for_paper`
  Here `target_prior_identity_rate` means the fraction of records flagged as high-identity / prior-copy, not the mean of `target_prior_identity`.
- `paper_stage_eligible_records >= min_nontrivial_target_count_for_paper`
- the fixed split protocol has been materialized
- the subset manifest only contains non-trivial, confidence-qualified, license-allowed records

## Buffer Validation Rules

The audit/validator distinguishes two notions:

- `effective_view_supervision`
  Minimum fields needed for actual view-to-UV supervision:
  - `rgba`
  - `uv`
  - `roughness`
  - `metallic`
- `strict_complete`
  Full canonical view supervision buffer:
  - `rgba`
  - `mask`
  - `depth`
  - `normal`
  - `position`
  - `uv`
  - `visibility`
  - `roughness`
  - `metallic`

If `effective_view_supervision_rate` is zero, `view_consistency` must be explicitly disabled or treated as unavailable.
