# Material Refine rebake_v2 Data Contract

## Scope

`rebake_v2` is the only contract allowed for future no-3D-FUTURE paper-stage candidate data. Legacy stored view targets and legacy UV target fields are not valid promotion evidence unless regenerated and audited under this contract.

This is a data contract only. It does not authorize R training, Phase4, paper-stage rehearsal, or paper claims.

## Required Record Assets

Each `rebake_v2` record must generate and reference:

- `canonical_buffer_root`
- per-view `uv`
- per-view `visibility`
- per-view `mask`
- per-view `roughness`
- per-view `metallic`
- `uv_target_roughness`
- `uv_target_metallic`
- `uv_target_confidence`
- `_field_sources.json`

The per-view target used for view-space supervision must be derived from the UV target sampled through the record's `view_uv` and valid `visibility/mask`, not from a legacy stored view target.

`scripts/prepare_material_refine_dataset.py --rebake-version rebake_v2` is the canonical data-side entrypoint for generating these fields. In this mode it disables render-cache reuse, disables prior-copy fallback, writes `_rebake_v2_contract.json`, and annotates the manifest with the target/view alignment metrics before promotion audit.

## Required Manifest Fields

Each `rebake_v2` record must write:

- `target_view_contract_version = "v2"`
- `rebake_version = "rebake_v2"`
- `stored_view_target_valid_for_paper`
- `prior_as_pred_pass`
- `target_as_pred_pass`
- `target_view_alignment_mean`
- `target_view_alignment_p95`
- `view_supervision_ready`
- `effective_view_supervision_rate`
- `target_source_type`
- `target_quality_tier`
- `target_prior_identity`
- `target_is_prior_copy`

## Hard Gates

A record cannot become `paper_pseudo` or `paper_strong` unless all of these are true:

- `target_view_contract_version == "v2"`
- `rebake_version == "rebake_v2"`
- `stored_view_target_valid_for_paper == true`
- `prior_as_pred_pass == true`
- `target_as_pred_pass == true`
- `target_view_alignment_mean < 0.03`
- `target_view_alignment_p95 < 0.08`
- `target_is_prior_copy == false`
- `copied_from_prior == false`
- `target_prior_identity <= 0.30` for paper-stage priority
- `target_source_type in {gt_render_baked, pseudo_from_multiview}`
- `uv_target_roughness`, `uv_target_metallic`, and `uv_target_confidence` exist
- license bucket is allowed for research training

## Promotion Rules

- `gt_render_baked` plus high confidence becomes `paper_strong`.
- `pseudo_from_multiview` plus high confidence and identity/alignment pass becomes `paper_pseudo`.
- Any contract, identity, confidence, quota, or license failure remains `diagnostic_only` or `rejected`.

Quota shortage must never force promotion. High `target_prior_identity` must never be relabeled as `paper_pseudo`. View-contract-unverified records must never become paper-ready.

## Balanced Candidate Policy

`stage1_v4_no3dfuture_rebake_v2_balanced` must satisfy:

- `metal_dominant >= 20%`
- `ceramic_glazed_lacquer >= 15%`
- `glass_metal >= 15%`
- `mixed_thin_boundary >= 15%`
- `glossy_non_metal <= 35%`
- `without_prior >= 15%`
- no single `source_name` above `60%`

If quota fails, the decision is `KEEP_AS_DATA_CANDIDATE_ONLY`. If target/view gate fails, the decision is `KEEP_AS_DIAGNOSTIC_ONLY`. Only a complete pass may become `READY_FOR_R_TRAINING_CANDIDATE`, and this still is not a paper claim.

The pilot builder first checks the no-3D-FUTURE material-family quota. If the pilot quota is impossible from the current candidate pool, GPU rebake is skipped and the 7-day longrun remains blocked. GPU0 rebake preparation starts only after the pilot composition is satisfiable.

## HDRI And Material Bank

PolyHaven HDRIs are lighting assets only. Each object must record reproducible `lighting_bank_id`, `lighting_asset_id`, and `lighting_stratum`.

PolyHaven material bank is material auxiliary/prior/diagnostic data only. It is not an object source and does not count toward no-3D-FUTURE main object quota.
