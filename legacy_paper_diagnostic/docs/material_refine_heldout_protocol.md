# Material Refine Heldout Protocol

## Goal

The heldout protocol must make later results interpretable, reproducible, and generator-aware.

This protocol fixes three evaluation layers:

- `train/val/test`: main paper-stage subset
- `material_sensitive_eval`: reserved analysis split for hard cases
- `ood_eval`: out-of-distribution source split

## Split Principles

1. Splits are object-level and written to a versioned split file.
2. No hidden re-hashing during later training runs.
3. `source_name`, `generator_id`, `has_material_prior`, `supervision_tier`, `category_bucket`, and `license_bucket` are audited per split.
4. `ood_eval` is source-held-out and never mixed into main training.
5. `material_sensitive_eval` is a deterministic heldout carved from paper-eligible main-train sources.

## Current v1 Defaults

- Main train sources:
  - `ABO_locked_core`
- OOD sources:
  - `3D-FUTURE_candidate`
  - `Objaverse-XL_filtered_candidate`

These defaults are conservative:

- `ABO_locked_core` is the current strongest in-repo source for main supervised experiments.
- `3D-FUTURE_candidate` and future `Objaverse-XL_filtered_candidate` are preserved as transfer/OOD probes unless explicitly promoted later.

## Material-Sensitive Holdout

Current manifests do not yet carry a robust material taxonomy field for every record.

Therefore the current P0 implementation uses a deterministic proxy:

- group by `category_bucket`
- if no explicit category/material bucket exists, fall back to `source_name|prior_label`
- reserve a small fixed fraction for `material_sensitive_eval`

This is a provisional protocol for P0.

For the final paper submission, the data pipeline should populate explicit material-category tags so `material_sensitive_eval` can be truly category-aware rather than proxy bucket-aware.

## Reproducibility

The split file must be treated as immutable once a paper-stage round begins.

Artifacts to preserve:

- split JSON
- split audit HTML/JSON
- readiness summary
- manifest digest
- config and git SHA

## Interpretation Rules

- `engineering smoke` results may use any manifest, including `smoke_only` targets.
- `valid experiment` requires non-trivial targets and a fixed split file.
- `paper-stage result` requires:
  - readiness gate pass
  - fixed split
  - stage1 subset manifest
  - explicit OOD and material-sensitive evaluation partitions

## Current Known Limitation

Because the current manifest still has `target_prior_identity=1.0`, the protocol can be materialized now, but the stage1 subset remains blocked until non-trivial targets arrive from the data process.
