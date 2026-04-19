# Material Refine Paper Protocol

## Core Claim

The paper claim should not be "SF3D gets better on one dataset." A stronger top-conference framing is:

> A lightweight generator-agnostic post-refinement module improves roughness/metallic material quality for single-image 3D asset generators after their outputs are canonicalized into a shared renderable representation.

This keeps the method scoped to `G -> C -> R`:

- `G`: upstream 3D asset generator, e.g. SF3D, SPAR3D, Hunyuan3D, or future renderable asset generators.
- `C`: canonicalization layer that normalizes mesh, UV, material priors, view buffers, target maps, confidence, provenance, and license buckets.
- `R`: lightweight material refiner that consumes only `CanonicalAssetRecordV1`, not generator-specific internals.

## Experimental Axes

The main paper tables should report every metric across these axes:

- `generator_id`: SF3D first, then SPAR3D/Hunyuan3D once adapters are available.
- `prior_label`: with material prior vs no material prior.
- `supervision_tier`: strong GT, pseudo GT, auxiliary highlight, eval-only.
- `source_name`: ABO, 3D-FUTURE, Objaverse-XL, and controlled auxiliary sets.
- `license_bucket`: kept visible for reproducibility and release constraints.
- `material_tag`: metal-dominant, ceramic/glazed/lacquer, glass+metal, mixed thin-boundary, glossy non-metal.

## Baselines

Minimum accepted baselines:

- `G raw`: upstream generator output with original roughness/metallic factors or maps.
- `G canonical prior`: same upstream output after canonicalization, before refinement.
- `Scalar broadcast`: roughness/metallic factors broadcast to UV maps without refinement.
- `Prior smoothing`: simple edge-aware or bilateral smoothing over prior RM maps.
- `No-view refiner`: refiner without multi-view RGB/depth/normal/position buffers.
- `No-prior refiner`: force prior dropout for all samples to test mode-B ability.
- `No-residual refiner`: direct RM prediction instead of residual refinement.
- `Ours full`: multi-view encoder, UV fusion, prior/coarse initialization, residual head.

Cross-generator baselines:

- Train on SF3D-only and test on SF3D.
- Train on SF3D-only and test on SPAR3D/Hunyuan3D canonical outputs.
- Train on mixed generators and test per generator.
- Leave-one-generator-out: train on all but one generator, test on the held-out generator.

## Metrics

UV-space metrics:

- confidence-weighted roughness MAE
- confidence-weighted metallic MAE
- confidence-weighted total RM MAE
- edge-band RM MAE around albedo/normal/material boundaries
- high-frequency preservation score, measured by gradient correlation against confident targets

View-space metrics:

- canonical 3-view roughness/metallic MAE
- stress-view roughness/metallic MAE under held-out HDRI/view bundles
- boundary bleed score: edge-band error minus interior error
- metal/non-metal confusion: metallic threshold F1, AUROC, and confusion rate
- highlight localization error: top-k highlight mask alignment under controlled lighting

Rendered appearance metrics:

- PSNR/SSIM/LPIPS between baseline/refined renders and GT renders under fixed HDRI bank
- specular highlight IoU or center-of-mass error where controlled GT highlights exist
- human/user preference study on a small blinded set, optional but useful for final submission

System metrics:

- training examples/sec
- eval assets/min
- GPU memory peak
- extra inference latency after upstream generator
- GLB validity: exported asset has roughness/metallic textures and preserves baseColor/normal textures

## Required Tables

Table 1: Main benchmark.

- Rows: baseline methods and ours.
- Columns: UV RM MAE, view RM MAE, boundary bleed, metal confusion, rendered LPIPS/SSIM, runtime.

Table 2: Cross-generator generalization.

- Rows: train generator mix.
- Columns: test on SF3D, SPAR3D, Hunyuan3D, and no-prior generator outputs.

Table 3: Ablations.

- Remove prior, remove multi-view, remove UV fusion, remove residual head, remove confidence weighting, remove edge-aware loss, remove prior dropout curriculum.

Table 4: Material category breakdown.

- Rows: metal, ceramic/glazed/lacquer, glass+metal, mixed thin-boundary, glossy non-metal.
- Columns: error reduction and failure-tag reduction.

Table 5: Data supervision breakdown.

- Rows: strong GT, pseudo GT, auxiliary highlight, material prior, OOD eval.
- Columns: per-metric reliability and confidence-weighted performance.

## Required Figures

Figure 1: Method overview.

- Show `G -> C -> R`, canonical manifest, multi-view buffers, UV fusion, residual RM refinement, and GLB export.

Figure 2: Qualitative comparison.

- Input image, upstream GLB render, refined GLB render, roughness before/after, metallic before/after, error maps.

Figure 3: Cross-generator examples.

- Same panel layout for SF3D, SPAR3D, Hunyuan3D.

Figure 4: Failure taxonomy.

- Representative successes and failures for `metal_nonmetal_confusion`, `boundary_bleed`, `local_highlight_misread`, and `over_smoothing`.

Figure 5: Data/metric reliability.

- Plot target/prior identity rate, confidence distribution, and per-source metric variance.

## W&B Run Layout

Use one project and stable groups:

- `paper-stage1`: main training runs and checkpoint artifacts.
- `paper-ablation`: ablation runs with one variable changed at a time.
- `paper-cross-generator`: train/test generator transfer runs.
- `paper-benchmark`: final eval runs and report artifacts.
- `paper-figures`: final selected panels, report HTML, and table CSV exports.

Each run must log:

- config snapshot
- manifest snapshot or manifest digest
- git SHA and dirty-state summary
- dataset summary by generator/source/tier/license/prior
- train/val/test metrics by generator/source/prior/material tag
- checkpoint artifact
- HTML report artifact
- validation comparison panels

## Multi-Upstream Adapter Contract

Each generator adapter only needs to emit `CanonicalAssetRecordV1` records and canonical bundles:

- `generator_id`: stable ID such as `sf3d`, `spar3d`, `hunyuan3d_2_1`.
- `generator_bundle_root`: original output root for traceability.
- `canonical_mesh_path` and `canonical_glb_path`.
- `uv_albedo_path` and `uv_normal_path` if present or generated by `C`.
- `uv_prior_roughness_path` and `uv_prior_metallic_path` if the generator has material priors.
- `scalar_prior_roughness` and `scalar_prior_metallic` for scalar-only generators.
- `prior_mode`: `uv_rm`, `scalar_rm`, or `none`.
- canonical buffers: `rgba`, `mask`, `depth`, `normal`, `position`, optional `uv`, rendered RM targets if available.

The training code must not branch on generator-specific files. It may filter or report by `generator_id`, but `R` should consume the same tensors for every upstream.

## Current Gaps Before Paper Claims

- Current audited manifest has target/prior identity at 100%, so it is only a smoke-test corpus.
- View UV buffers are still missing in the current canonical bundles, so view-to-UV consistency is not yet fully active.
- Rendered appearance metrics are not yet implemented in the eval script.
- SPAR3D/Hunyuan3D adapters are interface-ready but not yet staged or benchmarked.
- Material-tag fields need to be normalized in manifests so paper tables can group by category without filename heuristics.
