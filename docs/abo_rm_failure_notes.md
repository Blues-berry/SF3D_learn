# ABO RM Failure Notes

This is an initial roughness/metallic error pass for SF3D on a tiny ABO-derived probe set.
The goal here is not to lock the pipeline yet, but to identify the dominant failure modes
before scaling to a 200-object mini benchmark.

## Current Probe

- Objects: 6 ABO GLB assets
- Views per object: 3
- Input images: Blender Cycles renders with transparent background
- GT source: image-space metallic / roughness passes rendered from the original ABO glTF PBR materials
- SF3D run: local baseline with `texture_resolution=256`
- Output bundle: `output/abo_material_probe_run1/`

The exploratory scripts used for this pass are:

- `scripts/abo_material_passes_blender.py`
- `scripts/abo_material_probe.py`

## What the quick pass already shows

- SF3D is not just noisy on RM. It fails in a structured way.
- The dominant issue is not one single bias; it is a combination of:
  mixed-material averaging,
  metal / non-metal polarity flips,
  view-dependent highlight confusion,
  and edge-driven material leakage.
- A scalar RM baseline is already enough to expose the gap, especially on ABO objects
  with thin metallic parts, reflective boundaries, glass, or mixed fabric/metal layouts.

## Failure Types

### 1. Over-smoothing

Typical pattern:
- GT roughness has strong local variation, but prediction collapses toward a single mid value.
- This is most visible on fabric or multi-part assets where boundary, tuft, or frame regions differ.

Representative cases:

- `faux_linen_headboard / front_studio`
  - pred `R/M = 0.627 / 1.000`
  - GT mean `R/M = 0.687 / 0.121`
  - GT std `R/M = 0.302 / 0.089`
- `faux_linen_headboard / three_quarter_indoor`
  - pred `R/M = 0.569 / 1.000`
  - GT mean `R/M = 0.897 / 0.170`

Interpretation:
- High-roughness fabric is being compressed toward a smoother, more game-metal look.
- The model appears to average away local roughness structure instead of respecting the dominant fabric prior.

### 2. Metal / Non-metal Confusion

Typical pattern:
- Non-metal objects are pushed to `metallic ~= 1.0`, or mixed metal objects collapse to `metallic ~= 0.0`.
- The failure happens in both directions.

Representative cases:

- `wood_iron_mirror / side_neon`
  - pred `R/M = 0.380 / 1.000`
  - GT mean `R/M = 0.457 / 0.085`
- `faux_linen_headboard / front_studio`
  - pred `R/M = 0.627 / 1.000`
  - GT mean `R/M = 0.687 / 0.121`
- `lamp_metal_glass / front_studio`
  - pred `R/M = 0.810 / 0.000`
  - GT mean `R/M = 0.302 / 0.255`
- `lamp_metal_glass / three_quarter_indoor`
  - pred `R/M = 0.195 / 0.000`
  - GT mean `R/M = 0.306 / 0.264`

Interpretation:
- For fabric-heavy objects with small hard parts, the model can flip into an all-metal explanation.
- For mixed metal/glass objects, it can do the opposite and suppress metallic almost entirely.

### 3. Local Highlight Misread

Typical pattern:
- Bright local highlights do not stay as view-dependent illumination cues.
- Instead, they appear to distort RM inference, usually by pushing roughness away from the GT value.

Representative cases:

- `ceramic_stool / side_neon`
  - input brightness `p99 = 0.887`
  - pred `R/M = 0.403 / 0.000`
  - GT mean `R/M = 0.221 / 0.006`
- `ceramic_stool / front_studio`
  - input brightness `p99 = 0.935`
  - pred `R/M = 0.280 / 0.000`
  - GT mean `R/M = 0.221 / 0.006`
- `lamp_metal_glass / front_studio`
  - input brightness `p99 = 0.844`
  - pred `R/M = 0.810 / 0.000`
  - GT mean `R/M = 0.302 / 0.255`

Interpretation:
- On glossy non-metal ceramic, local glints are enough to lift predicted roughness well above GT.
- On the lamp, bright metal/glass highlights do not help metallic inference; they instead correlate with a much rougher, more matte prediction.

### 4. Boundary Bleed

Typical pattern:
- Thin metallic or reflective boundary structures leak into whole-object RM prediction.
- Edge statistics are much more metallic than the interior, and the prediction follows the edge cue too strongly.

Representative cases:

- `wood_iron_mirror / side_neon`
  - GT metallic edge mean `0.252`
  - GT metallic interior mean `0.016`
  - pred metallic `1.000`
- `wood_iron_mirror / three_quarter_indoor`
  - GT metallic edge mean `0.294`
  - GT metallic interior mean `0.088`
  - pred metallic `1.000`
- `leather_metal_stool / three_quarter_indoor`
  - GT metallic edge mean `0.472`
  - GT metallic interior mean `0.005`
  - pred metallic `0.327`

Interpretation:
- Thin frame / border geometry is disproportionately influential.
- This is exactly the kind of issue we should expect to see again when moving from scalar RM to map prediction unless alignment and visibility handling are done carefully.

## High-level Gap Summary

Object-level mean absolute error from the current 18-view probe:

- `lamp_metal_glass`: roughness `0.254`, metallic `0.211`
- `ceramic_stool`: roughness `0.151`, metallic `0.006`
- `leather_metal_stool`: roughness `0.115`, metallic `0.070`
- `wood_iron_mirror`: roughness `0.092`, metallic `0.722`
- `metal_desk`: roughness `0.264`, metallic `0.164`
- `faux_linen_headboard`: roughness `0.199`, metallic `0.628`

Takeaway:
- Metallic is the bigger failure channel on mixed-material and boundary-heavy objects.
- Roughness is the bigger failure channel on glossy / high-frequency appearance cues, especially when view lighting changes.

## What this means for the mini pipeline

Before scaling to 200 objects, the mini dataset should intentionally include:

- metal-dominant objects
- non-metal glossy objects
- mixed fabric/metal objects
- thin-boundary metallic structures
- mirror / glass or reflective trim cases

The current probe already suggests the alignment script should export not only whole-mask GT means,
but also:

- edge vs interior RM statistics
- visible-region RM variance
- highlight statistics from the input render

Those extra signals make the failure taxonomy much easier to stabilize.
