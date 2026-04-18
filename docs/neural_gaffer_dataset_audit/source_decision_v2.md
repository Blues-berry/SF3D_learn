# Source Decision v2

## Working Rule

- keep existing true source assets intact
- do not bulk download before pilot audit
- split every output and future release by `license_bucket`
- keep a single source-of-truth path for each asset source

## Pool Decisions

- `pool_A_direct_object_supervision`: keep `ABO_locked_core` as the only admitted object-level training core right now; add `3D-FUTURE_candidate` and `Objaverse-XL_filtered_candidate` only as pilot extensions pending audit
- `pool_B_controlled_highlight_supervision`: treat `OLATverse`, `OpenIllumination`, and `ICTPolarReal` as signal-check sources first, not bulk downloads
- `pool_C_part_material_highlight_semantics`: keep `3DCoMPaT++` as a gated follow-up source for part/material semantics
- `pool_D_natural_light_hdri_bank`: Poly Haven can be curated immediately from the official API; Laval stays in controlled slot planning until manual selection is confirmed
- `pool_E_material_prior_bank`: keep `OpenSVBRDF` and `MatSynth` separate from Pool-A and use them only as material prior sources
- `pool_F_real_world_eval_holdout`: reserve `Stanford-ORB` and `OmniObject3D` strictly for holdout evaluation

## Pilot Readiness

- ABO locked core ready now: 500 A-ready objects already present in the mini-v1 candidate pool
- 3D-FUTURE pilot queued now: 30 locally available objects
- Objaverse pilot queued now: 30 metadata-only candidates; asset access still blocked
- HDRI bank seed count: 20 Poly Haven + 20 Laval slots

## License Notes

- ABO is non-commercial on the official AWS registry, so it must remain a separate release bucket
- Objaverse-XL is per-object licensed; do not mix unknown object licenses into a single public bundle
- 3D-FUTURE, 3DCoMPaT++, OmniObject3D, Stanford-ORB, ICTPolarReal, and OLATverse all need custom/gated terms tracked independently

## Stop Condition

- after these pilots, stop before bulk expansion and report Pool-A / Pool-B / Pool-D suitability and risk
