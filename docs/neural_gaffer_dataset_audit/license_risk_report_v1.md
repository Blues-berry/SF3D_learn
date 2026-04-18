# License Risk Report v1

## Buckets

- `benchmark_project_terms_unstated`: Stanford-ORB
- `cc0`: PolyHaven_HDRI
- `cc_by_4_0`: OpenIllumination
- `cc_by_nc_4_0`: ABO_locked_core
- `custom_form_gated_license`: 3DCoMPaT++
- `custom_portal_terms_unstated`: OpenSVBRDF
- `custom_research_project_terms`: Laval_HDR
- `custom_tianchi_terms`: 3D-FUTURE_candidate
- `license_not_posted_preprint_project`: ICTPolarReal
- `license_not_posted_project_page`: OLATverse
- `mixed_cc0_cc_by`: MatSynth
- `mixed_per_object_license`: Objaverse-XL_filtered_candidate
- `portal_terms_plus_dataset_specific`: OmniObject3D

## Highest Risk

- `mixed_per_object_license`: Objaverse-XL must stay per-object and cannot be pushed into a single release bundle without filtering by object license
- `cc_by_nc_4_0`: ABO is usable for internal training and research, but must remain isolated from freer redistribution buckets
- custom/gated sources (`3D-FUTURE_candidate`, `3DCoMPaT++`, `OmniObject3D`, `Stanford-ORB`, `ICTPolarReal`, `OLATverse`, `OpenSVBRDF`, `Laval_HDR`) all need project-side term checks before public release

## Low Risk

- `cc0`: Poly Haven HDRIs are the cleanest natural light source for Pool-D
- `cc_by_4_0`: OpenIllumination is the cleanest Pool-B pilot source from a release perspective
- `mixed_cc0_cc_by`: MatSynth is manageable if material assets stay tagged and split by license bucket
