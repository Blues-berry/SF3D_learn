# SF3D Material-Refine Dataset Storage Policy

Effective: 2026-04-24

## Principle

Large dataset artifacts must not be written to the root-backed project tree by default.

Use `/4T` for future heavy outputs:

- Render/preprocess outputs: `/4T/CXY/sf3d_material_refine_dataset_factory`
- Auxiliary large downloads: `/4T/CXY/sf3d_material_refine_aux_downloads`

Keep the repository `output/` tree for lightweight artifacts only:

- manifests
- promotion reports
- audit summaries
- HTML/Markdown summaries
- compatibility indexes or symlinks when needed

## Active Runs

Do not move currently running dataset jobs while Blender/process workers are active. Existing root-backed runs may finish in place, then later be archived or migrated during a maintenance window.

## Enforcement

The dataset factory and 7-day supervisor configs declare this policy. New automatic longrun render/preprocess jobs should use the `/4T` render root, while promotion and audit code can read both repo-local and `/4T` manifests.
