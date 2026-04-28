# Material Refine Stage1-v4 No-3D-FUTURE Automation

## Scope

This is a data-only pipeline. It does not launch training and it keeps GPU1 idle. Heavy render/preprocess output is written under `/4T/CXY/sf3d_material_refine_dataset_factory`.

## Active Policy

- Future expansion excludes `3D-FUTURE_highlight_local_8k`.
- GPU render/preprocess is locked to GPU0.
- Promotion is enabled for `ABO_locked_core` and `Objaverse-XL_strict_filtered_increment`.
- Stage1-v4 output root is `output/material_refine_paper/stage1_v4_no3dfuture_latest`.
- HDRI/material assets remain auxiliary banks and are not mixed into object supervision.
- Rejected generated bundles are quarantined to `/4T/CXY/sf3d_material_refine_reject_trash` after safety checks; original source assets are not deleted by default.

## One-Command Restart

```bash
bash scripts/restart_material_refine_no3dfuture_v4_gpu0.sh
```

The restart script stops old data tmux sessions and orphan data processes, validates configs/scripts, and launches the 7-day data supervisor.

## Automated Loop

The supervisor launches `scripts/run_material_refine_dataset_factory.py` with:

- source refresh/download staging
- canonical manifest generation
- second-pass target promotion
- Stage1-v4 no-3D-FUTURE balanced/diagnostic/OOD/real-lighting subset build
- HDRI usage audit
- reject quarantine
- post-ingest cleanup
- GPU0 longrun render/preprocess
- manifest quality audit

## Current Gate

Stage1-v4 remains candidate-only until balanced records reach the configured paper gate:

- `min_paper_eligible >= 800`
- material families meet quota/min-count checks
- no-prior and secondary-source records meet minimum counts
- target/confidence/view gates pass
