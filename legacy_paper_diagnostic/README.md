# Legacy Paper Diagnostic

This directory stores historical material-refine paper diagnostics, old stage
builders, round logs, v1-fixed artifacts, paper-only configs, and source audit
records. It is intentionally separate from the TrainV5 active engineering
pipeline.

Active TrainV5 scripts must not import from this directory or call scripts here
as part of routine data processing, training, or evaluation. Use this tree only
for audit traceability, paper-result interpretation, or historical debugging.

See `MIGRATION_MANIFEST.json` for the moved file list.
