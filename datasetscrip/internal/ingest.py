from __future__ import annotations

import sys
from typing import Any

from .common import repo_root, run_cmd
from .status import refresh_status


def run_ingest(args: Any, config: dict[str, Any]) -> None:
    sources = args.sources or str(config.get("objaverse_sources", "smithsonian,thingiverse,sketchfab"))
    materials = args.priority_material_families or ",".join(config.get("material_priority", []))
    cmd = [
        sys.executable,
        "scripts/stage_material_refine_material_priority_sources.py",
        "--download-mode",
        args.download_mode or str(config.get("download_mode", "direct")),
        "--download-probe-size",
        str(args.download_probe_size or config.get("download_probe_size", 100)),
        "--min-download-success-rate",
        str(args.min_download_success_rate or config.get("min_download_success_rate", 0.2)),
        "--max-retry-rounds",
        str(args.max_retry_rounds or config.get("max_retry_rounds", 2)),
        "--max-actions-per-run",
        str(args.max_actions_per_run or config.get("max_actions_per_run", 4)),
        "--objaverse-target",
        str(args.objaverse_target or config.get("objaverse_target", 1200)),
        "--objaverse-sources",
        sources,
        "--priority-material-families",
        materials,
        "--target-material-family-ratios",
        args.target_material_family_ratios or str(config.get("target_material_family_ratios", "")),
    ]
    if args.download_objaverse:
        cmd.append("--download-objaverse")
    if args.skip_merge:
        cmd.append("--skip-merge")
    if args.skip_second_pass:
        cmd.append("--skip-second-pass")
    if args.skip_repair:
        cmd.append("--skip-repair")
    run_cmd(cmd, cwd=repo_root(config), dry_run=args.dry_run)
    refresh_status(config, dry_run=args.dry_run)

