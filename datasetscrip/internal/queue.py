from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .common import repo_path, repo_root, run_cmd
from .status import refresh_status


def run_queue(args: Any, config: dict[str, Any]) -> None:
    cwd = repo_root(config)
    second_pass_dir = repo_path(config, "expansion_second_pass_dir")
    input_manifest = Path(args.input_manifest) if args.input_manifest else repo_path(config, "merged_expansion_manifest")
    run_cmd(
        [
            sys.executable,
            "scripts/build_material_refine_trainV5_expansion_second_pass.py",
            "--input-manifest",
            str(input_manifest),
            "--output-dir",
            str(second_pass_dir),
        ],
        cwd=cwd,
        dry_run=args.dry_run,
    )
    run_cmd(
        [
            sys.executable,
            "scripts/build_material_refine_trainV5_repair_and_expansion_plan.py",
            "--second-pass-dir",
            str(second_pass_dir),
            "--output-dir",
            str(second_pass_dir / "repaired_second_pass"),
        ],
        cwd=cwd,
        dry_run=args.dry_run,
    )
    run_cmd(
        [
            sys.executable,
            "scripts/resolve_next_trainv5_b_batch.py",
            "--rolling-queue",
            str(Path(args.rolling_queue) if args.rolling_queue else repo_path(config, "rolling_queue")),
            "--min-launch-records",
            str(args.min_launch_records or config.get("min_launch_records", 256)),
        ],
        cwd=cwd,
        dry_run=args.dry_run,
    )
    refresh_status(config, dry_run=args.dry_run)

