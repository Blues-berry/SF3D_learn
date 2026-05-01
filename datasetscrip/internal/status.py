from __future__ import annotations

import sys
from typing import Any

from .common import repo_path, repo_root, run_cmd


def refresh_status(config: dict[str, Any], *, dry_run: bool = False) -> None:
    run_cmd(
        [
            sys.executable,
            "scripts/build_material_refine_trainV5_auto_status.py",
            "--output-dir",
            str(repo_path(config, "status_dir")),
            "--last-cycle-state",
            str(repo_path(config, "last_cycle_state")),
        ],
        cwd=repo_root(config),
        dry_run=dry_run,
    )


def brief_markdown_path(config: dict[str, Any]):
    return repo_path(config, "status_dir") / "pipeline_brief.md"

