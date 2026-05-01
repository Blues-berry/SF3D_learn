from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .common import read_json, repo_path, repo_root, run_cmd
from .status import refresh_status


def resolve_next(config: dict[str, Any], args: Any, *, dry_run: bool) -> dict[str, Any]:
    cwd = repo_root(config)
    run_cmd(
        [
            sys.executable,
            "scripts/resolve_next_trainv5_b_batch.py",
            "--rolling-queue",
            str(Path(args.queue) if getattr(args, "queue", None) else repo_path(config, "rolling_queue")),
            "--min-launch-records",
            str(getattr(args, "min_launch_records", None) or config.get("min_launch_records", 256)),
        ],
        cwd=cwd,
        dry_run=dry_run,
    )
    return read_json(repo_path(config, "expansion_second_pass_dir") / "rolling_next_batch_decision.json", {})


def launch_status_payload(config: dict[str, Any], decision: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    brief = read_json(repo_path(config, "status_dir") / "pipeline_brief.json", {})
    queue = brief.get("queue", {}) if isinstance(brief, dict) else {}
    rebake = brief.get("rebake", {}) if isinstance(brief, dict) else {}
    return {
        "dry_run": dry_run,
        "batch_name": decision.get("batch_name") or queue.get("next_batch_name"),
        "queue_path": decision.get("queue_path"),
        "expected_record_count": decision.get("expected_record_count") or queue.get("next_frozen_launch_record_count"),
        "launchable_now": decision.get("launchable_now"),
        "reason": decision.get("reason") or queue.get("launch_gate_reason"),
        "blocking_reason": queue.get("launch_blocking_reason") or decision.get("blocking_reason") or "",
        "current_rebake_batch": rebake.get("batch_name"),
        "current_rebake_status": rebake.get("status"),
        "current_rebake_processed": rebake.get("processed"),
        "current_rebake_total": rebake.get("total"),
    }


def run_launch(args: Any, config: dict[str, Any]) -> None:
    cwd = repo_root(config)
    decision = resolve_next(config, args, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(launch_status_payload(config, decision, dry_run=True), indent=2, ensure_ascii=False))
    batch_name = args.batch_name or str(decision.get("batch_name") or "")
    queue_path = Path(args.queue) if args.queue else Path(str(decision.get("queue_path") or ""))
    expected = int(args.expected_record_count or decision.get("expected_record_count") or args.batch_size or 0)
    if batch_name and queue_path:
        run_cmd(
            [
                sys.executable,
                "scripts/build_material_refine_trainV5_abc.py",
                "--run-b-preflight",
                "--b-queue",
                str(queue_path),
                "--b-batch-name",
                batch_name,
                "--b-batch-size",
                str(args.batch_size or expected),
                "--b-expected-record-count",
                str(args.expected_record_count or 0),
                "--b-parallel-workers",
                str(args.parallel_workers or config.get("parallel_workers", 2)),
                "--b-render-resolution",
                str(args.render_resolution or config.get("render_resolution", 320)),
                "--b-cycles-samples",
                str(args.cycles_samples or config.get("cycles_samples", 8)),
                "--b-view-light-protocol",
                args.view_light_protocol or str(config.get("view_light_protocol", "production_32")),
            ],
            cwd=cwd,
            dry_run=args.dry_run,
        )
    run_cmd(
        [
            sys.executable,
            "scripts/maybe_launch_next_trainv5_b_batch.py",
            "--session-prefix",
            args.session_prefix,
        ],
        cwd=cwd,
        dry_run=args.dry_run,
    )
    refresh_status(config, dry_run=args.dry_run)

