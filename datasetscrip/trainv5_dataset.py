#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from internal import pipeline


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = THIS_DIR / "trainv5_dataset_config.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path, {})
    if not isinstance(config, dict):
        raise SystemExit(f"invalid_config:{path}")
    config.setdefault("repo_root", str(THIS_DIR.parent))
    return config


def cmd_status(args: argparse.Namespace, config: dict[str, Any]) -> None:
    pipeline.refresh_status(config, dry_run=args.dry_run)
    brief_path = pipeline.repo_path(config, "status_dir") / "pipeline_brief.md"
    if brief_path.exists():
        print(brief_path.read_text(encoding="utf-8"))


def cmd_ingest(args: argparse.Namespace, config: dict[str, Any]) -> None:
    pipeline.run_ingest(args, config)


def cmd_queue(args: argparse.Namespace, config: dict[str, Any]) -> None:
    pipeline.run_queue(args, config)


def cmd_launch(args: argparse.Namespace, config: dict[str, Any]) -> None:
    pipeline.run_launch(args, config)


def cmd_finalize(args: argparse.Namespace, config: dict[str, Any]) -> None:
    pipeline.run_finalize(args, config)


def cmd_refresh(args: argparse.Namespace, config: dict[str, Any]) -> None:
    cmd_queue(args, config)
    cmd_launch(args, config)


def cmd_supervisor(args: argparse.Namespace, config: dict[str, Any]) -> None:
    iteration = 0
    while True:
        iteration += 1
        print(f"[{utc_now()}] trainv5_dataset supervisor iteration={iteration}")
        if args.with_ingest:
            cmd_ingest(args, config)
        if args.with_queue:
            cmd_queue(args, config)
        cmd_launch(args, config)
        cmd_status(args, config)
        if args.once:
            return
        time.sleep(max(float(args.interval_seconds), 30.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status")
    status.add_argument("--dry-run", action="store_true")
    status.set_defaults(func=cmd_status)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--dry-run", action="store_true")
    ingest.add_argument("--download-objaverse", action="store_true")
    ingest.add_argument("--download-mode", choices=["direct", "proxy-probe", "mirror-probe", "off"])
    ingest.add_argument("--download-probe-size", type=int)
    ingest.add_argument("--min-download-success-rate", type=float)
    ingest.add_argument("--max-retry-rounds", type=int)
    ingest.add_argument("--max-actions-per-run", type=int)
    ingest.add_argument("--objaverse-target", type=int)
    ingest.add_argument("--sources")
    ingest.add_argument("--priority-material-families")
    ingest.add_argument("--target-material-family-ratios")
    ingest.add_argument("--skip-merge", action="store_true")
    ingest.add_argument("--skip-second-pass", action="store_true")
    ingest.add_argument("--skip-repair", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    queue = sub.add_parser("queue")
    queue.add_argument("--dry-run", action="store_true")
    queue.add_argument("--input-manifest")
    queue.add_argument("--rolling-queue")
    queue.add_argument("--min-launch-records", type=int)
    queue.set_defaults(func=cmd_queue)

    launch = sub.add_parser("launch")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--queue")
    launch.add_argument("--batch-name")
    launch.add_argument("--batch-size", type=int)
    launch.add_argument("--expected-record-count", type=int)
    launch.add_argument("--min-launch-records", type=int)
    launch.add_argument("--parallel-workers", type=int)
    launch.add_argument("--render-resolution", type=int)
    launch.add_argument("--cycles-samples", type=int)
    launch.add_argument("--view-light-protocol")
    launch.add_argument("--session-prefix", default="trainv5_rolling")
    launch.set_defaults(func=cmd_launch)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--dry-run", action="store_true")
    finalize.add_argument("--batch")
    finalize.add_argument("--full-manifest")
    finalize.set_defaults(func=cmd_finalize)

    refresh = sub.add_parser("refresh")
    refresh.add_argument("--dry-run", action="store_true")
    refresh.add_argument("--input-manifest")
    refresh.add_argument("--rolling-queue")
    refresh.add_argument("--queue")
    refresh.add_argument("--batch-name")
    refresh.add_argument("--batch-size", type=int)
    refresh.add_argument("--expected-record-count", type=int)
    refresh.add_argument("--min-launch-records", type=int)
    refresh.add_argument("--parallel-workers", type=int)
    refresh.add_argument("--render-resolution", type=int)
    refresh.add_argument("--cycles-samples", type=int)
    refresh.add_argument("--view-light-protocol")
    refresh.add_argument("--session-prefix", default="trainv5_rolling")
    refresh.set_defaults(func=cmd_refresh)

    supervisor = sub.add_parser("supervisor")
    supervisor.add_argument("--dry-run", action="store_true")
    supervisor.add_argument("--once", action="store_true")
    supervisor.add_argument("--interval-seconds", type=float, default=1800.0)
    supervisor.add_argument("--with-ingest", action="store_true")
    supervisor.add_argument("--with-queue", action="store_true")
    supervisor.add_argument("--download-objaverse", action="store_true")
    supervisor.add_argument("--download-mode", choices=["direct", "proxy-probe", "mirror-probe", "off"])
    supervisor.add_argument("--download-probe-size", type=int)
    supervisor.add_argument("--min-download-success-rate", type=float)
    supervisor.add_argument("--max-retry-rounds", type=int)
    supervisor.add_argument("--max-actions-per-run", type=int)
    supervisor.add_argument("--objaverse-target", type=int)
    supervisor.add_argument("--sources")
    supervisor.add_argument("--priority-material-families")
    supervisor.add_argument("--target-material-family-ratios")
    supervisor.add_argument("--skip-merge", action="store_true")
    supervisor.add_argument("--skip-second-pass", action="store_true")
    supervisor.add_argument("--skip-repair", action="store_true")
    supervisor.add_argument("--input-manifest")
    supervisor.add_argument("--rolling-queue")
    supervisor.add_argument("--queue")
    supervisor.add_argument("--batch-name")
    supervisor.add_argument("--batch-size", type=int)
    supervisor.add_argument("--expected-record-count", type=int)
    supervisor.add_argument("--min-launch-records", type=int)
    supervisor.add_argument("--parallel-workers", type=int)
    supervisor.add_argument("--render-resolution", type=int)
    supervisor.add_argument("--cycles-samples", type=int)
    supervisor.add_argument("--view-light-protocol")
    supervisor.add_argument("--session-prefix", default="trainv5_rolling")
    supervisor.set_defaults(func=cmd_supervisor)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.func(args, config)


if __name__ == "__main__":
    main()
