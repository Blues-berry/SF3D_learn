#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = THIS_DIR / "trainv5_dataset_config.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def records(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path, {})
    if not isinstance(config, dict):
        raise SystemExit(f"invalid_config:{path}")
    config.setdefault("repo_root", str(THIS_DIR.parent))
    return config


def root(config: dict[str, Any]) -> Path:
    return Path(str(config["repo_root"])).resolve()


def repo_path(config: dict[str, Any], key: str) -> Path:
    value = Path(str(config[key]))
    return value if value.is_absolute() else root(config) / value


def run_cmd(cmd: list[str], *, cwd: Path, dry_run: bool = False, check: bool = True) -> subprocess.CompletedProcess[str] | None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return None
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def refresh_status(config: dict[str, Any], *, dry_run: bool = False) -> None:
    cwd = root(config)
    run_cmd(
        [
            sys.executable,
            "scripts/build_material_refine_trainV5_auto_status.py",
            "--output-dir",
            str(repo_path(config, "status_dir")),
            "--last-cycle-state",
            str(repo_path(config, "last_cycle_state")),
        ],
        cwd=cwd,
        dry_run=dry_run,
    )


def cmd_status(args: argparse.Namespace, config: dict[str, Any]) -> None:
    refresh_status(config, dry_run=args.dry_run)
    brief_path = repo_path(config, "status_dir") / "pipeline_brief.md"
    if brief_path.exists():
        print(brief_path.read_text(encoding="utf-8"))


def cmd_ingest(args: argparse.Namespace, config: dict[str, Any]) -> None:
    cwd = root(config)
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
    run_cmd(cmd, cwd=cwd, dry_run=args.dry_run)
    refresh_status(config, dry_run=args.dry_run)


def cmd_queue(args: argparse.Namespace, config: dict[str, Any]) -> None:
    cwd = root(config)
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


def resolve_next(config: dict[str, Any], args: argparse.Namespace, *, dry_run: bool) -> dict[str, Any]:
    cwd = root(config)
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
    decision_path = repo_path(config, "expansion_second_pass_dir") / "rolling_next_batch_decision.json"
    return read_json(decision_path, {})


def launch_status_summary(config: dict[str, Any], decision: dict[str, Any], *, dry_run: bool) -> None:
    brief = read_json(repo_path(config, "status_dir") / "pipeline_brief.json", {})
    queue = brief.get("queue", {}) if isinstance(brief, dict) else {}
    rebake = brief.get("rebake", {}) if isinstance(brief, dict) else {}
    payload = {
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
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_launch(args: argparse.Namespace, config: dict[str, Any]) -> None:
    cwd = root(config)
    decision = resolve_next(config, args, dry_run=args.dry_run)
    if args.dry_run:
        launch_status_summary(config, decision, dry_run=True)
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


def locate_batch_root(config: dict[str, Any], batch: str) -> Path:
    return repo_path(config, "abc_root") / "B_track" / batch


def cmd_finalize(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if not args.batch:
        raise SystemExit("--batch is required for finalize")
    b_root = locate_batch_root(config, args.batch)
    full_manifest = Path(args.full_manifest) if args.full_manifest else b_root / f"{args.batch}_manifest.json"
    if args.dry_run:
        manifest_used = full_manifest
        payload = read_json(full_manifest, {})
        if not records(payload):
            partial = b_root / f"{args.batch}_partial_manifest.json"
            partial_payload = read_json(partial, {})
            if records(partial_payload):
                manifest_used = partial
                payload = partial_payload
        prepared = len(records(payload))
        skipped = len(payload.get("skipped_records", [])) if isinstance(payload, dict) else 0
        print(
            json.dumps(
                {
                    "batch": args.batch,
                    "b_root": str(b_root),
                    "full_manifest": str(full_manifest),
                    "manifest_used_for_estimate": str(manifest_used),
                    "prepared_records": prepared,
                    "skipped_records": skipped,
                    "estimated_max_training_pairs": prepared * 5,
                    "dry_run": True,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    run_cmd(
        [
            sys.executable,
            "scripts/finalize_material_refine_trainV5_b_track.py",
            "--b-root",
            str(b_root),
            "--full-manifest",
            str(full_manifest),
        ],
        cwd=root(config),
    )
    refresh_status(config)


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
