#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEXT_BATCH = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/rolling_next_batch_decision.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/rolling_launch_gate_status.json"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--next-batch", type=Path, default=DEFAULT_NEXT_BATCH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--session-prefix", type=str, default="trainv5_rolling")
    return parser.parse_args()


def tmux_session_exists(name: str) -> bool:
    result = subprocess.run(["tmux", "has-session", "-t", name], cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def current_running_batch() -> dict[str, Any]:
    b_root = REPO_ROOT / "output/material_refine_trainV5_abc/B_track"
    if not b_root.exists():
        return {}
    running: list[dict[str, Any]] = []
    for progress_path in b_root.glob("*/progress_live.json"):
        payload = read_json(progress_path, {})
        if not isinstance(payload, dict):
            continue
        if str(payload.get("status") or "") != "running":
            continue
        running.append(
            {
                "batch_name": progress_path.parent.name,
                "progress_path": str(progress_path),
                "generated_at_utc": str(payload.get("generated_at_utc") or ""),
            }
        )
    running.sort(key=lambda item: (item["generated_at_utc"], item["progress_path"]), reverse=True)
    return running[0] if running else {}


def main() -> None:
    args = parse_args()
    next_batch = read_json(args.next_batch, {})
    batch_name = str(next_batch.get("batch_name") or "")
    batch_slug = batch_name
    status = {
        "generated_at_utc": utc_now(),
        "batch_name": batch_name,
        "launchable_now": bool(next_batch.get("launchable_now")),
        "frozen_launch_record_count": int(next_batch.get("expected_record_count") or 0),
        "launched": False,
        "reason": "",
        "blocking_reason": str(next_batch.get("blocking_reason") or ""),
    }
    if not status["launchable_now"] or not batch_name:
        status["reason"] = str(next_batch.get("reason") or "not_launchable")
        write_json(args.output_json, status)
        return

    running_batch = current_running_batch()
    if running_batch:
        status["reason"] = "running_batch_in_progress"
        status["blocking_reason"] = "gpu0_rebake_lane_busy"
        status["running_batch"] = running_batch
        write_json(args.output_json, status)
        return

    full_session = f"{args.session_prefix}_{batch_slug}_full_rebake"
    truth_session = f"{args.session_prefix}_{batch_slug}_truth_monitor"
    finalize_session = f"{args.session_prefix}_{batch_slug}_finalize_watcher"
    for session in (full_session, truth_session, finalize_session):
        if tmux_session_exists(session):
            status["launched"] = True
            status["reason"] = "already_launched"
            write_json(args.output_json, status)
            return

    cmd = [
        sys.executable,
        "scripts/launch_trainv5_b_rebake_batch.py",
        "--b-root",
        str(REPO_ROOT / "output/material_refine_trainV5_abc"),
        "--batch-name",
        batch_name,
        "--session-prefix",
        f"{args.session_prefix}_{batch_slug}",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    status["launch_command"] = cmd
    status["launch_returncode"] = result.returncode
    status["launch_stdout"] = result.stdout
    status["launch_stderr"] = result.stderr
    status["launched"] = result.returncode == 0
    status["reason"] = "launched" if result.returncode == 0 else "launch_failed"
    write_json(args.output_json, status)
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout or "launch_failed")


if __name__ == "__main__":
    main()
