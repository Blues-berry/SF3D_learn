#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_B1_STATUS = REPO_ROOT / "output/material_refine_trainV5_abc/B_track/full_1155_rebake/full_1155_finalize_status.json"
DEFAULT_OBJ1200_DECISION = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/objaverse_1200_serial/objaverse_1200_serial_decision.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/objaverse_1200_serial/objaverse_1200_launch_gate_status.json"


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
    parser.add_argument("--b1-status", type=Path, default=DEFAULT_B1_STATUS)
    parser.add_argument("--obj1200-decision", type=Path, default=DEFAULT_OBJ1200_DECISION)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--session-prefix", type=str, default="trainv5_obj1200")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    b1 = read_json(args.b1_status, {})
    obj = read_json(args.obj1200_decision, {})
    status = {
        "generated_at_utc": utc_now(),
        "b1_complete": str(b1.get("status") or "") == "complete",
        "obj1200_ready_exact_1200": bool(obj.get("ready_exact_1200")),
        "obj1200_queue_ready_exact_1200": bool(obj.get("queue_ready_exact_1200")),
        "obj1200_preflight_returncode": (obj.get("preflight_result") or {}).get("returncode"),
        "launched": False,
        "reason": "",
    }
    if not status["b1_complete"]:
        status["reason"] = "waiting_for_full_1155_finalize_complete"
        write_json(args.output_json, status)
        return
    if not status["obj1200_ready_exact_1200"] or not status["obj1200_queue_ready_exact_1200"]:
        status["reason"] = "waiting_for_objaverse_1200_ready_and_queue_ready_exact_1200"
        write_json(args.output_json, status)
        return
    if int(status["obj1200_preflight_returncode"] or 1) != 0:
        status["reason"] = "objaverse_1200_preflight_not_ready"
        write_json(args.output_json, status)
        return
    launch_status = Path(str(obj.get("abc_b_root") or "")) / f"{obj.get('batch_name')}_launch_status.json"
    if launch_status.exists():
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
        str(obj.get("batch_name") or "objaverse_1200_material_first_serial"),
        "--session-prefix",
        args.session_prefix,
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
