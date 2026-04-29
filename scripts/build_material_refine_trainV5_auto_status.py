#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/material_refine_trainV5_auto/status"
DEFAULT_LAST_CYCLE = REPO_ROOT / "output/material_refine_trainV5_auto/last_cycle_state.json"
DEFAULT_QUEUE = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/trainV5_plus_rebake_queue_latest.json"
DEFAULT_PENDING = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/pending_material_probe_by_source.json"
DEFAULT_B1_PROGRESS = REPO_ROOT / "output/material_refine_trainV5_abc/B_track/full_1155_rebake/progress_live.json"
DEFAULT_STAGE_SUMMARY = REPO_ROOT / "output/material_refine_expansion_candidates/material_priority_stage/material_priority_stage_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Write a TrainV5 automatic-material-priority status snapshot.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--last-cycle-state", type=Path, default=DEFAULT_LAST_CYCLE)
    parser.add_argument("--queue-path", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--pending-probe-path", type=Path, default=DEFAULT_PENDING)
    parser.add_argument("--b1-progress-path", type=Path, default=DEFAULT_B1_PROGRESS)
    parser.add_argument("--stage-summary-path", type=Path, default=DEFAULT_STAGE_SUMMARY)
    return parser.parse_args()


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def records(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def tmux_sessions() -> list[str]:
    result = subprocess.run(["tmux", "ls"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        return []
    sessions = []
    for line in result.stdout.splitlines():
        if ":" in line:
            sessions.append(line.split(":", 1)[0].strip())
    return sessions


def pgrep_lines(pattern: str) -> list[str]:
    result = subprocess.run(["pgrep", "-af", pattern], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def batch_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    return {
        "exists": path.exists(),
        "path": str(path),
        "summary": payload.get("summary", {}) if isinstance(payload, dict) else {},
        "batch_size": payload.get("batch_size") if isinstance(payload, dict) else None,
    }


def queue_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    rows = records(payload)
    return {
        "exists": path.exists(),
        "path": str(path),
        "records": len(rows),
        "material_family": dict(Counter(str(row.get("expected_material_family") or "unknown") for row in rows)),
        "source_name": dict(Counter(str(row.get("source_name") or "unknown") for row in rows)),
        "material_probe_status": dict(Counter(str(row.get("material_probe_status") or "unknown") for row in rows)),
    }


def main() -> None:
    args = parse_args()
    last_cycle = read_json(args.last_cycle_state, {})
    pending = read_json(args.pending_probe_path, {})
    b1 = read_json(args.b1_progress_path, {})
    stage_summary = read_json(args.stage_summary_path, {})
    sessions = tmux_sessions()

    payload = {
        "generated_at_utc": utc_now(),
        "tmux_sessions": sessions,
        "supervisor_session_alive": "trainv5_material_priority_supervisor" in sessions,
        "guard_session_alive": "trainv5_material_priority_guard" in sessions,
        "manual_stage_session_alive": "trainv5_material_priority_source_stage" in sessions,
        "active_stage_processes": pgrep_lines("stage_material_refine_material_priority_sources.py"),
        "active_objaverse_download_processes": pgrep_lines("stage_objaverse_cached_increment.py"),
        "last_cycle_state": last_cycle,
        "stage_summary": stage_summary.get("summary", {}) if isinstance(stage_summary, dict) else {},
        "latest_queue": queue_summary(args.queue_path),
        "pending_material_probe": pending,
        "b1_progress": {
            "status": b1.get("status"),
            "processed": b1.get("processed"),
            "total": b1.get("total"),
            "target_truth_gate_pass": b1.get("target_truth_gate_pass"),
            "target_truth_gate_fail": b1.get("target_truth_gate_fail"),
            "pass_rate": b1.get("pass_rate"),
            "material_family": b1.get("material_family", {}),
            "source_name": b1.get("source_name", {}),
        },
        "quota_batches": {
            "batch_0_64": batch_summary(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/batch_0_64_material_first.json"),
            "batch_1_256": batch_summary(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/batch_1_256_material_first.json"),
            "batch_1_512": batch_summary(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/batch_1_512_material_first.json"),
            "batch_2_1000": batch_summary(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/batch_2_1000_material_first.json"),
        },
    }
    write_json(args.output_dir / "auto_status.json", payload)
    write_text(
        args.output_dir / "auto_status.md",
        "\n".join(
            [
                "# TrainV5 Auto Status",
                "",
                f"- generated_at_utc: `{payload['generated_at_utc']}`",
                f"- supervisor_session_alive: `{str(payload['supervisor_session_alive']).lower()}`",
                f"- guard_session_alive: `{str(payload['guard_session_alive']).lower()}`",
                f"- manual_stage_session_alive: `{str(payload['manual_stage_session_alive']).lower()}`",
                f"- active_stage_processes: `{len(payload['active_stage_processes'])}`",
                f"- latest_queue_records: `{payload['latest_queue']['records']}`",
                f"- latest_queue_material_family: `{json.dumps(payload['latest_queue']['material_family'], ensure_ascii=False)}`",
                f"- pending_material_probe_by_source: `{json.dumps((payload['pending_material_probe'] or {}).get('source_name', {}), ensure_ascii=False)}`",
                f"- b1_progress: `{payload['b1_progress'].get('processed')}/{payload['b1_progress'].get('total')}`",
                f"- b1_pass_rate: `{payload['b1_progress'].get('pass_rate')}`",
                f"- batch_0_64_materials: `{json.dumps(payload['quota_batches']['batch_0_64']['summary'].get('material_family', {}), ensure_ascii=False)}`",
                f"- batch_1_256_materials: `{json.dumps(payload['quota_batches']['batch_1_256']['summary'].get('material_family', {}), ensure_ascii=False)}`",
                f"- batch_1_512_materials: `{json.dumps(payload['quota_batches']['batch_1_512']['summary'].get('material_family', {}), ensure_ascii=False)}`",
                f"- batch_2_1000_materials: `{json.dumps(payload['quota_batches']['batch_2_1000']['summary'].get('material_family', {}), ensure_ascii=False)}`",
                f"- last_cycle_state: `{json.dumps(last_cycle, ensure_ascii=False)}`",
            ]
        ),
    )


if __name__ == "__main__":
    main()
