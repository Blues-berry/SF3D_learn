#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
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
DEFAULT_OBJ1200_STATUS = REPO_ROOT / "output/material_refine_expansion_candidates/material_priority_stage/objaverse_1200_serial/objaverse_1200_download_status.json"
DEFAULT_OBJ1200_SERIAL_DECISION = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/objaverse_1200_serial/objaverse_1200_serial_decision.json"
DEFAULT_ROLLING_LAUNCH_GATE = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/rolling_launch_gate_status.json"
DEFAULT_ROLLING_NEXT_BATCH = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/rolling_next_batch_decision.json"
DEFAULT_ROLLING_READY = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/rolling_ready_queue.json"
DEFAULT_ROLLING_DEFERRED = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/rolling_deferred_queue.json"
DEFAULT_PLUS_FULL_READINESS = REPO_ROOT / "train/trainV5_plus_full/trainV5_plus_readiness_report.json"
DEFAULT_MERGED_READINESS = REPO_ROOT / "train/trainV5_merged_ab/trainV5_merged_readiness_report.json"


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
    parser.add_argument("--obj1200-status-path", type=Path, default=DEFAULT_OBJ1200_STATUS)
    parser.add_argument("--obj1200-serial-decision-path", type=Path, default=DEFAULT_OBJ1200_SERIAL_DECISION)
    parser.add_argument("--rolling-launch-gate-path", type=Path, default=DEFAULT_ROLLING_LAUNCH_GATE)
    parser.add_argument("--rolling-next-batch-path", type=Path, default=DEFAULT_ROLLING_NEXT_BATCH)
    parser.add_argument("--rolling-ready-path", type=Path, default=DEFAULT_ROLLING_READY)
    parser.add_argument("--rolling-deferred-path", type=Path, default=DEFAULT_ROLLING_DEFERRED)
    parser.add_argument("--plus-full-readiness-path", type=Path, default=DEFAULT_PLUS_FULL_READINESS)
    parser.add_argument("--merged-readiness-path", type=Path, default=DEFAULT_MERGED_READINESS)
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


def parse_etime_to_hours(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = text.split(":")
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        hours, minutes, seconds = nums
    elif len(nums) == 2:
        hours = 0
        minutes, seconds = nums
    else:
        return None
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds / 3600.0


def current_prepare_runtime(input_manifest: str) -> dict[str, Any]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,etime=,args="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "prepare_material_refine_dataset.py" not in line:
            continue
        if input_manifest and input_manifest not in line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, etime, cmd = parts
        runtime_hours = parse_etime_to_hours(etime)
        config = {
            "pid": int(pid),
            "elapsed": etime,
            "runtime_hours": runtime_hours,
            "command": cmd,
        }
        try:
            argv = shlex.split(cmd)
        except ValueError:
            argv = []
        mapping: dict[str, str] = {}
        for index, token in enumerate(argv):
            if token.startswith("--") and index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                mapping[token] = argv[index + 1]
        config["parallel_workers"] = mapping.get("--parallel-workers")
        config["render_resolution"] = mapping.get("--render-resolution")
        config["cycles_samples"] = mapping.get("--cycles-samples")
        config["view_light_protocol"] = mapping.get("--view-light-protocol")
        return config
    return {}


def batch_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    return {
        "exists": path.exists(),
        "path": str(path),
        "summary": payload.get("summary", {}) if isinstance(payload, dict) else {},
        "batch_size": payload.get("batch_size") if isinstance(payload, dict) else None,
    }


def current_rebake_batch() -> dict[str, Any]:
    b_root = REPO_ROOT / "output/material_refine_trainV5_abc/B_track"
    candidates: list[dict[str, Any]] = []
    status_priority = {"running": 3, "finalizing": 2, "blocked": 1, "complete": 0}
    if not b_root.exists():
        return {}
    for progress_path in sorted(b_root.glob("*/progress_live.json")):
        payload = read_json(progress_path, {})
        if not isinstance(payload, dict):
            continue
        if str(payload.get("status") or "") not in {"running", "finalizing", "complete", "blocked"}:
            continue
        candidates.append(
            {
                "batch_name": progress_path.parent.name,
                "progress_path": str(progress_path),
                "status": payload.get("status"),
                "processed": payload.get("processed"),
                "total": payload.get("total"),
                "target_truth_gate_pass": payload.get("target_truth_gate_pass"),
                "target_truth_gate_fail": payload.get("target_truth_gate_fail"),
                "pass_rate": payload.get("pass_rate"),
                "generated_at_utc": payload.get("generated_at_utc"),
                "input_manifest": payload.get("input_manifest"),
            }
        )
    candidates.sort(
        key=lambda item: (
            status_priority.get(str(item.get("status") or ""), -1),
            str(item.get("generated_at_utc") or ""),
            str(item.get("progress_path")),
        ),
        reverse=True,
    )
    return candidates[0] if candidates else {}


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
    obj1200_status = read_json(args.obj1200_status_path, {})
    obj1200_serial = read_json(args.obj1200_serial_decision_path, {})
    rolling_launch_gate = read_json(args.rolling_launch_gate_path, {})
    rolling_next_batch = read_json(args.rolling_next_batch_path, {})
    rolling_ready = read_json(args.rolling_ready_path, {})
    rolling_deferred = read_json(args.rolling_deferred_path, {})
    plus_full_readiness = read_json(args.plus_full_readiness_path, {})
    merged_readiness = read_json(args.merged_readiness_path, {})
    sessions = tmux_sessions()
    current_batch = current_rebake_batch()
    runtime_manifest = str(current_batch.get("input_manifest") or b1.get("input_manifest") or "")
    prepare_runtime = current_prepare_runtime(runtime_manifest)
    active_progress = current_batch or b1 or {}
    processed = active_progress.get("processed")
    total = active_progress.get("total")
    runtime_hours = prepare_runtime.get("runtime_hours")
    records_per_hour = None
    estimated_hours_remaining = None
    if isinstance(processed, (int, float)) and runtime_hours and runtime_hours > 0:
        records_per_hour = float(processed) / float(runtime_hours)
        remaining = max(float(total or 0) - float(processed), 0.0)
        if records_per_hour > 0:
            estimated_hours_remaining = remaining / records_per_hour

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
        "objaverse_1200_download_status": obj1200_status,
        "objaverse_1200_serial_decision": obj1200_serial,
        "rolling_next_batch": rolling_next_batch,
        "rolling_launch_gate": rolling_launch_gate,
        "rolling_ready_queue": queue_summary(args.rolling_ready_path) if args.rolling_ready_path.exists() else {
            "exists": False,
            "path": str(args.rolling_ready_path),
            "records": 0,
            "material_family": {},
            "source_name": {},
            "material_probe_status": {},
        },
        "rolling_deferred_queue": queue_summary(args.rolling_deferred_path) if args.rolling_deferred_path.exists() else {
            "exists": False,
            "path": str(args.rolling_deferred_path),
            "records": 0,
            "material_family": {},
            "source_name": {},
            "material_probe_status": {},
        },
        "latest_queue": queue_summary(args.queue_path),
        "pending_material_probe": pending,
        "current_rebake_batch": current_batch,
        "cumulative_trainv5": {
            "plus_full": {
                "target_bundles": plus_full_readiness.get("target_bundles"),
                "training_pairs": plus_full_readiness.get("training_pairs"),
                "generated_at_utc": plus_full_readiness.get("generated_at_utc"),
            },
            "merged_ab": {
                "target_bundles": merged_readiness.get("target_bundles"),
                "training_pairs": merged_readiness.get("training_pairs"),
                "generated_at_utc": merged_readiness.get("generated_at_utc"),
            },
        },
        "current_rebake_progress": {
            "batch_name": active_progress.get("batch_name"),
            "status": active_progress.get("status"),
            "processed": active_progress.get("processed"),
            "total": active_progress.get("total"),
            "target_truth_gate_pass": active_progress.get("target_truth_gate_pass"),
            "target_truth_gate_fail": active_progress.get("target_truth_gate_fail"),
            "pass_rate": active_progress.get("pass_rate"),
            "records_per_hour": records_per_hour,
            "estimated_hours_remaining": estimated_hours_remaining,
            "current_parallel_workers": prepare_runtime.get("parallel_workers"),
            "current_render_protocol": prepare_runtime.get("view_light_protocol"),
            "current_render_resolution": prepare_runtime.get("render_resolution"),
            "current_cycles_samples": prepare_runtime.get("cycles_samples"),
            "prepare_runtime": prepare_runtime,
            "material_family": active_progress.get("material_family", {}),
            "source_name": active_progress.get("source_name", {}),
        },
        "quota_batches": {
            "batch_0_64": batch_summary(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/batch_0_64_material_first.json"),
            "batch_1_256": batch_summary(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/batch_1_256_material_first.json"),
            "batch_1_512": batch_summary(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/batch_1_512_material_first.json"),
            "batch_2_1000": batch_summary(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/batch_2_1000_material_first.json"),
        },
    }
    write_json(args.output_dir / "auto_status.json", payload)
    pipeline_brief = {
        "generated_at_utc": payload["generated_at_utc"],
        "ingest": {
            "stage_process_running": len(payload["active_stage_processes"]) > 0,
            "last_cycle_started_utc": payload.get("last_cycle_state", {}).get("cycle_started_utc"),
            "last_cycle_finished_utc": payload.get("last_cycle_state", {}).get("cycle_finished_utc"),
            "download_target": payload.get("objaverse_1200_download_status", {}).get("target_total"),
            "downloaded": payload.get("objaverse_1200_download_status", {}).get("downloaded_total"),
            "missing": payload.get("objaverse_1200_download_status", {}).get("missing_total"),
            "retry_round": payload.get("objaverse_1200_download_status", {}).get("retry_round"),
            "topup_needed": payload.get("objaverse_1200_download_status", {}).get("topup_needed"),
        },
        "queue": {
            "ready_records": payload.get("rolling_ready_queue", {}).get("records"),
            "deferred_records": payload.get("rolling_deferred_queue", {}).get("records"),
            "obj1200_ready_records": payload.get("objaverse_1200_serial_decision", {}).get("queue_ready_records"),
            "obj1200_deferred_records": payload.get("objaverse_1200_serial_decision", {}).get("deferred_records"),
            "next_batch_name": payload.get("rolling_next_batch", {}).get("batch_name"),
            "next_launchable": payload.get("rolling_next_batch", {}).get("launchable_now"),
            "next_frozen_launch_record_count": payload.get("rolling_next_batch", {}).get("expected_record_count"),
            "launch_gate_reason": payload.get("rolling_launch_gate", {}).get("reason"),
            "launch_blocking_reason": payload.get("rolling_launch_gate", {}).get("blocking_reason"),
        },
        "rebake": payload.get("current_rebake_batch", {}),
        "accumulate": {
            "plus_full_target_bundles": payload.get("cumulative_trainv5", {}).get("plus_full", {}).get("target_bundles"),
            "plus_full_training_pairs": payload.get("cumulative_trainv5", {}).get("plus_full", {}).get("training_pairs"),
            "merged_ab_target_bundles": payload.get("cumulative_trainv5", {}).get("merged_ab", {}).get("target_bundles"),
            "merged_ab_training_pairs": payload.get("cumulative_trainv5", {}).get("merged_ab", {}).get("training_pairs"),
            "last_plus_full_update": payload.get("cumulative_trainv5", {}).get("plus_full", {}).get("generated_at_utc"),
            "last_merged_ab_update": payload.get("cumulative_trainv5", {}).get("merged_ab", {}).get("generated_at_utc"),
        },
    }
    write_json(args.output_dir / "pipeline_brief.json", pipeline_brief)
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
                f"- stage_download_mode: `{payload['stage_summary'].get('download_mode')}`",
                f"- stage_download_success_rate: `{payload['stage_summary'].get('download_success_rate')}`",
                f"- stage_download_failure_reason: `{payload['stage_summary'].get('download_failure_reason')}`",
                f"- obj1200_downloaded_total: `{(payload['objaverse_1200_download_status'] or {}).get('downloaded_total')}`",
                f"- obj1200_missing_total: `{(payload['objaverse_1200_download_status'] or {}).get('missing_total')}`",
                f"- obj1200_retry_round: `{(payload['objaverse_1200_download_status'] or {}).get('retry_round')}`",
                f"- rolling_ready_records: `{payload['rolling_ready_queue']['records']}`",
                f"- rolling_deferred_records: `{payload['rolling_deferred_queue']['records']}`",
                f"- next_batch_name: `{(payload['rolling_next_batch'] or {}).get('batch_name')}`",
                f"- next_batch_launchable: `{(payload['rolling_next_batch'] or {}).get('launchable_now')}`",
                f"- rolling_launch_reason: `{(payload['rolling_launch_gate'] or {}).get('reason')}`",
                f"- rolling_launch_blocking_reason: `{(payload['rolling_launch_gate'] or {}).get('blocking_reason')}`",
                f"- current_rebake_batch: `{payload['current_rebake_progress'].get('batch_name')}`",
                f"- current_rebake_progress: `{payload['current_rebake_progress'].get('processed')}/{payload['current_rebake_progress'].get('total')}`",
                f"- current_rebake_pass_rate: `{payload['current_rebake_progress'].get('pass_rate')}`",
                f"- current_rebake_records_per_hour: `{payload['current_rebake_progress'].get('records_per_hour')}`",
                f"- current_rebake_estimated_hours_remaining: `{payload['current_rebake_progress'].get('estimated_hours_remaining')}`",
                f"- current_rebake_parallel_workers: `{payload['current_rebake_progress'].get('current_parallel_workers')}`",
                f"- current_rebake_render_protocol: `{payload['current_rebake_progress'].get('current_render_protocol')}`",
                f"- current_rebake_render_resolution: `{payload['current_rebake_progress'].get('current_render_resolution')}`",
                f"- current_rebake_cycles_samples: `{payload['current_rebake_progress'].get('current_cycles_samples')}`",
                f"- batch_0_64_materials: `{json.dumps(payload['quota_batches']['batch_0_64']['summary'].get('material_family', {}), ensure_ascii=False)}`",
                f"- batch_1_256_materials: `{json.dumps(payload['quota_batches']['batch_1_256']['summary'].get('material_family', {}), ensure_ascii=False)}`",
                f"- batch_1_512_materials: `{json.dumps(payload['quota_batches']['batch_1_512']['summary'].get('material_family', {}), ensure_ascii=False)}`",
                f"- batch_2_1000_materials: `{json.dumps(payload['quota_batches']['batch_2_1000']['summary'].get('material_family', {}), ensure_ascii=False)}`",
                f"- last_cycle_state: `{json.dumps(last_cycle, ensure_ascii=False)}`",
            ]
        ),
    )
    write_text(
        args.output_dir / "pipeline_brief.md",
        "\n".join(
            [
                "# TrainV5 Pipeline Brief",
                "",
                f"- generated_at_utc: `{pipeline_brief['generated_at_utc']}`",
                "",
                "## ingest/material_priority_stage",
                "",
                f"- stage_process_running: `{str(pipeline_brief['ingest']['stage_process_running']).lower()}`",
                f"- last_cycle_started_utc: `{pipeline_brief['ingest']['last_cycle_started_utc']}`",
                f"- last_cycle_finished_utc: `{pipeline_brief['ingest']['last_cycle_finished_utc']}`",
                f"- download_target/downloaded/missing: `{pipeline_brief['ingest']['download_target']}/{pipeline_brief['ingest']['downloaded']}/{pipeline_brief['ingest']['missing']}`",
                f"- retry_round: `{pipeline_brief['ingest']['retry_round']}`",
                f"- topup_needed: `{pipeline_brief['ingest']['topup_needed']}`",
                "",
                "## queue/second_pass_repair",
                "",
                f"- ready_records: `{pipeline_brief['queue']['ready_records']}`",
                f"- deferred_records: `{pipeline_brief['queue']['deferred_records']}`",
                f"- obj1200_ready_records: `{pipeline_brief['queue']['obj1200_ready_records']}`",
                f"- obj1200_deferred_records: `{pipeline_brief['queue']['obj1200_deferred_records']}`",
                f"- next_batch_name: `{pipeline_brief['queue']['next_batch_name']}`",
                f"- next_launchable: `{pipeline_brief['queue']['next_launchable']}`",
                f"- next_frozen_launch_record_count: `{pipeline_brief['queue']['next_frozen_launch_record_count']}`",
                f"- launch_gate_reason: `{pipeline_brief['queue']['launch_gate_reason']}`",
                f"- launch_blocking_reason: `{pipeline_brief['queue']['launch_blocking_reason']}`",
                "",
                "## rebake/current_batch",
                "",
                f"- batch_name: `{pipeline_brief['rebake'].get('batch_name')}`",
                f"- status: `{pipeline_brief['rebake'].get('status')}`",
                f"- processed/total: `{pipeline_brief['rebake'].get('processed')}/{pipeline_brief['rebake'].get('total')}`",
                f"- target_truth_gate_pass/fail: `{pipeline_brief['rebake'].get('target_truth_gate_pass')}/{pipeline_brief['rebake'].get('target_truth_gate_fail')}`",
                f"- pass_rate: `{pipeline_brief['rebake'].get('pass_rate')}`",
                "",
                "## finalize/accumulate",
                "",
                f"- plus_full_target_bundles/training_pairs: `{pipeline_brief['accumulate']['plus_full_target_bundles']}/{pipeline_brief['accumulate']['plus_full_training_pairs']}`",
                f"- merged_ab_target_bundles/training_pairs: `{pipeline_brief['accumulate']['merged_ab_target_bundles']}/{pipeline_brief['accumulate']['merged_ab_training_pairs']}`",
                f"- last_plus_full_update: `{pipeline_brief['accumulate']['last_plus_full_update']}`",
                f"- last_merged_ab_update: `{pipeline_brief['accumulate']['last_merged_ab_update']}`",
            ]
        ),
    )


if __name__ == "__main__":
    main()
