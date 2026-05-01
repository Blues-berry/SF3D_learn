#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "output/material_refine_expansion_candidates/material_priority_objaverse_increment/objaverse_1200_source_candidate_manifest.json"
DEFAULT_SECOND_PASS_DIR = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass"
DEFAULT_OUTPUT_DIR = DEFAULT_SECOND_PASS_DIR / "objaverse_1200_serial"
DEFAULT_ABC_ROOT = REPO_ROOT / "output/material_refine_trainV5_abc"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--second-pass-dir", type=Path, default=DEFAULT_SECOND_PASS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--abc-root", type=Path, default=DEFAULT_ABC_ROOT)
    parser.add_argument("--target-total", type=int, default=1200)
    parser.add_argument("--batch-name", type=str, default="objaverse_1200_material_first_serial")
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--render-resolution", type=int, default=320)
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument("--view-light-protocol", type=str, default="production_32")
    parser.add_argument("--min-launch-records", type=int, default=256)
    parser.add_argument("--max-fail-rate", type=float, default=0.30)
    return parser.parse_args()


def records(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "material_family": dict(Counter(str(row.get("expected_material_family") or row.get("material_family") or "unknown") for row in rows)),
        "source_name": dict(Counter(str(row.get("source_name") or "unknown") for row in rows)),
        "license_bucket": dict(Counter(str(row.get("license_bucket") or "unknown") for row in rows)),
    }


def filter_rows(path: Path, object_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in records(read_json(path, {})) if str(row.get("object_id") or "") in object_ids]


def run_preflight(args: argparse.Namespace, queue_manifest: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/build_material_refine_trainV5_abc.py",
        "--run-b-preflight",
        "--b-queue",
        str(queue_manifest),
        "--b-batch-name",
        args.batch_name,
        "--b-batch-size",
        str(args.target_total),
        "--b-expected-record-count",
        str(args.target_total),
        "--b-parallel-workers",
        str(args.parallel_workers),
        "--b-render-resolution",
        str(args.render_resolution),
        "--b-cycles-samples",
        str(args.cycles_samples),
        "--b-view-light-protocol",
        str(args.view_light_protocol),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_payload = read_json(args.source_manifest, {})
    source_rows = records(source_payload)
    object_ids = {str(row.get("object_id") or "") for row in source_rows if str(row.get("object_id") or "")}

    queue_path = args.second_pass_dir / "trainV5_plus_rebake_queue_latest.json"
    repaired_queue_path = args.second_pass_dir / "repaired_second_pass/repaired_trainV5_plus_rebake_queue_latest.json"
    active_queue_path = repaired_queue_path if repaired_queue_path.exists() else queue_path
    queue_rows = filter_rows(active_queue_path, object_ids)
    pending_rows = filter_rows(args.second_pass_dir / "repaired_second_pass/repaired_pending_material_probe_candidates.json", object_ids)
    rejected_rows = filter_rows(args.second_pass_dir / "repaired_second_pass/repaired_hard_block_or_unusable_candidates.json", object_ids)
    repair_unknown_rows = filter_rows(args.second_pass_dir / "repair/still_unknown_material_candidates.json", object_ids)
    repair_license_pending_rows = filter_rows(args.second_pass_dir / "repair/license_pending_after_repair.json", object_ids)
    repair_path_missing_rows = filter_rows(args.second_pass_dir / "repair/path_still_missing_candidates.json", object_ids)

    second_pass_summary = {
        "generated_at_utc": utc_now(),
        "source_manifest": str(args.source_manifest),
        "queue_path": str(active_queue_path),
        "source_records": len(source_rows),
        "queue_ready_records": len(queue_rows),
        "summary": summarize(queue_rows),
    }
    repair_summary = {
        "generated_at_utc": utc_now(),
        "source_manifest": str(args.source_manifest),
        "pending_material_probe": len(pending_rows),
        "hard_block_or_unusable": len(rejected_rows),
        "still_unknown_after_repair": len(repair_unknown_rows),
        "license_pending_after_repair": len(repair_license_pending_rows),
        "path_still_missing_after_repair": len(repair_path_missing_rows),
        "pending_summary": summarize(pending_rows),
        "rejected_summary": summarize(rejected_rows),
    }
    write_json(args.output_dir / "objaverse_1200_second_pass_summary.json", second_pass_summary)
    write_json(args.output_dir / "objaverse_1200_repair_summary.json", repair_summary)

    download_stage_complete = len(source_rows) >= args.target_total
    deferred_records = len(pending_rows) + len(rejected_rows)
    source_records = len(source_rows)
    queue_ready_records = len(queue_rows)
    predicted_fail_rate = (deferred_records / source_records) if source_records > 0 else 1.0
    launchable_now = (
        download_stage_complete
        and queue_ready_records >= int(args.min_launch_records)
        and predicted_fail_rate <= float(args.max_fail_rate)
    )
    queue_manifest = args.output_dir / "objaverse_1200_rebake_input_manifest.json"
    preflight_result: dict[str, Any] = {"skipped": True}
    deferred_manifest = args.output_dir / "objaverse_1200_deferred_manifest.json"
    write_json(
        deferred_manifest,
        {
            "generated_at_utc": utc_now(),
            "batch_name": args.batch_name,
            "summary": summarize(pending_rows + rejected_rows),
            "records": pending_rows + rejected_rows,
        },
    )
    if launchable_now:
        write_json(
            queue_manifest,
            {
                "generated_at_utc": utc_now(),
                "queue_policy": "objaverse_1200_serial_ready_subset_after_download_and_repair",
                "batch_name": args.batch_name,
                "summary": summarize(queue_rows),
                "records": queue_rows,
            },
        )
        preflight_result = run_preflight(args, queue_manifest)
    decision = {
        "generated_at_utc": utc_now(),
        "source_manifest": str(args.source_manifest),
        "queue_path": str(active_queue_path),
        "download_stage_complete": download_stage_complete,
        "source_records": source_records,
        "queue_ready_records": queue_ready_records,
        "deferred_records": deferred_records,
        "predicted_fail_rate": predicted_fail_rate,
        "launchable_now": launchable_now,
        "frozen_launch_record_count": queue_ready_records if launchable_now else 0,
        "batch_name": args.batch_name,
        "rebake_input_manifest": str(queue_manifest),
        "deferred_manifest": str(deferred_manifest),
        "abc_b_root": str(args.abc_root / "B_track" / args.batch_name),
        "preflight_result": preflight_result,
    }
    write_json(args.output_dir / "objaverse_1200_serial_decision.json", decision)
    write_text(
        args.output_dir / "objaverse_1200_serial_decision.md",
        "\n".join(
            [
                "# Objaverse 1200 Serial Decision",
                "",
                f"- generated_at_utc: `{decision['generated_at_utc']}`",
                f"- source_records: `{decision['source_records']}`",
                f"- queue_ready_records: `{decision['queue_ready_records']}`",
                f"- deferred_records: `{decision['deferred_records']}`",
                f"- download_stage_complete: `{str(decision['download_stage_complete']).lower()}`",
                f"- launchable_now: `{str(decision['launchable_now']).lower()}`",
                f"- predicted_fail_rate: `{decision['predicted_fail_rate']}`",
                f"- rebake_input_manifest: `{decision['rebake_input_manifest']}`",
                f"- deferred_manifest: `{decision['deferred_manifest']}`",
                f"- abc_b_root: `{decision['abc_b_root']}`",
                f"- preflight_returncode: `{preflight_result.get('returncode')}`",
            ]
        ),
    )


if __name__ == "__main__":
    main()
