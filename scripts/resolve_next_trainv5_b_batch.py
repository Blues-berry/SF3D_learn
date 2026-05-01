#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/rolling_next_batch_decision.json"
DEFAULT_READY_ALIAS = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/rolling_ready_queue.json"
DEFAULT_DEFERRED_ALIAS = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/rolling_deferred_queue.json"


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
    parser.add_argument(
        "--obj1200-decision",
        type=Path,
        default=REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/objaverse_1200_serial/objaverse_1200_serial_decision.json",
    )
    parser.add_argument(
        "--rolling-queue",
        type=Path,
        default=REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/batch_1_512_material_first.json",
    )
    parser.add_argument(
        "--ready-queue-source",
        type=Path,
        default=REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/repaired_second_pass/repaired_target_rebake_candidates.json",
    )
    parser.add_argument(
        "--deferred-queue-source",
        type=Path,
        default=REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/repaired_second_pass/repaired_pending_material_probe_candidates.json",
    )
    parser.add_argument("--ready-alias-output", type=Path, default=DEFAULT_READY_ALIAS)
    parser.add_argument("--deferred-alias-output", type=Path, default=DEFAULT_DEFERRED_ALIAS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-launch-records", type=int, default=256)
    return parser.parse_args()


def records(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def object_ids_from_manifest(path: Path) -> set[str]:
    payload = read_json(path, {})
    ids: set[str] = set()
    for row in records(payload):
        oid = str(row.get("object_id") or row.get("canonical_object_id") or "")
        if oid:
            ids.add(oid)
    return ids


def launched_batch_object_ids() -> set[str]:
    used: set[str] = set()
    b_root = REPO_ROOT / "output/material_refine_trainV5_abc/B_track"
    if not b_root.exists():
        return used
    for launch_status in b_root.glob("*/*_launch_status.json"):
        payload = read_json(launch_status, {})
        manifest_path = Path(str(payload.get("input_manifest") or ""))
        if manifest_path.exists():
            used.update(object_ids_from_manifest(manifest_path))
    return used


def batch_state(batch_name: str) -> str:
    if not batch_name:
        return ""
    b_root = REPO_ROOT / "output/material_refine_trainV5_abc/B_track" / batch_name
    progress = read_json(b_root / "progress_live.json", {})
    status = str(progress.get("status") or "")
    if status:
        return status
    finalize = read_json(b_root / f"{batch_name}_finalize_status.json", {})
    return str(finalize.get("status") or "")


def write_frozen_manifest(path: Path, *, batch_name: str, queue_policy: str, source_path: Path, source: str, rows: list[dict[str, Any]], excluded_objects: int) -> None:
    payload = {
        "generated_at_utc": utc_now(),
        "batch_name": batch_name,
        "queue_policy": queue_policy,
        "source": source,
        "source_path": str(source_path),
        "excluded_objects": excluded_objects,
        "records": rows,
        "summary": {
            "records": len(rows),
            "source_name": {},
            "material_family": {},
        },
    }
    source_counts: dict[str, int] = {}
    material_counts: dict[str, int] = {}
    for row in rows:
        src = str(row.get("source_name") or "unknown")
        fam = str(row.get("expected_material_family") or row.get("material_family") or "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        material_counts[fam] = material_counts.get(fam, 0) + 1
    payload["summary"]["source_name"] = source_counts
    payload["summary"]["material_family"] = material_counts
    payload["queue_sha256"] = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    write_json(path, payload)


def main() -> None:
    args = parse_args()
    obj1200 = read_json(args.obj1200_decision, {})
    rolling = read_json(args.rolling_queue, {})
    ready_payload = read_json(args.ready_queue_source, {})
    deferred_payload = read_json(args.deferred_queue_source, {})
    used_object_ids = launched_batch_object_ids()

    write_json(
        args.ready_alias_output,
        {
            "generated_at_utc": utc_now(),
            "queue_role": "rolling_ready_queue",
            "source_path": str(args.ready_queue_source),
            "records": records(ready_payload),
            "summary": ready_payload.get("summary", {}) if isinstance(ready_payload, dict) else {},
        },
    )
    write_json(
        args.deferred_alias_output,
        {
            "generated_at_utc": utc_now(),
            "queue_role": "rolling_deferred_queue",
            "source_path": str(args.deferred_queue_source),
            "records": records(deferred_payload),
            "summary": deferred_payload.get("summary", {}) if isinstance(deferred_payload, dict) else {},
        },
    )

    decision: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "batch_name": None,
        "queue_path": None,
        "expected_record_count": 0,
        "launchable_now": False,
        "reason": "",
        "blocking_reason": "",
        "source": "",
        "ready_queue_path": str(args.ready_alias_output),
        "deferred_queue_path": str(args.deferred_alias_output),
        "ready_queue_records": len(records(ready_payload)),
        "deferred_queue_records": len(records(deferred_payload)),
    }

    obj1200_batch_name = str(obj1200.get("batch_name") or "objaverse_1200_material_first_serial")
    obj1200_state = batch_state(obj1200_batch_name)
    obj1200_manifest_path = Path(str(obj1200.get("rebake_input_manifest") or ""))
    if (
        bool(obj1200.get("launchable_now"))
        and int(obj1200.get("frozen_launch_record_count") or 0) >= int(args.min_launch_records)
        and obj1200_state not in {"running", "finalizing", "complete"}
    ):
        decision.update(
            {
                "batch_name": obj1200_batch_name,
                "queue_path": str(obj1200_manifest_path),
                "expected_record_count": int(obj1200.get("frozen_launch_record_count") or 0),
                "launchable_now": True,
                "reason": "obj1200_ready_subset",
                "blocking_reason": "",
                "source": "obj1200",
            }
        )
    else:
        rolling_rows = records(rolling)
        filtered_rows = []
        for row in rolling_rows:
            oid = str(row.get("object_id") or row.get("canonical_object_id") or "")
            if oid and oid in used_object_ids:
                continue
            filtered_rows.append(row)
        frozen_rolling_path = args.output_json.parent / "rolling_next_batch_input_manifest.json"
        rolling_batch_name = str(rolling.get("batch_name") or "material_first_complement_512")
        write_frozen_manifest(
            frozen_rolling_path,
            batch_name=rolling_batch_name,
            queue_policy="rolling_material_priority_complement_after_launched_batches",
            source_path=args.rolling_queue,
            source="rolling_queue",
            rows=filtered_rows,
            excluded_objects=max(len(rolling_rows) - len(filtered_rows), 0),
        )
        rolling_records = len(filtered_rows)
        if rolling_records >= int(args.min_launch_records):
            decision.update(
                {
                    "batch_name": rolling_batch_name,
                    "queue_path": str(frozen_rolling_path),
                    "expected_record_count": rolling_records,
                    "launchable_now": True,
                    "reason": "rolling_material_queue_ready",
                    "blocking_reason": "",
                    "source": "rolling_queue",
                }
            )
        else:
            decision.update(
                {
                    "queue_path": str(frozen_rolling_path),
                    "expected_record_count": rolling_records,
                    "launchable_now": False,
                    "reason": "insufficient_ready_records",
                    "blocking_reason": "ready_queue_below_min_launch_records",
                    "source": "rolling_queue",
                }
            )

    write_json(args.output_json, decision)


if __name__ == "__main__":
    main()
