#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.trainv5_target_gate import (  # noqa: E402
    TARGET_GATE_VERSION,
    target_prior_relation_diagnostics,
    trainv5_target_truth_gate,
)


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


def skipped(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("skipped_records", [])
        return [row for row in rows if isinstance(row, dict)]
    return []


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def free_gb(path: Path) -> float | None:
    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    return float(stat.f_bavail * stat.f_frsize / (1024**3))


def output_size_gb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return float(total / (1024**3))


def numeric_stats(values: list[float]) -> dict[str, Any]:
    values = sorted(values)
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    p95_idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
    return {"mean": sum(values) / len(values), "p50": values[len(values) // 2], "p95": values[p95_idx]}


def build_snapshot(
    *,
    input_manifest: Path,
    partial_manifest: Path,
    final_manifest: Path,
    output_root: Path,
    dataoutput_root: Path,
    total: int,
) -> dict[str, Any]:
    input_payload = read_json(input_manifest, {"records": []})
    partial_payload = read_json(partial_manifest, {"records": [], "skipped_records": []})
    final_payload = read_json(final_manifest, None)
    final_is_complete = (
        isinstance(final_payload, dict)
        and "counts" in final_payload
        and str(final_payload.get("status") or "") != "not_started"
    )
    active_payload = final_payload if final_is_complete else partial_payload
    prepared = records(active_payload)
    # Partial manifests report not-yet-finished bundles as missing files. Those
    # are progress placeholders, not actual skipped records from the rebake run.
    skips = skipped(active_payload) if final_is_complete else []
    processed = len(prepared) + len(skips)
    means = [finite_float(row.get("target_view_alignment_mean")) for row in prepared]
    p95s = [finite_float(row.get("target_view_alignment_p95")) for row in prepared]
    means_clean = [float(value) for value in means if value is not None]
    p95s_clean = [float(value) for value in p95s if value is not None]
    gate_results = [trainv5_target_truth_gate(row) for row in prepared]
    gate_pass_count = sum(1 for ok, _blockers in gate_results if ok)
    gate_blockers = Counter(reason for _ok, blockers in gate_results for reason in blockers)
    prior_relation = [target_prior_relation_diagnostics(row) for row in prepared]
    target_prior_copy_count = sum(1 for row in prior_relation if bool_value(row.get("target_is_prior_copy")))
    elapsed_status = "complete" if final_is_complete else "running"
    failure_counts = Counter(str(row.get("reason") or "unknown") for row in skips)
    return {
        "generated_at_utc": utc_now(),
        "status": elapsed_status,
        "input_manifest": str(input_manifest),
        "partial_manifest": str(partial_manifest),
        "final_manifest": str(final_manifest),
        "total": total,
        "processed": processed,
        "prepared_records": len(prepared),
        "skipped_records": len(skips),
        "target_gate_version": TARGET_GATE_VERSION,
        "target_truth_gate_pass": gate_pass_count,
        "target_truth_gate_fail": max(len(prepared) - gate_pass_count, 0),
        "target_truth_gate_blockers": dict(gate_blockers),
        "pass_rate": (gate_pass_count / len(prepared)) if prepared else None,
        "target_prior_relation_diagnostic": {
            "target_is_prior_copy": target_prior_copy_count,
            "target_not_prior_copy": max(len(prepared) - target_prior_copy_count, 0),
        },
        "material_family": dict(Counter(str(row.get("material_family") or "unknown") for row in prepared)),
        "source_name": dict(Counter(str(row.get("source_name") or "unknown") for row in prepared)),
        "prior_mode": dict(Counter(str(row.get("prior_mode") or "unknown") for row in prepared)),
        "target_view_alignment_mean": numeric_stats(means_clean),
        "target_view_alignment_p95": numeric_stats(p95s_clean),
        "failure_reason_counts": dict(failure_counts),
        "ssd_free_gb": free_gb(dataoutput_root),
        "output_size_gb": output_size_gb(output_root),
        "estimated_remaining_records": max(total - processed, 0),
    }


def write_progress(output_dir: Path, snapshot: dict[str, Any], force_final: bool = False) -> None:
    processed = int(snapshot.get("processed") or 0)
    milestone = ((processed + 99) // 100) * 100 if processed else 0
    if force_final:
        path = output_dir / "progress_final.md"
    elif milestone > 0:
        path = output_dir / f"progress_{milestone:04d}.md"
    else:
        path = output_dir / "progress_0000.md"
    lines = [
        f"# B-track Full Rebake Progress {processed}/{snapshot.get('total')}",
        "",
        f"- generated_at_utc: `{snapshot['generated_at_utc']}`",
        f"- status: `{snapshot.get('status')}`",
        f"- processed: `{processed}/{snapshot.get('total')}`",
        f"- prepared_records: `{snapshot.get('prepared_records')}`",
        f"- skipped_records: `{snapshot.get('skipped_records')}`",
        f"- target_gate_version: `{snapshot.get('target_gate_version')}`",
        f"- target_truth_gate_pass: `{snapshot.get('target_truth_gate_pass')}`",
        f"- target_truth_gate_fail: `{snapshot.get('target_truth_gate_fail')}`",
        f"- target_truth_gate_blockers: `{json.dumps(snapshot.get('target_truth_gate_blockers', {}), ensure_ascii=False)}`",
        f"- pass_rate: `{snapshot.get('pass_rate')}`",
        f"- target_prior_relation_diagnostic: `{json.dumps(snapshot.get('target_prior_relation_diagnostic', {}), ensure_ascii=False)}`",
        f"- material_family distribution: `{json.dumps(snapshot.get('material_family', {}), ensure_ascii=False)}`",
        f"- source distribution: `{json.dumps(snapshot.get('source_name', {}), ensure_ascii=False)}`",
        f"- no_prior / scalar / spatial prior hints: `{json.dumps(snapshot.get('prior_mode', {}), ensure_ascii=False)}`",
        f"- target_view_alignment_mean: `{json.dumps(snapshot.get('target_view_alignment_mean', {}), ensure_ascii=False)}`",
        f"- target_view_alignment_p95: `{json.dumps(snapshot.get('target_view_alignment_p95', {}), ensure_ascii=False)}`",
        f"- failure reason counts: `{json.dumps(snapshot.get('failure_reason_counts', {}), ensure_ascii=False)}`",
        f"- SSD free GB: `{snapshot.get('ssd_free_gb')}`",
        f"- output size GB: `{snapshot.get('output_size_gb')}`",
        f"- estimated remaining records: `{snapshot.get('estimated_remaining_records')}`",
    ]
    write_text(path, "\n".join(lines))
    write_text(output_dir / "progress_live.md", "\n".join(lines))
    write_json(output_dir / "progress_live.json", snapshot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--partial-manifest", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataoutput-root", type=Path, default=Path("dataoutput"))
    parser.add_argument("--total", type=int, default=1155)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.once:
        snapshot = build_snapshot(
            input_manifest=args.input_manifest,
            partial_manifest=args.partial_manifest,
            final_manifest=args.final_manifest,
            output_root=args.output_root,
            dataoutput_root=args.dataoutput_root,
            total=args.total,
        )
        write_progress(args.output_dir, snapshot, force_final=args.force_final)
        return

    while True:
        snapshot = build_snapshot(
            input_manifest=args.input_manifest,
            partial_manifest=args.partial_manifest,
            final_manifest=args.final_manifest,
            output_root=args.output_root,
            dataoutput_root=args.dataoutput_root,
            total=args.total,
        )
        write_progress(args.output_dir, snapshot, force_final=False)
        if snapshot.get("status") == "complete":
            write_progress(args.output_dir, snapshot, force_final=True)
            break
        time.sleep(max(float(args.interval_seconds), 5.0))


if __name__ == "__main__":
    main()
