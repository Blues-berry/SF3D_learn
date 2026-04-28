#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REJECT = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/reject_or_unknown_candidates.json"
DEFAULT_PENDING = REPO_ROOT / "output/material_refine_trainV5_abc/B_track/pending_repair/pending_repair_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/material_refine_trainV5_abc/B_track/reject_unknown_provenance"
DEFAULT_QUEUE_COPY = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/trainV5_plus_rebake_queue_preview_v3.json"

STRICT_ALLOWED_LICENSE_BUCKETS = {
    "Creative Commons Zero v1.0 Universal",
    "MIT License",
    "Apache License 2.0",
    'BSD 3-Clause "New" or "Revised" License',
    "The Unlicense",
    "Creative Commons - Attribution",
    "Creative Commons - Attribution - Share Alike",
}
PENDING_LICENSE_BUCKETS = {
    "cc_by_4_0_pending_reconcile",
    "smithsonian_open_access_pending_reconcile",
    "omniobject3d_license_pending_reconcile",
    "khronos_sample_license_pending_reconcile",
    "cc_by_nc_4_0_pending_reconcile",
}
SOURCE_DIVERSITY_BONUS = {
    "ABO_locked_core": 0.0,
    "Local_GSO_highlight_increment": 25.0,
    "Local_Kenney_CC0_increment": 18.0,
    "Local_Smithsonian_selected_increment": 28.0,
    "Local_OmniObject3D_increment": 24.0,
    "Khronos_highlight_reference_samples": 18.0,
}


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
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def path_exists(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).exists()


def object_id(record: dict[str, Any]) -> str:
    source = record.get("source_record") if isinstance(record.get("source_record"), dict) else {}
    return str(record.get("object_id") or source.get("object_id") or "")


def candidate_paths(record: dict[str, Any]) -> list[str]:
    source = record.get("source_record") if isinstance(record.get("source_record"), dict) else {}
    paths: list[str] = []
    for key in (
        "asset_path",
        "physical_path",
        "source_model_path",
        "raw_asset_path",
        "canonical_glb_path",
        "canonical_mesh_path",
    ):
        value = record.get(key) or source.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    return paths


def first_existing_path(record: dict[str, Any]) -> str:
    for path in candidate_paths(record):
        if path_exists(path):
            return str(Path(path).resolve())
    return ""


def original_reasons(record: dict[str, Any], reject_index: dict[str, dict[str, Any]]) -> list[str]:
    reasons = record.get("blocked_reason") or record.get("blocked_reasons") or []
    if not reasons and object_id(record) in reject_index:
        reasons = reject_index[object_id(record)].get("blocked_reason") or []
    if isinstance(reasons, str):
        return [reasons]
    return [str(reason) for reason in reasons if reason]


def material_status(record: dict[str, Any]) -> str:
    family = str(record.get("expected_material_family") or record.get("material_family") or "")
    if family in {"", "unknown", "still_unknown", "unknown_material", "unknown_pending_second_pass", "pending_abo_semantic_classification"}:
        return "unknown_but_allowed"
    return "known"


def normalized_material_family(record: dict[str, Any]) -> str:
    return (
        str(record.get("expected_material_family") or record.get("material_family") or "")
        if material_status(record) == "known"
        else "unknown_material_pending_probe"
    )


def license_status(record: dict[str, Any]) -> str:
    bucket = str(record.get("license_bucket") or "")
    status = str(record.get("license_status") or "").lower()
    if bucket in STRICT_ALLOWED_LICENSE_BUCKETS or bool_value(record.get("license_allowed_for_training")):
        return "allowed"
    if bucket in PENDING_LICENSE_BUCKETS or "pending" in bucket.lower() or status == "pending_review":
        return "pending_engineering_only"
    if any(token in status or token in bucket.lower() for token in ("hard_block", "forbidden", "no_training", "blocked")):
        return "hard_blocked"
    return "pending_engineering_only"


def trace_status(record: dict[str, Any], physical_path: str, license_state: str) -> str:
    if license_state == "hard_blocked":
        return "license_hard_blocked"
    if not physical_path or not object_id(record) or not str(record.get("source_name") or ""):
        return "path_unresolved"
    return "trace_ok"


def priority(record: dict[str, Any], family: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons = [f"material:{family}:0"]
    if not bool_value(record.get("has_material_prior")):
        score += 45.0
        reasons.append("no_prior:+45")
    source = str(record.get("source_name") or "unknown")
    bonus = SOURCE_DIVERSITY_BONUS.get(source, 12.0 if source != "ABO_locked_core" else 0.0)
    score += bonus
    reasons.append(f"source_diversity:{bonus:.0f}")
    if license_status(record) == "pending_engineering_only":
        score -= 2.0
        reasons.append("license_pending_engineering_only:-2")
    return round(score, 4), reasons


def storage_tier(path: str) -> str:
    if not path:
        return "unknown"
    if path.startswith("/4T/"):
        return "hdd_archive"
    return "external_or_project_storage"


def expected_prior_variant_types(record: dict[str, Any]) -> list[str]:
    if not bool_value(record.get("has_material_prior")):
        return ["no_prior_bootstrap", "synthetic_large_gap_prior"]
    prior_mode = str(record.get("prior_mode") or "").lower()
    if prior_mode == "scalar_rm":
        return ["scalar_broadcast_prior", "synthetic_mild_gap_prior", "synthetic_medium_gap_prior"]
    if prior_mode in {"uv_rm", "texture_rm", "spatial_map"}:
        return ["texture_rm_prior", "synthetic_mild_gap_prior", "synthetic_medium_gap_prior"]
    return ["existing_pipeline_prior", "synthetic_medium_gap_prior"]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "rebake_flow_status": dict(Counter(str(row.get("rebake_flow_status") or "unknown") for row in rows)),
        "source_trace_status": dict(Counter(str(row.get("source_trace_status") or "unknown") for row in rows)),
        "material_status": dict(Counter(str(row.get("material_status") or "unknown") for row in rows)),
        "license_status": dict(Counter(str(row.get("license_status") or "unknown") for row in rows)),
        "material_family": dict(Counter(str(row.get("material_family") or "unknown") for row in rows)),
        "source_name": dict(Counter(str(row.get("source_name") or "unknown") for row in rows)),
        "original_blocked_reason": dict(Counter(reason for row in rows for reason in row.get("original_blocked_reason", []))),
    }


def build_record(record: dict[str, Any], reject_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    physical = first_existing_path(record)
    license_state = license_status(record)
    source_trace = trace_status(record, physical, license_state)
    flow = "queue_ready" if source_trace == "trace_ok" and license_state != "hard_blocked" else "quarantine"
    family = normalized_material_family(record)
    score, reasons = priority(record, family)
    item = dict(record)
    item.update(
        {
            "candidate_status": "target_rebake_candidate" if flow == "queue_ready" else "quarantine",
            "rebake_flow_status": flow,
            "source_trace_status": source_trace,
            "original_blocked_reason": original_reasons(record, reject_index),
            "material_status": material_status(record),
            "license_status": license_state,
            "engineering_only_no_paper_claim": license_state == "pending_engineering_only",
            "material_family": family,
            "expected_material_family": family,
            "path_resolved_ok": bool(physical),
            "physical_path": physical,
            "asset_path": physical or str(record.get("asset_path") or ""),
            "logical_path": physical or str(record.get("logical_path") or ""),
            "storage_tier": storage_tier(physical),
            "priority_score": score if flow == "queue_ready" else 0.0,
            "priority_reason": reasons if flow == "queue_ready" else [source_trace, license_state],
            "blocked_reason": [] if flow == "queue_ready" else [source_trace, license_state],
            "expected_prior_variant_types": expected_prior_variant_types(record),
            "recommended_storage_tier": "hdd_archive",
        }
    )
    if physical and not item.get("source_model_path"):
        item["source_model_path"] = physical
    return item


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "object_id",
                "source_name",
                "source_trace_status",
                "rebake_flow_status",
                "material_status",
                "license_status",
                "material_family",
                "physical_path",
                "original_blocked_reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            item = {key: row.get(key) for key in writer.fieldnames}
            item["original_blocked_reason"] = ";".join(row.get("original_blocked_reason", []))
            writer.writerow(item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reject-manifest", type=Path, default=DEFAULT_REJECT)
    parser.add_argument("--pending-manifest", type=Path, default=DEFAULT_PENDING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--queue-copy", type=Path, default=DEFAULT_QUEUE_COPY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_rows = records(read_json(args.reject_manifest, {"records": []}))
    pending_rows = records(read_json(args.pending_manifest, {"records": []}))
    reject_index = {object_id(row): row for row in reject_rows if object_id(row)}
    source_rows = pending_rows or reject_rows
    seen: set[str] = set()
    provenance: list[dict[str, Any]] = []
    for row in source_rows:
        oid = object_id(row)
        if not oid or oid in seen:
            continue
        seen.add(oid)
        provenance.append(build_record(row, reject_index))
    queue_ready = sorted(
        [row for row in provenance if row["rebake_flow_status"] == "queue_ready"],
        key=lambda row: (float(row.get("priority_score", 0.0)), str(row.get("object_id") or "")),
        reverse=True,
    )
    quarantine = [row for row in provenance if row["rebake_flow_status"] == "quarantine"]
    base = {
        "generated_at_utc": utc_now(),
        "provenance_version": "trainV5_reject_unknown_provenance_v1",
        "source_reject_manifest": str(args.reject_manifest.resolve()),
        "source_pending_manifest": str(args.pending_manifest.resolve()),
        "queue_scope": "reject_unknown_pending_only",
    }
    write_json(args.output_dir / "reject_unknown_provenance_manifest.json", {**base, "summary": summarize(provenance), "records": provenance})
    write_json(args.output_dir / "trace_ok_queue_ready_manifest.json", {**base, "summary": summarize(queue_ready), "records": queue_ready})
    write_json(args.output_dir / "quarantine_manifest.json", {**base, "summary": summarize(quarantine), "records": quarantine})
    queue_payload = {**base, "summary": summarize(queue_ready), "records": queue_ready}
    write_json(args.output_dir / "trainV5_plus_rebake_queue_preview_v3.json", queue_payload)
    write_json(args.queue_copy, queue_payload)
    write_csv(args.output_dir / "reject_unknown_provenance_table.csv", provenance)
    report = [
        "# Reject/Unknown Provenance Report",
        "",
        f"- generated_at_utc: `{base['generated_at_utc']}`",
        f"- input_reject_records: `{len(reject_rows)}`",
        f"- input_pending_records: `{len(pending_rows)}`",
        f"- provenance_records: `{len(provenance)}`",
        f"- queue_ready: `{len(queue_ready)}`",
        f"- quarantine: `{len(quarantine)}`",
        f"- source_trace_status: `{json.dumps(summarize(provenance)['source_trace_status'], ensure_ascii=False)}`",
        f"- material_status: `{json.dumps(summarize(provenance)['material_status'], ensure_ascii=False)}`",
        f"- license_status: `{json.dumps(summarize(provenance)['license_status'], ensure_ascii=False)}`",
        f"- source_name: `{json.dumps(summarize(provenance)['source_name'], ensure_ascii=False)}`",
        "",
        "Unknown material is no longer a TrainV5 queue blocker. Pending-license rows are engineering-only and carry `engineering_only_no_paper_claim=true`.",
        "Raw assets and source manifests are not deleted.",
    ]
    write_text(args.output_dir / "reject_unknown_provenance_report.md", "\n".join(report))
    print(json.dumps({"queue_ready": len(queue_ready), "quarantine": len(quarantine), "summary": summarize(provenance)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
