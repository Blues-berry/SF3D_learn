#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECOND_PASS_DIR = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass"
REPAIRABLE_REASONS = {"unknown_material", "path_unresolved", "missing_asset", "license_blocked", "duplicate_object_id"}
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
MATERIAL_PRIORITY = {
    "mixed_thin_boundary": 120.0,
    "glass_metal": 112.0,
    "ceramic_glazed_lacquer": 105.0,
    "metal_dominant": 100.0,
    "glossy_non_metal": 35.0,
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def records(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Repair TrainV5 reject/unknown candidates and write queue-v2/expansion planning artifacts.",
    )
    parser.add_argument("--second-pass-dir", type=Path, default=DEFAULT_SECOND_PASS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def object_id(record: dict[str, Any]) -> str:
    return str(record.get("object_id") or (record.get("source_record") or {}).get("object_id") or "")


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def path_exists(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).exists()


def resolved(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        return str(Path(value).resolve())
    except OSError:
        return str(value)


def storage_tier(value: Any) -> str:
    physical = resolved(value)
    if not physical:
        return "unknown"
    if physical.startswith("/4T/"):
        return "hdd_archive"
    if physical.startswith(str((REPO_ROOT / "dataoutput").resolve())):
        return "ssd_active"
    if physical.startswith(str(REPO_ROOT.resolve())):
        return "ssd_project_or_output_symlink"
    return "external_or_unknown"


def repo_logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        return str(Path(value).resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(value)


def candidate_text(record: dict[str, Any]) -> str:
    source = record.get("source_record") or {}
    keys = (
        "object_id",
        "source_name",
        "source_dataset",
        "generator_id",
        "asset_path",
        "physical_path",
        "logical_path",
        "source_model_path",
        "raw_asset_path",
        "canonical_glb_path",
        "canonical_mesh_path",
        "source_texture_root",
        "category",
        "category_bucket",
        "tags",
        "name",
        "title",
        "notes",
    )
    return " ".join(str(record.get(key) or source.get(key) or "") for key in keys).lower()


def material_repair(record: dict[str, Any]) -> str:
    text = candidate_text(record)
    if any(token in text for token in ("glass", "bottle", "transparent", "transmission", "window", "vase")) and any(token in text for token in ("metal", "chrome", "steel", "aluminum", "iron")):
        return "glass_metal"
    if any(token in text for token in ("metal", "chrome", "steel", "aluminum", "brass", "copper", "iron", "silver", "gold")):
        return "metal_dominant"
    if any(token in text for token in ("ceramic", "porcelain", "lacquer", "glazed", "enamel", "tile", "mug", "plate", "bowl")):
        return "ceramic_glazed_lacquer"
    if any(token in text for token in ("thin", "wire", "frame", "grille", "fence", "chain", "mesh", "leaf", "plant", "basket")):
        return "mixed_thin_boundary"
    if any(token in text for token in ("gloss", "plastic", "polished", "shiny", "toy", "shoe", "helmet", "case", "chair", "lamp")):
        return "glossy_non_metal"
    return "still_unknown"


def license_repair(record: dict[str, Any]) -> dict[str, Any]:
    bucket = str(record.get("license_bucket") or "")
    if bucket in STRICT_ALLOWED_LICENSE_BUCKETS:
        status = "allowed"
        research = True
        training = True
    elif bucket in PENDING_LICENSE_BUCKETS or "pending" in bucket.lower():
        status = "pending_review"
        research = True
        training = False
    else:
        status = "hard_blocked"
        research = False
        training = False
    return {
        "license_bucket": bucket,
        "license_status": status,
        "license_allowed_for_research": research,
        "license_allowed_for_training": training,
    }


def candidate_paths(record: dict[str, Any]) -> list[str]:
    source = record.get("source_record") or {}
    paths = []
    for key in ("asset_path", "physical_path", "source_model_path", "raw_asset_path", "canonical_glb_path", "canonical_mesh_path", "source_texture_root"):
        value = record.get(key) or source.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    return paths


def path_repair(record: dict[str, Any]) -> dict[str, Any]:
    for path in candidate_paths(record):
        if path_exists(path):
            return {
                "path_resolved_ok": True,
                "physical_path": resolved(path),
                "logical_path": repo_logical_path(path),
                "storage_tier": storage_tier(path),
                "asset_path": resolved(path),
            }
    return {
        "path_resolved_ok": False,
        "physical_path": "",
        "logical_path": "",
        "storage_tier": "unknown",
        "asset_path": "",
    }


def expected_prior_variant_types(record: dict[str, Any]) -> list[str]:
    has_prior = bool_value(record.get("has_material_prior"))
    prior_mode = str(record.get("prior_mode") or "").lower()
    if not has_prior or prior_mode in {"none", "no_prior", ""}:
        return ["no_prior_bootstrap", "synthetic_large_gap_prior"]
    if prior_mode == "scalar_rm":
        return ["scalar_broadcast_prior", "synthetic_mild_gap_prior", "synthetic_medium_gap_prior"]
    if prior_mode in {"uv_rm", "texture_rm", "spatial_map"}:
        return ["texture_rm_prior", "synthetic_mild_gap_prior", "synthetic_medium_gap_prior"]
    return ["existing_pipeline_prior", "synthetic_medium_gap_prior"]


def priority(record: dict[str, Any]) -> tuple[float, list[str]]:
    family = str(record.get("expected_material_family") or "still_unknown")
    score = MATERIAL_PRIORITY.get(family, 0.0)
    reasons = [f"material:{family}:{score:.0f}"]
    if not bool_value(record.get("has_material_prior")):
        score += 45.0
        reasons.append("no_prior:+45")
    if str(record.get("prior_mode") or "").lower() == "scalar_rm":
        score += 18.0
        reasons.append("scalar_broadcast_prior:+18")
    source = str(record.get("source_name") or "unknown")
    bonus = SOURCE_DIVERSITY_BONUS.get(source, 12.0 if source != "ABO_locked_core" else 0.0)
    score += bonus
    reasons.append(f"source_diversity:{bonus:.0f}")
    return round(score, 4), reasons


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "candidate_status": dict(Counter(str(row.get("candidate_status") or "unknown") for row in rows)),
        "blocked_reason": dict(Counter(reason for row in rows for reason in row.get("blocked_reason", []))),
        "material_family": dict(Counter(str(row.get("expected_material_family") or "unknown") for row in rows)),
        "source_name": dict(Counter(str(row.get("source_name") or "unknown") for row in rows)),
        "license_status": dict(Counter(str(row.get("license_status") or "unknown") for row in rows)),
        "prior_mode": dict(Counter(str(row.get("prior_mode") or "unknown") for row in rows)),
        "has_material_prior": {
            "true": sum(bool_value(row.get("has_material_prior")) for row in rows),
            "false": sum(not bool_value(row.get("has_material_prior")) for row in rows),
        },
    }


def split_reason_buckets(rejected: list[dict[str, Any]], out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "repairable_candidates": [],
        "hard_reject_candidates": [],
        "pending_repair_candidates": [],
        "missing_asset_or_path_unresolved": [],
        "unknown_material_candidates": [],
        "license_pending_candidates": [],
        "license_hard_blocked_candidates": [],
        "duplicate_candidates": [],
        "source_policy_blocked_candidates": [],
        "polyhaven_auxiliary_records": [],
    }
    for row in rejected:
        reasons = set(row.get("blocked_reason") or [])
        license_info = license_repair(row)
        if "unknown_material" in reasons:
            buckets["unknown_material_candidates"].append(row)
        if reasons & {"missing_asset", "path_unresolved"}:
            buckets["missing_asset_or_path_unresolved"].append(row)
        if "duplicate_object_id" in reasons:
            buckets["duplicate_candidates"].append(row)
        if reasons & {"is_3dfuture", "polyhaven_not_object"}:
            buckets["source_policy_blocked_candidates"].append(row)
        if "polyhaven_not_object" in reasons:
            buckets["polyhaven_auxiliary_records"].append(row)
        if license_info["license_status"] == "pending_review":
            buckets["license_pending_candidates"].append(row)
        if license_info["license_status"] == "hard_blocked":
            buckets["license_hard_blocked_candidates"].append(row)
        if reasons and reasons <= REPAIRABLE_REASONS:
            buckets["repairable_candidates"].append(row)
        elif reasons:
            buckets["hard_reject_candidates"].append(row)
        else:
            buckets["pending_repair_candidates"].append(row)
    for name, rows in buckets.items():
        write_json(out_dir / f"{name}.json", {"generated_at_utc": utc_now(), "summary": summarize(rows), "records": rows})
    reason_counts = Counter(reason for row in rejected for reason in row.get("blocked_reason", []))
    write_md(
        out_dir / "reject_reason_breakdown.md",
        [
            "# Reject Reason Breakdown",
            "",
            f"- generated_at_utc: `{utc_now()}`",
            f"- records: `{len(rejected)}`",
            f"- blocked_reason: `{json.dumps(dict(reason_counts), ensure_ascii=False)}`",
            f"- repairable_candidates: `{len(buckets['repairable_candidates'])}`",
            f"- hard_reject_candidates: `{len(buckets['hard_reject_candidates'])}`",
            f"- pending_repair_candidates: `{len(buckets['pending_repair_candidates'])}`",
        ],
    )
    return buckets


def build_repaired_candidates(rejected: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    repaired: list[dict[str, Any]] = []
    material_repaired: list[dict[str, Any]] = []
    still_unknown: list[dict[str, Any]] = []
    license_repaired: list[dict[str, Any]] = []
    license_pending: list[dict[str, Any]] = []
    license_hard: list[dict[str, Any]] = []
    path_repaired: list[dict[str, Any]] = []
    path_missing: list[dict[str, Any]] = []
    for row in rejected:
        item = dict(row)
        material = str(item.get("expected_material_family") or "")
        if material in {"", "unknown", "unknown_pending_second_pass", "pending_abo_semantic_classification"}:
            material = material_repair(item)
        item["expected_material_family"] = material
        if material == "still_unknown":
            still_unknown.append(item)
        else:
            material_repaired.append(item)
        license_info = license_repair(item)
        item.update(license_info)
        if license_info["license_status"] == "allowed":
            license_repaired.append(item)
        elif license_info["license_status"] == "pending_review":
            license_pending.append(item)
        else:
            license_hard.append(item)
        path_info = path_repair(item)
        item.update(path_info)
        if path_info["path_resolved_ok"]:
            path_repaired.append(item)
        else:
            path_missing.append(item)
        repaired.append(item)

    write_json(out_dir / "material_repaired_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(material_repaired), "records": material_repaired})
    write_json(out_dir / "still_unknown_material_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(still_unknown), "records": still_unknown})
    write_json(out_dir / "license_repaired_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(license_repaired), "records": license_repaired})
    write_json(out_dir / "license_hard_blocked_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(license_hard), "records": license_hard})
    write_json(out_dir / "license_pending_after_repair.json", {"generated_at_utc": utc_now(), "summary": summarize(license_pending), "records": license_pending})
    write_json(out_dir / "path_repaired_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(path_repaired), "records": path_repaired})
    write_json(out_dir / "path_still_missing_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(path_missing), "records": path_missing})
    write_json(out_dir / "path_remap.json", {"generated_at_utc": utc_now(), "records": [{"object_id": object_id(row), "physical_path": row.get("physical_path"), "logical_path": row.get("logical_path")} for row in path_repaired]})
    write_md(out_dir / "material_repair_report.md", ["# Material Repair Report", "", f"- material_repaired: `{len(material_repaired)}`", f"- still_unknown: `{len(still_unknown)}`", f"- distribution: `{json.dumps(summarize(material_repaired)['material_family'], ensure_ascii=False)}`"])
    write_md(out_dir / "license_repair_report.md", ["# License Repair Report", "", f"- allowed: `{len(license_repaired)}`", f"- pending_review: `{len(license_pending)}`", f"- hard_blocked: `{len(license_hard)}`", "", "Pending licenses are not silently promoted to training-allowed."])
    write_md(out_dir / "path_repair_report.md", ["# Path Repair Report", "", f"- path_repaired: `{len(path_repaired)}`", f"- path_still_missing: `{len(path_missing)}`"])
    return repaired, {
        "path_repaired": len(path_repaired),
        "material_repaired": len(material_repaired),
        "still_unknown": len(still_unknown),
        "license_hard_blocked": len(license_hard),
        "license_pending": len(license_pending),
    }


def dedup_candidates(rows: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    suppressed: list[dict[str, Any]] = []

    def score(row: dict[str, Any]) -> tuple[int, int, int, int, float]:
        return (
            int(bool_value(row.get("path_resolved_ok"))),
            int(row.get("license_status") == "allowed"),
            int(str(row.get("expected_material_family")) != "still_unknown"),
            int(str(row.get("source_name")) != "ABO_locked_core"),
            float(row.get("priority_score", 0.0) or 0.0),
        )

    for row in rows:
        oid = object_id(row)
        if not oid:
            suppressed.append(row)
            continue
        current = selected.get(oid)
        if current is None or score(row) > score(current):
            if current is not None:
                suppressed.append(current)
            selected[oid] = row
        else:
            suppressed.append(row)
    chosen = list(selected.values())
    write_json(out_dir / "dedup_selected_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(chosen), "records": chosen})
    write_json(out_dir / "duplicate_suppressed_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(suppressed), "records": suppressed})
    write_md(out_dir / "dedup_report.md", ["# Dedup Report", "", f"- selected: `{len(chosen)}`", f"- duplicate_suppressed: `{len(suppressed)}`"])
    return chosen, suppressed


def second_pass(rows: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    usable: list[dict[str, Any]] = []
    target: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        blockers: list[str] = []
        if not bool_value(item.get("path_resolved_ok")):
            blockers.append("path_unresolved")
        if item.get("license_status") != "allowed":
            blockers.append("license_blocked" if item.get("license_status") == "hard_blocked" else "license_pending")
        if str(item.get("expected_material_family")) == "still_unknown":
            blockers.append("unknown_material")
        item["blocked_reason"] = blockers
        if blockers:
            item["candidate_status"] = "reject_or_unknown"
            item["priority_score"] = 0.0
            rejected.append(item)
        else:
            item["candidate_status"] = "target_rebake_candidate"
            item["expected_prior_variant_types"] = expected_prior_variant_types(item)
            item["priority_score"], item["priority_reason"] = priority(item)
            item["recommended_storage_tier"] = "ssd_active_for_rebake_cache_then_hdd_archive" if item["priority_score"] >= 120 else "hdd_archive"
            target.append(item)
            usable.append(item)
    target = sorted(target, key=lambda row: (float(row.get("priority_score", 0.0)), str(row.get("object_id") or "")), reverse=True)
    base = {"generated_at_utc": utc_now(), "second_pass_version": "trainV5_repaired_second_pass_v1"}
    write_json(out_dir / "repaired_usable_candidates.json", {**base, "summary": summarize(usable), "records": usable})
    write_json(out_dir / "repaired_target_rebake_candidates.json", {**base, "summary": summarize(target), "records": target})
    write_json(out_dir / "repaired_reject_or_unknown_candidates.json", {**base, "summary": summarize(rejected), "records": rejected})
    write_json(out_dir / "repaired_trainV5_plus_rebake_queue_preview.json", {**base, "summary": summarize(target), "records": target})
    write_md(out_dir / "repaired_second_pass_report.md", ["# Repaired Second-Pass Report", "", f"- repaired_target_rebake_candidates: `{len(target)}`", f"- repaired_reject_or_unknown_candidates: `{len(rejected)}`", f"- material_family: `{json.dumps(summarize(target)['material_family'], ensure_ascii=False)}`"])
    return usable, target, rejected


def write_quota(path: Path, target: list[dict[str, Any]]) -> None:
    def batch_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "records": len(rows),
            "material_family": dict(Counter(str(row.get("expected_material_family") or "unknown") for row in rows)),
            "source_name": dict(Counter(str(row.get("source_name") or "unknown") for row in rows)),
            "has_material_prior": {"true": sum(bool_value(row.get("has_material_prior")) for row in rows), "false": sum(not bool_value(row.get("has_material_prior")) for row in rows)},
            "expected_prior_variant_types": dict(Counter(variant for row in rows for variant in row.get("expected_prior_variant_types", []))),
            "recommended_storage_tier": dict(Counter(str(row.get("recommended_storage_tier") or "unknown") for row in rows)),
        }
    lines = [
        "# TrainV5 Plus Quota Recommendation V2",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- full_sorted_target_rebake_candidates: `{len(target)}`",
        "",
        "## Batch Summaries",
        "",
        f"- batch_0_64: `{json.dumps(batch_summary(target[:64]), ensure_ascii=False)}`",
        f"- batch_1_256: `{json.dumps(batch_summary(target[:256]), ensure_ascii=False)}`",
        f"- batch_1_512: `{json.dumps(batch_summary(target[:512]), ensure_ascii=False)}`",
        f"- batch_2_1000: `{json.dumps(batch_summary(target[:1000]), ensure_ascii=False)}`",
        "",
        "This is a queue preview only. It does not launch GPU rebake.",
    ]
    write_md(path, lines)


def write_expansion_plan(base_dir: Path) -> None:
    out_dir = base_dir / "data_expansion_plan"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Smithsonian_OpenAccess", "mixed_thin_boundary/glass/ceramic", "high", "pending_policy", "medium", "glb/obj/usdz", "medium", "non_ABO_material_diversity", "license_review"),
        ("Kenney_CC0", "stylized glossy/nonmetal/thin", "high", "low", "low", "glb/obj", "high", "no_prior_bootstrap_and_large_gap", "expand_now"),
        ("GSO", "household ceramic/metal/plastic", "medium", "pending_policy", "medium", "obj/glb", "medium", "pilot_source_diversity", "probe_first"),
        ("OmniObject3D", "broad real object coverage", "high", "pending_policy", "high", "glb/obj", "medium", "future_spatial_prior_eval", "license_review"),
        ("Objaverse-XL_strict", "broad long tail", "high", "medium", "medium", "glb", "medium", "material_gap_repair_pool", "probe_first"),
        ("PolyHaven_aux", "materials only/non-object", "low", "low", "low", "textures/hdr", "high", "diagnostics_only", "diagnostic_only"),
    ]
    with (out_dir / "trainV5_external_source_priority_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_name", "expected_material_strength", "expected_no_prior_value", "license_risk", "integration_cost", "asset_format", "expected_path_reliability", "expected_trainV5_role", "recommended_action"])
        writer.writerows(rows)
    write_md(
        out_dir / "trainV5_external_expansion_plan.md",
        [
            "# TrainV5 External Expansion Plan",
            "",
            f"- generated_at_utc: `{utc_now()}`",
            "",
            "## Goals",
            "",
            "- Fill material gaps: `metal_dominant`, `ceramic_glazed_lacquer`, `glass_metal`, `mixed_thin_boundary`.",
            "- Fill prior gaps: no-prior, scalar broadcast, spatial map, and future upstream-native priors.",
            "- Reduce ABO dependence with license-clear local and external object sources.",
            "- Stage future upstream shapes as sf3d-like scalar/low-frequency, spar3d-like coarse spatial, and hunyuan3d-like richer spatial priors without launching online generation in this pass.",
        ],
    )


def main() -> None:
    args = parse_args()
    second_pass_dir = args.second_pass_dir
    repair_dir = args.output_dir or (second_pass_dir / "repair")
    repair_dir.mkdir(parents=True, exist_ok=True)
    rejected = records(second_pass_dir / "reject_or_unknown_candidates.json")
    original_target = records(second_pass_dir / "trainV5_plus_rebake_queue_preview.json")
    split_reason_buckets(rejected, repair_dir)
    repaired_rows, repair_counts = build_repaired_candidates(rejected, repair_dir)
    dedup_selected, duplicate_suppressed = dedup_candidates(repaired_rows, repair_dir)
    repaired_second_pass_dir = second_pass_dir / "repaired_second_pass"
    _, repaired_target, repaired_rejected = second_pass(dedup_selected, repaired_second_pass_dir)

    merged: dict[str, dict[str, Any]] = {}
    for row in original_target + repaired_target:
        oid = object_id(row)
        if not oid:
            continue
        current = merged.get(oid)
        if current is None or float(row.get("priority_score", 0.0) or 0.0) > float(current.get("priority_score", 0.0) or 0.0):
            merged[oid] = row
    final_target = sorted(merged.values(), key=lambda row: (float(row.get("priority_score", 0.0) or 0.0), str(row.get("object_id") or "")), reverse=True)
    write_json(second_pass_dir / "trainV5_plus_rebake_queue_preview_v2.json", {"generated_at_utc": utc_now(), "queue_policy": "original_1153_plus_repaired_dedup_priority_sort", "summary": summarize(final_target), "records": final_target})
    write_quota(second_pass_dir / "trainV5_plus_quota_recommendation_v2.md", final_target)
    final_counts = {
        "original_target_rebake_candidates": len(original_target),
        "repaired_target_rebake_candidates": len(repaired_target),
        "final_target_rebake_candidates": len(final_target),
        "hard_reject": repair_counts["license_hard_blocked"],
        "pending_repair": len(repaired_rejected),
        "still_unknown": repair_counts["still_unknown"],
        "duplicate_suppressed": len(duplicate_suppressed),
        "license_hard_blocked": repair_counts["license_hard_blocked"],
        "path_repaired": repair_counts["path_repaired"],
        "material_repaired": repair_counts["material_repaired"],
    }
    write_md(
        second_pass_dir / "final_candidate_pool_report.md",
        [
            "# TrainV5 Final Candidate Pool Report",
            "",
            f"- generated_at_utc: `{utc_now()}`",
            *[f"- {key}: `{value}`" for key, value in final_counts.items()],
            f"- material_family_before: `{json.dumps(summarize(original_target)['material_family'], ensure_ascii=False)}`",
            f"- material_family_after: `{json.dumps(summarize(final_target)['material_family'], ensure_ascii=False)}`",
            f"- source_distribution_before: `{json.dumps(summarize(original_target)['source_name'], ensure_ascii=False)}`",
            f"- source_distribution_after: `{json.dumps(summarize(final_target)['source_name'], ensure_ascii=False)}`",
            f"- prior_availability_after: `{json.dumps(summarize(final_target)['has_material_prior'], ensure_ascii=False)}`",
            "- recommended_next_rebake_batch_size: `64`",
            "",
            "No raw assets were deleted and no full GPU rebake was launched.",
        ],
    )
    write_expansion_plan(second_pass_dir.parent)
    print(json.dumps(final_counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
