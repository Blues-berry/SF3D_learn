#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output/material_refine_expansion_candidates/merged_expansion_candidate_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass"
QUEUE_GENERATION = "trainV5_material_first_screening_v1"

BLOCKED_REASONS = {
    "missing_asset",
    "license_blocked",
    "is_3dfuture",
    "polyhaven_not_object",
    "duplicate_object_id",
    "path_unresolved",
    "unknown_material",
}

ALLOWED_LICENSE_BUCKETS = {
    "cc_by_nc_4_0",
    "cc_by_nc_4_0_pending_reconcile",
    "custom_tianchi_research_noncommercial_no_redistribution",
    "Creative Commons Zero v1.0 Universal",
    "MIT License",
    "Apache License 2.0",
    'BSD 3-Clause "New" or "Revised" License',
    "The Unlicense",
    "Creative Commons - Attribution",
    "Creative Commons - Attribution - Share Alike",
    "cc_by_4_0_pending_reconcile",
    "smithsonian_open_access_pending_reconcile",
    "omniobject3d_license_pending_reconcile",
    "khronos_sample_license_pending_reconcile",
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
    "Local_Quaternius_CC0_increment": 16.0,
    "Local_Smithsonian_selected_increment": 28.0,
    "Local_OmniObject3D_increment": 24.0,
    "Khronos_highlight_reference_samples": 18.0,
}

KNOWN_MATERIAL_CONFIDENCE = {
    "source_semantic_locked_or_validated": 0.92,
    "manifest_keyword_or_metadata": 0.60,
    "pending_static_probe": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="CPU-only TrainV5 second-pass over merged expansion candidates using material-first screening.",
    )
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ssd-active-root", type=Path, default=REPO_ROOT / "dataoutput")
    parser.add_argument("--min-ssd-free-gb", type=float, default=20.0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def object_id(record: dict[str, Any]) -> str:
    return str(record.get("canonical_object_id") or record.get("object_id") or record.get("source_uid") or "")


def normalize_material_family(value: Any) -> str:
    family = str(value or "").strip()
    if family in {"", "unknown", "unknown_pending_second_pass", "pending_abo_semantic_classification"}:
        return "unknown_material_pending_probe"
    return family


def material_family(record: dict[str, Any]) -> str:
    return normalize_material_family(record.get("material_family") or record.get("highlight_material_class"))


def candidate_text(record: dict[str, Any]) -> str:
    keys = (
        "source_name",
        "source_name_detail",
        "source_dataset",
        "generator_id",
        "source_model_path",
        "canonical_glb_path",
        "canonical_mesh_path",
        "raw_asset_path",
        "pool_name",
        "notes",
    )
    return " ".join(str(record.get(key) or "") for key in keys).lower()


def is_3dfuture(record: dict[str, Any]) -> bool:
    text = candidate_text(record)
    if any(token in text for token in ("non_3dfuture", "no_3dfuture", "no-3dfuture", "no 3d-future", "no3dfuture")):
        return False
    return any(token in text for token in ("3d-future", "3d_future", "3dfuture"))


def is_polyhaven_non_object(record: dict[str, Any]) -> bool:
    text = candidate_text(record)
    return "polyhaven" in text or "poly haven" in text


def asset_paths(record: dict[str, Any]) -> list[str]:
    paths = []
    for key in ("source_model_path", "canonical_glb_path", "canonical_mesh_path", "raw_asset_path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    return paths


def path_exists(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.exists()


def usable_asset_path(record: dict[str, Any]) -> str:
    for value in asset_paths(record):
        if path_exists(value):
            path = Path(value)
            if not path.is_absolute():
                path = REPO_ROOT / path
            return str(path.resolve())
    return ""


def resolved(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return str(path.resolve())
    except OSError:
        return str(value)


def repo_logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        return str(Path(value).resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(value)


def storage_tier_for_path(value: Any) -> str:
    physical = resolved(value)
    if not physical:
        return "unknown"
    if physical.startswith("/4T/"):
        return "hdd_archive"
    if physical.startswith(str((REPO_ROOT / "dataoutput").resolve())):
        return "ssd_active"
    if physical.startswith(str(REPO_ROOT.resolve())):
        return "ssd_project"
    return "external_or_unknown"


def asset_size_mb(path: str) -> float | None:
    if not path:
        return None
    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        return candidate.stat().st_size / (1024 * 1024)
    except OSError:
        return None


def estimated_cost_level(size_mb: float | None) -> str:
    if size_mb is None:
        return "unknown"
    if size_mb < 20:
        return "low"
    if size_mb < 120:
        return "medium"
    return "high"


def expected_prior_variant_types(record: dict[str, Any]) -> list[str]:
    variants: list[str] = []
    has_prior = bool_value(record.get("has_material_prior"))
    prior_mode = str(record.get("prior_mode") or "").lower()
    if not has_prior or prior_mode in {"none", "no_prior", ""}:
        variants.append("no_prior_bootstrap")
        variants.append("synthetic_large_gap_prior")
    elif prior_mode == "scalar_rm":
        variants.append("scalar_broadcast_prior")
        variants.append("synthetic_mild_gap_prior")
        variants.append("synthetic_medium_gap_prior")
    elif prior_mode in {"uv_rm", "texture_rm", "spatial_map"}:
        variants.append("texture_rm_prior")
        variants.append("synthetic_mild_gap_prior")
        variants.append("synthetic_medium_gap_prior")
    else:
        variants.append("existing_pipeline_prior")
        variants.append("synthetic_medium_gap_prior")
    return variants


def classify_base_blockers(record: dict[str, Any], seen: set[str]) -> list[str]:
    blockers: list[str] = []
    oid = object_id(record)
    if not oid or oid in seen:
        blockers.append("duplicate_object_id")
    paths = asset_paths(record)
    if not paths:
        blockers.append("missing_asset")
    elif not any(path_exists(path) for path in paths):
        blockers.append("path_unresolved")
    license_bucket = str(record.get("license_bucket") or "unknown")
    license_status = str(record.get("license_status") or "").lower()
    if license_bucket and license_bucket not in ALLOWED_LICENSE_BUCKETS:
        if any(token in license_bucket.lower() or token in license_status for token in ("hard_block", "forbidden", "no_training")):
            blockers.append("license_blocked")
    if is_3dfuture(record):
        blockers.append("is_3dfuture")
    if is_polyhaven_non_object(record):
        blockers.append("polyhaven_not_object")
    return [reason for reason in blockers if reason in BLOCKED_REASONS]


def material_probe_stub(record: dict[str, Any], family: str) -> tuple[str, float, str]:
    if family == "unknown_material_pending_probe":
        return (
            "pending_static_probe",
            KNOWN_MATERIAL_CONFIDENCE["pending_static_probe"],
            "pending_static_probe",
        )
    source = str(record.get("material_class_source") or "").lower()
    if any(token in source for token in ("semantic", "validated", "locked_core")):
        return (
            "known_material_preclassified",
            KNOWN_MATERIAL_CONFIDENCE["source_semantic_locked_or_validated"],
            "manifest_material_label",
        )
    return (
        "known_material_preclassified",
        KNOWN_MATERIAL_CONFIDENCE["manifest_keyword_or_metadata"],
        "manifest_material_label",
    )


def priority(record: dict[str, Any], asset_path: str, family: str) -> tuple[float, list[str]]:
    score = MATERIAL_PRIORITY.get(family, 0.0)
    reasons = [f"material:{family}:{score:.0f}"]

    source = str(record.get("source_name") or "unknown")
    source_bonus = SOURCE_DIVERSITY_BONUS.get(source, 12.0 if source != "ABO_locked_core" else 0.0)
    score += source_bonus
    reasons.append(f"source_bonus:{source_bonus:.0f}")

    if asset_path:
        score += 6.0
        reasons.append("path_resolved:+6")

    license_bucket = str(record.get("license_bucket") or "")
    if license_bucket in ALLOWED_LICENSE_BUCKETS:
        bonus = 8.0 if "pending" not in license_bucket.lower() else 4.0
        score += bonus
        reasons.append(f"license_bonus:+{bonus:.0f}")

    size = asset_size_mb(asset_path)
    cost = estimated_cost_level(size)
    if cost == "medium":
        score -= 2.0
        reasons.append("cost_penalty:-2")
    elif cost == "high":
        score -= 8.0
        reasons.append("cost_penalty:-8")

    return round(score, 4), reasons


def recommended_storage_tier(score: float, asset_path: str, family: str) -> str:
    size = asset_size_mb(asset_path)
    if family in {"mixed_thin_boundary", "glass_metal", "ceramic_glazed_lacquer", "metal_dominant"} and score >= 100.0:
        if size is None or size <= 300:
            return "ssd_active_for_rebake_cache_then_hdd_archive"
    return "hdd_archive"


def build_candidate(record: dict[str, Any], seen: set[str]) -> dict[str, Any]:
    oid = object_id(record)
    family = material_family(record)
    blockers = classify_base_blockers(record, seen)
    if family == "unknown_material_pending_probe":
        blockers.append("unknown_material")
    if oid:
        seen.add(oid)

    asset_path = usable_asset_path(record)
    size = asset_size_mb(asset_path)
    score, reasons = priority(record, asset_path, family)
    probe_status, probe_confidence, probe_source = material_probe_stub(record, family)

    hard_blockers = {
        reason
        for reason in blockers
        if reason in {"missing_asset", "path_unresolved", "license_blocked", "is_3dfuture", "polyhaven_not_object", "duplicate_object_id"}
    }
    if hard_blockers:
        status = "hard_block_or_unusable"
        rebake_ready = False
    elif "unknown_material" in blockers:
        status = "pending_material_probe"
        rebake_ready = False
    else:
        status = "target_rebake_candidate"
        rebake_ready = True

    out = {
        "candidate_status": status,
        "blocked_reason": blockers,
        "priority_score": score,
        "priority_reason": reasons if not hard_blockers else sorted(hard_blockers),
        "screening_priority_score": score,
        "screening_priority_reason": reasons,
        "expected_material_family": family,
        "material_probe_status": probe_status,
        "material_probe_confidence": round(float(probe_confidence), 4),
        "material_probe_source": probe_source,
        "rebake_ready": rebake_ready,
        "queue_generation": QUEUE_GENERATION,
        "expected_prior_variant_types": expected_prior_variant_types(record),
        "recommended_storage_tier": (
            recommended_storage_tier(score, asset_path, family)
            if rebake_ready
            else ("hdd_archive_or_repair" if status == "pending_material_probe" else "hdd_archive_or_rejects")
        ),
        "estimated_cost_level": estimated_cost_level(size),
        "object_id": oid,
        "source_name": str(record.get("source_name") or ""),
        "source_dataset": str(record.get("source_dataset") or ""),
        "generator_id": str(record.get("generator_id") or ""),
        "license_bucket": str(record.get("license_bucket") or ""),
        "has_material_prior": bool_value(record.get("has_material_prior")),
        "prior_mode": str(record.get("prior_mode") or ""),
        "asset_path": asset_path,
        "asset_size_mb": size,
        "storage_tier": storage_tier_for_path(asset_path),
        "logical_path": repo_logical_path(asset_path),
        "physical_path": resolved(asset_path),
        "path_resolved_ok": bool(asset_path),
        "source_record": record,
    }
    unknown = [reason for reason in out["blocked_reason"] if reason not in BLOCKED_REASONS]
    if unknown:
        raise RuntimeError(f"unexpected_blocked_reason:{unknown}")
    return out


def read_records(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload if isinstance(payload, list) else payload.get("records", [])
    return [record for record in rows if isinstance(record, dict)]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "candidate_status": dict(Counter(str(record.get("candidate_status") or "unknown") for record in records)),
        "blocked_reason": dict(Counter(reason for record in records for reason in record.get("blocked_reason", []))),
        "material_family": dict(Counter(str(record.get("expected_material_family") or "unknown") for record in records)),
        "source_name": dict(Counter(str(record.get("source_name") or "unknown") for record in records)),
        "has_material_prior": {
            "true": sum(bool_value(record.get("has_material_prior")) for record in records),
            "false": sum(not bool_value(record.get("has_material_prior")) for record in records),
        },
        "prior_mode": dict(Counter(str(record.get("prior_mode") or "unknown") for record in records)),
        "estimated_cost_level": dict(Counter(str(record.get("estimated_cost_level") or "unknown") for record in records)),
        "material_probe_status": dict(Counter(str(record.get("material_probe_status") or "unknown") for record in records)),
    }


def inventory(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = Counter(
        variant
        for record in records
        for variant in record.get("expected_prior_variant_types", [])
    )
    priorities = [float(record.get("screening_priority_score", 0.0)) for record in records]
    return {
        "summary": summarize(records),
        "expected_prior_variant_types": dict(by_variant),
        "screening_priority_score": {
            "count": len(priorities),
            "mean": statistics.mean(priorities) if priorities else None,
            "max": max(priorities) if priorities else None,
            "min": min(priorities) if priorities else None,
        },
    }


def top_batch(records: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    return records[: min(size, len(records))]


def batch_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "material_family": dict(Counter(str(record.get("expected_material_family") or "unknown") for record in records)),
        "source_name": dict(Counter(str(record.get("source_name") or "unknown") for record in records)),
        "has_material_prior": {
            "true": sum(bool_value(record.get("has_material_prior")) for record in records),
            "false": sum(not bool_value(record.get("has_material_prior")) for record in records),
        },
        "expected_prior_variant_types": dict(
            Counter(variant for record in records for variant in record.get("expected_prior_variant_types", []))
        ),
        "recommended_storage_tier": dict(Counter(str(record.get("recommended_storage_tier") or "unknown") for record in records)),
    }


def write_report(path: Path, all_records: list[dict[str, Any]], target: list[dict[str, Any]], pending: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    pending_sources = Counter(str(record.get("source_name") or "unknown") for record in pending)
    lines = [
        "# TrainV5 Expansion Second-Pass Report",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- total_candidates: `{len(all_records)}`",
        f"- target_rebake_candidates: `{len(target)}`",
        f"- pending_material_probe: `{len(pending)}`",
        f"- hard_block_or_unusable: `{len(rejected)}`",
        f"- status_counts: `{json.dumps(summarize(all_records)['candidate_status'], ensure_ascii=False)}`",
        f"- blocked_reason_counts: `{json.dumps(summarize(all_records)['blocked_reason'], ensure_ascii=False)}`",
        f"- target_material_family: `{json.dumps(summarize(target)['material_family'], ensure_ascii=False)}`",
        f"- target_source_name: `{json.dumps(summarize(target)['source_name'], ensure_ascii=False)}`",
        f"- pending_unknown_by_source: `{json.dumps(dict(pending_sources), ensure_ascii=False)}`",
        "",
        "This pass is CPU/metadata only. It does not launch GPU, Blender, R training, or upstream prior generators.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quota(path: Path, target: list[dict[str, Any]]) -> None:
    batch0 = top_batch(target, 64)
    batch256 = top_batch(target, 256)
    batch512 = top_batch(target, 512)
    batch1000 = top_batch(target, 1000)
    lines = [
        "# TrainV5 Plus Quota Recommendation",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- full_sorted_target_rebake_candidates: `{len(target)}`",
        "",
        "## Batch Plan",
        "",
        "- Batch-0 sanity rebake: `64`",
        "- Batch-1 pilot rebake: `256/512`",
        "- Batch-2 expansion rebake: `1000`",
        "- Batch-3 large rebake: expand based on success rate, storage, and material coverage.",
        "",
        "## Batch Summaries",
        "",
        f"- batch_0_64: `{json.dumps(batch_summary(batch0), ensure_ascii=False)}`",
        f"- batch_1_256: `{json.dumps(batch_summary(batch256), ensure_ascii=False)}`",
        f"- batch_1_512: `{json.dumps(batch_summary(batch512), ensure_ascii=False)}`",
        f"- batch_2_1000: `{json.dumps(batch_summary(batch1000), ensure_ascii=False)}`",
        "",
        "## Material-First Rule",
        "",
        "- Sorting is driven by `expected_material_family` first, then source/path/license/cost bonuses.",
        "- `expected_prior_variant_types` remain planning hints only and do not control queue order.",
        "- `no_prior` and `scalar_broadcast` are not primary screening weights in this pass.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def free_gb(path: Path) -> float | None:
    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    return float(stat.f_bavail * stat.f_frsize / (1024**3))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ssd_active_root.mkdir(parents=True, exist_ok=True)
    source_records = read_records(args.input_manifest)
    seen: set[str] = set()
    candidates = [build_candidate(record, seen) for record in source_records]
    target = sorted(
        [record for record in candidates if record["candidate_status"] == "target_rebake_candidate"],
        key=lambda record: (float(record.get("screening_priority_score", 0.0)), str(record.get("object_id") or "")),
        reverse=True,
    )
    pending = [
        record
        for record in candidates
        if record["candidate_status"] == "pending_material_probe"
    ]
    rejected = [
        record
        for record in candidates
        if record["candidate_status"] == "hard_block_or_unusable"
    ]
    usable = [record for record in candidates if record.get("rebake_ready")]

    base = {
        "generated_at_utc": utc_now(),
        "source_manifest": str(args.input_manifest.resolve()),
        "second_pass_version": "trainV5_expansion_second_pass_material_first_v2",
        "queue_generation": QUEUE_GENERATION,
        "full_input_count": len(source_records),
        "dataoutput_free_gb": free_gb(args.ssd_active_root),
        "min_ssd_free_gb": args.min_ssd_free_gb,
    }
    write_json(args.output_dir / "usable_candidates.json", {**base, "summary": summarize(usable), "records": usable})
    write_json(args.output_dir / "target_rebake_candidates.json", {**base, "summary": summarize(target), "records": target})
    write_json(args.output_dir / "reject_or_unknown_candidates.json", {**base, "summary": summarize(pending + rejected), "records": pending + rejected})
    write_json(args.output_dir / "material_prior_gap_inventory.json", {**base, **inventory(target)})
    queue_payload = {
        **base,
        "queue_policy": {
            "sort": "screening_priority_score_desc_then_object_id",
            "material_first": True,
            "prior_hints_non_blocking": True,
            "batch_plan": {
                "batch_0_sanity_rebake": 64,
                "batch_1_pilot_rebake": [256, 512],
                "batch_2_expansion_rebake": 1000,
                "batch_3_large_rebake": "expand based on success rate, storage, and material coverage",
            },
        },
        "summary": summarize(target),
        "records": target,
    }
    write_json(args.output_dir / "trainV5_plus_rebake_queue_preview.json", queue_payload)
    write_json(args.output_dir / "trainV5_plus_rebake_queue_latest.json", queue_payload)
    write_report(args.output_dir / "second_pass_report.md", candidates, target, pending, rejected)
    write_quota(args.output_dir / "trainV5_plus_quota_recommendation.md", target)
    print(
        json.dumps(
            {
                "input_candidates": len(source_records),
                "processed_candidates": len(candidates),
                "target_rebake_candidates": len(target),
                "pending_material_probe": len(pending),
                "hard_block_or_unusable": len(rejected),
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
