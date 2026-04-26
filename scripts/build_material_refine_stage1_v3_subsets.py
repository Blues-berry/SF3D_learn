#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.manifest_quality import (
    DEFAULT_MIN_TARGET_CONFIDENCE_ACTIVE_COVERAGE_FOR_PAPER,
    DEFAULT_MIN_TARGET_CONFIDENCE_ACTIVE_MEAN_FOR_PAPER,
    audit_manifest,
    confidence_active_mean,
)


PAPER_TIERS = {"paper_strong", "paper_pseudo"}
BLOCKED_TARGET_SOURCES = {"copied_from_prior", "unknown", "real_benchmark_no_uv_target"}
DEFAULT_LICENSE_BUCKETS = {
    "cc_by_nc_4_0",
    "cc_by_nc_4_0_pending_reconcile",
    "custom_tianchi_research_noncommercial_no_redistribution",
    "Creative Commons Zero v1.0 Universal",
    "MIT License",
    "Apache License 2.0",
    "BSD 3-Clause \"New\" or \"Revised\" License",
    "The Unlicense",
    "Creative Commons - Attribution",
}
DEFAULT_QUOTAS = {
    "metal_dominant": 0.30,
    "ceramic_glazed_lacquer": 0.20,
    "glass_metal": 0.15,
    "mixed_thin_boundary": 0.20,
    "glossy_non_metal": 0.15,
}
DEFAULT_MAIN_TRAIN_SOURCES = {
    "ABO_locked_core",
    "3D-FUTURE_highlight_local_8k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Build Stage1-v3 data-only material-refine subsets from promoted manifests: "
            "strict candidates, quota-balanced paper subset, diagnostic subset, OOD subset, "
            "auxiliary upgrade queue, rejects, and an audit report."
        ),
    )
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-records", type=int, default=900)
    parser.add_argument("--min-paper-eligible", type=int, default=800)
    parser.add_argument("--min-material-family-records", type=int, default=64)
    parser.add_argument("--max-material-family-ratio", type=float, default=0.40)
    parser.add_argument("--min-no-prior-records", type=int, default=100)
    parser.add_argument("--min-secondary-source-records", type=int, default=100)
    parser.add_argument("--min-confidence-mean", type=float, default=0.70)
    parser.add_argument("--target-confidence-mean", type=float, default=0.75)
    parser.add_argument("--min-confidence-nonzero-rate", type=float, default=0.50)
    parser.add_argument("--min-target-coverage", type=float, default=0.50)
    parser.add_argument("--identity-like-threshold", type=float, default=0.999)
    parser.add_argument("--min-valid-view-count", type=int, default=1)
    parser.add_argument("--max-diagnostic-records", type=int, default=1024)
    parser.add_argument("--max-ood-records", type=int, default=512)
    parser.add_argument("--diagnostic-min-per-material-family", type=int, default=32)
    parser.add_argument("--ood-min-per-material-family", type=int, default=24)
    parser.add_argument("--paper-license-buckets", type=str, default=",".join(sorted(DEFAULT_LICENSE_BUCKETS)))
    parser.add_argument(
        "--material-quotas",
        type=str,
        default=",".join(f"{key}:{value}" for key, value in DEFAULT_QUOTAS.items()),
        help="Comma-separated material quota pairs, for example metal_dominant:0.30,glass_metal:0.15.",
    )
    parser.add_argument(
        "--main-train-source-names",
        type=str,
        default=",".join(sorted(DEFAULT_MAIN_TRAIN_SOURCES)),
        help="Sources allowed to receive train/val/IID/material-holdout splits. Other strict sources become OOD test.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--iid-test-ratio", type=float, default=0.10)
    parser.add_argument("--material-holdout-ratio", type=float, default=0.08)
    parser.add_argument(
        "--fill-deficits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, fill quota deficits with surplus materials. Default preserves quota honesty and reports blockers.",
    )
    return parser.parse_args()


def parse_csv(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def parse_quotas(value: str | None) -> dict[str, float]:
    if not value:
        return dict(DEFAULT_QUOTAS)
    quotas: dict[str, float] = {}
    for part in str(value).split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError(f"invalid_quota:{part}")
        key, raw = part.split(":", 1)
        quotas[key.strip()] = float(raw)
    total = sum(quotas.values())
    if total <= 0:
        raise ValueError("material_quotas_sum_to_zero")
    return {key: value / total for key, value in quotas.items()}


def stable_hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def object_key(record: dict[str, Any]) -> str:
    return str(
        record.get("canonical_object_id")
        or record.get("object_id")
        or record.get("source_uid")
        or record.get("source_model_path")
        or ""
    )


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or payload.get("objects") or payload.get("rows") or []
    if not isinstance(records, list):
        raise TypeError(f"manifest_missing_records:{path}")
    return payload, [record for record in records if isinstance(record, dict)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key) if record.get(key) is not None else "unknown") for record in records))


def bool_distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {
        "true": sum(bool(record.get(key)) for record in records),
        "false": sum(not bool(record.get(key)) for record in records),
    }


def numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def numeric_stats(records: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = sorted(numeric_values(records, key))
    if not values:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p90": None, "max": None}
    p50_index = min(len(values) - 1, int(round((len(values) - 1) * 0.50)))
    p90_index = min(len(values) - 1, int(round((len(values) - 1) * 0.90)))
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": values[0],
        "p50": values[p50_index],
        "p90": values[p90_index],
        "max": values[-1],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    material = distribution(records, "material_family")
    max_material_ratio = max(material.values()) / max(len(records), 1) if material else 0.0
    return {
        "records": len(records),
        "paper_split": distribution(records, "paper_split"),
        "default_split": distribution(records, "default_split"),
        "source_name": distribution(records, "source_name"),
        "generator_id": distribution(records, "generator_id"),
        "material_family": material,
        "license_bucket": distribution(records, "license_bucket"),
        "target_quality_tier": distribution(records, "target_quality_tier"),
        "target_source_type": distribution(records, "target_source_type"),
        "supervision_role": distribution(records, "supervision_role"),
        "experiment_role": distribution(records, "experiment_role"),
        "has_material_prior": bool_distribution(records, "has_material_prior"),
        "view_supervision_ready": bool_distribution(records, "view_supervision_ready"),
        "paper_stage_eligible": bool_distribution(records, "paper_stage_eligible"),
        "target_is_prior_copy": bool_distribution(records, "target_is_prior_copy"),
        "max_material_family_ratio": max_material_ratio,
        "target_prior_identity_stats": numeric_stats(records, "target_prior_identity"),
        "target_prior_distance_stats": numeric_stats(records, "target_prior_distance"),
        "target_confidence_mean_stats": numeric_stats(records, "target_confidence_mean"),
        "target_confidence_nonzero_rate_stats": numeric_stats(records, "target_confidence_nonzero_rate"),
        "target_coverage_stats": numeric_stats(records, "target_coverage"),
    }


def row_by_object_id(audit_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("object_id")): row
        for row in audit_payload.get("records", [])
        if row.get("object_id") is not None
    }


def merge_audit_fields(record: dict[str, Any], row: dict[str, Any] | None, *, source_manifest: Path) -> dict[str, Any]:
    out = dict(record)
    out["source_manifest"] = str(source_manifest.resolve())
    if row is None:
        return out
    for key in (
        "source_name",
        "supervision_role",
        "supervision_tier",
        "target_source_type",
        "target_is_prior_copy",
        "target_prior_identity",
        "target_prior_similarity",
        "target_prior_distance",
        "target_quality_tier",
        "target_confidence_summary",
        "target_confidence_mean",
        "target_confidence_nonzero_rate",
        "target_coverage",
        "paper_stage_eligible",
        "paper_license_allowed",
        "is_complete",
        "category_bucket",
        "category_label",
        "material_family",
        "thin_boundary_flag",
        "lighting_bank_id",
        "effective_view_supervision_rate",
        "strict_complete_view_rate",
        "view_supervision_ready",
        "valid_view_count",
    ):
        if key in row:
            out[key] = row[key]
    return out


def record_rank(record: dict[str, Any]) -> tuple[int, int, float, float, int, str]:
    quality_score = {"paper_strong": 4, "paper_pseudo": 3, "research_only": 2, "smoke_only": 1}.get(
        str(record.get("target_quality_tier") or "unknown"),
        0,
    )
    source_score = {"gt_render_baked": 3, "pseudo_from_multiview": 2, "pseudo_from_material_bank": 1}.get(
        str(record.get("target_source_type") or "unknown"),
        0,
    )
    confidence = float(record.get("target_confidence_mean") or 0.0)
    distance = float(record.get("target_prior_distance") or 0.0)
    eligible = int(bool(record.get("paper_stage_eligible")))
    return eligible, quality_score, confidence, distance, source_score, stable_hash_key(object_key(record))


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        key = object_key(record)
        if not key:
            continue
        current = selected.get(key)
        if current is None or record_rank(record) > record_rank(current):
            selected[key] = record
    return sorted(selected.values(), key=object_key)


def strict_blockers(record: dict[str, Any], args: argparse.Namespace, allowed_license_buckets: set[str]) -> list[str]:
    blockers: list[str] = []
    target_quality = str(record.get("target_quality_tier") or "unknown")
    target_source = str(record.get("target_source_type") or "unknown")
    license_bucket = str(record.get("license_bucket") or "unknown")
    if target_quality not in PAPER_TIERS:
        blockers.append(f"target_quality_not_paper:{target_quality}")
    if target_source in BLOCKED_TARGET_SOURCES:
        blockers.append(f"blocked_target_source:{target_source}")
    if bool(record.get("target_is_prior_copy")):
        blockers.append("target_is_prior_copy")
    if not bool(record.get("paper_stage_eligible")):
        blockers.append("paper_stage_eligible_false")
    if not bool(record.get("is_complete", True)):
        blockers.append("incomplete_paths_or_buffers")
    if not bool(record.get("paper_license_allowed", True)):
        blockers.append("paper_license_not_allowed")
    if allowed_license_buckets and license_bucket not in allowed_license_buckets:
        blockers.append(f"license_not_allowed:{license_bucket}")
    if not bool(record.get("view_supervision_ready")):
        blockers.append("view_supervision_not_ready")
    if int(record.get("valid_view_count") or 0) < int(args.min_valid_view_count):
        blockers.append(f"valid_view_count_low:{int(record.get('valid_view_count') or 0)}")
    target_prior_identity = record.get("target_prior_identity")
    if not isinstance(target_prior_identity, (int, float)):
        blockers.append("missing_target_prior_identity")
    elif float(target_prior_identity) >= float(args.identity_like_threshold):
        blockers.append(f"identity_like:{float(target_prior_identity):.4f}")
    confidence_summary = (
        record.get("target_confidence_summary") if isinstance(record.get("target_confidence_summary"), dict) else {}
    )
    confidence_mean = float(record.get("target_confidence_mean") or confidence_summary.get("mean", 0.0) or 0.0)
    nonzero_rate = float(
        record.get("target_confidence_nonzero_rate") or confidence_summary.get("nonzero_rate", 0.0) or 0.0
    )
    active_mean = confidence_active_mean(confidence_summary)
    confidence_mean_pass = confidence_mean >= float(args.min_confidence_mean) and nonzero_rate >= float(
        args.min_confidence_nonzero_rate
    )
    active_confidence_pass = (
        active_mean >= DEFAULT_MIN_TARGET_CONFIDENCE_ACTIVE_MEAN_FOR_PAPER
        and nonzero_rate >= DEFAULT_MIN_TARGET_CONFIDENCE_ACTIVE_COVERAGE_FOR_PAPER
    )
    if not (confidence_mean_pass or active_confidence_pass):
        if confidence_mean < float(args.min_confidence_mean):
            blockers.append(f"low_confidence_mean:{confidence_mean:.3f}:active_mean={active_mean:.3f}")
        if nonzero_rate < float(args.min_confidence_nonzero_rate):
            blockers.append(f"low_confidence_nonzero_rate:{nonzero_rate:.3f}")
        elif nonzero_rate < DEFAULT_MIN_TARGET_CONFIDENCE_ACTIVE_COVERAGE_FOR_PAPER:
            blockers.append(f"active_confidence_coverage_low:{nonzero_rate:.3f}")
    if float(record.get("target_coverage") or 0.0) < float(args.min_target_coverage):
        blockers.append(f"low_target_coverage:{float(record.get('target_coverage') or 0.0):.3f}")
    return blockers


def is_diagnostic_candidate(record: dict[str, Any]) -> bool:
    if str(record.get("target_source_type") or "unknown") in BLOCKED_TARGET_SOURCES:
        return False
    if bool(record.get("target_is_prior_copy")):
        return False
    if not bool(record.get("is_complete", True)):
        return False
    if not bool(record.get("view_supervision_ready")):
        return False
    return str(record.get("target_quality_tier") or "unknown") in PAPER_TIERS | {"research_only", "smoke_only"}


def priority(record: dict[str, Any]) -> tuple[int, int, int, float, float, str]:
    quality_score = {"paper_strong": 4, "paper_pseudo": 3, "research_only": 2, "smoke_only": 1}.get(
        str(record.get("target_quality_tier") or "unknown"),
        0,
    )
    source_score = {"gt_render_baked": 3, "pseudo_from_multiview": 2, "pseudo_from_material_bank": 1}.get(
        str(record.get("target_source_type") or "unknown"),
        0,
    )
    no_prior = int(not bool(record.get("has_material_prior")))
    confidence = float(record.get("target_confidence_mean") or 0.0)
    distance = float(record.get("target_prior_distance") or 0.0)
    return quality_score, source_score, no_prior, confidence, distance, stable_hash_key(object_key(record))


def select_balanced_records(
    strict_records: list[dict[str, Any]],
    *,
    quotas: dict[str, float],
    target_records: int,
    fill_deficits: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in strict_records:
        grouped[str(record.get("material_family") or "unknown")].append(record)
    for records in grouped.values():
        records.sort(key=priority, reverse=True)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    quota_targets: dict[str, int] = {
        family: int(math.floor(float(target_records) * fraction))
        for family, fraction in quotas.items()
    }
    # Give any rounding remainder to the highest-quota families deterministically.
    remainder = max(0, int(target_records) - sum(quota_targets.values()))
    for family, _fraction in sorted(quotas.items(), key=lambda item: (-item[1], item[0]))[:remainder]:
        quota_targets[family] += 1

    deficits: dict[str, int] = {}
    available: dict[str, int] = {}
    selected_by_family: dict[str, int] = {}
    for family, target in quota_targets.items():
        candidates = grouped.get(family, [])
        available[family] = len(candidates)
        take = min(target, len(candidates))
        if take < target:
            deficits[family] = target - take
        for record in candidates[:take]:
            key = object_key(record)
            selected.append(record)
            selected_keys.add(key)
        selected_by_family[family] = take

    if fill_deficits and len(selected) < target_records:
        remaining = [
            record
            for record in strict_records
            if object_key(record) not in selected_keys
        ]
        remaining.sort(key=priority, reverse=True)
        for record in remaining:
            if len(selected) >= target_records:
                break
            selected.append(record)
            selected_keys.add(object_key(record))
            family = str(record.get("material_family") or "unknown")
            selected_by_family[family] = selected_by_family.get(family, 0) + 1

    selected.sort(key=lambda record: (str(record.get("material_family") or "unknown"), object_key(record)))
    return selected, {
        "target_records": int(target_records),
        "quota_fraction": quotas,
        "quota_target_records": quota_targets,
        "available_records": available,
        "selected_records": selected_by_family,
        "quota_deficits": deficits,
        "fill_deficits": bool(fill_deficits),
    }


def take_balanced_by_material(
    records: list[dict[str, Any]],
    *,
    limit: int,
    min_per_material_family: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("material_family") or "unknown")].append(record)
    for family_records in grouped.values():
        family_records.sort(key=priority, reverse=True)
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for family in sorted(grouped):
        for record in grouped[family][: max(0, min_per_material_family)]:
            if len(selected) >= limit:
                return selected
            selected.append(record)
            selected_keys.add(object_key(record))
    remaining = [record for record in records if object_key(record) not in selected_keys]
    remaining.sort(key=priority, reverse=True)
    for record in remaining:
        if len(selected) >= limit:
            break
        selected.append(record)
        selected_keys.add(object_key(record))
    return selected


def assign_splits(
    records: list[dict[str, Any]],
    *,
    main_train_sources: set[str],
    train_ratio: float,
    val_ratio: float,
    iid_test_ratio: float,
    material_holdout_ratio: float,
) -> list[dict[str, Any]]:
    by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ood_records: list[dict[str, Any]] = []
    for record in records:
        if main_train_sources and str(record.get("source_name") or "unknown") not in main_train_sources:
            ood_records.append(record)
        else:
            by_material[str(record.get("material_family") or "unknown")].append(record)

    out: list[dict[str, Any]] = []
    for record in sorted(ood_records, key=lambda item: stable_hash_key(object_key(item))):
        updated = dict(record)
        updated["paper_split"] = "paper_test_ood_object"
        updated["default_split"] = "test"
        out.append(updated)

    for _family, family_records in sorted(by_material.items()):
        ordered = sorted(family_records, key=lambda item: stable_hash_key("stage1_v3:" + object_key(item)))
        count = len(ordered)
        material_holdout_count = int(round(count * material_holdout_ratio))
        val_count = int(round(count * val_ratio))
        iid_test_count = int(round(count * iid_test_ratio))
        # Tiny groups still need at least train coverage; do not consume all records as eval.
        eval_budget = min(max(0, count - 1), material_holdout_count + val_count + iid_test_count)
        material_holdout_count = min(material_holdout_count, eval_budget)
        eval_budget -= material_holdout_count
        val_count = min(val_count, eval_budget)
        eval_budget -= val_count
        iid_test_count = min(iid_test_count, eval_budget)
        for index, record in enumerate(ordered):
            updated = dict(record)
            if index < material_holdout_count:
                updated["paper_split"] = "paper_test_material_holdout"
                updated["default_split"] = "test"
            elif index < material_holdout_count + val_count:
                updated["paper_split"] = "paper_val_iid"
                updated["default_split"] = "val"
            elif index < material_holdout_count + val_count + iid_test_count:
                updated["paper_split"] = "paper_test_iid"
                updated["default_split"] = "test"
            else:
                updated["paper_split"] = "paper_train"
                updated["default_split"] = "train"
            out.append(updated)

    return sorted(out, key=object_key)


def make_manifest(
    *,
    records: list[dict[str, Any]],
    subset_name: str,
    experiment_role: str,
    source_manifests: list[Path],
    selection_policy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_version": "canonical_asset_record_v1_stage1_v3",
        "subset_name": subset_name,
        "experiment_role": experiment_role,
        "source_manifests": [str(path.resolve()) for path in source_manifests],
        "selection_policy": selection_policy,
        "summary": summarize(records),
        "records": records,
    }


def add_role(records: list[dict[str, Any]], *, role: str, paper_split: str | None = None, default_split: str | None = None) -> list[dict[str, Any]]:
    out = []
    for record in records:
        updated = dict(record)
        updated["experiment_role"] = role
        if paper_split is not None:
            updated["paper_split"] = paper_split
        if default_split is not None:
            updated["default_split"] = default_split
        out.append(updated)
    return out


def build_quality_blockers(
    *,
    strict_records: list[dict[str, Any]],
    balanced_records: list[dict[str, Any]],
    args: argparse.Namespace,
    quotas: dict[str, float],
) -> list[str]:
    blockers: list[str] = []
    strict_summary = summarize(strict_records)
    balanced_summary = summarize(balanced_records)
    strict_count = len(strict_records)
    balanced_count = len(balanced_records)
    if strict_count < int(args.min_paper_eligible):
        blockers.append(f"strict_paper_candidates={strict_count} below {int(args.min_paper_eligible)}")
    if balanced_count < int(args.min_paper_eligible):
        blockers.append(f"balanced_paper_records={balanced_count} below {int(args.min_paper_eligible)}")
    material_counts = balanced_summary["material_family"]
    for family in quotas:
        count = int(material_counts.get(family, 0))
        if count < int(args.min_material_family_records):
            blockers.append(f"material_family[{family}]={count} below {int(args.min_material_family_records)}")
    max_material_ratio = float(balanced_summary.get("max_material_family_ratio") or 0.0)
    if max_material_ratio > float(args.max_material_family_ratio):
        blockers.append(
            f"max_material_family_ratio={max_material_ratio:.3f} above {float(args.max_material_family_ratio):.3f}"
        )
    no_prior = int(balanced_summary["has_material_prior"].get("false", 0))
    if no_prior < int(args.min_no_prior_records):
        blockers.append(f"no_prior_records={no_prior} below {int(args.min_no_prior_records)}")
    secondary = sum(
        count
        for source, count in balanced_summary["source_name"].items()
        if source != "ABO_locked_core"
    )
    if secondary < int(args.min_secondary_source_records):
        blockers.append(f"secondary_source_records={secondary} below {int(args.min_secondary_source_records)}")
    confidence_mean = balanced_summary["target_confidence_mean_stats"].get("mean")
    if confidence_mean is None or float(confidence_mean) < float(args.target_confidence_mean):
        blockers.append(
            f"target_confidence_mean={float(confidence_mean or 0.0):.3f} below {float(args.target_confidence_mean):.3f}"
        )
    strict_max_ratio = float(strict_summary.get("max_material_family_ratio") or 0.0)
    if strict_max_ratio > float(args.max_material_family_ratio):
        blockers.append(
            f"strict_candidates_max_material_family_ratio={strict_max_ratio:.3f} above {float(args.max_material_family_ratio):.3f}"
        )
    return blockers


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stage1-v3 Dataset Build Report",
        "",
        f"- recommendation: `{report['recommendation']}`",
        f"- stage1_v3_ready: `{report['stage1_v3_ready']}`",
        f"- strict_candidates: `{report['subset_summaries']['strict_paper_candidates']['records']}`",
        f"- balanced_paper: `{report['subset_summaries']['balanced_paper']['records']}`",
        f"- diagnostic: `{report['subset_summaries']['diagnostic']['records']}`",
        f"- ood_eval: `{report['subset_summaries']['ood_eval']['records']}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in report.get("blockers") or ["none"]:
        lines.append(f"- {blocker}")
    lines.extend(["", "## Material Quota", ""])
    lines.append(f"- quota: `{json.dumps(report['selection_policy']['material_quotas'], ensure_ascii=False)}`")
    lines.append(f"- selected: `{json.dumps(report['balanced_selection']['selected_records'], ensure_ascii=False)}`")
    lines.append(f"- deficits: `{json.dumps(report['balanced_selection']['quota_deficits'], ensure_ascii=False)}`")
    lines.extend(["", "## Subsets", ""])
    for name, summary in report["subset_summaries"].items():
        lines.append(f"### {name}")
        for key in (
            "records",
            "paper_split",
            "source_name",
            "material_family",
            "license_bucket",
            "target_quality_tier",
            "target_source_type",
            "has_material_prior",
            "target_confidence_mean_stats",
        ):
            lines.append(f"- {key}: `{json.dumps(summary.get(key), ensure_ascii=False)}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report_html(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for name, summary in report["subset_summaries"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{summary['records']}</td>"
            f"<td><code>{html.escape(json.dumps(summary.get('material_family', {}), ensure_ascii=False))}</code></td>"
            f"<td><code>{html.escape(json.dumps(summary.get('source_name', {}), ensure_ascii=False))}</code></td>"
            f"<td><code>{html.escape(json.dumps(summary.get('target_quality_tier', {}), ensure_ascii=False))}</code></td>"
            f"<td><code>{html.escape(json.dumps(summary.get('paper_split', {}), ensure_ascii=False))}</code></td>"
            "</tr>"
        )
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Stage1-v3 Dataset Build Report</title>",
        "<style>body{font-family:Arial,sans-serif;background:#111820;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1380px;margin:auto}.card{background:#1a2430;border-radius:18px;padding:18px;margin:16px 0}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #344256;padding:8px;text-align:left;vertical-align:top}code{color:#b9e6ff;white-space:pre-wrap}</style>",
        "</head><body><div class='wrap'>",
        "<h1>Stage1-v3 Dataset Build Report</h1>",
        "<div class='card'>",
        f"<p>recommendation: <code>{html.escape(str(report['recommendation']))}</code></p>",
        f"<p>stage1_v3_ready: <code>{html.escape(str(report['stage1_v3_ready']))}</code></p>",
        f"<p>blockers: <code>{html.escape(json.dumps(report.get('blockers', []), ensure_ascii=False))}</code></p>",
        "</div>",
        "<div class='card'><table><thead><tr><th>Subset</th><th>Records</th><th>Material</th><th>Source</th><th>Target quality</th><th>Split</th></tr></thead><tbody>",
        *rows,
        "</tbody></table></div>",
        "</div></body></html>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    allowed_license_buckets = parse_csv(args.paper_license_buckets)
    main_train_sources = parse_csv(args.main_train_source_names)
    quotas = parse_quotas(args.material_quotas)

    source_audits: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for manifest in args.manifest:
        payload, records = load_manifest(manifest)
        audit_payload = audit_manifest(
            manifest,
            max_records=-1,
            identity_warning_threshold=float(args.identity_like_threshold),
            max_target_prior_identity_rate_for_paper=0.30,
            min_nontrivial_target_count_for_paper=128,
            allowed_paper_license_buckets=allowed_license_buckets,
        )
        rows = row_by_object_id(audit_payload)
        for record in records:
            all_records.append(merge_audit_fields(record, rows.get(object_key(record)), source_manifest=manifest))
        source_audits[str(manifest.resolve())] = audit_payload.get("summary", {})

    unique_records = dedupe_records(all_records)

    strict_records: list[dict[str, Any]] = []
    auxiliary_records: list[dict[str, Any]] = []
    reject_records: list[dict[str, Any]] = []
    blocker_counts: Counter[str] = Counter()
    blocker_counts_by_material: dict[str, Counter[str]] = defaultdict(Counter)
    for record in unique_records:
        blockers = strict_blockers(record, args, allowed_license_buckets)
        updated = dict(record)
        updated["stage1_v3_gate_blockers"] = blockers
        if not blockers:
            updated["experiment_role"] = "paper_stage"
            strict_records.append(updated)
            continue
        for blocker in blockers:
            key = blocker.split(":", 1)[0]
            blocker_counts[key] += 1
            blocker_counts_by_material[str(record.get("material_family") or "unknown")][key] += 1
        if is_diagnostic_candidate(updated):
            updated["experiment_role"] = "auxiliary_upgrade_queue"
            updated["paper_split"] = "auxiliary_upgrade_queue"
            updated["default_split"] = "test"
            auxiliary_records.append(updated)
        else:
            updated["experiment_role"] = "rejected"
            updated["paper_split"] = "rejected"
            updated["default_split"] = "excluded"
            reject_records.append(updated)

    strict_records.sort(key=priority, reverse=True)
    balanced_records, balanced_selection = select_balanced_records(
        strict_records,
        quotas=quotas,
        target_records=int(args.target_records),
        fill_deficits=bool(args.fill_deficits),
    )
    balanced_records = assign_splits(
        add_role(balanced_records, role="paper_stage"),
        main_train_sources=main_train_sources,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        iid_test_ratio=float(args.iid_test_ratio),
        material_holdout_ratio=float(args.material_holdout_ratio),
    )

    strict_candidates_records = assign_splits(
        add_role(strict_records, role="paper_stage"),
        main_train_sources=main_train_sources,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        iid_test_ratio=float(args.iid_test_ratio),
        material_holdout_ratio=float(args.material_holdout_ratio),
    )

    diagnostic_candidates = [
        dict(record)
        for record in unique_records
        if is_diagnostic_candidate(record)
    ]
    diagnostic_records = take_balanced_by_material(
        diagnostic_candidates,
        limit=max(0, int(args.max_diagnostic_records)),
        min_per_material_family=max(0, int(args.diagnostic_min_per_material_family)),
    )
    diagnostic_records = add_role(
        diagnostic_records,
        role="diagnostic_only",
        paper_split="diagnostic_only",
        default_split="test",
    )

    ood_candidates = [
        record
        for record in diagnostic_candidates
        if str(record.get("source_name") or "unknown") != "ABO_locked_core"
        or str(record.get("generator_id") or "unknown") != "abo_locked_core"
    ]
    ood_records = take_balanced_by_material(
        ood_candidates,
        limit=max(0, int(args.max_ood_records)),
        min_per_material_family=max(0, int(args.ood_min_per_material_family)),
    )
    ood_records = add_role(
        ood_records,
        role="ood_eval_only",
        paper_split="paper_test_ood_object",
        default_split="test",
    )

    selection_policy = {
        "paper_tiers": sorted(PAPER_TIERS),
        "blocked_target_sources": sorted(BLOCKED_TARGET_SOURCES),
        "allowed_license_buckets": sorted(allowed_license_buckets),
        "material_quotas": quotas,
        "target_records": int(args.target_records),
        "min_confidence_mean": float(args.min_confidence_mean),
        "target_confidence_mean": float(args.target_confidence_mean),
        "min_confidence_nonzero_rate": float(args.min_confidence_nonzero_rate),
        "min_target_coverage": float(args.min_target_coverage),
        "identity_like_threshold": float(args.identity_like_threshold),
        "main_train_source_names": sorted(main_train_sources),
        "fill_deficits": bool(args.fill_deficits),
    }
    source_manifests = [path for path in args.manifest]

    strict_manifest = make_manifest(
        records=strict_candidates_records,
        subset_name="stage1_v3_strict_paper_candidates",
        experiment_role="paper_stage_candidates",
        source_manifests=source_manifests,
        selection_policy=selection_policy,
    )
    balanced_manifest = make_manifest(
        records=balanced_records,
        subset_name="stage1_v3_balanced_paper",
        experiment_role="paper_stage_balanced",
        source_manifests=source_manifests,
        selection_policy={**selection_policy, "balanced_selection": balanced_selection},
    )
    diagnostic_manifest = make_manifest(
        records=diagnostic_records,
        subset_name="stage1_v3_diagnostic",
        experiment_role="diagnostic_only",
        source_manifests=source_manifests,
        selection_policy=selection_policy,
    )
    ood_manifest = make_manifest(
        records=ood_records,
        subset_name="stage1_v3_ood_eval",
        experiment_role="ood_eval_only",
        source_manifests=source_manifests,
        selection_policy=selection_policy,
    )
    auxiliary_manifest = make_manifest(
        records=auxiliary_records,
        subset_name="stage1_v3_auxiliary_upgrade_queue",
        experiment_role="auxiliary_upgrade_queue",
        source_manifests=source_manifests,
        selection_policy=selection_policy,
    )
    rejects_manifest = make_manifest(
        records=reject_records,
        subset_name="stage1_v3_rejects",
        experiment_role="rejected",
        source_manifests=source_manifests,
        selection_policy=selection_policy,
    )

    blockers = build_quality_blockers(
        strict_records=strict_candidates_records,
        balanced_records=balanced_records,
        args=args,
        quotas=quotas,
    )
    stage1_v3_ready = not blockers
    recommendation = "STAGE1_V3_BALANCED_READY_FOR_DATA_HANDOFF" if stage1_v3_ready else "KEEP_AS_DATA_CANDIDATE_ONLY"

    paths = {
        "strict_paper_candidates": args.output_root / "stage1_v3_strict_paper_candidates.json",
        "balanced_paper": args.output_root / "stage1_v3_balanced_paper_manifest.json",
        "diagnostic": args.output_root / "stage1_v3_diagnostic_manifest.json",
        "ood_eval": args.output_root / "stage1_v3_ood_eval_manifest.json",
        "auxiliary_upgrade_queue": args.output_root / "stage1_v3_auxiliary_upgrade_queue.json",
        "rejects": args.output_root / "stage1_v3_rejects.json",
    }
    write_json(paths["strict_paper_candidates"], strict_manifest)
    write_json(paths["balanced_paper"], balanced_manifest)
    write_json(paths["diagnostic"], diagnostic_manifest)
    write_json(paths["ood_eval"], ood_manifest)
    write_json(paths["auxiliary_upgrade_queue"], auxiliary_manifest)
    write_json(paths["rejects"], rejects_manifest)

    report = {
        "source_manifests": [str(path.resolve()) for path in source_manifests],
        "source_audit_summaries": source_audits,
        "unique_records": len(unique_records),
        "stage1_v3_ready": stage1_v3_ready,
        "recommendation": recommendation,
        "blockers": blockers,
        "strict_gate_blocker_counts": dict(blocker_counts),
        "strict_gate_blocker_counts_by_material": {
            key: dict(value)
            for key, value in sorted(blocker_counts_by_material.items())
        },
        "balanced_selection": balanced_selection,
        "selection_policy": selection_policy,
        "subset_paths": {key: str(path.resolve()) for key, path in paths.items()},
        "subset_summaries": {
            "strict_paper_candidates": strict_manifest["summary"],
            "balanced_paper": balanced_manifest["summary"],
            "diagnostic": diagnostic_manifest["summary"],
            "ood_eval": ood_manifest["summary"],
            "auxiliary_upgrade_queue": auxiliary_manifest["summary"],
            "rejects": rejects_manifest["summary"],
        },
    }
    write_json(args.output_root / "stage1_v3_dataset_audit.json", report)
    write_report_md(args.output_root / "stage1_v3_dataset_audit.md", report)
    write_report_html(args.output_root / "stage1_v3_dataset_audit.html", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
