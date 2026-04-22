from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.manifest_quality import (
    DEFAULT_MAX_TARGET_PRIOR_IDENTITY_RATE_FOR_PAPER,
    DEFAULT_MIN_NONTRIVIAL_TARGET_COUNT_FOR_PAPER,
    audit_manifest,
)


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed paper-stage split and high-quality stage1 subset from a canonical material-refine manifest.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--max-target-prior-identity-rate-for-paper",
        type=float,
        default=DEFAULT_MAX_TARGET_PRIOR_IDENTITY_RATE_FOR_PAPER,
    )
    parser.add_argument(
        "--min-nontrivial-target-count-for-paper",
        type=int,
        default=DEFAULT_MIN_NONTRIVIAL_TARGET_COUNT_FOR_PAPER,
    )
    parser.add_argument("--paper-license-buckets", type=str, default=None)
    parser.add_argument(
        "--main-train-source-names",
        type=str,
        default="ABO_locked_core,3D-FUTURE_candidate,Objaverse-XL_strict_filtered_increment,Objaverse-XL_filtered_candidate",
    )
    parser.add_argument("--ood-source-names", type=str, default="")
    parser.add_argument("--real-lighting-source-names", type=str, default="OLATverse,OpenIllumination,ICTPolarReal")
    parser.add_argument("--promote-eligible-auxiliary", type=str, default="true")
    parser.add_argument(
        "--required-material-families",
        type=str,
        default="metal_dominant,glass_metal,mixed_thin_boundary,glossy_non_metal",
    )
    parser.add_argument("--min-no-prior-records", type=int, default=16)
    parser.add_argument("--min-secondary-generator-records", type=int, default=16)
    parser.add_argument("--min-paper-strong-records", type=int, default=16)
    parser.add_argument("--material-sensitive-holdout-fraction", type=float, default=0.10)
    parser.add_argument("--max-material-sensitive-holdout", type=int, default=64)
    return parser.parse_args()


def stable_hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key, "unknown")) for record in records))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def enrich_records_from_audit(
    original_records: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    *,
    promote_eligible_auxiliary: bool = False,
    main_train_sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    row_by_object = {str(row["object_id"]): row for row in audit_rows}
    enriched = []
    for record in original_records:
        row = row_by_object.get(str(record.get("object_id")))
        merged = dict(record)
        if row is not None:
            promoted_auxiliary = False
            paper_stage_eligible = bool(row["paper_stage_eligible"])
            if (
                promote_eligible_auxiliary
                and not paper_stage_eligible
                and str(row.get("supervision_role")) == "auxiliary_upgrade_queue"
                and (not main_train_sources or str(row.get("source_name")) in main_train_sources)
                and str(row.get("target_quality_tier")) in {"paper_strong", "paper_pseudo"}
                and not bool(row.get("target_is_prior_copy"))
                and bool(row.get("is_complete"))
                and bool(row.get("paper_license_allowed"))
            ):
                paper_stage_eligible = True
                promoted_auxiliary = True
            merged.update(
                {
                    "target_source_type": row["target_source_type"],
                    "target_is_prior_copy": row["target_is_prior_copy"],
                    "target_prior_similarity": row.get("target_prior_similarity"),
                    "target_prior_distance": row.get("target_prior_distance"),
                    "target_quality_tier": row["target_quality_tier"],
                    "target_confidence_summary": row["target_confidence_summary"],
                    "paper_stage_eligible": paper_stage_eligible,
                    "paper_role_promotion": "auxiliary_upgrade_queue_to_stage1" if promoted_auxiliary else "none",
                    "category_bucket": row["category_bucket"],
                    "category_label": row["category_label"],
                    "material_family": row["material_family"],
                    "thin_boundary_flag": row["thin_boundary_flag"],
                    "lighting_bank_id": row["lighting_bank_id"],
                    "effective_view_supervision_rate": row["effective_view_supervision_rate"],
                }
            )
        enriched.append(merged)
    return enriched


def choose_material_sensitive_holdout(
    records: list[dict[str, Any]],
    *,
    fraction: float,
    limit: int,
    min_remaining_records: int = 3,
) -> set[str]:
    if not records:
        return set()
    available_for_holdout = max(0, len(records) - max(int(min_remaining_records), 0))
    if available_for_holdout <= 0:
        return set()
    effective_limit = min(max(int(limit), 0), available_for_holdout)
    if effective_limit <= 0:
        return set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        bucket = str(record.get("category_bucket") or "unknown")
        grouped[bucket].append(record)
    selected: set[str] = set()
    for bucket, items in grouped.items():
        items = sorted(items, key=lambda item: stable_hash_key(str(item["object_id"])))
        take = max(1, int(round(len(items) * fraction)))
        for item in items[:take]:
            selected.add(str(item["object_id"]))
    if len(selected) > effective_limit:
        selected = set(sorted(selected, key=stable_hash_key)[:effective_limit])
    return selected


def assign_stage1_split(
    record: dict[str, Any],
    *,
    main_train_sources: set[str],
    ood_sources: set[str],
    real_lighting_sources: set[str],
    material_sensitive_holdout_ids: set[str],
) -> str:
    object_id = str(record["object_id"])
    source_name = str(record.get("source_name", "unknown"))
    if not bool(record.get("paper_stage_eligible")):
        return "excluded"
    if source_name in real_lighting_sources:
        return "paper_test_real_lighting"
    if object_id in material_sensitive_holdout_ids:
        return "paper_test_material_holdout"
    if source_name in ood_sources:
        return "paper_test_ood_object"
    if main_train_sources and source_name not in main_train_sources:
        return "excluded"
    default_split = str(record.get("default_split", "train"))
    if default_split == "val":
        return "paper_val_iid"
    if default_split == "test":
        return "paper_test_iid"
    return "paper_train"


def build_split_payload(
    records: list[dict[str, Any]],
    *,
    main_train_sources: set[str],
    ood_sources: set[str],
    real_lighting_sources: set[str],
    material_sensitive_holdout_fraction: float,
    max_material_sensitive_holdout: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    trainable_candidates = [
        record
        for record in records
        if bool(record.get("paper_stage_eligible"))
        and str(record.get("source_name", "unknown")) in (main_train_sources or {str(record.get("source_name", "unknown"))})
    ]
    material_sensitive_holdout_ids = choose_material_sensitive_holdout(
        trainable_candidates,
        fraction=material_sensitive_holdout_fraction,
        limit=max_material_sensitive_holdout,
        min_remaining_records=3,
    )

    split_records = []
    stage1_subset_records = []
    ood_subset_records = []
    for record in records:
        split_name = assign_stage1_split(
            record,
            main_train_sources=main_train_sources,
            ood_sources=ood_sources,
            real_lighting_sources=real_lighting_sources,
            material_sensitive_holdout_ids=material_sensitive_holdout_ids,
        )
        split_record = dict(record)
        split_record["paper_split"] = split_name
        split_records.append(split_record)
        if split_name in {
            "paper_train",
            "paper_val_iid",
            "paper_test_iid",
            "paper_test_material_holdout",
            "paper_test_real_lighting",
        }:
            subset_record = dict(split_record)
            if split_name == "paper_train":
                subset_record["default_split"] = "train"
            elif split_name == "paper_val_iid":
                subset_record["default_split"] = "val"
            else:
                subset_record["default_split"] = "test"
            stage1_subset_records.append(subset_record)
        elif split_name == "paper_test_ood_object":
            ood_record = dict(split_record)
            ood_record["default_split"] = "test"
            ood_subset_records.append(ood_record)

    split_payload = {
        "split_version": "paper_stage1_split_v1",
        "counts": distribution(split_records, "paper_split"),
        "records": [
            {
                "object_id": record["object_id"],
                "source_name": record.get("source_name", "unknown"),
                "generator_id": record.get("generator_id", "unknown"),
                "paper_split": record["paper_split"],
                "default_split": record.get("default_split", "train"),
                "paper_stage_eligible": bool(record.get("paper_stage_eligible")),
                "target_quality_tier": record.get("target_quality_tier", "unknown"),
                "category_bucket": record.get("category_bucket", "unknown"),
                "license_bucket": record.get("license_bucket", "unknown"),
            }
            for record in sorted(split_records, key=lambda item: stable_hash_key(str(item["object_id"])))
        ],
    }
    return split_payload, stage1_subset_records, ood_subset_records


def split_count(split_counts: dict[str, int], split_name: str) -> int:
    return int(split_counts.get(split_name, 0) or 0)


def split_readiness_blockers(split_counts: dict[str, int], *, stage1_subset_records: int) -> list[str]:
    blockers: list[str] = []
    if stage1_subset_records <= 0:
        blockers.append("stage1_subset_records=0 after eligibility and split filtering")
        return blockers
    if split_count(split_counts, "paper_train") <= 0:
        blockers.append("paper_train_records=0")
    if split_count(split_counts, "paper_val_iid") <= 0:
        blockers.append("paper_val_iid_records=0")
    iid_test = split_count(split_counts, "paper_test_iid")
    material_test = split_count(split_counts, "paper_test_material_holdout")
    real_test = split_count(split_counts, "paper_test_real_lighting")
    if iid_test + material_test + real_test <= 0:
        blockers.append("paper_test_records=0")
    return blockers


def distribution_readiness(
    records: list[dict[str, Any]],
    *,
    required_material_families: set[str],
    min_no_prior_records: int,
    min_secondary_generator_records: int,
    min_paper_strong_records: int,
) -> dict[str, Any]:
    train_records = [record for record in records if str(record.get("paper_split")) == "paper_train"]
    generator_counts = Counter(str(record.get("generator_id", "unknown")) for record in train_records)
    non_primary_generators = sum(
        count for generator, count in generator_counts.items() if generator not in {"abo_locked_core", "abo_locked_core_v1"}
    )
    material_counts = Counter(str(record.get("material_family", "unknown")) for record in train_records)
    no_prior_records = sum(not bool(record.get("has_material_prior")) for record in train_records)
    paper_strong_records = sum(str(record.get("target_quality_tier")) == "paper_strong" for record in train_records)
    warnings = []
    for family in sorted(required_material_families):
        if material_counts.get(family, 0) <= 0:
            warnings.append(f"missing_material_family:{family}")
    if no_prior_records < int(min_no_prior_records):
        warnings.append(f"no_prior_records={no_prior_records} below {int(min_no_prior_records)}")
    if non_primary_generators < int(min_secondary_generator_records):
        warnings.append(
            f"secondary_generator_records={non_primary_generators} below {int(min_secondary_generator_records)}"
        )
    if paper_strong_records < int(min_paper_strong_records):
        warnings.append(f"paper_strong_records={paper_strong_records} below {int(min_paper_strong_records)}")
    return {
        "train_records": len(train_records),
        "material_family": dict(material_counts),
        "generator_id": dict(generator_counts),
        "no_prior_records": no_prior_records,
        "secondary_generator_records": non_primary_generators,
        "paper_strong_records": paper_strong_records,
        "warnings": warnings,
    }


def summarize_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name in sorted({str(record.get("paper_split", "unknown")) for record in records}):
        split_records = [record for record in records if str(record.get("paper_split")) == split_name]
        summary[split_name] = {
            "records": len(split_records),
            "source_name": distribution(split_records, "source_name"),
            "generator_id": distribution(split_records, "generator_id"),
            "has_material_prior": {
                "true": sum(bool(record.get("has_material_prior")) for record in split_records),
                "false": sum(not bool(record.get("has_material_prior")) for record in split_records),
            },
            "supervision_tier": distribution(split_records, "supervision_tier"),
            "category_bucket": distribution(split_records, "category_bucket"),
            "license_bucket": distribution(split_records, "license_bucket"),
            "target_quality_tier": distribution(split_records, "target_quality_tier"),
        }
    return summary


def save_split_html(
    summary: dict[str, Any],
    readiness_summary: dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Material Refine Paper Split Audit</title>",
        "<style>body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1320px;margin:auto}.card{background:#18202b;border-radius:18px;padding:18px;margin:16px 0}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #334052;padding:8px;text-align:left}code{color:#b9e6ff}</style>",
        "</head><body><div class='wrap'>",
        "<h1>Material Refine Paper Split Audit</h1>",
        "<div class='card'>",
        f"<p>paper_stage_ready: <code>{readiness_summary['paper_stage_ready']}</code></p>",
        f"<p>stage1_subset_records: <code>{readiness_summary['stage1_subset_records']}</code> | ood_eval_records: <code>{readiness_summary['ood_eval_records']}</code></p>",
        f"<p>round2_recommendation: <code>{html.escape(readiness_summary['round2_recommendation'])}</code></p>",
        f"<p>distribution_warnings: <code>{html.escape(json.dumps(readiness_summary.get('distribution_warnings', []), ensure_ascii=False))}</code></p>",
        "</div>",
    ]
    for split_name, bucket in summary.items():
        lines.extend(
            [
                "<div class='card'>",
                f"<h2>{html.escape(split_name)}</h2>",
                f"<p>records: <code>{bucket['records']}</code></p>",
                "<table><thead><tr><th>Axis</th><th>Distribution</th></tr></thead><tbody>",
            ]
        )
        for axis in (
            "source_name",
            "generator_id",
            "has_material_prior",
            "supervision_tier",
            "category_bucket",
            "license_bucket",
            "target_quality_tier",
        ):
            lines.append(
                f"<tr><td>{html.escape(axis)}</td><td><code>{html.escape(json.dumps(bucket[axis], ensure_ascii=False))}</code></td></tr>"
            )
        lines.extend(["</tbody></table>", "</div>"])
    lines.extend(["</div></body></html>"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_readiness_md(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Material Refine Paper-Stage Readiness Summary",
        "",
        f"- manifest: `{summary['manifest']}`",
        f"- paper_stage_ready: `{summary['paper_stage_ready']}`",
        f"- paper_stage_eligible_records: `{summary['paper_stage_eligible_records']}`",
        f"- nontrivial_target_records: `{summary['nontrivial_target_records']}`",
        f"- target_prior_identity_rate: `{summary['target_prior_identity_rate']:.4f}`",
        f"- target_prior_similarity_mean: `{summary.get('target_prior_similarity_mean', 0.0):.4f}`",
        f"- target_prior_distance_mean: `{summary.get('target_prior_distance_mean', 0.0):.4f}`",
        f"- effective_view_supervision_rate: `{summary['effective_view_supervision_rate']:.4f}`",
        f"- stage1_subset_records: `{summary['stage1_subset_records']}`",
        f"- ood_eval_records: `{summary['ood_eval_records']}`",
        f"- split_counts: `{json.dumps(summary['split_counts'], ensure_ascii=False)}`",
        f"- round2_recommendation: `{summary['round2_recommendation']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = summary.get("readiness_blockers") or ["none"]
    for blocker in blockers:
        lines.append(f"- {blocker}")
    lines.extend(["", "## Distribution Warnings", ""])
    for warning in summary.get("distribution_warnings") or ["none"]:
        lines.append(f"- {warning}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_payload = json.loads(args.manifest.read_text())
    original_records = manifest_payload.get("records") or []
    if not isinstance(original_records, list):
        raise TypeError(f"unsupported_manifest_records:{args.manifest}")

    allowed_license_buckets = set(parse_csv_list(args.paper_license_buckets)) or None
    audit_payload = audit_manifest(
        args.manifest,
        max_records=-1,
        identity_warning_threshold=args.max_target_prior_identity_rate_for_paper,
        max_target_prior_identity_rate_for_paper=args.max_target_prior_identity_rate_for_paper,
        min_nontrivial_target_count_for_paper=args.min_nontrivial_target_count_for_paper,
        allowed_paper_license_buckets=allowed_license_buckets,
    )
    main_train_sources = set(parse_csv_list(args.main_train_source_names))
    ood_sources = set(parse_csv_list(args.ood_source_names))
    real_lighting_sources = set(parse_csv_list(args.real_lighting_source_names))
    enriched_records = enrich_records_from_audit(
        original_records,
        audit_payload["records"],
        promote_eligible_auxiliary=parse_bool(args.promote_eligible_auxiliary),
        main_train_sources=main_train_sources,
    )

    split_payload, stage1_subset_records, ood_subset_records = build_split_payload(
        enriched_records,
        main_train_sources=main_train_sources,
        ood_sources=ood_sources,
        real_lighting_sources=real_lighting_sources,
        material_sensitive_holdout_fraction=max(0.0, min(1.0, float(args.material_sensitive_holdout_fraction))),
        max_material_sensitive_holdout=max(int(args.max_material_sensitive_holdout), 0),
    )
    split_by_object = {
        str(record["object_id"]): str(record["paper_split"])
        for record in split_payload["records"]
    }
    split_records = [
        dict(record, paper_split=split_by_object.get(str(record["object_id"]), "excluded"))
        for record in enriched_records
    ]
    split_summary = summarize_split(split_records)
    distribution_summary = distribution_readiness(
        split_records,
        required_material_families=set(parse_csv_list(args.required_material_families)),
        min_no_prior_records=max(int(args.min_no_prior_records), 0),
        min_secondary_generator_records=max(int(args.min_secondary_generator_records), 0),
        min_paper_strong_records=max(int(args.min_paper_strong_records), 0),
    )

    enriched_manifest = dict(manifest_payload)
    enriched_manifest["records"] = enriched_records
    enriched_manifest["manifest_version"] = "canonical_asset_record_v1_enriched"
    enriched_manifest["paper_stage_quality_schema"] = {
        "target_source_type": [
            "gt_render_baked",
            "pseudo_from_multiview",
            "pseudo_from_material_bank",
            "copied_from_prior",
            "unknown",
        ],
        "target_quality_tier": [
            "paper_strong",
            "paper_pseudo",
            "research_only",
            "smoke_only",
            "unknown",
        ],
    }

    stage1_manifest = {
        "manifest_version": "canonical_asset_record_v1_stage1_subset",
        "source_manifest": str(args.manifest.resolve()),
        "summary": {
            "records": len(stage1_subset_records),
            "source_name": distribution(stage1_subset_records, "source_name"),
            "generator_id": distribution(stage1_subset_records, "generator_id"),
            "license_bucket": distribution(stage1_subset_records, "license_bucket"),
            "target_quality_tier": distribution(stage1_subset_records, "target_quality_tier"),
            "target_source_type": distribution(stage1_subset_records, "target_source_type"),
            "material_family": distribution(stage1_subset_records, "material_family"),
            "paper_role_promotion": distribution(stage1_subset_records, "paper_role_promotion"),
        },
        "records": stage1_subset_records,
    }
    ood_manifest = {
        "manifest_version": "canonical_asset_record_v1_stage1_ood_subset",
        "source_manifest": str(args.manifest.resolve()),
        "summary": {
            "records": len(ood_subset_records),
            "source_name": distribution(ood_subset_records, "source_name"),
            "generator_id": distribution(ood_subset_records, "generator_id"),
        },
        "records": ood_subset_records,
    }
    split_summary_payload = {
        "manifest": str(args.manifest.resolve()),
        "split_counts": split_payload["counts"],
        "split_distribution": split_summary,
        "distribution_readiness": distribution_summary,
        "material_sensitive_eval_count": int(split_payload["counts"].get("paper_test_material_holdout", 0)),
        "ood_eval_count": int(split_payload["counts"].get("paper_test_ood_object", 0)),
        "real_lighting_eval_count": int(split_payload["counts"].get("paper_test_real_lighting", 0)),
    }
    split_blockers = split_readiness_blockers(
        split_payload["counts"],
        stage1_subset_records=len(stage1_subset_records),
    )
    readiness_blockers = list(audit_payload["summary"]["readiness_blockers"]) + split_blockers
    paper_stage_ready = (
        bool(audit_payload["summary"]["paper_stage_ready"])
        and bool(stage1_subset_records)
        and not split_blockers
    )
    readiness_summary = {
        "manifest": str(args.manifest.resolve()),
        "paper_stage_ready": paper_stage_ready,
        "paper_stage_eligible_records": int(audit_payload["summary"]["paper_stage_eligible_records"]),
        "nontrivial_target_records": int(audit_payload["summary"]["nontrivial_target_records"]),
        "target_prior_identity_rate": float(audit_payload["summary"]["target_prior_identity_rate"]),
        "target_prior_similarity_mean": float(audit_payload["summary"].get("target_prior_similarity_mean", 0.0)),
        "target_prior_distance_mean": float(audit_payload["summary"].get("target_prior_distance_mean", 0.0)),
        "effective_view_supervision_rate": float(audit_payload["summary"]["effective_view_supervision_record_rate"]),
        "stage1_subset_records": len(stage1_subset_records),
        "ood_eval_records": len(ood_subset_records),
        "split_counts": split_payload["counts"],
        "split_distribution": split_summary,
        "source_counts": stage1_manifest["summary"]["source_name"],
        "generator_counts": stage1_manifest["summary"]["generator_id"],
        "prior_counts": {
            "true": sum(bool(record.get("has_material_prior")) for record in stage1_subset_records),
            "false": sum(not bool(record.get("has_material_prior")) for record in stage1_subset_records),
        },
        "license_bucket_counts": stage1_manifest["summary"]["license_bucket"],
        "supervision_tier_counts": distribution(stage1_subset_records, "supervision_tier"),
        "target_source_type_counts": stage1_manifest["summary"]["target_source_type"],
        "target_quality_tier_counts": stage1_manifest["summary"]["target_quality_tier"],
        "material_family_counts": stage1_manifest["summary"]["material_family"],
        "paper_role_promotion_counts": stage1_manifest["summary"]["paper_role_promotion"],
        "distribution_readiness": distribution_summary,
        "distribution_warnings": distribution_summary["warnings"],
        "readiness_blockers": readiness_blockers,
        "round2_recommendation": (
            "DO_NOT_START_FULL_TRAINING"
            if not paper_stage_ready
            else "START_CONFIG_material_refine_train_paper_stage1_yaml"
        ),
    }

    enriched_manifest_path = args.output_root / "paper_stage1_enriched_manifest.json"
    stage1_manifest_path = args.output_root / "paper_stage1_subset_manifest.json"
    ood_manifest_path = args.output_root / "paper_stage1_ood_manifest.json"
    split_path = args.output_root / "paper_stage1_split_v1.json"
    split_summary_path = args.output_root / "paper_stage1_split_audit_summary.json"
    split_html_path = args.output_root / "paper_stage1_split_audit.html"
    readiness_json_path = args.output_root / "paper_stage1_readiness_summary.json"
    readiness_md_path = args.output_root / "paper_stage1_readiness_summary.md"

    write_json(enriched_manifest_path, enriched_manifest)
    write_json(stage1_manifest_path, stage1_manifest)
    write_json(ood_manifest_path, ood_manifest)
    write_json(split_path, split_payload)
    write_json(split_summary_path, split_summary_payload)
    save_split_html(split_summary, readiness_summary, split_html_path)
    write_json(readiness_json_path, readiness_summary)
    save_readiness_md(readiness_summary, readiness_md_path)

    print(
        json.dumps(
            {
                "enriched_manifest": str(enriched_manifest_path),
                "stage1_subset_manifest": str(stage1_manifest_path),
                "ood_manifest": str(ood_manifest_path),
                "split_file": str(split_path),
                "split_audit": str(split_html_path),
                "readiness_summary": str(readiness_json_path),
                "paper_stage_ready": readiness_summary["paper_stage_ready"],
                "stage1_subset_records": len(stage1_subset_records),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
