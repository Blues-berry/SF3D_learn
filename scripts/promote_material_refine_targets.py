#!/usr/bin/env python3
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
    DEFAULT_MIN_TARGET_CONFIDENCE_ACTIVE_COVERAGE_FOR_PAPER,
    DEFAULT_MIN_TARGET_CONFIDENCE_ACTIVE_MEAN_FOR_PAPER,
    audit_record,
    confidence_active_mean,
    summarize_audit_rows,
)


PROMOTION_VERSION = "material_refine_target_promotion_v1"
PAPER_TIERS = {"paper_strong", "paper_pseudo"}
BLOCKED_TARGET_SOURCES = {"copied_from_prior", "unknown", "real_benchmark_no_uv_target"}
DEFAULT_ALLOWED_LICENSE_BUCKETS = {
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
DEFAULT_PROMOTABLE_SOURCE_NAMES = {
    "ABO_locked_core",
    "3D-FUTURE_highlight_local_8k",
    "Objaverse-XL_strict_filtered_increment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run second-pass material-refine target validation and promote "
            "eligible smoke/auxiliary records to paper_pseudo or paper_strong."
        ),
    )
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--report-html", type=Path)
    parser.add_argument(
        "--allowed-license-buckets",
        type=str,
        default=",".join(sorted(DEFAULT_ALLOWED_LICENSE_BUCKETS)),
        help="Comma-separated license buckets allowed to become paper_main in this pass.",
    )
    parser.add_argument(
        "--promotable-source-names",
        type=str,
        default=",".join(sorted(DEFAULT_PROMOTABLE_SOURCE_NAMES)),
        help="Comma-separated source_name allowlist. Empty means any source can be considered.",
    )
    parser.add_argument("--min-confidence-mean", type=float, default=0.70)
    parser.add_argument("--min-confidence-nonzero-rate", type=float, default=0.50)
    parser.add_argument("--min-target-coverage", type=float, default=0.50)
    parser.add_argument("--max-target-prior-identity", type=float, default=0.95)
    parser.add_argument("--min-valid-view-count", type=int, default=1)
    parser.add_argument("--min-strict-complete-view-rate", type=float, default=0.80)
    parser.add_argument(
        "--promote-auxiliary-to-paper-main",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Temporarily evaluate auxiliary records as paper_main candidates, then write paper_main only if all gates pass.",
    )
    parser.add_argument(
        "--keep-unpromoted-audit-fields",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write second-pass audit metadata onto records that do not pass promotion while preserving their old quality tier.",
    )
    return parser.parse_args()


def parse_csv(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or payload.get("objects") or payload.get("rows") or []
    if not isinstance(records, list):
        raise TypeError(f"manifest_missing_records:{path}")
    return payload, [record for record in records if isinstance(record, dict)]


def object_key(record: dict[str, Any]) -> str:
    return str(
        record.get("canonical_object_id")
        or record.get("object_id")
        or record.get("source_uid")
        or record.get("source_model_path")
        or ""
    )


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def record_rank(record: dict[str, Any]) -> tuple[int, int, float, int, str]:
    quality = str(record.get("target_quality_tier") or "unknown")
    source = str(record.get("target_source_type") or "unknown")
    promoted = int(str(record.get("target_promotion_status") or "") == "promoted")
    quality_score = {"paper_strong": 4, "paper_pseudo": 3, "research_only": 2, "smoke_only": 1}.get(quality, 0)
    source_score = {"gt_render_baked": 3, "pseudo_from_multiview": 2, "pseudo_from_material_bank": 1}.get(source, 0)
    distance = record.get("target_prior_distance")
    distance_score = float(distance) if isinstance(distance, (int, float)) else 0.0
    return promoted, quality_score, distance_score, source_score, stable_hash(object_key(record))


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        key = object_key(record)
        if not key:
            continue
        current = by_key.get(key)
        if current is None or record_rank(record) > record_rank(current):
            by_key[key] = record
    return sorted(by_key.values(), key=object_key)


def make_candidate_record(
    record: dict[str, Any],
    *,
    promote_auxiliary_to_paper_main: bool,
) -> dict[str, Any]:
    candidate = dict(record)
    # The entire point of this pass is to re-check old smoke labels instead of
    # trusting them forever. Keep source-type evidence, but force tier inference.
    candidate["target_quality_tier"] = ""
    if promote_auxiliary_to_paper_main:
        candidate["supervision_role"] = "paper_main"
    return candidate


def promotion_blockers(
    row: dict[str, Any],
    record: dict[str, Any],
    *,
    allowed_license_buckets: set[str],
    promotable_source_names: set[str],
    min_confidence_mean: float,
    min_confidence_nonzero_rate: float,
    min_target_coverage: float,
    max_target_prior_identity: float,
    min_valid_view_count: int,
    min_strict_complete_view_rate: float,
) -> list[str]:
    blockers: list[str] = []
    source_name = str(row.get("source_name") or record.get("source_name") or "unknown")
    license_bucket = str(row.get("license_bucket") or record.get("license_bucket") or "unknown")
    target_source_type = str(row.get("target_source_type") or "unknown")
    target_quality_tier = str(row.get("target_quality_tier") or "unknown")
    if promotable_source_names and source_name not in promotable_source_names:
        blockers.append(f"source_not_promotable:{source_name}")
    if allowed_license_buckets and license_bucket not in allowed_license_buckets:
        blockers.append(f"license_not_allowed:{license_bucket}")
    if not bool(row.get("is_complete")):
        missing = ",".join(str(item) for item in row.get("missing_fields") or [])
        blockers.append(f"incomplete:{missing or 'unknown'}")
    if bool(row.get("target_is_prior_copy")):
        blockers.append("target_is_prior_copy")
    if target_source_type in BLOCKED_TARGET_SOURCES:
        blockers.append(f"blocked_target_source:{target_source_type}")
    if target_quality_tier not in PAPER_TIERS:
        blockers.append(f"non_paper_target_tier:{target_quality_tier}")
    confidence_summary = row.get("target_confidence_summary") if isinstance(row.get("target_confidence_summary"), dict) else {}
    confidence_mean = float(row.get("target_confidence_mean") or confidence_summary.get("mean", 0.0) or 0.0)
    nonzero_rate = float(
        row.get("target_confidence_nonzero_rate") or confidence_summary.get("nonzero_rate", 0.0) or 0.0
    )
    active_mean = confidence_active_mean(confidence_summary)
    confidence_mean_pass = confidence_mean >= float(min_confidence_mean) and nonzero_rate >= float(min_confidence_nonzero_rate)
    active_confidence_pass = (
        active_mean >= DEFAULT_MIN_TARGET_CONFIDENCE_ACTIVE_MEAN_FOR_PAPER
        and nonzero_rate >= DEFAULT_MIN_TARGET_CONFIDENCE_ACTIVE_COVERAGE_FOR_PAPER
    )
    if not (confidence_mean_pass or active_confidence_pass):
        if confidence_mean < float(min_confidence_mean):
            blockers.append(f"low_confidence_mean:{confidence_mean:.3f}:active_mean={active_mean:.3f}")
        if nonzero_rate < float(min_confidence_nonzero_rate):
            blockers.append(f"low_confidence_nonzero_rate:{nonzero_rate:.3f}")
        elif nonzero_rate < DEFAULT_MIN_TARGET_CONFIDENCE_ACTIVE_COVERAGE_FOR_PAPER:
            blockers.append(f"active_confidence_coverage_low:{nonzero_rate:.3f}")
    if float(row.get("target_coverage") or 0.0) < float(min_target_coverage):
        blockers.append(f"low_target_coverage:{float(row.get('target_coverage') or 0.0):.3f}")
    target_prior_identity = row.get("target_prior_identity")
    if target_prior_identity is None:
        blockers.append("missing_target_prior_identity")
    elif float(target_prior_identity) > float(max_target_prior_identity):
        blockers.append(f"target_prior_identity_high:{float(target_prior_identity):.3f}")
    if int(row.get("valid_view_count") or 0) < int(min_valid_view_count):
        blockers.append(f"valid_view_count_low:{int(row.get('valid_view_count') or 0)}")
    if float(row.get("strict_complete_view_rate") or 0.0) < float(min_strict_complete_view_rate):
        blockers.append(f"strict_complete_view_rate_low:{float(row.get('strict_complete_view_rate') or 0.0):.3f}")
    return blockers


def apply_row_fields(record: dict[str, Any], row: dict[str, Any]) -> None:
    for key in (
        "target_source_type",
        "target_is_prior_copy",
        "target_prior_identity",
        "target_prior_similarity",
        "target_prior_distance",
        "target_confidence_summary",
        "target_confidence_mean",
        "target_confidence_nonzero_rate",
        "target_coverage",
        "material_family",
        "thin_boundary_flag",
        "lighting_bank_id",
        "view_supervision_ready",
        "valid_view_count",
        "paper_license_allowed",
    ):
        if key in row:
            record[key] = row[key]


def promote_record(
    *,
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    record: dict[str, Any],
    args: argparse.Namespace,
    allowed_license_buckets: set[str],
    promotable_source_names: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = make_candidate_record(
        record,
        promote_auxiliary_to_paper_main=bool(args.promote_auxiliary_to_paper_main),
    )
    row = audit_record(
        manifest_path,
        manifest_payload,
        candidate,
        allowed_paper_license_buckets=allowed_license_buckets,
    )
    blockers = promotion_blockers(
        row,
        record,
        allowed_license_buckets=allowed_license_buckets,
        promotable_source_names=promotable_source_names,
        min_confidence_mean=float(args.min_confidence_mean),
        min_confidence_nonzero_rate=float(args.min_confidence_nonzero_rate),
        min_target_coverage=float(args.min_target_coverage),
        max_target_prior_identity=float(args.max_target_prior_identity),
        min_valid_view_count=int(args.min_valid_view_count),
        min_strict_complete_view_rate=float(args.min_strict_complete_view_rate),
    )

    out = dict(record)
    old_quality = str(out.get("target_quality_tier") or "unknown")
    old_role = str(out.get("supervision_role") or "unknown")
    apply_row_fields(out, row)
    out["target_second_pass_version"] = PROMOTION_VERSION
    out["target_second_pass_source_manifest"] = str(manifest_path.resolve())
    out["target_second_pass_candidate_quality_tier"] = row.get("target_quality_tier")
    out["target_second_pass_candidate_supervision_role"] = row.get("supervision_role")
    out["target_second_pass_blockers"] = blockers

    promoted = not blockers
    if promoted:
        out["target_quality_tier"] = row["target_quality_tier"]
        out["supervision_role"] = "paper_main"
        out["paper_stage_eligible"] = True
        out["target_promotion_status"] = "promoted"
        out["target_promotion_from_quality_tier"] = old_quality
        out["target_promotion_from_supervision_role"] = old_role
        out["target_promotion_reason"] = "second_pass_nontrivial_target_validated"
    else:
        if bool(args.keep_unpromoted_audit_fields):
            out["paper_stage_eligible"] = False
        else:
            out.pop("paper_stage_eligible", None)
        out["target_quality_tier"] = old_quality
        out["supervision_role"] = old_role
        out["target_promotion_status"] = "blocked"
    return out, {"row": row, "blockers": blockers, "promoted": promoted, "old_quality": old_quality, "old_role": old_role}


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key) or "unknown") for record in records))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_report(
    *,
    source_manifests: list[Path],
    input_records: int,
    output_records: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    audit_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    promoted_records = [record for record in output_records if record.get("target_promotion_status") == "promoted"]
    blocked_records = [record for record in output_records if record.get("target_promotion_status") == "blocked"]
    blocker_counts: Counter[str] = Counter()
    blocker_counts_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for record in blocked_records:
        source_name = str(record.get("source_name") or "unknown")
        blockers = record.get("target_second_pass_blockers") or ["unknown"]
        for blocker in blockers:
            blocker_key = str(blocker).split(":", 1)[0]
            blocker_counts[blocker_key] += 1
            blocker_counts_by_source[source_name][blocker_key] += 1
    return {
        "promotion_version": PROMOTION_VERSION,
        "source_manifests": [str(path.resolve()) for path in source_manifests],
        "output_manifest": str(args.output_manifest.resolve()),
        "input_records": input_records,
        "output_records": len(output_records),
        "promoted_records": len(promoted_records),
        "blocked_records": len(blocked_records),
        "promotion_by_source": distribution(promoted_records, "source_name"),
        "promotion_by_material_family": distribution(promoted_records, "material_family"),
        "promotion_by_target_quality_tier": distribution(promoted_records, "target_quality_tier"),
        "promotion_by_target_source_type": distribution(promoted_records, "target_source_type"),
        "output_target_quality_tier": distribution(output_records, "target_quality_tier"),
        "output_target_source_type": distribution(output_records, "target_source_type"),
        "output_supervision_role": distribution(output_records, "supervision_role"),
        "output_material_family": distribution(output_records, "material_family"),
        "output_source_name": distribution(output_records, "source_name"),
        "blocker_counts": dict(blocker_counts),
        "blocker_counts_by_source": {
            source: dict(counter)
            for source, counter in sorted(blocker_counts_by_source.items())
        },
        "audit_summary": audit_summary,
        "promotion_policy": {
            "allowed_license_buckets": sorted(parse_csv(args.allowed_license_buckets)),
            "promotable_source_names": sorted(parse_csv(args.promotable_source_names)),
            "min_confidence_mean": float(args.min_confidence_mean),
            "min_confidence_nonzero_rate": float(args.min_confidence_nonzero_rate),
            "min_target_coverage": float(args.min_target_coverage),
            "max_target_prior_identity": float(args.max_target_prior_identity),
            "min_valid_view_count": int(args.min_valid_view_count),
            "min_strict_complete_view_rate": float(args.min_strict_complete_view_rate),
            "promote_auxiliary_to_paper_main": bool(args.promote_auxiliary_to_paper_main),
        },
        "diagnostic_counts": {
            "old_quality": dict(Counter(str(item["old_quality"]) for item in diagnostics)),
            "old_role": dict(Counter(str(item["old_role"]) for item in diagnostics)),
        },
    }


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Material Refine Target Promotion Report",
        "",
        f"- output_manifest: `{report['output_manifest']}`",
        f"- input_records: `{report['input_records']}`",
        f"- output_records: `{report['output_records']}`",
        f"- promoted_records: `{report['promoted_records']}`",
        f"- paper_stage_eligible_records: `{report['audit_summary'].get('paper_stage_eligible_records')}`",
        f"- target_prior_identity_rate: `{float(report['audit_summary'].get('target_prior_identity_rate', 0.0)):.4f}`",
        "",
        "## Promoted",
        "",
        f"- by_source: `{json.dumps(report['promotion_by_source'], ensure_ascii=False)}`",
        f"- by_material_family: `{json.dumps(report['promotion_by_material_family'], ensure_ascii=False)}`",
        f"- by_target_quality_tier: `{json.dumps(report['promotion_by_target_quality_tier'], ensure_ascii=False)}`",
        "",
        "## Blockers",
        "",
    ]
    for key, value in report.get("blocker_counts", {}).items() or {"none": 0}.items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_html(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for key in (
        "promotion_by_source",
        "promotion_by_material_family",
        "promotion_by_target_quality_tier",
        "blocker_counts",
        "output_target_quality_tier",
        "output_supervision_role",
    ):
        rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td><code>{html.escape(json.dumps(report.get(key, {}), ensure_ascii=False))}</code></td>"
            "</tr>"
        )
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Target Promotion Report</title>",
        "<style>body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1200px;margin:auto}.card{background:#18202b;border-radius:16px;padding:18px;margin:16px 0}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #334052;padding:8px;text-align:left;vertical-align:top}code{color:#b9e6ff;white-space:pre-wrap}</style>",
        "</head><body><div class='wrap'>",
        "<h1>Material Refine Target Promotion Report</h1>",
        "<div class='card'>",
        f"<p>output_manifest: <code>{html.escape(report['output_manifest'])}</code></p>",
        f"<p>promoted_records: <code>{report['promoted_records']}</code> / <code>{report['output_records']}</code></p>",
        f"<p>paper_stage_eligible_records: <code>{report['audit_summary'].get('paper_stage_eligible_records')}</code></p>",
        "</div>",
        "<div class='card'><table><tbody>",
        *rows,
        "</tbody></table></div>",
        "</div></body></html>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    allowed_license_buckets = parse_csv(args.allowed_license_buckets)
    promotable_source_names = parse_csv(args.promotable_source_names)

    promoted_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    input_records = 0
    for manifest in args.manifest:
        payload, records = load_manifest(manifest)
        input_records += len(records)
        for record in records:
            out, diagnostic = promote_record(
                manifest_path=manifest,
                manifest_payload=payload,
                record=record,
                args=args,
                allowed_license_buckets=allowed_license_buckets,
                promotable_source_names=promotable_source_names,
            )
            promoted_records.append(out)
            diagnostics.append(diagnostic)

    output_records = dedupe_records(promoted_records)
    output_payload = {
        "manifest_version": "canonical_asset_record_v1_target_promoted",
        "promotion_version": PROMOTION_VERSION,
        "source_manifests": [str(path.resolve()) for path in args.manifest],
        "summary": {
            "records": len(output_records),
            "source_name": distribution(output_records, "source_name"),
            "material_family": distribution(output_records, "material_family"),
            "license_bucket": distribution(output_records, "license_bucket"),
            "supervision_role": distribution(output_records, "supervision_role"),
            "target_quality_tier": distribution(output_records, "target_quality_tier"),
            "target_source_type": distribution(output_records, "target_source_type"),
            "target_promotion_status": distribution(output_records, "target_promotion_status"),
            "paper_split": distribution(output_records, "paper_split"),
            "default_split": distribution(output_records, "default_split"),
        },
        "records": output_records,
    }
    write_json(args.output_manifest, output_payload)

    output_audit_rows = [
        audit_record(
            args.output_manifest,
            output_payload,
            record,
            allowed_paper_license_buckets=allowed_license_buckets,
        )
        for record in output_records
    ]
    audit_summary = summarize_audit_rows(
        output_audit_rows,
        identity_warning_threshold=0.95,
    )
    report = build_report(
        source_manifests=args.manifest,
        input_records=input_records,
        output_records=output_records,
        diagnostics=diagnostics,
        audit_summary=audit_summary,
        args=args,
    )
    report_json = args.report_json or args.output_manifest.with_suffix(".promotion_report.json")
    report_md = args.report_md or args.output_manifest.with_suffix(".promotion_report.md")
    report_html = args.report_html or args.output_manifest.with_suffix(".promotion_report.html")
    write_json(report_json, report)
    write_report_md(report_md, report)
    write_report_html(report_html, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
