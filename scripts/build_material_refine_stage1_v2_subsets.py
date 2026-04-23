from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.manifest_quality import audit_manifest


PAPER_TARGET_QUALITY_TIERS = {"paper_pseudo", "paper_strong"}
BLOCKED_TARGET_SOURCE_TYPES = {"copied_from_prior", "unknown"}
DEFAULT_PAPER_LICENSE_BUCKETS = {
    "cc_by_nc_4_0",
    "cc_by_nc_4_0_pending_reconcile",
    "custom_tianchi_research_noncommercial_no_redistribution",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage1-v2 strict paper, diverse diagnostic, and OOD eval subsets.",
    )
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--paper-license-buckets", type=str, default=",".join(sorted(DEFAULT_PAPER_LICENSE_BUCKETS)))
    parser.add_argument("--max-target-prior-identity-rate-for-paper", type=float, default=0.30)
    parser.add_argument("--min-nontrivial-target-count-for-paper", type=int, default=128)
    parser.add_argument("--max-diagnostic-records", type=int, default=512)
    parser.add_argument("--max-ood-records", type=int, default=256)
    parser.add_argument(
        "--diagnostic-min-per-material-family",
        type=int,
        default=48,
        help="Reserve at least this many diagnostic records per material family before filling focus records.",
    )
    parser.add_argument(
        "--ood-min-per-material-family",
        type=int,
        default=32,
        help="Reserve at least this many OOD records per material family before filling focus records.",
    )
    parser.add_argument("--strict-val-ratio", type=float, default=0.12)
    parser.add_argument("--strict-test-ratio", type=float, default=0.18)
    parser.add_argument("--diagnostic-focus-source", type=str, default="3D-FUTURE_highlight_local_8k")
    parser.add_argument("--diagnostic-focus-material-family", type=str, default="metal_dominant")
    return parser.parse_args()


def parse_csv(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def stable_hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def deterministic_split(object_id: str, *, val_ratio: float, test_ratio: float) -> str:
    bucket = int(stable_hash_key(object_id)[:8], 16) / 0xFFFFFFFF
    if bucket < val_ratio:
        return "val"
    if bucket < val_ratio + test_ratio:
        return "test"
    return "train"


def load_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError(f"manifest_missing_records:{path}")
    return payload, records


def audit_rows_by_object(manifest: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    audit_payload = audit_manifest(
        manifest,
        max_records=-1,
        identity_warning_threshold=float(args.max_target_prior_identity_rate_for_paper),
        max_target_prior_identity_rate_for_paper=float(args.max_target_prior_identity_rate_for_paper),
        min_nontrivial_target_count_for_paper=int(args.min_nontrivial_target_count_for_paper),
        allowed_paper_license_buckets=parse_csv(args.paper_license_buckets),
    )
    rows = {
        str(row.get("object_id")): row
        for row in audit_payload.get("records", [])
        if row.get("object_id") is not None
    }
    return audit_payload, rows


def enrich_record(record: dict[str, Any], row: dict[str, Any] | None, *, manifest: Path) -> dict[str, Any]:
    out = dict(record)
    out["source_manifest"] = str(manifest.resolve())
    if row is not None:
        for key in (
            "target_source_type",
            "target_is_prior_copy",
            "target_prior_similarity",
            "target_prior_distance",
            "target_quality_tier",
            "target_confidence_summary",
            "paper_stage_eligible",
            "paper_license_allowed",
            "is_complete",
            "category_bucket",
            "category_label",
            "material_family",
            "thin_boundary_flag",
            "lighting_bank_id",
            "effective_view_supervision_rate",
            "view_supervision_ready",
            "supervision_role",
            "supervision_tier",
            "source_name",
        ):
            if key in row:
                out[key] = row[key]
    out.setdefault("target_source_type", "unknown")
    out.setdefault("target_quality_tier", "unknown")
    out.setdefault("supervision_role", "unknown")
    out.setdefault("material_family", "unknown")
    out.setdefault("source_name", out.get("generator_id", "unknown"))
    return out


def record_key(record: dict[str, Any]) -> str:
    return str(record.get("object_id") or record.get("id") or record.get("source_uid"))


def record_score(record: dict[str, Any]) -> tuple[int, float, int, int, int]:
    quality = str(record.get("target_quality_tier", "unknown"))
    source = str(record.get("target_source_type", "unknown"))
    license_bucket = str(record.get("license_bucket", "unknown"))
    eligible = int(bool(record.get("paper_stage_eligible")))
    quality_score = {"paper_strong": 3, "paper_pseudo": 2, "research_only": 1, "smoke_only": 0}.get(quality, -1)
    source_score = {"gt_render_baked": 3, "pseudo_from_multiview": 2, "pseudo_from_material_bank": 1}.get(source, 0)
    license_score = 2 if license_bucket == "cc_by_nc_4_0" else 1 if license_bucket in DEFAULT_PAPER_LICENSE_BUCKETS else 0
    similarity = record.get("target_prior_similarity")
    distance_score = 0.0 if not isinstance(similarity, (int, float)) else 1.0 - float(similarity)
    return eligible, distance_score, quality_score, source_score, license_score


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        if not key:
            continue
        current = selected.get(key)
        if current is None or record_score(record) > record_score(current):
            selected[key] = record
    return list(selected.values())


def is_strict_paper_record(record: dict[str, Any], allowed_license_buckets: set[str]) -> bool:
    if str(record.get("target_quality_tier")) not in PAPER_TARGET_QUALITY_TIERS:
        return False
    if str(record.get("target_source_type")) in BLOCKED_TARGET_SOURCE_TYPES:
        return False
    if bool(record.get("target_is_prior_copy")):
        return False
    if not bool(record.get("paper_stage_eligible")):
        return False
    if not bool(record.get("paper_license_allowed", True)):
        return False
    if allowed_license_buckets and str(record.get("license_bucket", "unknown")) not in allowed_license_buckets:
        return False
    effective_view_rate = record.get("effective_view_supervision_rate")
    view_ready = bool(record.get("view_supervision_ready")) or (
        isinstance(effective_view_rate, (int, float)) and float(effective_view_rate) > 0.0
    )
    return view_ready


def diagnostic_priority(record: dict[str, Any], args: argparse.Namespace) -> tuple[int, int, str]:
    source_match = int(str(record.get("source_name")) == str(args.diagnostic_focus_source))
    material_match = int(str(record.get("material_family")) == str(args.diagnostic_focus_material_family))
    return source_match, material_match, stable_hash_key(record_key(record))


def balanced_take_by_key(
    records: list[dict[str, Any]],
    *,
    key: str,
    limit: int,
    min_per_group: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if min_per_group <= 0:
        return records[:limit]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get(key, "unknown")), []).append(record)
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for group_name in sorted(grouped):
        group_records = sorted(grouped[group_name], key=lambda record: diagnostic_priority(record, args), reverse=True)
        for record in group_records[:min_per_group]:
            if len(selected) >= limit:
                return selected
            selected.append(record)
            selected_keys.add(record_key(record))
    for record in records:
        if len(selected) >= limit:
            break
        key_value = record_key(record)
        if key_value in selected_keys:
            continue
        selected.append(record)
        selected_keys.add(key_value)
    return selected


def assign_paper_split(record: dict[str, Any], *, val_ratio: float, test_ratio: float) -> tuple[str, str]:
    existing = str(record.get("paper_split") or "")
    if existing in {"paper_train", "paper_val_iid", "paper_test_iid", "paper_test_material_holdout"}:
        if existing == "paper_train":
            return "paper_train", "train"
        if existing == "paper_val_iid":
            return "paper_val_iid", "val"
        return existing, "test"
    split = deterministic_split(record_key(record), val_ratio=val_ratio, test_ratio=test_ratio)
    if split == "val":
        return "paper_val_iid", "val"
    if split == "test":
        return "paper_test_iid", "test"
    return "paper_train", "train"


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key, "unknown")) for record in records))


def manifest_payload(
    *,
    records: list[dict[str, Any]],
    source_manifests: list[Path],
    subset_name: str,
    experiment_role: str,
) -> dict[str, Any]:
    return {
        "manifest_version": "canonical_asset_record_v1_stage1_v2",
        "subset_name": subset_name,
        "experiment_role": experiment_role,
        "source_manifests": [str(path.resolve()) for path in source_manifests],
        "summary": summarize(records),
        "records": records,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "default_split": distribution(records, "default_split"),
        "paper_split": distribution(records, "paper_split"),
        "source_name": distribution(records, "source_name"),
        "generator_id": distribution(records, "generator_id"),
        "material_family": distribution(records, "material_family"),
        "license_bucket": distribution(records, "license_bucket"),
        "target_quality_tier": distribution(records, "target_quality_tier"),
        "target_source_type": distribution(records, "target_source_type"),
        "supervision_role": distribution(records, "supervision_role"),
        "has_material_prior": {
            "true": sum(bool(record.get("has_material_prior")) for record in records),
            "false": sum(not bool(record.get("has_material_prior")) for record in records),
        },
        "view_supervision_ready": {
            "true": sum(bool(record.get("view_supervision_ready")) for record in records),
            "false": sum(not bool(record.get("view_supervision_ready")) for record in records),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
            f"<td><code>{html.escape(json.dumps(summary.get('license_bucket', {}), ensure_ascii=False))}</code></td>"
            "</tr>"
        )
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Stage1-v2 Dataset Sync Report</title>",
        "<style>body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1320px;margin:auto}.card{background:#18202b;border-radius:18px;padding:18px;margin:16px 0}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #334052;padding:8px;text-align:left;vertical-align:top}code{color:#b9e6ff}</style>",
        "</head><body><div class='wrap'>",
        "<h1>Stage1-v2 Dataset Sync Report</h1>",
        "<div class='card'>",
        f"<p>recommendation: <code>{html.escape(report['recommendation'])}</code></p>",
        f"<p>blockers: <code>{html.escape(json.dumps(report.get('blockers', []), ensure_ascii=False))}</code></p>",
        "</div>",
        "<div class='card'><table><thead><tr><th>Subset</th><th>Records</th><th>Material</th><th>Source</th><th>Target quality</th><th>License</th></tr></thead><tbody>",
        *rows,
        "</tbody></table></div>",
        "</div></body></html>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stage1-v2 Dataset Sync Report",
        "",
        f"- recommendation: `{report['recommendation']}`",
        f"- strict_records: `{report['subset_summaries']['strict_paper']['records']}`",
        f"- diagnostic_records: `{report['subset_summaries']['diverse_diagnostic']['records']}`",
        f"- ood_records: `{report['subset_summaries']['ood_eval']['records']}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in report.get("blockers") or ["none"]:
        lines.append(f"- {blocker}")
    lines.extend(["", "## Subsets", ""])
    for name, summary in report["subset_summaries"].items():
        lines.append(f"### {name}")
        for key in ("records", "source_name", "material_family", "license_bucket", "target_quality_tier", "target_source_type", "paper_split"):
            lines.append(f"- {key}: `{json.dumps(summary.get(key), ensure_ascii=False)}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    allowed_license_buckets = parse_csv(args.paper_license_buckets)
    all_records: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    for manifest in args.manifest:
        _payload, records = load_records(manifest)
        audit_payload, rows = audit_rows_by_object(manifest, args)
        enriched = [enrich_record(record, rows.get(record_key(record)), manifest=manifest) for record in records]
        all_records.extend(enriched)
        source_summaries[str(manifest.resolve())] = audit_payload.get("summary", {})

    unique_records = dedupe_records(all_records)
    strict_records = [
        dict(record, experiment_role="paper_stage")
        for record in unique_records
        if is_strict_paper_record(record, allowed_license_buckets)
    ]
    for record in strict_records:
        paper_split, default_split = assign_paper_split(
            record,
            val_ratio=float(args.strict_val_ratio),
            test_ratio=float(args.strict_test_ratio),
        )
        record["paper_split"] = paper_split
        record["default_split"] = default_split

    diagnostic_candidates = [
        dict(record, experiment_role="diagnostic_only", paper_split="diagnostic_only", default_split="test")
        for record in unique_records
        if (
            str(record.get("target_quality_tier")) in {"paper_pseudo", "paper_strong", "research_only", "smoke_only"}
            and str(record.get("target_source_type")) not in {"copied_from_prior", "unknown"}
            and not bool(record.get("target_is_prior_copy"))
        )
    ]
    diagnostic_candidates.sort(key=lambda record: diagnostic_priority(record, args), reverse=True)
    diagnostic_records = balanced_take_by_key(
        diagnostic_candidates,
        key="material_family",
        limit=max(int(args.max_diagnostic_records), 0),
        min_per_group=max(int(args.diagnostic_min_per_material_family), 0),
        args=args,
    )

    ood_candidates = [
        dict(record, experiment_role="ood_eval", paper_split="paper_test_ood_object", default_split="test")
        for record in diagnostic_candidates
        if str(record.get("source_name")) != "ABO_locked_core"
        or str(record.get("generator_id")) != "abo_locked_core"
    ]
    ood_records = balanced_take_by_key(
        ood_candidates,
        key="material_family",
        limit=max(int(args.max_ood_records), 0),
        min_per_group=max(int(args.ood_min_per_material_family), 0),
        args=args,
    )

    strict_manifest = manifest_payload(
        records=strict_records,
        source_manifests=args.manifest,
        subset_name="stage1_v2_strict_paper",
        experiment_role="paper_stage",
    )
    diagnostic_manifest = manifest_payload(
        records=diagnostic_records,
        source_manifests=args.manifest,
        subset_name="stage1_v2_diverse_diagnostic",
        experiment_role="diagnostic_only",
    )
    ood_manifest = manifest_payload(
        records=ood_records,
        source_manifests=args.manifest,
        subset_name="stage1_v2_ood_eval",
        experiment_role="ood_eval",
    )

    strict_count = len(strict_records)
    material_coverage = len([key for key, value in distribution(strict_records, "material_family").items() if value > 0])
    blockers = []
    if strict_count <= 346:
        blockers.append(f"strict_paper_records={strict_count} not greater than current_locked_346")
    if material_coverage <= 1:
        blockers.append(f"strict_material_family_coverage={material_coverage} insufficient_for_replacing_current_main")
    recommendation = "KEEP_CURRENT_346_AS_MAIN"
    if strict_count > 346 and material_coverage > 1:
        recommendation = "STAGE1_V2_STRICT_CAN_REPLACE_CURRENT_AFTER_SMOKE"
    report = {
        "source_manifests": [str(path.resolve()) for path in args.manifest],
        "source_audit_summaries": source_summaries,
        "unique_records": len(unique_records),
        "subset_paths": {
            "strict_paper": str((args.output_root / "stage1_v2_strict_paper_manifest.json").resolve()),
            "diverse_diagnostic": str((args.output_root / "stage1_v2_diverse_diagnostic_manifest.json").resolve()),
            "ood_eval": str((args.output_root / "stage1_v2_ood_eval_manifest.json").resolve()),
        },
        "subset_summaries": {
            "strict_paper": strict_manifest["summary"],
            "diverse_diagnostic": diagnostic_manifest["summary"],
            "ood_eval": ood_manifest["summary"],
        },
        "blockers": blockers,
        "recommendation": recommendation,
        "selection_policy": {
            "diagnostic_min_per_material_family": int(args.diagnostic_min_per_material_family),
            "ood_min_per_material_family": int(args.ood_min_per_material_family),
            "diagnostic_focus_source": str(args.diagnostic_focus_source),
            "diagnostic_focus_material_family": str(args.diagnostic_focus_material_family),
        },
    }

    write_json(args.output_root / "stage1_v2_strict_paper_manifest.json", strict_manifest)
    write_json(args.output_root / "stage1_v2_diverse_diagnostic_manifest.json", diagnostic_manifest)
    write_json(args.output_root / "stage1_v2_ood_eval_manifest.json", ood_manifest)
    write_json(args.output_root / "stage1_v2_dataset_sync_report.json", report)
    write_report_html(args.output_root / "stage1_v2_dataset_sync_report.html", report)
    write_report_md(args.output_root / "stage1_v2_dataset_sync_report.md", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
