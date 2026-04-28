#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER_TARGET_SOURCES = {"gt_render_baked", "pseudo_from_multiview"}
BLOCKED_TARGET_SOURCES = {"copied_from_prior", "unknown", "real_benchmark_no_uv_target"}
MATERIAL_FAMILIES = (
    "metal_dominant",
    "ceramic_glazed_lacquer",
    "glass_metal",
    "mixed_thin_boundary",
    "glossy_non_metal",
)
REQUIRED_UV_FIELDS = (
    "uv_target_roughness_path",
    "uv_target_metallic_path",
    "uv_target_confidence_path",
)
REQUIRED_VIEW_FIELDS = ("uv", "visibility", "mask", "roughness", "metallic")
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Audit rebake_v2 target/view contract and produce candidate/promoted/reject manifests.",
    )
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mean-threshold", type=float, default=0.03)
    parser.add_argument("--p95-threshold", type=float, default=0.08)
    parser.add_argument("--max-target-prior-identity", type=float, default=0.30)
    parser.add_argument("--min-target-confidence-mean", type=float, default=0.70)
    parser.add_argument("--min-target-confidence-nonzero-rate", type=float, default=0.50)
    parser.add_argument("--min-balanced-records", type=int, default=256)
    parser.add_argument("--min-source-diversity", type=int, default=2)
    parser.add_argument("--max-source-ratio", type=float, default=0.60)
    parser.add_argument("--min-no-prior-ratio", type=float, default=0.15)
    parser.add_argument("--allow-license-buckets", type=str, default=",".join(sorted(ALLOWED_LICENSE_BUCKETS)))
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_csv(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []) or []:
            if isinstance(record, dict):
                item = dict(record)
                item["rebake_v2_source_manifest"] = str(path)
                records.append(item)
    return dedupe(records)


def object_key(record: dict[str, Any]) -> str:
    return str(
        record.get("canonical_object_id")
        or record.get("object_id")
        or record.get("source_uid")
        or record.get("source_model_path")
        or ""
    )


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        key = object_key(record)
        if not key:
            continue
        current = selected.get(key)
        if current is None or record_rank(record) > record_rank(current):
            selected[key] = record
    return sorted(selected.values(), key=object_key)


def record_rank(record: dict[str, Any]) -> tuple[int, float, int]:
    quality = {"paper_strong": 3, "paper_pseudo": 2, "research_only": 1, "smoke_only": 0}.get(
        str(record.get("target_quality_tier") or "unknown"),
        0,
    )
    confidence = float(record.get("target_confidence_mean") or 0.0)
    return quality, confidence, len(record)


def existing_path(value: Any) -> bool:
    if not value or not isinstance(value, str):
        return False
    return Path(value).exists()


def record_has_view_field(record: dict[str, Any], field: str) -> bool:
    direct = record.get(f"view_{field}_paths")
    if isinstance(direct, dict) and direct:
        return True
    nested_keys = ("view_buffer_paths", "view_rgba_paths", "view_buffers")
    for key in nested_keys:
        payload = record.get(key)
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, dict) and value.get(field):
                    return True
    buffer_root = record.get("canonical_buffer_root")
    if isinstance(buffer_root, str) and Path(buffer_root).exists():
        candidates = list(Path(buffer_root).glob(f"*/{field}.png")) + list(Path(buffer_root).glob(f"*/{field}.exr"))
        if candidates:
            return True
    return False


def bool_field(record: dict[str, Any], key: str) -> bool:
    value = record.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "pass", "passed"}
    return bool(value)


def audit_record(record: dict[str, Any], args: argparse.Namespace, allowed_licenses: set[str]) -> dict[str, Any]:
    blockers: list[str] = []
    if str(record.get("target_view_contract_version") or "") != "v2":
        blockers.append("target_view_contract_version_not_v2")
    if str(record.get("rebake_version") or "") != "rebake_v2":
        blockers.append("rebake_version_not_rebake_v2")
    if not bool_field(record, "stored_view_target_valid_for_paper"):
        blockers.append("stored_view_target_not_valid_for_paper")
    if not bool_field(record, "prior_as_pred_pass"):
        blockers.append("prior_as_pred_pass_false")
    if not bool_field(record, "target_as_pred_pass"):
        blockers.append("target_as_pred_pass_false")
    mean = record.get("target_view_alignment_mean")
    p95 = record.get("target_view_alignment_p95")
    if not isinstance(mean, (int, float)):
        blockers.append("missing_target_view_alignment_mean")
    elif float(mean) >= float(args.mean_threshold):
        blockers.append(f"target_view_alignment_mean_high:{float(mean):.5f}")
    if not isinstance(p95, (int, float)):
        blockers.append("missing_target_view_alignment_p95")
    elif float(p95) >= float(args.p95_threshold):
        blockers.append(f"target_view_alignment_p95_high:{float(p95):.5f}")
    if not existing_path(record.get("canonical_buffer_root")):
        blockers.append("missing_canonical_buffer_root")
    for field in REQUIRED_VIEW_FIELDS:
        if not record_has_view_field(record, field):
            blockers.append(f"missing_view_{field}")
    for field in REQUIRED_UV_FIELDS:
        if not existing_path(record.get(field)):
            blockers.append(f"missing_{field}")
    target_source = str(record.get("target_source_type") or "unknown")
    if target_source not in PAPER_TARGET_SOURCES:
        blockers.append(f"target_source_not_promotable:{target_source}")
    if target_source in BLOCKED_TARGET_SOURCES:
        blockers.append(f"target_source_blocked:{target_source}")
    if bool_field(record, "target_is_prior_copy") or bool_field(record, "copied_from_prior"):
        blockers.append("target_is_prior_copy")
    identity = record.get("target_prior_identity")
    if not isinstance(identity, (int, float)):
        blockers.append("missing_target_prior_identity")
    elif float(identity) > float(args.max_target_prior_identity):
        blockers.append(f"target_prior_identity_high:{float(identity):.5f}")
    confidence_mean = float(record.get("target_confidence_mean") or 0.0)
    confidence_nonzero = float(record.get("target_confidence_nonzero_rate") or 0.0)
    if confidence_mean < float(args.min_target_confidence_mean):
        blockers.append(f"target_confidence_mean_low:{confidence_mean:.5f}")
    if confidence_nonzero < float(args.min_target_confidence_nonzero_rate):
        blockers.append(f"target_confidence_nonzero_rate_low:{confidence_nonzero:.5f}")
    if not bool_field(record, "view_supervision_ready"):
        blockers.append("view_supervision_ready_false")
    license_bucket = str(record.get("license_bucket") or "unknown")
    if allowed_licenses and license_bucket not in allowed_licenses:
        blockers.append(f"license_not_allowed:{license_bucket}")

    out = dict(record)
    out["rebake_v2_gate_blockers"] = blockers
    out["rebake_v2_gate_pass"] = not blockers
    out["stored_view_target_valid_for_paper"] = bool(out.get("stored_view_target_valid_for_paper")) and not blockers
    out["paper_stage_eligible_rebake_v2"] = not blockers
    if blockers:
        out["supervision_role"] = "diagnostic_only" if is_diagnostic_like(record) else "rejected"
        out["target_quality_tier"] = str(record.get("target_quality_tier") or "smoke_only")
        out["reject_reason"] = blockers
        out["candidate_pool_only"] = True
    else:
        out["supervision_role"] = "paper_main"
        out["candidate_pool_only"] = False
        if target_source == "gt_render_baked" and confidence_mean >= 0.80:
            out["target_quality_tier"] = "paper_strong"
        else:
            out["target_quality_tier"] = "paper_pseudo"
    return out


def is_diagnostic_like(record: dict[str, Any]) -> bool:
    if bool_field(record, "target_is_prior_copy") or str(record.get("target_source_type") or "") in BLOCKED_TARGET_SOURCES:
        return False
    return True


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key) if record.get(key) is not None else "unknown") for record in records))


def bool_distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {"true": sum(bool_field(record, key) for record in records), "false": sum(not bool_field(record, key) for record in records)}


def numeric_stats(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = sorted(float(record[key]) for record in records if isinstance(record.get(key), (int, float)))
    if not vals:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(vals),
        "mean": statistics.mean(vals),
        "min": vals[0],
        "p50": vals[int((len(vals) - 1) * 0.50)],
        "p95": vals[int((len(vals) - 1) * 0.95)],
        "max": vals[-1],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "source_name": distribution(records, "source_name"),
        "generator_id": distribution(records, "generator_id"),
        "material_family": distribution(records, "material_family"),
        "target_quality_tier": distribution(records, "target_quality_tier"),
        "target_source_type": distribution(records, "target_source_type"),
        "supervision_role": distribution(records, "supervision_role"),
        "has_material_prior": bool_distribution(records, "has_material_prior"),
        "rebake_v2_gate_pass": bool_distribution(records, "rebake_v2_gate_pass"),
        "paper_stage_eligible_rebake_v2": bool_distribution(records, "paper_stage_eligible_rebake_v2"),
        "target_view_alignment_mean_stats": numeric_stats(records, "target_view_alignment_mean"),
        "target_view_alignment_p95_stats": numeric_stats(records, "target_view_alignment_p95"),
        "target_prior_identity_stats": numeric_stats(records, "target_prior_identity"),
    }


def priority(record: dict[str, Any]) -> tuple[int, float, float, str]:
    source_score = {"gt_render_baked": 2, "pseudo_from_multiview": 1}.get(str(record.get("target_source_type") or ""), 0)
    confidence = float(record.get("target_confidence_mean") or 0.0)
    distance = 1.0 - float(record.get("target_prior_identity") or 1.0)
    return source_score, confidence, distance, object_key(record)


def build_balanced(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("material_family") or "unknown")].append(record)
    for group in grouped.values():
        group.sort(key=priority, reverse=True)
    selected: list[dict[str, Any]] = []
    # Keep the quotas as lower-bound policy; do not fill with glossy if scarce families are absent.
    targets = {
        "metal_dominant": 0.20,
        "ceramic_glazed_lacquer": 0.15,
        "glass_metal": 0.15,
        "mixed_thin_boundary": 0.15,
        "glossy_non_metal": 0.35,
    }
    total = len(records)
    for family, ratio in targets.items():
        take = int(total * ratio)
        if family == "glossy_non_metal":
            take = min(take, len(grouped.get(family, [])))
        selected.extend(grouped.get(family, [])[:take])
    return dedupe(selected)


def quota_blockers(records: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    blockers: list[str] = []
    total = max(len(records), 1)
    material = Counter(str(record.get("material_family") or "unknown") for record in records)
    required_min = {
        "metal_dominant": 0.20,
        "ceramic_glazed_lacquer": 0.15,
        "glass_metal": 0.15,
        "mixed_thin_boundary": 0.15,
    }
    for family, ratio in required_min.items():
        if material.get(family, 0) / total < ratio:
            blockers.append(f"material_family[{family}]={material.get(family, 0)}/{total} below {ratio:.2f}")
    if material.get("glossy_non_metal", 0) / total > 0.35:
        blockers.append(f"glossy_non_metal_ratio={material.get('glossy_non_metal', 0) / total:.3f} above 0.350")
    no_prior = sum(not bool_field(record, "has_material_prior") for record in records)
    if no_prior / total < float(args.min_no_prior_ratio):
        blockers.append(f"without_prior_ratio={no_prior / total:.3f} below {float(args.min_no_prior_ratio):.3f}")
    source = Counter(str(record.get("source_name") or "unknown") for record in records)
    if len(source) < int(args.min_source_diversity):
        blockers.append(f"source_diversity={len(source)} below {int(args.min_source_diversity)}")
    if source and max(source.values()) / total > float(args.max_source_ratio):
        blockers.append(f"max_source_ratio={max(source.values()) / total:.3f} above {float(args.max_source_ratio):.3f}")
    if len(records) < int(args.min_balanced_records):
        blockers.append(f"balanced_records={len(records)} below {int(args.min_balanced_records)}")
    return blockers


def write_manifest(path: Path, *, records: list[dict[str, Any]], name: str, role: str, source_manifests: list[Path]) -> None:
    payload = {
        "manifest_version": "canonical_asset_record_v1_rebake_v2",
        "subset_name": name,
        "experiment_role": role,
        "source_manifests": [str(path) for path in source_manifests],
        "summary": summarize(records),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    allowed = parse_csv(args.allow_license_buckets)
    records = load_records(args.manifest)
    audited = [audit_record(record, args, allowed) for record in records]
    promoted = [record for record in audited if bool_field(record, "rebake_v2_gate_pass")]
    diagnostic = [record for record in audited if record.get("supervision_role") == "diagnostic_only"]
    rejects = [record for record in audited if record.get("supervision_role") == "rejected"]
    balanced = build_balanced(promoted)
    quota_issues = quota_blockers(balanced, args)
    target_view_issues = []
    if promoted:
        mean_stats = numeric_stats(promoted, "target_view_alignment_mean")
        p95_stats = numeric_stats(promoted, "target_view_alignment_p95")
        if (mean_stats.get("mean") or 0.0) >= float(args.mean_threshold):
            target_view_issues.append("target_view_alignment_mean_mean_above_threshold")
        if (p95_stats.get("p95") or 0.0) >= float(args.p95_threshold):
            target_view_issues.append("target_view_alignment_p95_p95_above_threshold")
    else:
        target_view_issues.append("no_rebake_v2_gate_pass_records")
    if target_view_issues:
        decision = "KEEP_AS_DIAGNOSTIC_ONLY"
    elif quota_issues:
        decision = "KEEP_AS_DATA_CANDIDATE_ONLY"
    else:
        decision = "READY_FOR_R_TRAINING_CANDIDATE"

    write_manifest(args.output_root / "merged_manifest_rebake_v2.json", records=audited, name="merged_manifest_rebake_v2", role="rebake_v2_all_audited", source_manifests=args.manifest)
    write_manifest(args.output_root / "stage1_v4_no3dfuture_rebake_v2_balanced_manifest.json", records=balanced, name="stage1_v4_no3dfuture_rebake_v2_balanced", role="paper_candidate_after_rebake_v2_gate", source_manifests=args.manifest)
    write_manifest(args.output_root / "diagnostic_manifest_rebake_v2.json", records=diagnostic, name="diagnostic_manifest_rebake_v2", role="diagnostic_only", source_manifests=args.manifest)
    write_manifest(args.output_root / "ood_manifest_rebake_v2.json", records=[r for r in diagnostic if str(r.get("source_name") or "") != "ABO_locked_core"], name="ood_manifest_rebake_v2", role="ood_eval_only", source_manifests=args.manifest)
    real_lighting = [r for r in diagnostic if any(token in str(r.get("lighting_bank_id") or "").lower() for token in ("real", "olat", "illumination", "orb"))]
    write_manifest(args.output_root / "real_lighting_manifest_rebake_v2.json", records=real_lighting, name="real_lighting_manifest_rebake_v2", role="real_lighting_eval_only" if real_lighting else "insufficient_real_lighting", source_manifests=args.manifest)
    write_manifest(args.output_root / "rejects_manifest_rebake_v2.json", records=rejects, name="rejects_manifest_rebake_v2", role="rejected", source_manifests=args.manifest)

    blocker_counts: Counter[str] = Counter()
    for record in audited:
        for blocker in record.get("rebake_v2_gate_blockers") or []:
            blocker_counts[str(blocker).split(":", 1)[0]] += 1
    report = {
        "generated_at_utc": utc_now(),
        "decision": decision,
        "approve_r_training": decision == "READY_FOR_R_TRAINING_CANDIDATE",
        "approve_stage1_v4_no3dfuture_as_train_candidate": decision == "READY_FOR_R_TRAINING_CANDIDATE",
        "approve_paper_claim": False,
        "source_manifests": [str(path) for path in args.manifest],
        "thresholds": {
            "target_view_alignment_mean": float(args.mean_threshold),
            "target_view_alignment_p95": float(args.p95_threshold),
            "max_target_prior_identity": float(args.max_target_prior_identity),
        },
        "records": len(audited),
        "promoted_records": len(promoted),
        "balanced_records": len(balanced),
        "diagnostic_records": len(diagnostic),
        "reject_records": len(rejects),
        "quota_blockers": quota_issues,
        "target_view_blockers": target_view_issues,
        "gate_blocker_counts": dict(blocker_counts),
        "summaries": {
            "all": summarize(audited),
            "promoted": summarize(promoted),
            "balanced": summarize(balanced),
            "diagnostic": summarize(diagnostic),
            "rejects": summarize(rejects),
        },
    }
    (args.output_root / "stage1_v4_no3dfuture_rebake_v2_dataset_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(
        args.output_root / "target_promotion_audit.md",
        [
            "# rebake_v2 Target Promotion Audit",
            "",
            f"- decision: `{decision}`",
            f"- records: `{len(audited)}`",
            f"- promoted_records: `{len(promoted)}`",
            f"- diagnostic_records: `{len(diagnostic)}`",
            f"- reject_records: `{len(rejects)}`",
            f"- gate_blocker_counts: `{json.dumps(dict(blocker_counts), ensure_ascii=False)}`",
        ],
    )
    write_md(
        args.output_root / "stage1_v4_no3dfuture_rebake_v2_quota_report.md",
        [
            "# rebake_v2 Quota Report",
            "",
            f"- decision: `{decision}`",
            f"- quota_blockers: `{json.dumps(quota_issues, ensure_ascii=False)}`",
            f"- balanced_summary: `{json.dumps(summarize(balanced), ensure_ascii=False)}`",
        ],
    )
    write_md(
        args.output_root / "no3dfuture_rebake_v2_decision.md",
        [
            "# no-3D-FUTURE rebake_v2 Decision",
            "",
            f"- approve_r_training = `{decision == 'READY_FOR_R_TRAINING_CANDIDATE'}`",
            f"- approve_stage1_v4_no3dfuture_as_train_candidate = `{decision == 'READY_FOR_R_TRAINING_CANDIDATE'}`",
            "- approve_paper_claim = `false`",
            f"- decision = `{decision}`",
            f"- target_view_blockers = `{json.dumps(target_view_issues, ensure_ascii=False)}`",
            f"- quota_blockers = `{json.dumps(quota_issues, ensure_ascii=False)}`",
        ],
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
