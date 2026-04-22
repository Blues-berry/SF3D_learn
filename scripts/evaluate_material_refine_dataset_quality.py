#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.manifest_quality import audit_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate dataset-production quality, split safety, view/light coverage, and paper readiness.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-paper-eligible", type=int, default=128)
    parser.add_argument("--min-view-ready-rate", type=float, default=0.8)
    parser.add_argument("--min-strict-complete-rate", type=float, default=0.8)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key) or "unknown") for record in records))


def find_duplicate_object_splits(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    splits_by_object: dict[str, set[str]] = {}
    for record in records:
        object_id = str(record.get("object_id") or "")
        if not object_id:
            continue
        split = str(record.get("paper_split") or record.get("default_split") or "unknown")
        splits_by_object.setdefault(object_id, set()).add(split)
    return {
        object_id: sorted(splits)
        for object_id, splits in splits_by_object.items()
        if len(splits) > 1
    }


def summarize_view_light(records: list[dict[str, Any]]) -> dict[str, Any]:
    view_counts = []
    valid_view_counts = []
    hdri_asset_counts = []
    hdri_assets = Counter()
    for record in records:
        view_count = record.get("view_count")
        if view_count not in {None, ""}:
            view_counts.append(int(view_count))
        valid_view_counts.append(int(record.get("valid_view_count") or 0))
        asset_ids = record.get("hdri_asset_ids") or []
        if isinstance(asset_ids, list):
            hdri_asset_counts.append(len(asset_ids))
            hdri_assets.update(str(item) for item in asset_ids)
    return {
        "view_count_min": min(view_counts) if view_counts else 0,
        "view_count_mean": statistics.mean(view_counts) if view_counts else 0.0,
        "view_count_max": max(view_counts) if view_counts else 0,
        "valid_view_count_min": min(valid_view_counts) if valid_view_counts else 0,
        "valid_view_count_mean": statistics.mean(valid_view_counts) if valid_view_counts else 0.0,
        "valid_view_count_max": max(valid_view_counts) if valid_view_counts else 0,
        "hdri_assets_per_object_mean": statistics.mean(hdri_asset_counts) if hdri_asset_counts else 0.0,
        "unique_hdri_assets": len(hdri_assets),
        "top_hdri_assets": dict(hdri_assets.most_common(20)),
        "lighting_bank_id": distribution(records, "lighting_bank_id"),
        "view_light_protocol": distribution(records, "view_light_protocol"),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = audit_manifest(
        args.manifest,
        max_records=-1,
        min_nontrivial_target_count_for_paper=args.min_paper_eligible,
    )
    manifest_payload = load_json(args.manifest)
    records = manifest_payload.get("records") or []
    duplicate_object_splits = find_duplicate_object_splits(records)
    audit_summary = audit_payload["summary"]
    blockers = list(audit_summary.get("readiness_blockers") or [])
    if duplicate_object_splits:
        blockers.append(f"object_split_leakage={len(duplicate_object_splits)}")
    if float(audit_summary.get("view_supervision_ready_rate", 0.0)) < float(args.min_view_ready_rate):
        blockers.append(
            f"view_supervision_ready_rate={float(audit_summary.get('view_supervision_ready_rate', 0.0)):.3f}"
            f" below {float(args.min_view_ready_rate):.3f}"
        )
    if float(audit_summary.get("strict_complete_record_rate", 0.0)) < float(args.min_strict_complete_rate):
        blockers.append(
            f"strict_complete_record_rate={float(audit_summary.get('strict_complete_record_rate', 0.0)):.3f}"
            f" below {float(args.min_strict_complete_rate):.3f}"
        )

    quality = {
        "manifest": str(args.manifest.resolve()),
        "records": len(records),
        "paper_stage_ready": bool(audit_summary.get("paper_stage_ready")) and not duplicate_object_splits,
        "readiness_blockers": blockers,
        "audit_summary": audit_summary,
        "object_split_leakage_count": len(duplicate_object_splits),
        "object_split_leakage_examples": dict(list(duplicate_object_splits.items())[:20]),
        "source_name": distribution(records, "source_name"),
        "supervision_role": distribution(records, "supervision_role"),
        "paper_split": distribution(records, "paper_split"),
        "default_split": distribution(records, "default_split"),
        "license_bucket": distribution(records, "license_bucket"),
        "material_family": distribution(records, "material_family"),
        "target_quality_tier": distribution(records, "target_quality_tier"),
        "target_source_type": distribution(records, "target_source_type"),
        "view_light": summarize_view_light(records),
    }
    write_json(args.output_dir / "dataset_quality_summary.json", quality)
    write_json(args.output_dir / "manifest_audit_summary.json", audit_payload)
    lines = [
        "# Material Refine Dataset Quality",
        "",
        f"- manifest: `{quality['manifest']}`",
        f"- records: `{quality['records']}`",
        f"- paper_stage_ready: `{quality['paper_stage_ready']}`",
        f"- paper_stage_eligible_records: `{audit_summary.get('paper_stage_eligible_records')}`",
        f"- target_prior_identity_rate: `{float(audit_summary.get('target_prior_identity_rate', 0.0)):.4f}`",
        f"- view_supervision_ready_rate: `{float(audit_summary.get('view_supervision_ready_rate', 0.0)):.4f}`",
        f"- strict_complete_record_rate: `{float(audit_summary.get('strict_complete_record_rate', 0.0)):.4f}`",
        f"- unique_hdri_assets: `{quality['view_light']['unique_hdri_assets']}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in blockers or ["none"]:
        lines.append(f"- {blocker}")
    (args.output_dir / "dataset_quality_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.output_dir / "dataset_quality_summary.json"), "blockers": blockers}, indent=2))


if __name__ == "__main__":
    main()
