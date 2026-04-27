#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output/material_refine_v1_fixed/releases/stage1_v1_fixed_rebaked_full_manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output/material_refine_v1_fixed/releases"
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
        description="Release stage1_v1_fixed trainable/paper-candidate manifests from a freshly rebaked v1 manifest.",
    )
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trainable-mean-threshold", type=float, default=0.08)
    parser.add_argument("--trainable-p95-threshold", type=float, default=0.20)
    parser.add_argument("--paper-mean-threshold", type=float, default=0.03)
    parser.add_argument("--paper-p95-threshold", type=float, default=0.08)
    parser.add_argument("--paper-max-target-prior-identity", type=float, default=0.30)
    parser.add_argument("--min-trainable-records", type=int, default=128)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def bool_field(record: dict[str, Any], key: str) -> bool:
    value = record.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "passed"}
    return bool(value)


def path_exists(record: dict[str, Any], key: str) -> bool:
    value = record.get(key)
    return isinstance(value, str) and bool(value) and Path(value).exists()


def numeric(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def paper_split_to_default(value: Any) -> str:
    text = str(value or "").lower()
    if "val" in text:
        return "val"
    if "test" in text or "holdout" in text or "ood" in text:
        return "test"
    return "train"


def object_id(record: dict[str, Any]) -> str:
    return str(record.get("canonical_object_id") or record.get("object_id") or record.get("source_uid") or "")


def stats(values: list[float]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values if value is not None)
    if not finite:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": statistics.mean(finite),
        "p50": finite[int((len(finite) - 1) * 0.50)],
        "p95": finite[int((len(finite) - 1) * 0.95)],
        "min": finite[0],
        "max": finite[-1],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "default_split": dict(Counter(str(record.get("default_split") or "unknown") for record in records)),
        "paper_split": dict(Counter(str(record.get("paper_split") or "unknown") for record in records)),
        "source_name": dict(Counter(str(record.get("source_name") or "unknown") for record in records)),
        "generator_id": dict(Counter(str(record.get("generator_id") or "unknown") for record in records)),
        "material_family": dict(Counter(str(record.get("material_family") or "unknown") for record in records)),
        "license_bucket": dict(Counter(str(record.get("license_bucket") or "unknown") for record in records)),
        "has_material_prior": {
            "true": sum(bool(record.get("has_material_prior")) for record in records),
            "false": sum(not bool(record.get("has_material_prior")) for record in records),
        },
        "target_source_type": dict(Counter(str(record.get("target_source_type") or "unknown") for record in records)),
        "target_quality_tier": dict(Counter(str(record.get("target_quality_tier") or "unknown") for record in records)),
        "target_view_alignment_mean": stats(
            [value for record in records if (value := numeric(record, "target_view_alignment_mean")) is not None]
        ),
        "target_view_alignment_p95": stats(
            [value for record in records if (value := numeric(record, "target_view_alignment_p95")) is not None]
        ),
        "target_prior_identity": stats(
            [value for record in records if (value := numeric(record, "target_prior_identity")) is not None]
        ),
    }


def audit_record(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
    item = dict(record)
    if not item.get("default_split"):
        item["default_split"] = paper_split_to_default(item.get("paper_split"))
    item["paper_split"] = item.get("paper_split") or item["default_split"]
    item["dataset_role"] = "stage1_v1_fixed_candidate"
    train_blockers: list[str] = []
    paper_blockers: list[str] = []

    if str(item.get("rebake_version") or "") != "v1_fixed_rebake":
        train_blockers.append("rebake_version_not_v1_fixed")
    if str(item.get("target_view_contract_version") or "") != "v1_fixed":
        train_blockers.append("target_view_contract_not_v1_fixed")
    if not bool_field(item, "prior_as_pred_pass"):
        train_blockers.append("prior_as_pred_pass_false")
    if not bool_field(item, "target_as_pred_pass"):
        train_blockers.append("target_as_pred_pass_false")
    mean = numeric(item, "target_view_alignment_mean")
    p95 = numeric(item, "target_view_alignment_p95")
    if mean is None:
        train_blockers.append("missing_target_view_alignment_mean")
    elif mean >= float(args.trainable_mean_threshold):
        train_blockers.append(f"target_view_alignment_mean_high:{mean:.5f}")
    if p95 is None:
        train_blockers.append("missing_target_view_alignment_p95")
    elif p95 >= float(args.trainable_p95_threshold):
        train_blockers.append(f"target_view_alignment_p95_high:{p95:.5f}")
    if bool_field(item, "target_is_prior_copy") or bool_field(item, "copied_from_prior"):
        train_blockers.append("target_is_prior_copy")
    if str(item.get("license_bucket") or "unknown") not in ALLOWED_LICENSE_BUCKETS:
        train_blockers.append(f"license_not_allowed:{item.get('license_bucket')}")
    if not any(path_exists(item, key) for key in ("source_model_path", "canonical_glb_path", "canonical_mesh_path")):
        train_blockers.append("missing_raw_or_canonical_asset")
    for key in ("canonical_buffer_root", "uv_target_roughness_path", "uv_target_metallic_path", "uv_target_confidence_path"):
        if not path_exists(item, key):
            train_blockers.append(f"missing_{key}")

    paper_blockers.extend(train_blockers)
    identity = numeric(item, "target_prior_identity")
    confidence_mean = numeric(item, "target_confidence_mean")
    confidence_nonzero = numeric(item, "target_confidence_nonzero_rate")
    if mean is None or mean >= float(args.paper_mean_threshold):
        paper_blockers.append("paper_target_view_alignment_mean_not_strict")
    if p95 is None or p95 >= float(args.paper_p95_threshold):
        paper_blockers.append("paper_target_view_alignment_p95_not_strict")
    if identity is None or identity > float(args.paper_max_target_prior_identity):
        paper_blockers.append("paper_target_prior_identity_high")
    if confidence_mean is None or confidence_mean < 0.70:
        paper_blockers.append("paper_confidence_mean_low")
    if confidence_nonzero is None or confidence_nonzero < 0.50:
        paper_blockers.append("paper_confidence_nonzero_low")

    if not train_blockers:
        item["dataset_role"] = "stage1_v1_fixed_trainable"
        item["supervision_role"] = "paper_main"
        item["target_quality_tier"] = "trainable_pseudo"
        item["paper_stage_eligible"] = False
        item["paper_stage_eligible_v1_fixed"] = False
        item["candidate_pool_only"] = False
    else:
        item["dataset_role"] = "stage1_v1_fixed_diagnostic"
        item["supervision_role"] = "diagnostic_only"
        item["candidate_pool_only"] = True
        item["trainable_gate_blockers"] = train_blockers

    if not paper_blockers:
        item["dataset_role"] = "stage1_v1_fixed_paper_candidate"
        item["target_quality_tier"] = (
            "paper_strong" if str(item.get("target_source_type")) == "gt_render_baked" else "paper_pseudo"
        )
        item["paper_stage_eligible"] = True
        item["paper_stage_eligible_v1_fixed"] = True
        item["candidate_pool_only"] = False
    else:
        item["paper_gate_blockers"] = paper_blockers
    return item, train_blockers, paper_blockers


def write_manifest(path: Path, *, name: str, records: list[dict[str, Any]], source_manifest: Path) -> None:
    payload = {
        "manifest_version": "canonical_asset_record_v1_stage1_v1_fixed_release",
        "generated_at_utc": utc_now(),
        "subset_name": name,
        "source_manifest": str(source_manifest.resolve()),
        "summary": summarize(records),
        "records": records,
    }
    write_json(path, payload)


def write_sampler_config(path: Path, trainable: list[dict[str, Any]]) -> None:
    material_counts = Counter(str(record.get("material_family") or "unknown") for record in trainable)
    source_counts = Counter(str(record.get("source_name") or "unknown") for record in trainable)
    no_prior = sum(not bool(record.get("has_material_prior")) for record in trainable)
    payload = {
        "sampler_version": "stage1_v1_fixed_sampler_v1",
        "notes": [
            "Use for engineering/R-v2.1 validation only; not a paper claim.",
            "Material distribution is not a hard blocker for v1_fixed, but sampler weights compensate for imbalance.",
        ],
        "weighting": {
            "base": 1.0,
            "glossy_non_metal_downsample": 0.50 if material_counts.get("glossy_non_metal", 0) > 0 else 1.0,
            "rare_material_oversample": 2.0,
            "no_prior_oversample": 2.0 if no_prior > 0 else 1.0,
            "source_balance": True,
            "prior_mode_balance": True,
        },
        "material_family_counts": dict(material_counts),
        "source_counts": dict(source_counts),
        "has_material_prior": {
            "true": len(trainable) - no_prior,
            "false": no_prior,
        },
    }
    write_json(path, payload)


def write_decision(
    path: Path,
    *,
    trainable: list[dict[str, Any]],
    paper: list[dict[str, Any]],
    diagnostic: list[dict[str, Any]],
    rejects: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    train_summary = summarize(trainable)
    split_counts = train_summary["default_split"]
    split_nonempty = all(split_counts.get(split, 0) > 0 for split in ("train", "val", "test"))
    enough_records = len(trainable) >= int(args.min_trainable_records)
    approve_train = bool(enough_records and split_nonempty)
    material_family_count = len([key for key, value in train_summary["material_family"].items() if value > 0])
    lines = [
        "# stage1_v1_fixed Decision",
        "",
        f"- v1_fixed_trainable_records = `{len(trainable)}`",
        f"- v1_fixed_paper_candidate_records = `{len(paper)}`",
        f"- diagnostic_records = `{len(diagnostic)}`",
        f"- rejects_records = `{len(rejects)}`",
        f"- train/val/test counts = `{json.dumps(split_counts, ensure_ascii=False)}`",
        f"- target_view_alignment_mean = `{json.dumps(train_summary['target_view_alignment_mean'], ensure_ascii=False)}`",
        f"- target_view_alignment_p95 = `{json.dumps(train_summary['target_view_alignment_p95'], ensure_ascii=False)}`",
        f"- target_prior_identity = `{json.dumps(train_summary['target_prior_identity'], ensure_ascii=False)}`",
        f"- material_family distribution = `{json.dumps(train_summary['material_family'], ensure_ascii=False)}`",
        f"- source distribution = `{json.dumps(train_summary['source_name'], ensure_ascii=False)}`",
        f"- with_prior / without_prior counts = `{json.dumps(train_summary['has_material_prior'], ensure_ascii=False)}`",
        f"- material_family_count = `{material_family_count}`",
        f"- material_family_warning = `{material_family_count < 3}`",
        f"- approve_r_training_on_v1_fixed = `{str(approve_train).lower()}`",
        "- approve_paper_claim = `false`",
        f"- approve_expand_to_longrun_rebake = `{str(approve_train).lower()}`",
        "",
        "## Notes",
        "",
        "- `approve_r_training_on_v1_fixed=true` only means a later manual R-v2.1 launch is allowed; this script never starts training.",
        "- Material-family imbalance is reported and handled by sampler config; it is not a hard blocker for this v1_fixed engineering release.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = read_json(args.input_manifest)
    records = [record for record in payload.get("records", []) if isinstance(record, dict)]

    trainable: list[dict[str, Any]] = []
    paper: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    gate_counter: Counter[str] = Counter()
    seen: set[str] = set()
    for record in records:
        key = object_id(record)
        if not key or key in seen:
            continue
        seen.add(key)
        item, train_blockers, paper_blockers = audit_record(record, args)
        for blocker in train_blockers:
            gate_counter[blocker.split(":", 1)[0]] += 1
        if not train_blockers:
            trainable.append(item)
            if not paper_blockers:
                paper.append(item)
            else:
                diagnostic.append(item)
        else:
            if any(blocker.startswith(("missing_raw", "license_not_allowed")) for blocker in train_blockers):
                rejects.append(item)
            else:
                diagnostic.append(item)

    write_manifest(
        args.output_root / "stage1_v1_fixed_trainable_manifest.json",
        name="stage1_v1_fixed_trainable",
        records=trainable,
        source_manifest=args.input_manifest,
    )
    write_manifest(
        args.output_root / "stage1_v1_fixed_paper_candidate_manifest.json",
        name="stage1_v1_fixed_paper_candidate",
        records=paper,
        source_manifest=args.input_manifest,
    )
    write_manifest(
        args.output_root / "stage1_v1_fixed_diagnostic_manifest.json",
        name="stage1_v1_fixed_diagnostic",
        records=diagnostic,
        source_manifest=args.input_manifest,
    )
    write_manifest(
        args.output_root / "stage1_v1_fixed_rejects_manifest.json",
        name="stage1_v1_fixed_rejects",
        records=rejects,
        source_manifest=args.input_manifest,
    )
    write_sampler_config(args.output_root / "stage1_v1_fixed_sampler_config.json", trainable)
    write_json(
        args.output_root / "stage1_v1_fixed_gate_audit.json",
        {
            "generated_at_utc": utc_now(),
            "input_manifest": str(args.input_manifest.resolve()),
            "gate_blocker_counts": dict(gate_counter),
            "trainable_summary": summarize(trainable),
            "paper_candidate_summary": summarize(paper),
            "diagnostic_summary": summarize(diagnostic),
            "rejects_summary": summarize(rejects),
        },
    )
    write_decision(
        REPO_ROOT / "output/material_refine_v1_fixed/stage1_v1_fixed_decision.md",
        trainable=trainable,
        paper=paper,
        diagnostic=diagnostic,
        rejects=rejects,
        args=args,
    )
    print(
        json.dumps(
            {
                "trainable_records": len(trainable),
                "paper_candidate_records": len(paper),
                "diagnostic_records": len(diagnostic),
                "rejects_records": len(rejects),
                "decision": str(REPO_ROOT / "output/material_refine_v1_fixed/stage1_v1_fixed_decision.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
