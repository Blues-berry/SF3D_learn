#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output/material_refine_v1_fixed/releases/stage1_v1_fixed_trainable_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "train/trainV5_initial"
DEFAULT_MIN_FREE_GB = 20.0

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
        description="Build lightweight TrainV5 initial manifests from the A-track v1_fixed trainable release.",
    )
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ssd-active-root", type=Path, default=REPO_ROOT / "dataoutput")
    parser.add_argument("--min-ssd-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--mean-threshold", type=float, default=0.08)
    parser.add_argument("--p95-threshold", type=float, default=0.20)
    parser.add_argument("--near-gt-threshold", type=float, default=0.85)
    parser.add_argument("--mild-gap-threshold", type=float, default=0.60)
    parser.add_argument("--medium-gap-threshold", type=float, default=0.30)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
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


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def object_id(record: dict[str, Any]) -> str:
    return str(record.get("canonical_object_id") or record.get("object_id") or record.get("source_uid") or "")


def normalize_split(record: dict[str, Any]) -> str:
    value = str(record.get("default_split") or record.get("paper_split") or "train").lower()
    if "val" in value:
        return "val"
    if "test" in value or "holdout" in value or "ood" in value:
        return "test"
    return "train"


def path_or_empty(value: Any) -> str:
    return str(value) if isinstance(value, str) and value else ""


def path_exists(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).exists()


def resolved(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        return str(Path(value).resolve())
    except OSError:
        return str(value)


def repo_logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    path = Path(value)
    try:
        absolute = path.resolve()
        return str(absolute.relative_to(REPO_ROOT))
    except (OSError, ValueError):
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
        return "ssd_project"
    return "external_or_unknown"


def storage_ref(value: Any) -> dict[str, Any]:
    return {
        "logical_path": repo_logical_path(value),
        "physical_path": resolved(value),
        "storage_tier": storage_tier(value),
        "path_resolved_ok": path_exists(value),
    }


def sha256_file(value: Any) -> str | None:
    if not path_exists(value):
        return None
    digest = hashlib.sha256()
    path = Path(str(value))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def prior_quality_bin(similarity: float | None, has_prior: bool, args: argparse.Namespace) -> str:
    if not has_prior:
        return "no_prior"
    if similarity is None:
        return "unknown"
    if similarity >= float(args.near_gt_threshold):
        return "near_gt"
    if similarity >= float(args.mild_gap_threshold):
        return "mild_gap"
    if similarity >= float(args.medium_gap_threshold):
        return "medium_gap"
    return "large_gap"


def target_gate(record: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not bool_field(record, "target_as_pred_pass"):
        blockers.append("target_as_pred_fail")
    mean = finite_float(record.get("target_view_alignment_mean"))
    p95 = finite_float(record.get("target_view_alignment_p95"))
    if mean is None or mean >= float(args.mean_threshold):
        blockers.append("target_view_alignment_mean_fail")
    if p95 is None or p95 >= float(args.p95_threshold):
        blockers.append("target_view_alignment_p95_fail")
    if bool_field(record, "target_is_prior_copy") or bool_field(record, "copied_from_prior"):
        blockers.append("target_is_prior_copy")
    if str(record.get("license_bucket") or "unknown") not in ALLOWED_LICENSE_BUCKETS:
        blockers.append("license_blocked")
    if not any(path_exists(record.get(key)) for key in ("source_model_path", "canonical_glb_path", "canonical_mesh_path")):
        blockers.append("missing_asset")
    for key in ("canonical_buffer_root", "uv_target_roughness_path", "uv_target_metallic_path", "uv_target_confidence_path"):
        if not path_exists(record.get(key)):
            blockers.append(f"missing_{key}")
    return not blockers, blockers


def prior_spatiality(record: dict[str, Any], roughness_prior: str, metallic_prior: str) -> str:
    if roughness_prior and metallic_prior:
        if str(record.get("prior_mode") or "").lower() == "scalar_rm":
            return "scalar_broadcast"
        return "spatial_map"
    if finite_float(record.get("scalar_prior_roughness")) is not None and finite_float(record.get("scalar_prior_metallic")) is not None:
        return "scalar_broadcast"
    return "no_prior"


def leakage_audit(record: dict[str, Any], spatiality: str) -> dict[str, Any]:
    roughness_prior = path_or_empty(record.get("uv_prior_roughness_path"))
    metallic_prior = path_or_empty(record.get("uv_prior_metallic_path"))
    roughness_target = path_or_empty(record.get("uv_target_roughness_path"))
    metallic_target = path_or_empty(record.get("uv_target_metallic_path"))
    payload: dict[str, Any] = {
        "prior_spatiality": spatiality,
        "prior_not_target_leakage": True,
        "leakage_reason": "",
        "roughness_prior_hash": "",
        "roughness_target_hash": "",
        "metallic_prior_hash": "",
        "metallic_target_hash": "",
        "roughness_path_same_as_target": False,
        "metallic_path_same_as_target": False,
        "roughness_hash_same_as_target": False,
        "metallic_hash_same_as_target": False,
    }
    if spatiality == "no_prior":
        payload["leakage_audit_method"] = "skipped_no_prior"
        return payload

    for channel, prior_path, target_path in (
        ("roughness", roughness_prior, roughness_target),
        ("metallic", metallic_prior, metallic_target),
    ):
        prior_resolved = resolved(prior_path)
        target_resolved = resolved(target_path)
        path_same = bool(prior_resolved and target_resolved and prior_resolved == target_resolved)
        prior_hash = sha256_file(prior_path)
        target_hash = sha256_file(target_path)
        hash_same = bool(prior_hash and target_hash and prior_hash == target_hash)
        payload[f"{channel}_prior_hash"] = prior_hash or ""
        payload[f"{channel}_target_hash"] = target_hash or ""
        payload[f"{channel}_path_same_as_target"] = path_same
        payload[f"{channel}_hash_same_as_target"] = hash_same
        if path_same or hash_same:
            payload["prior_not_target_leakage"] = False
            payload["leakage_reason"] = f"{channel}_prior_matches_target"
    payload["leakage_audit_method"] = "path_and_hash"
    return payload


def build_target_bundle(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    oid = object_id(record)
    gate_pass, blockers = target_gate(record, args)
    bundle_id = f"tb_{oid}"
    paths = {
        "canonical_bundle_root": storage_ref(record.get("canonical_bundle_root")),
        "canonical_buffer_root": storage_ref(record.get("canonical_buffer_root")),
        "uv_target_roughness_path": storage_ref(record.get("uv_target_roughness_path")),
        "uv_target_metallic_path": storage_ref(record.get("uv_target_metallic_path")),
        "uv_target_confidence_path": storage_ref(record.get("uv_target_confidence_path")),
        "source_model_path": storage_ref(record.get("source_model_path")),
        "canonical_glb_path": storage_ref(record.get("canonical_glb_path")),
    }
    return {
        "target_bundle_id": bundle_id,
        "object_id": oid,
        "source_name": str(record.get("source_name") or ""),
        "source_dataset": str(record.get("source_dataset") or ""),
        "generator_id": str(record.get("generator_id") or ""),
        "license_bucket": str(record.get("license_bucket") or ""),
        "material_family": str(record.get("material_family") or "unknown"),
        "split": normalize_split(record),
        "default_split": str(record.get("default_split") or normalize_split(record)),
        "paper_split": str(record.get("paper_split") or ""),
        "target_source_type": str(record.get("target_source_type") or ""),
        "target_quality_tier": str(record.get("target_quality_tier") or ""),
        "target_as_pred_pass": bool_field(record, "target_as_pred_pass"),
        "target_view_alignment_mean": finite_float(record.get("target_view_alignment_mean")),
        "target_view_alignment_p95": finite_float(record.get("target_view_alignment_p95")),
        "target_is_prior_copy": bool_field(record, "target_is_prior_copy") or bool_field(record, "copied_from_prior"),
        "target_gate_pass": gate_pass,
        "target_gate_blockers": blockers,
        "storage_tier": paths["canonical_bundle_root"]["storage_tier"],
        "logical_path": paths["canonical_bundle_root"]["logical_path"],
        "physical_path": paths["canonical_bundle_root"]["physical_path"],
        "path_resolved_ok": all(
            paths[key]["path_resolved_ok"]
            for key in ("canonical_bundle_root", "canonical_buffer_root", "uv_target_roughness_path", "uv_target_metallic_path", "uv_target_confidence_path")
        ),
        "storage_paths": paths,
        "source_record": record,
    }


def build_prior_variant(record: dict[str, Any], target_bundle_id: str, args: argparse.Namespace) -> dict[str, Any]:
    oid = object_id(record)
    roughness_prior = path_or_empty(record.get("uv_prior_roughness_path"))
    metallic_prior = path_or_empty(record.get("uv_prior_metallic_path"))
    spatiality = prior_spatiality(record, roughness_prior, metallic_prior)
    leakage = leakage_audit(record, spatiality)
    has_prior = spatiality != "no_prior"
    similarity = finite_float(record.get("target_prior_identity"))
    gap = None if similarity is None else 1.0 - similarity
    if spatiality == "no_prior":
        variant_type = "no_prior_bootstrap"
        training_role = "bootstrap_recovery"
        sample_weight = 1.0
    elif leakage["prior_not_target_leakage"]:
        variant_type = "a_track_existing_pipeline_prior"
        training_role = "safe_refinement"
        sample_weight = 1.0
    else:
        variant_type = "identity_control"
        training_role = "no_overedit_regularization"
        sample_weight = 0.10

    prior_paths = {
        "uv_prior_roughness_path": storage_ref(roughness_prior),
        "uv_prior_metallic_path": storage_ref(metallic_prior),
    }
    prior_path_resolved_ok = (
        True
        if spatiality == "no_prior"
        else bool(prior_paths["uv_prior_roughness_path"]["path_resolved_ok"] and prior_paths["uv_prior_metallic_path"]["path_resolved_ok"])
    )
    prior_as_pred_required = spatiality in {"spatial_map", "scalar_broadcast"}
    prior_as_pred_pass = bool_field(record, "prior_as_pred_pass") if prior_as_pred_required else None
    prior_variant_gate_pass = bool(
        prior_path_resolved_ok
        and bool(leakage["prior_not_target_leakage"])
        and (prior_as_pred_pass is not False)
        and variant_type != "identity_control"
    )
    return {
        "prior_variant_id": f"pv_{oid}_existing_pipeline_prior",
        "target_bundle_id": target_bundle_id,
        "object_id": oid,
        "split": normalize_split(record),
        "prior_source_type": "existing_pipeline_prior",
        "upstream_model_id": "none_or_existing_pipeline",
        "prior_variant_type": variant_type,
        "training_role": training_role,
        "sample_weight": sample_weight,
        "prior_spatiality": spatiality,
        "prior_mode": str(record.get("prior_mode") or ""),
        "has_prior": has_prior,
        "prior_quality_bin": prior_quality_bin(similarity, has_prior, args),
        "prior_target_similarity": similarity,
        "prior_target_gap": gap,
        "scalar_prior_roughness": finite_float(record.get("scalar_prior_roughness")),
        "scalar_prior_metallic": finite_float(record.get("scalar_prior_metallic")),
        "uv_prior_roughness_path": roughness_prior,
        "uv_prior_metallic_path": metallic_prior,
        "prior_path_resolved_ok": prior_path_resolved_ok,
        "prior_as_pred_required": prior_as_pred_required,
        "prior_as_pred_pass": prior_as_pred_pass,
        "prior_variant_gate_pass": prior_variant_gate_pass,
        "prior_not_target_leakage": bool(leakage["prior_not_target_leakage"]),
        "leakage_audit": leakage,
        "storage_tier": prior_paths["uv_prior_roughness_path"]["storage_tier"] if has_prior else "none",
        "logical_path": prior_paths["uv_prior_roughness_path"]["logical_path"] if has_prior else "",
        "physical_path": prior_paths["uv_prior_roughness_path"]["physical_path"] if has_prior else "",
        "path_resolved_ok": prior_path_resolved_ok,
        "storage_paths": prior_paths,
    }


def build_training_pair(bundle: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any] | None:
    if not bundle["target_gate_pass"]:
        return None
    if not variant["prior_variant_gate_pass"]:
        return None
    if variant["prior_variant_type"] == "identity_control":
        return None
    pair_id = f"pair_{bundle['target_bundle_id']}__{variant['prior_variant_id']}"
    return {
        "training_pair_id": pair_id,
        "target_bundle_id": bundle["target_bundle_id"],
        "prior_variant_id": variant["prior_variant_id"],
        "object_id": bundle["object_id"],
        "split": bundle["split"],
        "source_name": bundle["source_name"],
        "material_family": bundle["material_family"],
        "training_role": variant["training_role"],
        "sample_weight": variant["sample_weight"],
        "prior_quality_bin": variant["prior_quality_bin"],
        "prior_spatiality": variant["prior_spatiality"],
        "prior_source_type": variant["prior_source_type"],
        "upstream_model_id": variant["upstream_model_id"],
        "uv_prior_roughness_path": variant["uv_prior_roughness_path"],
        "uv_prior_metallic_path": variant["uv_prior_metallic_path"],
        "uv_target_roughness_path": bundle["source_record"].get("uv_target_roughness_path"),
        "uv_target_metallic_path": bundle["source_record"].get("uv_target_metallic_path"),
        "uv_target_confidence_path": bundle["source_record"].get("uv_target_confidence_path"),
        "canonical_buffer_root": bundle["source_record"].get("canonical_buffer_root"),
        "path_resolved_ok": bool(bundle["path_resolved_ok"] and variant["path_resolved_ok"]),
        "storage_tier": bundle["storage_tier"],
        "logical_path": bundle["logical_path"],
        "physical_path": bundle["physical_path"],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "split": dict(Counter(str(record.get("split") or "unknown") for record in records)),
        "source_name": dict(Counter(str(record.get("source_name") or "unknown") for record in records)),
        "material_family": dict(Counter(str(record.get("material_family") or "unknown") for record in records)),
        "prior_quality_bin": dict(Counter(str(record.get("prior_quality_bin") or "unknown") for record in records)),
        "prior_spatiality": dict(Counter(str(record.get("prior_spatiality") or "unknown") for record in records)),
        "training_role": dict(Counter(str(record.get("training_role") or "unknown") for record in records)),
    }


def split_leakage_ok(pairs: list[dict[str, Any]]) -> tuple[bool, dict[str, list[str]]]:
    object_splits: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        object_splits[str(pair.get("object_id"))].add(str(pair.get("split")))
    offenders = {key: sorted(value) for key, value in object_splits.items() if len(value) > 1}
    return not offenders, offenders


def free_gb(path: Path) -> float | None:
    try:
        usage = os.statvfs(path)
    except OSError:
        return None
    return float(usage.f_bavail * usage.f_frsize / (1024**3))


def write_inventory(
    path: Path,
    *,
    bundles: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> None:
    lines = [
        "# TrainV5 Initial Inventory",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- target_bundles: `{len(bundles)}`",
        f"- prior_variants: `{len(variants)}`",
        f"- training_pairs: `{len(pairs)}`",
        f"- split: `{json.dumps(summarize(pairs)['split'], ensure_ascii=False)}`",
        f"- material_family: `{json.dumps(summarize(pairs)['material_family'], ensure_ascii=False)}`",
        f"- prior_quality_bin: `{json.dumps(summarize(pairs)['prior_quality_bin'], ensure_ascii=False)}`",
        f"- prior_spatiality: `{json.dumps(summarize(pairs)['prior_spatiality'], ensure_ascii=False)}`",
        f"- sanity_ready: `{str(readiness['sanity_ready']).lower()}`",
        "",
        "Large view/UV/buffer images are not copied into `train/trainV5_initial`; manifests keep logical/physical paths to `output` or `/4T`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readiness(path: Path, readiness: dict[str, Any]) -> None:
    lines = [
        "# TrainV5 Initial Readiness",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- sanity_ready: `{str(readiness['sanity_ready']).lower()}`",
        f"- target_bundles: `{readiness['target_bundles']}`",
        f"- training_pairs: `{readiness['training_pairs']}`",
        f"- split_nonempty: `{readiness['split_nonempty']}`",
        f"- all_paths_resolved: `{readiness['all_paths_resolved']}`",
        f"- prior_target_leakage_audit_pass: `{readiness['prior_target_leakage_audit_pass']}`",
        f"- object_level_split_check_pass: `{readiness['object_level_split_check_pass']}`",
        f"- at_least_one_prior_variant_per_target_bundle: `{readiness['at_least_one_prior_variant_per_target_bundle']}`",
        f"- ssd_active_free_gb: `{readiness['ssd_active_free_gb']}`",
        f"- blockers: `{json.dumps(readiness['blockers'], ensure_ascii=False)}`",
        "",
        "No R training is launched by this builder. The command draft requires manual review.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_command_draft(path: Path) -> None:
    text = """#!/usr/bin/env bash
set -euo pipefail

# Manual draft only. Review manifests/readiness before removing this guard.
echo "TrainV5 R-v2.1 sanity command draft only; not auto-starting training."
exit 1

# Example draft, adjust config/CLI after human review:
# cd /home/ubuntu/ssd_work/projects/stable-fast-3d
# /home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/train_material_refiner.py \\
#   --config configs/material_refine_train_r_v2_1_view_aware.yaml \\
#   --train-manifest train/trainV5_initial/trainV5_training_pairs.json
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ssd_active_root.mkdir(parents=True, exist_ok=True)
    payload = read_json(args.input_manifest)
    records = payload["records"] if isinstance(payload, dict) else payload
    records = [record for record in records if isinstance(record, dict)]

    bundles: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    identity_controls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        oid = object_id(record)
        if not oid or oid in seen:
            continue
        seen.add(oid)
        bundle = build_target_bundle(record, args)
        variant = build_prior_variant(record, bundle["target_bundle_id"], args)
        pair = build_training_pair(bundle, variant)
        bundles.append(bundle)
        variants.append(variant)
        if variant["prior_variant_type"] == "identity_control":
            identity_controls.append(variant)
        if pair is not None:
            pairs.append(pair)

    split_ok, split_offenders = split_leakage_ok(pairs)
    pair_counts_by_bundle = Counter(str(pair["target_bundle_id"]) for pair in pairs)
    split_counts = Counter(str(pair.get("split") or "unknown") for pair in pairs)
    ssd_free = free_gb(args.ssd_active_root)
    readiness = {
        "sanity_ready": False,
        "target_bundles": len(bundles),
        "prior_variants": len(variants),
        "training_pairs": len(pairs),
        "split_nonempty": all(split_counts.get(split, 0) > 0 for split in ("train", "val", "test")),
        "all_paths_resolved": all(bool(pair.get("path_resolved_ok")) for pair in pairs),
        "prior_target_leakage_audit_pass": all(bool(variant.get("prior_not_target_leakage")) for variant in variants if variant.get("prior_variant_type") != "identity_control"),
        "object_level_split_check_pass": split_ok,
        "split_leakage_offenders": split_offenders,
        "at_least_one_prior_variant_per_target_bundle": all(pair_counts_by_bundle.get(str(bundle["target_bundle_id"]), 0) >= 1 for bundle in bundles),
        "ssd_active_root": str(args.ssd_active_root.resolve()),
        "ssd_active_free_gb": ssd_free,
        "ssd_min_free_gb": float(args.min_ssd_free_gb),
        "identity_control_variants": len(identity_controls),
        "blockers": [],
    }
    if len(bundles) < 300:
        readiness["blockers"].append("target_bundles_below_300")
    if len(pairs) < 300:
        readiness["blockers"].append("training_pairs_below_300")
    for key in (
        "split_nonempty",
        "all_paths_resolved",
        "prior_target_leakage_audit_pass",
        "object_level_split_check_pass",
        "at_least_one_prior_variant_per_target_bundle",
    ):
        if not readiness[key]:
            readiness["blockers"].append(key)
    if ssd_free is not None and ssd_free < float(args.min_ssd_free_gb):
        readiness["blockers"].append("ssd_active_free_space_below_threshold")
    readiness["sanity_ready"] = not readiness["blockers"]

    manifest_base = {
        "generated_at_utc": utc_now(),
        "source_manifest": str(args.input_manifest.resolve()),
        "data_contract": "trainV5_initial_r_only_v1",
        "notes": [
            "R-only initial data. No online SF3D/SPAR3D/Hunyuan3D priors are generated.",
            "Large images are referenced through logical/physical paths and are not copied into train/trainV5_initial.",
        ],
    }
    write_json(args.output_dir / "trainV5_target_bundles.json", {**manifest_base, "records": bundles, "summary": summarize(bundles)})
    write_json(args.output_dir / "trainV5_prior_variants.json", {**manifest_base, "records": variants, "summary": summarize(variants)})
    write_json(args.output_dir / "trainV5_training_pairs.json", {**manifest_base, "records": pairs, "summary": summarize(pairs)})
    write_json(
        args.output_dir / "trainV5_sampler_config.json",
        {
            "generated_at_utc": utc_now(),
            "sampler_version": "trainV5_initial_sampler_v1",
            "weights": {
                "safe_refinement": 1.0,
                "identity_control": 0.10,
                "rare_material_oversample": 2.0,
                "no_prior_oversample": 2.0,
                "scalar_broadcast_oversample": 1.25,
            },
            "summary": summarize(pairs),
        },
    )
    write_json(args.output_dir / "trainV5_readiness_report.json", readiness)
    write_inventory(args.output_dir / "trainV5_inventory.md", bundles=bundles, variants=variants, pairs=pairs, readiness=readiness)
    write_readiness(args.output_dir / "trainV5_readiness_report.md", readiness)
    write_command_draft(args.output_dir / "r_v2_1_sanity_command_draft.sh")
    write_json(
        args.output_dir / "trainV5_lightweight_path_index.json",
        {
            "generated_at_utc": utc_now(),
            "target_bundle_paths": {
                bundle["target_bundle_id"]: bundle["storage_paths"]
                for bundle in bundles
            },
            "prior_variant_paths": {
                variant["prior_variant_id"]: variant["storage_paths"]
                for variant in variants
            },
        },
    )
    print(
        json.dumps(
            {
                "target_bundles": len(bundles),
                "prior_variants": len(variants),
                "training_pairs": len(pairs),
                "identity_control_variants": len(identity_controls),
                "sanity_ready": readiness["sanity_ready"],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
