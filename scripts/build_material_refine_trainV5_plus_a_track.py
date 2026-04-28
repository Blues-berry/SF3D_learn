#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.trainv5_target_gate import (  # noqa: E402
    TARGET_GATE_VERSION,
    target_prior_relation_diagnostics,
    trainv5_target_truth_gate,
)

DEFAULT_INITIAL_DIR = REPO_ROOT / "train/trainV5_initial"
DEFAULT_FALLBACK_MANIFEST = REPO_ROOT / "output/material_refine_v1_fixed/releases/stage1_v1_fixed_trainable_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "train/trainV5_plus_a_track"

VARIANT_SPECS = {
    "near_gt_prior": {
        "quality": "near_gt",
        "role": "safe_refinement",
        "similarity": 0.90,
        "weight": 1.0,
        "recipe": "existing_pipeline_prior_or_synthetic_near_gt_v1",
    },
    "mild_gap_prior": {
        "quality": "mild_gap",
        "role": "mild_correction",
        "similarity": 0.72,
        "weight": 1.0,
        "recipe": "synthetic_degradation_v1:blur9_bias0.065",
    },
    "medium_gap_prior": {
        "quality": "medium_gap",
        "role": "main_refinement",
        "similarity": 0.45,
        "weight": 1.0,
        "recipe": "synthetic_degradation_v1:low_frequency_collapse_bias0.14",
    },
    "large_gap_prior": {
        "quality": "large_gap",
        "role": "strong_correction",
        "similarity": 0.18,
        "weight": 1.0,
        "recipe": "synthetic_degradation_v1:scalarization_low_frequency_bias0.28",
    },
    "no_prior_bootstrap": {
        "quality": "no_prior",
        "role": "bootstrap_refinement",
        "similarity": 0.0,
        "weight": 1.0,
        "recipe": "no_prior_bootstrap_v1",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build TrainV5 Plus A-track pair-based manifests with five prior variants per target.",
    )
    parser.add_argument("--initial-dir", type=Path, default=DEFAULT_INITIAL_DIR)
    parser.add_argument("--fallback-manifest", type=Path, default=DEFAULT_FALLBACK_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-targets", type=int, default=322)
    parser.add_argument("--min-ssd-free-gb", type=float, default=20.0)
    parser.add_argument("--ssd-active-root", type=Path, default=REPO_ROOT / "dataoutput")
    return parser.parse_args()


def record_list(payload: Any) -> list[dict[str, Any]]:
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [record for record in records if isinstance(record, dict)]


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def path_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
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
    try:
        return str(Path(value).resolve().relative_to(REPO_ROOT))
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
        return "ssd_project_or_output_symlink"
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
    with Path(str(value)).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_id(record: dict[str, Any]) -> str:
    return str(record.get("object_id") or record.get("canonical_object_id") or record.get("source_uid") or "")


def normalize_split(record: dict[str, Any]) -> str:
    value = str(record.get("split") or record.get("default_split") or record.get("paper_split") or "train").lower()
    if "val" in value:
        return "val"
    if "test" in value or "holdout" in value or "ood" in value:
        return "test"
    return "train"


def load_source_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    target_path = args.initial_dir / "trainV5_target_bundles.json"
    if target_path.exists():
        bundles = record_list(read_json(target_path))
        out = []
        for bundle in bundles:
            source = dict(bundle.get("source_record") or {})
            merged = {**source, **{k: v for k, v in bundle.items() if k != "source_record"}}
            for key, value in source.items():
                merged.setdefault(key, value)
            out.append(merged)
        return out, str(target_path.resolve())
    return record_list(read_json(args.fallback_manifest)), str(args.fallback_manifest.resolve())


def target_gate(record: dict[str, Any]) -> tuple[bool, list[str]]:
    return trainv5_target_truth_gate(record)


def leakage_audit(prior_roughness: str, prior_metallic: str, target_roughness: str, target_metallic: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prior_not_target_leakage": True,
        "leakage_reason": "",
        "roughness_path_same_as_target": False,
        "metallic_path_same_as_target": False,
        "roughness_hash_same_as_target": False,
        "metallic_hash_same_as_target": False,
    }
    for channel, prior_path, target_path in (
        ("roughness", prior_roughness, target_roughness),
        ("metallic", prior_metallic, target_metallic),
    ):
        prior_resolved = resolved(prior_path)
        target_resolved = resolved(target_path)
        path_same = bool(prior_resolved and target_resolved and prior_resolved == target_resolved)
        prior_hash = sha256_file(prior_path)
        target_hash = sha256_file(target_path)
        hash_same = bool(prior_hash and target_hash and prior_hash == target_hash)
        out[f"{channel}_path_same_as_target"] = path_same
        out[f"{channel}_hash_same_as_target"] = hash_same
        if path_same or hash_same:
            out["prior_not_target_leakage"] = False
            out["leakage_reason"] = f"{channel}_prior_matches_target"
    return out


def build_target_bundle(record: dict[str, Any]) -> dict[str, Any]:
    oid = object_id(record)
    bundle_root = (
        path_text(record, "canonical_bundle_root")
        or str(Path(path_text(record, "uv_target_roughness_path")).parents[1])
    )
    gate_pass, gate_blockers = target_gate(record)
    paths = {
        "canonical_bundle_root": storage_ref(bundle_root),
        "canonical_buffer_root": storage_ref(record.get("canonical_buffer_root")),
        "uv_albedo_path": storage_ref(record.get("uv_albedo_path")),
        "uv_normal_path": storage_ref(record.get("uv_normal_path")),
        "uv_target_roughness_path": storage_ref(record.get("uv_target_roughness_path")),
        "uv_target_metallic_path": storage_ref(record.get("uv_target_metallic_path")),
        "uv_target_confidence_path": storage_ref(record.get("uv_target_confidence_path")),
        "canonical_views_json": storage_ref(record.get("canonical_views_json")),
        "source_model_path": storage_ref(record.get("source_model_path")),
        "canonical_glb_path": storage_ref(record.get("canonical_glb_path")),
    }
    return {
        "target_bundle_id": str(record.get("target_bundle_id") or f"tb_{oid}"),
        "object_id": oid,
        "source_name": str(record.get("source_name") or ""),
        "source_dataset": str(record.get("source_dataset") or ""),
        "generator_id": str(record.get("generator_id") or ""),
        "license_bucket": str(record.get("license_bucket") or ""),
        "material_family": str(record.get("material_family") or "unknown"),
        "split": normalize_split(record),
        "default_split": normalize_split(record),
        "paper_split": str(record.get("paper_split") or ""),
        "target_source_type": str(record.get("target_source_type") or "rebaked_target"),
        "target_quality_tier": str(record.get("target_quality_tier") or "v1_fixed_trainable"),
        "target_as_pred_pass": bool_value(record.get("target_as_pred_pass", True)),
        "target_view_alignment_mean": finite_float(record.get("target_view_alignment_mean")),
        "target_view_alignment_p95": finite_float(record.get("target_view_alignment_p95")),
        "target_is_prior_copy": bool_value(record.get("target_is_prior_copy")) or bool_value(record.get("copied_from_prior")),
        "target_gate_version": TARGET_GATE_VERSION,
        "target_truth_gate_pass": gate_pass,
        "target_truth_gate_blockers": gate_blockers,
        "target_prior_relation_diagnostic": target_prior_relation_diagnostics(record),
        "uv_albedo_path": path_text(record, "uv_albedo_path"),
        "uv_normal_path": path_text(record, "uv_normal_path"),
        "uv_target_roughness_path": path_text(record, "uv_target_roughness_path"),
        "uv_target_metallic_path": path_text(record, "uv_target_metallic_path"),
        "uv_target_confidence_path": path_text(record, "uv_target_confidence_path"),
        "canonical_views_json": path_text(record, "canonical_views_json"),
        "canonical_buffer_root": path_text(record, "canonical_buffer_root"),
        "canonical_mesh_path": path_text(record, "canonical_mesh_path"),
        "canonical_glb_path": path_text(record, "canonical_glb_path"),
        "view_supervision_ready": bool_value(record.get("view_supervision_ready", True)),
        "valid_view_count": int(record.get("valid_view_count", 0) or 0),
        "scalar_prior_roughness": finite_float(record.get("scalar_prior_roughness")),
        "scalar_prior_metallic": finite_float(record.get("scalar_prior_metallic")),
        "storage_tier": paths["canonical_bundle_root"]["storage_tier"],
        "logical_path": paths["canonical_bundle_root"]["logical_path"],
        "physical_path": paths["canonical_bundle_root"]["physical_path"],
        "path_resolved_ok": all(paths[key]["path_resolved_ok"] for key in ("canonical_buffer_root", "uv_target_roughness_path", "uv_target_metallic_path", "uv_target_confidence_path")),
        "storage_paths": paths,
    }


def build_prior_variant(bundle: dict[str, Any], variant_type: str, source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    spec = VARIANT_SPECS[variant_type]
    oid = bundle["object_id"]
    variant_id = f"pv_{bundle['target_bundle_id']}__{variant_type}"
    prior: dict[str, Any] = {
        "prior_variant_id": variant_id,
        "target_bundle_id": bundle["target_bundle_id"],
        "object_id": oid,
        "split": bundle["split"],
        "prior_variant_type": variant_type,
        "prior_quality_bin": spec["quality"],
        "prior_generation_recipe": spec["recipe"],
        "prior_target_similarity": spec["similarity"],
        "prior_target_gap": 1.0 - float(spec["similarity"]),
        "upstream_model_id": "none_or_existing_pipeline",
        "training_role": spec["role"],
        "sample_weight": spec["weight"],
        "prior_source_type": "synthetic_degradation",
        "prior_spatiality": "spatial_map",
        "prior_mode": "synthetic_rm",
        "has_material_prior": True,
        "prior_confidence": 1.0,
        "scalar_prior_roughness": None,
        "scalar_prior_metallic": None,
        "uv_prior_roughness_path": "",
        "uv_prior_metallic_path": "",
        "prior_path_resolved_ok": True,
        "prior_as_pred_required": False,
        "prior_as_pred_pass": None,
        "prior_variant_status": "ordinary_training_input",
        "prior_not_target_leakage": True,
        "leakage_audit": {"prior_not_target_leakage": True, "leakage_audit_method": "synthetic_lazy_generation"},
        "storage_tier": "lazy_generated",
        "logical_path": "",
        "physical_path": "",
        "path_resolved_ok": True,
    }
    identity_control = None
    if variant_type == "near_gt_prior":
        scalar_r = finite_float(source.get("scalar_prior_roughness"))
        scalar_m = finite_float(source.get("scalar_prior_metallic"))
        prior_r = path_text(source, "uv_prior_roughness_path")
        prior_m = path_text(source, "uv_prior_metallic_path")
        target_r = bundle["uv_target_roughness_path"]
        target_m = bundle["uv_target_metallic_path"]
        source_prior_mode = str(source.get("prior_mode") or "").lower()
        use_scalar = scalar_r is not None and scalar_m is not None and source_prior_mode == "scalar_rm"
        audit = leakage_audit(prior_r, prior_m, target_r, target_m) if prior_r or prior_m else {"prior_not_target_leakage": True, "leakage_audit_method": "scalar_or_missing"}
        if not audit.get("prior_not_target_leakage", True):
            identity_control = {
                **prior,
                "prior_variant_id": f"pv_{bundle['target_bundle_id']}__identity_control",
                "prior_variant_type": "identity_control",
                "prior_quality_bin": "identity",
                "training_role": "no_overedit_regularization",
                "sample_weight": 0.1,
                "prior_generation_recipe": "existing_pipeline_identity_control_excluded_from_ordinary_pairs",
                "leakage_audit": audit,
                "prior_not_target_leakage": False,
                "prior_variant_status": "identity_control_excluded_from_ordinary_pairs",
            }
            prior["prior_variant_type"] = "near_gt_prior"
            prior["prior_source_type"] = "synthetic_degradation"
            prior["prior_generation_recipe"] = "synthetic_degradation_v1:near_gt_leakage_replacement"
        elif use_scalar:
            prior.update(
                {
                    "prior_source_type": "existing_pipeline_prior",
                    "prior_spatiality": "scalar_broadcast",
                    "prior_mode": "scalar_rm",
                    "prior_generation_recipe": "existing_scalar_broadcast_prior",
                    "scalar_prior_roughness": scalar_r,
                    "scalar_prior_metallic": scalar_m,
                    "prior_path_resolved_ok": True,
                    "storage_tier": "scalar_broadcast",
                    "leakage_audit": {"prior_not_target_leakage": True, "leakage_audit_method": "scalar_broadcast"},
                }
            )
        elif prior_r and prior_m and path_exists(prior_r) and path_exists(prior_m):
            prior.update(
                {
                    "prior_source_type": "existing_pipeline_prior",
                    "prior_spatiality": "spatial_map",
                    "prior_mode": "uv_rm",
                    "prior_generation_recipe": "existing_uv_rm_prior",
                    "uv_prior_roughness_path": prior_r,
                    "uv_prior_metallic_path": prior_m,
                    "storage_tier": storage_tier(prior_r),
                    "logical_path": repo_logical_path(prior_r),
                    "physical_path": resolved(prior_r),
                    "leakage_audit": audit,
                    "prior_not_target_leakage": bool(audit["prior_not_target_leakage"]),
                }
            )
        else:
            prior["prior_generation_recipe"] = "synthetic_degradation_v1:near_gt_missing_existing_prior_replacement"
    elif variant_type == "no_prior_bootstrap":
        prior.update(
            {
                "prior_source_type": "no_prior",
                "prior_spatiality": "no_prior",
                "prior_mode": "none",
                "has_material_prior": False,
                "prior_confidence": 0.0,
                "prior_path_resolved_ok": True,
                "prior_as_pred_required": False,
                "storage_tier": "none",
                "leakage_audit": {"prior_not_target_leakage": True, "leakage_audit_method": "skipped_no_prior"},
            }
        )
    return prior, identity_control


def build_pair(bundle: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    pair_id = f"pair_{bundle['target_bundle_id']}__{variant['prior_variant_type']}"
    return {
        "pair_id": pair_id,
        "training_pair_id": pair_id,
        "target_bundle_id": bundle["target_bundle_id"],
        "prior_variant_id": variant["prior_variant_id"],
        "object_id": bundle["object_id"],
        "split": bundle["split"],
        "default_split": bundle["split"],
        "paper_split": bundle["paper_split"],
        "source_name": bundle["source_name"],
        "source_dataset": bundle["source_dataset"],
        "generator_id": bundle["generator_id"],
        "license_bucket": bundle["license_bucket"],
        "material_family": bundle["material_family"],
        "target_source_type": bundle["target_source_type"],
        "target_quality_tier": bundle["target_quality_tier"],
        "target_is_prior_copy": bundle["target_is_prior_copy"],
        "target_prior_identity": variant["prior_target_similarity"],
        "has_material_prior": variant["has_material_prior"],
        "prior_mode": variant["prior_mode"],
        "prior_source_type": variant["prior_source_type"],
        "prior_generation_mode": variant["prior_source_type"],
        "prior_variant_type": variant["prior_variant_type"],
        "prior_quality_bin": variant["prior_quality_bin"],
        "prior_spatiality": variant["prior_spatiality"],
        "prior_generation_recipe": variant["prior_generation_recipe"],
        "prior_target_similarity": variant["prior_target_similarity"],
        "prior_target_gap": variant["prior_target_gap"],
        "training_role": variant["training_role"],
        "upstream_model_id": variant["upstream_model_id"],
        "sample_weight": variant["sample_weight"],
        "prior_confidence": variant["prior_confidence"],
        "scalar_prior_roughness": variant["scalar_prior_roughness"],
        "scalar_prior_metallic": variant["scalar_prior_metallic"],
        "uv_prior_roughness_path": variant["uv_prior_roughness_path"],
        "uv_prior_metallic_path": variant["uv_prior_metallic_path"],
        "prior_path_resolved_ok": variant["prior_path_resolved_ok"],
        "prior_not_target_leakage": variant["prior_not_target_leakage"],
        "uv_albedo_path": bundle["uv_albedo_path"],
        "uv_normal_path": bundle["uv_normal_path"],
        "uv_target_roughness_path": bundle["uv_target_roughness_path"],
        "uv_target_metallic_path": bundle["uv_target_metallic_path"],
        "uv_target_confidence_path": bundle["uv_target_confidence_path"],
        "canonical_views_json": bundle["canonical_views_json"],
        "canonical_buffer_root": bundle["canonical_buffer_root"],
        "canonical_mesh_path": bundle["canonical_mesh_path"],
        "canonical_glb_path": bundle["canonical_glb_path"],
        "view_supervision_ready": bundle["view_supervision_ready"],
        "valid_view_count": bundle["valid_view_count"],
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
        "prior_variant_type": dict(Counter(str(record.get("prior_variant_type") or "unknown") for record in records)),
        "prior_quality_bin": dict(Counter(str(record.get("prior_quality_bin") or "unknown") for record in records)),
        "prior_spatiality": dict(Counter(str(record.get("prior_spatiality") or "unknown") for record in records)),
        "training_role": dict(Counter(str(record.get("training_role") or "unknown") for record in records)),
    }


def free_gb(path: Path) -> float | None:
    try:
        usage = os.statvfs(path)
    except OSError:
        return None
    return float(usage.f_bavail * usage.f_frsize / (1024**3))


def split_leakage_ok(pairs: list[dict[str, Any]]) -> tuple[bool, dict[str, list[str]]]:
    splits: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        splits[str(pair["object_id"])].add(str(pair["split"]))
    offenders = {key: sorted(value) for key, value in splits.items() if len(value) > 1}
    return not offenders, offenders


def write_inventory(path: Path, bundles: list[dict[str, Any]], variants: list[dict[str, Any]], pairs: list[dict[str, Any]], readiness: dict[str, Any]) -> None:
    summary = summarize(pairs)
    lines = [
        "# TrainV5 Plus A-track Inventory",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- target_bundles: `{len(bundles)}`",
        f"- ordinary_prior_variants: `{len(variants)}`",
        f"- training_pairs: `{len(pairs)}`",
        f"- split: `{json.dumps(summary['split'], ensure_ascii=False)}`",
        f"- prior_variant_type: `{json.dumps(summary['prior_variant_type'], ensure_ascii=False)}`",
        f"- prior_spatiality: `{json.dumps(summary['prior_spatiality'], ensure_ascii=False)}`",
        f"- material_family: `{json.dumps(summary['material_family'], ensure_ascii=False)}`",
        f"- source_name: `{json.dumps(summary['source_name'], ensure_ascii=False)}`",
        f"- identity_controls: `{readiness['identity_control_variants']}`",
        f"- sanity_ready: `{str(readiness['sanity_ready']).lower()}`",
        "",
        "A-track TrainV5 Plus expands prior-gap coverage only. It does not claim material/source diversity beyond the 322 inherited targets.",
        "Large UV/view/buffer assets are referenced by paths and are not copied into this train directory.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readiness(path: Path, readiness: dict[str, Any]) -> None:
    lines = [
        "# TrainV5 Plus A-track Readiness",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- sanity_ready: `{str(readiness['sanity_ready']).lower()}`",
        f"- target_bundles: `{readiness['target_bundles']}`",
        f"- ordinary_prior_variants: `{readiness['prior_variants']}`",
        f"- training_pairs: `{readiness['training_pairs']}`",
        f"- each_target_has_five_ordinary_variants: `{readiness['each_target_has_five_ordinary_variants']}`",
        f"- split_nonempty: `{readiness['split_nonempty']}`",
        f"- all_paths_resolved: `{readiness['all_paths_resolved']}`",
        f"- prior_target_leakage_audit_pass: `{readiness['prior_target_leakage_audit_pass']}`",
        f"- object_level_split_check_pass: `{readiness['object_level_split_check_pass']}`",
        f"- ssd_active_free_gb: `{readiness['ssd_active_free_gb']}`",
        f"- blockers: `{json.dumps(readiness['blockers'], ensure_ascii=False)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_command_draft(path: Path) -> None:
    text = """#!/usr/bin/env bash
set -euo pipefail

# Engineering sanity draft only. Keep GPU1 idle.
cd /home/ubuntu/ssd_work/projects/stable-fast-3d
CUDA_VISIBLE_DEVICES=0 python scripts/train_material_refiner.py \\
  --config configs/material_refine_train_r_v2_1_view_aware.yaml \\
  --train-manifest train/trainV5_plus_a_track/trainV5_training_pairs.json \\
  --val-manifest train/trainV5_plus_a_track/trainV5_training_pairs.json \\
  --split-strategy manifest \\
  --train-split train \\
  --val-split val \\
  --output-dir output/material_refine_trainV5_plus_a_track_sanity \\
  --cuda-device-index 0 \\
  --epochs 1 \\
  --max-train-steps 8 \\
  --max-validation-batches 2 \\
  --batch-size 2 \\
  --val-batch-size 2 \\
  --num-workers 0 \\
  --log-every 1 \\
  --wandb-mode disabled
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ssd_active_root.mkdir(parents=True, exist_ok=True)
    source_records, source_manifest = load_source_records(args)

    bundles: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    identity_controls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in source_records:
        oid = object_id(source)
        if not oid or oid in seen:
            continue
        seen.add(oid)
        bundle = build_target_bundle(source)
        bundles.append(bundle)
        for variant_type in VARIANT_SPECS:
            variant, identity_control = build_prior_variant(bundle, variant_type, source)
            variants.append(variant)
            pairs.append(build_pair(bundle, variant))
            if identity_control is not None:
                identity_controls.append(identity_control)

    split_ok, split_offenders = split_leakage_ok(pairs)
    pair_counts = Counter(str(pair["target_bundle_id"]) for pair in pairs)
    split_counts = Counter(str(pair["split"]) for pair in pairs)
    ssd_free = free_gb(args.ssd_active_root)
    readiness = {
        "sanity_ready": False,
        "target_bundles": len(bundles),
        "prior_variants": len(variants),
        "training_pairs": len(pairs),
        "identity_control_variants": len(identity_controls),
        "each_target_has_five_ordinary_variants": all(pair_counts.get(str(bundle["target_bundle_id"]), 0) == 5 for bundle in bundles),
        "split_nonempty": all(split_counts.get(split, 0) > 0 for split in ("train", "val", "test")),
        "all_paths_resolved": all(bool(pair.get("path_resolved_ok")) for pair in pairs),
        "prior_target_leakage_audit_pass": all(bool(variant.get("prior_not_target_leakage")) for variant in variants),
        "object_level_split_check_pass": split_ok,
        "split_leakage_offenders": split_offenders,
        "ssd_active_root": str(args.ssd_active_root.resolve()),
        "ssd_active_free_gb": ssd_free,
        "ssd_min_free_gb": float(args.min_ssd_free_gb),
        "blockers": [],
    }
    if len(bundles) != int(args.expected_targets):
        readiness["blockers"].append(f"target_bundles_expected_{args.expected_targets}_got_{len(bundles)}")
    if len(variants) != len(bundles) * 5:
        readiness["blockers"].append("prior_variants_not_five_per_target")
    if len(pairs) != len(bundles) * 5:
        readiness["blockers"].append("training_pairs_not_five_per_target")
    for key in (
        "each_target_has_five_ordinary_variants",
        "split_nonempty",
        "all_paths_resolved",
        "prior_target_leakage_audit_pass",
        "object_level_split_check_pass",
    ):
        if not readiness[key]:
            readiness["blockers"].append(key)
    if ssd_free is not None and ssd_free < float(args.min_ssd_free_gb):
        readiness["blockers"].append("ssd_active_free_space_below_threshold")
    readiness["sanity_ready"] = not readiness["blockers"]

    base = {
        "generated_at_utc": utc_now(),
        "source_manifest": source_manifest,
        "data_contract": "trainV5_plus_a_track_pair_manifest_v1",
        "notes": [
            "Pair-based R-v2.1 data. Each ordinary pair is target_bundle + prior_variant.",
            "Synthetic prior variants are generated lazily by dataset.py from recorded recipes; no large images are copied into train/.",
            "A-track source is currently ABO/glossy/scalar-broadcast and does not satisfy material/source diversity alone.",
        ],
    }
    write_json(args.output_dir / "trainV5_target_bundles.json", {**base, "records": bundles, "summary": summarize(bundles)})
    write_json(args.output_dir / "trainV5_prior_variants.json", {**base, "records": variants, "summary": summarize(variants)})
    write_json(args.output_dir / "trainV5_training_pairs.json", {**base, "records": pairs, "summary": summarize(pairs)})
    write_json(args.output_dir / "trainV5_identity_controls.json", {**base, "records": identity_controls, "summary": summarize(identity_controls)})
    write_json(
        args.output_dir / "trainV5_sampler_config.json",
        {
            "generated_at_utc": utc_now(),
            "sampler_version": "trainV5_plus_pair_sampler_v1",
            "balance_key": "prior_variant_type",
            "weights": {
                "near_gt_prior": 1.0,
                "mild_gap_prior": 1.0,
                "medium_gap_prior": 1.0,
                "large_gap_prior": 1.0,
                "no_prior_bootstrap": 0.75,
            },
            "summary": summarize(pairs),
        },
    )
    write_json(args.output_dir / "trainV5_readiness_report.json", readiness)
    write_inventory(args.output_dir / "trainV5_inventory.md", bundles, variants, pairs, readiness)
    write_readiness(args.output_dir / "trainV5_readiness_report.md", readiness)
    write_command_draft(args.output_dir / "r_v2_1_trainV5_plus_sanity.sh")
    print(json.dumps({"target_bundles": len(bundles), "prior_variants": len(variants), "training_pairs": len(pairs), "sanity_ready": readiness["sanity_ready"], "output_dir": str(args.output_dir.resolve())}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
