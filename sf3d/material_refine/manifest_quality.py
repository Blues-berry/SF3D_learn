from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TARGET_SOURCE_TYPES = {
    "gt_render_baked",
    "pseudo_from_multiview",
    "pseudo_from_material_bank",
    "copied_from_prior",
    "unknown",
}
TARGET_QUALITY_TIERS = {
    "paper_strong",
    "paper_pseudo",
    "research_only",
    "smoke_only",
    "unknown",
}
PAPER_ELIGIBLE_TARGET_QUALITY_TIERS = {"paper_strong", "paper_pseudo"}
SUPERVISION_ROLES = {
    "paper_main",
    "auxiliary_upgrade_queue",
    "auxiliary_highlight",
    "material_prior",
    "lighting_bank",
    "benchmark_ood",
    "unknown",
}
PAPER_ELIGIBLE_SUPERVISION_ROLES = {"paper_main"}
MATERIAL_FAMILIES = {
    "metal_dominant",
    "ceramic_glazed_lacquer",
    "glass_metal",
    "mixed_thin_boundary",
    "glossy_non_metal",
    "unknown",
}

DEFAULT_MAX_TARGET_PRIOR_IDENTITY_RATE_FOR_PAPER = 0.30
DEFAULT_MIN_NONTRIVIAL_TARGET_COUNT_FOR_PAPER = 128
DEFAULT_MIN_TARGET_CONFIDENCE_MEAN_FOR_PAPER = 0.70
DEFAULT_MIN_TARGET_CONFIDENCE_NONZERO_RATE_FOR_PAPER = 0.50
NEAR_COPY_TARGET_PRIOR_IDENTITY_THRESHOLD = 0.95
AUDIT_SCHEMA_VERSION = "material_refine_quality_v2"

VIEW_BUFFER_FIELD_CANDIDATES = {
    "rgba": ("rgba.png", "rgb.png"),
    "mask": ("mask.png",),
    "depth": ("depth.npy", "depth.npz", "depth.png"),
    "normal": ("normal.npy", "normal.npz", "normal.png"),
    "position": ("position.npy", "position.npz", "position.png"),
    "uv": ("uv.npy", "uv.npz", "uv.png"),
    "visibility": ("visibility.npy", "visibility.npz", "visibility.png"),
    "roughness": ("roughness.png",),
    "metallic": ("metallic.png",),
}
STRICT_REQUIRED_VIEW_FIELDS = (
    "rgba",
    "mask",
    "depth",
    "normal",
    "position",
    "uv",
    "visibility",
    "roughness",
    "metallic",
)
VIEW_BUFFER_FIELD_SOURCES_FILENAME = "_field_sources.json"


def resolve_record_path(
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    record: dict[str, Any],
    value: str | None,
) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    bundle_root_value = (
        record.get("bundle_root")
        or record.get("canonical_bundle_root")
        or manifest_payload.get("canonical_bundle_root")
        or manifest_payload.get("bundle_root")
    )
    if bundle_root_value:
        bundle_root = Path(str(bundle_root_value))
        if not bundle_root.is_absolute():
            bundle_root = manifest_path.parent / bundle_root
        candidate = bundle_root / path
        if candidate.exists():
            return candidate
    return manifest_path.parent / path


def resolve_optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def file_digest(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confidence_summary_from_path(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    nonzero = arr > 1e-6
    high_conf = arr >= 0.75
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.quantile(arr, 0.50)),
        "max": float(arr.max()),
        "nonzero_rate": float(nonzero.mean()),
        "high_conf_rate": float(high_conf.mean()),
    }


def grayscale_stats_from_path(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "high_rate": float((arr >= 0.55).mean()),
        "low_rate": float((arr <= 0.10).mean()),
    }


def parse_confidence_summary(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {
            str(key): float(item)
            for key, item in value.items()
            if isinstance(item, (int, float))
        }
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parse_confidence_summary(parsed)
    return {}


def load_view_field_sources(buffer_root: Path | None) -> dict[str, dict[str, str]]:
    if buffer_root is None:
        return {}
    metadata_path = buffer_root / VIEW_BUFFER_FIELD_SOURCES_FILENAME
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return {}
    views = payload.get("views")
    if not isinstance(views, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for view_name, item in views.items():
        if not isinstance(item, dict):
            continue
        fields = item.get("fields")
        if isinstance(fields, dict):
            out[str(view_name)] = {str(key): str(value) for key, value in fields.items()}
    return out


def view_field_is_effective(source: str | None) -> bool:
    if not source:
        return False
    return str(source).lower() not in {"missing", "placeholder", "synthetic_placeholder"}


def _candidate_present(
    *,
    view_dir: Path,
    field: str,
    field_sources: dict[str, str] | None,
) -> bool:
    if field_sources is not None:
        source = field_sources.get(field)
        if source is not None and not view_field_is_effective(source):
            return False
    return any((view_dir / candidate).exists() for candidate in VIEW_BUFFER_FIELD_CANDIDATES[field])


def view_buffer_field_presence(buffer_root: Path | None) -> dict[str, int]:
    field_counts = {field: 0 for field in VIEW_BUFFER_FIELD_CANDIDATES}
    field_counts["views"] = 0
    field_counts["effective_view_supervision_views"] = 0
    field_counts["strict_complete_views"] = 0
    if buffer_root is None or not buffer_root.exists():
        return field_counts

    field_sources_by_view = load_view_field_sources(buffer_root)
    view_dirs = [path for path in buffer_root.iterdir() if path.is_dir()]
    field_counts["views"] = len(view_dirs)
    for view_dir in view_dirs:
        field_sources = field_sources_by_view.get(view_dir.name)
        present = {}
        for field, candidates in VIEW_BUFFER_FIELD_CANDIDATES.items():
            present[field] = _candidate_present(
                view_dir=view_dir,
                field=field,
                field_sources=field_sources,
            )
            field_counts[field] += int(present[field])
        effective_view_supervision = (
            present["rgba"] and present["uv"] and present["roughness"] and present["metallic"]
        )
        strict_complete = all(present[field] for field in STRICT_REQUIRED_VIEW_FIELDS)
        field_counts["effective_view_supervision_views"] += int(effective_view_supervision)
        field_counts["strict_complete_views"] += int(strict_complete)
    return field_counts


def infer_target_source_type(
    record: dict[str, Any],
    *,
    target_is_prior_copy: bool,
    view_counts: dict[str, int],
) -> str:
    explicit = str(record.get("target_source_type") or "").strip()
    if explicit in TARGET_SOURCE_TYPES:
        return explicit
    if target_is_prior_copy:
        return "copied_from_prior"
    notes = str(record.get("notes") or "").lower()
    source_name = str(record.get("source_name") or "").lower()
    if "material_bank" in notes or "material_bank" in source_name:
        return "pseudo_from_material_bank"
    if view_counts.get("roughness", 0) > 0 and view_counts.get("metallic", 0) > 0:
        return "pseudo_from_multiview"
    return "unknown"


def infer_target_quality_tier(
    record: dict[str, Any],
    *,
    target_source_type: str,
    target_is_prior_copy: bool,
    confidence_summary: dict[str, float],
    is_complete: bool,
) -> str:
    explicit = str(record.get("target_quality_tier") or "").strip()
    if explicit in TARGET_QUALITY_TIERS:
        return explicit
    if not is_complete or target_is_prior_copy or target_source_type == "copied_from_prior":
        return "smoke_only"
    confidence_mean = float(confidence_summary.get("mean", 0.0))
    nonzero_rate = float(confidence_summary.get("nonzero_rate", 0.0))
    if target_source_type == "gt_render_baked":
        if (
            confidence_mean >= DEFAULT_MIN_TARGET_CONFIDENCE_MEAN_FOR_PAPER
            and nonzero_rate >= DEFAULT_MIN_TARGET_CONFIDENCE_NONZERO_RATE_FOR_PAPER
        ):
            return "paper_strong"
        return "research_only"
    if target_source_type in {"pseudo_from_multiview", "pseudo_from_material_bank"}:
        if (
            confidence_mean >= DEFAULT_MIN_TARGET_CONFIDENCE_MEAN_FOR_PAPER
            and nonzero_rate >= DEFAULT_MIN_TARGET_CONFIDENCE_NONZERO_RATE_FOR_PAPER
        ):
            return "paper_pseudo"
        if confidence_mean > 0.0 and nonzero_rate > 0.0:
            return "research_only"
        return "smoke_only"
    return "smoke_only"


def infer_supervision_role(record: dict[str, Any]) -> str:
    explicit = str(record.get("supervision_role") or "").strip()
    if explicit in SUPERVISION_ROLES:
        return explicit
    source_name = str(record.get("source_name") or "").lower()
    supervision_tier = str(record.get("supervision_tier") or "").lower()
    if source_name == "abo_locked_core":
        return "paper_main"
    if any(token in source_name for token in ("olatverse", "openillumination", "ictpolarreal")):
        return "auxiliary_highlight"
    if any(token in source_name for token in ("polyhaven", "laval", "hdri", "stress")):
        return "lighting_bank"
    if any(token in source_name for token in ("stanford-orb", "benchmark", "ood")):
        return "benchmark_ood"
    if supervision_tier == "eval_only":
        return "benchmark_ood"
    if supervision_tier == "material_prior":
        return "material_prior"
    if supervision_tier in {"strong", "aux_highlight", "part_semantic"}:
        return "auxiliary_upgrade_queue"
    return "unknown"


def infer_material_family(record: dict[str, Any], *, metallic_stats: dict[str, float] | None = None) -> str:
    explicit = str(record.get("material_family") or "").strip()
    stats = metallic_stats or {}
    metallic_mean = float(stats.get("mean", record.get("avg_gt_metallic_mean", 0.0) or 0.0))
    metallic_high_rate = float(stats.get("high_rate", 0.0) or 0.0)
    text = " ".join(
        [
            str(record.get("highlight_priority_class") or ""),
            str(record.get("material_bucket") or ""),
            str(record.get("category_bucket") or ""),
            str(record.get("label") or ""),
            str(record.get("name") or ""),
            str(record.get("category") or ""),
            str(record.get("super_category") or ""),
            str(record.get("material") or ""),
            str(record.get("notes") or ""),
        ]
    ).lower()
    has_metal_text = "metal_dominant" in text or "metal dominant" in text or any(
        token in text for token in ("chrome", "steel", "aluminum", "aluminium", "brass", "copper")
    ) or bool(
        re.search(r"(?:material\s*=\s*metal|\bmetal\b|\bmetal frame\b|\bmetallic object\b)", text)
    )
    # Older manifests stamped many unknowns as glossy_non_metal. Do not override
    # that label from pseudo metallic maps alone; noisy pseudo targets can
    # otherwise make the distribution look solved when it is not.
    if explicit in MATERIAL_FAMILIES and explicit != "unknown":
        if explicit == "glossy_non_metal" and has_metal_text:
            return "metal_dominant"
        return explicit
    if "mixed_thin_boundary" in text or "thin-boundary" in text or "thin boundary" in text:
        return "mixed_thin_boundary"
    if "glass_metal" in text or "glass" in text:
        return "glass_metal"
    if "ceramic" in text or "glazed" in text or "lacquer" in text:
        return "ceramic_glazed_lacquer"
    if has_metal_text:
        return "metal_dominant"
    if metallic_mean >= 0.35 or metallic_high_rate >= 0.20:
        return "metal_dominant"
    if any(token in text for token in ("frame", "wire", "rack", "handle", "lamp", "shelf")):
        return "mixed_thin_boundary"
    return "glossy_non_metal"


def infer_thin_boundary_flag(record: dict[str, Any], *, material_family: str) -> bool:
    explicit = record.get("thin_boundary_flag")
    if explicit is not None:
        if isinstance(explicit, bool):
            return explicit
        return str(explicit).strip().lower() in {"1", "true", "yes", "y"}
    if material_family == "mixed_thin_boundary":
        return True
    text = " ".join(
        [
            str(record.get("label") or ""),
            str(record.get("name") or ""),
            str(record.get("category") or ""),
            str(record.get("notes") or ""),
        ]
    ).lower()
    return any(token in text for token in ("thin", "frame", "wire", "handle", "boundary"))


def infer_lighting_bank_id(record: dict[str, Any]) -> str:
    explicit = str(record.get("lighting_bank_id") or "").strip()
    if explicit:
        return explicit
    source_name = str(record.get("source_name") or "").lower()
    if any(token in source_name for token in ("olatverse", "openillumination", "ictpolarreal")):
        return "controlled_real_capture_v1"
    if any(token in source_name for token in ("polyhaven", "laval", "hdri", "stress")):
        return "stress_hdri_bank_v1"
    return "canonical_triplet_v1"


def derive_category_bucket(record: dict[str, Any]) -> str:
    for key in (
        "material_bucket",
        "sampling_bucket",
        "_material_class",
        "priority_bucket",
        "smoke_source_bucket",
        "category_bucket",
    ):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    source_name = str(record.get("source_name") or record.get("generator_id") or "unknown")
    prior_label = "with_prior" if bool(record.get("has_material_prior")) else "without_prior"
    return f"{source_name}|{prior_label}"


def derive_category_label(record: dict[str, Any]) -> str:
    for key in ("category", "material_category", "super_category", "super-category"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return "unknown"


def target_prior_identity_from_paths(
    *,
    target_roughness_path: Path | None,
    target_metallic_path: Path | None,
    prior_roughness_path: Path | None,
    prior_metallic_path: Path | None,
    confidence_path: Path | None,
) -> float | None:
    if (
        target_roughness_path is None
        or target_metallic_path is None
        or prior_roughness_path is None
        or prior_metallic_path is None
        or not target_roughness_path.exists()
        or not target_metallic_path.exists()
        or not prior_roughness_path.exists()
        or not prior_metallic_path.exists()
    ):
        return None
    target_roughness_image = Image.open(target_roughness_path).convert("L")
    target_size = target_roughness_image.size
    target_roughness = np.asarray(target_roughness_image, dtype=np.float32) / 255.0
    target_metallic = (
        np.asarray(Image.open(target_metallic_path).convert("L").resize(target_size, Image.BILINEAR), dtype=np.float32)
        / 255.0
    )
    prior_roughness = (
        np.asarray(Image.open(prior_roughness_path).convert("L").resize(target_size, Image.BILINEAR), dtype=np.float32)
        / 255.0
    )
    prior_metallic = (
        np.asarray(Image.open(prior_metallic_path).convert("L").resize(target_size, Image.BILINEAR), dtype=np.float32)
        / 255.0
    )
    if confidence_path is not None and confidence_path.exists():
        confidence = (
            np.asarray(Image.open(confidence_path).convert("L").resize(target_size, Image.BILINEAR), dtype=np.float32)
            / 255.0
        )
    else:
        confidence = np.ones_like(target_roughness, dtype=np.float32)
    weight = np.maximum(confidence, 1e-6)
    rough_error = float(np.sum(np.abs(target_roughness - prior_roughness) * weight) / np.sum(weight))
    metal_error = float(np.sum(np.abs(target_metallic - prior_metallic) * weight) / np.sum(weight))
    return max(0.0, min(1.0, 1.0 - 0.5 * (rough_error + metal_error)))


def audit_record(
    manifest_path: Path,
    payload: dict[str, Any],
    record: dict[str, Any],
    *,
    allowed_paper_license_buckets: set[str] | None = None,
) -> dict[str, Any]:
    required_fields = [
        "uv_albedo_path",
        "uv_normal_path",
        "uv_prior_roughness_path",
        "uv_prior_metallic_path",
        "uv_target_roughness_path",
        "uv_target_metallic_path",
        "uv_target_confidence_path",
        "canonical_views_json",
        "canonical_buffer_root",
    ]
    resolved = {
        field: resolve_record_path(manifest_path, payload, record, record.get(field))
        for field in required_fields
    }
    canonical_asset_candidates = {
        "canonical_mesh_path": resolve_record_path(
            manifest_path,
            payload,
            record,
            record.get("canonical_mesh_path"),
        ),
        "canonical_glb_path": resolve_record_path(
            manifest_path,
            payload,
            record,
            record.get("canonical_glb_path"),
        ),
    }
    missing = [
        field
        for field, path in resolved.items()
        if path is None or not path.exists()
    ]
    if not any(path is not None and path.exists() for path in canonical_asset_candidates.values()):
        missing.append("canonical_asset_path")
    prior_roughness_digest = file_digest(resolved["uv_prior_roughness_path"])
    target_roughness_digest = file_digest(resolved["uv_target_roughness_path"])
    prior_metallic_digest = file_digest(resolved["uv_prior_metallic_path"])
    target_metallic_digest = file_digest(resolved["uv_target_metallic_path"])
    same_roughness = (
        prior_roughness_digest is not None
        and prior_roughness_digest == target_roughness_digest
    )
    same_metallic = (
        prior_metallic_digest is not None
        and prior_metallic_digest == target_metallic_digest
    )
    same_rm_pair = same_roughness and same_metallic
    explicit_confidence_summary = parse_confidence_summary(record.get("target_confidence_summary"))
    confidence_summary = (
        explicit_confidence_summary
        if explicit_confidence_summary
        else confidence_summary_from_path(resolved["uv_target_confidence_path"])
    )
    view_counts = view_buffer_field_presence(resolved["canonical_buffer_root"])
    target_metallic_stats = grayscale_stats_from_path(resolved["uv_target_metallic_path"])
    target_roughness_stats = grayscale_stats_from_path(resolved["uv_target_roughness_path"])
    target_prior_identity = (
        float(record.get("target_prior_identity"))
        if record.get("target_prior_identity") not in {None, "", "None"}
        else target_prior_identity_from_paths(
            target_roughness_path=resolved["uv_target_roughness_path"],
            target_metallic_path=resolved["uv_target_metallic_path"],
            prior_roughness_path=resolved["uv_prior_roughness_path"],
            prior_metallic_path=resolved["uv_prior_metallic_path"],
            confidence_path=resolved["uv_target_confidence_path"],
        )
    )
    target_is_near_copy = bool(
        target_prior_identity is not None
        and float(target_prior_identity) >= NEAR_COPY_TARGET_PRIOR_IDENTITY_THRESHOLD
    )
    target_is_prior_copy = bool(record.get("target_is_prior_copy")) or same_rm_pair or target_is_near_copy
    target_source_type = infer_target_source_type(
        record,
        target_is_prior_copy=target_is_prior_copy,
        view_counts=view_counts,
    )
    supervision_role = infer_supervision_role(record)
    material_family = infer_material_family(record, metallic_stats=target_metallic_stats)
    thin_boundary_flag = infer_thin_boundary_flag(record, material_family=material_family)
    lighting_bank_id = infer_lighting_bank_id(record)
    target_confidence_mean = (
        float(record.get("target_confidence_mean"))
        if record.get("target_confidence_mean") not in {None, "", "None"}
        else float(confidence_summary.get("mean", 0.0))
    )
    target_confidence_nonzero_rate = (
        float(record.get("target_confidence_nonzero_rate"))
        if record.get("target_confidence_nonzero_rate") not in {None, "", "None"}
        else float(confidence_summary.get("nonzero_rate", 0.0))
    )
    target_coverage = (
        float(record.get("target_coverage"))
        if record.get("target_coverage") not in {None, "", "None"}
        else float(confidence_summary.get("nonzero_rate", 0.0))
    )
    target_quality_tier = infer_target_quality_tier(
        record,
        target_source_type=target_source_type,
        target_is_prior_copy=target_is_prior_copy,
        confidence_summary=confidence_summary,
        is_complete=not missing,
    )
    license_bucket = str(record.get("license_bucket", "unknown"))
    license_allowed = (
        True
        if allowed_paper_license_buckets is None
        else license_bucket in allowed_paper_license_buckets
    )
    paper_stage_eligible = (
        target_quality_tier in PAPER_ELIGIBLE_TARGET_QUALITY_TIERS
        and not target_is_prior_copy
        and not missing
        and license_allowed
        and supervision_role in PAPER_ELIGIBLE_SUPERVISION_ROLES
    )
    views = max(int(view_counts.get("views", 0)), 0)
    effective_view_supervision_views = int(view_counts.get("effective_view_supervision_views", 0))
    strict_complete_views = int(view_counts.get("strict_complete_views", 0))
    valid_view_count = int(
        record.get("valid_view_count", max(effective_view_supervision_views, strict_complete_views))
        or 0
    )
    view_supervision_ready = bool(record.get("view_supervision_ready"))
    if not view_supervision_ready:
        view_supervision_ready = effective_view_supervision_views > 0 and strict_complete_views > 0
    return {
        "object_id": record.get("object_id"),
        "source_name": record.get("source_name", record.get("generator_id", "unknown")),
        "generator_id": record.get("generator_id", "unknown"),
        "license_bucket": license_bucket,
        "supervision_role": supervision_role,
        "supervision_tier": record.get("supervision_tier", "unknown"),
        "prior_mode": record.get("prior_mode", "unknown"),
        "has_material_prior": bool(record.get("has_material_prior")),
        "category_bucket": derive_category_bucket(record),
        "category_label": derive_category_label(record),
        "material_family": material_family,
        "target_metallic_stats": target_metallic_stats,
        "target_roughness_stats": target_roughness_stats,
        "thin_boundary_flag": thin_boundary_flag,
        "lighting_bank_id": lighting_bank_id,
        "missing_fields": missing,
        "is_complete": not missing,
        "same_roughness": same_roughness,
        "same_metallic": same_metallic,
        "same_rm_pair": same_rm_pair,
        "target_is_near_copy": target_is_near_copy,
        "target_is_prior_copy": target_is_prior_copy,
        "target_prior_identity": target_prior_identity,
        "target_prior_similarity": target_prior_identity,
        "target_prior_distance": (
            None if target_prior_identity is None else max(0.0, min(1.0, 1.0 - float(target_prior_identity)))
        ),
        "target_source_type": target_source_type,
        "target_quality_tier": target_quality_tier,
        "target_confidence_summary": confidence_summary,
        "target_confidence_mean": target_confidence_mean,
        "target_confidence_nonzero_rate": target_confidence_nonzero_rate,
        "target_coverage": target_coverage,
        "paper_stage_eligible": paper_stage_eligible,
        "paper_license_allowed": license_allowed,
        "views": views,
        "view_field_counts": {field: int(value) for field, value in view_counts.items()},
        "effective_view_supervision_views": effective_view_supervision_views,
        "effective_view_supervision_rate": (
            effective_view_supervision_views / float(views)
            if views > 0
            else 0.0
        ),
        "strict_complete_views": strict_complete_views,
        "strict_complete_view_rate": (
            strict_complete_views / float(views)
            if views > 0
            else 0.0
        ),
        "view_supervision_ready": view_supervision_ready,
        "valid_view_count": valid_view_count,
    }


def summarize_audit_rows(
    rows: list[dict[str, Any]],
    *,
    identity_warning_threshold: float,
    max_target_prior_identity_rate_for_paper: float = DEFAULT_MAX_TARGET_PRIOR_IDENTITY_RATE_FOR_PAPER,
    min_nontrivial_target_count_for_paper: int = DEFAULT_MIN_NONTRIVIAL_TARGET_COUNT_FOR_PAPER,
) -> dict[str, Any]:
    count = len(rows)
    source_counts = Counter(str(row["source_name"]) for row in rows)
    generator_counts = Counter(str(row["generator_id"]) for row in rows)
    prior_counts = Counter("with_prior" if row["has_material_prior"] else "without_prior" for row in rows)
    license_counts = Counter(str(row["license_bucket"]) for row in rows)
    supervision_role_counts = Counter(str(row["supervision_role"]) for row in rows)
    supervision_counts = Counter(str(row["supervision_tier"]) for row in rows)
    target_source_type_counts = Counter(str(row["target_source_type"]) for row in rows)
    target_quality_tier_counts = Counter(str(row["target_quality_tier"]) for row in rows)
    category_bucket_counts = Counter(str(row["category_bucket"]) for row in rows)
    material_family_counts = Counter(str(row["material_family"]) for row in rows)
    lighting_bank_counts = Counter(str(row["lighting_bank_id"]) for row in rows)
    missing_counts = Counter()
    for row in rows:
        missing_counts.update(row["missing_fields"])

    same_rm_pair = sum(1 for row in rows if row["same_rm_pair"])
    high_identity_count = sum(1 for row in rows if row["target_is_prior_copy"])
    target_is_prior_copy_count = sum(1 for row in rows if row["target_is_prior_copy"])
    complete = sum(1 for row in rows if row["is_complete"])
    nontrivial_target_records = sum(
        1 for row in rows if not row["target_is_prior_copy"] and row["target_source_type"] != "unknown"
    )
    paper_stage_eligible_records = sum(1 for row in rows if row["paper_stage_eligible"])
    paper_license_allowed_records = sum(1 for row in rows if row["paper_license_allowed"])
    views = sum(int(row["views"]) for row in rows)
    field_totals = Counter()
    effective_view_supervision_records = 0
    strict_complete_records = 0
    ready_view_supervision_records = 0
    for row in rows:
        field_totals.update(row["view_field_counts"])
        effective_view_supervision_records += int(row["effective_view_supervision_views"] > 0 and row["view_supervision_ready"])
        strict_complete_records += int(row["strict_complete_views"] > 0 and row["view_supervision_ready"])
        ready_view_supervision_records += int(row["view_supervision_ready"])

    confidence_means = [
        float(row["target_confidence_summary"].get("mean", 0.0))
        for row in rows
        if row.get("target_confidence_summary")
    ]
    confidence_nonzero_rates = [
        float(row["target_confidence_summary"].get("nonzero_rate", 0.0))
        for row in rows
        if row.get("target_confidence_summary")
    ]
    identity_means = [
        float(row["target_prior_identity"])
        for row in rows
        if row.get("target_prior_identity") is not None
    ]
    target_prior_similarity_mean = float(np.mean(identity_means)) if identity_means else 0.0

    by_source: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "paper_stage_eligible": 0.0,
            "target_is_prior_copy": 0.0,
            "effective_view_supervision_records": 0.0,
        }
    )
    for row in rows:
        bucket = by_source[str(row["source_name"])]
        bucket["count"] += 1.0
        bucket["paper_stage_eligible"] += float(row["paper_stage_eligible"])
        bucket["target_is_prior_copy"] += float(row["target_is_prior_copy"])
        bucket["effective_view_supervision_records"] += float(
            row["effective_view_supervision_views"] > 0 and row["view_supervision_ready"]
        )
    by_source_final = {}
    for key, bucket in by_source.items():
        denom = max(bucket["count"], 1.0)
        by_source_final[key] = {
            "count": int(bucket["count"]),
            "paper_stage_eligible_rate": bucket["paper_stage_eligible"] / denom,
            "target_prior_identity_rate": bucket["target_is_prior_copy"] / denom,
            "effective_view_supervision_record_rate": bucket["effective_view_supervision_records"] / denom,
        }

    target_prior_identity_rate = high_identity_count / max(count, 1)
    paper_stage_ready = True
    readiness_blockers = []
    if target_prior_identity_rate > max_target_prior_identity_rate_for_paper:
        paper_stage_ready = False
        readiness_blockers.append(
            f"target_prior_identity_rate={target_prior_identity_rate:.3f} exceeds {max_target_prior_identity_rate_for_paper:.3f}"
        )
    if paper_stage_eligible_records < int(min_nontrivial_target_count_for_paper):
        paper_stage_ready = False
        readiness_blockers.append(
            f"paper_stage_eligible_records={paper_stage_eligible_records} below {int(min_nontrivial_target_count_for_paper)}"
        )

    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "records": count,
        "complete_records": complete,
        "complete_rate": complete / max(count, 1),
        "same_rm_pair": same_rm_pair,
        "same_rm_pair_rate": same_rm_pair / max(count, 1),
        "high_identity_records": high_identity_count,
        "target_prior_identity_rate": target_prior_identity_rate,
        "identity_warning_threshold": identity_warning_threshold,
        "identity_warning": target_prior_identity_rate >= identity_warning_threshold,
        "target_is_prior_copy_count": target_is_prior_copy_count,
        "target_is_prior_copy_rate": target_is_prior_copy_count / max(count, 1),
        "target_prior_identity_mean": target_prior_similarity_mean,
        "target_prior_similarity_mean": target_prior_similarity_mean,
        "target_prior_distance_mean": 1.0 - target_prior_similarity_mean if identity_means else 0.0,
        "nontrivial_target_records": nontrivial_target_records,
        "nontrivial_target_rate": nontrivial_target_records / max(count, 1),
        "paper_license_allowed_records": paper_license_allowed_records,
        "paper_license_allowed_rate": paper_license_allowed_records / max(count, 1),
        "paper_stage_eligible_records": paper_stage_eligible_records,
        "paper_stage_eligible_rate": paper_stage_eligible_records / max(count, 1),
        "paper_stage_ready": paper_stage_ready,
        "readiness_blockers": readiness_blockers,
        "max_target_prior_identity_rate_for_paper": max_target_prior_identity_rate_for_paper,
        "min_nontrivial_target_count_for_paper": int(min_nontrivial_target_count_for_paper),
        "views": views,
        "buffer_field_totals": dict(field_totals),
        "buffer_field_rates": {
            key: value / max(views, 1)
            for key, value in field_totals.items()
            if key != "views"
        },
        "view_supervision_ready_records": ready_view_supervision_records,
        "view_supervision_ready_rate": ready_view_supervision_records / max(count, 1),
        "effective_view_supervision_records": effective_view_supervision_records,
        "effective_view_supervision_record_rate": effective_view_supervision_records / max(count, 1),
        "effective_view_supervision_views": int(field_totals.get("effective_view_supervision_views", 0)),
        "effective_view_supervision_view_rate": field_totals.get("effective_view_supervision_views", 0) / max(views, 1),
        "strict_complete_records": strict_complete_records,
        "strict_complete_record_rate": strict_complete_records / max(count, 1),
        "strict_complete_views": int(field_totals.get("strict_complete_views", 0)),
        "strict_complete_view_rate": field_totals.get("strict_complete_views", 0) / max(views, 1),
        "source_counts": dict(source_counts),
        "generator_counts": dict(generator_counts),
        "prior_counts": dict(prior_counts),
        "license_bucket_counts": dict(license_counts),
        "supervision_role_counts": dict(supervision_role_counts),
        "supervision_tier_counts": dict(supervision_counts),
        "target_source_type_counts": dict(target_source_type_counts),
        "target_quality_tier_counts": dict(target_quality_tier_counts),
        "category_bucket_counts": dict(category_bucket_counts),
        "material_family_counts": dict(material_family_counts),
        "lighting_bank_id_counts": dict(lighting_bank_counts),
        "missing_field_counts": dict(missing_counts),
        "confidence_summary_aggregate": {
            "mean_of_mean": float(np.mean(confidence_means)) if confidence_means else 0.0,
            "mean_nonzero_rate": float(np.mean(confidence_nonzero_rates)) if confidence_nonzero_rates else 0.0,
        },
        "by_source": by_source_final,
    }


def load_manifest_records(manifest_path: Path, *, max_records: int = -1) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(manifest_path.read_text())
    raw_records = payload.get("records") or payload.get("objects") or payload.get("rows") or []
    if not isinstance(raw_records, list):
        raise TypeError(f"unsupported_manifest_records:{manifest_path}")
    records = [record for record in raw_records if isinstance(record, dict)]
    if max_records >= 0:
        records = records[:max_records]
    return payload, records


def audit_manifest(
    manifest_path: Path,
    *,
    max_records: int = -1,
    identity_warning_threshold: float = 0.95,
    max_target_prior_identity_rate_for_paper: float = DEFAULT_MAX_TARGET_PRIOR_IDENTITY_RATE_FOR_PAPER,
    min_nontrivial_target_count_for_paper: int = DEFAULT_MIN_NONTRIVIAL_TARGET_COUNT_FOR_PAPER,
    allowed_paper_license_buckets: set[str] | None = None,
) -> dict[str, Any]:
    payload, records = load_manifest_records(manifest_path, max_records=max_records)
    rows = [
        audit_record(
            manifest_path,
            payload,
            record,
            allowed_paper_license_buckets=allowed_paper_license_buckets,
        )
        for record in records
    ]
    summary = summarize_audit_rows(
        rows,
        identity_warning_threshold=identity_warning_threshold,
        max_target_prior_identity_rate_for_paper=max_target_prior_identity_rate_for_paper,
        min_nontrivial_target_count_for_paper=min_nontrivial_target_count_for_paper,
    )
    return {
        "manifest": str(manifest_path.resolve()),
        "audited_records": len(rows),
        "summary": summary,
        "records": rows,
    }


def enrich_record_with_quality_fields(record: dict[str, Any]) -> dict[str, Any]:
    row = audit_record(
        Path(record.get("uv_target_confidence_path") or "."),
        {},
        record,
        allowed_paper_license_buckets=None,
    )
    record.update(
        {
            "supervision_role": row["supervision_role"],
            "target_source_type": row["target_source_type"],
            "target_is_prior_copy": row["target_is_prior_copy"],
            "target_prior_identity": row["target_prior_identity"],
            "target_quality_tier": row["target_quality_tier"],
            "target_confidence_summary": row["target_confidence_summary"],
            "target_confidence_mean": row["target_confidence_mean"],
            "target_confidence_nonzero_rate": row["target_confidence_nonzero_rate"],
            "target_coverage": row["target_coverage"],
            "paper_split": record.get("paper_split") or record.get("default_split"),
            "material_family": row["material_family"],
            "thin_boundary_flag": row["thin_boundary_flag"],
            "lighting_bank_id": row["lighting_bank_id"],
            "view_supervision_ready": row["view_supervision_ready"],
            "valid_view_count": row["valid_view_count"],
        }
    )
    return record
