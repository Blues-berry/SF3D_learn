from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_optional_float(value: Any) -> float | None:
    if value in {None, "", "None"}:
        return None
    return float(value)


def parse_optional_str(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def parse_optional_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass
class CanonicalAssetRecordV1:
    object_id: str
    generator_id: str = "sf3d"
    license_bucket: str = "unknown"
    supervision_role: str = "unknown"
    has_material_prior: bool = False
    prior_mode: str = "none"
    canonical_mesh_path: str | None = None
    canonical_glb_path: str | None = None
    uv_albedo_path: str | None = None
    uv_normal_path: str | None = None
    uv_prior_roughness_path: str | None = None
    uv_prior_metallic_path: str | None = None
    scalar_prior_roughness: float | None = None
    scalar_prior_metallic: float | None = None
    uv_target_roughness_path: str | None = None
    uv_target_metallic_path: str | None = None
    uv_target_confidence_path: str | None = None
    target_source_type: str = "unknown"
    target_is_prior_copy: bool = False
    target_prior_identity: float | None = None
    target_quality_tier: str = "unknown"
    target_confidence_summary: dict[str, Any] = field(default_factory=dict)
    target_confidence_mean: float | None = None
    target_confidence_nonzero_rate: float | None = None
    target_coverage: float | None = None
    canonical_views_json: str | None = None
    canonical_buffer_root: str | None = None
    paper_split: str | None = None
    material_family: str = "unknown"
    thin_boundary_flag: bool = False
    lighting_bank_id: str = "unknown"
    view_supervision_ready: bool = False
    valid_view_count: int = 0
    supervision_tier: str = "strong"
    default_split: str = "train"
    bundle_root: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalAssetRecordV1":
        known_fields = {
            "object_id",
            "generator_id",
            "license_bucket",
            "supervision_role",
            "has_material_prior",
            "prior_mode",
            "canonical_mesh_path",
            "canonical_glb_path",
            "uv_albedo_path",
            "uv_normal_path",
            "uv_prior_roughness_path",
            "uv_prior_metallic_path",
            "scalar_prior_roughness",
            "scalar_prior_metallic",
            "uv_target_roughness_path",
            "uv_target_metallic_path",
            "uv_target_confidence_path",
            "target_source_type",
            "target_is_prior_copy",
            "target_prior_identity",
            "target_quality_tier",
            "target_confidence_summary",
            "target_confidence_mean",
            "target_confidence_nonzero_rate",
            "target_coverage",
            "canonical_views_json",
            "canonical_buffer_root",
            "paper_split",
            "material_family",
            "thin_boundary_flag",
            "lighting_bank_id",
            "view_supervision_ready",
            "valid_view_count",
            "supervision_tier",
            "default_split",
            "bundle_root",
        }
        metadata = {k: v for k, v in payload.items() if k not in known_fields}
        object_id = str(payload.get("object_id") or payload.get("id") or payload.get("source_uid"))
        if not object_id:
            raise ValueError("canonical material record is missing object_id")
        return cls(
            object_id=object_id,
            generator_id=str(payload.get("generator_id", "sf3d")),
            license_bucket=str(payload.get("license_bucket", "unknown")),
            supervision_role=str(payload.get("supervision_role", "unknown")),
            has_material_prior=parse_bool(payload.get("has_material_prior")),
            prior_mode=str(payload.get("prior_mode", "none")),
            canonical_mesh_path=parse_optional_str(payload.get("canonical_mesh_path")),
            canonical_glb_path=parse_optional_str(payload.get("canonical_glb_path")),
            uv_albedo_path=parse_optional_str(payload.get("uv_albedo_path")),
            uv_normal_path=parse_optional_str(payload.get("uv_normal_path")),
            uv_prior_roughness_path=parse_optional_str(
                payload.get("uv_prior_roughness_path")
            ),
            uv_prior_metallic_path=parse_optional_str(
                payload.get("uv_prior_metallic_path")
            ),
            scalar_prior_roughness=parse_optional_float(
                payload.get("scalar_prior_roughness")
            ),
            scalar_prior_metallic=parse_optional_float(
                payload.get("scalar_prior_metallic")
            ),
            uv_target_roughness_path=parse_optional_str(
                payload.get("uv_target_roughness_path")
            ),
            uv_target_metallic_path=parse_optional_str(
                payload.get("uv_target_metallic_path")
            ),
            uv_target_confidence_path=parse_optional_str(
                payload.get("uv_target_confidence_path")
            ),
            target_source_type=str(payload.get("target_source_type", "unknown")),
            target_is_prior_copy=parse_bool(payload.get("target_is_prior_copy")),
            target_prior_identity=parse_optional_float(payload.get("target_prior_identity")),
            target_quality_tier=str(payload.get("target_quality_tier", "unknown")),
            target_confidence_summary=parse_optional_dict(payload.get("target_confidence_summary")),
            target_confidence_mean=parse_optional_float(payload.get("target_confidence_mean")),
            target_confidence_nonzero_rate=parse_optional_float(payload.get("target_confidence_nonzero_rate")),
            target_coverage=parse_optional_float(payload.get("target_coverage")),
            canonical_views_json=parse_optional_str(payload.get("canonical_views_json")),
            canonical_buffer_root=parse_optional_str(payload.get("canonical_buffer_root")),
            paper_split=parse_optional_str(payload.get("paper_split")),
            material_family=str(payload.get("material_family", "unknown")),
            thin_boundary_flag=parse_bool(payload.get("thin_boundary_flag")),
            lighting_bank_id=str(payload.get("lighting_bank_id", "unknown")),
            view_supervision_ready=parse_bool(payload.get("view_supervision_ready")),
            valid_view_count=int(payload.get("valid_view_count", 0) or 0),
            supervision_tier=str(payload.get("supervision_tier", "strong")),
            default_split=str(payload.get("default_split", payload.get("split", "train"))),
            bundle_root=parse_optional_str(payload.get("bundle_root")),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "object_id": self.object_id,
            "generator_id": self.generator_id,
            "license_bucket": self.license_bucket,
            "supervision_role": self.supervision_role,
            "has_material_prior": self.has_material_prior,
            "prior_mode": self.prior_mode,
            "canonical_mesh_path": self.canonical_mesh_path,
            "canonical_glb_path": self.canonical_glb_path,
            "uv_albedo_path": self.uv_albedo_path,
            "uv_normal_path": self.uv_normal_path,
            "uv_prior_roughness_path": self.uv_prior_roughness_path,
            "uv_prior_metallic_path": self.uv_prior_metallic_path,
            "scalar_prior_roughness": self.scalar_prior_roughness,
            "scalar_prior_metallic": self.scalar_prior_metallic,
            "uv_target_roughness_path": self.uv_target_roughness_path,
            "uv_target_metallic_path": self.uv_target_metallic_path,
            "uv_target_confidence_path": self.uv_target_confidence_path,
            "target_source_type": self.target_source_type,
            "target_is_prior_copy": self.target_is_prior_copy,
            "target_prior_identity": self.target_prior_identity,
            "target_quality_tier": self.target_quality_tier,
            "target_confidence_summary": self.target_confidence_summary,
            "target_confidence_mean": self.target_confidence_mean,
            "target_confidence_nonzero_rate": self.target_confidence_nonzero_rate,
            "target_coverage": self.target_coverage,
            "canonical_views_json": self.canonical_views_json,
            "canonical_buffer_root": self.canonical_buffer_root,
            "paper_split": self.paper_split,
            "material_family": self.material_family,
            "thin_boundary_flag": self.thin_boundary_flag,
            "lighting_bank_id": self.lighting_bank_id,
            "view_supervision_ready": self.view_supervision_ready,
            "valid_view_count": self.valid_view_count,
            "supervision_tier": self.supervision_tier,
            "default_split": self.default_split,
            "bundle_root": self.bundle_root,
        }
        payload.update(self.metadata)
        return payload


@dataclass
class CanonicalManifestV1:
    manifest_version: str
    records: list[CanonicalAssetRecordV1]
    bundle_root: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
