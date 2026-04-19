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


@dataclass
class CanonicalAssetRecordV1:
    object_id: str
    generator_id: str = "sf3d"
    license_bucket: str = "unknown"
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
    canonical_views_json: str | None = None
    canonical_buffer_root: str | None = None
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
            "canonical_views_json",
            "canonical_buffer_root",
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
            canonical_views_json=parse_optional_str(payload.get("canonical_views_json")),
            canonical_buffer_root=parse_optional_str(payload.get("canonical_buffer_root")),
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
            "canonical_views_json": self.canonical_views_json,
            "canonical_buffer_root": self.canonical_buffer_root,
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

