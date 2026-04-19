from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image

from sf3d.material_refine.types import (
    CanonicalAssetRecordV1,
    CanonicalManifestV1,
)


def _resolve_path(
    value: str | None,
    *,
    manifest_dir: Path,
    bundle_root: Path | None = None,
) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if bundle_root is not None:
        candidate = bundle_root / path
        if candidate.exists():
            return candidate
    return manifest_dir / path


def load_canonical_manifest(manifest_path: str | Path) -> CanonicalManifestV1:
    path = Path(manifest_path)
    payload = json.loads(path.read_text())
    metadata = {}
    bundle_root = None
    if isinstance(payload, dict):
        records_payload = payload.get("records") or payload.get("objects") or payload.get("rows")
        if not isinstance(records_payload, list):
            raise TypeError(f"Unsupported manifest payload in {path}")
        bundle_root_value = (
            payload.get("canonical_bundle_root")
            or payload.get("bundle_root")
            or payload.get("canonical_root")
        )
        if bundle_root_value:
            bundle_root = _resolve_path(
                str(bundle_root_value),
                manifest_dir=path.parent,
                bundle_root=None,
            )
        metadata = {
            k: v for k, v in payload.items() if k not in {"records", "objects", "rows"}
        }
    elif isinstance(payload, list):
        records_payload = payload
    else:
        raise TypeError(f"Unsupported manifest payload in {path}")

    records = [CanonicalAssetRecordV1.from_dict(record) for record in records_payload]
    if bundle_root is not None:
        for record in records:
            if record.bundle_root is None:
                record.bundle_root = str(bundle_root)
    return CanonicalManifestV1(
        manifest_version=str(metadata.get("manifest_version", "canonical_material_refine_v1")),
        records=records,
        bundle_root=bundle_root,
        metadata=metadata,
    )


def pil_to_tensor(image: Image.Image, grayscale: bool = False) -> torch.Tensor:
    if grayscale:
        image = image.convert("L")
        array = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)
    image = image.convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_to_pil(tensor: torch.Tensor, grayscale: bool = False) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1)
    if grayscale:
        data = (array.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(data, mode="L")
    data = (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(data, mode="RGB")


def resize_tensor_image(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.shape[-2:] == (size, size):
        return image
    return torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


def default_normal_map(size: int) -> torch.Tensor:
    normal = torch.zeros(3, size, size, dtype=torch.float32)
    normal[0].fill_(0.5)
    normal[1].fill_(0.5)
    normal[2].fill_(1.0)
    return normal


def _load_texture_image(texture: Any) -> Image.Image | None:
    if texture is None:
        return None
    if isinstance(texture, Image.Image):
        return texture
    if isinstance(texture, (str, Path)):
        return Image.open(texture)
    return None


def _load_metallic_roughness_texture(texture: Image.Image, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = texture.convert("RGB").resize((size, size))
    array = np.asarray(image).astype(np.float32) / 255.0
    metallic = torch.from_numpy(array[..., 0]).unsqueeze(0)
    roughness = torch.from_numpy(array[..., 1]).unsqueeze(0)
    return roughness, metallic


def pack_metallic_roughness_texture(
    roughness: torch.Tensor,
    metallic: torch.Tensor,
) -> Image.Image:
    rough = roughness.detach().cpu().clamp(0, 1).squeeze(0).numpy()
    metal = metallic.detach().cpu().clamp(0, 1).squeeze(0).numpy()
    packed = np.zeros((rough.shape[0], rough.shape[1], 3), dtype=np.float32)
    packed[..., 0] = metal
    packed[..., 1] = rough
    packed_uint8 = (packed * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(packed_uint8, mode="RGB")


def extract_mesh_material_inputs(
    mesh: trimesh.Trimesh,
    atlas_size: int,
) -> dict[str, torch.Tensor]:
    material = None
    if hasattr(mesh.visual, "material"):
        material = mesh.visual.material
    albedo = torch.ones(3, atlas_size, atlas_size, dtype=torch.float32)
    normal = default_normal_map(atlas_size)
    roughness = torch.full((1, atlas_size, atlas_size), 0.5, dtype=torch.float32)
    metallic = torch.zeros(1, atlas_size, atlas_size, dtype=torch.float32)
    prior_confidence = torch.zeros(1, atlas_size, atlas_size, dtype=torch.float32)

    if material is None:
        return {
            "uv_albedo": albedo,
            "uv_normal": normal,
            "uv_prior_roughness": roughness,
            "uv_prior_metallic": metallic,
            "uv_prior_confidence": prior_confidence,
        }

    basecolor = _load_texture_image(getattr(material, "baseColorTexture", None))
    if basecolor is not None:
        albedo = resize_tensor_image(pil_to_tensor(basecolor), atlas_size)

    bump = _load_texture_image(getattr(material, "normalTexture", None))
    if bump is not None:
        normal = resize_tensor_image(pil_to_tensor(bump), atlas_size)

    metallic_roughness = _load_texture_image(
        getattr(material, "metallicRoughnessTexture", None)
    )
    if metallic_roughness is not None:
        roughness, metallic = _load_metallic_roughness_texture(
            metallic_roughness, atlas_size
        )
        prior_confidence.fill_(1.0)
    else:
        roughness_value = getattr(material, "roughnessFactor", None)
        metallic_value = getattr(material, "metallicFactor", None)
        if roughness_value is not None:
            roughness.fill_(float(roughness_value))
            prior_confidence.fill_(1.0)
        if metallic_value is not None:
            metallic.fill_(float(metallic_value))
            prior_confidence.fill_(1.0)

    return {
        "uv_albedo": albedo,
        "uv_normal": normal,
        "uv_prior_roughness": roughness,
        "uv_prior_metallic": metallic,
        "uv_prior_confidence": prior_confidence,
    }


def apply_refined_maps_to_mesh(
    mesh: trimesh.Trimesh,
    refined_roughness: torch.Tensor,
    refined_metallic: torch.Tensor,
    *,
    basecolor: torch.Tensor | None = None,
    normal_map: torch.Tensor | None = None,
) -> trimesh.Trimesh:
    refined_mesh = copy.deepcopy(mesh)
    old_material = getattr(refined_mesh.visual, "material", None)
    basecolor_texture = None
    normal_texture = None
    if basecolor is not None:
        basecolor_texture = tensor_to_pil(basecolor)
    elif old_material is not None:
        basecolor_texture = _load_texture_image(getattr(old_material, "baseColorTexture", None))

    if normal_map is not None:
        normal_texture = tensor_to_pil(normal_map)
    elif old_material is not None:
        normal_texture = _load_texture_image(getattr(old_material, "normalTexture", None))

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=basecolor_texture,
        normalTexture=normal_texture,
        roughnessFactor=1.0,
        metallicFactor=1.0,
        metallicRoughnessTexture=pack_metallic_roughness_texture(
            refined_roughness, refined_metallic
        ),
    )
    refined_mesh.visual.material = material
    return refined_mesh


def save_atlas_bundle(
    output_dir: Path,
    *,
    baseline_roughness: torch.Tensor,
    baseline_metallic: torch.Tensor,
    refined_roughness: torch.Tensor,
    refined_metallic: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline_roughness": output_dir / "baseline_roughness.png",
        "baseline_metallic": output_dir / "baseline_metallic.png",
        "refined_roughness": output_dir / "refined_roughness.png",
        "refined_metallic": output_dir / "refined_metallic.png",
    }
    tensor_to_pil(baseline_roughness, grayscale=True).save(paths["baseline_roughness"])
    tensor_to_pil(baseline_metallic, grayscale=True).save(paths["baseline_metallic"])
    tensor_to_pil(refined_roughness, grayscale=True).save(paths["refined_roughness"])
    tensor_to_pil(refined_metallic, grayscale=True).save(paths["refined_metallic"])
    if confidence is not None:
        confidence_path = output_dir / "confidence.png"
        tensor_to_pil(confidence, grayscale=True).save(confidence_path)
        paths["confidence"] = confidence_path
    return paths
