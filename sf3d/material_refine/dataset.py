from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from sf3d.material_refine.data_utils import filter_records, select_split_records
from sf3d.material_refine.io import (
    default_normal_map,
    load_canonical_manifest,
    pil_to_tensor,
    resize_tensor_image,
)
from sf3d.material_refine.types import CanonicalAssetRecordV1

DEFAULT_VIEWS = ["front_studio", "three_quarter_indoor", "side_neon"]


def _resolve_record_path(
    record: CanonicalAssetRecordV1,
    value: str | None,
    manifest_dir: Path,
) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if record.bundle_root:
        bundle_path = Path(record.bundle_root)
        if not bundle_path.is_absolute():
            bundle_path = manifest_dir / bundle_path
        candidate = bundle_path / path
        if candidate.exists():
            return candidate
    return manifest_dir / path


def _load_png_tensor(
    path: Path | None,
    *,
    grayscale: bool = False,
) -> torch.Tensor | None:
    if path is None or not path.exists():
        return None
    return pil_to_tensor(Image.open(path), grayscale=grayscale)


def _load_array_like(path: Path | None) -> torch.Tensor | None:
    if path is None or not path.exists():
        return None
    if path.suffix == ".npy":
        array = np.load(path)
    elif path.suffix == ".npz":
        payload = np.load(path)
        if "arr_0" in payload:
            array = payload["arr_0"]
        elif "uv" in payload:
            array = payload["uv"]
        else:
            array = payload[list(payload.keys())[0]]
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return pil_to_tensor(Image.open(path))
    else:
        return None
    tensor = torch.from_numpy(np.asarray(array)).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] in {2, 3, 4}:
        tensor = tensor.permute(2, 0, 1)
    return tensor


def _load_view_names(record: CanonicalAssetRecordV1, manifest_dir: Path) -> list[str]:
    views_path = _resolve_record_path(record, record.canonical_views_json, manifest_dir)
    if views_path is None or not views_path.exists():
        return list(DEFAULT_VIEWS)
    payload = json.loads(views_path.read_text())
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return [str(item["name"]) for item in payload if "name" in item]
        return [str(item) for item in payload]
    return list(DEFAULT_VIEWS)


def _load_view_bundle(
    record: CanonicalAssetRecordV1,
    manifest_dir: Path,
    buffer_resolution: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    buffer_root = _resolve_record_path(record, record.canonical_buffer_root, manifest_dir)
    view_names = _load_view_names(record, manifest_dir)
    if buffer_root is None or not buffer_root.exists():
        zero_features = torch.zeros(1, 11, buffer_resolution, buffer_resolution)
        zero_mask = torch.zeros(1, 1, buffer_resolution, buffer_resolution)
        return zero_features, None, zero_mask, {
            "view_names": view_names,
            "view_targets": None,
        }

    features = []
    view_uvs = []
    view_masks = []
    view_targets = []

    for view_name in view_names:
        view_dir = buffer_root / view_name
        rgba = _load_png_tensor(view_dir / "rgba.png")
        if rgba is None:
            rgba = _load_png_tensor(view_dir / "rgb.png")
        if rgba is None:
            rgba = torch.zeros(3, buffer_resolution, buffer_resolution)
        rgba = resize_tensor_image(rgba[:3], buffer_resolution)

        mask = _load_png_tensor(view_dir / "mask.png", grayscale=True)
        if mask is None:
            rgba_full = Image.open(view_dir / "rgba.png") if (view_dir / "rgba.png").exists() else None
            if rgba_full is not None and rgba_full.mode == "RGBA":
                mask = pil_to_tensor(rgba_full.getchannel("A"), grayscale=True)
        if mask is None:
            mask = torch.ones(1, rgba.shape[-2], rgba.shape[-1], dtype=torch.float32)
        mask = resize_tensor_image(mask, buffer_resolution)

        depth = _load_array_like(view_dir / "depth.npy")
        if depth is None:
            depth = _load_array_like(view_dir / "depth.npz")
        if depth is None:
            depth = _load_png_tensor(view_dir / "depth.png", grayscale=True)
        if depth is None:
            depth = torch.zeros(1, buffer_resolution, buffer_resolution)
        depth = resize_tensor_image(depth[:1], buffer_resolution)

        normal = _load_array_like(view_dir / "normal.npy")
        if normal is None:
            normal = _load_array_like(view_dir / "normal.npz")
        if normal is None:
            normal = _load_png_tensor(view_dir / "normal.png")
        if normal is None:
            normal = default_normal_map(buffer_resolution)
        normal = resize_tensor_image(normal[:3], buffer_resolution)

        position = _load_array_like(view_dir / "position.npy")
        if position is None:
            position = _load_array_like(view_dir / "position.npz")
        if position is None:
            position = torch.zeros(3, buffer_resolution, buffer_resolution)
        position = resize_tensor_image(position[:3], buffer_resolution)

        uv = _load_array_like(view_dir / "uv.npy")
        if uv is None:
            uv = _load_array_like(view_dir / "uv.npz")
        if uv is not None:
            uv = resize_tensor_image(uv[:2], buffer_resolution).permute(1, 2, 0).contiguous()
        view_uvs.append(uv)
        view_masks.append(mask)

        roughness_gt = _load_png_tensor(view_dir / "roughness.png", grayscale=True)
        metallic_gt = _load_png_tensor(view_dir / "metallic.png", grayscale=True)
        if roughness_gt is not None and metallic_gt is not None:
            roughness_gt = resize_tensor_image(roughness_gt[:1], buffer_resolution)
            metallic_gt = resize_tensor_image(metallic_gt[:1], buffer_resolution)
            view_targets.append(torch.cat([roughness_gt, metallic_gt], dim=0))
        else:
            view_targets.append(None)

        features.append(torch.cat([rgba, mask, depth, normal, position], dim=0))

    stacked_features = torch.stack(features, dim=0)
    stacked_masks = torch.stack(view_masks, dim=0)

    if any(uv is not None for uv in view_uvs):
        uv_tensors = []
        for uv in view_uvs:
            if uv is None:
                uv = torch.zeros(buffer_resolution, buffer_resolution, 2)
            uv_tensors.append(uv)
        stacked_uvs = torch.stack(uv_tensors, dim=0)
    else:
        stacked_uvs = None

    if any(target is not None for target in view_targets):
        target_tensors = []
        for target in view_targets:
            if target is None:
                target = torch.zeros(2, buffer_resolution, buffer_resolution)
            target_tensors.append(target)
        stacked_targets = torch.stack(target_tensors, dim=0)
    else:
        stacked_targets = None

    return stacked_features, stacked_uvs, stacked_masks, {
        "view_names": view_names,
        "view_targets": stacked_targets,
    }


class CanonicalMaterialDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str = "train",
        split_strategy: str = "auto",
        hash_val_ratio: float = 0.1,
        hash_test_ratio: float = 0.1,
        source_names: list[str] | None = None,
        generator_ids: list[str] | None = None,
        supervision_tiers: list[str] | None = None,
        license_buckets: list[str] | None = None,
        require_prior: bool | None = None,
        max_records: int | None = None,
        atlas_size: int = 512,
        buffer_resolution: int = 256,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        manifest = load_canonical_manifest(self.manifest_path)
        self.atlas_size = atlas_size
        self.buffer_resolution = buffer_resolution
        records = filter_records(
            manifest.records,
            generator_ids=generator_ids,
            source_names=source_names,
            supervision_tiers=supervision_tiers,
            license_buckets=license_buckets,
            require_prior=require_prior,
        )
        self.records = select_split_records(
            records,
            split=split,
            split_strategy=split_strategy,
            hash_val_ratio=hash_val_ratio,
            hash_test_ratio=hash_test_ratio,
        )
        if max_records is not None:
            self.records = self.records[:max_records]
        if not self.records:
            raise RuntimeError(f"No canonical material records found for split={split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        albedo_path = _resolve_record_path(record, record.uv_albedo_path, self.manifest_dir)
        normal_path = _resolve_record_path(record, record.uv_normal_path, self.manifest_dir)
        prior_roughness_path = _resolve_record_path(
            record, record.uv_prior_roughness_path, self.manifest_dir
        )
        prior_metallic_path = _resolve_record_path(
            record, record.uv_prior_metallic_path, self.manifest_dir
        )
        target_roughness_path = _resolve_record_path(
            record, record.uv_target_roughness_path, self.manifest_dir
        )
        target_metallic_path = _resolve_record_path(
            record, record.uv_target_metallic_path, self.manifest_dir
        )
        target_confidence_path = _resolve_record_path(
            record, record.uv_target_confidence_path, self.manifest_dir
        )

        uv_albedo = _load_png_tensor(albedo_path)
        if uv_albedo is None:
            uv_albedo = torch.ones(3, self.atlas_size, self.atlas_size)
        uv_albedo = resize_tensor_image(uv_albedo[:3], self.atlas_size)

        uv_normal = _load_png_tensor(normal_path)
        if uv_normal is None:
            uv_normal = default_normal_map(self.atlas_size)
        uv_normal = resize_tensor_image(uv_normal[:3], self.atlas_size)

        prior_roughness = _load_png_tensor(prior_roughness_path, grayscale=True)
        if prior_roughness is None:
            value = 0.5 if record.scalar_prior_roughness is None else record.scalar_prior_roughness
            prior_roughness = torch.full(
                (1, self.atlas_size, self.atlas_size),
                float(value),
                dtype=torch.float32,
            )
        prior_roughness = resize_tensor_image(prior_roughness[:1], self.atlas_size)

        prior_metallic = _load_png_tensor(prior_metallic_path, grayscale=True)
        if prior_metallic is None:
            value = 0.0 if record.scalar_prior_metallic is None else record.scalar_prior_metallic
            prior_metallic = torch.full(
                (1, self.atlas_size, self.atlas_size),
                float(value),
                dtype=torch.float32,
            )
        prior_metallic = resize_tensor_image(prior_metallic[:1], self.atlas_size)

        target_roughness = _load_png_tensor(target_roughness_path, grayscale=True)
        target_metallic = _load_png_tensor(target_metallic_path, grayscale=True)
        target_confidence = _load_png_tensor(target_confidence_path, grayscale=True)

        if target_roughness is None or target_metallic is None:
            target_roughness = prior_roughness.clone()
            target_metallic = prior_metallic.clone()
            target_confidence = torch.ones(
                1, self.atlas_size, self.atlas_size, dtype=torch.float32
            )
        else:
            target_roughness = resize_tensor_image(target_roughness[:1], self.atlas_size)
            target_metallic = resize_tensor_image(target_metallic[:1], self.atlas_size)
            if target_confidence is None:
                target_confidence = torch.ones_like(target_roughness)
            else:
                target_confidence = resize_tensor_image(
                    target_confidence[:1], self.atlas_size
                )

        prior_confidence = torch.ones_like(prior_roughness)
        if record.prior_mode == "none" or not record.has_material_prior:
            prior_confidence.zero_()
        if target_confidence.max().item() == 0:
            target_confidence.fill_(1.0)

        view_features, view_uvs, view_masks, view_meta = _load_view_bundle(
            record,
            self.manifest_dir,
            self.buffer_resolution,
        )

        return {
            "object_id": record.object_id,
            "split": record.default_split,
            "generator_id": record.generator_id,
            "source_name": str(record.metadata.get("source_name", record.generator_id)),
            "supervision_tier": record.supervision_tier,
            "license_bucket": record.license_bucket,
            "has_material_prior": bool(record.has_material_prior),
            "prior_mode": record.prior_mode,
            "uv_albedo": uv_albedo,
            "uv_normal": uv_normal,
            "uv_prior_roughness": prior_roughness,
            "uv_prior_metallic": prior_metallic,
            "uv_prior_confidence": prior_confidence,
            "uv_target_roughness": target_roughness,
            "uv_target_metallic": target_metallic,
            "uv_target_confidence": target_confidence,
            "view_features": view_features,
            "view_uvs": view_uvs,
            "view_masks": view_masks,
            "view_targets": view_meta["view_targets"],
            "view_names": view_meta["view_names"],
            "metadata": record.metadata,
        }


def collate_material_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "object_id": [sample["object_id"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "generator_id": [sample["generator_id"] for sample in samples],
        "source_name": [sample["source_name"] for sample in samples],
        "supervision_tier": [sample["supervision_tier"] for sample in samples],
        "license_bucket": [sample["license_bucket"] for sample in samples],
        "has_material_prior": [sample["has_material_prior"] for sample in samples],
        "prior_mode": [sample["prior_mode"] for sample in samples],
        "view_names": [sample["view_names"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
    }
    tensor_keys = [
        "uv_albedo",
        "uv_normal",
        "uv_prior_roughness",
        "uv_prior_metallic",
        "uv_prior_confidence",
        "uv_target_roughness",
        "uv_target_metallic",
        "uv_target_confidence",
    ]
    for key in tensor_keys:
        batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

    max_views = max(sample["view_features"].shape[0] for sample in samples)
    feature_shape = samples[0]["view_features"].shape[1:]
    padded_features = []
    padded_masks = []
    padded_uvs = []
    padded_targets = []
    has_uvs = any(sample["view_uvs"] is not None for sample in samples)
    has_targets = any(sample["view_targets"] is not None for sample in samples)

    for sample in samples:
        views = sample["view_features"]
        masks = sample["view_masks"]
        pad_count = max_views - views.shape[0]
        if pad_count > 0:
            views = torch.cat(
                [views, torch.zeros((pad_count, *feature_shape), dtype=views.dtype)],
                dim=0,
            )
            masks = torch.cat(
                [
                    masks,
                    torch.zeros(
                        (pad_count, *masks.shape[1:]),
                        dtype=masks.dtype,
                    ),
                ],
                dim=0,
            )
        padded_features.append(views)
        padded_masks.append(masks)

        if has_uvs:
            uvs = sample["view_uvs"]
            if uvs is None:
                uvs = torch.zeros(
                    max_views,
                    feature_shape[-2],
                    feature_shape[-1],
                    2,
                    dtype=torch.float32,
                )
            elif pad_count > 0:
                uvs = torch.cat(
                    [
                        uvs,
                        torch.zeros(
                            (pad_count, *uvs.shape[1:]),
                            dtype=uvs.dtype,
                        ),
                    ],
                    dim=0,
                )
            padded_uvs.append(uvs)

        if has_targets:
            targets = sample["view_targets"]
            if targets is None:
                targets = torch.zeros(
                    max_views,
                    2,
                    feature_shape[-2],
                    feature_shape[-1],
                    dtype=torch.float32,
                )
            elif pad_count > 0:
                targets = torch.cat(
                    [
                        targets,
                        torch.zeros(
                            (pad_count, *targets.shape[1:]),
                            dtype=targets.dtype,
                        ),
                    ],
                    dim=0,
                )
            padded_targets.append(targets)

    batch["view_features"] = torch.stack(padded_features, dim=0)
    batch["view_masks"] = torch.stack(padded_masks, dim=0)
    batch["view_uvs"] = torch.stack(padded_uvs, dim=0) if has_uvs else None
    batch["view_targets"] = (
        torch.stack(padded_targets, dim=0) if has_targets else None
    )
    return batch
