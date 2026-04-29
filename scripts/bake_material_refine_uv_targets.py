#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_gray(path: Path, value: np.ndarray) -> None:
    _ensure_parent(path)
    clipped = np.clip(value, 0.0, 1.0)
    Image.fromarray((clipped * 255.0).round().astype(np.uint8), mode="L").save(path)


def _load_image(path: Path, mode: str, resolution: int | None = None) -> Image.Image | None:
    if not path.exists():
        return None
    image = Image.open(path).convert(mode)
    if resolution is not None and image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.BILINEAR)
    return image


def load_gray_image(path: Path | str | None, resolution: int | None = None) -> np.ndarray | None:
    if path in (None, "", "None"):
        return None
    image = _load_image(Path(path), "L", resolution)
    if image is None:
        return None
    return np.asarray(image, dtype=np.float32) / 255.0


def load_uv_map(path: Path | str | None, resolution: int | None = None) -> np.ndarray | None:
    if path in (None, "", "None"):
        return None
    image = _load_image(Path(path), "RGB", resolution)
    if image is None:
        return None
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    return rgb[..., :2]


def _load_alpha_mask(uv_albedo_path: Path, atlas_resolution: int) -> np.ndarray:
    rgba = _load_image(uv_albedo_path, "RGBA", atlas_resolution)
    if rgba is None:
        return np.ones((atlas_resolution, atlas_resolution), dtype=np.float32)
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
    return alpha


def _constant_map(value: float, resolution: int) -> np.ndarray:
    return np.full((resolution, resolution), float(value), dtype=np.float32)


def _synthesize_prior_from_target(target: np.ndarray, default_value: float) -> np.ndarray:
    pil = Image.fromarray((np.clip(target, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    low = pil.resize((max(4, pil.width // 8), max(4, pil.height // 8)), Image.BILINEAR).resize(pil.size, Image.BILINEAR)
    low_arr = np.asarray(low, dtype=np.float32) / 255.0
    prior = 0.75 * low_arr + 0.25 * float(default_value)
    return np.clip(prior, 0.0, 1.0).astype(np.float32)


def _bake_from_multiview(
    canonical_buffer_root: Path,
    atlas_resolution: int,
    default_roughness: float,
    default_metallic: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not canonical_buffer_root.exists():
        return None, None, None
    sum_r = np.zeros((atlas_resolution, atlas_resolution), dtype=np.float32)
    sum_m = np.zeros((atlas_resolution, atlas_resolution), dtype=np.float32)
    sum_w = np.zeros((atlas_resolution, atlas_resolution), dtype=np.float32)
    for view_dir in sorted(path for path in canonical_buffer_root.iterdir() if path.is_dir()):
        uv = load_uv_map(view_dir / "uv.png")
        mask = load_gray_image(view_dir / "mask.png")
        visibility = load_gray_image(view_dir / "visibility.png")
        roughness = load_gray_image(view_dir / "roughness.png")
        metallic = load_gray_image(view_dir / "metallic.png")
        if any(value is None for value in (uv, mask, visibility, roughness, metallic)):
            continue
        assert uv is not None
        assert mask is not None
        assert visibility is not None
        assert roughness is not None
        assert metallic is not None
        valid = (
            np.isfinite(uv[..., 0])
            & np.isfinite(uv[..., 1])
            & (mask > 0.5)
            & (visibility > 0.5)
            & (uv[..., 0] >= 0.0)
            & (uv[..., 0] <= 1.0)
            & (uv[..., 1] >= 0.0)
            & (uv[..., 1] <= 1.0)
        )
        if not bool(valid.any()):
            continue
        u = np.clip(np.rint(uv[..., 0][valid] * (atlas_resolution - 1)).astype(np.int32), 0, atlas_resolution - 1)
        v = np.clip(np.rint((1.0 - uv[..., 1][valid]) * (atlas_resolution - 1)).astype(np.int32), 0, atlas_resolution - 1)
        weight = np.clip(mask[valid] * visibility[valid], 0.0, 1.0).astype(np.float32)
        weight = np.where(weight > 0.0, weight, 1.0)
        np.add.at(sum_r, (v, u), roughness[valid] * weight)
        np.add.at(sum_m, (v, u), metallic[valid] * weight)
        np.add.at(sum_w, (v, u), weight)
    if not bool((sum_w > 0).any()):
        return None, None, None
    target_r = np.where(sum_w > 0, sum_r / np.maximum(sum_w, 1e-6), float(default_roughness)).astype(np.float32)
    target_m = np.where(sum_w > 0, sum_m / np.maximum(sum_w, 1e-6), float(default_metallic)).astype(np.float32)
    confidence = np.zeros_like(sum_w, dtype=np.float32)
    if float(sum_w.max()) > 0:
        confidence = np.clip(sum_w / float(sum_w.max()), 0.0, 1.0)
    return target_r, target_m, confidence.astype(np.float32)


def _similarity(prior_r: np.ndarray, prior_m: np.ndarray, target_r: np.ndarray, target_m: np.ndarray, active_mask: np.ndarray) -> float:
    active = active_mask > 0.5
    if not bool(active.any()):
        return 1.0
    delta = 0.5 * (np.abs(prior_r[active] - target_r[active]) + np.abs(prior_m[active] - target_m[active]))
    return float(np.clip(1.0 - float(delta.mean()), 0.0, 1.0))


def _confidence_summary(confidence: np.ndarray, active_mask: np.ndarray) -> dict[str, float]:
    active = active_mask > 0.5
    values = confidence[active] if bool(active.any()) else confidence.reshape(-1)
    if values.size <= 0:
        values = np.zeros((1,), dtype=np.float32)
    values = values.astype(np.float32)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p50": float(np.quantile(values, 0.50)),
        "max": float(values.max()),
        "nonzero_rate": float((values > 0.0).mean()),
        "high_conf_rate": float((values >= 0.80).mean()),
        "active_mean": float(values[values > 0.0].mean()) if bool((values > 0.0).any()) else 0.0,
    }


def bake_uv_targets(
    *,
    bundle_dir: Path,
    uv_albedo_path: Path,
    uv_prior_roughness_path: Path | None,
    uv_prior_metallic_path: Path | None,
    uv_target_roughness_source_path: Path | None,
    uv_target_metallic_source_path: Path | None,
    canonical_buffer_root: Path | None,
    atlas_resolution: int,
    default_roughness: float,
    default_metallic: float,
    synthesize_nontrivial_prior: bool,
    allow_prior_copy_fallback: bool,
) -> dict[str, Any]:
    uv_dir = bundle_dir / "uv"
    uv_dir.mkdir(parents=True, exist_ok=True)
    prior_r_path = Path(uv_prior_roughness_path) if uv_prior_roughness_path else uv_dir / "uv_prior_roughness.png"
    prior_m_path = Path(uv_prior_metallic_path) if uv_prior_metallic_path else uv_dir / "uv_prior_metallic.png"
    target_r_path = uv_dir / "uv_target_roughness.png"
    target_m_path = uv_dir / "uv_target_metallic.png"
    confidence_path = uv_dir / "uv_target_confidence.png"

    active_mask = _load_alpha_mask(uv_albedo_path, atlas_resolution)
    target_r = load_gray_image(uv_target_roughness_source_path, atlas_resolution)
    target_m = load_gray_image(uv_target_metallic_source_path, atlas_resolution)
    source_type = ""
    confidence = None

    if target_r is not None and target_m is not None:
        source_type = "gt_render_baked"
        confidence = np.where(active_mask > 0.5, 0.95, 0.0).astype(np.float32)
    else:
        baked = _bake_from_multiview(
            Path(canonical_buffer_root) if canonical_buffer_root else Path("."),
            atlas_resolution,
            default_roughness,
            default_metallic,
        )
        if baked[0] is not None and baked[1] is not None and baked[2] is not None:
            target_r, target_m, confidence = baked
            source_type = "pseudo_from_multiview"
        elif allow_prior_copy_fallback:
            source_type = "copied_from_prior"
        else:
            source_type = "default_fallback"

    prior_r = load_gray_image(prior_r_path, atlas_resolution)
    prior_m = load_gray_image(prior_m_path, atlas_resolution)
    if prior_r is None or prior_m is None:
        if synthesize_nontrivial_prior and target_r is not None and target_m is not None:
            prior_r = _synthesize_prior_from_target(target_r, default_roughness)
            prior_m = _synthesize_prior_from_target(target_m, default_metallic)
        else:
            prior_r = _constant_map(default_roughness, atlas_resolution)
            prior_m = _constant_map(default_metallic, atlas_resolution)
        _save_gray(prior_r_path, prior_r)
        _save_gray(prior_m_path, prior_m)

    assert prior_r is not None
    assert prior_m is not None
    if target_r is None or target_m is None:
        target_r = prior_r.copy() if source_type == "copied_from_prior" else _constant_map(default_roughness, atlas_resolution)
        target_m = prior_m.copy() if source_type == "copied_from_prior" else _constant_map(default_metallic, atlas_resolution)
    if confidence is None:
        confidence = np.where(active_mask > 0.5, 0.90 if source_type != "default_fallback" else 0.50, 0.0).astype(np.float32)

    _save_gray(target_r_path, target_r)
    _save_gray(target_m_path, target_m)
    _save_gray(confidence_path, confidence)

    identity = _similarity(prior_r, prior_m, target_r, target_m, active_mask)
    target_is_prior_copy = bool(source_type == "copied_from_prior" or identity >= 0.995)
    coverage_mask = (active_mask > 0.5) & (confidence > 0.0)
    target_coverage = float(coverage_mask.mean())
    confidence_summary = _confidence_summary(confidence, active_mask)
    payload = {
        "generated_at_utc": json.dumps(utc_now())[1:-1] if False else None,
        "uv_prior_roughness_path": str(prior_r_path.resolve()),
        "uv_prior_metallic_path": str(prior_m_path.resolve()),
        "uv_target_roughness_path": str(target_r_path.resolve()),
        "uv_target_metallic_path": str(target_m_path.resolve()),
        "uv_target_confidence_path": str(confidence_path.resolve()),
        "target_source_type": source_type,
        "target_is_prior_copy": target_is_prior_copy,
        "target_prior_identity": float(identity),
        "target_confidence_summary": confidence_summary,
        "target_confidence_mean": float(confidence_summary["mean"]),
        "target_confidence_nonzero_rate": float(confidence_summary["nonzero_rate"]),
        "target_coverage": target_coverage,
    }
    payload.pop("generated_at_utc", None)
    return payload
