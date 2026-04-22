#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bake non-trivial UV-space roughness/metallic targets for the material refinement dataset."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--uv-albedo-path", type=Path, default=None)
    parser.add_argument("--uv-prior-roughness-path", type=Path, default=None)
    parser.add_argument("--uv-prior-metallic-path", type=Path, default=None)
    parser.add_argument("--uv-target-roughness-source-path", type=Path, default=None)
    parser.add_argument("--uv-target-metallic-source-path", type=Path, default=None)
    parser.add_argument("--canonical-buffer-root", type=Path, default=None)
    parser.add_argument("--atlas-resolution", type=int, default=1024)
    parser.add_argument("--default-roughness", type=float, default=0.5)
    parser.add_argument("--default-metallic", type=float, default=0.0)
    parser.add_argument("--synthesize-nontrivial-prior", action="store_true")
    parser.add_argument("--allow-prior-copy-fallback", action="store_true")
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_gray_image(path: Path | None, resolution: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    image = Image.open(path).convert("L").resize((resolution, resolution), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_rgba_alpha(path: Path | None, resolution: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    image = Image.open(path)
    if "A" not in image.getbands():
        return None
    alpha = image.getchannel("A").resize((resolution, resolution), Image.BILINEAR)
    return np.asarray(alpha, dtype=np.float32) / 255.0


def load_uv_map(path: Path | None, resolution: int) -> np.ndarray | None:
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
    else:
        image = Image.open(path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        return None
    if array.shape[-1] >= 2:
        uv = array[..., :2]
    elif array.shape[0] >= 2:
        uv = np.moveaxis(array[:2], 0, -1)
    else:
        return None
    if uv.shape[:2] != (resolution, resolution):
        uv_image = Image.fromarray((uv.clip(0.0, 1.0) * 255.0).astype(np.uint8), mode="LA")
        uv = np.asarray(uv_image.resize((resolution, resolution), Image.BILINEAR), dtype=np.float32) / 255.0
        uv = uv[..., :2]
    return uv


def save_gray_png(path: Path, values: np.ndarray) -> None:
    clipped = np.clip(values, 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((clipped * 255.0).round().astype(np.uint8), mode="L").save(path)


def confidence_summary(values: np.ndarray) -> dict[str, float]:
    nonzero = values > 1e-6
    high_conf = values >= 0.75
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p50": float(np.quantile(values, 0.50)),
        "max": float(values.max()),
        "nonzero_rate": float(nonzero.mean()),
        "high_conf_rate": float(high_conf.mean()),
    }


def valid_mask_from_sources(
    *,
    resolution: int,
    albedo_path: Path | None,
    target_roughness: np.ndarray | None,
    target_metallic: np.ndarray | None,
) -> np.ndarray:
    alpha = load_rgba_alpha(albedo_path, resolution)
    if alpha is not None:
        return alpha > 1e-3
    if target_roughness is not None and target_metallic is not None:
        signal = np.maximum(target_roughness, target_metallic)
        return signal > 1e-6
    return np.ones((resolution, resolution), dtype=bool)


def soften_confidence(valid_mask: np.ndarray) -> np.ndarray:
    base = valid_mask.astype(np.float32) * 0.95
    if not valid_mask.any():
        return base
    mask_image = Image.fromarray(valid_mask.astype(np.uint8) * 255, mode="L")
    eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(5)), dtype=np.float32) > 0
    boundary = valid_mask & ~eroded
    base[boundary] = 0.68
    return base


def downsample_upsample(values: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray((np.clip(values, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    small = image.resize((size, size), Image.BILINEAR)
    return np.asarray(small.resize(image.size, Image.BILINEAR), dtype=np.float32) / 255.0


def synthesize_prior_from_target(
    values: np.ndarray,
    *,
    scalar_default: float,
    is_metallic: bool,
) -> np.ndarray:
    coarse = downsample_upsample(values, max(8, values.shape[0] // 16))
    ultra_coarse = downsample_upsample(values, max(4, values.shape[0] // 32))
    blurred = np.asarray(
        Image.fromarray((np.clip(values, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=6.0 if not is_metallic else 3.0)
        ),
        dtype=np.float32,
    ) / 255.0
    mean_value = float(values.mean()) if values.size else scalar_default
    scalar_anchor = clamp01(
        (0.80 * scalar_default + 0.20 * mean_value)
        if is_metallic
        else (0.65 * scalar_default + 0.35 * mean_value)
    )
    scalar_map = np.full_like(values, scalar_anchor)
    levels = 4 if is_metallic else 8
    quantized = np.round(values * float(levels)) / float(levels)
    height, width = values.shape
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    directional = np.clip(0.55 + 0.30 * (grid_x - 0.5) - 0.20 * (grid_y - 0.5), 0.0, 1.0)
    if is_metallic:
        compressed = scalar_map + 0.18 * (ultra_coarse - scalar_map)
        mixed = (
            0.45 * compressed
            + 0.25 * scalar_map
            + 0.20 * (directional * compressed)
            + 0.10 * quantized
        )
    else:
        compressed = scalar_map + 0.28 * (coarse - scalar_map)
        mixed = (
            0.35 * compressed
            + 0.25 * scalar_map
            + 0.20 * ultra_coarse
            + 0.10 * blurred
            + 0.10 * (directional * quantized)
        )
    if float(values.std()) < 0.02:
        # Uniform baked maps otherwise recreate the scalar prior almost exactly;
        # inject a deterministic low-frequency corruption so the target remains
        # meaningfully non-trivial without changing the supervision target.
        amplitude = 0.18 if is_metallic else 0.36
        mixed = mixed + amplitude * (directional - 0.5)
    return np.clip(mixed, 0.0, 1.0)


def prior_identity(
    target_roughness: np.ndarray,
    target_metallic: np.ndarray,
    prior_roughness: np.ndarray,
    prior_metallic: np.ndarray,
    confidence: np.ndarray,
) -> float:
    weight = np.maximum(confidence, 1e-6)
    rough_error = float(np.sum(np.abs(target_roughness - prior_roughness) * weight) / np.sum(weight))
    metal_error = float(np.sum(np.abs(target_metallic - prior_metallic) * weight) / np.sum(weight))
    return clamp01(1.0 - 0.5 * (rough_error + metal_error))


def multiview_bake_to_uv(
    *,
    buffer_root: Path | None,
    atlas_resolution: int,
    default_roughness: float,
    default_metallic: float,
) -> dict[str, Any] | None:
    if buffer_root is None or not buffer_root.exists():
        return None
    sum_roughness = np.zeros((atlas_resolution, atlas_resolution), dtype=np.float32)
    sum_metallic = np.zeros((atlas_resolution, atlas_resolution), dtype=np.float32)
    sum_weight = np.zeros((atlas_resolution, atlas_resolution), dtype=np.float32)
    valid_view_count = 0
    field_sources_by_view: dict[str, dict[str, str]] = {}
    metadata_path = buffer_root / "_field_sources.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        views_payload = payload.get("views")
        if isinstance(views_payload, dict):
            for view_name, item in views_payload.items():
                if not isinstance(item, dict):
                    continue
                fields = item.get("fields")
                if isinstance(fields, dict):
                    field_sources_by_view[str(view_name)] = {
                        str(key): str(value) for key, value in fields.items()
                    }

    for view_dir in sorted(path for path in buffer_root.iterdir() if path.is_dir()):
        view_field_sources = field_sources_by_view.get(view_dir.name, {})
        if any(
            view_field_sources.get(field) in {"synthetic_placeholder", "missing", "placeholder"}
            for field in ("uv", "roughness", "metallic")
        ):
            continue
        uv = load_uv_map(
            (view_dir / "uv.npy") if (view_dir / "uv.npy").exists() else (
                (view_dir / "uv.npz") if (view_dir / "uv.npz").exists() else (view_dir / "uv.png")
            ),
            atlas_resolution,
        )
        roughness = load_gray_image(view_dir / "roughness.png", atlas_resolution)
        metallic = load_gray_image(view_dir / "metallic.png", atlas_resolution)
        if uv is None or roughness is None or metallic is None:
            continue
        mask = load_gray_image(view_dir / "mask.png", atlas_resolution)
        if mask is None:
            rgba_alpha = load_rgba_alpha(view_dir / "rgba.png", atlas_resolution)
            mask = rgba_alpha if rgba_alpha is not None else np.ones((atlas_resolution, atlas_resolution), dtype=np.float32)
        visibility = load_gray_image(view_dir / "visibility.png", atlas_resolution)
        if visibility is None:
            visibility = mask
        valid = (
            (mask > 0.5)
            & (visibility > 0.5)
            & (uv[..., 0] >= 0.0)
            & (uv[..., 0] <= 1.0)
            & (uv[..., 1] >= 0.0)
            & (uv[..., 1] <= 1.0)
        )
        if not np.any(valid):
            continue
        valid_view_count += 1
        u = np.clip(np.rint(uv[..., 0][valid] * (atlas_resolution - 1)).astype(np.int32), 0, atlas_resolution - 1)
        v = np.clip(np.rint((1.0 - uv[..., 1][valid]) * (atlas_resolution - 1)).astype(np.int32), 0, atlas_resolution - 1)
        weight = np.clip(mask[valid] * visibility[valid], 0.0, 1.0)
        np.add.at(sum_roughness, (v, u), roughness[valid] * weight)
        np.add.at(sum_metallic, (v, u), metallic[valid] * weight)
        np.add.at(sum_weight, (v, u), weight)

    if valid_view_count <= 0 or float(sum_weight.max()) <= 0.0:
        return None

    roughness_target = np.full((atlas_resolution, atlas_resolution), clamp01(default_roughness), dtype=np.float32)
    metallic_target = np.full((atlas_resolution, atlas_resolution), clamp01(default_metallic), dtype=np.float32)
    filled = sum_weight > 1e-6
    roughness_target[filled] = sum_roughness[filled] / sum_weight[filled]
    metallic_target[filled] = sum_metallic[filled] / sum_weight[filled]

    normalized_weight = np.zeros_like(sum_weight)
    normalized_weight[filled] = sum_weight[filled] / float(sum_weight.max())
    confidence = np.clip(np.maximum(normalized_weight, filled.astype(np.float32) * 0.45), 0.0, 1.0)
    confidence = np.asarray(
        Image.fromarray((confidence * 255.0).round().astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=1.5)
        ),
        dtype=np.float32,
    ) / 255.0
    confidence[filled] = np.maximum(confidence[filled], 0.45)

    return {
        "target_roughness": roughness_target,
        "target_metallic": metallic_target,
        "confidence": confidence,
        "valid_view_count": int(valid_view_count),
        "target_source_type": "pseudo_from_multiview",
    }


def maybe_copy(src_path: Path | None, dst_path: Path) -> bool:
    if src_path is None or not src_path.exists():
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() != dst_path.resolve():
        shutil.copy2(src_path, dst_path)
    return True


def bake_uv_targets(
    *,
    bundle_dir: Path,
    uv_albedo_path: Path | None,
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
    uv_dir = ensure_dir(bundle_dir / "uv")
    prior_roughness_path = uv_dir / "uv_prior_roughness.png"
    prior_metallic_path = uv_dir / "uv_prior_metallic.png"
    target_roughness_path = uv_dir / "uv_target_roughness.png"
    target_metallic_path = uv_dir / "uv_target_metallic.png"
    confidence_path = uv_dir / "uv_target_confidence.png"

    target_roughness = load_gray_image(uv_target_roughness_source_path, atlas_resolution)
    target_metallic = load_gray_image(uv_target_metallic_source_path, atlas_resolution)
    target_source_type = "unknown"
    valid_view_count = 0

    if target_roughness is not None and target_metallic is not None:
        target_source_type = "gt_render_baked"
    else:
        pseudo = multiview_bake_to_uv(
            buffer_root=canonical_buffer_root,
            atlas_resolution=atlas_resolution,
            default_roughness=default_roughness,
            default_metallic=default_metallic,
        )
        if pseudo is not None:
            target_roughness = pseudo["target_roughness"]
            target_metallic = pseudo["target_metallic"]
            target_source_type = str(pseudo["target_source_type"])
            valid_view_count = int(pseudo["valid_view_count"])

    if target_roughness is None or target_metallic is None:
        if not allow_prior_copy_fallback:
            target_roughness = np.full(
                (atlas_resolution, atlas_resolution),
                clamp01(default_roughness),
                dtype=np.float32,
            )
            target_metallic = np.full(
                (atlas_resolution, atlas_resolution),
                clamp01(default_metallic),
                dtype=np.float32,
            )
        else:
            prior_roughness = load_gray_image(uv_prior_roughness_path, atlas_resolution)
            prior_metallic = load_gray_image(uv_prior_metallic_path, atlas_resolution)
            target_roughness = (
                prior_roughness
                if prior_roughness is not None
                else np.full((atlas_resolution, atlas_resolution), clamp01(default_roughness), dtype=np.float32)
            )
            target_metallic = (
                prior_metallic
                if prior_metallic is not None
                else np.full((atlas_resolution, atlas_resolution), clamp01(default_metallic), dtype=np.float32)
            )
        target_source_type = "copied_from_prior" if allow_prior_copy_fallback else "unknown"

    target_roughness = np.asarray(target_roughness, dtype=np.float32)
    target_metallic = np.asarray(target_metallic, dtype=np.float32)
    valid_mask = valid_mask_from_sources(
        resolution=atlas_resolution,
        albedo_path=uv_albedo_path,
        target_roughness=target_roughness,
        target_metallic=target_metallic,
    )
    confidence = soften_confidence(valid_mask)

    if target_source_type == "pseudo_from_multiview":
        pseudo = multiview_bake_to_uv(
            buffer_root=canonical_buffer_root,
            atlas_resolution=atlas_resolution,
            default_roughness=default_roughness,
            default_metallic=default_metallic,
        )
        if pseudo is not None:
            confidence = np.maximum(confidence, np.asarray(pseudo["confidence"], dtype=np.float32))
            valid_view_count = int(pseudo["valid_view_count"])

    copied_prior_roughness = load_gray_image(uv_prior_roughness_path, atlas_resolution)
    copied_prior_metallic = load_gray_image(uv_prior_metallic_path, atlas_resolution)
    if copied_prior_roughness is None:
        copied_prior_roughness = np.full((atlas_resolution, atlas_resolution), clamp01(default_roughness), dtype=np.float32)
    if copied_prior_metallic is None:
        copied_prior_metallic = np.full((atlas_resolution, atlas_resolution), clamp01(default_metallic), dtype=np.float32)

    if synthesize_nontrivial_prior and target_source_type in {"gt_render_baked", "pseudo_from_multiview"}:
        prior_roughness = synthesize_prior_from_target(
            target_roughness,
            scalar_default=default_roughness,
            is_metallic=False,
        )
        prior_metallic = synthesize_prior_from_target(
            target_metallic,
            scalar_default=default_metallic,
            is_metallic=True,
        )
    else:
        prior_roughness = copied_prior_roughness
        prior_metallic = copied_prior_metallic

    save_gray_png(prior_roughness_path, prior_roughness)
    save_gray_png(prior_metallic_path, prior_metallic)
    save_gray_png(target_roughness_path, target_roughness)
    save_gray_png(target_metallic_path, target_metallic)
    save_gray_png(confidence_path, confidence)

    target_is_prior_copy = bool(
        np.allclose(target_roughness, prior_roughness, atol=1.0 / 255.0)
        and np.allclose(target_metallic, prior_metallic, atol=1.0 / 255.0)
    )
    target_confidence_summary = confidence_summary(confidence)
    target_coverage = float((confidence > 1e-6).mean())
    return {
        "uv_prior_roughness_path": str(prior_roughness_path.resolve()),
        "uv_prior_metallic_path": str(prior_metallic_path.resolve()),
        "uv_target_roughness_path": str(target_roughness_path.resolve()),
        "uv_target_metallic_path": str(target_metallic_path.resolve()),
        "uv_target_confidence_path": str(confidence_path.resolve()),
        "uv_target_size": [atlas_resolution, atlas_resolution],
        "roughness_seed": clamp01(default_roughness),
        "metallic_seed": clamp01(default_metallic),
        "target_source_type": target_source_type,
        "target_is_prior_copy": target_is_prior_copy,
        "target_confidence_summary": target_confidence_summary,
        "target_confidence_mean": float(target_confidence_summary["mean"]),
        "target_confidence_nonzero_rate": float(target_confidence_summary["nonzero_rate"]),
        "target_coverage": target_coverage,
        "target_prior_identity": prior_identity(
            target_roughness,
            target_metallic,
            prior_roughness,
            prior_metallic,
            confidence,
        ),
        "valid_view_count": int(valid_view_count),
        "prior_generation_mode": (
            "synthetic_degraded_from_target"
            if synthesize_nontrivial_prior and target_source_type in {"gt_render_baked", "pseudo_from_multiview"}
            else "copied_or_scalar_default"
        ),
    }


def main() -> None:
    args = parse_args()
    payload = bake_uv_targets(
        bundle_dir=args.bundle_dir,
        uv_albedo_path=args.uv_albedo_path,
        uv_prior_roughness_path=args.uv_prior_roughness_path,
        uv_prior_metallic_path=args.uv_prior_metallic_path,
        uv_target_roughness_source_path=args.uv_target_roughness_source_path,
        uv_target_metallic_source_path=args.uv_target_metallic_source_path,
        canonical_buffer_root=args.canonical_buffer_root,
        atlas_resolution=args.atlas_resolution,
        default_roughness=args.default_roughness,
        default_metallic=args.default_metallic,
        synthesize_nontrivial_prior=args.synthesize_nontrivial_prior,
        allow_prior_copy_fallback=args.allow_prior_copy_fallback,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
