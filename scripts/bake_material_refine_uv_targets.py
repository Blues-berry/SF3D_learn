#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import struct
import zlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bake v1 UV-space roughness/metallic targets for the material refinement dataset."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--uv-prior-roughness-path", type=Path, default=None)
    parser.add_argument("--uv-prior-metallic-path", type=Path, default=None)
    parser.add_argument("--atlas-resolution", type=int, default=1024)
    parser.add_argument("--default-roughness", type=float, default=0.5)
    parser.add_argument("--default-metallic", type=float, default=0.0)
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def gray_level(value: float) -> int:
    return int(round(clamp01(value) * 255.0))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def write_png(path: Path, width: int, height: int, color_type: int, rows: list[bytes]) -> None:
    bit_depth = 8
    raw = b"".join(b"\x00" + row for row in rows)
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            png_chunk(b"IHDR", ihdr),
            png_chunk(b"IDAT", zlib.compress(raw, level=9)),
            png_chunk(b"IEND", b""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_constant_gray_png(path: Path, width: int, height: int, value: float) -> None:
    row = bytes([gray_level(value)]) * width
    write_png(path, width, height, 0, [row] * height)


def maybe_copy(src_path: Path | None, dst_path: Path) -> bool:
    if src_path is None or not src_path.exists():
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return True


def bake_uv_targets(
    *,
    bundle_dir: Path,
    uv_prior_roughness_path: Path | None,
    uv_prior_metallic_path: Path | None,
    atlas_resolution: int,
    default_roughness: float,
    default_metallic: float,
) -> dict[str, object]:
    uv_dir = ensure_dir(bundle_dir / "uv")
    roughness_path = uv_dir / "uv_target_roughness.png"
    metallic_path = uv_dir / "uv_target_metallic.png"
    confidence_path = uv_dir / "uv_target_confidence.png"

    if not maybe_copy(uv_prior_roughness_path, roughness_path):
        write_constant_gray_png(
            roughness_path,
            atlas_resolution,
            atlas_resolution,
            default_roughness,
        )
    if not maybe_copy(uv_prior_metallic_path, metallic_path):
        write_constant_gray_png(
            metallic_path,
            atlas_resolution,
            atlas_resolution,
            default_metallic,
        )
    write_constant_gray_png(confidence_path, atlas_resolution, atlas_resolution, 1.0)

    return {
        "uv_target_roughness_path": str(roughness_path.resolve()),
        "uv_target_metallic_path": str(metallic_path.resolve()),
        "uv_target_confidence_path": str(confidence_path.resolve()),
        "uv_target_size": [atlas_resolution, atlas_resolution],
        "roughness_seed": clamp01(default_roughness),
        "metallic_seed": clamp01(default_metallic),
    }


def main() -> None:
    args = parse_args()
    payload = bake_uv_targets(
        bundle_dir=args.bundle_dir,
        uv_prior_roughness_path=args.uv_prior_roughness_path,
        uv_prior_metallic_path=args.uv_prior_metallic_path,
        atlas_resolution=args.atlas_resolution,
        default_roughness=args.default_roughness,
        default_metallic=args.default_metallic,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
