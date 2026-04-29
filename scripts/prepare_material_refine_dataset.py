#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
import os
import shutil
import struct
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import numpy as np
from bake_material_refine_uv_targets import (
    bake_uv_targets,
    load_gray_image as load_gray_map,
    load_uv_map as load_view_uv_map,
)
from PIL import Image
from refresh_material_refine_partial_manifest import (
    PREPARED_RECORD_FILENAME,
    build_partial_manifest,
    build_prepared_record,
    find_missing_paths,
)
from sf3d.material_refine.manifest_quality import enrich_record_with_quality_fields

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_MANIFEST = REPO_ROOT / "docs" / "neural_gaffer_dataset_audit" / "material_refine_manifest_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "material_refine"
DEFAULT_ABO_RENDER_CACHE = REPO_ROOT / "output" / "abo_rm_mini" / "renders"
DEFAULT_BLENDER_BIN = Path(
    "/4T/CXY/Neural_Gaffer_original/scripts/Objavarse_rendering/blender-3.2.2-linux-x64/blender"
)
BLENDER_SCRIPT = REPO_ROOT / "scripts" / "abo_material_passes_blender.py"
DEFAULT_HDRI_BANK_JSON = REPO_ROOT / "output" / "highlight_pool_a_8k" / "aux_sources" / "polyhaven_hdri_bank.json"
CANONICAL_TRIPLET_VIEWS = [
    {
        "name": "front_studio",
        "azimuth": 0.0,
        "elevation": 18.0,
        "distance": 2.0,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "studio_small_08_1k.hdr"),
    },
    {
        "name": "three_quarter_indoor",
        "azimuth": 45.0,
        "elevation": 28.0,
        "distance": 2.0,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "peppermint_powerplant_1k.hdr"),
    },
    {
        "name": "side_neon",
        "azimuth": 110.0,
        "elevation": 16.0,
        "distance": 2.05,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "neon_photostudio_1k.hdr"),
    },
]
CAMERA_PROTOCOLS = {
    "canonical_triplet": [
        {"label": "front_studio", "azimuth": 0.0, "elevation": 18.0, "distance": 2.0},
        {"label": "three_quarter_indoor", "azimuth": 45.0, "elevation": 28.0, "distance": 2.0},
        {"label": "side_neon", "azimuth": 110.0, "elevation": 16.0, "distance": 2.05},
    ],
    "standard_12": [
        {"label": "front_mid", "azimuth": 0.0, "elevation": 18.0, "distance": 2.0},
        {"label": "front_high", "azimuth": 0.0, "elevation": 34.0, "distance": 2.05},
        {"label": "three_quarter_mid", "azimuth": 45.0, "elevation": 24.0, "distance": 2.0},
        {"label": "side_low", "azimuth": 95.0, "elevation": 12.0, "distance": 2.05},
        {"label": "grazing_left", "azimuth": 150.0, "elevation": 10.0, "distance": 2.25},
        {"label": "top_oblique", "azimuth": 315.0, "elevation": 42.0, "distance": 2.15},
    ],
    "stress_24": [
        {"label": "front_mid", "azimuth": 0.0, "elevation": 18.0, "distance": 2.0},
        {"label": "front_high", "azimuth": 0.0, "elevation": 34.0, "distance": 2.05},
        {"label": "three_quarter_mid", "azimuth": 45.0, "elevation": 24.0, "distance": 2.0},
        {"label": "side_low", "azimuth": 95.0, "elevation": 12.0, "distance": 2.05},
        {"label": "grazing_left", "azimuth": 150.0, "elevation": 10.0, "distance": 2.25},
        {"label": "grazing_right", "azimuth": 230.0, "elevation": 10.0, "distance": 2.25},
        {"label": "top_oblique", "azimuth": 315.0, "elevation": 42.0, "distance": 2.15},
        {"label": "thin_boundary_close", "azimuth": 35.0, "elevation": 8.0, "distance": 1.55},
    ],
    "production_32": [
        {"label": "front_mid", "azimuth": 0.0, "elevation": 18.0, "distance": 2.0},
        {"label": "front_high", "azimuth": 0.0, "elevation": 34.0, "distance": 2.05},
        {"label": "three_quarter_mid", "azimuth": 45.0, "elevation": 24.0, "distance": 2.0},
        {"label": "side_low", "azimuth": 95.0, "elevation": 12.0, "distance": 2.05},
        {"label": "rear_three_quarter", "azimuth": 210.0, "elevation": 22.0, "distance": 2.05},
        {"label": "grazing_left", "azimuth": 150.0, "elevation": 10.0, "distance": 2.25},
        {"label": "grazing_right", "azimuth": 230.0, "elevation": 10.0, "distance": 2.25},
        {"label": "top_oblique", "azimuth": 315.0, "elevation": 42.0, "distance": 2.15},
    ],
}
PROTOCOL_LIGHT_COUNTS = {
    "canonical_triplet": 0,
    "standard_12": 2,
    "stress_24": 3,
    "production_32": 4,
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the CanonicalAssetRecordV1 bundle for GPU0 dataset work."
    )
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--split", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--atlas-resolution", type=int, default=1024)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--cycles-samples", type=int, default=32)
    parser.add_argument(
        "--view-light-protocol",
        choices=sorted(CAMERA_PROTOCOLS),
        default="canonical_triplet",
        help="Object-level view/light bank. Non-canonical protocols combine multiple camera poses with HDRIs.",
    )
    parser.add_argument("--hdri-bank-json", type=Path, default=DEFAULT_HDRI_BANK_JSON)
    parser.add_argument(
        "--min-hdri-count",
        type=int,
        default=0,
        help="Fail early for non-canonical protocols unless the HDRI bank has at least this many local entries.",
    )
    parser.add_argument(
        "--max-hdri-lights",
        type=int,
        default=0,
        help="Override HDRI count per object. 0 uses the protocol default.",
    )
    parser.add_argument(
        "--hdri-selection-offset",
        type=int,
        default=0,
        help="Deterministic offset for object-specific HDRI selection.",
    )
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--abo-render-cache", type=Path, default=DEFAULT_ABO_RENDER_CACHE)
    parser.add_argument("--cuda-device-index", type=str, default="0")
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument(
        "--rebake-version",
        choices=("legacy", "rebake_v2", "v1_fixed_rebake"),
        default="legacy",
        help="When rebaking, write strict target/view contract fields and do not mark legacy targets paper-ready.",
    )
    parser.add_argument(
        "--disable-render-cache",
        action="store_true",
        help="Force fresh view buffers instead of reusing cached render roots.",
    )
    parser.add_argument(
        "--disallow-prior-copy-fallback",
        action="store_true",
        help="Do not use prior-copy fallback for target baking; failed targets remain non-promotable.",
    )
    parser.add_argument("--target-view-alignment-mean-threshold", type=float, default=0.03)
    parser.add_argument("--target-view-alignment-p95-threshold", type=float, default=0.08)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--partial-manifest", type=Path, default=None)
    parser.add_argument("--refresh-partial-every", type=int, default=25)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "records" not in payload:
        raise RuntimeError(f"unexpected_manifest_payload:{path}")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.lower()).strip("_")


def load_hdri_bank(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    hdri_records = []
    for record in records:
        if not isinstance(record, dict):
            continue
        local_path = record.get("local_path") or record.get("path")
        if not local_path:
            continue
        hdri_path = Path(str(local_path))
        if not hdri_path.is_absolute():
            hdri_path = REPO_ROOT / hdri_path
        if not hdri_path.exists():
            continue
        hdri_records.append(
            {
                "asset_id": str(record.get("asset_id") or hdri_path.stem),
                "name": str(record.get("name") or record.get("asset_id") or hdri_path.stem),
                "stratum": str(record.get("stratum") or "unknown"),
                "license_bucket": str(record.get("license_bucket") or "unknown"),
                "path": str(hdri_path.resolve()),
            }
        )
    return sorted(hdri_records, key=lambda item: (item["stratum"], item["asset_id"]))


def select_hdri_records(
    *,
    hdri_bank: list[dict],
    object_id: str,
    count: int,
    offset: int,
) -> list[dict]:
    if count <= 0 or not hdri_bank:
        return []
    grouped: dict[str, list[dict]] = {}
    for record in hdri_bank:
        grouped.setdefault(str(record.get("stratum") or "unknown"), []).append(record)
    strata = sorted(grouped)
    selected: list[dict] = []
    seed = stable_int(f"{object_id}|{offset}")
    for idx in range(count):
        stratum = strata[(seed + idx) % len(strata)]
        bucket = grouped[stratum]
        selected.append(bucket[(seed // max(idx + 1, 1) + idx) % len(bucket)])
    return selected


def build_views_for_record(
    record: dict,
    *,
    protocol: str,
    hdri_bank: list[dict],
    max_hdri_lights: int,
    hdri_selection_offset: int,
) -> list[dict]:
    if protocol == "canonical_triplet":
        return [dict(view) for view in CANONICAL_TRIPLET_VIEWS]
    cameras = CAMERA_PROTOCOLS[protocol]
    light_count = max_hdri_lights if max_hdri_lights > 0 else PROTOCOL_LIGHT_COUNTS[protocol]
    selected_hdris = select_hdri_records(
        hdri_bank=hdri_bank,
        object_id=str(record.get("object_id") or record.get("source_uid") or "unknown"),
        count=light_count,
        offset=hdri_selection_offset,
    )
    if not selected_hdris:
        return [dict(view) for view in CANONICAL_TRIPLET_VIEWS]
    views: list[dict] = []
    for light_index, hdri in enumerate(selected_hdris):
        hdri_id = sanitize_name(str(hdri["asset_id"]))
        for camera in cameras:
            camera_label = sanitize_name(str(camera["label"]))
            view = {
                "name": f"{camera_label}__{light_index:02d}_{hdri_id}",
                "azimuth": float(camera["azimuth"]),
                "elevation": float(camera["elevation"]),
                "distance": float(camera["distance"]),
                "hdri": str(hdri["path"]),
                "camera_label": str(camera["label"]),
                "lighting_bank_id": f"polyhaven_{protocol}_v1",
                "lighting_asset_id": str(hdri["asset_id"]),
                "lighting_stratum": str(hdri.get("stratum") or "unknown"),
                "lighting_license_bucket": str(hdri.get("license_bucket") or "unknown"),
                "view_light_protocol": protocol,
            }
            views.append(view)
    return views


def select_records(payload: dict, split: str, max_records: int | None) -> list[dict]:
    selected = []
    for record in payload["records"]:
        include_flag = record.get("include_in_smoke") if split == "smoke" else record.get("include_in_full", True)
        if not include_flag:
            continue
        if not record.get("source_asset_available", False):
            continue
        selected.append(dict(record))
    if max_records is not None:
        selected = selected[:max_records]
    return selected


def first_existing_path(*candidates: str) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path.resolve()
    return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def resolve_prepared_prior_fields(
    record: dict,
    *,
    roughness_src: Path | None,
    metallic_src: Path | None,
) -> tuple[bool, str]:
    has_material_prior = bool(record.get("has_material_prior"))
    prior_mode = str(record.get("prior_mode") or "none")
    if has_material_prior and prior_mode in {"", "none"}:
        if roughness_src is not None or metallic_src is not None:
            prior_mode = "uv_rm"
        else:
            prior_mode = "scalar_rm"
    elif not has_material_prior:
        prior_mode = "none"
    return has_material_prior, prior_mode


def should_synthesize_nontrivial_prior(record: dict) -> bool:
    """Treat locked/qualified strong tiers as strong, not only the literal string."""
    supervision_tier = str(record.get("supervision_tier") or "").lower()
    supervision_role = str(record.get("supervision_role") or "").lower()
    target_source_type = str(record.get("target_source_type") or "").lower()
    if supervision_role == "paper_main":
        return True
    if supervision_tier.startswith("strong") or "a_ready" in supervision_tier:
        return True
    if target_source_type in {"gt_render_baked", "pseudo_from_multiview"}:
        return True
    if "smoke" in supervision_tier or "copy" in supervision_tier:
        return False
    return True


def write_prepared_record_file(bundle_dir: Path, prepared: dict) -> None:
    write_json(bundle_dir / PREPARED_RECORD_FILENAME, prepared)


def build_prepare_manifest_payload(
    *,
    input_manifest: Path,
    split: str,
    bundle_root: Path,
    selected: list[dict],
    prepared_records: list[dict],
    skipped_records: list[dict],
) -> dict:
    return {
        "manifest_version": "canonical_asset_record_v1_prepared",
        "input_manifest": str(input_manifest.resolve()),
        "split": split,
        "canonical_bundle_root": str(bundle_root.resolve()),
        "counts": {
            "selected_records": len(selected),
            "prepared_records": len(prepared_records),
            "skipped_records": len(skipped_records),
            "with_prior": sum(bool(record.get("has_material_prior")) for record in prepared_records),
            "without_prior": sum(not bool(record.get("has_material_prior")) for record in prepared_records),
            "by_source_name": dict(Counter(str(record.get("source_name", "")) for record in prepared_records)),
        },
        "skipped_records": skipped_records,
        "records": prepared_records,
    }


def build_run_summary(
    *,
    output_manifest: Path,
    split: str,
    selected: list[dict],
    prepared_records: list[dict],
    skipped_records: list[dict],
) -> dict:
    return {
        "split": split,
        "output_manifest": str(output_manifest.resolve()),
        "selected_records": len(selected),
        "prepared_records": len(prepared_records),
        "skipped_records": len(skipped_records),
        "source_mix": dict(Counter(str(record.get("source_name", "")) for record in prepared_records)),
        "with_prior": sum(bool(record.get("has_material_prior")) for record in prepared_records),
        "without_prior": sum(not bool(record.get("has_material_prior")) for record in prepared_records),
        "render_modes": dict(Counter(str(record.get("render_mode", "unknown")) for record in prepared_records)),
        "skipped_reasons": dict(Counter(str(record.get("reason", "unknown")) for record in skipped_records)),
    }


def write_summary_md(path: Path, summary: dict) -> None:
    lines = [
        "# Material Refine Prepare Summary",
        "",
        f"- split: {summary['split']}",
        f"- output_manifest: {summary['output_manifest']}",
        f"- selected_records: {summary['selected_records']}",
        f"- prepared_records: {summary['prepared_records']}",
        f"- skipped_records: {summary['skipped_records']}",
        f"- with_prior: {summary['with_prior']}",
        f"- without_prior: {summary['without_prior']}",
        "",
        "## Source Mix",
        "",
    ]
    for source_name, count in sorted(summary["source_mix"].items()):
        lines.append(f"- {source_name}: {count}")
    lines.extend(["", "## Render Modes", ""])
    for render_mode, count in sorted(summary["render_modes"].items()):
        lines.append(f"- {render_mode}: {count}")
    lines.extend(["", "## Skipped Reasons", ""])
    if summary["skipped_reasons"]:
        for reason, count in sorted(summary["skipped_reasons"].items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_constant_rgb(path: Path, color: tuple[int, int, int], size: int) -> None:
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[..., 0] = color[0]
    array[..., 1] = color[1]
    array[..., 2] = color[2]
    Image.fromarray(array, mode="RGB").save(path)


def write_flat_normal(path: Path, size: int) -> None:
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[..., 0] = 128
    array[..., 1] = 128
    array[..., 2] = 255
    Image.fromarray(array, mode="RGB").save(path)


def write_constant_gray(path: Path, value: float, size: int) -> None:
    array = np.full((size, size), int(round(clamp01(value) * 255.0)), dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


def write_rgb_array(path: Path, array: np.ndarray) -> None:
    clipped = np.clip(array, 0.0, 1.0)
    Image.fromarray((clipped * 255.0).round().astype(np.uint8), mode="RGB").save(path)


def load_rgb_array(path: Path, size: int) -> np.ndarray | None:
    if not path.exists():
        return None
    image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_glb_json(path: Path) -> dict | None:
    if path.suffix.lower() != ".glb" or not path.exists():
        return None
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12:
            return None
        magic, version, _length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            return None
        chunk_header = handle.read(8)
        if len(chunk_header) < 8:
            return None
        chunk_len, chunk_type = struct.unpack("<I4s", chunk_header)
        if chunk_type != b"JSON":
            return None
        chunk = handle.read(chunk_len)
    return json.loads(chunk.decode("utf-8"))


def infer_scalar_priors(source_model_path: Path, record: dict) -> tuple[float, float]:
    if record.get("scalar_prior_roughness") not in {None, ""} and record.get("scalar_prior_metallic") not in {None, ""}:
        return float(record["scalar_prior_roughness"]), float(record["scalar_prior_metallic"])

    roughness = 0.5
    metallic = 0.0
    doc = load_glb_json(source_model_path)
    if doc is not None:
        materials = doc.get("materials", [])
        if materials:
            pbr = materials[0].get("pbrMetallicRoughness", {})
            roughness = float(pbr.get("roughnessFactor", roughness))
            metallic = float(pbr.get("metallicFactor", metallic))
    return clamp01(roughness), clamp01(metallic)


def discover_texture(texture_root: Path | None, keywords: tuple[str, ...]) -> Path | None:
    if texture_root is None or not texture_root.exists():
        return None
    if texture_root.is_file():
        if texture_root.suffix.lower() in IMAGE_EXTENSIONS:
            return texture_root.resolve()
        return None
    candidates = []
    for path in texture_root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = path.stem.lower()
        if any(keyword in name for keyword in keywords):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort()
    return candidates[0].resolve()


def copy_or_placeholder(
    src_path: Path | None,
    dst_path: Path,
    *,
    fallback_kind: str,
    fallback_value: float | None = None,
    size: int,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path is not None and src_path.exists():
        if src_path.resolve() != dst_path.resolve():
            shutil.copy2(src_path, dst_path)
        return
    if fallback_kind == "albedo":
        write_constant_rgb(dst_path, (180, 180, 180), size)
    elif fallback_kind == "normal":
        write_flat_normal(dst_path, size)
    elif fallback_kind == "gray":
        write_constant_gray(dst_path, fallback_value or 0.0, size)
    else:
        raise ValueError(f"unsupported_fallback_kind:{fallback_kind}")


def render_bundle_complete(buffer_root: Path, views: list[dict]) -> bool:
    if not buffer_root.exists():
        return False
    for view in views:
        view_dir = buffer_root / view["name"]
        if not view_dir.is_dir():
            return False
        for name in (
            "rgba.png",
            "normal.png",
            "position.png",
            "uv.png",
            "roughness.png",
            "metallic.png",
        ):
            if not (view_dir / name).exists():
                return False
    return True


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def resolve_cached_render_root(
    record: dict,
    source_model_path: Path,
    abo_render_cache: Path | None,
    views: list[dict],
) -> Path | None:
    if abo_render_cache is None or not abo_render_cache.exists():
        return None
    source_name = str(record.get("source_name", "")).lower()
    generator_id = str(record.get("generator_id", "")).lower()
    if "abo" not in source_name and "abo" not in generator_id:
        return None
    candidate = abo_render_cache / source_model_path.stem
    if render_bundle_complete(candidate, views):
        return candidate.resolve()
    return None


def link_or_copy_render_root(src_root: Path, dst_root: Path) -> None:
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    if dst_root.exists() or dst_root.is_symlink():
        remove_path(dst_root)
    try:
        dst_root.symlink_to(src_root, target_is_directory=True)
    except OSError:
        shutil.copytree(src_root, dst_root)


def derive_depth_from_position(position: np.ndarray, camera_location: list[float]) -> np.ndarray:
    world_position = np.clip(position * 2.0 - 1.0, -1.0, 1.0)
    camera = np.asarray(camera_location, dtype=np.float32).reshape(1, 1, 3)
    depth = np.linalg.norm(world_position - camera, axis=-1)
    if float(depth.max()) > float(depth.min()):
        depth = (depth - float(depth.min())) / (float(depth.max()) - float(depth.min()))
    else:
        depth = np.zeros_like(depth, dtype=np.float32)
    return depth.astype(np.float32)


def finalize_view_buffers(buffer_root: Path, render_resolution: int, views: list[dict]) -> dict[str, object]:
    rgba_paths: dict[str, dict[str, str]] = {}
    field_sources: dict[str, dict[str, object]] = {}
    valid_view_count = 0
    for view in views:
        view_dir = ensure_dir(buffer_root / view["name"])
        rgba_path = view_dir / "rgba.png"
        view_sources: dict[str, str] = {}
        if not rgba_path.exists():
            rgba = np.zeros((render_resolution, render_resolution, 4), dtype=np.uint8)
            rgba[..., :3] = 180
            rgba[..., 3] = 255
            Image.fromarray(rgba, mode="RGBA").save(rgba_path)
            view_sources["rgba"] = "synthetic_placeholder"
        else:
            view_sources["rgba"] = "rendered"
        roughness_path = view_dir / "roughness.png"
        if not roughness_path.exists():
            write_constant_gray(roughness_path, 0.5, render_resolution)
            view_sources["roughness"] = "synthetic_placeholder"
        else:
            view_sources["roughness"] = "rendered"
        metallic_path = view_dir / "metallic.png"
        if not metallic_path.exists():
            write_constant_gray(metallic_path, 0.0, render_resolution)
            view_sources["metallic"] = "synthetic_placeholder"
        else:
            view_sources["metallic"] = "rendered"
        mask_path = view_dir / "mask.png"
        if mask_path.exists():
            view_sources["mask"] = "rendered"
        else:
            rgba = Image.open(rgba_path).convert("RGBA").resize((render_resolution, render_resolution), Image.BILINEAR)
            mask = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
            Image.fromarray(mask, mode="L").save(mask_path)
            view_sources["mask"] = "derived_from_rgba"

        visibility_path = view_dir / "visibility.png"
        if visibility_path.exists():
            view_sources["visibility"] = "rendered"
        else:
            shutil.copy2(mask_path, visibility_path)
            view_sources["visibility"] = "derived_from_mask"

        normal_path = view_dir / "normal.png"
        if normal_path.exists():
            view_sources["normal"] = "rendered"
        else:
            write_flat_normal(normal_path, render_resolution)
            view_sources["normal"] = "synthetic_placeholder"

        position_path = view_dir / "position.png"
        if position_path.exists():
            view_sources["position"] = "rendered"
        else:
            write_constant_rgb(position_path, (128, 128, 128), render_resolution)
            view_sources["position"] = "synthetic_placeholder"

        uv_path = view_dir / "uv.png"
        if uv_path.exists():
            view_sources["uv"] = "rendered"
        else:
            write_constant_rgb(uv_path, (0, 0, 0), render_resolution)
            view_sources["uv"] = "synthetic_placeholder"

        depth_path = view_dir / "depth.png"
        if depth_path.exists():
            view_sources["depth"] = "rendered"
        else:
            position = load_rgb_array(position_path, render_resolution)
            view_json_path = view_dir / "view.json"
            if position is not None and view_json_path.exists():
                view_json = json.loads(view_json_path.read_text())
                camera_location = view_json.get("camera_location")
                if (
                    isinstance(camera_location, list)
                    and len(camera_location) == 3
                    and view_sources.get("position") != "synthetic_placeholder"
                ):
                    depth = derive_depth_from_position(position, camera_location)
                    Image.fromarray((depth * 255.0).round().astype(np.uint8), mode="L").save(depth_path)
                    view_sources["depth"] = "derived_from_position"
                else:
                    write_constant_gray(depth_path, 0.0, render_resolution)
                    view_sources["depth"] = "synthetic_placeholder"
            else:
                write_constant_gray(depth_path, 0.0, render_resolution)
                view_sources["depth"] = "synthetic_placeholder"

        effective_ready = all(
            view_sources.get(field, "missing") not in {"synthetic_placeholder", "missing"}
            for field in ("rgba", "uv", "roughness", "metallic")
        )
        strict_ready = all(
            view_sources.get(field, "missing") not in {"synthetic_placeholder", "missing"}
            for field in ("rgba", "mask", "depth", "normal", "position", "uv", "visibility", "roughness", "metallic")
        )
        valid_view_count += int(effective_ready)
        field_sources[view["name"]] = {
            "fields": view_sources,
            "view_supervision_ready": effective_ready,
            "strict_complete_ready": strict_ready,
        }
        rgba_paths[view["name"]] = {"rgba": str(rgba_path.resolve())}
    metadata_path = buffer_root / "_field_sources.json"
    metadata_path.write_text(
        json.dumps(
            {
                "view_quality_version": "v1",
                "view_count": len(views),
                "views": field_sources,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "view_rgba_paths": rgba_paths,
        "view_buffer_field_sources_path": str(metadata_path.resolve()),
        "view_supervision_ready": bool(valid_view_count > 0),
        "valid_view_count": int(valid_view_count),
    }


def render_views(
    *,
    source_model_path: Path,
    buffer_root: Path,
    views_json: Path,
    views: list[dict],
    blender_bin: Path,
    render_resolution: int,
    cycles_samples: int,
    cuda_device_index: str,
) -> None:
    views_json.write_text(json.dumps(views, indent=2, ensure_ascii=False), encoding="utf-8")
    cmd = [
        str(blender_bin),
        "-b",
        "-P",
        str(BLENDER_SCRIPT),
        "--",
        "--object-path",
        str(source_model_path),
        "--output-dir",
        str(buffer_root),
        "--views-json",
        str(views_json),
        "--resolution",
        str(render_resolution),
        "--cycles-samples",
        str(cycles_samples),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_device_index
    env["BLENDER_CUDA_DEVICE_INDEX"] = "0"
    subprocess.run(cmd, check=True, env=env)


def _field_source_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _view_field_sources(field_sources_payload: dict, view_name: str) -> dict[str, str]:
    views = field_sources_payload.get("views")
    if not isinstance(views, dict):
        return {}
    view_payload = views.get(view_name)
    if not isinstance(view_payload, dict):
        return {}
    fields = view_payload.get("fields")
    if not isinstance(fields, dict):
        return {}
    return {str(key): str(value) for key, value in fields.items()}


def _alignment_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _valid_uv_mask(
    *,
    uv: np.ndarray,
    mask: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    return (
        np.isfinite(uv[..., 0])
        & np.isfinite(uv[..., 1])
        & (mask > 0.5)
        & (visibility > 0.5)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] <= 1.0)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] <= 1.0)
    )


def compute_rebake_v2_contract(
    *,
    bundle_dir: Path,
    buffer_root: Path,
    views: list[dict],
    uv_prior_roughness_path: Path,
    uv_prior_metallic_path: Path,
    uv_target_roughness_path: Path,
    uv_target_metallic_path: Path,
    view_buffer_field_sources_path: Path,
    atlas_resolution: int,
    render_resolution: int,
    mean_threshold: float,
    p95_threshold: float,
    rebake_version: str = "rebake_v2",
    target_view_contract_version: str = "v2",
) -> dict[str, object]:
    target_roughness = load_gray_map(uv_target_roughness_path, atlas_resolution)
    target_metallic = load_gray_map(uv_target_metallic_path, atlas_resolution)
    prior_roughness = load_gray_map(uv_prior_roughness_path, atlas_resolution)
    prior_metallic = load_gray_map(uv_prior_metallic_path, atlas_resolution)
    contract_path = bundle_dir / "_rebake_v2_contract.json"
    blockers: list[str] = []
    if target_roughness is None:
        blockers.append("missing_uv_target_roughness")
    if target_metallic is None:
        blockers.append("missing_uv_target_metallic")
    if prior_roughness is None:
        blockers.append("missing_uv_prior_roughness")
    if prior_metallic is None:
        blockers.append("missing_uv_prior_metallic")

    field_sources_payload = _field_source_payload(view_buffer_field_sources_path)
    target_errors: list[float] = []
    prior_sampled_pixels = 0
    prior_invalid_pixels = 0
    valid_view_count = 0
    strict_view_count = 0
    per_view: dict[str, dict[str, object]] = {}
    required_fields = ("uv", "visibility", "mask", "roughness", "metallic")
    placeholder_sources = {"synthetic_placeholder", "missing", "placeholder"}

    if not blockers:
        assert target_roughness is not None
        assert target_metallic is not None
        assert prior_roughness is not None
        assert prior_metallic is not None
        for view in views:
            view_name = str(view.get("name") or "")
            view_dir = buffer_root / view_name
            sources = _view_field_sources(field_sources_payload, view_name)
            placeholder_fields = [
                field
                for field in required_fields
                if sources.get(field, "missing") in placeholder_sources
            ]
            if placeholder_fields:
                per_view[view_name] = {
                    "ready": False,
                    "reason": "placeholder_view_fields",
                    "placeholder_fields": placeholder_fields,
                }
                continue

            uv_path = view_dir / "uv.png"
            uv = load_view_uv_map(uv_path, render_resolution)
            mask = load_gray_map(view_dir / "mask.png", render_resolution)
            visibility = load_gray_map(view_dir / "visibility.png", render_resolution)
            view_roughness = load_gray_map(view_dir / "roughness.png", render_resolution)
            view_metallic = load_gray_map(view_dir / "metallic.png", render_resolution)
            if any(value is None for value in (uv, mask, visibility, view_roughness, view_metallic)):
                per_view[view_name] = {"ready": False, "reason": "missing_view_arrays"}
                continue
            assert uv is not None
            assert mask is not None
            assert visibility is not None
            assert view_roughness is not None
            assert view_metallic is not None
            valid = _valid_uv_mask(uv=uv, mask=mask, visibility=visibility)
            valid_rate = float(valid.mean())
            if not bool(valid.any()):
                per_view[view_name] = {"ready": False, "reason": "empty_valid_uv_mask", "valid_rate": valid_rate}
                continue

            u = np.clip(
                np.rint(uv[..., 0][valid] * (atlas_resolution - 1)).astype(np.int32),
                0,
                atlas_resolution - 1,
            )
            v = np.clip(
                np.rint((1.0 - uv[..., 1][valid]) * (atlas_resolution - 1)).astype(np.int32),
                0,
                atlas_resolution - 1,
            )
            sampled_target_roughness = target_roughness[v, u]
            sampled_target_metallic = target_metallic[v, u]
            sampled_prior_roughness = prior_roughness[v, u]
            sampled_prior_metallic = prior_metallic[v, u]
            prior_finite = np.isfinite(sampled_prior_roughness) & np.isfinite(sampled_prior_metallic)
            prior_sampled_pixels += int(prior_finite.size)
            prior_invalid_pixels += int((~prior_finite).sum())
            target_error = 0.5 * (
                np.abs(sampled_target_roughness - view_roughness[valid])
                + np.abs(sampled_target_metallic - view_metallic[valid])
            )
            target_errors.extend(float(value) for value in target_error.tolist())
            valid_view_count += 1
            strict_view_count += int(
                all(sources.get(field, "missing") not in placeholder_sources for field in required_fields)
            )
            per_view[view_name] = {
                "ready": True,
                "valid_rate": valid_rate,
                "target_alignment_mean": float(target_error.mean()),
                "target_alignment_p95": float(np.quantile(target_error, 0.95)),
            }

    target_stats = _alignment_stats(target_errors)
    prior_sampling_valid_rate = (
        float((prior_sampled_pixels - prior_invalid_pixels) / max(prior_sampled_pixels, 1))
        if prior_sampled_pixels > 0
        else 0.0
    )
    target_mean = target_stats["mean"]
    target_p95 = target_stats["p95"]
    prior_as_pred_pass = bool(valid_view_count > 0 and prior_sampling_valid_rate >= 0.999)
    target_as_pred_pass = bool(
        valid_view_count > 0
        and target_mean is not None
        and target_p95 is not None
        and float(target_mean) < float(mean_threshold)
        and float(target_p95) < float(p95_threshold)
    )
    if valid_view_count <= 0:
        blockers.append("no_valid_rebake_v2_views")
    if target_mean is None:
        blockers.append("missing_target_view_alignment_mean")
    elif float(target_mean) >= float(mean_threshold):
        blockers.append(f"target_view_alignment_mean_high:{float(target_mean):.5f}")
    if target_p95 is None:
        blockers.append("missing_target_view_alignment_p95")
    elif float(target_p95) >= float(p95_threshold):
        blockers.append(f"target_view_alignment_p95_high:{float(target_p95):.5f}")

    payload = {
        "target_view_contract_version": target_view_contract_version,
        "rebake_version": rebake_version,
        "stored_view_target_valid_for_paper": bool(prior_as_pred_pass and target_as_pred_pass),
        "prior_as_pred_pass": prior_as_pred_pass,
        "target_as_pred_pass": target_as_pred_pass,
        "target_view_alignment_mean": target_mean,
        "target_view_alignment_p95": target_p95,
        "target_view_alignment_max": target_stats["max"],
        "prior_sampling_valid_rate": prior_sampling_valid_rate,
        "prior_sampling_invalid_pixels": int(prior_invalid_pixels),
        "view_supervision_ready": bool(valid_view_count > 0),
        "effective_view_supervision_rate": float(valid_view_count / max(len(views), 1)),
        "strict_complete_view_rate": float(strict_view_count / max(len(views), 1)),
        "valid_view_count": int(valid_view_count),
        "rebake_v2_contract_blockers": blockers,
        "rebake_v2_contract_debug_path": str(contract_path.resolve()),
        "per_view": per_view,
    }
    contract_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {key: value for key, value in payload.items() if key != "per_view"}


def prepare_record(
    record: dict,
    *,
    output_root: Path,
    split: str,
    atlas_resolution: int,
    render_resolution: int,
    cycles_samples: int,
    blender_bin: Path,
    abo_render_cache: Path | None,
    cuda_device_index: str,
    skip_render: bool,
    views: list[dict],
    view_light_protocol: str,
    rebake_version: str,
    disable_render_cache: bool,
    disallow_prior_copy_fallback: bool,
    target_view_alignment_mean_threshold: float,
    target_view_alignment_p95_threshold: float,
) -> dict:
    source_model_path = Path(record["source_model_path"]).resolve()
    texture_root = first_existing_path(record.get("source_texture_root", ""))
    bundle_root = ensure_dir(output_root / "canonical_bundle")
    bundle_dir = ensure_dir(bundle_root / record["object_id"])
    uv_dir = ensure_dir(bundle_dir / "uv")
    buffer_root = bundle_dir / "buffers"
    views_json = bundle_dir / "views.json"
    views_json.write_text(json.dumps(views, indent=2, ensure_ascii=False), encoding="utf-8")

    roughness_seed, metallic_seed = infer_scalar_priors(source_model_path, record)
    albedo_src = discover_texture(texture_root, ("albedo", "basecolor", "base_color", "diffuse"))
    normal_src = discover_texture(texture_root, ("normal", "norm"))
    roughness_src = discover_texture(texture_root, ("roughness", "rough"))
    metallic_src = discover_texture(texture_root, ("metallic", "metalness", "metal"))
    has_material_prior, prior_mode = resolve_prepared_prior_fields(
        record,
        roughness_src=roughness_src,
        metallic_src=metallic_src,
    )

    albedo_dst = uv_dir / "uv_albedo.png"
    normal_dst = uv_dir / "uv_normal.png"
    roughness_dst = uv_dir / "uv_prior_roughness.png"
    metallic_dst = uv_dir / "uv_prior_metallic.png"

    if not find_missing_paths(bundle_dir):
        print(f"[prepare reuse] upgrading existing bundle for {record['object_id']}")

    copy_or_placeholder(albedo_src, albedo_dst, fallback_kind="albedo", size=atlas_resolution)
    copy_or_placeholder(normal_src, normal_dst, fallback_kind="normal", size=atlas_resolution)
    copy_or_placeholder(
        roughness_src,
        roughness_dst,
        fallback_kind="gray",
        fallback_value=roughness_seed,
        size=atlas_resolution,
    )
    copy_or_placeholder(
        metallic_src,
        metallic_dst,
        fallback_kind="gray",
        fallback_value=metallic_seed,
        size=atlas_resolution,
    )

    is_contract_rebake = rebake_version in {"rebake_v2", "v1_fixed_rebake"}
    cached_render_root = (
        None
        if disable_render_cache or is_contract_rebake
        else resolve_cached_render_root(record, source_model_path, abo_render_cache, views)
    )
    if render_bundle_complete(buffer_root, views):
        render_mode = "existing_bundle"
        print(f"[prepare reuse] existing bundle buffers for {record['object_id']}")
    elif cached_render_root is not None:
        link_or_copy_render_root(cached_render_root, buffer_root)
        render_mode = "abo_render_cache"
        print(
            f"[prepare reuse] cached ABO render for {record['object_id']} <- {cached_render_root.name}"
        )
    elif not skip_render and blender_bin.exists():
        ensure_dir(buffer_root)
        render_views(
            source_model_path=source_model_path,
            buffer_root=buffer_root,
            views_json=views_json,
            views=views,
            blender_bin=blender_bin,
            render_resolution=render_resolution,
            cycles_samples=cycles_samples,
            cuda_device_index=cuda_device_index,
        )
        render_mode = "rendered"
    else:
        ensure_dir(buffer_root)
        render_mode = "placeholder"
    view_buffer_payload = finalize_view_buffers(buffer_root, render_resolution, views)

    uv_target_payload = bake_uv_targets(
        bundle_dir=bundle_dir,
        uv_albedo_path=albedo_dst,
        uv_prior_roughness_path=roughness_dst,
        uv_prior_metallic_path=metallic_dst,
        uv_target_roughness_source_path=roughness_src,
        uv_target_metallic_source_path=metallic_src,
        canonical_buffer_root=buffer_root,
        atlas_resolution=atlas_resolution,
        default_roughness=roughness_seed,
        default_metallic=metallic_seed,
        synthesize_nontrivial_prior=should_synthesize_nontrivial_prior(record),
        allow_prior_copy_fallback=not disallow_prior_copy_fallback,
    )

    rebake_v2_payload: dict[str, object] = {}
    if is_contract_rebake:
        contract_version = "v1_fixed" if rebake_version == "v1_fixed_rebake" else "v2"
        rebake_v2_payload = compute_rebake_v2_contract(
            bundle_dir=bundle_dir,
            buffer_root=buffer_root,
            views=views,
            uv_prior_roughness_path=Path(uv_target_payload["uv_prior_roughness_path"]),
            uv_prior_metallic_path=Path(uv_target_payload["uv_prior_metallic_path"]),
            uv_target_roughness_path=Path(uv_target_payload["uv_target_roughness_path"]),
            uv_target_metallic_path=Path(uv_target_payload["uv_target_metallic_path"]),
            view_buffer_field_sources_path=Path(view_buffer_payload["view_buffer_field_sources_path"]),
            atlas_resolution=atlas_resolution,
            render_resolution=render_resolution,
            mean_threshold=target_view_alignment_mean_threshold,
            p95_threshold=target_view_alignment_p95_threshold,
            rebake_version=rebake_version,
            target_view_contract_version=contract_version,
        )

    missing = find_missing_paths(bundle_dir)
    if missing:
        raise RuntimeError(f"incomplete_bundle_after_prepare:{','.join(missing)}")

    prepared = build_prepared_record(record, bundle_root, bundle_dir, split)
    prepared.update(
        {
            "scalar_prior_roughness": roughness_seed,
            "scalar_prior_metallic": metallic_seed,
            "uv_target_roughness_path": uv_target_payload["uv_target_roughness_path"],
            "uv_target_metallic_path": uv_target_payload["uv_target_metallic_path"],
            "uv_target_confidence_path": uv_target_payload["uv_target_confidence_path"],
            "has_material_prior": has_material_prior,
            "prior_mode": prior_mode,
            "render_mode": render_mode,
            "target_source_type": uv_target_payload["target_source_type"],
            "target_is_prior_copy": uv_target_payload["target_is_prior_copy"],
            "target_prior_identity": uv_target_payload["target_prior_identity"],
            "target_confidence_summary": uv_target_payload["target_confidence_summary"],
            "target_confidence_mean": uv_target_payload["target_confidence_mean"],
            "target_confidence_nonzero_rate": uv_target_payload["target_confidence_nonzero_rate"],
            "target_coverage": uv_target_payload["target_coverage"],
            "target_quality_tier": "",
            "view_rgba_paths": view_buffer_payload["view_rgba_paths"],
            "view_buffer_field_sources_path": view_buffer_payload["view_buffer_field_sources_path"],
            "view_supervision_ready": view_buffer_payload["view_supervision_ready"],
            "valid_view_count": view_buffer_payload["valid_view_count"],
            "view_light_protocol": view_light_protocol,
            "view_count": len(views),
            "lighting_bank_id": (
                f"polyhaven_{view_light_protocol}_v1"
                if view_light_protocol != "canonical_triplet"
                else str(record.get("lighting_bank_id") or "canonical_triplet_v1")
            ),
            "hdri_asset_ids": sorted(
                {
                    str(view.get("lighting_asset_id"))
                    for view in views
                    if view.get("lighting_asset_id")
                }
            ),
            "paper_split": str(record.get("paper_split") or record.get("default_split", "train")),
        }
    )
    if rebake_v2_payload:
        prepared.update(rebake_v2_payload)
        prepared["candidate_pool_only"] = True
        prepared["paper_stage_eligible_rebake_v2"] = False
        prepared["target_quality_tier"] = str(prepared.get("target_quality_tier") or "smoke_only")
        prepared["old_view_reuse"] = False
        prepared["old_uv_target_reuse"] = False
    prepared = enrich_record_with_quality_fields(prepared)
    write_prepared_record_file(bundle_dir, prepared)
    return prepared


def main() -> None:
    args = parse_args()
    payload = read_manifest(args.input_manifest)
    hdri_bank = load_hdri_bank(args.hdri_bank_json)
    if args.view_light_protocol != "canonical_triplet" and int(args.min_hdri_count) > 0:
        if len(hdri_bank) < int(args.min_hdri_count):
            raise RuntimeError(
                f"insufficient_hdri_bank:{len(hdri_bank)}<{int(args.min_hdri_count)}:{args.hdri_bank_json}"
            )
    output_root = ensure_dir(args.output_root / args.split)
    bundle_root = ensure_dir(output_root / "canonical_bundle")
    output_manifest = (
        args.output_manifest
        if args.output_manifest is not None
        else output_root / f"canonical_manifest_{args.split}.json"
    )
    partial_manifest_path = args.partial_manifest
    if partial_manifest_path is None and args.split == "full" and output_manifest.name == "canonical_manifest_full.json":
        partial_manifest_path = output_root / "canonical_manifest_partial.json"
    summary_json_path = (
        args.summary_json
        if args.summary_json is not None
        else output_root / f"prepare_{args.split}_summary.json"
    )
    summary_md_path = (
        args.summary_md
        if args.summary_md is not None
        else output_root / f"prepare_{args.split}_summary.md"
    )
    selected = select_records(payload, args.split, args.max_records)
    skipped_records: list[dict] = []
    prepared_indexed: list[tuple[int, dict]] = []
    last_partial_refresh_completed = -1

    def refresh_partial_manifest(force: bool = False) -> None:
        nonlocal last_partial_refresh_completed
        if partial_manifest_path is None:
            return
        completed = len(prepared_indexed) + len(skipped_records)
        if not force:
            if args.refresh_partial_every <= 0:
                return
            if completed == 0:
                return
            if completed < last_partial_refresh_completed + args.refresh_partial_every:
                return
        partial_payload = build_partial_manifest(payload, bundle_root, args.split)
        partial_payload["input_manifest"] = str(args.input_manifest.resolve())
        write_json(partial_manifest_path, partial_payload)
        last_partial_refresh_completed = completed
        print(
            "[prepare partial] "
            f"prepared={partial_payload['counts']['prepared_records']} "
            f"skipped={partial_payload['counts']['skipped_records']} "
            f"path={partial_manifest_path}"
        )

    def run_prepare(index: int, record: dict) -> tuple[int, dict]:
        views = build_views_for_record(
            record,
            protocol=args.view_light_protocol,
            hdri_bank=hdri_bank,
            max_hdri_lights=args.max_hdri_lights,
            hdri_selection_offset=args.hdri_selection_offset,
        )
        prepared = prepare_record(
            record,
            output_root=output_root,
            split=args.split,
            atlas_resolution=args.atlas_resolution,
            render_resolution=args.render_resolution,
            cycles_samples=args.cycles_samples,
            blender_bin=args.blender_bin,
            abo_render_cache=args.abo_render_cache,
            cuda_device_index=args.cuda_device_index,
            skip_render=args.skip_render,
            views=views,
            view_light_protocol=args.view_light_protocol,
            rebake_version=args.rebake_version,
            disable_render_cache=args.disable_render_cache,
            disallow_prior_copy_fallback=args.disallow_prior_copy_fallback
            or args.rebake_version in {"rebake_v2", "v1_fixed_rebake"},
            target_view_alignment_mean_threshold=args.target_view_alignment_mean_threshold,
            target_view_alignment_p95_threshold=args.target_view_alignment_p95_threshold,
        )
        return index, prepared

    refresh_partial_manifest(force=True)

    if args.parallel_workers <= 1:
        for index, record in enumerate(selected):
            try:
                prepared_indexed.append(run_prepare(index, record))
            except Exception as exc:  # noqa: BLE001
                skipped_records.append(
                    {
                        "object_id": record["object_id"],
                        "source_name": record["source_name"],
                        "reason": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
            refresh_partial_manifest()
    else:
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
            future_to_record = {
                executor.submit(run_prepare, index, record): (index, record)
                for index, record in enumerate(selected)
            }
            for future in as_completed(future_to_record):
                _index, record = future_to_record[future]
                try:
                    prepared_indexed.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    skipped_records.append(
                        {
                            "object_id": record["object_id"],
                            "source_name": record["source_name"],
                            "reason": type(exc).__name__,
                            "detail": str(exc),
                        }
                    )
                refresh_partial_manifest()

    prepared_records = [
        prepared
        for _index, prepared in sorted(prepared_indexed, key=lambda item: item[0])
    ]

    out_payload = build_prepare_manifest_payload(
        input_manifest=args.input_manifest,
        split=args.split,
        bundle_root=bundle_root,
        selected=selected,
        prepared_records=prepared_records,
        skipped_records=skipped_records,
    )
    write_json(output_manifest, out_payload)
    refresh_partial_manifest(force=True)
    summary_payload = build_run_summary(
        output_manifest=output_manifest,
        split=args.split,
        selected=selected,
        prepared_records=prepared_records,
        skipped_records=skipped_records,
    )
    write_json(summary_json_path, summary_payload)
    write_summary_md(summary_md_path, summary_payload)
    print(f"wrote {output_manifest}")
    print(json.dumps(out_payload["counts"], indent=2))
    print(f"wrote {summary_json_path}")
    print(f"wrote {summary_md_path}")


if __name__ == "__main__":
    main()
