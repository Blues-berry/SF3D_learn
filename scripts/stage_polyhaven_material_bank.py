#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "highlight_pool_a_8k" / "aux_sources" / "polyhaven_materials"
USER_AGENT = "stable-fast-3d-dataset-builder/0.1 (research; local contact: ubuntu)"
ASSETS_URL = "https://api.polyhaven.com/assets?t=textures"
FILES_URL = "https://api.polyhaven.com/files/{asset_id}"

MATERIAL_BUCKETS: dict[str, tuple[str, ...]] = {
    "metal_dominant": (
        "metal",
        "steel",
        "iron",
        "brass",
        "copper",
        "aluminium",
        "aluminum",
        "rust",
        "chrome",
    ),
    "ceramic_glazed_lacquer": (
        "ceramic",
        "porcelain",
        "tile",
        "tiles",
        "marble",
        "glazed",
        "terrazzo",
    ),
    "glass_metal": (
        "glass",
        "mirror",
        "window",
        "transparent",
    ),
    "mixed_thin_boundary": (
        "mosaic",
        "woven",
        "wire",
        "mesh",
        "panel",
        "planks",
        "trim",
    ),
    "glossy_non_metal": (
        "plastic",
        "leather",
        "paint",
        "painted",
        "polished",
        "varnished",
        "lacquer",
        "wood",
    ),
}

MAP_ALIASES: dict[str, tuple[str, ...]] = {
    "diffuse": ("Diffuse", "Color", "Base Color", "BaseColor", "Albedo"),
    "roughness": ("Rough", "Roughness"),
    "normal": ("nor_gl", "NormalGL", "Normal", "nor_dx", "NormalDX"),
    "height": ("Displacement", "Height", "Bump"),
    "metallic": ("Metalness", "Metallic", "Metal"),
    "ao": ("AO", "Ambient Occlusion"),
    "opacity": ("Opacity", "Alpha"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage a CC0 Poly Haven PBR material prior bank.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--resolution", type=str, default="1k")
    parser.add_argument("--format", dest="file_format", choices=("jpg", "png", "exr"), default="jpg")
    parser.add_argument(
        "--maps",
        type=str,
        default="diffuse,roughness,normal,metallic,height,ao,opacity",
        help="Comma-separated canonical maps to try per material.",
    )
    parser.add_argument("--request-delay", type=float, default=0.08)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def request_json(url: str) -> dict[str, Any]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    response.raise_for_status()
    return response.json()


def material_bucket(asset_id: str, metadata: dict[str, Any]) -> tuple[str, int]:
    haystack = " ".join(
        [
            asset_id,
            str(metadata.get("name") or ""),
            str(metadata.get("description") or ""),
            " ".join(str(tag) for tag in metadata.get("tags") or []),
            " ".join(str(category) for category in metadata.get("categories") or []),
        ]
    ).lower()
    scores = {
        bucket: sum(1 for keyword in keywords if keyword in haystack)
        for bucket, keywords in MATERIAL_BUCKETS.items()
    }
    bucket, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return "glossy_non_metal", 0
    return bucket, score


def select_assets(assets: dict[str, Any], target: int) -> list[tuple[str, dict[str, Any], str]]:
    buckets: dict[str, list[tuple[int, str, dict[str, Any]]]] = {bucket: [] for bucket in MATERIAL_BUCKETS}
    for asset_id, metadata in assets.items():
        bucket, score = material_bucket(asset_id, metadata)
        buckets.setdefault(bucket, []).append((score, asset_id, metadata))
    for rows in buckets.values():
        rows.sort(key=lambda item: (-item[0], item[1]))

    quota = {
        "metal_dominant": 0.30,
        "ceramic_glazed_lacquer": 0.20,
        "glass_metal": 0.10,
        "mixed_thin_boundary": 0.20,
        "glossy_non_metal": 0.20,
    }
    selected: list[tuple[str, dict[str, Any], str]] = []
    used: set[str] = set()
    for bucket, ratio in quota.items():
        bucket_target = max(1, round(target * ratio))
        for _score, asset_id, metadata in buckets.get(bucket, [])[:bucket_target]:
            if asset_id not in used:
                selected.append((asset_id, metadata, bucket))
                used.add(asset_id)
    if len(selected) < target:
        remaining = [
            (score, asset_id, metadata, bucket)
            for bucket, rows in buckets.items()
            for score, asset_id, metadata in rows
            if asset_id not in used
        ]
        remaining.sort(key=lambda item: (-item[0], item[3], item[1]))
        for _score, asset_id, metadata, bucket in remaining[: target - len(selected)]:
            selected.append((asset_id, metadata, bucket))
            used.add(asset_id)
    return selected[:target]


def choose_file(
    file_tree: dict[str, Any],
    canonical_map: str,
    resolution: str,
    file_format: str,
) -> tuple[str, dict[str, Any]] | None:
    for map_name in MAP_ALIASES.get(canonical_map, (canonical_map,)):
        variants = file_tree.get(map_name)
        if not isinstance(variants, dict):
            continue
        resolution_options = [resolution, "2k", "1k", "4k", "8k"]
        for candidate_resolution in resolution_options:
            formats = variants.get(candidate_resolution)
            if not isinstance(formats, dict):
                continue
            if file_format in formats:
                return map_name, formats[file_format]
            for fallback_format in ("jpg", "png", "exr"):
                if fallback_format in formats:
                    return map_name, formats[fallback_format]
    return None


def download_file(url: str, path: Path, expected_size: int | None) -> str:
    if path.exists() and (not expected_size or path.stat().st_size == expected_size):
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=120) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return "downloaded"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["material_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    requested_maps = [name.strip().lower() for name in args.maps.split(",") if name.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    assets = request_json(ASSETS_URL)
    selected = select_assets(assets, args.target)

    records: list[dict[str, Any]] = []
    map_records: list[dict[str, Any]] = []
    for index, (asset_id, metadata, bucket) in enumerate(selected, start=1):
        file_tree = request_json(FILES_URL.format(asset_id=asset_id))
        time.sleep(args.request_delay)
        material_id = f"polyhaven_mat_{stable_id(asset_id)}"
        material_dir = args.output_root / "materials" / material_id
        downloaded_maps: dict[str, str] = {}
        for canonical_map in requested_maps:
            chosen = choose_file(file_tree, canonical_map, args.resolution, args.file_format)
            if chosen is None:
                continue
            source_map_name, file_info = chosen
            url = str(file_info["url"])
            suffix = Path(url.split("?")[0]).suffix or f".{args.file_format}"
            local_path = material_dir / f"{canonical_map}{suffix}"
            status = download_file(url, local_path, int(file_info.get("size") or 0) or None)
            downloaded_maps[canonical_map] = str(local_path)
            map_records.append(
                {
                    "material_id": material_id,
                    "asset_id": asset_id,
                    "canonical_map": canonical_map,
                    "source_map_name": source_map_name,
                    "status": status,
                    "local_path": str(local_path),
                    "url": url,
                    "bytes": file_info.get("size") or "",
                }
            )
        records.append(
            {
                "source_name": "PolyHaven_CC0_PBR_material_bank",
                "material_id": material_id,
                "asset_id": asset_id,
                "name": metadata.get("name") or asset_id,
                "license_bucket": "cc0",
                "highlight_material_class": bucket,
                "downloaded_map_count": len(downloaded_maps),
                "local_dir": str(material_dir),
                "polyhaven_categories": "|".join(str(item) for item in metadata.get("categories") or []),
                "polyhaven_tags": "|".join(str(item) for item in metadata.get("tags") or []),
            }
        )
        print(f"[{index}/{len(selected)}] {asset_id}: {len(downloaded_maps)} maps -> {bucket}", flush=True)

    payload = {
        "generated_at_utc": utc_now(),
        "source_name": "PolyHaven_CC0_PBR_material_bank",
        "target": args.target,
        "selected_count": len(records),
        "resolution": args.resolution,
        "preferred_format": args.file_format,
        "requested_maps": requested_maps,
        "downloaded_map_count": len(map_records),
        "license_policy": "Poly Haven asset pages state assets are CC0; API used with a User-Agent for research staging.",
        "material_counts": dict(Counter(row["highlight_material_class"] for row in records)),
        "records": records,
        "maps": map_records,
    }
    write_csv(args.output_root / "polyhaven_material_bank_manifest.csv", records)
    write_csv(args.output_root / "polyhaven_material_bank_maps.csv", map_records)
    write_json(args.output_root / "polyhaven_material_bank_manifest.json", payload)
    print(json.dumps({key: payload[key] for key in payload if key != "maps" and key != "records"}, indent=2))


if __name__ == "__main__":
    main()
