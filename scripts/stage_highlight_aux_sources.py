#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "highlight_pool_a_8k" / "aux_sources"
POLYHAVEN_ASSETS_URL = "https://api.polyhaven.com/assets?t=hdris"
POLYHAVEN_FILES_URL = "https://api.polyhaven.com/files/{asset_id}"


HDRI_STRATA = {
    "indoor_high_contrast": ("indoor", "high contrast", "window", "artificial light"),
    "indoor_soft_window": ("indoor", "natural light", "windows", "studio"),
    "outdoor_sun_hard": ("outdoor", "sunny", "clear", "urban"),
    "outdoor_overcast_soft": ("outdoor", "overcast", "cloudy", "forest"),
    "night_urban_neon": ("night", "urban", "street", "artificial light"),
    "studio_product": ("studio", "interior", "soft", "artificial light"),
}

OBJAVERSE_CLASS_KEYWORDS = {
    "metal_dominant": ("metal", "chrome", "steel", "aluminum", "brass", "copper", "hardware"),
    "ceramic_glazed_lacquer": ("ceramic", "porcelain", "vase", "tile", "glazed", "lacquer", "marble"),
    "glass_metal": ("glass", "lamp", "lantern", "mirror", "transparent", "chandelier"),
    "mixed_thin_boundary": ("frame", "handle", "wire", "rack", "shelf", "chair", "table", "furniture"),
    "glossy_non_metal": ("glossy", "plastic", "leather", "polished", "painted", "acrylic", "resin"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage auxiliary highlight sources while Pool-A rendering is already running."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--polyhaven-count", type=int, default=60)
    parser.add_argument("--polyhaven-workers", type=int, default=8)
    parser.add_argument("--objaverse-target", type=int, default=1500)
    parser.add_argument("--objaverse-processes", type=int, default=8)
    parser.add_argument(
        "--objaverse-allowed-sources",
        type=str,
        default="sketchfab,smithsonian",
        help="Comma-separated Objaverse-XL sources to keep before download; exclude github by default to avoid git-lfs stalls.",
    )
    parser.add_argument("--download-objaverse", action="store_true")
    parser.add_argument("--skip-polyhaven", action="store_true")
    return parser.parse_args()


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "stable-fast-3d-dataset-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "item"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def score_asset_for_stratum(asset: dict[str, Any], keywords: tuple[str, ...]) -> int:
    terms = " ".join(
        [
            str(asset.get("name") or ""),
            " ".join(str(item) for item in asset.get("categories", []) or []),
            " ".join(str(item) for item in asset.get("tags", []) or []),
        ]
    ).lower()
    return sum(1 for keyword in keywords if keyword in terms)


def choose_polyhaven_assets(asset_payload: dict[str, Any], target_count: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    per_round = max(1, target_count // len(HDRI_STRATA))
    for stratum, keywords in HDRI_STRATA.items():
        scored = []
        for asset_id, asset in asset_payload.items():
            if asset_id in used:
                continue
            score = score_asset_for_stratum(asset, keywords)
            if score <= 0:
                continue
            scored.append((score, asset_id, asset))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for _score, asset_id, asset in scored[:per_round]:
            used.add(asset_id)
            chosen.append({"asset_id": asset_id, "stratum": stratum, **asset})
    if len(chosen) < target_count:
        leftovers = [
            (asset_id, asset)
            for asset_id, asset in sorted(asset_payload.items())
            if asset_id not in used
        ]
        for asset_id, asset in leftovers[: target_count - len(chosen)]:
            used.add(asset_id)
            chosen.append({"asset_id": asset_id, "stratum": "fallback_diverse", **asset})
    return chosen[:target_count]


def select_hdri_file(files_payload: dict[str, Any]) -> tuple[str, int, str]:
    hdri = files_payload.get("hdri", {})
    for resolution in ("1k", "2k", "4k"):
        entry = hdri.get(resolution, {}).get("hdr")
        if isinstance(entry, dict) and entry.get("url"):
            return str(entry["url"]), int(entry.get("size") or 0), resolution
    for resolution, by_format in hdri.items():
        entry = by_format.get("hdr") if isinstance(by_format, dict) else None
        if isinstance(entry, dict) and entry.get("url"):
            return str(entry["url"]), int(entry.get("size") or 0), str(resolution)
    raise RuntimeError("no_hdr_file_in_polyhaven_payload")


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "stable-fast-3d-dataset-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp_path.replace(output_path)


def stage_polyhaven(output_root: Path, count: int, workers: int) -> dict[str, Any]:
    hdri_root = output_root / "polyhaven_hdri"
    assets = fetch_json(POLYHAVEN_ASSETS_URL)
    selected = choose_polyhaven_assets(assets, count)
    jobs = []
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for asset in selected:
            asset_id = str(asset["asset_id"])
            files = fetch_json(POLYHAVEN_FILES_URL.format(asset_id=asset_id))
            url, size, resolution = select_hdri_file(files)
            local_path = hdri_root / f"{slugify(asset_id)}_{resolution}.hdr"
            rows.append(
                {
                    "source_name": "PolyHaven_HDRI",
                    "asset_id": asset_id,
                    "name": asset.get("name", asset_id),
                    "stratum": asset.get("stratum", ""),
                    "license_bucket": "cc0",
                    "resolution": resolution,
                    "download_url": url,
                    "expected_size_bytes": size,
                    "local_path": str(local_path),
                    "download_status": "queued",
                }
            )
            jobs.append((asset_id, executor.submit(download_file, url, local_path)))
        for asset_id, future in jobs:
            status = "downloaded"
            error = ""
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            for row in rows:
                if row["asset_id"] == asset_id:
                    row["download_status"] = status
                    row["error"] = error
                    break
    manifest_csv = output_root / "polyhaven_hdri_bank_60.csv"
    manifest_json = output_root / "polyhaven_hdri_bank_60.json"
    write_csv(manifest_csv, rows)
    write_json(manifest_json, {"source": "PolyHaven_HDRI", "count": len(rows), "records": rows})
    return {
        "source": "PolyHaven_HDRI",
        "records": len(rows),
        "downloaded": sum(row["download_status"] == "downloaded" for row in rows),
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
    }


def objaverse_material_class(row: Any) -> tuple[str, str]:
    values = []
    for column in row.index:
        value = row[column]
        if isinstance(value, str):
            values.append(value)
    text = " ".join(values).lower()
    scores = {
        material_class: sum(1 for keyword in keywords if keyword in text)
        for material_class, keywords in OBJAVERSE_CLASS_KEYWORDS.items()
    }
    material_class, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return "unclassified", "metadata_keyword:none"
    return material_class, "metadata_keyword:objaverse_text"


def license_allowed(value: Any) -> bool:
    text = str(value or "").lower()
    if not text:
        return False
    blocked = ("noncommercial", "non-commercial", "nc", "nd", "noai", "unknown")
    if any(token in text for token in blocked):
        return False
    allowed = ("cc0", "public", "by", "odc", "mit", "apache", "bsd")
    return any(token in text for token in allowed)


def select_balanced_objaverse_rows(scored: list[tuple[str, str, Any, str, Any]], target: int) -> list[tuple[str, str, Any, str, Any]]:
    grouped: dict[str, list[tuple[str, str, Any, str, Any]]] = {}
    for item in scored:
        grouped.setdefault(item[0], []).append(item)
    for rows in grouped.values():
        rows.sort(key=lambda item: (str(item[2]), item[1]))
    selected: list[tuple[str, str, Any, str, Any]] = []
    while len(selected) < target:
        before = len(selected)
        for material_class in sorted(grouped):
            rows = grouped[material_class]
            if rows:
                selected.append(rows.pop(0))
                if len(selected) >= target:
                    break
        if len(selected) == before:
            break
    return selected


def stage_objaverse(
    output_root: Path,
    target: int,
    processes: int,
    download: bool,
    allowed_sources: str,
) -> dict[str, Any]:
    import objaverse.xl as oxl

    objaverse_root = output_root / "objaverse_xl"
    annotations = oxl.get_annotations(download_dir=str(objaverse_root))
    rows = []
    license_column = "license" if "license" in annotations.columns else None
    file_type_column = "fileType" if "fileType" in annotations.columns else None
    source_column = "source" if "source" in annotations.columns else None
    allowed_source_set = {
        item.strip().lower()
        for item in allowed_sources.split(",")
        if item.strip()
    }
    filtered = annotations
    if source_column is not None and allowed_source_set:
        filtered = filtered[filtered[source_column].astype(str).str.lower().isin(allowed_source_set)]
    if license_column is not None:
        filtered = filtered[filtered[license_column].map(license_allowed)]
    if file_type_column is not None:
        filtered = filtered[filtered[file_type_column].astype(str).str.lower().isin(["glb", "gltf", "obj", "fbx"])]

    scored = []
    fallback_items = []
    for index, row in filtered.iterrows():
        material_class, reason = objaverse_material_class(row)
        if material_class == "unclassified":
            fallback_items.append(
                (
                    "unknown_pending_second_pass",
                    str(index),
                    index,
                    "license_format_filtered_pending_material_probe",
                    row,
                )
            )
            continue
        scored.append((material_class, str(index), index, reason, row))
    selected_items = select_balanced_objaverse_rows(scored, target)
    if len(selected_items) < target:
        fallback_items.sort(key=lambda item: (str(item[4].get(source_column, "")) if source_column else "", item[1]))
        selected_items.extend(fallback_items[: target - len(selected_items)])
    selected_indices = [item[2] for item in selected_items]
    selected = filtered.loc[selected_indices] if selected_indices else filtered.head(0)
    class_by_index = {str(item[2]): (item[0], item[3]) for item in selected_items}

    for index, row in selected.iterrows():
        material_class, reason = class_by_index.get(str(index), ("unclassified", "metadata_keyword:none"))
        rows.append(
            {
                "source_name": "Objaverse-XL_strict_filtered",
                "object_id": f"objaverse_{index}",
                "source_uid": str(index),
                "source": str(row.get(source_column, "unknown")) if source_column else "unknown",
                "license_bucket": str(row.get(license_column, "unknown")) if license_column else "unknown",
                "format": str(row.get(file_type_column, "unknown")) if file_type_column else "unknown",
                "highlight_material_class": material_class,
                "material_class_source": reason,
                "download_status": "metadata_selected",
                "local_path": "",
            }
        )

    download_results: dict[str, str] = {}
    download_error = ""
    if download and len(selected) > 0:
        try:
            download_results = oxl.download_objects(
                objects=selected,
                download_dir=str(objaverse_root),
                processes=processes,
            )
        except Exception as exc:  # noqa: BLE001
            download_error = f"{type(exc).__name__}: {exc}"
        for row in rows:
            local_path = download_results.get(row["source_uid"], "")
            row["local_path"] = local_path
            row["download_status"] = "downloaded" if local_path else "download_missing"

    manifest_csv = output_root / "objaverse_xl_strict_filtered_manifest.csv"
    manifest_json = output_root / "objaverse_xl_strict_filtered_manifest.json"
    write_csv(manifest_csv, rows)
    write_json(
        manifest_json,
        {
            "source": "Objaverse-XL_strict_filtered",
            "allowed_sources": sorted(allowed_source_set),
            "download_requested": download,
            "download_error": download_error,
            "annotations_columns": list(annotations.columns),
            "filtered_count": int(len(filtered)),
            "selected_count": len(rows),
            "download_result_count": len(download_results),
            "records": rows,
        },
    )
    return {
        "source": "Objaverse-XL_strict_filtered",
        "selected": len(rows),
        "download_requested": download,
        "download_error": download_error,
        "downloaded": sum(row["download_status"] == "downloaded" for row in rows),
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summary: dict[str, Any] = {"started_unix": started, "outputs": []}
    if not args.skip_polyhaven:
        summary["outputs"].append(
            stage_polyhaven(args.output_root, args.polyhaven_count, args.polyhaven_workers)
        )
    summary["outputs"].append(
        stage_objaverse(
            args.output_root,
            target=args.objaverse_target,
            processes=args.objaverse_processes,
            download=args.download_objaverse,
            allowed_sources=args.objaverse_allowed_sources,
        )
    )
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    write_json(args.output_root / "aux_source_stage_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
