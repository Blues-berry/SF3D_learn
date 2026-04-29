#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DEFAULT_OBJAVERSE_ROOT = PROJECT_ROOT / "output" / "highlight_pool_a_8k" / "aux_sources" / "objaverse_xl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "highlight_pool_a_8k" / "objaverse_cached_increment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and download Objaverse-XL increment rows from already cached source parquet files."
    )
    parser.add_argument("--objaverse-root", type=Path, default=DEFAULT_OBJAVERSE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sources", type=str, default="sketchfab,smithsonian")
    parser.add_argument("--target", type=int, default=1500)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--file-types", type=str, default="glb,gltf,obj,fbx")
    parser.add_argument(
        "--priority-material-families",
        type=str,
        default="",
        help="Comma-separated material families used as a deterministic tie-breaker during selection.",
    )
    parser.add_argument(
        "--target-material-family-ratios",
        type=str,
        default="",
        help="Comma-separated material quotas, for example glass_metal=0.30,ceramic_glazed_lacquer=0.30.",
    )
    parser.add_argument(
        "--source-priority",
        type=str,
        default="",
        help="Comma-separated source priority used as a tie-breaker, for example smithsonian,sketchfab.",
    )
    parser.add_argument(
        "--github-save-repo-format",
        choices=("files", "zip", "tar", "tar.gz"),
        default="files",
        help="GitHub Objaverse rows are repository-based; saving files keeps object-local paths usable.",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=None,
        help="Existing merged/source manifest used to exclude already-staged Objaverse source_uids/fileIdentifiers.",
    )
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:24]


def license_allowed(value: Any) -> bool:
    text = str(value or "").lower()
    if not text:
        return False
    blocked_tokens = (
        "noncommercial",
        "non-commercial",
        "nc",
        "nd",
        "noai",
        "unknown",
        "gpl",
        "lgpl",
        "agpl",
        "mozilla public",
        "eclipse public",
        "common public",
        "copyleft",
    )
    if any(token in text for token in blocked_tokens):
        return False
    cc_attribution = text.startswith("creative commons - attribution")
    return any(
        token in text
        for token in (
            "cc0",
            "zero",
            "public domain",
            "mit license",
            "apache license",
            "bsd",
            "the unlicense",
            "isc license",
            "zlib license",
            "boost software license",
        )
    ) or cc_attribution


def material_class_from_row(row: pd.Series) -> tuple[str, str]:
    text = " ".join(
        str(row.get(column) or "")
        for column in ("fileIdentifier", "metadata")
    ).lower()
    checks = {
        "metal_dominant": (
            "metal", "metallic", "chrome", "steel", "aluminum", "aluminium", "brass", "copper",
            "hardware", "silver", "gold", "iron", "tool", "gear", "bolt", "screw", "hinge",
        ),
        "ceramic_glazed_lacquer": (
            "ceramic", "porcelain", "pottery", "vase", "tile", "glazed", "glaze", "lacquer",
            "marble", "stoneware", "bowl", "plate", "dish", "cup", "mug", "sink", "toilet",
            "statue", "figurine",
        ),
        "glass_metal": (
            "glass", "lamp", "lantern", "mirror", "transparent", "translucent", "chandelier",
            "bottle", "jar", "goblet", "window", "vitrine", "bulb", "crystal",
        ),
        "mixed_thin_boundary": (
            "frame", "handle", "wire", "rack", "shelf", "chair", "table", "furniture",
            "fence", "rail", "basket", "cage", "stand",
        ),
        "glossy_non_metal": (
            "glossy", "plastic", "leather", "polished", "painted", "acrylic", "resin",
            "varnish", "paint", "synthetic",
        ),
    }
    scores = {
        material_class: sum(1 for token in tokens if token in text)
        for material_class, tokens in checks.items()
    }
    material_class, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return "unknown_pending_second_pass", "license_format_filtered_pending_material_probe"
    return material_class, "cached_metadata_keyword"


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_ratio_csv(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    ratios: dict[str, float] = {}
    for item in str(value).split(","):
        if not item.strip() or "=" not in item:
            continue
        key, raw = item.split("=", 1)
        key = key.strip()
        try:
            ratio = float(raw)
        except ValueError:
            continue
        if ratio > 0.0:
            ratios[key] = ratio
    total = sum(ratios.values())
    if total > 0.0:
        ratios = {key: value / total for key, value in ratios.items()}
    return ratios


def load_filtered_rows(objaverse_root: Path, sources: list[str], file_types: set[str]) -> pd.DataFrame:
    frames = []
    for source in sources:
        parquet_path = objaverse_root / source / f"{source}.parquet"
        if not parquet_path.exists():
            continue
        frame = pd.read_parquet(parquet_path)
        frame = frame[frame["license"].map(license_allowed)]
        frame = frame[frame["fileType"].astype(str).str.lower().isin(file_types)]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["fileIdentifier", "source", "license", "fileType", "sha256", "metadata"])
    return pd.concat(frames, axis=0, ignore_index=True)


def load_excluded_source_uids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    excluded: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_name = str(row.get("source_name") or "")
        source_uid = str(row.get("source_uid") or "")
        object_id = str(row.get("object_id") or "")
        if "objaverse" not in source_name.lower() and not object_id.startswith("objaverse_"):
            continue
        if source_uid:
            excluded.add(source_uid)
    return excluded


def select_rows(
    frame: pd.DataFrame,
    target: int,
    *,
    priority_material_families: list[str],
    target_material_family_ratios: dict[str, float],
    source_priority: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    for index, row in frame.iterrows():
        material_class, material_reason = material_class_from_row(row)
        rows.append((material_class, str(row["source"]), str(row["fileIdentifier"]), material_reason, index))
    material_rank = {
        material_family: rank
        for rank, material_family in enumerate(priority_material_families)
    }
    source_rank = {source_name: rank for rank, source_name in enumerate(source_priority)}
    rows.sort(
        key=lambda item: (
            item[0] == "unknown_pending_second_pass",
            material_rank.get(item[0], len(material_rank)),
            item[0],
            source_rank.get(item[1], len(source_rank)),
            item[1],
            item[2],
        )
    )

    if target_material_family_ratios:
        grouped: dict[str, list[tuple[str, str, str, str, int]]] = {}
        for row in rows:
            grouped.setdefault(row[0], []).append(row)
        raw_quotas = {
            material_family: ratio * int(target)
            for material_family, ratio in target_material_family_ratios.items()
        }
        quotas = {material_family: int(raw) for material_family, raw in raw_quotas.items()}
        remainder = max(0, int(target) - sum(quotas.values()))
        fractional = sorted(
            raw_quotas.items(),
            key=lambda pair: (
                pair[1] - int(pair[1]),
                -material_rank.get(pair[0], 999),
            ),
            reverse=True,
        )
        for material_family, _raw in fractional[:remainder]:
            quotas[material_family] += 1
        selected_indices = []
        used_indices: set[int] = set()
        for material_family in sorted(
            target_material_family_ratios,
            key=lambda family: material_rank.get(family, 999),
        ):
            take = min(quotas.get(material_family, 0), len(grouped.get(material_family, [])))
            for item in grouped.get(material_family, [])[:take]:
                selected_indices.append(item[4])
                used_indices.add(item[4])
        if len(selected_indices) < target:
            for item in rows:
                if item[4] in used_indices:
                    continue
                selected_indices.append(item[4])
                used_indices.add(item[4])
                if len(selected_indices) >= target:
                    break
    else:
        selected_indices = [item[4] for item in rows[:target]]

    selected = frame.loc[selected_indices].copy()
    material_by_identifier = {
        str(frame.loc[index, "fileIdentifier"]): material_class_from_row(frame.loc[index])
        for index in selected_indices
    }
    selected["highlight_material_class"] = [
        material_by_identifier[str(identifier)][0] for identifier in selected["fileIdentifier"]
    ]
    selected["material_class_source"] = [
        material_by_identifier[str(identifier)][1] for identifier in selected["fileIdentifier"]
    ]
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["object_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconcile_cached_downloads(download_root: Path, selected: pd.DataFrame) -> dict[str, str]:
    if selected.empty or "sha256" not in selected.columns:
        return {}
    wanted = {
        str(row["sha256"]).lower(): str(row["fileIdentifier"])
        for _index, row in selected.iterrows()
        if str(row.get("sha256") or "").strip()
    }
    if not wanted:
        return {}
    search_roots = [
        download_root / "hf-objaverse-v1" / "glbs",
        download_root / "smithsonian",
    ]
    recovered: dict[str, str] = {}
    extensions = {".glb", ".gltf", ".obj", ".fbx"}
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if len(recovered) >= len(wanted):
                return recovered
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                digest = file_sha256(path)
            except OSError:
                continue
            identifier = wanted.get(digest)
            if identifier and identifier not in recovered:
                recovered[identifier] = str(path)
    return recovered


def main() -> None:
    args = parse_args()
    sources = [source.strip().lower() for source in args.sources.split(",") if source.strip()]
    file_types = {file_type.strip().lower() for file_type in args.file_types.split(",") if file_type.strip()}
    args.output_root.mkdir(parents=True, exist_ok=True)
    priority_material_families = parse_csv(args.priority_material_families)
    target_material_family_ratios = parse_ratio_csv(args.target_material_family_ratios)
    frame = load_filtered_rows(args.objaverse_root, sources, file_types)
    excluded_source_uids = load_excluded_source_uids(args.exclude_manifest)
    if excluded_source_uids and not frame.empty:
        frame = frame[~frame["fileIdentifier"].astype(str).isin(excluded_source_uids)].copy()
    selected = select_rows(
        frame,
        args.target,
        priority_material_families=priority_material_families,
        target_material_family_ratios=target_material_family_ratios,
        source_priority=parse_csv(args.source_priority),
    )
    selected_parquet = args.output_root / "objaverse_cached_selected.parquet"
    selected.to_parquet(selected_parquet)

    selection_preview = {
        "generated_at_utc": utc_now(),
        "sources": sources,
        "target": args.target,
        "selected_count": int(len(selected)),
        "selection_policy": {
            "priority_material_families": priority_material_families,
            "target_material_family_ratios": target_material_family_ratios,
            "source_priority": parse_csv(args.source_priority),
        },
        "exclude_manifest": str(args.exclude_manifest.resolve()) if args.exclude_manifest else "",
        "excluded_existing_source_uid_count": len(excluded_source_uids),
        "file_types": sorted(file_types),
        "source_counts": selected["source"].value_counts(dropna=False).to_dict() if not selected.empty else {},
        "license_counts": selected["license"].value_counts(dropna=False).to_dict() if not selected.empty else {},
        "file_type_counts": selected["fileType"].value_counts(dropna=False).to_dict() if not selected.empty else {},
        "material_counts": selected["highlight_material_class"].value_counts(dropna=False).to_dict() if not selected.empty else {},
        "selected_parquet": str(selected_parquet),
    }
    write_json(args.output_root / "objaverse_cached_selection_manifest.json", selection_preview)

    download_results: dict[str, str] = {}
    download_error = ""
    if args.download and not selected.empty:
        import objaverse.xl as oxl

        try:
            download_kwargs: dict[str, Any] = {}
            if "github" in set(selected["source"].astype(str).str.lower()):
                download_kwargs["save_repo_format"] = args.github_save_repo_format
            download_results = oxl.download_objects(
                selected[["fileIdentifier", "source", "license", "fileType", "sha256", "metadata"]],
                download_dir=str(args.objaverse_root.parent),
                processes=args.processes,
                **download_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            download_error = f"{type(exc).__name__}: {exc}"

    recovered_results = reconcile_cached_downloads(args.objaverse_root.parent, selected)
    if recovered_results:
        download_results = {**recovered_results, **download_results}

    rows: list[dict[str, Any]] = []
    for _index, row in selected.iterrows():
        identifier = str(row["fileIdentifier"])
        local_path = download_results.get(identifier, "")
        rows.append(
            {
                "source_name": "Objaverse-XL_cached_strict_increment",
                "object_id": f"objaverse_{stable_id(identifier)}",
                "source_uid": identifier,
                "source": str(row["source"]),
                "license_bucket": str(row["license"]),
                "format": str(row["fileType"]),
                "highlight_material_class": str(row["highlight_material_class"]),
                "material_class_source": str(row["material_class_source"]),
                "download_status": "downloaded" if local_path else ("metadata_selected" if not args.download else "download_missing"),
                "local_path": local_path,
            }
        )

    manifest_csv = args.output_root / "objaverse_cached_increment_manifest.csv"
    manifest_json = args.output_root / "objaverse_cached_increment_manifest.json"
    payload = {
        "generated_at_utc": utc_now(),
        "sources": sources,
        "target": args.target,
        "selection_policy": {
            "priority_material_families": priority_material_families,
            "target_material_family_ratios": target_material_family_ratios,
            "source_priority": parse_csv(args.source_priority),
        },
        "exclude_manifest": str(args.exclude_manifest.resolve()) if args.exclude_manifest else "",
        "excluded_existing_source_uid_count": len(excluded_source_uids),
        "selected_count": len(rows),
        "download_requested": args.download,
        "download_error": download_error,
        "downloaded_count": sum(row["download_status"] == "downloaded" for row in rows),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "license_counts": dict(Counter(row["license_bucket"] for row in rows)),
        "material_counts": dict(Counter(row["highlight_material_class"] for row in rows)),
        "selected_parquet": str(selected_parquet),
        "records": rows,
    }
    write_csv(manifest_csv, rows)
    write_json(manifest_json, payload)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "records"},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
