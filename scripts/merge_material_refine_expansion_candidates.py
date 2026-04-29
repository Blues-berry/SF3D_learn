#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPANSION_ROOT = REPO_ROOT / "output/material_refine_expansion_candidates"
DEFAULT_OUTPUT = DEFAULT_EXPANSION_ROOT / "merged_expansion_candidate_manifest.json"
DEFAULT_EXCLUDE_DIRS = {
    "polyhaven_hdri_bank",
    "polyhaven_material_bank",
}
MODEL_SUFFIXES = {".glb", ".gltf", ".fbx", ".obj"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Merge staged expansion source manifests into one object-candidate manifest.",
    )
    parser.add_argument("--expansion-root", type=Path, default=DEFAULT_EXPANSION_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exclude-dirs",
        type=str,
        default=",".join(sorted(DEFAULT_EXCLUDE_DIRS)),
        help="Comma-separated subdirectories ignored during merge.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def source_records(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def record_text(record: dict[str, Any]) -> str:
    keys = (
        "source_name",
        "source_name_detail",
        "source_dataset",
        "generator_id",
        "pool_name",
        "notes",
        "source_model_path",
        "canonical_glb_path",
        "canonical_mesh_path",
        "raw_asset_path",
    )
    return " ".join(str(record.get(key) or "") for key in keys).lower()


def asset_candidates(record: dict[str, Any]) -> list[str]:
    paths = []
    for key in ("source_model_path", "canonical_glb_path", "canonical_mesh_path", "raw_asset_path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    return paths


def resolved_existing_model_path(record: dict[str, Any]) -> str:
    for value in asset_candidates(record):
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists() and path.is_file() and path.suffix.lower() in MODEL_SUFFIXES:
            try:
                return str(path.resolve())
            except OSError:
                return str(path)
    return ""


def logical_path(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return path


def storage_tier(path: str) -> str:
    if not path:
        return "unknown"
    resolved = str(Path(path).resolve())
    if resolved.startswith("/4T/"):
        return "hdd_archive"
    if resolved.startswith(str((REPO_ROOT / "dataoutput").resolve())):
        return "ssd_active"
    if resolved.startswith(str(REPO_ROOT.resolve())):
        return "ssd_project_or_output_symlink"
    return "external_or_unknown"


def material_family(record: dict[str, Any]) -> str:
    return str(
        record.get("material_family")
        or record.get("highlight_material_class")
        or "unknown_pending_second_pass"
    )


def is_object_candidate(record: dict[str, Any], manifest_dir: str) -> bool:
    if manifest_dir in DEFAULT_EXCLUDE_DIRS:
        return False
    text = record_text(record)
    if "polyhaven" in text:
        return False
    return bool(asset_candidates(record))


def dedup_key(record: dict[str, Any]) -> str:
    return str(record.get("object_id") or record.get("source_uid") or "")


def selection_score(record: dict[str, Any], manifest_dir: str) -> tuple[int, int, int, int, int]:
    physical = resolved_existing_model_path(record)
    return (
        int(bool(physical)),
        int(boolish(record.get("source_asset_available"))),
        int("local_available" in str(record.get("asset_access_status") or "")),
        int(manifest_dir not in DEFAULT_EXCLUDE_DIRS),
        int(material_family(record) != "unknown_pending_second_pass"),
    )


def normalize_record(record: dict[str, Any], manifest_dir: str, source_manifest: Path) -> dict[str, Any]:
    out = dict(record)
    physical = resolved_existing_model_path(record)
    out["material_family"] = material_family(record)
    out["raw_or_canonical_asset_available"] = bool(physical)
    out["path_resolved_ok"] = bool(physical)
    out["physical_path"] = physical
    out["logical_path"] = logical_path(physical)
    out["storage_tier"] = storage_tier(physical)
    out["source_manifest_dir"] = manifest_dir
    out["source_manifest_path"] = str(source_manifest.resolve())
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "split": dict(Counter(str(record.get("default_split") or "unknown") for record in records)),
        "paper_split": dict(Counter(str(record.get("paper_split") or "unknown") for record in records)),
        "raw_or_canonical_asset_available": {
            "true": sum(boolish(record.get("raw_or_canonical_asset_available")) for record in records),
            "false": sum(not boolish(record.get("raw_or_canonical_asset_available")) for record in records),
        },
        "with_prior": sum(boolish(record.get("has_material_prior")) for record in records),
        "without_prior": sum(not boolish(record.get("has_material_prior")) for record in records),
        "source_name": dict(Counter(str(record.get("source_name") or "unknown") for record in records)),
        "material_family": dict(Counter(str(record.get("material_family") or "unknown") for record in records)),
        "license_bucket": dict(Counter(str(record.get("license_bucket") or "unknown") for record in records)),
    }


def main() -> None:
    args = parse_args()
    excluded_dirs = {
        item.strip()
        for item in str(args.exclude_dirs).split(",")
        if item.strip()
    }
    manifests = sorted(args.expansion_root.glob("*/source_candidate_manifest.json"))
    merged: dict[str, dict[str, Any]] = {}
    source_paths: list[str] = []
    skipped_by_dir: Counter[str] = Counter()
    duplicates = 0

    for manifest_path in manifests:
        manifest_dir = manifest_path.parent.name
        if manifest_dir in excluded_dirs:
            skipped_by_dir[manifest_dir] += 1
            continue
        source_paths.append(str(manifest_path.resolve()))
        for record in source_records(manifest_path):
            if not is_object_candidate(record, manifest_dir):
                continue
            key = dedup_key(record)
            if not key:
                continue
            normalized = normalize_record(record, manifest_dir, manifest_path)
            current = merged.get(key)
            if current is None or selection_score(normalized, manifest_dir) > selection_score(current, str(current.get("source_manifest_dir") or "")):
                if current is not None:
                    duplicates += 1
                merged[key] = normalized
            else:
                duplicates += 1

    records = sorted(
        merged.values(),
        key=lambda record: (
            str(record.get("material_family") or "unknown"),
            str(record.get("source_name") or "unknown"),
            str(record.get("object_id") or ""),
        ),
    )
    payload = {
        "manifest_version": "trainV5_expansion_candidate_merged_v2",
        "generated_at_utc": utc_now(),
        "policy": "object_candidate_only_merge_excluding_polyhaven_auxiliary_sources",
        "source_paths": source_paths,
        "summary": summarize(records),
        "records": records,
    }
    write_json(args.output_manifest, payload)
    write_text(
        args.output_manifest.with_suffix(".md"),
        "\n".join(
            [
                "# Merged Expansion Candidate Manifest",
                "",
                f"- generated_at_utc: `{payload['generated_at_utc']}`",
                f"- records: `{len(records)}`",
                f"- source_manifest_count: `{len(source_paths)}`",
                f"- skipped_source_dirs: `{json.dumps(dict(skipped_by_dir), ensure_ascii=False)}`",
                f"- duplicate_suppressed: `{duplicates}`",
                f"- source_name: `{json.dumps(payload['summary']['source_name'], ensure_ascii=False)}`",
                f"- material_family: `{json.dumps(payload['summary']['material_family'], ensure_ascii=False)}`",
            ]
        ),
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "source_manifest_count": len(source_paths),
                "skipped_source_dirs": dict(skipped_by_dir),
                "duplicate_suppressed": duplicates,
                "output_manifest": str(args.output_manifest.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
