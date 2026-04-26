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


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DEFAULT_INPUT_JSON = PROJECT_ROOT / "output" / "highlight_pool_a_8k" / "aux_sources" / "objaverse_xl_strict_filtered_manifest.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "highlight_pool_a_8k" / "objaverse_increment"
MANIFEST_VERSION = "canonical_asset_record_v1_objaverse_increment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert downloaded Objaverse-XL strict-filtered rows into CanonicalAssetRecordV1 increment manifest."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def deterministic_split(object_id: str) -> str:
    bucket = int(stable_hash(object_id)[:8], 16) % 100
    if bucket < 10:
        return "val"
    if bucket < 20:
        return "test"
    return "train"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["object_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_record(row: dict[str, Any]) -> dict[str, Any] | None:
    local_path = str(row.get("local_path") or "")
    if not local_path or not Path(local_path).exists():
        return None
    object_id = str(row.get("object_id") or f"objaverse_{row.get('source_uid')}")
    source_format = str(row.get("format") or Path(local_path).suffix.lstrip(".") or "unknown").lower()
    provider = str(row.get("source") or row.get("source_provider") or "mixed").strip().lower() or "mixed"
    provider_safe = "".join(character if character.isalnum() else "_" for character in provider).strip("_") or "mixed"
    return {
        "record_version": MANIFEST_VERSION,
        "object_id": object_id,
        "source_uid": str(row.get("source_uid") or object_id),
        "source_name": "Objaverse-XL_strict_filtered_increment",
        "source_name_detail": f"Objaverse-XL_{provider_safe}_strict_filtered_increment",
        "source_provider": provider,
        "source_dataset": "objaverse_xl_strict_filtered",
        "source_dataset_detail": f"objaverse_xl_{provider_safe}_strict_filtered",
        "pool_name": "pool_A_direct_object_supervision",
        "generator_id": "objaverse_xl_strict_filtered_increment",
        "generator_id_detail": f"objaverse_xl_{provider_safe}_strict_filtered_increment",
        "license_bucket": str(row.get("license_bucket") or "unknown_per_object"),
        "source_model_path": local_path,
        "source_texture_root": str(Path(local_path).parent),
        "source_format": source_format,
        "source_asset_available": True,
        "asset_access_status": "local_available",
        "default_split": deterministic_split(object_id),
        "has_material_prior": False,
        "prior_mode": "none",
        "supervision_tier": "material_prior_candidate_needs_second_pass",
        "highlight_material_class": str(row.get("highlight_material_class") or "unknown"),
        "material_class_source": str(row.get("material_class_source") or "objaverse_metadata"),
        "material_slot_count": 0,
        "canonical_mesh_path": "",
        "canonical_glb_path": "",
        "uv_albedo_path": "",
        "uv_normal_path": "",
        "uv_prior_roughness_path": "",
        "uv_prior_metallic_path": "",
        "scalar_prior_roughness": "",
        "scalar_prior_metallic": "",
        "uv_target_roughness_path": "",
        "uv_target_metallic_path": "",
        "uv_target_confidence_path": "",
        "canonical_views_json": "",
        "canonical_buffer_root": "",
        "include_in_smoke": False,
        "include_in_full": True,
        "processing_status": "unprepared",
        "smoke_source_bucket": "",
        "notes": f"downloaded_objaverse_xl_strict_filtered_increment; provider={provider}",
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Objaverse-XL Increment Manifest",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- input_json: `{payload['input_json']}`",
        f"- total_downloaded_records: {payload['counts']['total_records']}",
        f"- output_json: `{payload['output_files']['json']}`",
        "",
        "## Split Counts",
        "",
    ]
    for split, count in sorted(payload["split_counts"].items()):
        lines.append(f"- {split}: {count}")
    lines.extend(["", "## Highlight Material Counts", ""])
    for material_class, count in sorted(payload["highlight_material_counts"].items()):
        lines.append(f"- {material_class}: {count}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.output_root / "material_refine_manifest_objaverse_increment.json"
    output_csv = args.output_csv or args.output_root / "material_refine_manifest_objaverse_increment.csv"
    output_md = args.output_md or args.output_root / "material_refine_manifest_objaverse_increment_summary.md"
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    records = [
        record
        for row in payload.get("records", [])
        if (record := build_record(row)) is not None
    ]
    records.sort(key=lambda record: (record["highlight_material_class"], record["object_id"]))
    out_payload = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_utc": utc_now(),
        "input_json": str(args.input_json),
        "output_files": {
            "json": str(output_json),
            "csv": str(output_csv),
            "md": str(output_md),
        },
        "counts": {
            "total_records": len(records),
            "local_assets": len(records),
            "blocked_assets": 0,
        },
        "split_counts": dict(Counter(record["default_split"] for record in records)),
        "highlight_material_counts": dict(Counter(record["highlight_material_class"] for record in records)),
        "records": records,
    }
    write_json(output_json, out_payload)
    write_csv(output_csv, records)
    write_summary(output_md, out_payload)
    print(f"wrote {output_json}")
    print(json.dumps(out_payload["counts"], indent=2))


if __name__ == "__main__":
    main()
