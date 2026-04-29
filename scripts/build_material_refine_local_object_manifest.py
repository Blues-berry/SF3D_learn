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
DEFAULT_SOURCE_ROOT = Path("/4T/CXY/Neural_Gaffer/external_data/neural_gaffer_original/external_sources/downloads")
MANIFEST_VERSION = "canonical_asset_record_v1_local_object_increment"
MODEL_EXTENSIONS = {".glb", ".gltf", ".obj", ".fbx"}

DEFAULT_SOURCES = {
    "Local_GSO_highlight_increment": {
        "root": DEFAULT_SOURCE_ROOT / "gso_selected",
        "license_bucket": "cc_by_4_0_pending_reconcile",
        "generator_id": "local_gso_highlight_increment",
        "source_dataset": "google_scanned_objects_local_selected",
        "max_records": 800,
    },
    "Local_Kenney_CC0_increment": {
        "root": DEFAULT_SOURCE_ROOT / "kenney_selected",
        "license_bucket": "Creative Commons Zero v1.0 Universal",
        "generator_id": "local_kenney_cc0_increment",
        "source_dataset": "kenney_selected_local",
        "max_records": 1200,
    },
    "Local_Quaternius_CC0_increment": {
        "root": DEFAULT_SOURCE_ROOT / "quaternius_selected",
        "license_bucket": "Creative Commons Zero v1.0 Universal",
        "generator_id": "local_quaternius_cc0_increment",
        "source_dataset": "quaternius_selected_local",
        "max_records": 800,
    },
    "Local_Smithsonian_selected_increment": {
        "root": DEFAULT_SOURCE_ROOT / "smithsonian_selected",
        "license_bucket": "smithsonian_open_access_pending_reconcile",
        "generator_id": "local_smithsonian_selected_increment",
        "source_dataset": "smithsonian_selected_local",
        "max_records": 800,
    },
    "Local_OmniObject3D_increment": {
        "root": DEFAULT_SOURCE_ROOT / "omni_selected",
        "license_bucket": "omniobject3d_license_pending_reconcile",
        "generator_id": "local_omniobject3d_increment",
        "source_dataset": "omniobject3d_local_selected",
        "max_records": 800,
    },
    "Khronos_highlight_reference_samples": {
        "root": DEFAULT_SOURCE_ROOT / "highlight_test_samples" / "khronos",
        "license_bucket": "khronos_sample_license_pending_reconcile",
        "generator_id": "khronos_highlight_reference_samples",
        "source_dataset": "khronos_gltf_sample_models_local",
        "max_records": 128,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build a CanonicalAssetRecordV1 manifest from local non-3D-FUTURE object sources.",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output" / "material_refine_aux_downloads" / "local_object_sources_canonical")
    parser.add_argument("--max-total-records", type=int, default=3600)
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


def material_family_from_text(text: str) -> tuple[str, str]:
    lowered = text.lower().replace("_", " ").replace("-", " ")
    checks = {
        "metal_dominant": (
            "metal", "metallic", "chrome", "steel", "aluminum", "aluminium", "brass", "copper",
            "iron", "gold", "silver", "watch", "clock", "gear", "bolt", "screw", "cannon", "tool",
            "cable", "wire",
        ),
        "ceramic_glazed_lacquer": (
            "ceramic", "porcelain", "pottery", "stoneware", "ramekin", "teapot", "bowl", "plate",
            "dish", "cup", "mug", "vase", "tile", "glazed", "lacquer", "marble",
        ),
        "glass_metal": (
            "glass", "mirror", "transparent", "transmission", "crystal", "bottle", "jar", "lamp",
            "lantern", "bulb", "candle", "hurricane", "window",
        ),
        "mixed_thin_boundary": (
            "frame", "handle", "chair", "table", "fence", "rail", "rack", "basket", "cage",
            "stand", "ship", "boat", "wheel", "bike", "car", "armchair", "forelimb", "skeleton",
        ),
        "glossy_non_metal": (
            "plastic", "toy", "polished", "painted", "leather", "resin", "acrylic", "silk",
            "synthetic", "glossy", "varnish",
        ),
    }
    scores = {
        family: sum(1 for token in tokens if token in lowered)
        for family, tokens in checks.items()
    }
    family, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return "unknown_pending_second_pass", "local_path_keyword_pending_material_probe"
    return family, "local_path_keyword"


def discover_models(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in MODEL_EXTENSIONS
        and "3d_future" not in {part.lower() for part in path.parts}
    )


def build_record(path: Path, *, source_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    relative = path.relative_to(spec["root"])
    object_id = f"{spec['generator_id']}_{stable_hash(str(relative))[:24]}"
    family, family_source = material_family_from_text(" ".join(relative.parts))
    return {
        "record_version": MANIFEST_VERSION,
        "object_id": object_id,
        "source_uid": str(relative),
        "source_name": source_name,
        "source_name_detail": source_name,
        "source_provider": source_name,
        "source_dataset": spec["source_dataset"],
        "source_dataset_detail": spec["source_dataset"],
        "pool_name": "pool_A_auxiliary_upgrade_queue",
        "generator_id": spec["generator_id"],
        "generator_id_detail": spec["generator_id"],
        "license_bucket": spec["license_bucket"],
        "source_model_path": str(path),
        "source_texture_root": str(path.parent),
        "source_format": path.suffix.lstrip(".").lower(),
        "source_asset_available": True,
        "asset_access_status": "local_available",
        "default_split": deterministic_split(object_id),
        "has_material_prior": False,
        "prior_mode": "none",
        "supervision_tier": "material_prior_candidate_needs_second_pass",
        "highlight_material_class": family,
        "material_class_source": family_source,
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
        "notes": "local_non_3dfuture_source; requires import/material/target second pass before paper promotion",
    }


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


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for source_name, spec in DEFAULT_SOURCES.items():
        models = discover_models(Path(spec["root"]))[: int(spec["max_records"])]
        for path in models:
            records.append(build_record(path, source_name=source_name, spec=spec))
            source_counts[source_name] += 1
    records.sort(key=lambda record: (record["highlight_material_class"], record["source_name"], record["object_id"]))
    if args.max_total_records > 0:
        records = records[: args.max_total_records]

    output_json = args.output_root / "material_refine_manifest_local_object_increment.json"
    output_csv = args.output_root / "material_refine_manifest_local_object_increment.csv"
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_utc": utc_now(),
        "source_policy": "non_3dfuture_local_sources_only_auxiliary_until_second_pass",
        "output_files": {"json": str(output_json), "csv": str(output_csv)},
        "counts": {
            "total_records": len(records),
            "local_assets": len(records),
            "blocked_assets": 0,
        },
        "source_counts": dict(Counter(record["source_name"] for record in records)),
        "configured_source_counts": dict(source_counts),
        "highlight_material_counts": dict(Counter(record["highlight_material_class"] for record in records)),
        "split_counts": dict(Counter(record["default_split"] for record in records)),
        "records": records,
    }
    write_json(output_json, payload)
    write_csv(output_csv, records)
    print(f"wrote {output_json}")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
