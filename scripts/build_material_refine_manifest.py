#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DOCS_ROOT = PROJECT_ROOT / "docs" / "neural_gaffer_dataset_audit"

DEFAULT_MINI_MANIFEST_JSON = DOCS_ROOT / "mini_v1_manifest.json"
DEFAULT_THREE_FUTURE_JSON = DOCS_ROOT / "pool_A_pilot_3dfuture_30.json"
DEFAULT_OBJAVERSE_CSV = DOCS_ROOT / "pool_A_pilot_objaverse_30.csv"
DEFAULT_OUTPUT_JSON = DOCS_ROOT / "material_refine_manifest_v1.json"
DEFAULT_OUTPUT_CSV = DOCS_ROOT / "material_refine_manifest_v1.csv"
DEFAULT_OUTPUT_MD = DOCS_ROOT / "material_refine_manifest_v1_summary.md"

MANIFEST_VERSION = "canonical_asset_record_v1"
SMOKE_SOURCE_COUNTS = {
    "ABO_locked_core": 8,
    "3D-FUTURE_candidate": 4,
    "Objaverse-XL_filtered_candidate": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the CanonicalAssetRecordV1 manifest for the material refinement data pipeline."
    )
    parser.add_argument("--mini-manifest-json", type=Path, default=DEFAULT_MINI_MANIFEST_JSON)
    parser.add_argument("--three-future-json", type=Path, default=DEFAULT_THREE_FUTURE_JSON)
    parser.add_argument("--objaverse-csv", type=Path, default=DEFAULT_OBJAVERSE_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise RuntimeError("cannot_write_empty_material_refine_manifest_csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def stable_hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def deterministic_split(object_id: str) -> str:
    bucket = int(stable_hash_key(object_id)[:8], 16) % 100
    if bucket < 5:
        return "val"
    if bucket < 10:
        return "test"
    return "train"


def path_exists(path_str: str) -> bool:
    return bool(path_str) and Path(path_str).exists()


def first_existing_path(*candidates: str) -> str:
    for candidate in candidates:
        if path_exists(candidate):
            return str(Path(candidate).resolve())
    return ""


def infer_prior_mode(notes: str, has_roughness: bool, has_metallic: bool) -> str:
    note_text = notes.lower()
    if "roughness=texture" in note_text or "metallic=texture" in note_text:
        return "uv_rm"
    if has_roughness or has_metallic:
        return "scalar_rm"
    return "none"


def source_rank(record: dict) -> tuple[int, str]:
    availability_rank = 0 if record["source_asset_available"] else 1
    return availability_rank, stable_hash_key(record["object_id"])


def infer_material_family_from_row(row: dict) -> str:
    text = " ".join(
        str(row.get(key, ""))
        for key in (
            "label",
            "highlight_priority_class",
            "semantic_stratum",
            "selection_reason",
            "category",
            "super_category",
            "material",
            "notes",
        )
    ).lower()
    if "mixed_thin_boundary" in text or "thin-boundary" in text or "thin boundary" in text:
        return "mixed_thin_boundary"
    if "glass_metal" in text or ("glass" in text and "metal" in text):
        return "glass_metal"
    if "ceramic" in text or "glazed" in text or "lacquer" in text:
        return "ceramic_glazed_lacquer"
    if "metal_dominant" in text or float(row.get("avg_gt_metallic_mean", 0.0) or 0.0) >= 0.45:
        return "metal_dominant"
    return "glossy_non_metal"


def infer_thin_boundary_flag_from_row(row: dict, *, material_family: str) -> bool:
    if material_family == "mixed_thin_boundary":
        return True
    text = " ".join(
        str(row.get(key, ""))
        for key in ("label", "category", "name", "notes")
    ).lower()
    return any(token in text for token in ("thin", "frame", "wire", "handle", "boundary"))


def csv_record(record: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in record.items():
        if isinstance(value, bool):
            out[key] = bool_str(value)
        elif value is None:
            out[key] = ""
        else:
            out[key] = str(value)
    return out


def build_abo_records(path: Path) -> list[dict]:
    payload = read_json(path)
    if not isinstance(payload, dict) or "records" not in payload:
        raise RuntimeError(f"unexpected_abo_manifest_shape:{path}")
    rows = payload["records"]
    if not isinstance(rows, list):
        raise RuntimeError(f"unexpected_abo_records_shape:{path}")

    records: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_model_path = str(row.get("source_model_path", ""))
        texture_root = str(row.get("texture_root", ""))
        notes = str(row.get("notes", ""))
        material_family = infer_material_family_from_row(row)
        prior_mode = infer_prior_mode(
            notes=notes,
            has_roughness=True,
            has_metallic=True,
        )
        record = {
            "record_version": MANIFEST_VERSION,
            "object_id": str(row["object_id"]),
            "source_uid": str(row.get("source_uid") or row["object_id"]),
            "source_name": "ABO_locked_core",
            "source_dataset": "mini_v1_abo_ecommerce",
            "generator_id": "abo_locked_core",
            "license_bucket": "cc_by_nc_4_0",
            "supervision_role": "paper_main",
            "source_model_path": source_model_path,
            "source_texture_root": texture_root,
            "source_format": str(row.get("format", "glb")),
            "source_asset_available": path_exists(source_model_path),
            "asset_access_status": "local_available" if path_exists(source_model_path) else "missing_source_path",
            "default_split": str(row.get("default_split", "train")),
            "has_material_prior": prior_mode != "none",
            "prior_mode": prior_mode,
            "supervision_tier": "strong",
            "material_slot_count": int(row.get("material_slot_count", 0) or 0),
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
            "paper_split": str(row.get("default_split", "train")),
            "material_family": material_family,
            "thin_boundary_flag": infer_thin_boundary_flag_from_row(row, material_family=material_family),
            "lighting_bank_id": "canonical_triplet_v1",
            "view_supervision_ready": False,
            "valid_view_count": 0,
            "include_in_smoke": False,
            "include_in_full": True,
            "processing_status": "unprepared",
            "smoke_source_bucket": "",
            "notes": notes,
        }
        records.append(record)
    return records


def build_three_future_records(path: Path) -> list[dict]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"unexpected_3dfuture_payload:{path}")

    records: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_model_path = first_existing_path(
            str(row.get("local_path", "")),
            str(row.get("source_model_path", "")),
        )
        texture_root = first_existing_path(str(row.get("texture_root", "")))
        audit_status = str(row.get("audit_status", "unknown"))
        has_roughness = str(row.get("has_roughness", "")).lower() == "true"
        has_metallic = str(row.get("has_metallic", "")).lower() == "true"
        material_family = infer_material_family_from_row(row)
        prior_mode = infer_prior_mode(
            notes=str(row.get("notes", "")),
            has_roughness=has_roughness,
            has_metallic=has_metallic,
        )
        supervision_tier = "strong" if audit_status == "A_ready" else "material_prior"
        record = {
            "record_version": MANIFEST_VERSION,
            "object_id": str(row["object_id"]),
            "source_uid": str(row.get("source_uid") or row["object_id"]),
            "source_name": "3D-FUTURE_candidate",
            "source_dataset": "pool_a_pilot_3dfuture",
            "generator_id": "3d_future_candidate",
            "license_bucket": str(row.get("license_bucket", "custom_tianchi_terms")),
            "supervision_role": "auxiliary_upgrade_queue",
            "source_model_path": source_model_path or str(row.get("source_model_path", "")),
            "source_texture_root": texture_root or str(row.get("texture_root", "")),
            "source_format": str(row.get("format", "obj")),
            "source_asset_available": bool(source_model_path),
            "asset_access_status": "local_available" if source_model_path else str(row.get("download_status", "missing")),
            "default_split": deterministic_split(str(row["object_id"])),
            "has_material_prior": prior_mode != "none",
            "prior_mode": prior_mode,
            "supervision_tier": supervision_tier,
            "material_slot_count": int(row.get("material_slot_count", 0) or 0),
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
            "paper_split": deterministic_split(str(row["object_id"])),
            "material_family": material_family,
            "thin_boundary_flag": infer_thin_boundary_flag_from_row(row, material_family=material_family),
            "lighting_bank_id": "canonical_triplet_v1",
            "view_supervision_ready": False,
            "valid_view_count": 0,
            "include_in_smoke": False,
            "include_in_full": True,
            "processing_status": "unprepared",
            "smoke_source_bucket": "",
            "notes": str(row.get("notes", "")),
        }
        records.append(record)
    return records


def build_objaverse_records(path: Path) -> list[dict]:
    rows = read_csv_rows(path)
    records: list[dict] = []
    for row in rows:
        source_model_path = first_existing_path(
            str(row.get("local_path", "")),
            str(row.get("source_model_path", "")),
        )
        material_family = infer_material_family_from_row(row)
        record = {
            "record_version": MANIFEST_VERSION,
            "object_id": str(row["object_id"]),
            "source_uid": str(row.get("source_uid") or row["object_id"]),
            "source_name": "Objaverse-XL_filtered_candidate",
            "source_dataset": "pool_a_pilot_objaverse",
            "generator_id": "objaverse_xl_filtered_candidate",
            "license_bucket": str(row.get("license_bucket", "mixed_per_object_license")),
            "supervision_role": "auxiliary_upgrade_queue",
            "source_model_path": source_model_path or str(row.get("source_model_path", "")),
            "source_texture_root": str(row.get("texture_root", "")),
            "source_format": str(row.get("format", "glb")),
            "source_asset_available": bool(source_model_path),
            "asset_access_status": "local_available" if source_model_path else str(row.get("download_status", "metadata_only")),
            "default_split": deterministic_split(str(row["object_id"])),
            "has_material_prior": False,
            "prior_mode": "none",
            "supervision_tier": "eval_only",
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
            "paper_split": deterministic_split(str(row["object_id"])),
            "material_family": material_family,
            "thin_boundary_flag": infer_thin_boundary_flag_from_row(row, material_family=material_family),
            "lighting_bank_id": "canonical_triplet_v1",
            "view_supervision_ready": False,
            "valid_view_count": 0,
            "include_in_smoke": False,
            "include_in_full": True,
            "processing_status": "unprepared",
            "smoke_source_bucket": "",
            "notes": str(row.get("notes", "")),
        }
        records.append(record)
    return records


def apply_smoke_membership(records: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["source_name"]].append(record)

    for source_name, count in SMOKE_SOURCE_COUNTS.items():
        candidates = sorted(grouped[source_name], key=source_rank)
        for record in candidates[:count]:
            record["include_in_smoke"] = True
            record["smoke_source_bucket"] = source_name


def write_summary(path: Path, payload: dict) -> None:
    counts = payload["counts"]
    source_counts = payload["source_counts"]
    split_counts = payload["split_counts"]
    smoke_ids = set(payload["splits"]["smoke"]["object_ids"])
    lines = [
        "# Material Refine Manifest v1",
        "",
        f"- manifest_version: {payload['manifest_version']}",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- output_json: {payload['output_files']['json']}",
        f"- output_csv: {payload['output_files']['csv']}",
        "",
        "## Record Counts",
        "",
        f"- total_records: {counts['total_records']}",
        f"- smoke_records: {counts['smoke_records']}",
        f"- full_records: {counts['full_records']}",
        f"- local_assets: {counts['local_assets']}",
        f"- blocked_assets: {counts['blocked_assets']}",
        "",
        "## Source Counts",
        "",
    ]
    for source_name, count in sorted(source_counts.items()):
        lines.append(f"- {source_name}: {count}")
    lines.extend(
        [
            "",
            "## Default Split Counts",
            "",
            f"- train: {split_counts['train']}",
            f"- val: {split_counts['val']}",
            f"- test: {split_counts['test']}",
            "",
            "## Smoke Selection",
            "",
            "| object_id | source_name | default_split | asset_access_status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in payload["records"]:
        if record["object_id"] not in smoke_ids:
            continue
        lines.append(
            "| "
            + f"{record['object_id']} | {record['source_name']} | {record['default_split']} | "
            + f"{record['asset_access_status']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- smoke keeps the fixed 8/4/4 source allocation even when some Objaverse rows remain metadata-only.",
            "- full keeps every current candidate row so GPU0 can prepare local assets and placeholder bundles through the same contract.",
            "- blocked assets stay explicit in the manifest via `source_asset_available = false` and `asset_access_status`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = []
    records.extend(build_abo_records(args.mini_manifest_json))
    records.extend(build_three_future_records(args.three_future_json))
    records.extend(build_objaverse_records(args.objaverse_csv))
    apply_smoke_membership(records)
    records.sort(key=lambda record: (record["source_name"], record["object_id"]))

    split_counts = Counter(record["default_split"] for record in records)
    source_counts = Counter(record["source_name"] for record in records)
    local_assets = sum(1 for record in records if record["source_asset_available"])
    smoke_ids = [record["object_id"] for record in records if record["include_in_smoke"]]
    full_ids = [record["object_id"] for record in records if record["include_in_full"]]

    payload = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_utc": utc_now(),
        "input_files": {
            "mini_manifest_json": str(args.mini_manifest_json),
            "three_future_json": str(args.three_future_json),
            "objaverse_csv": str(args.objaverse_csv),
        },
        "output_files": {
            "json": str(args.output_json),
            "csv": str(args.output_csv),
            "md": str(args.output_md),
        },
        "split_policy": {
            "smoke_fixed_source_counts": SMOKE_SOURCE_COUNTS,
            "full_policy": "include_every_current_candidate_row",
            "non_abo_default_split": "sha1_hash_bucket_90_5_5",
        },
        "counts": {
            "total_records": len(records),
            "smoke_records": len(smoke_ids),
            "full_records": len(full_ids),
            "local_assets": local_assets,
            "blocked_assets": len(records) - local_assets,
        },
        "source_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "splits": {
            "smoke": {
                "object_ids": smoke_ids,
                "record_count": len(smoke_ids),
            },
            "full": {
                "object_ids": full_ids,
                "record_count": len(full_ids),
            },
        },
        "records": records,
    }
    write_json(args.output_json, payload)
    write_csv(args.output_csv, [csv_record(record) for record in records])
    write_summary(args.output_md, payload)

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
