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
DEFAULT_ABO_JSON = DOCS_ROOT / "mini_v1_manifest.json"
DEFAULT_THREE_FUTURE_ROOT = Path(
    "/4T/CXY/Neural_Gaffer/external_data/neural_gaffer_original/external_sources/downloads/3d_future/model_extracted/3D-FUTURE-model"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "highlight_pool_a_8k"
MANIFEST_VERSION = "canonical_asset_record_v1_highlight_8k"


MATERIAL_QUOTAS = {
    "metal_dominant": 0.30,
    "ceramic_glazed_lacquer": 0.20,
    "glass_metal": 0.15,
    "mixed_thin_boundary": 0.20,
    "glossy_non_metal": 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an 8k+ local Pool-A manifest from ABO plus locally staged 3D-FUTURE assets."
    )
    parser.add_argument("--abo-json", type=Path, default=DEFAULT_ABO_JSON)
    parser.add_argument("--three-future-root", type=Path, default=DEFAULT_THREE_FUTURE_ROOT)
    parser.add_argument("--target-count", type=int, default=8200)
    parser.add_argument("--abo-count", type=int, default=500)
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


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def path_exists(path: str | Path) -> bool:
    return bool(path) and Path(path).exists()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("cannot_write_empty_csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: bool_str(value) if isinstance(value, bool) else "" if value is None else str(value)
                    for key, value in row.items()
                }
            )


def infer_prior_mode(notes: str, has_roughness: bool, has_metallic: bool) -> str:
    lowered = notes.lower()
    if "roughness=texture" in lowered or "metallic=texture" in lowered:
        return "uv_rm"
    if has_roughness or has_metallic:
        return "scalar_rm"
    return "none"


def classify_three_future(row: dict) -> tuple[str, str]:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("super-category", "category", "style", "theme", "material")
    ).lower()
    if any(token in text for token in ("lighting", "lamp", "chandelier", "pendant", "glass", "mirror", "transparent")):
        return "glass_metal", "metadata_keyword:lighting_or_glass"
    if any(token in text for token in ("metal", "iron", "steel", "chrome", "aluminum", "stainless", "gold", "foil", "plating", "copper", "brass")):
        return "metal_dominant", "metadata_keyword:metal_or_metallic_finish"
    if any(token in text for token in ("ceramic", "porcelain", "marble", "stone", "glaze", "lacquer", "varnish")):
        return "ceramic_glazed_lacquer", "metadata_keyword:ceramic_stone_or_lacquer"
    if any(token in text for token in ("smooth", "gloss", "polished", "paint", "plastic", "leather", "synthetic", "acrylic", "resin", "pvc", "wood", "board")):
        return "glossy_non_metal", "metadata_keyword:smooth_glossy_nonmetal"
    if any(token in text for token in ("frame", "border", "handle", "shelf", "cabinet", "chair", "stool", "table", "desk", "door", "bed", "sofa")):
        return "mixed_thin_boundary", "metadata_keyword:part_boundary_or_furniture"
    return "unclassified", "metadata_keyword:none"


def build_abo_records(path: Path, limit: int) -> list[dict]:
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise RuntimeError(f"unexpected_abo_manifest:{path}")
    records = []
    for row in payload["records"][:limit]:
        source_model_path = str(row.get("source_model_path", ""))
        notes = str(row.get("notes", ""))
        prior_mode = infer_prior_mode(notes, has_roughness=True, has_metallic=True)
        object_id = str(row["object_id"])
        records.append(
            {
                "record_version": MANIFEST_VERSION,
                "object_id": object_id,
                "source_uid": str(row.get("source_uid") or object_id),
                "source_name": "ABO_locked_core",
                "source_dataset": "mini_v1_abo_ecommerce",
                "pool_name": "pool_A_direct_object_supervision",
                "generator_id": "abo_locked_core",
                "license_bucket": "cc_by_nc_4_0_pending_reconcile",
                "source_model_path": source_model_path,
                "source_texture_root": str(row.get("texture_root", "")),
                "source_format": str(row.get("format", "glb")),
                "source_asset_available": path_exists(source_model_path),
                "asset_access_status": "local_available" if path_exists(source_model_path) else "missing_source_path",
                "default_split": str(row.get("default_split") or deterministic_split(object_id)),
                "has_material_prior": prior_mode != "none",
                "prior_mode": prior_mode,
                "supervision_tier": "strong_A_ready_locked",
                "highlight_material_class": "pending_abo_semantic_classification",
                "material_class_source": "abo_core_pending_metadata_reconcile",
                "material_quota_target_fraction": "",
                "material_quota_deficit": "",
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
                "include_in_smoke": False,
                "include_in_full": True,
                "processing_status": "unprepared",
                "smoke_source_bucket": "",
                "notes": notes,
            }
        )
    return records


def three_future_candidate(row: dict, root: Path) -> dict | None:
    model_id = str(row.get("model_id") or "")
    if not model_id:
        return None
    model_dir = root / model_id
    obj_path = model_dir / "raw_model.obj"
    if not obj_path.exists():
        return None
    texture_png = model_dir / "texture.png"
    material_class, class_reason = classify_three_future(row)
    object_id = f"3dfuture_{model_id}"
    return {
        "source_row": row,
        "object_id": object_id,
        "source_uid": model_id,
        "material_class": material_class,
        "class_reason": class_reason,
        "source_model_path": str(obj_path.resolve()),
        "source_texture_root": str(texture_png.resolve() if texture_png.exists() else model_dir.resolve()),
        "sort_key": stable_hash(
            "|".join(
                [
                    material_class,
                    str(row.get("super-category") or ""),
                    str(row.get("category") or ""),
                    model_id,
                ]
            )
        ),
    }


def select_three_future(candidates: list[dict], target_three_future_count: int, target_total: int) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["material_class"]].append(candidate)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["sort_key"])

    quota_targets = {
        material_class: int(round(target_total * fraction))
        for material_class, fraction in MATERIAL_QUOTAS.items()
    }
    selected: list[dict] = []
    deficits: dict[str, int] = {}
    selected_counts: Counter[str] = Counter()
    selected_ids: set[str] = set()

    for material_class, desired_count in quota_targets.items():
        available = grouped.get(material_class, [])
        take = min(desired_count, len(available), target_three_future_count - len(selected))
        for item in available[:take]:
            selected.append(item)
            selected_ids.add(item["object_id"])
            selected_counts[material_class] += 1
        deficits[material_class] = max(0, desired_count - take)

    if len(selected) < target_three_future_count:
        leftovers = [
            item
            for bucket in grouped.values()
            for item in bucket
            if item["object_id"] not in selected_ids
        ]
        leftovers.sort(key=lambda item: (item["material_class"] == "unclassified", item["sort_key"]))
        for item in leftovers[: target_three_future_count - len(selected)]:
            selected.append(item)
            selected_ids.add(item["object_id"])
            selected_counts[item["material_class"]] += 1

    selected.sort(key=lambda item: (item["material_class"], item["sort_key"]))
    diagnostics = {
        "quota_targets": quota_targets,
        "available_by_class": dict(Counter(item["material_class"] for item in candidates)),
        "selected_by_class": dict(selected_counts),
        "quota_deficits_before_fill": deficits,
        "target_three_future_count": target_three_future_count,
    }
    return selected, diagnostics


def build_three_future_records(root: Path, target_three_future_count: int, target_total: int) -> tuple[list[dict], dict]:
    info_path = root / "model_info.json"
    rows = read_json(info_path)
    if not isinstance(rows, list):
        raise RuntimeError(f"unexpected_3dfuture_model_info:{info_path}")
    candidates = [
        candidate
        for row in rows
        if (candidate := three_future_candidate(row, root)) is not None
    ]
    selected, diagnostics = select_three_future(candidates, target_three_future_count, target_total)
    records = []
    for item in selected:
        row = item["source_row"]
        material_class = item["material_class"]
        records.append(
            {
                "record_version": MANIFEST_VERSION,
                "object_id": item["object_id"],
                "source_uid": item["source_uid"],
                "source_name": "3D-FUTURE_highlight_local_8k",
                "source_dataset": "3D-FUTURE-model-local",
                "pool_name": "pool_A_direct_object_supervision",
                "generator_id": "3d_future_highlight_local_8k",
                "license_bucket": "custom_tianchi_research_noncommercial_no_redistribution",
                "source_model_path": item["source_model_path"],
                "source_texture_root": item["source_texture_root"],
                "source_format": "obj",
                "source_asset_available": True,
                "asset_access_status": "local_available",
                "default_split": deterministic_split(item["object_id"]),
                "has_material_prior": False,
                "prior_mode": "none",
                "supervision_tier": "material_prior_candidate_needs_second_pass",
                "highlight_material_class": material_class,
                "material_class_source": item["class_reason"],
                "material_quota_target_fraction": MATERIAL_QUOTAS.get(material_class, ""),
                "material_quota_deficit": "",
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
                "notes": (
                    "selected_for_highlight_pool_a_8k; "
                    f"super_category={row.get('super-category')}; "
                    f"category={row.get('category')}; style={row.get('style')}; "
                    f"theme={row.get('theme')}; material={row.get('material')}"
                ),
            }
        )
    return records, diagnostics


def write_summary(path: Path, payload: dict) -> None:
    counts = payload["counts"]
    lines = [
        "# Highlight Pool-A 8k Manifest",
        "",
        f"- manifest_version: {payload['manifest_version']}",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- target_count: {payload['target_count']}",
        f"- total_records: {counts['total_records']}",
        f"- local_assets: {counts['local_assets']}",
        f"- output_json: `{payload['output_files']['json']}`",
        f"- output_csv: `{payload['output_files']['csv']}`",
        "",
        "## Source Counts",
        "",
    ]
    for source, count in sorted(payload["source_counts"].items()):
        lines.append(f"- {source}: {count}")
    lines.extend(["", "## Default Split Counts", ""])
    for split, count in sorted(payload["split_counts"].items()):
        lines.append(f"- {split}: {count}")
    lines.extend(["", "## Highlight Material Counts", ""])
    for material_class, count in sorted(payload["highlight_material_counts"].items()):
        lines.append(f"- {material_class}: {count}")
    lines.extend(["", "## 3D-FUTURE Quota Diagnostics", ""])
    diagnostics = payload["three_future_diagnostics"]
    for material_class, target in sorted(diagnostics["quota_targets"].items()):
        available = diagnostics["available_by_class"].get(material_class, 0)
        selected = diagnostics["selected_by_class"].get(material_class, 0)
        deficit = diagnostics["quota_deficits_before_fill"].get(material_class, 0)
        lines.append(
            f"- {material_class}: target={target}, available={available}, selected={selected}, initial_deficit={deficit}"
        )
    lines.extend(
        [
            "",
            "## Operating Notes",
            "",
            "- This manifest is intentionally local-first so GPU rendering can run while remote Objaverse-XL/HDRI/material pools download in parallel.",
            "- 3D-FUTURE remains in a separate non-commercial/no-redistribution license bucket.",
            "- ABO is kept as the locked core but marked for license reconciliation because local docs and the current official page differ.",
            "- Ceramic/glazed coverage is the main local deficit; fill it from Objaverse-XL strict-filtered objects and material-prior augmentation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.target_count < args.abo_count:
        raise RuntimeError("target_count_must_be_at_least_abo_count")
    output_json = args.output_json or args.output_root / "material_refine_manifest_pool_a_8k.json"
    output_csv = args.output_csv or args.output_root / "material_refine_manifest_pool_a_8k.csv"
    output_md = args.output_md or args.output_root / "material_refine_manifest_pool_a_8k_summary.md"

    abo_records = build_abo_records(args.abo_json, args.abo_count)
    three_future_target = args.target_count - len(abo_records)
    three_future_records, diagnostics = build_three_future_records(
        args.three_future_root,
        target_three_future_count=three_future_target,
        target_total=args.target_count,
    )
    records = [*three_future_records, *abo_records]
    records.sort(key=lambda record: (record["source_name"], record["object_id"]))

    payload = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_utc": utc_now(),
        "target_count": args.target_count,
        "input_files": {
            "abo_json": str(args.abo_json),
            "three_future_root": str(args.three_future_root),
        },
        "output_files": {
            "json": str(output_json),
            "csv": str(output_csv),
            "md": str(output_md),
        },
        "material_quotas": MATERIAL_QUOTAS,
        "split_policy": {
            "object_level_split": "deterministic_sha1_80_train_10_val_10_test",
            "license_bucket_isolation": True,
        },
        "counts": {
            "total_records": len(records),
            "local_assets": sum(1 for record in records if record["source_asset_available"]),
            "blocked_assets": sum(1 for record in records if not record["source_asset_available"]),
        },
        "source_counts": dict(Counter(record["source_name"] for record in records)),
        "split_counts": dict(Counter(record["default_split"] for record in records)),
        "highlight_material_counts": dict(Counter(record["highlight_material_class"] for record in records)),
        "three_future_diagnostics": diagnostics,
        "records": records,
    }
    write_json(output_json, payload)
    write_csv(output_csv, records)
    write_summary(output_md, payload)
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")
    print(f"wrote {output_md}")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
