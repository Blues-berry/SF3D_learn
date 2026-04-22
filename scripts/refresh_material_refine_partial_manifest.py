#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.manifest_quality import enrich_record_with_quality_fields

DEFAULT_INPUT_MANIFEST = REPO_ROOT / "docs" / "neural_gaffer_dataset_audit" / "material_refine_manifest_v1.json"
DEFAULT_CANONICAL_BUNDLE_ROOT = REPO_ROOT / "output" / "material_refine" / "prepared" / "full" / "canonical_bundle"
DEFAULT_OUTPUT_MANIFEST = REPO_ROOT / "output" / "material_refine" / "prepared" / "full" / "canonical_manifest_partial.json"
DEFAULT_VIEW_NAMES = ("front_studio", "three_quarter_indoor", "side_neon")
PREPARED_RECORD_FILENAME = "prepared_record.json"
REQUIRED_VIEW_BUFFER_FILES = (
    "rgba.png",
    "mask.png",
    "depth.png",
    "normal.png",
    "position.png",
    "uv.png",
    "visibility.png",
    "roughness.png",
    "metallic.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh a partial CanonicalAssetRecordV1 manifest from completed canonical bundles."
    )
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--canonical-bundle-root", type=Path, default=DEFAULT_CANONICAL_BUNDLE_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--split", choices=("smoke", "full"), default="full")
    return parser.parse_args()


def read_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "records" not in payload:
        raise RuntimeError(f"unexpected_manifest_payload:{path}")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def select_records(payload: dict, split: str) -> list[dict]:
    selected = []
    for record in payload["records"]:
        include_flag = record.get("include_in_smoke") if split == "smoke" else record.get("include_in_full", True)
        if not include_flag:
            continue
        if not record.get("source_asset_available", False):
            continue
        selected.append(dict(record))
    return selected


def load_bundle_views(bundle_dir: Path) -> list[dict]:
    views_json = bundle_dir / "views.json"
    if not views_json.exists():
        return [{"name": view_name} for view_name in DEFAULT_VIEW_NAMES]
    try:
        payload = json.loads(views_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [{"name": view_name} for view_name in DEFAULT_VIEW_NAMES]
    views = payload.get("views") if isinstance(payload, dict) else payload
    if not isinstance(views, list):
        return [{"name": view_name} for view_name in DEFAULT_VIEW_NAMES]
    normalized = []
    for view in views:
        if not isinstance(view, dict):
            continue
        name = str(view.get("name") or "")
        if name:
            normalized.append({"name": name, **view})
    return normalized or [{"name": view_name} for view_name in DEFAULT_VIEW_NAMES]


def expected_bundle_paths(bundle_dir: Path) -> dict[str, Path]:
    uv_dir = bundle_dir / "uv"
    buffer_root = bundle_dir / "buffers"
    paths = {
        "canonical_bundle_root": bundle_dir,
        "canonical_views_json": bundle_dir / "views.json",
        "canonical_buffer_root": buffer_root,
        "uv_albedo_path": uv_dir / "uv_albedo.png",
        "uv_normal_path": uv_dir / "uv_normal.png",
        "uv_prior_roughness_path": uv_dir / "uv_prior_roughness.png",
        "uv_prior_metallic_path": uv_dir / "uv_prior_metallic.png",
        "uv_target_roughness_path": uv_dir / "uv_target_roughness.png",
        "uv_target_metallic_path": uv_dir / "uv_target_metallic.png",
        "uv_target_confidence_path": uv_dir / "uv_target_confidence.png",
    }
    for view in load_bundle_views(bundle_dir):
        view_name = str(view["name"])
        view_dir = buffer_root / view_name
        for file_name in REQUIRED_VIEW_BUFFER_FILES:
            paths[f"view:{view_name}:{file_name.removesuffix('.png')}"] = view_dir / file_name
    return paths


def find_missing_paths(bundle_dir: Path) -> list[str]:
    missing = []
    for label, path in expected_bundle_paths(bundle_dir).items():
        if not path.exists():
            missing.append(label)
    return missing


def load_prepared_record(bundle_dir: Path) -> dict | None:
    path = bundle_dir / PREPARED_RECORD_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def build_view_rgba_paths(bundle_dir: Path) -> dict[str, dict[str, str]]:
    buffer_root = bundle_dir / "buffers"
    return {
        view_name: {"rgba": str((buffer_root / view_name / "rgba.png").resolve())}
        for view_name in [str(view["name"]) for view in load_bundle_views(bundle_dir)]
    }


def build_fallback_prepared_record(record: dict, bundle_root: Path, bundle_dir: Path, split: str) -> dict:
    uv_dir = bundle_dir / "uv"
    buffer_root = bundle_dir / "buffers"
    execution_split = "smoke" if record.get("include_in_smoke") else split
    prepared = dict(record)
    prepared.update(
        {
            "bundle_root": str(bundle_root.resolve()),
            "canonical_bundle_root": str(bundle_dir.resolve()),
            "canonical_buffer_root": str(buffer_root.resolve()),
            "canonical_views_json": str((bundle_dir / "views.json").resolve()),
            "uv_albedo_path": str((uv_dir / "uv_albedo.png").resolve()),
            "uv_normal_path": str((uv_dir / "uv_normal.png").resolve()),
            "uv_prior_roughness_path": str((uv_dir / "uv_prior_roughness.png").resolve()),
            "uv_prior_metallic_path": str((uv_dir / "uv_prior_metallic.png").resolve()),
            "uv_target_roughness_path": str((uv_dir / "uv_target_roughness.png").resolve()),
            "uv_target_metallic_path": str((uv_dir / "uv_target_metallic.png").resolve()),
            "uv_target_confidence_path": str((uv_dir / "uv_target_confidence.png").resolve()),
            "processing_status": "prepared",
            "execution_split": execution_split,
            "is_smoke": bool(record.get("include_in_smoke")),
            "view_rgba_paths": build_view_rgba_paths(bundle_dir),
        }
    )
    if not prepared.get("canonical_mesh_path") and prepared.get("source_model_path"):
        prepared["canonical_mesh_path"] = prepared["source_model_path"]
    if not prepared.get("canonical_glb_path"):
        source_model_path = Path(str(prepared.get("canonical_mesh_path") or prepared.get("source_model_path") or ""))
        prepared["canonical_glb_path"] = (
            str(source_model_path.resolve())
            if source_model_path.suffix.lower() in {".glb", ".gltf"} and source_model_path.exists()
            else ""
        )
    return enrich_record_with_quality_fields(prepared)


def build_prepared_record(record: dict, bundle_root: Path, bundle_dir: Path, split: str) -> dict:
    prepared = build_fallback_prepared_record(record, bundle_root, bundle_dir, split)
    cached_payload = load_prepared_record(bundle_dir)
    if cached_payload is not None:
        prepared.update(cached_payload)
        # Preserve source-level distribution labels from the active input
        # manifest. Older cached prepared records may contain fallback
        # material_family values such as glossy_non_metal, which would hide
        # the intended metal/thin-boundary quotas on resume.
        for key in (
            "material_family",
            "highlight_material_class",
            "thin_boundary_flag",
            "failure_tags",
            "sampling_bucket",
            "supervision_role",
            "license_bucket",
            "default_split",
        ):
            value = record.get(key)
            if key in record and value is not None and value != "":
                prepared[key] = value
    has_material_prior = bool(record.get("has_material_prior"))
    prior_mode = str(record.get("prior_mode") or "none")
    prepared.update(
        {
            "bundle_root": str(bundle_root.resolve()),
            "canonical_bundle_root": str(bundle_dir.resolve()),
            "canonical_buffer_root": str((bundle_dir / "buffers").resolve()),
            "canonical_views_json": str((bundle_dir / "views.json").resolve()),
            "uv_albedo_path": str((bundle_dir / "uv" / "uv_albedo.png").resolve()),
            "uv_normal_path": str((bundle_dir / "uv" / "uv_normal.png").resolve()),
            "uv_prior_roughness_path": str((bundle_dir / "uv" / "uv_prior_roughness.png").resolve()),
            "uv_prior_metallic_path": str((bundle_dir / "uv" / "uv_prior_metallic.png").resolve()),
            "uv_target_roughness_path": str((bundle_dir / "uv" / "uv_target_roughness.png").resolve()),
            "uv_target_metallic_path": str((bundle_dir / "uv" / "uv_target_metallic.png").resolve()),
            "uv_target_confidence_path": str((bundle_dir / "uv" / "uv_target_confidence.png").resolve()),
            "processing_status": "prepared",
            "execution_split": "smoke" if record.get("include_in_smoke") else split,
            "is_smoke": bool(record.get("include_in_smoke")),
            "has_material_prior": has_material_prior,
            "prior_mode": prior_mode,
            "view_rgba_paths": build_view_rgba_paths(bundle_dir),
        }
    )
    if not prepared.get("canonical_mesh_path") and prepared.get("source_model_path"):
        prepared["canonical_mesh_path"] = prepared["source_model_path"]
    if not prepared.get("canonical_glb_path"):
        source_model_path = Path(str(prepared.get("canonical_mesh_path") or prepared.get("source_model_path") or ""))
        prepared["canonical_glb_path"] = (
            str(source_model_path.resolve())
            if source_model_path.suffix.lower() in {".glb", ".gltf"} and source_model_path.exists()
            else ""
        )
    return enrich_record_with_quality_fields(prepared)


def build_partial_manifest(manifest: dict, bundle_root: Path, split: str) -> dict:
    selected = select_records(manifest, split)
    resolved_bundle_root = bundle_root.resolve()

    prepared_records: list[dict] = []
    skipped_records: list[dict[str, object]] = []

    for record in selected:
        object_id = str(record["object_id"])
        bundle_dir = resolved_bundle_root / object_id
        missing = find_missing_paths(bundle_dir)
        if missing:
            skipped_records.append(
                {
                    "object_id": object_id,
                    "source_name": record.get("source_name", ""),
                    "reason": "missing_bundle_files",
                    "missing": missing,
                }
            )
            continue
        prepared_records.append(build_prepared_record(record, resolved_bundle_root, bundle_dir, split))

    counts = {
        "prepared_records": len(prepared_records),
        "skipped_records": len(skipped_records),
        "with_prior": sum(bool(record.get("has_material_prior")) for record in prepared_records),
        "without_prior": sum(not bool(record.get("has_material_prior")) for record in prepared_records),
        "by_source_name": dict(Counter(str(record.get("source_name", "")) for record in prepared_records)),
    }

    return {
        "manifest_version": manifest.get("manifest_version", "canonical_asset_record_v1"),
        "input_manifest": str(Path(manifest.get("input_manifest", "")).resolve()) if manifest.get("input_manifest") else None,
        "split": split,
        "canonical_bundle_root": str(resolved_bundle_root),
        "counts": counts,
        "records": prepared_records,
        "skipped_records": skipped_records,
    }


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.input_manifest)
    payload = build_partial_manifest(
        manifest,
        args.canonical_bundle_root,
        args.split,
    )
    payload["input_manifest"] = str(args.input_manifest.resolve())
    write_json(args.output_manifest, payload)


if __name__ == "__main__":
    main()
