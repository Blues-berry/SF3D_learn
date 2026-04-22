#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DEFAULT_TAR_ROOT = PROJECT_ROOT / "output" / "material_refine_real_benchmarks" / "stanford_orb_pool_f" / "stanford_orb"
DEFAULT_EXTRACT_ROOT = PROJECT_ROOT / "output" / "material_refine_real_benchmarks" / "stanford_orb_pool_f" / "extracted"
DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "output"
    / "material_refine_real_benchmarks"
    / "stanford_orb_pool_f"
    / "stanford_orb_benchmark_manifest.json"
)

ARCHIVES = {
    "ground_truth": "ground_truth.tar.gz",
    "blender_LDR": "blender_LDR.tar.gz",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tif", ".tiff"}
SIDECAR_EXTS = {".json", ".txt", ".npy", ".npz", ".pkl", ".csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract and index Stanford-ORB as a Pool-F benchmark-only manifest.",
    )
    parser.add_argument("--tar-root", type=Path, default=DEFAULT_TAR_ROOT)
    parser.add_argument("--extract-root", type=Path, default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--extract", action="store_true", help="Extract downloaded Stanford-ORB archives before indexing.")
    parser.add_argument("--max-records", type=int, default=0, help="0 means keep all discovered RGB/HDR image records.")
    parser.add_argument("--overwrite-extract", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_extract(archive_path: Path, destination: Path, *, overwrite: bool) -> dict[str, Any]:
    marker = destination / ".sf3d_extract_complete"
    if marker.exists() and not overwrite:
        return {
            "archive": str(archive_path),
            "destination": str(destination),
            "status": "already_extracted",
            "elapsed_seconds": 0.0,
        }
    if not archive_path.exists():
        return {
            "archive": str(archive_path),
            "destination": str(destination),
            "status": "missing_archive",
            "elapsed_seconds": 0.0,
        }
    destination.mkdir(parents=True, exist_ok=True)
    if overwrite and marker.exists():
        marker.unlink()
    started = time.time()
    process = subprocess.run(
        ["tar", "-xzf", str(archive_path), "-C", str(destination)],
        check=False,
    )
    elapsed = round(time.time() - started, 3)
    status = "extracted" if process.returncode == 0 else "extract_failed"
    if process.returncode == 0:
        marker.write_text(json.dumps({"archive": str(archive_path), "completed_unix": time.time()}), encoding="utf-8")
    return {
        "archive": str(archive_path),
        "destination": str(destination),
        "status": status,
        "exit_code": int(process.returncode),
        "elapsed_seconds": elapsed,
    }


def classify_path(path: Path) -> str:
    lower = str(path).lower()
    name = path.name.lower()
    if "/mesh_blender/" in lower or name.startswith("texture_"):
        return "material_sidecar"
    if any(token in lower for token in ("/mask", "_mask", "segmentation", "alpha")):
        return "mask"
    if "normal" in lower:
        return "normal"
    if "depth" in lower or "distance" in lower:
        return "depth"
    if "roughness" in lower:
        return "roughness"
    if "metallic" in lower or "metalness" in lower:
        return "metallic"
    if "albedo" in lower or "diffuse" in lower:
        return "albedo"
    if any(token in lower for token in ("envmap", "environment", "lighting", "light_probe", "hdr_env")):
        return "environment"
    if path.suffix.lower() in IMAGE_EXTS:
        if path.suffix.lower() in {".hdr", ".exr"}:
            return "hdr_image"
        if any(token in name for token in ("rgb", "image", "render", "photo", "ldr", "linear")):
            return "rgb_image"
        return "image"
    if path.suffix.lower() in SIDECAR_EXTS:
        return "sidecar"
    return "other"


def infer_object_id(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = [part for part in rel.parts if part not in {"ground_truth", "blender_LDR", "images", "renders", "rgb", "ldr"}]
    for part in parts:
        lower = part.lower()
        if lower.startswith(("object", "obj_", "orb_", "scene", "test_")):
            return part
    if len(parts) >= 2:
        return parts[0]
    return path.parent.name or "stanford_orb_unknown"


def stable_split(object_id: str) -> str:
    # Benchmark pool stays holdout-only; this field is for compatibility with manifest-driven evaluators.
    return "test"


def build_records(extract_root: Path, *, max_records: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = [path for path in extract_root.rglob("*") if path.is_file() and path.name != ".sf3d_extract_complete"]
    role_counts = Counter(classify_path(path) for path in files)
    sidecars_by_object: dict[str, list[str]] = defaultdict(list)
    envs_by_object: dict[str, list[str]] = defaultdict(list)
    image_candidates: list[Path] = []
    for path in files:
        role = classify_path(path)
        object_id = infer_object_id(path, extract_root)
        if role in {"sidecar", "material_sidecar", "mask", "normal", "depth", "roughness", "metallic", "albedo"}:
            sidecars_by_object[object_id].append(str(path))
        elif role == "environment":
            envs_by_object[object_id].append(str(path))
        elif role in {"rgb_image", "hdr_image", "image"}:
            image_candidates.append(path)

    image_candidates = sorted(image_candidates)
    if max_records > 0:
        image_candidates = image_candidates[:max_records]

    records = []
    for idx, image_path in enumerate(image_candidates):
        object_id = infer_object_id(image_path, extract_root)
        role = classify_path(image_path)
        rel = image_path.relative_to(extract_root).as_posix()
        digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
        records.append(
            {
                "record_version": "sf3d_real_benchmark_record_v1",
                "record_id": f"stanford_orb_{digest}",
                "canonical_object_id": f"stanford_orb::{object_id}",
                "object_id": f"stanford_orb::{object_id}",
                "source_name": "Stanford-ORB",
                "source_dataset": "Stanford-ORB",
                "pool_name": "pool_F_real_world_eval_holdout",
                "benchmark_role": "real_world_eval_holdout",
                "supervision_role": "benchmark_only",
                "target_source_type": "real_benchmark_no_uv_target",
                "target_quality_tier": "benchmark_only",
                "license_bucket": "benchmark_project_terms_unstated",
                "paper_split": "paper_test_real_lighting",
                "default_split": stable_split(object_id),
                "view_supervision_ready": False,
                "valid_view_count": 1,
                "image_role": role,
                "image_path": str(image_path),
                "relative_image_path": rel,
                "sidecar_paths": sorted(sidecars_by_object.get(object_id, []))[:32],
                "environment_paths": sorted(envs_by_object.get(object_id, []))[:32],
                "notes": "Pool-F benchmark-only record; do not mix into Pool-A training or paper_stage1 train split.",
            }
        )

    summary = {
        "manifest_version": "sf3d_stanford_orb_benchmark_v1",
        "created_unix": time.time(),
        "extract_root": str(extract_root),
        "records": len(records),
        "file_count": len(files),
        "role_counts": dict(sorted(role_counts.items())),
        "object_count": len({record["canonical_object_id"] for record in records}),
        "paper_split_counts": dict(Counter(record["paper_split"] for record in records)),
        "supervision_role_counts": dict(Counter(record["supervision_role"] for record in records)),
        "target_quality_tier_counts": dict(Counter(record["target_quality_tier"] for record in records)),
    }
    return records, summary


def main() -> None:
    args = parse_args()
    extraction_results = []
    if args.extract:
        for archive_key, archive_name in ARCHIVES.items():
            extraction_results.append(
                run_extract(
                    args.tar_root / archive_name,
                    args.extract_root / archive_key,
                    overwrite=args.overwrite_extract,
                )
            )

    records, summary = build_records(args.extract_root, max_records=args.max_records)
    summary["extraction_results"] = extraction_results
    write_json(args.output_manifest, {"summary": summary, "records": records})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
