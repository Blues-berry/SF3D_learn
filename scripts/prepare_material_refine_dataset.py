#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from bake_material_refine_uv_targets import bake_uv_targets
from PIL import Image
from refresh_material_refine_partial_manifest import (
    PREPARED_RECORD_FILENAME,
    build_partial_manifest,
    build_prepared_record,
    find_missing_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_MANIFEST = REPO_ROOT / "docs" / "neural_gaffer_dataset_audit" / "material_refine_manifest_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "material_refine"
DEFAULT_ABO_RENDER_CACHE = REPO_ROOT / "output" / "abo_rm_mini" / "renders"
DEFAULT_BLENDER_BIN = Path(
    "/4T/CXY/Neural_Gaffer_original/scripts/Objavarse_rendering/blender-3.2.2-linux-x64/blender"
)
BLENDER_SCRIPT = REPO_ROOT / "scripts" / "abo_material_passes_blender.py"
VIEWS = [
    {
        "name": "front_studio",
        "azimuth": 0.0,
        "elevation": 18.0,
        "distance": 2.0,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "studio_small_08_1k.hdr"),
    },
    {
        "name": "three_quarter_indoor",
        "azimuth": 45.0,
        "elevation": 28.0,
        "distance": 2.0,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "peppermint_powerplant_1k.hdr"),
    },
    {
        "name": "side_neon",
        "azimuth": 110.0,
        "elevation": 16.0,
        "distance": 2.05,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "neon_photostudio_1k.hdr"),
    },
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the CanonicalAssetRecordV1 bundle for GPU0 dataset work."
    )
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--split", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--atlas-resolution", type=int, default=1024)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--cycles-samples", type=int, default=32)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--abo-render-cache", type=Path, default=DEFAULT_ABO_RENDER_CACHE)
    parser.add_argument("--cuda-device-index", type=str, default="0")
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--partial-manifest", type=Path, default=None)
    parser.add_argument("--refresh-partial-every", type=int, default=25)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "records" not in payload:
        raise RuntimeError(f"unexpected_manifest_payload:{path}")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def select_records(payload: dict, split: str, max_records: int | None) -> list[dict]:
    selected = []
    for record in payload["records"]:
        include_flag = record.get("include_in_smoke") if split == "smoke" else record.get("include_in_full", True)
        if not include_flag:
            continue
        if not record.get("source_asset_available", False):
            continue
        selected.append(dict(record))
    if max_records is not None:
        selected = selected[:max_records]
    return selected


def first_existing_path(*candidates: str) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path.resolve()
    return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def resolve_prepared_prior_fields(
    record: dict,
    *,
    roughness_src: Path | None,
    metallic_src: Path | None,
) -> tuple[bool, str]:
    has_material_prior = bool(record.get("has_material_prior"))
    prior_mode = str(record.get("prior_mode") or "none")
    if has_material_prior and prior_mode in {"", "none"}:
        if roughness_src is not None or metallic_src is not None:
            prior_mode = "uv_rm"
        else:
            prior_mode = "scalar_rm"
    elif not has_material_prior:
        prior_mode = "none"
    return has_material_prior, prior_mode


def write_prepared_record_file(bundle_dir: Path, prepared: dict) -> None:
    write_json(bundle_dir / PREPARED_RECORD_FILENAME, prepared)


def build_prepare_manifest_payload(
    *,
    input_manifest: Path,
    split: str,
    bundle_root: Path,
    selected: list[dict],
    prepared_records: list[dict],
    skipped_records: list[dict],
) -> dict:
    return {
        "manifest_version": "canonical_asset_record_v1_prepared",
        "input_manifest": str(input_manifest.resolve()),
        "split": split,
        "canonical_bundle_root": str(bundle_root.resolve()),
        "counts": {
            "selected_records": len(selected),
            "prepared_records": len(prepared_records),
            "skipped_records": len(skipped_records),
            "with_prior": sum(bool(record.get("has_material_prior")) for record in prepared_records),
            "without_prior": sum(not bool(record.get("has_material_prior")) for record in prepared_records),
            "by_source_name": dict(Counter(str(record.get("source_name", "")) for record in prepared_records)),
        },
        "skipped_records": skipped_records,
        "records": prepared_records,
    }


def build_run_summary(
    *,
    output_manifest: Path,
    split: str,
    selected: list[dict],
    prepared_records: list[dict],
    skipped_records: list[dict],
) -> dict:
    return {
        "split": split,
        "output_manifest": str(output_manifest.resolve()),
        "selected_records": len(selected),
        "prepared_records": len(prepared_records),
        "skipped_records": len(skipped_records),
        "source_mix": dict(Counter(str(record.get("source_name", "")) for record in prepared_records)),
        "with_prior": sum(bool(record.get("has_material_prior")) for record in prepared_records),
        "without_prior": sum(not bool(record.get("has_material_prior")) for record in prepared_records),
        "render_modes": dict(Counter(str(record.get("render_mode", "unknown")) for record in prepared_records)),
        "skipped_reasons": dict(Counter(str(record.get("reason", "unknown")) for record in skipped_records)),
    }


def write_summary_md(path: Path, summary: dict) -> None:
    lines = [
        "# Material Refine Prepare Summary",
        "",
        f"- split: {summary['split']}",
        f"- output_manifest: {summary['output_manifest']}",
        f"- selected_records: {summary['selected_records']}",
        f"- prepared_records: {summary['prepared_records']}",
        f"- skipped_records: {summary['skipped_records']}",
        f"- with_prior: {summary['with_prior']}",
        f"- without_prior: {summary['without_prior']}",
        "",
        "## Source Mix",
        "",
    ]
    for source_name, count in sorted(summary["source_mix"].items()):
        lines.append(f"- {source_name}: {count}")
    lines.extend(["", "## Render Modes", ""])
    for render_mode, count in sorted(summary["render_modes"].items()):
        lines.append(f"- {render_mode}: {count}")
    lines.extend(["", "## Skipped Reasons", ""])
    if summary["skipped_reasons"]:
        for reason, count in sorted(summary["skipped_reasons"].items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_constant_rgb(path: Path, color: tuple[int, int, int], size: int) -> None:
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[..., 0] = color[0]
    array[..., 1] = color[1]
    array[..., 2] = color[2]
    Image.fromarray(array, mode="RGB").save(path)


def write_flat_normal(path: Path, size: int) -> None:
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[..., 0] = 128
    array[..., 1] = 128
    array[..., 2] = 255
    Image.fromarray(array, mode="RGB").save(path)


def write_constant_gray(path: Path, value: float, size: int) -> None:
    array = np.full((size, size), int(round(clamp01(value) * 255.0)), dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


def load_glb_json(path: Path) -> dict | None:
    if path.suffix.lower() != ".glb" or not path.exists():
        return None
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12:
            return None
        magic, version, _length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            return None
        chunk_header = handle.read(8)
        if len(chunk_header) < 8:
            return None
        chunk_len, chunk_type = struct.unpack("<I4s", chunk_header)
        if chunk_type != b"JSON":
            return None
        chunk = handle.read(chunk_len)
    return json.loads(chunk.decode("utf-8"))


def infer_scalar_priors(source_model_path: Path, record: dict) -> tuple[float, float]:
    if record.get("scalar_prior_roughness") not in {None, ""} and record.get("scalar_prior_metallic") not in {None, ""}:
        return float(record["scalar_prior_roughness"]), float(record["scalar_prior_metallic"])

    roughness = 0.5
    metallic = 0.0
    doc = load_glb_json(source_model_path)
    if doc is not None:
        materials = doc.get("materials", [])
        if materials:
            pbr = materials[0].get("pbrMetallicRoughness", {})
            roughness = float(pbr.get("roughnessFactor", roughness))
            metallic = float(pbr.get("metallicFactor", metallic))
    return clamp01(roughness), clamp01(metallic)


def discover_texture(texture_root: Path | None, keywords: tuple[str, ...]) -> Path | None:
    if texture_root is None or not texture_root.exists():
        return None
    if texture_root.is_file():
        if texture_root.suffix.lower() in IMAGE_EXTENSIONS:
            return texture_root.resolve()
        return None
    candidates = []
    for path in texture_root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = path.stem.lower()
        if any(keyword in name for keyword in keywords):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort()
    return candidates[0].resolve()


def copy_or_placeholder(
    src_path: Path | None,
    dst_path: Path,
    *,
    fallback_kind: str,
    fallback_value: float | None = None,
    size: int,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path is not None and src_path.exists():
        if src_path.resolve() != dst_path.resolve():
            shutil.copy2(src_path, dst_path)
        return
    if fallback_kind == "albedo":
        write_constant_rgb(dst_path, (180, 180, 180), size)
    elif fallback_kind == "normal":
        write_flat_normal(dst_path, size)
    elif fallback_kind == "gray":
        write_constant_gray(dst_path, fallback_value or 0.0, size)
    else:
        raise ValueError(f"unsupported_fallback_kind:{fallback_kind}")


def render_bundle_complete(buffer_root: Path) -> bool:
    if not buffer_root.exists():
        return False
    for view in VIEWS:
        view_dir = buffer_root / view["name"]
        if not view_dir.is_dir():
            return False
        for name in ("rgba.png", "roughness.png", "metallic.png"):
            if not (view_dir / name).exists():
                return False
    return True


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def resolve_cached_render_root(
    record: dict,
    source_model_path: Path,
    abo_render_cache: Path | None,
) -> Path | None:
    if abo_render_cache is None or not abo_render_cache.exists():
        return None
    source_name = str(record.get("source_name", "")).lower()
    generator_id = str(record.get("generator_id", "")).lower()
    if "abo" not in source_name and "abo" not in generator_id:
        return None
    candidate = abo_render_cache / source_model_path.stem
    if render_bundle_complete(candidate):
        return candidate.resolve()
    return None


def link_or_copy_render_root(src_root: Path, dst_root: Path) -> None:
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    if dst_root.exists() or dst_root.is_symlink():
        remove_path(dst_root)
    try:
        dst_root.symlink_to(src_root, target_is_directory=True)
    except OSError:
        shutil.copytree(src_root, dst_root)


def ensure_view_placeholders(buffer_root: Path, render_resolution: int) -> dict[str, dict[str, str]]:
    rgba_paths: dict[str, dict[str, str]] = {}
    for view in VIEWS:
        view_dir = ensure_dir(buffer_root / view["name"])
        rgba_path = view_dir / "rgba.png"
        if not rgba_path.exists():
            rgba = np.zeros((render_resolution, render_resolution, 4), dtype=np.uint8)
            rgba[..., :3] = 180
            rgba[..., 3] = 255
            Image.fromarray(rgba, mode="RGBA").save(rgba_path)
        roughness_path = view_dir / "roughness.png"
        if not roughness_path.exists():
            write_constant_gray(roughness_path, 0.5, render_resolution)
        metallic_path = view_dir / "metallic.png"
        if not metallic_path.exists():
            write_constant_gray(metallic_path, 0.0, render_resolution)
        rgba_paths[view["name"]] = {"rgba": str(rgba_path.resolve())}
    return rgba_paths


def render_views(
    *,
    source_model_path: Path,
    buffer_root: Path,
    views_json: Path,
    blender_bin: Path,
    render_resolution: int,
    cycles_samples: int,
    cuda_device_index: str,
) -> None:
    views_json.write_text(json.dumps(VIEWS, indent=2))
    cmd = [
        str(blender_bin),
        "-b",
        "-P",
        str(BLENDER_SCRIPT),
        "--",
        "--object-path",
        str(source_model_path),
        "--output-dir",
        str(buffer_root),
        "--views-json",
        str(views_json),
        "--resolution",
        str(render_resolution),
        "--cycles-samples",
        str(cycles_samples),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_device_index
    env["BLENDER_CUDA_DEVICE_INDEX"] = "0"
    subprocess.run(cmd, check=True, env=env)


def prepare_record(
    record: dict,
    *,
    output_root: Path,
    split: str,
    atlas_resolution: int,
    render_resolution: int,
    cycles_samples: int,
    blender_bin: Path,
    abo_render_cache: Path | None,
    cuda_device_index: str,
    skip_render: bool,
) -> dict:
    source_model_path = Path(record["source_model_path"]).resolve()
    texture_root = first_existing_path(record.get("source_texture_root", ""))
    bundle_root = ensure_dir(output_root / "canonical_bundle")
    bundle_dir = ensure_dir(bundle_root / record["object_id"])
    uv_dir = ensure_dir(bundle_dir / "uv")
    buffer_root = bundle_dir / "buffers"
    views_json = bundle_dir / "views.json"
    views_json.write_text(json.dumps(VIEWS, indent=2))

    roughness_seed, metallic_seed = infer_scalar_priors(source_model_path, record)
    albedo_src = discover_texture(texture_root, ("albedo", "basecolor", "base_color", "diffuse"))
    normal_src = discover_texture(texture_root, ("normal", "norm"))
    roughness_src = discover_texture(texture_root, ("roughness", "rough"))
    metallic_src = discover_texture(texture_root, ("metallic", "metalness", "metal"))
    has_material_prior, prior_mode = resolve_prepared_prior_fields(
        record,
        roughness_src=roughness_src,
        metallic_src=metallic_src,
    )

    albedo_dst = uv_dir / "uv_albedo.png"
    normal_dst = uv_dir / "uv_normal.png"
    roughness_dst = uv_dir / "uv_prior_roughness.png"
    metallic_dst = uv_dir / "uv_prior_metallic.png"

    if not find_missing_paths(bundle_dir):
        prepared = build_prepared_record(record, bundle_root, bundle_dir, split)
        prepared.update(
            {
                "scalar_prior_roughness": roughness_seed,
                "scalar_prior_metallic": metallic_seed,
                "has_material_prior": has_material_prior,
                "prior_mode": prior_mode,
                "render_mode": str(prepared.get("render_mode") or "existing_bundle"),
            }
        )
        write_prepared_record_file(bundle_dir, prepared)
        print(f"[prepare reuse] complete bundle for {record['object_id']}")
        return prepared

    copy_or_placeholder(albedo_src, albedo_dst, fallback_kind="albedo", size=atlas_resolution)
    copy_or_placeholder(normal_src, normal_dst, fallback_kind="normal", size=atlas_resolution)
    copy_or_placeholder(
        roughness_src,
        roughness_dst,
        fallback_kind="gray",
        fallback_value=roughness_seed,
        size=atlas_resolution,
    )
    copy_or_placeholder(
        metallic_src,
        metallic_dst,
        fallback_kind="gray",
        fallback_value=metallic_seed,
        size=atlas_resolution,
    )

    cached_render_root = resolve_cached_render_root(record, source_model_path, abo_render_cache)
    if render_bundle_complete(buffer_root):
        render_mode = "existing_bundle"
        print(f"[prepare reuse] existing bundle buffers for {record['object_id']}")
    elif cached_render_root is not None:
        link_or_copy_render_root(cached_render_root, buffer_root)
        render_mode = "abo_render_cache"
        print(
            f"[prepare reuse] cached ABO render for {record['object_id']} <- {cached_render_root.name}"
        )
    elif not skip_render and blender_bin.exists():
        ensure_dir(buffer_root)
        render_views(
            source_model_path=source_model_path,
            buffer_root=buffer_root,
            views_json=views_json,
            blender_bin=blender_bin,
            render_resolution=render_resolution,
            cycles_samples=cycles_samples,
            cuda_device_index=cuda_device_index,
        )
        render_mode = "rendered"
    else:
        ensure_dir(buffer_root)
        render_mode = "placeholder"
    ensure_view_placeholders(buffer_root, render_resolution)

    uv_target_payload = bake_uv_targets(
        bundle_dir=bundle_dir,
        uv_prior_roughness_path=roughness_dst,
        uv_prior_metallic_path=metallic_dst,
        atlas_resolution=atlas_resolution,
        default_roughness=roughness_seed,
        default_metallic=metallic_seed,
    )

    missing = find_missing_paths(bundle_dir)
    if missing:
        raise RuntimeError(f"incomplete_bundle_after_prepare:{','.join(missing)}")

    prepared = build_prepared_record(record, bundle_root, bundle_dir, split)
    prepared.update(
        {
            "scalar_prior_roughness": roughness_seed,
            "scalar_prior_metallic": metallic_seed,
            "uv_target_roughness_path": uv_target_payload["uv_target_roughness_path"],
            "uv_target_metallic_path": uv_target_payload["uv_target_metallic_path"],
            "uv_target_confidence_path": uv_target_payload["uv_target_confidence_path"],
            "has_material_prior": has_material_prior,
            "prior_mode": prior_mode,
            "render_mode": render_mode,
        }
    )
    write_prepared_record_file(bundle_dir, prepared)
    return prepared


def main() -> None:
    args = parse_args()
    payload = read_manifest(args.input_manifest)
    output_root = ensure_dir(args.output_root / args.split)
    bundle_root = ensure_dir(output_root / "canonical_bundle")
    output_manifest = (
        args.output_manifest
        if args.output_manifest is not None
        else output_root / f"canonical_manifest_{args.split}.json"
    )
    partial_manifest_path = args.partial_manifest
    if partial_manifest_path is None and args.split == "full" and output_manifest.name == "canonical_manifest_full.json":
        partial_manifest_path = output_root / "canonical_manifest_partial.json"
    summary_json_path = (
        args.summary_json
        if args.summary_json is not None
        else output_root / f"prepare_{args.split}_summary.json"
    )
    summary_md_path = (
        args.summary_md
        if args.summary_md is not None
        else output_root / f"prepare_{args.split}_summary.md"
    )
    selected = select_records(payload, args.split, args.max_records)
    skipped_records: list[dict] = []
    prepared_indexed: list[tuple[int, dict]] = []
    last_partial_refresh_completed = -1

    def refresh_partial_manifest(force: bool = False) -> None:
        nonlocal last_partial_refresh_completed
        if partial_manifest_path is None:
            return
        completed = len(prepared_indexed) + len(skipped_records)
        if not force:
            if args.refresh_partial_every <= 0:
                return
            if completed == 0:
                return
            if completed < last_partial_refresh_completed + args.refresh_partial_every:
                return
        partial_payload = build_partial_manifest(payload, bundle_root, args.split)
        partial_payload["input_manifest"] = str(args.input_manifest.resolve())
        write_json(partial_manifest_path, partial_payload)
        last_partial_refresh_completed = completed
        print(
            "[prepare partial] "
            f"prepared={partial_payload['counts']['prepared_records']} "
            f"skipped={partial_payload['counts']['skipped_records']} "
            f"path={partial_manifest_path}"
        )

    def run_prepare(index: int, record: dict) -> tuple[int, dict]:
        prepared = prepare_record(
            record,
            output_root=output_root,
            split=args.split,
            atlas_resolution=args.atlas_resolution,
            render_resolution=args.render_resolution,
            cycles_samples=args.cycles_samples,
            blender_bin=args.blender_bin,
            abo_render_cache=args.abo_render_cache,
            cuda_device_index=args.cuda_device_index,
            skip_render=args.skip_render,
        )
        return index, prepared

    refresh_partial_manifest(force=True)

    if args.parallel_workers <= 1:
        for index, record in enumerate(selected):
            try:
                prepared_indexed.append(run_prepare(index, record))
            except Exception as exc:  # noqa: BLE001
                skipped_records.append(
                    {
                        "object_id": record["object_id"],
                        "source_name": record["source_name"],
                        "reason": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
            refresh_partial_manifest()
    else:
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
            future_to_record = {
                executor.submit(run_prepare, index, record): (index, record)
                for index, record in enumerate(selected)
            }
            for future in as_completed(future_to_record):
                _index, record = future_to_record[future]
                try:
                    prepared_indexed.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    skipped_records.append(
                        {
                            "object_id": record["object_id"],
                            "source_name": record["source_name"],
                            "reason": type(exc).__name__,
                            "detail": str(exc),
                        }
                    )
                refresh_partial_manifest()

    prepared_records = [
        prepared
        for _index, prepared in sorted(prepared_indexed, key=lambda item: item[0])
    ]

    out_payload = build_prepare_manifest_payload(
        input_manifest=args.input_manifest,
        split=args.split,
        bundle_root=bundle_root,
        selected=selected,
        prepared_records=prepared_records,
        skipped_records=skipped_records,
    )
    write_json(output_manifest, out_payload)
    refresh_partial_manifest(force=True)
    summary_payload = build_run_summary(
        output_manifest=output_manifest,
        split=args.split,
        selected=selected,
        prepared_records=prepared_records,
        skipped_records=skipped_records,
    )
    write_json(summary_json_path, summary_payload)
    write_summary_md(summary_md_path, summary_payload)
    print(f"wrote {output_manifest}")
    print(json.dumps(out_payload["counts"], indent=2))
    print(f"wrote {summary_json_path}")
    print(f"wrote {summary_md_path}")


if __name__ == "__main__":
    main()
