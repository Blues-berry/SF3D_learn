#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DEFAULT_PREPARED_MANIFEST = (
    PROJECT_ROOT / "output" / "highlight_pool_a_8k" / "material_refine_manifest_pool_a_no_3dfuture.json"
)
DEFAULT_HDRI_MANIFEST = (
    PROJECT_ROOT / "output" / "highlight_pool_a_8k" / "aux_sources" / "polyhaven_hdri_bank_60.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "highlight_pool_a_8k" / "polyhaven_stress_gpu0_no_3dfuture"
DEFAULT_BLENDER_BIN = Path(
    "/4T/CXY/Neural_Gaffer_original/scripts/Objavarse_rendering/blender-3.2.2-linux-x64/blender"
)
BLENDER_SCRIPT = PROJECT_ROOT / "scripts" / "abo_material_passes_blender.py"


VIEW_SPECS = {
    "indoor_high_contrast": {"elevation": 15.0, "azimuth": 25.0, "distance": 2.2},
    "indoor_soft_window": {"elevation": 18.0, "azimuth": 65.0, "distance": 2.25},
    "outdoor_sun_hard": {"elevation": 12.0, "azimuth": 115.0, "distance": 2.3},
    "outdoor_overcast_soft": {"elevation": 22.0, "azimuth": 175.0, "distance": 2.25},
    "night_urban_neon": {"elevation": 10.0, "azimuth": 235.0, "distance": 2.2},
    "studio_product": {"elevation": 20.0, "azimuth": 315.0, "distance": 2.15},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Poly Haven HDRI stress views for prepared Pool-A objects on a selected GPU."
    )
    parser.add_argument("--prepared-manifest", type=Path, default=DEFAULT_PREPARED_MANIFEST)
    parser.add_argument("--hdri-manifest", type=Path, default=DEFAULT_HDRI_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--cuda-device-index", type=str, default="0")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--refresh-every", type=int, default=100)
    parser.add_argument(
        "--hdri-mode",
        choices=["stratum-one", "all-hdri"],
        default="stratum-one",
        help="Use one deterministic HDRI per stratum or every downloaded HDRI for each object.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_hdri_by_stratum(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = read_json(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("records", []):
        if row.get("download_status") != "downloaded":
            continue
        local_path = Path(str(row.get("local_path") or ""))
        if not local_path.is_absolute():
            local_path = (PROJECT_ROOT / local_path).resolve()
        if not local_path.exists():
            continue
        grouped[str(row.get("stratum") or "unknown")].append({**row, "local_path": str(local_path)})
    for rows in grouped.values():
        rows.sort(key=lambda item: str(item.get("asset_id") or ""))
    return grouped


def choose_views(
    object_id: str,
    hdri_by_stratum: dict[str, list[dict[str, Any]]],
    *,
    hdri_mode: str,
) -> list[dict[str, Any]]:
    offset = sum(ord(ch) for ch in object_id)
    views = []
    if hdri_mode == "all-hdri":
        for stratum in sorted(hdri_by_stratum):
            spec = dict(VIEW_SPECS.get(stratum, VIEW_SPECS["studio_product"]))
            for local_index, hdri in enumerate(hdri_by_stratum[stratum]):
                asset_id = str(hdri.get("asset_id") or f"hdri_{local_index}")
                view_spec = dict(spec)
                view_spec["azimuth"] = float(view_spec["azimuth"]) + 11.0 * (local_index % 6)
                view_spec["elevation"] = float(view_spec["elevation"]) + 2.0 * ((local_index // 6) % 3)
                views.append(
                    {
                        "name": f"polyhaven_{stratum}_{asset_id}",
                        "hdri": hdri["local_path"],
                        "hdri_asset_id": asset_id,
                        "hdri_stratum": stratum,
                        **view_spec,
                    }
                )
        if not views:
            raise RuntimeError("no_downloaded_hdri_rows")
        return views

    for stratum, spec in VIEW_SPECS.items():
        candidates = hdri_by_stratum.get(stratum, [])
        if not candidates:
            continue
        hdri = candidates[offset % len(candidates)]
        views.append(
            {
                "name": f"polyhaven_{stratum}",
                "hdri": hdri["local_path"],
                "hdri_asset_id": hdri.get("asset_id", ""),
                "hdri_stratum": stratum,
                **spec,
            }
        )
    if len(views) != len(VIEW_SPECS):
        raise RuntimeError(f"missing_hdri_strata:expected={len(VIEW_SPECS)} actual={len(views)}")
    return views


def stress_bundle_complete(buffer_root: Path, views: list[dict[str, Any]]) -> bool:
    for view in views:
        view_dir = buffer_root / view["name"]
        for filename in ("rgba.png", "roughness.png", "metallic.png", "view.json"):
            if not (view_dir / filename).exists():
                return False
    return True


def render_record(
    record: dict[str, Any],
    *,
    hdri_by_stratum: dict[str, list[dict[str, Any]]],
    output_root: Path,
    blender_bin: Path,
    cuda_device_index: str,
    resolution: int,
    cycles_samples: int,
    hdri_mode: str,
) -> dict[str, Any]:
    object_id = str(record["object_id"])
    source_model_path = Path(str(record.get("source_model_path") or ""))
    if not source_model_path.exists():
        raise RuntimeError(f"missing_source_model_path:{source_model_path}")
    bundle_dir = output_root / "render_bundle" / object_id
    buffer_root = bundle_dir / "buffers"
    views_json = bundle_dir / "views_polyhaven_stress.json"
    views = choose_views(object_id, hdri_by_stratum, hdri_mode=hdri_mode)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    write_json(views_json, views)
    if stress_bundle_complete(buffer_root, views):
        return {
            "object_id": object_id,
            "source_name": record.get("source_name", ""),
            "default_split": record.get("default_split", ""),
            "highlight_material_class": record.get("highlight_material_class", ""),
            "render_mode": "existing_stress_bundle",
            "buffer_root": str(buffer_root.resolve()),
            "views_json": str(views_json.resolve()),
            "view_count": len(views),
        }

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
        str(resolution),
        "--cycles-samples",
        str(cycles_samples),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device_index)
    env["BLENDER_CUDA_DEVICE_INDEX"] = "0"
    subprocess.run(cmd, check=True, env=env)
    if not stress_bundle_complete(buffer_root, views):
        raise RuntimeError(f"incomplete_polyhaven_stress_bundle:{object_id}")
    return {
        "object_id": object_id,
        "source_name": record.get("source_name", ""),
        "default_split": record.get("default_split", ""),
        "highlight_material_class": record.get("highlight_material_class", ""),
        "render_mode": "rendered",
        "buffer_root": str(buffer_root.resolve()),
        "views_json": str(views_json.resolve()),
        "view_count": len(views),
    }


def build_payload(
    *,
    input_manifest: Path,
    hdri_manifest: Path,
    output_root: Path,
    hdri_mode: str,
    selected_count: int,
    rendered_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "manifest_version": "polyhaven_hdri_stress_v1",
        "generated_at_utc": utc_now(),
        "input_manifest": str(input_manifest.resolve()),
        "hdri_manifest": str(hdri_manifest.resolve()),
        "hdri_mode": hdri_mode,
        "output_root": str(output_root.resolve()),
        "counts": {
            "selected_records": selected_count,
            "rendered_records": len(rendered_rows),
            "skipped_records": len(skipped_rows),
            "by_render_mode": dict(Counter(row.get("render_mode", "") for row in rendered_rows)),
            "by_source_name": dict(Counter(row.get("source_name", "") for row in rendered_rows)),
            "by_highlight_material_class": dict(Counter(row.get("highlight_material_class", "") for row in rendered_rows)),
        },
        "skipped_records": skipped_rows,
        "records": rendered_rows,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    counts = payload["counts"]
    lines = [
        "# Poly Haven HDRI Stress Render",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- selected_records: {counts['selected_records']}",
        f"- rendered_records: {counts['rendered_records']}",
        f"- skipped_records: {counts['skipped_records']}",
        f"- output_root: `{payload['output_root']}`",
        "",
        "## Render Modes",
        "",
    ]
    for mode, count in sorted(counts["by_render_mode"].items()):
        lines.append(f"- {mode}: {count}")
    lines.extend(["", "## Source Mix", ""])
    for source, count in sorted(counts["by_source_name"].items()):
        lines.append(f"- {source}: {count}")
    if payload["skipped_records"]:
        lines.extend(["", "## Skipped", ""])
        for row in payload["skipped_records"][:50]:
            lines.append(f"- {row['object_id']}: {row['reason']} - {row['detail']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_manifest = args.output_manifest or args.output_root / "polyhaven_stress_manifest.json"
    output_csv = args.output_root / "polyhaven_stress_manifest.csv"
    summary_md = args.summary_md or args.output_root / "polyhaven_stress_summary.md"
    prepared_payload = read_json(args.prepared_manifest)
    records = list(prepared_payload.get("records", []))
    if args.max_records is not None:
        records = records[: args.max_records]
    hdri_by_stratum = load_hdri_by_stratum(args.hdri_manifest)
    rendered_indexed: list[tuple[int, dict[str, Any]]] = []
    skipped_rows: list[dict[str, Any]] = []
    started = time.time()

    def flush_snapshot(force: bool = False) -> None:
        completed = len(rendered_indexed) + len(skipped_rows)
        if not force and (completed == 0 or completed % max(1, args.refresh_every) != 0):
            return
        rows = [row for _index, row in sorted(rendered_indexed, key=lambda item: item[0])]
        payload = build_payload(
            input_manifest=args.prepared_manifest,
            hdri_manifest=args.hdri_manifest,
            output_root=args.output_root,
            hdri_mode=args.hdri_mode,
            selected_count=len(records),
            rendered_rows=rows,
            skipped_rows=skipped_rows,
        )
        payload["elapsed_seconds"] = round(time.time() - started, 3)
        write_json(output_manifest, payload)
        write_csv(output_csv, rows)
        write_summary(summary_md, payload)
        print(
            "[polyhaven stress] "
            f"completed={completed}/{len(records)} rendered={len(rows)} skipped={len(skipped_rows)}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    flush_snapshot(force=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_record = {
            executor.submit(
                render_record,
                record,
                hdri_by_stratum=hdri_by_stratum,
                output_root=args.output_root,
                blender_bin=args.blender_bin,
                cuda_device_index=args.cuda_device_index,
                resolution=args.resolution,
                cycles_samples=args.cycles_samples,
                hdri_mode=args.hdri_mode,
            ): (index, record)
            for index, record in enumerate(records)
        }
        for future in as_completed(future_to_record):
            index, record = future_to_record[future]
            try:
                rendered_indexed.append((index, future.result()))
            except Exception as exc:  # noqa: BLE001
                skipped_rows.append(
                    {
                        "object_id": record.get("object_id", ""),
                        "source_name": record.get("source_name", ""),
                        "reason": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
            flush_snapshot()
    flush_snapshot(force=True)


if __name__ == "__main__":
    main()
