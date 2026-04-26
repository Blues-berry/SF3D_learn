from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.experiment import (  # noqa: E402
    log_path_artifact,
    make_json_serializable,
    maybe_init_wandb,
    wandb,
)

DEFAULT_VIEWS = ["front_studio", "three_quarter_indoor", "side_neon"]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export validation panels comparing SF3D baseline material maps against refined maps.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-panels", type=int, default=64)
    parser.add_argument("--panel-size", type=int, default=192)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="wandb")
    parser.add_argument("--tracker-project-name", type=str, default="stable-fast-3d-material-refine")
    parser.add_argument("--tracker-run-name", type=str, default=None)
    parser.add_argument("--tracker-group", type=str, default="material-refine-validation-panels")
    parser.add_argument("--tracker-tags", type=str, default="material-refine,validation,comparison")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-log-artifacts", type=parse_bool, default=True)
    parser.add_argument(
        "--wandb-max-panel-images",
        type=int,
        default=8,
        help="Only upload this many sample panels to W&B. All panels remain saved locally.",
    )
    parser.add_argument(
        "--wandb-log-panel-table",
        type=parse_bool,
        default=False,
        help="Legacy compatibility switch. Defaults off to avoid uploading per-object detailed panel tables.",
    )
    parser.add_argument(
        "--wandb-log-full-panel-artifact",
        type=parse_bool,
        default=False,
        help="If true, upload the full panel directory as an artifact. Defaults off; local files are the source of truth.",
    )
    return parser


def load_records(manifest_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(manifest_path.read_text())
    records = payload.get("records") or payload.get("objects") or payload.get("rows") or []
    return payload, {
        str(record.get("object_id")): record
        for record in records
        if isinstance(record, dict) and record.get("object_id")
    }


def resolve_record_path(
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    record: dict[str, Any],
    value: str | None,
) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    bundle_root_value = (
        record.get("bundle_root")
        or record.get("canonical_bundle_root")
        or manifest_payload.get("canonical_bundle_root")
        or manifest_payload.get("bundle_root")
    )
    if bundle_root_value:
        bundle_root = Path(str(bundle_root_value))
        if not bundle_root.is_absolute():
            bundle_root = manifest_path.parent / bundle_root
        candidate = bundle_root / path
        if candidate.exists():
            return candidate
    return manifest_path.parent / path


def load_view_names(manifest_path: Path, manifest_payload: dict[str, Any], record: dict[str, Any]) -> list[str]:
    views_path = resolve_record_path(
        manifest_path,
        manifest_payload,
        record,
        record.get("canonical_views_json"),
    )
    if views_path is None or not views_path.exists():
        return list(DEFAULT_VIEWS)
    try:
        payload = json.loads(views_path.read_text())
    except Exception:
        return list(DEFAULT_VIEWS)
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            names = [str(item.get("name")) for item in payload if item.get("name")]
        else:
            names = [str(item) for item in payload]
        return names or list(DEFAULT_VIEWS)
    return list(DEFAULT_VIEWS)


def load_image(path: str | Path | None, *, size: int, mode: str = "RGB") -> Image.Image:
    if path is None:
        return Image.new("RGB", (size, size), (28, 32, 38))
    path = Path(path)
    if not path.exists():
        return Image.new("RGB", (size, size), (28, 32, 38))
    image = Image.open(path)
    if mode == "L":
        image = image.convert("L").convert("RGB")
    else:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (24, 27, 32, 255))
        background.alpha_composite(image)
        image = background.convert("RGB")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt_metric(value: Any) -> str:
    number = finite_float(value)
    return "N/A" if number is None else f"{number:.5f}"


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [finite_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def label_image(image: Image.Image, label: str) -> Image.Image:
    header_height = 28
    canvas = Image.new("RGB", (image.width, image.height + header_height), (12, 16, 22))
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, fill=(236, 242, 248))
    return canvas


def delta_image(path_a: str | Path | None, path_b: str | Path | None, *, size: int) -> Image.Image:
    if path_a is None or path_b is None or not Path(path_a).exists() or not Path(path_b).exists():
        return Image.new("RGB", (size, size), (28, 32, 38))
    arr_a = np.asarray(Image.open(path_a).convert("L")).astype(np.float32) / 255.0
    arr_b = np.asarray(Image.open(path_b).convert("L")).astype(np.float32) / 255.0
    delta = np.abs(arr_b - arr_a)
    delta = np.clip(delta * 4.0, 0.0, 1.0)
    heat = np.zeros((*delta.shape, 3), dtype=np.uint8)
    heat[..., 0] = (delta * 255).astype(np.uint8)
    heat[..., 1] = (np.sqrt(delta) * 120).astype(np.uint8)
    heat[..., 2] = ((1.0 - delta) * 45).astype(np.uint8)
    return Image.fromarray(heat, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)


def view_rgb_paths(
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    record: dict[str, Any],
) -> list[tuple[str, Path | None]]:
    buffer_root = resolve_record_path(
        manifest_path,
        manifest_payload,
        record,
        record.get("canonical_buffer_root"),
    )
    paths = []
    for view_name in load_view_names(manifest_path, manifest_payload, record)[:3]:
        candidate = None
        if buffer_root is not None:
            rgba = buffer_root / view_name / "rgba.png"
            rgb = buffer_root / view_name / "rgb.png"
            candidate = rgba if rgba.exists() else rgb if rgb.exists() else None
        paths.append((view_name, candidate))
    while len(paths) < 3:
        paths.append((f"view_{len(paths)}", None))
    return paths


def choose_object_rows(rows: list[dict[str, Any]], max_panels: int) -> list[dict[str, Any]]:
    by_object: dict[str, dict[str, Any]] = {}
    for row in rows:
        object_id = str(row.get("object_id", ""))
        if not object_id:
            continue
        current = by_object.get(object_id)
        if current is None or float(row.get("improvement_total", 0.0)) < float(
            current.get("improvement_total", 0.0)
        ):
            by_object[object_id] = row
    ranked = sorted(
        by_object.values(),
        key=lambda row: (
            float(row.get("improvement_total", 0.0)),
            -float(row.get("refined_total_mae", 0.0)),
        ),
    )
    return ranked[:max_panels]


def build_panel(
    *,
    row: dict[str, Any],
    record: dict[str, Any] | None,
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    size: int,
) -> Image.Image:
    object_id = str(row["object_id"])
    paths = row.get("paths", {})
    tiles: list[Image.Image] = []
    if record is not None:
        for view_name, view_path in view_rgb_paths(manifest_path, manifest_payload, record):
            tiles.append(label_image(load_image(view_path, size=size), f"SF3D RGB {view_name}"))
    else:
        for index in range(3):
            tiles.append(label_image(load_image(None, size=size), f"SF3D RGB view_{index}"))

    tiles.extend(
        [
            label_image(load_image(paths.get("baseline_roughness"), size=size, mode="L"), "SF3D roughness"),
            label_image(load_image(paths.get("baseline_metallic"), size=size, mode="L"), "SF3D metallic"),
            label_image(load_image(paths.get("refined_roughness"), size=size, mode="L"), "R refined roughness"),
            label_image(load_image(paths.get("refined_metallic"), size=size, mode="L"), "R refined metallic"),
            label_image(
                delta_image(paths.get("baseline_roughness"), paths.get("refined_roughness"), size=size),
                "|delta roughness|",
            ),
            label_image(
                delta_image(paths.get("baseline_metallic"), paths.get("refined_metallic"), size=size),
                "|delta metallic|",
            ),
            label_image(load_image(paths.get("confidence"), size=size, mode="L"), "target confidence"),
        ]
    )

    columns = 5
    rows = int(np.ceil(len(tiles) / columns))
    header_height = 76
    tile_width, tile_height = tiles[0].size
    canvas = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * tile_height),
        (9, 13, 19),
    )
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{object_id} | generator={row.get('generator_id', 'unknown')} | "
        f"source={row.get('source_name')} | {row.get('prior_label')} | {row.get('view_name')}"
    )
    metrics = (
        f"baseline_total={fmt_metric(row.get('baseline_total_mae'))}  "
        f"refined_total={fmt_metric(row.get('refined_total_mae'))}  "
        f"improvement={fmt_metric(row.get('improvement_total'))}"
    )
    draw.text((14, 14), title, fill=(242, 247, 255))
    draw.text((14, 42), metrics, fill=(180, 205, 224))
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_width
        y = header_height + (index // columns) * tile_height
        canvas.paste(tile, (x, y))
    return canvas


def build_html(rows: list[dict[str, Any]], panel_paths: list[Path], output_dir: Path) -> Path:
    html_path = output_dir / "validation_comparison_index.html"
    cards = []
    for row, panel_path in zip(rows, panel_paths):
        cards.append(
            "\n".join(
                [
                    "<section class='card'>",
                    f"<h2>{row.get('object_id')}</h2>",
                    "<div class='meta'>"
                    f"generator={row.get('generator_id', 'unknown')} | "
                    f"source={row.get('source_name')} | prior={row.get('prior_label')} | "
                    f"improvement={fmt_metric(row.get('improvement_total'))}"
                    "</div>",
                    f"<img src='{panel_path.name}' alt='{row.get('object_id')} comparison panel'>",
                    "</section>",
                ]
            )
        )
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Material Validation Comparison Panels</title>",
                "<style>",
                "body{margin:0;background:#081019;color:#eef5ff;font-family:Arial,sans-serif}",
                ".wrap{max-width:1400px;margin:0 auto;padding:28px}",
                ".card{background:#111b27;border:1px solid #263648;border-radius:18px;padding:18px;margin:18px 0}",
                ".meta{color:#a9c4d8;margin:6px 0 14px}",
                "img{width:100%;border-radius:12px;background:#090d13}",
                "</style></head><body><main class='wrap'>",
                "<h1>SF3D Baseline vs Material Refiner Validation Panels</h1>",
                "<p>Each panel shows original SF3D RGB views, baseline RM atlas maps, refined RM atlas maps, deltas, and target confidence.</p>",
                *cards,
                "</main></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return html_path


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload, records = load_records(args.manifest)
    metrics_rows = json.loads(args.metrics.read_text())
    selected_rows = choose_object_rows(metrics_rows, args.max_panels)
    missing_baseline_rows = [
        row.get("object_id")
        for row in selected_rows
        if finite_float(row.get("baseline_total_mae")) is None
    ]

    panel_paths = []
    for row in selected_rows:
        object_id = str(row["object_id"])
        panel = build_panel(
            row=row,
            record=records.get(object_id),
            manifest_path=args.manifest,
            manifest_payload=manifest_payload,
            size=args.panel_size,
        )
        panel_path = output_dir / f"{object_id}.png"
        panel.save(panel_path)
        panel_paths.append(panel_path)

    summary = {
        "manifest": str(args.manifest.resolve()),
        "metrics": str(args.metrics.resolve()),
        "output_dir": str(output_dir.resolve()),
        "panels": len(panel_paths),
        "wandb_policy": {
            "upload_per_object_table": bool(args.wandb_log_panel_table),
            "max_panel_images": int(args.wandb_max_panel_images),
            "full_panel_artifact": bool(args.wandb_log_full_panel_artifact),
            "details_saved_locally": True,
        },
        "metrics_mean": {
            "baseline_total_mae": mean_metric(selected_rows, "baseline_total_mae"),
            "refined_total_mae": mean_metric(selected_rows, "refined_total_mae"),
            "improvement_total": mean_metric(selected_rows, "improvement_total"),
        },
        "warnings": [
            f"missing_baseline_total_mae:{object_id}"
            for object_id in missing_baseline_rows
        ],
        "selected_object_ids": [row.get("object_id") for row in selected_rows],
    }
    summary_path = output_dir / "validation_comparison_summary.json"
    summary_path.write_text(
        json.dumps(make_json_serializable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    html_path = build_html(selected_rows, panel_paths, output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    run = maybe_init_wandb(
        enabled=args.report_to == "wandb",
        project=args.tracker_project_name,
        job_type="validation-panels",
        config=make_json_serializable(vars(args)),
        mode=args.wandb_mode,
        name=args.tracker_run_name or output_dir.name,
        group=args.tracker_group,
        tags=args.tracker_tags,
        dir_path=args.wandb_dir,
    )
    if run is not None:
        run.log(
            {
                "validation_panels/count": len(panel_paths),
                "validation_panels/uploaded_panel_count": min(
                    len(panel_paths),
                    max(int(args.wandb_max_panel_images), 0),
                ),
                "validation_panels/baseline_total_mae_mean": summary["metrics_mean"]["baseline_total_mae"],
                "validation_panels/refined_total_mae_mean": summary["metrics_mean"]["refined_total_mae"],
                "validation_panels/improvement_total_mean": summary["metrics_mean"]["improvement_total"],
                "validation_panels/missing_baseline_total_mae_count": len(missing_baseline_rows),
            }
        )
        if wandb is not None and int(args.wandb_max_panel_images) > 0:
            sample_panels = [
                wandb.Image(
                    str(panel_path),
                    caption=(
                        f"{row.get('object_id')} | "
                        f"SF3D={fmt_metric(row.get('baseline_total_mae'))} "
                        f"Pred={fmt_metric(row.get('refined_total_mae'))} "
                        f"gain={fmt_metric(row.get('improvement_total'))}"
                    ),
                )
                for row, panel_path in zip(selected_rows, panel_paths)
            ][: max(int(args.wandb_max_panel_images), 0)]
            run.log({"validation_panels/sample_panels": sample_panels})
        if wandb is not None and bool(args.wandb_log_panel_table):
            table = wandb.Table(
                columns=[
                    "object_id",
                    "generator_id",
                    "source_name",
                    "prior_label",
                    "baseline_total_mae",
                    "refined_total_mae",
                    "improvement_total",
                    "panel",
                ]
            )
            for row, panel_path in zip(selected_rows, panel_paths):
                table.add_data(
                    row.get("object_id"),
                    row.get("generator_id", "unknown"),
                    row.get("source_name"),
                    row.get("prior_label"),
                    row.get("baseline_total_mae"),
                    row.get("refined_total_mae"),
                    row.get("improvement_total"),
                    wandb.Image(str(panel_path)),
                )
            run.log({"validation_panels/table": table})
        if args.wandb_log_artifacts:
            artifact_paths = [summary_path, html_path]
            if bool(args.wandb_log_full_panel_artifact):
                artifact_paths.append(output_dir)
            log_path_artifact(
                run,
                name=f"{run.name}-validation-panels",
                artifact_type="validation-panels",
                paths=artifact_paths,
            )
        run.finish()


if __name__ == "__main__":
    main()
