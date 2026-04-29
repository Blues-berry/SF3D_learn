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
from sf3d.material_refine.training.preview import select_variant_balanced_records  # noqa: E402

DEFAULT_VIEWS = ["front_studio", "three_quarter_indoor", "side_neon"]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_path_component(value: Any, *, default: str = "unknown", max_length: int = 96) -> str:
    text = default if value is None else str(value).strip()
    if not text:
        text = default
    safe_chars = []
    last_was_separator = False
    for char in text:
        if char.isascii() and (char.isalnum() or char in {"-", "_", "."}):
            safe_chars.append(char)
            last_was_separator = False
        elif not last_was_separator:
            safe_chars.append("_")
            last_was_separator = True
    safe = "".join(safe_chars).strip("._")
    if not safe:
        safe = default
    return safe[:max_length].strip("._") or default


def first_case_identity(*values: Any) -> str:
    for value in values:
        text = "" if value is None else str(value).strip()
        if text and text.lower() not in {"none", "null", "unknown"}:
            return text
    return "unknown"


def row_case_key(row: dict[str, Any]) -> str:
    case_id = row.get("case_id")
    if case_id:
        return safe_path_component(case_id, max_length=220)
    suffix = first_case_identity(row.get("pair_id"), row.get("prior_variant_id"), row.get("target_bundle_id"))
    return "__".join(
        [
            safe_path_component(row.get("object_id")),
            safe_path_component(row.get("prior_variant_type")),
            safe_path_component(suffix, max_length=128),
        ]
    )


def row_case_label(row: dict[str, Any]) -> str:
    return (
        f"{row.get('object_id')} | "
        f"pair={row.get('pair_id', 'unknown')} | "
        f"variant={row.get('prior_variant_type', 'unknown')} | "
        f"quality={row.get('prior_quality_bin', 'unknown')} | "
        f"role={row.get('training_role', 'unknown')}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export validation panels comparing input prior material maps against refined maps.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-panels", type=int, default=64)
    parser.add_argument("--panel-size", type=int, default=192)
    parser.add_argument(
        "--selection-mode",
        choices=["balanced_by_variant", "effect_showcase", "balanced", "first", "best_gain", "worst_regression"],
        default="balanced_by_variant",
    )
    parser.add_argument("--report-to", choices=["none", "wandb"], default="wandb")
    parser.add_argument("--tracker-project-name", type=str, default="stable-fast-3d-material-refine")
    parser.add_argument("--tracker-run-name", type=str, default=None)
    parser.add_argument("--tracker-group", type=str, default="material-refine-validation-panels")
    parser.add_argument("--tracker-tags", type=str, default="material-refine,validation,comparison")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-log-artifacts", type=parse_bool, default=True)
    parser.add_argument("--wandb-max-panel-images", type=int, default=8)
    parser.add_argument("--wandb-log-panel-table", type=parse_bool, default=False)
    parser.add_argument("--wandb-log-full-panel-artifact", type=parse_bool, default=False)
    return parser


def load_records(manifest_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(manifest_path.read_text())
    records = payload.get("records") or payload.get("objects") or payload.get("rows") or []
    return payload, {str(r.get("object_id")): r for r in records if isinstance(r, dict) and r.get("object_id")}


def get_value(row: dict[str, Any], record: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    value = row.get(key)
    if value not in (None, ""):
        return value
    if record is not None:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def prior_display_label(row: dict[str, Any], record: dict[str, Any] | None) -> str:
    source = str(get_value(row, record, "prior_source_type", get_value(row, record, "prior_generation_mode", "")) or "").lower().strip()
    generation = str(get_value(row, record, "prior_generation_mode", "") or "").lower().strip()
    mode = str(get_value(row, record, "prior_mode", "") or "").lower().strip()
    prior_label = str(get_value(row, record, "prior_label", "") or "").lower().strip()
    has_prior = str(get_value(row, record, "has_material_prior", "") or "").lower().strip()
    mapping = {
        "true_sf3d_rm_texture": "SF3D RM Texture",
        "sf3d_rm_texture": "SF3D RM Texture",
        "true_sf3d_scalar_broadcast": "SF3D Scalar Prior",
        "sf3d_scalar_broadcast": "SF3D Scalar Prior",
        "sf3d_missing_rm_fallback": "Fallback Prior",
        "external_asset_rm_texture": "External Asset Prior",
        "external_asset_scalar_broadcast": "External Scalar Prior",
        "synthetic_degraded_prior": "Synthetic Prior",
        "synthetic_degraded_from_target": "Synthetic Prior",
        "fallback_default": "Fallback Prior",
        "no_prior_placeholder": "No-prior Default",
    }
    for key in (source, generation):
        if key in mapping:
            return mapping[key]
    if has_prior in {"false", "0", "none", "no"} or mode == "none" or prior_label in {"without_prior", "no_prior", "none"}:
        return "No-prior Default"
    if mode == "scalar_rm":
        return "Input Scalar Prior"
    if mode == "uv_rm":
        return "Input UV Prior"
    return "Input Prior"


def baseline_metadata(row: dict[str, Any], record: dict[str, Any] | None) -> str:
    prior_source = get_value(row, record, "prior_source_type", get_value(row, record, "prior_generation_mode", "unknown"))
    prior_generation = get_value(row, record, "prior_generation_mode", prior_source)
    target_source = get_value(row, record, "target_source_type", "unknown")
    target_identity = get_value(row, record, "target_prior_identity", "unknown")
    prior_label = prior_display_label(row, record)
    return (
        f"baseline_label={prior_label} | prior_label={prior_label} | "
        f"prior_mode={get_value(row, record, 'prior_mode', 'unknown')} | "
        f"prior_source_type={prior_source} | prior_generation_mode={prior_generation} | "
        f"target_source_type={target_source} | target_prior_identity={target_identity}"
    )


def resolve_record_path(manifest_path: Path, manifest_payload: dict[str, Any], record: dict[str, Any], value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    bundle = record.get("bundle_root") or record.get("canonical_bundle_root") or manifest_payload.get("canonical_bundle_root") or manifest_payload.get("bundle_root")
    if bundle:
        root = Path(str(bundle))
        if not root.is_absolute():
            root = manifest_path.parent / root
        candidate = root / path
        if candidate.exists():
            return candidate
    return manifest_path.parent / path


def load_view_names(manifest_path: Path, manifest_payload: dict[str, Any], record: dict[str, Any]) -> list[str]:
    path = resolve_record_path(manifest_path, manifest_payload, record, record.get("canonical_views_json"))
    if path is None or not path.exists():
        return list(DEFAULT_VIEWS)
    try:
        payload = json.loads(path.read_text())
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
    if path is None or not Path(path).exists():
        return Image.new("RGB", (size, size), (28, 32, 38))
    image = Image.open(path)
    if mode == "L":
        image = image.convert("L").convert("RGB")
    else:
        image = image.convert("RGBA")
        bg = Image.new("RGBA", image.size, (24, 27, 32, 255))
        bg.alpha_composite(image)
        image = bg.convert("RGB")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def finite_float(value: Any) -> float | None:
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
    return float(np.mean(values)) if values else None


def label_image(image: Image.Image, label: str) -> Image.Image:
    header_height = 28
    canvas = Image.new("RGB", (image.width, image.height + header_height), (12, 16, 22))
    canvas.paste(image, (0, header_height))
    ImageDraw.Draw(canvas).text((8, 7), label, fill=(236, 242, 248))
    return canvas


def delta_image(path_a: str | Path | None, path_b: str | Path | None, *, size: int) -> Image.Image:
    if path_a is None or path_b is None or not Path(path_a).exists() or not Path(path_b).exists():
        return Image.new("RGB", (size, size), (28, 32, 38))
    arr_a = np.asarray(Image.open(path_a).convert("L")).astype(np.float32) / 255.0
    arr_b = np.asarray(Image.open(path_b).convert("L")).astype(np.float32) / 255.0
    delta = np.clip(np.abs(arr_b - arr_a) * 4.0, 0.0, 1.0)
    heat = np.zeros((*delta.shape, 3), dtype=np.uint8)
    heat[..., 0] = (delta * 255).astype(np.uint8)
    heat[..., 1] = (np.sqrt(delta) * 120).astype(np.uint8)
    heat[..., 2] = ((1.0 - delta) * 45).astype(np.uint8)
    return Image.fromarray(heat, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)


def improvement_image(
    prior_path: str | Path | None,
    pred_path: str | Path | None,
    target_path: str | Path | None,
    *,
    size: int,
) -> Image.Image:
    paths = [prior_path, pred_path, target_path]
    if any(path is None or not Path(path).exists() for path in paths):
        return Image.new("RGB", (size, size), (28, 32, 38))
    prior = np.asarray(Image.open(prior_path).convert("L")).astype(np.float32) / 255.0
    pred = np.asarray(Image.open(pred_path).convert("L")).astype(np.float32) / 255.0
    target = np.asarray(Image.open(target_path).convert("L")).astype(np.float32) / 255.0
    gain = np.clip((np.abs(prior - target) - np.abs(pred - target)) * 5.0, -1.0, 1.0)
    heat = np.zeros((*gain.shape, 3), dtype=np.uint8)
    positive = np.clip(gain, 0.0, 1.0)
    negative = np.clip(-gain, 0.0, 1.0)
    heat[..., 0] = (negative * 230.0 + 34.0).astype(np.uint8)
    heat[..., 1] = (positive * 230.0 + 34.0).astype(np.uint8)
    heat[..., 2] = ((1.0 - np.maximum(positive, negative)) * 70.0).astype(np.uint8)
    return Image.fromarray(heat, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)


def view_space_tiles(paths: dict[str, Any], size: int) -> list[Image.Image]:
    keys = [
        ("reference_rgb", "Input RGB view", "RGB"),
        ("sampled_input_prior_view_roughness", "View Prior rough", "L"),
        ("sampled_input_prior_view_metallic", "View Prior metal", "L"),
        ("sampled_gt_view_roughness", "View GT rough", "L"),
        ("sampled_gt_view_metallic", "View GT metal", "L"),
        ("sampled_pred_view_roughness", "View Pred rough", "L"),
        ("sampled_pred_view_metallic", "View Pred metal", "L"),
        ("prior_gt_view_error", "|Prior-GT| view", "RGB"),
        ("pred_gt_view_error", "|Pred-GT| view", "RGB"),
        ("view_mask", "View mask", "L"),
        ("view_uv_valid", "view_uv valid", "L"),
    ]
    if not any(paths.get(key) for key, _, _ in keys):
        return []
    return [
        label_image(load_image(paths.get(key), size=size, mode=mode), label)
        for key, label, mode in keys
    ]


def view_rgb_paths(manifest_path: Path, manifest_payload: dict[str, Any], record: dict[str, Any]) -> list[tuple[str, Path | None]]:
    root = resolve_record_path(manifest_path, manifest_payload, record, record.get("canonical_buffer_root"))
    paths: list[tuple[str, Path | None]] = []
    for name in load_view_names(manifest_path, manifest_payload, record)[:3]:
        candidate = None
        if root is not None:
            rgba = root / name / "rgba.png"
            rgb = root / name / "rgb.png"
            candidate = rgba if rgba.exists() else rgb if rgb.exists() else None
        paths.append((name, candidate))
    while len(paths) < 3:
        paths.append((f"view_{len(paths)}", None))
    return paths


def choose_object_rows(
    rows: list[dict[str, Any]],
    max_panels: int,
    *,
    selection_mode: str,
) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_key = row_case_key(row)
        if not case_key:
            continue
        current = by_case.get(case_key)
        if current is None or float(row.get("improvement_total", 0.0)) < float(current.get("improvement_total", 0.0)):
            by_case[case_key] = row
    representatives = list(by_case.values())
    if max_panels <= 0:
        return representatives

    def gain(row: dict[str, Any]) -> float:
        return float(row.get("gain_total", row.get("improvement_total", 0.0)) or 0.0)

    if selection_mode == "first":
        return representatives[:max_panels]
    if selection_mode == "best_gain":
        return sorted(representatives, key=gain, reverse=True)[:max_panels]
    if selection_mode == "worst_regression":
        return sorted(representatives, key=gain)[:max_panels]
    return select_variant_balanced_records(
        representatives,
        max_count=max_panels,
        mode=selection_mode,
    )


def build_panel(row: dict[str, Any], record: dict[str, Any] | None, manifest_path: Path, manifest_payload: dict[str, Any], size: int) -> Image.Image:
    paths = row.get("paths", {})
    tiles: list[Image.Image] = []
    if record is not None:
        for name, path in view_rgb_paths(manifest_path, manifest_payload, record):
            tiles.append(label_image(load_image(path, size=size), f"Input RGB {name}"))
    else:
        for idx in range(3):
            tiles.append(label_image(load_image(None, size=size), f"Input RGB view_{idx}"))

    prior_label = prior_display_label(row, record)
    tiles.extend([
        label_image(load_image(paths.get("baseline_roughness"), size=size, mode="L"), "Input Prior rough"),
        label_image(load_image(paths.get("baseline_metallic"), size=size, mode="L"), "Input Prior metal"),
        label_image(load_image(paths.get("confidence"), size=size, mode="L"), "Prior/target conf"),
        label_image(load_image(paths.get("prior_reliability"), size=size, mode="L"), "Prior reliability"),
        label_image(load_image(paths.get("change_gate"), size=size, mode="L"), "Change gate"),
        label_image(load_image(paths.get("rm_init_roughness"), size=size, mode="L"), "RM init rough"),
        label_image(load_image(paths.get("rm_init_metallic"), size=size, mode="L"), "RM init metal"),
        label_image(load_image(paths.get("bootstrap_roughness"), size=size, mode="L"), "Bootstrap rough"),
        label_image(load_image(paths.get("bootstrap_metallic"), size=size, mode="L"), "Bootstrap metal"),
        label_image(load_image(paths.get("target_roughness"), size=size, mode="L"), "GT rough"),
        label_image(load_image(paths.get("target_metallic"), size=size, mode="L"), "GT metal"),
        label_image(load_image(paths.get("refined_roughness"), size=size, mode="L"), "Pred rough"),
        label_image(load_image(paths.get("refined_metallic"), size=size, mode="L"), "Pred metal"),
        label_image(load_image(paths.get("delta_abs"), size=size, mode="L"), "|Pred-Init| delta"),
        label_image(delta_image(paths.get("baseline_roughness"), paths.get("target_roughness"), size=size), "|Prior-GT| rough"),
        label_image(delta_image(paths.get("refined_roughness"), paths.get("target_roughness"), size=size), "|Pred-GT| rough"),
        label_image(improvement_image(paths.get("baseline_roughness"), paths.get("refined_roughness"), paths.get("target_roughness"), size=size), "Gain rough"),
        label_image(delta_image(paths.get("refined_roughness"), paths.get("baseline_roughness"), size=size), "|Pred-Prior| rough"),
        label_image(delta_image(paths.get("baseline_metallic"), paths.get("target_metallic"), size=size), "|Prior-GT| metal"),
        label_image(delta_image(paths.get("refined_metallic"), paths.get("target_metallic"), size=size), "|Pred-GT| metal"),
        label_image(improvement_image(paths.get("baseline_metallic"), paths.get("refined_metallic"), paths.get("target_metallic"), size=size), "Gain metal"),
        label_image(delta_image(paths.get("refined_metallic"), paths.get("baseline_metallic"), size=size), "|Pred-Prior| metal"),
    ])
    tiles.extend(view_space_tiles(paths, size))

    columns = 5
    rows = int(np.ceil(len(tiles) / columns))
    header_height = 76
    tile_width, tile_height = tiles[0].size
    canvas = Image.new("RGB", (columns * tile_width, header_height + rows * tile_height), (9, 13, 19))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{row_case_label(row)} | generator={row.get('generator_id', 'unknown')} | "
        f"source={row.get('source_name')} | material={row.get('material_family', 'unknown')} | "
        f"{row.get('view_name')} | {prior_label}"
    )
    regression_flag = bool((finite_float(row.get("gain_total", row.get("improvement_total"))) or 0.0) < 0.0)
    uv_gain = finite_float(row.get("uv_gain"))
    view_gain = finite_float(row.get("view_gain", row.get("gain_total", row.get("improvement_total"))))
    uv_improved = bool(uv_gain is not None and uv_gain > 0.0)
    view_regressed = bool(view_gain is not None and view_gain < 0.0)
    metrics = (
        f"{baseline_metadata(row, record)} | "
        f"input_prior_total={fmt_metric(row.get('input_prior_total_mae', row.get('baseline_total_mae')))}  "
        f"refined_total={fmt_metric(row.get('refined_total_mae'))}  "
        f"gain={fmt_metric(row.get('gain_total', row.get('improvement_total')))}  "
        f"regression_flag={regression_flag}  "
        f"uv_gain={fmt_metric(uv_gain)} view_gain={fmt_metric(view_gain)} "
        f"uv_improved={uv_improved} view_regressed={view_regressed}"
    )
    draw.text((14, 14), title, fill=(242, 247, 255))
    draw.text((14, 42), metrics, fill=(180, 205, 224))
    for idx, tile in enumerate(tiles):
        canvas.paste(tile, ((idx % columns) * tile_width, header_height + (idx // columns) * tile_height))
    return canvas


def build_html(rows: list[dict[str, Any]], panel_paths: list[Path], output_dir: Path, records: dict[str, dict[str, Any]]) -> Path:
    cards = []
    for row, panel_path in zip(rows, panel_paths):
        record = records.get(str(row.get("object_id")))
        uv_gain = finite_float(row.get("uv_gain"))
        view_gain = finite_float(row.get("view_gain", row.get("gain_total", row.get("improvement_total"))))
        cards.append("\n".join([
            "<section class='card'>",
            f"<h2>{row_case_label(row)}</h2>",
            "<div class='meta'>" + f"generator={row.get('generator_id', 'unknown')} | source={row.get('source_name')} | material={row.get('material_family', 'unknown')} | {baseline_metadata(row, record)} | gain={fmt_metric(row.get('gain_total', row.get('improvement_total')))} | uv_gain={fmt_metric(uv_gain)} | view_gain={fmt_metric(view_gain)}" + "</div>",
            f"<img src='{panel_path.name}' alt='{row.get('object_id')} comparison panel'>",
            "</section>",
        ]))
    html_path = output_dir / "validation_comparison_index.html"
    html_path.write_text("\n".join([
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Material Validation Comparison Panels</title>",
        "<style>body{margin:0;background:#081019;color:#eef5ff;font-family:Arial,sans-serif}.wrap{max-width:1400px;margin:0 auto;padding:28px}.card{background:#111b27;border:1px solid #263648;border-radius:18px;padding:18px;margin:18px 0}.meta{color:#a9c4d8;margin:6px 0 14px}img{width:100%;border-radius:12px;background:#090d13}</style></head><body><main class='wrap'>",
        "<h1>Input Prior vs Material Refiner Validation Panels</h1>",
        "<p>Each panel shows canonical RGB views, UV RM maps, GT/target maps, predicted maps, error maps, and a view-space row when the eval exported sampled view artifacts. The prior label is source-aware and should only say SF3D when the atlas is traceable to an SF3D output.</p>",
        *cards,
        "</main></body></html>",
    ]), encoding="utf-8")
    return html_path


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload, records = load_records(args.manifest)
    metrics_rows = json.loads(args.metrics.read_text())
    selected_rows = choose_object_rows(
        metrics_rows,
        args.max_panels,
        selection_mode=str(args.selection_mode),
    )
    panel_paths = []
    for row in selected_rows:
        object_id = str(row["object_id"])
        panel = build_panel(row, records.get(object_id), args.manifest, manifest_payload, args.panel_size)
        panel_path = args.output_dir / f"{row_case_key(row)}.png"
        panel.save(panel_path)
        panel_paths.append(panel_path)

    missing_baseline_rows = [row.get("object_id") for row in selected_rows if finite_float(row.get("baseline_total_mae")) is None]
    summary = {
        "manifest": str(args.manifest.resolve()),
        "metrics": str(args.metrics.resolve()),
        "selection_mode": str(args.selection_mode),
        "output_dir": str(args.output_dir.resolve()),
        "panels": len(panel_paths),
        "wandb_policy": {
            "upload_per_object_table": bool(args.wandb_log_panel_table),
            "max_panel_images": int(args.wandb_max_panel_images),
            "full_panel_artifact": bool(args.wandb_log_full_panel_artifact),
            "details_saved_locally": True,
        },
        "metrics_mean": {
            "baseline_total_mae": mean_metric(selected_rows, "baseline_total_mae"),
            "input_prior_total_mae": mean_metric(selected_rows, "input_prior_total_mae"),
            "refined_total_mae": mean_metric(selected_rows, "refined_total_mae"),
            "gain_total": mean_metric(selected_rows, "gain_total"),
            "improvement_total": mean_metric(selected_rows, "improvement_total"),
        },
        "warnings": [f"missing_baseline_total_mae:{object_id}" for object_id in missing_baseline_rows],
        "selected_object_ids": [row.get("object_id") for row in selected_rows],
        "selected_case_ids": [row_case_key(row) for row in selected_rows],
    }
    summary_path = args.output_dir / "validation_comparison_summary.json"
    summary_path.write_text(json.dumps(make_json_serializable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = build_html(selected_rows, panel_paths, args.output_dir, records)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    run = maybe_init_wandb(
        enabled=args.report_to == "wandb",
        project=args.tracker_project_name,
        job_type="validation-panels",
        config=make_json_serializable(vars(args)),
        mode=args.wandb_mode,
        name=args.tracker_run_name or args.output_dir.name,
        group=args.tracker_group,
        tags=args.tracker_tags,
        dir_path=args.wandb_dir,
    )
    if run is not None:
        run.log({
            "validation_panels/count": len(panel_paths),
            "validation_panels/uploaded_panel_count": min(len(panel_paths), max(int(args.wandb_max_panel_images), 0)),
            "validation_panels/input_prior_total_mae_mean": summary["metrics_mean"]["input_prior_total_mae"] or summary["metrics_mean"]["baseline_total_mae"],
            "validation_panels/refined_total_mae_mean": summary["metrics_mean"]["refined_total_mae"],
            "validation_panels/gain_total_mean": summary["metrics_mean"]["gain_total"] or summary["metrics_mean"]["improvement_total"],
            "validation_panels/missing_baseline_total_mae_count": len(missing_baseline_rows),
        })
        if wandb is not None and int(args.wandb_max_panel_images) > 0:
            sample_panels = [
                wandb.Image(str(panel_path), caption=f"{row_case_label(row)} | Prior={fmt_metric(row.get('input_prior_total_mae', row.get('baseline_total_mae')))} Pred={fmt_metric(row.get('refined_total_mae'))} gain={fmt_metric(row.get('gain_total', row.get('improvement_total')))} | {prior_display_label(row, records.get(str(row.get('object_id'))))}")
                for row, panel_path in zip(selected_rows, panel_paths)
            ][: max(int(args.wandb_max_panel_images), 0)]
            run.log({"validation_panels/sample_panels": sample_panels})
        if wandb is not None and bool(args.wandb_log_panel_table):
            table = wandb.Table(columns=["case_id", "object_id", "pair_id", "prior_variant_type", "prior_quality_bin", "training_role", "generator_id", "source_name", "prior_source_type", "target_source_type", "baseline_label", "prior_label", "baseline_total_mae", "input_prior_total_mae", "refined_total_mae", "gain_total", "panel"])
            for row, panel_path in zip(selected_rows, panel_paths):
                record = records.get(str(row.get("object_id")))
                table.add_data(row_case_key(row), row.get("object_id"), row.get("pair_id"), row.get("prior_variant_type"), row.get("prior_quality_bin"), row.get("training_role"), row.get("generator_id", "unknown"), row.get("source_name"), get_value(row, record, "prior_source_type", "unknown"), get_value(row, record, "target_source_type", "unknown"), prior_display_label(row, record), row.get("prior_label"), row.get("baseline_total_mae"), row.get("input_prior_total_mae", row.get("baseline_total_mae")), row.get("refined_total_mae"), row.get("gain_total", row.get("improvement_total")), wandb.Image(str(panel_path)))
            run.log({"validation_panels/table": table})
        if args.wandb_log_artifacts:
            artifact_paths = [summary_path, html_path]
            if bool(args.wandb_log_full_panel_artifact):
                artifact_paths.append(args.output_dir)
            log_path_artifact(run, name=f"{run.name}-validation-panels", artifact_type="validation-panels", paths=artifact_paths)
        run.finish()


if __name__ == "__main__":
    main()
