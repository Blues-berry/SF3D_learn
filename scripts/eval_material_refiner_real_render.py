from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.training.preview import (
    format_prior_distribution,
    prior_variant_distribution,
    select_variant_balanced_records,
)


HDRI_PRESETS = {
    "hdri1": REPO_ROOT / "demo_files" / "hdri" / "studio_small_08_1k.hdr",
    "hdri2": REPO_ROOT / "demo_files" / "hdri" / "metro_noord_1k.hdr",
    "hdri3": REPO_ROOT / "demo_files" / "hdri" / "peppermint_powerplant_1k.hdr",
}
DEFAULT_BLENDER_BIN = "/4T/CXY/Neural_Gaffer_original/scripts/Objavarse_rendering/blender-3.2.2-linux-x64/blender"
_LPIPS_MODEL: Any | None = None
_LPIPS_DEVICE: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run a balanced Blender re-render benchmark on eval outputs.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument(
        "--selection-mode",
        choices=["balanced_by_variant", "random_balanced_by_variant", "first"],
        default="balanced_by_variant",
    )
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--hdri-preset", choices=sorted(HDRI_PRESETS), default="hdri1")
    parser.add_argument("--blender-bin", type=str, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--blender-script", type=Path, default=REPO_ROOT / "scripts" / "abo_material_relight_blender.py")
    parser.add_argument("--cuda-device-index", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--cycles-samples", type=int, default=32)
    parser.add_argument("--enable-lpips", type=str, default="true")
    parser.add_argument("--lpips-device", type=str, default=None)
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def make_json_serializable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): make_json_serializable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [make_json_serializable(value) for value in payload]
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_serializable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported_manifest_payload:{path}")


def safe_component(value: Any, default: str = "unknown") -> str:
    text = default if value is None else str(value).strip()
    if not text:
        text = default
    safe = []
    last_sep = False
    for char in text:
        if char.isascii() and (char.isalnum() or char in {"-", "_", "."}):
            safe.append(char)
            last_sep = False
        elif not last_sep:
            safe.append("_")
            last_sep = True
    return "".join(safe).strip("._") or default


def case_unique_key(row: dict[str, Any]) -> str:
    pair_id = str(row.get("pair_id") or "").strip()
    prior_variant_id = str(row.get("prior_variant_id") or "").strip()
    object_id = str(row.get("object_id") or "unknown")
    prior_variant_type = str(row.get("prior_variant_type") or "unknown")
    return "|".join([object_id, pair_id or prior_variant_id or object_id, prior_variant_type])


def aggregate_case_rows(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        grouped[case_unique_key(row)].append(row)
    cases: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        first = rows[0]
        improvements = [float(row.get("improvement_total") or 0.0) for row in rows]
        representative = sorted(
            rows,
            key=lambda item: (
                len(item.get("paths") or {}),
                abs(float(item.get("improvement_total") or 0.0)),
            ),
            reverse=True,
        )[0]
        cases.append(
            {
                "case_key": key,
                "object_id": first.get("object_id"),
                "pair_id": first.get("pair_id"),
                "prior_variant_id": first.get("prior_variant_id"),
                "prior_variant_type": first.get("prior_variant_type"),
                "prior_quality_bin": first.get("prior_quality_bin"),
                "training_role": first.get("training_role"),
                "split": first.get("split"),
                "gain_total": float(sum(improvements) / max(len(improvements), 1)),
                "view_name": representative.get("view_name"),
                "paths": representative.get("paths") or {},
            }
        )
    return cases


def resolve_case_render_paths(case: dict[str, Any], grouped_rows: list[dict[str, Any]]) -> dict[str, str]:
    required = [
        "baseline_roughness",
        "baseline_metallic",
        "refined_roughness",
        "refined_metallic",
    ]
    resolved = dict(case.get("paths") or {})
    for name in required:
        if name in resolved:
            continue
        for row in grouped_rows:
            row_paths = row.get("paths") or {}
            if name in row_paths:
                resolved[name] = row_paths[name]
                break
    missing = [name for name in required if name not in resolved]
    if missing:
        raise KeyError("missing_case_render_paths:" + ",".join(missing))
    return resolved


def choose_view_spec(record: dict[str, Any], preferred_view_name: str | None) -> dict[str, Any]:
    views_path = Path(str(record["canonical_views_json"]))
    views = json.loads(views_path.read_text(encoding="utf-8"))
    if not isinstance(views, list) or not views:
        raise RuntimeError(f"invalid_canonical_views_json:{views_path}")
    for view in views:
        if preferred_view_name and str(view.get("name")) == str(preferred_view_name):
            return view
    preferred_tokens = ("front_mid", "three_quarter_mid", "front_high")
    for token in preferred_tokens:
        for view in views:
            if str(view.get("name", "")).startswith(token):
                return view
    return views[0]


def load_rgba(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    return torch.from_numpy(np.moveaxis(array, -1, 0))


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (prediction - target).square() * mask
    denom = mask.sum().clamp_min(1.0) * float(prediction.shape[0])
    return float(diff.sum().item() / denom.item())


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def masked_global_ssim(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    weight = mask.expand_as(prediction)
    denom = weight.sum().clamp_min(1.0)
    mu_x = (prediction * weight).sum() / denom
    mu_y = (target * weight).sum() / denom
    var_x = ((prediction - mu_x).square() * weight).sum() / denom
    var_y = ((target - mu_y).square() * weight).sum() / denom
    cov_xy = ((prediction - mu_x) * (target - mu_y) * weight).sum() / denom
    c1 = 0.01**2
    c2 = 0.03**2
    value = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return float(value.clamp(-1.0, 1.0).item())


def compute_lpips(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, device: str) -> float | None:
    global _LPIPS_DEVICE, _LPIPS_MODEL
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    if _LPIPS_MODEL is None or _LPIPS_DEVICE != device:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device).eval()
        _LPIPS_DEVICE = device
    model = _LPIPS_MODEL
    pred = (prediction * mask).unsqueeze(0).to(device) * 2.0 - 1.0
    tgt = (target * mask).unsqueeze(0).to(device) * 2.0 - 1.0
    with torch.no_grad():
        distance = model(pred, tgt)
    return float(distance.mean().detach().cpu().item())


def save_error_heatmap(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, output_path: Path) -> None:
    error = (prediction - target).abs().mean(dim=0).numpy()
    error = error * mask.squeeze(0).numpy()
    error = np.clip(error * 4.0, 0.0, 1.0)
    heat = np.zeros((error.shape[0], error.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = (error * 255.0).round().astype(np.uint8)
    heat[..., 1] = (np.sqrt(error) * 180.0).round().astype(np.uint8)
    heat[..., 2] = ((1.0 - error) * 60.0).round().astype(np.uint8)
    Image.fromarray(heat, mode="RGB").save(output_path)


def render_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    font_names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def image_tile(path: Path, label: str, *, size: int = 196) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + 30), (246, 248, 251))
    x0 = (size - image.width) // 2
    y0 = 30 + (size - image.height) // 2
    tile.paste(image, (x0, y0))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size - 1, 29), fill=(232, 237, 244))
    draw.text((8, 7), label, font=render_font(14, bold=True), fill=(18, 24, 32))
    return tile


def fit_text(draw: ImageDraw.ImageDraw, text: str, *, font: ImageFont.ImageFont, max_width: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    candidate = text
    while candidate and draw.textbbox((0, 0), candidate + suffix, font=font)[2] > max_width:
        candidate = candidate[:-1]
    return candidate.rstrip() + suffix if candidate else suffix


def metric_delta(prior_metrics: dict[str, Any], ours_metrics: dict[str, Any], name: str, *, higher_is_better: bool) -> float | None:
    prior_value = prior_metrics.get(name)
    ours_value = ours_metrics.get(name)
    if prior_value is None or ours_value is None:
        return None
    return float(ours_value - prior_value) if higher_is_better else float(prior_value - ours_value)


def format_optional(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "n/a"


def save_comparison_panel(
    *,
    case: dict[str, Any],
    case_dir: Path,
    view_spec: dict[str, Any],
    prior_metrics: dict[str, Any],
    ours_metrics: dict[str, Any],
    hdri_preset: str,
    prior_distribution_text: str,
) -> Path:
    tile_size = 196
    gutter = 10
    header_height = 112
    footer_height = 34
    labels = [
        ("gt.png", "GT RGB"),
        ("prior.png", "Input Prior RGB"),
        ("ours.png", "Ours RGB"),
        ("prior_error.png", "Prior Error"),
        ("ours_error.png", "Ours Error"),
    ]
    tiles = [image_tile(case_dir / filename, label, size=tile_size) for filename, label in labels]
    width = gutter + len(tiles) * tile_size + (len(tiles) - 1) * gutter + gutter
    height = header_height + tiles[0].height + gutter + footer_height
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = render_font(20, bold=True)
    detail_font = render_font(13)
    object_id = str(case.get("object_id") or "unknown")
    pair_id = str(case.get("pair_id") or "unknown")
    variant = str(case.get("prior_variant_type") or "unknown")
    quality = str(case.get("prior_quality_bin") or "unknown")
    draw.text((12, 10), fit_text(draw, object_id, font=title_font, max_width=width - 24), font=title_font, fill=(8, 12, 18))
    metric_line = (
        f"variant={variant} | quality={quality} | gain={float(case.get('gain_total') or 0.0):+.4f} | "
        f"PSNR d={format_optional(metric_delta(prior_metrics, ours_metrics, 'psnr', higher_is_better=True))} | "
        f"SSIM d={format_optional(metric_delta(prior_metrics, ours_metrics, 'ssim', higher_is_better=True))} | "
        f"MSE d={format_optional(metric_delta(prior_metrics, ours_metrics, 'mse', higher_is_better=False))} | "
        f"LPIPS d={format_optional(metric_delta(prior_metrics, ours_metrics, 'lpips', higher_is_better=False))}"
    )
    view_line = f"pair={pair_id} | hdri={hdri_preset} | view={view_spec.get('name') or 'unknown'}"
    distribution_line = f"selected prior distribution: {prior_distribution_text}"
    for y0, text in ((38, metric_line), (58, view_line), (78, distribution_line)):
        draw.text((12, y0), fit_text(draw, text, font=detail_font, max_width=width - 24), font=detail_font, fill=(68, 76, 88))
    y_tile = header_height
    for index, tile in enumerate(tiles):
        x0 = gutter + index * (tile_size + gutter)
        canvas.paste(tile, (x0, y_tile))
    draw.rectangle((0, height - footer_height, width, height), fill=(239, 243, 248))
    draw.text(
        (12, height - footer_height + 9),
        fit_text(draw, distribution_line, font=detail_font, max_width=width - 24),
        font=detail_font,
        fill=(54, 63, 78),
    )
    output_path = case_dir / "comparison.png"
    canvas.save(output_path)
    return output_path


def metric_pair(values: list[float], refined_values: list[float], *, higher_is_better: bool) -> dict[str, Any]:
    baseline = float(np.mean(values)) if values else None
    refined = float(np.mean(refined_values)) if refined_values else None
    if baseline is None or refined is None:
        delta = None
    else:
        delta = (refined - baseline) if higher_is_better else (baseline - refined)
    return {
        "baseline": baseline,
        "refined": refined,
        "delta": delta,
        "higher_is_better": higher_is_better,
        "available_count": min(len(values), len(refined_values)),
    }


def summarize_by_variant(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row.get("prior_variant_type") or "unknown")].append(row)
    summary: dict[str, Any] = {}
    for variant, rows in sorted(grouped.items()):
        summary[variant] = {
            "count": len(rows),
            "ours_psnr": metric_pair(
                [item["prior_metrics"]["psnr"] for item in rows],
                [item["ours_metrics"]["psnr"] for item in rows],
                higher_is_better=True,
            ),
            "ours_mse": metric_pair(
                [item["prior_metrics"]["mse"] for item in rows],
                [item["ours_metrics"]["mse"] for item in rows],
                higher_is_better=False,
            ),
            "ours_ssim": metric_pair(
                [item["prior_metrics"]["ssim"] for item in rows],
                [item["ours_metrics"]["ssim"] for item in rows],
                higher_is_better=True,
            ),
            "ours_lpips": metric_pair(
                [item["prior_metrics"]["lpips"] for item in rows if item["prior_metrics"]["lpips"] is not None],
                [item["ours_metrics"]["lpips"] for item in rows if item["ours_metrics"]["lpips"] is not None],
                higher_is_better=False,
            ),
        }
    return summary


def render_case(
    *,
    blender_bin: str,
    blender_script: Path,
    object_path: Path,
    uv_albedo_path: Path,
    uv_normal_path: Path,
    uv_roughness_path: Path,
    uv_metallic_path: Path,
    hdri_path: Path,
    output_path: Path,
    view_spec: dict[str, Any],
    resolution: int,
    cycles_samples: int,
    cuda_device_index: int,
) -> None:
    env = dict(os.environ)
    env["BLENDER_CUDA_DEVICE_INDEX"] = str(cuda_device_index)
    command = [
        blender_bin,
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--object-path",
        str(object_path),
        "--uv-albedo",
        str(uv_albedo_path),
        "--uv-normal",
        str(uv_normal_path),
        "--uv-roughness",
        str(uv_roughness_path),
        "--uv-metallic",
        str(uv_metallic_path),
        "--hdri-path",
        str(hdri_path),
        "--output-path",
        str(output_path),
        "--elevation",
        str(float(view_spec["elevation"])),
        "--azimuth",
        str(float(view_spec["azimuth"])),
        "--distance",
        str(float(view_spec["distance"])),
        "--resolution",
        str(int(resolution)),
        "--cycles-samples",
        str(int(cycles_samples)),
    ]
    subprocess.run(command, check=True, env=env)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = load_manifest_records(args.manifest)
    manifest_by_pair = {str(record.get("pair_id") or ""): record for record in manifest_records}
    metrics_rows = json.loads(args.metrics.read_text(encoding="utf-8"))
    metrics_rows = [row for row in metrics_rows if str(row.get("split")) == args.split]
    grouped_metric_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        grouped_metric_rows[case_unique_key(row)].append(row)
    case_candidates = aggregate_case_rows(metrics_rows)
    selected_cases = select_variant_balanced_records(
        case_candidates,
        max_count=int(args.max_cases),
        mode=str(args.selection_mode),
    )
    selected_prior_distribution = prior_variant_distribution(selected_cases)
    selected_prior_distribution_text = format_prior_distribution(selected_prior_distribution)
    hdri_path = HDRI_PRESETS[args.hdri_preset]
    lpips_device = args.lpips_device or (f"cuda:{args.cuda_device_index}" if torch.cuda.is_available() else "cpu")
    case_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(selected_cases):
        pair_id = str(case.get("pair_id") or "")
        record = manifest_by_pair.get(pair_id)
        if record is None:
            failures.append({"case_key": case.get("case_key"), "reason": f"missing_manifest_pair:{pair_id}"})
            continue
        case_dir = args.output_dir / "cases" / (
            f"{index:03d}__{safe_component(case.get('object_id'))}__"
            f"{safe_component(case.get('prior_variant_type'))}__{safe_component(pair_id or case.get('case_key'))}"
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        view_spec = choose_view_spec(record, case.get("view_name"))
        resolved_paths = resolve_case_render_paths(case, grouped_metric_rows.get(str(case.get("case_key")), []))
        render_inputs = {
            "gt": {
                "roughness": Path(str(record["uv_target_roughness_path"])),
                "metallic": Path(str(record["uv_target_metallic_path"])),
            },
            "prior": {
                "roughness": Path(str(resolved_paths["baseline_roughness"])),
                "metallic": Path(str(resolved_paths["baseline_metallic"])),
            },
            "ours": {
                "roughness": Path(str(resolved_paths["refined_roughness"])),
                "metallic": Path(str(resolved_paths["refined_metallic"])),
            },
        }
        try:
            for label, inputs in render_inputs.items():
                render_case(
                    blender_bin=args.blender_bin,
                    blender_script=args.blender_script,
                    object_path=Path(str(record["canonical_mesh_path"])),
                    uv_albedo_path=Path(str(record["uv_albedo_path"])),
                    uv_normal_path=Path(str(record["uv_normal_path"])),
                    uv_roughness_path=inputs["roughness"],
                    uv_metallic_path=inputs["metallic"],
                    hdri_path=hdri_path,
                    output_path=case_dir / f"{label}.png",
                    view_spec=view_spec,
                    resolution=args.resolution,
                    cycles_samples=args.cycles_samples,
                    cuda_device_index=args.cuda_device_index,
                )
            gt_rgba = load_rgba(case_dir / "gt.png")
            prior_rgba = load_rgba(case_dir / "prior.png")
            ours_rgba = load_rgba(case_dir / "ours.png")
            gt_rgb = gt_rgba[:3]
            prior_rgb = prior_rgba[:3]
            ours_rgb = ours_rgba[:3]
            mask = (gt_rgba[3:4] > 0.01).to(torch.float32)
            prior_mse = masked_mse(prior_rgb, gt_rgb, mask)
            ours_mse = masked_mse(ours_rgb, gt_rgb, mask)
            prior_ssim = masked_global_ssim(prior_rgb, gt_rgb, mask)
            ours_ssim = masked_global_ssim(ours_rgb, gt_rgb, mask)
            prior_lpips = compute_lpips(prior_rgb, gt_rgb, mask, device=lpips_device) if parse_bool(args.enable_lpips) else None
            ours_lpips = compute_lpips(ours_rgb, gt_rgb, mask, device=lpips_device) if parse_bool(args.enable_lpips) else None
            save_error_heatmap(prior_rgb, gt_rgb, mask, case_dir / "prior_error.png")
            save_error_heatmap(ours_rgb, gt_rgb, mask, case_dir / "ours_error.png")
            prior_metrics = {
                "mse": prior_mse,
                "psnr": psnr_from_mse(prior_mse),
                "ssim": prior_ssim,
                "lpips": prior_lpips,
            }
            ours_metrics = {
                "mse": ours_mse,
                "psnr": psnr_from_mse(ours_mse),
                "ssim": ours_ssim,
                "lpips": ours_lpips,
            }
            comparison_path = save_comparison_panel(
                case=case,
                case_dir=case_dir,
                view_spec=view_spec,
                prior_metrics=prior_metrics,
                ours_metrics=ours_metrics,
                hdri_preset=args.hdri_preset,
                prior_distribution_text=selected_prior_distribution_text,
            )
            case_row = {
                "selection_slot": index,
                "case_key": case.get("case_key"),
                "object_id": case.get("object_id"),
                "pair_id": case.get("pair_id"),
                "prior_variant_id": case.get("prior_variant_id"),
                "prior_variant_type": case.get("prior_variant_type"),
                "prior_quality_bin": case.get("prior_quality_bin"),
                "training_role": case.get("training_role"),
                "hdri_preset": args.hdri_preset,
                "view_name": view_spec.get("name"),
                "case_gain_total": case.get("gain_total"),
                "selected_prior_distribution": selected_prior_distribution,
                "selected_prior_distribution_text": selected_prior_distribution_text,
                "prior_metrics": prior_metrics,
                "ours_metrics": ours_metrics,
                "paths": {
                    "gt": str((case_dir / "gt.png").resolve()),
                    "prior": str((case_dir / "prior.png").resolve()),
                    "ours": str((case_dir / "ours.png").resolve()),
                    "prior_error": str((case_dir / "prior_error.png").resolve()),
                    "ours_error": str((case_dir / "ours_error.png").resolve()),
                    "comparison_panel": str(comparison_path.resolve()),
                },
                "comparison_panel": str(comparison_path.resolve()),
            }
            save_json(case_dir / "case.json", case_row)
            case_rows.append(case_row)
        except Exception as exc:  # noqa: BLE001
            failures.append({"case_key": case.get("case_key"), "reason": f"{type(exc).__name__}: {exc}"})
    real_render_metrics = {
        "psnr": metric_pair(
            [row["prior_metrics"]["psnr"] for row in case_rows],
            [row["ours_metrics"]["psnr"] for row in case_rows],
            higher_is_better=True,
        ),
        "mse": metric_pair(
            [row["prior_metrics"]["mse"] for row in case_rows],
            [row["ours_metrics"]["mse"] for row in case_rows],
            higher_is_better=False,
        ),
        "ssim": metric_pair(
            [row["prior_metrics"]["ssim"] for row in case_rows],
            [row["ours_metrics"]["ssim"] for row in case_rows],
            higher_is_better=True,
        ),
        "lpips": metric_pair(
            [row["prior_metrics"]["lpips"] for row in case_rows if row["prior_metrics"]["lpips"] is not None],
            [row["ours_metrics"]["lpips"] for row in case_rows if row["ours_metrics"]["lpips"] is not None],
            higher_is_better=False,
        ),
    }
    summary_payload = {
        "manifest": str(args.manifest.resolve()),
        "metrics": str(args.metrics.resolve()),
        "split": args.split,
        "evaluation_basis": f"benchmark_{args.split}_realrender_{int(args.max_cases)}",
        "selection_mode": args.selection_mode,
        "selected_cases": len(selected_cases),
        "completed_cases": len(case_rows),
        "failed_cases": len(failures),
        "selected_prior_distribution": selected_prior_distribution,
        "selected_prior_distribution_text": selected_prior_distribution_text,
        "hdri_preset": args.hdri_preset,
        "hdri_path": str(hdri_path.resolve()),
        "real_render_metrics": real_render_metrics,
        "psnr_delta": (real_render_metrics.get("psnr") or {}).get("delta"),
        "mse_delta": (real_render_metrics.get("mse") or {}).get("delta"),
        "ssim_delta": (real_render_metrics.get("ssim") or {}).get("delta"),
        "lpips_delta": (real_render_metrics.get("lpips") or {}).get("delta"),
        "by_prior_variant_type": summarize_by_variant(case_rows),
        "cases": case_rows,
        "failures": failures,
    }
    save_json(args.output_dir / "summary.json", summary_payload)
    print(json.dumps({"real_render_summary": {k: v for k, v in summary_payload.items() if k not in {"cases", "failures"}}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
