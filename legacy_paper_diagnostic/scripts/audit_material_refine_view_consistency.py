from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine import (  # noqa: E402
    CanonicalMaterialDataset,
    MaterialRefinementPipeline,
    collate_material_samples,
)
from sf3d.material_refine.experiment import make_json_serializable  # noqa: E402


DEFAULT_EVAL_DIR = REPO_ROOT / "output/material_refine_r_v2_dayrun/acceptance_128_eval"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/material_refine_r_v2_dayrun/view_consistency_audit"
MODEL_CFG_OVERRIDE_KEYS = [
    "enable_prior_source_embedding",
    "enable_no_prior_bootstrap",
    "enable_boundary_safety",
    "enable_change_gate",
    "enable_material_aux_head",
    "enable_render_proxy_loss",
    "enable_material_evidence_calibration",
    "material_evidence_channels",
    "material_evidence_strength",
    "enable_evidence_update_budget",
    "evidence_update_budget_strength",
    "evidence_update_budget_floor",
]
CASE_COLUMNS = [
    "object_id",
    "view_name",
    "uv_gain",
    "view_gain",
    "uv_prior_error",
    "uv_pred_error",
    "view_prior_error",
    "view_pred_error",
    "target_prior_identity",
    "prior_label",
    "prior_mode",
    "prior_source_type",
    "material_family",
    "source_name",
    "target_source_type",
    "target_quality_tier",
    "thin_boundary_flag",
    "sampled_target_stored_total_mae",
    "uv_valid_in_mask_rate",
]


@dataclass(frozen=True)
class CaseKey:
    object_id: str
    view_name: str


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "any", "none"}:
        return None
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"unsupported_optional_bool:{value}")


def parse_csv_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",")]
    items = [item for item in items if item]
    return items or None


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def optional_mean(values: list[Any]) -> float | None:
    finite_values = [finite_float(value) for value in values]
    finite_values = [value for value in finite_values if value is not None]
    if not finite_values:
        return None
    return float(np.mean(finite_values))


def pair_direction_rates(baseline_values: list[Any], refined_values: list[Any]) -> dict[str, float]:
    improved = regressed = tied = total = 0
    for baseline, refined in zip(baseline_values, refined_values, strict=False):
        base = finite_float(baseline)
        pred = finite_float(refined)
        if base is None or pred is None:
            continue
        total += 1
        if pred < base:
            improved += 1
        elif pred > base:
            regressed += 1
        else:
            tied += 1
    denom = max(total, 1)
    return {
        "count": total,
        "improvement_rate": float(improved / denom),
        "regression_rate": float(regressed / denom),
        "tie_rate": float(tied / denom),
    }


def first_nonempty(*values: Any, default: str = "unknown") -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return str(default)


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_json_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def load_csv_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(make_json_serializable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)[:180]


def rel(path: Path, start: Path) -> str:
    try:
        return str(path.resolve().relative_to(start.resolve()))
    except ValueError:
        return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Audit why UV-space material refinement improves while view/render rows regress.",
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--rows", type=Path, default=None, help="Eval metrics rows JSON. Defaults to eval_dir/metrics.json.")
    parser.add_argument(
        "--metric-debug-csv",
        type=Path,
        default=None,
        help="metric_consistency_debug.csv used as the case selection source when available.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--split-strategy", choices=["auto", "manifest", "hash"], default=None)
    parser.add_argument("--hash-val-ratio", type=float, default=None)
    parser.add_argument("--hash-test-ratio", type=float, default=None)
    parser.add_argument("--generator-ids", type=str, default=None)
    parser.add_argument("--source-names", type=str, default=None)
    parser.add_argument("--supervision-tiers", type=str, default=None)
    parser.add_argument("--supervision-roles", type=str, default=None)
    parser.add_argument("--license-buckets", type=str, default=None)
    parser.add_argument("--target-quality-tiers", type=str, default=None)
    parser.add_argument("--paper-splits", type=str, default=None)
    parser.add_argument("--material-families", type=str, default=None)
    parser.add_argument("--lighting-bank-ids", type=str, default=None)
    parser.add_argument("--require-prior", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--view-sample-count", type=int, default=None)
    parser.add_argument("--min-hard-views", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="CPU is the default so this audit does not compete with active render/train jobs.",
    )
    parser.add_argument("--cuda-device-index", type=int, default=None)
    parser.add_argument("--panel-size", type=int, default=128)
    parser.add_argument("--debug-case-count", type=int, default=16)
    parser.add_argument("--case-count", type=int, default=16)
    parser.add_argument("--prior-identity-atol", type=float, default=1.0e-8)
    parser.add_argument("--target-view-mae-threshold", type=float, default=0.05)
    parser.add_argument("--enable-prior-source-embedding", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-no-prior-bootstrap", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-boundary-safety", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-change-gate", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-material-aux-head", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-render-proxy-loss", type=parse_optional_bool, default=None)
    return parser


def apply_eval_arg_defaults(args: argparse.Namespace, eval_args: dict[str, Any]) -> None:
    defaults = {
        "summary": args.eval_dir / "summary.json",
        "rows": args.eval_dir / "metrics.json",
        "metric_debug_csv": args.eval_dir / "metric_consistency_debug.csv",
        "split": "all",
        "split_strategy": "manifest",
        "hash_val_ratio": 0.1,
        "hash_test_ratio": 0.1,
        "batch_size": 2,
        "num_workers": 0,
        "view_sample_count": 4,
        "min_hard_views": 0,
        "max_samples": None,
        "cuda_device_index": 0,
    }
    for key, default in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, eval_args.get(key, default))
    if args.manifest is None:
        args.manifest = Path(eval_args.get("manifest") or DEFAULT_EVAL_DIR / "manifest_snapshot.json")
    if args.checkpoint is None:
        checkpoint = eval_args.get("checkpoint")
        if checkpoint:
            args.checkpoint = Path(checkpoint)
    for key in MODEL_CFG_OVERRIDE_KEYS:
        if hasattr(args, key) and getattr(args, key) is None and key in eval_args:
            setattr(args, key, eval_args[key])


def build_model_cfg_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in MODEL_CFG_OVERRIDE_KEYS:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                overrides[key] = value
    return overrides


def build_dataset(args: argparse.Namespace, atlas_size: int, buffer_resolution: int) -> CanonicalMaterialDataset:
    require_prior = args.require_prior
    if require_prior in (None, "any", ""):
        parsed_require_prior = None
    else:
        parsed_require_prior = parse_bool(require_prior)
    return CanonicalMaterialDataset(
        args.manifest,
        split=args.split,
        split_strategy=args.split_strategy,
        hash_val_ratio=float(args.hash_val_ratio),
        hash_test_ratio=float(args.hash_test_ratio),
        generator_ids=parse_csv_list(args.generator_ids),
        source_names=parse_csv_list(args.source_names),
        supervision_tiers=parse_csv_list(args.supervision_tiers),
        supervision_roles=parse_csv_list(args.supervision_roles),
        license_buckets=parse_csv_list(args.license_buckets),
        target_quality_tiers=parse_csv_list(args.target_quality_tiers),
        paper_splits=parse_csv_list(args.paper_splits),
        material_families=parse_csv_list(args.material_families),
        lighting_bank_ids=parse_csv_list(args.lighting_bank_ids),
        require_prior=parsed_require_prior,
        max_records=args.max_samples,
        atlas_size=atlas_size,
        buffer_resolution=buffer_resolution,
        max_views_per_sample=int(args.view_sample_count or 0),
        min_hard_views_per_sample=int(args.min_hard_views or 0),
        randomize_view_subset=False,
    )


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def detach_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in outputs.items()
    }


def sample_uv_maps_to_view_options(
    uv_maps: torch.Tensor,
    view_uvs: torch.Tensor,
    *,
    y_flip: bool = True,
    align_corners: bool = True,
) -> torch.Tensor:
    grid = view_uvs.clone()
    grid[..., 0] = grid[..., 0] * 2.0 - 1.0
    if y_flip:
        grid[..., 1] = (1.0 - grid[..., 1]) * 2.0 - 1.0
    else:
        grid[..., 1] = grid[..., 1] * 2.0 - 1.0
    batch, views, height, width, _ = grid.shape
    repeated_maps = (
        uv_maps.unsqueeze(1)
        .expand(-1, views, -1, -1, -1)
        .reshape(batch * views, uv_maps.shape[1], uv_maps.shape[2], uv_maps.shape[3])
    )
    sampled = F.grid_sample(
        repeated_maps,
        grid.reshape(batch * views, height, width, 2),
        mode="bilinear",
        align_corners=align_corners,
    )
    return sampled.view(batch, views, sampled.shape[1], height, width)


def sample_uv_maps_to_view(uv_maps: torch.Tensor, view_uvs: torch.Tensor) -> torch.Tensor:
    return sample_uv_maps_to_view_options(uv_maps, view_uvs, y_flip=True, align_corners=True)


def uv_total_error(pred: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor) -> tuple[float, float, float]:
    weighted = (pred - target).abs() * confidence
    weight = float(confidence.sum().item()) or 1.0
    roughness = float(weighted[0].sum().item() / weight)
    metallic = float(weighted[1].sum().item() / weight)
    return roughness, metallic, roughness + metallic


def masked_view_total_error(pred_view: torch.Tensor, gt_view: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float]:
    mask_bool = mask > 0.5
    if not bool(mask_bool.any()):
        return math.nan, math.nan, math.nan
    rough = float((pred_view[0][mask_bool] - gt_view[0][mask_bool]).abs().mean().item())
    metal = float((pred_view[1][mask_bool] - gt_view[1][mask_bool]).abs().mean().item())
    return rough, metal, rough + metal


def view_uv_valid_map(view_uv: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(view_uv).all(dim=-1)
    in_range = (
        (view_uv[..., 0] >= 0.0)
        & (view_uv[..., 0] <= 1.0)
        & (view_uv[..., 1] >= 0.0)
        & (view_uv[..., 1] <= 1.0)
    )
    return finite & in_range


def target_identity_bucket(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "unknown"
    if number < 0.30:
        return "<0.30"
    if number <= 0.70:
        return "0.30-0.70"
    return ">0.70"


def add_mode_value(
    aggregates: dict[str, dict[str, list[float]]],
    mode: str,
    *,
    uv_baseline: float | None = None,
    uv_refined: float | None = None,
    view_baseline: float | None = None,
    view_refined: float | None = None,
) -> None:
    item = aggregates[mode]
    if uv_baseline is not None and math.isfinite(uv_baseline):
        item["uv_baseline"].append(float(uv_baseline))
    if uv_refined is not None and math.isfinite(uv_refined):
        item["uv_refined"].append(float(uv_refined))
    if view_baseline is not None and math.isfinite(view_baseline):
        item["view_baseline"].append(float(view_baseline))
    if view_refined is not None and math.isfinite(view_refined):
        item["view_refined"].append(float(view_refined))


def batch_metadata(batch: dict[str, Any], item_idx: int) -> dict[str, Any]:
    metadata = batch["metadata"][item_idx] if isinstance(batch.get("metadata"), list) else {}
    has_prior = bool(batch["has_material_prior"][item_idx])
    prior_mode = str(batch["prior_mode"][item_idx])
    prior_source_type = first_nonempty(
        batch["prior_source_type"][item_idx] if isinstance(batch.get("prior_source_type"), list) else None,
        metadata.get("prior_source_type") if isinstance(metadata, dict) else None,
        prior_mode,
    )
    return {
        "generator_id": str(batch["generator_id"][item_idx]),
        "source_name": str(batch["source_name"][item_idx]),
        "category_bucket": str(batch["category_bucket"][item_idx]),
        "prior_label": "with_prior" if has_prior else "without_prior",
        "prior_mode": prior_mode,
        "prior_source_type": prior_source_type,
        "has_material_prior": has_prior,
        "target_source_type": str(batch["target_source_type"][item_idx]),
        "target_prior_identity": float(batch["target_prior_identity"][item_idx].item()),
        "target_quality_tier": str(batch["target_quality_tier"][item_idx]),
        "paper_split": str(batch["paper_split"][item_idx]),
        "material_family": str(batch["material_family"][item_idx]),
        "lighting_bank_id": str(batch["lighting_bank_id"][item_idx]),
        "thin_boundary_flag": bool(batch["thin_boundary_flag"][item_idx]),
    }


def input_rgb_image(batch: dict[str, Any], item_idx: int, view_idx: int) -> np.ndarray:
    rgb = batch["view_features"][item_idx, view_idx, 0:3].detach().cpu().numpy()
    return np.moveaxis(rgb, 0, -1)


def gray_tile(value: Any, size: int) -> Image.Image:
    arr = tensor_or_array_2d(value)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    image = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").convert("RGB")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def rgb_tile(value: Any, size: int) -> Image.Image:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in {3, 4} and arr.shape[-1] not in {3, 4}:
        arr = np.moveaxis(arr[:3], 0, -1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr[..., :3], 0.0, 1.0)
    image = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def heat_tile(value: Any, size: int, scale: float = 4.0) -> Image.Image:
    arr = tensor_or_array_2d(value)
    arr = np.nan_to_num(np.abs(arr) * scale, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    heat = np.zeros((*arr.shape, 3), dtype=np.uint8)
    heat[..., 0] = (arr * 255.0).astype(np.uint8)
    heat[..., 1] = (np.sqrt(arr) * 130.0).astype(np.uint8)
    heat[..., 2] = ((1.0 - arr) * 45.0).astype(np.uint8)
    return Image.fromarray(heat, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)


def tensor_or_array_2d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected_2d_array_got_shape:{arr.shape}")
    return arr.astype(np.float32)


def label_tile(image: Image.Image, label: str) -> Image.Image:
    header_height = 26
    canvas = Image.new("RGB", (image.width, image.height + header_height), (10, 14, 20))
    canvas.paste(image, (0, header_height))
    ImageDraw.Draw(canvas).text((7, 7), label[:32], fill=(236, 242, 248))
    return canvas


def grid_panel(
    tiles: list[tuple[str, Image.Image]],
    *,
    title: str,
    subtitle: str,
    columns: int = 6,
) -> Image.Image:
    labeled = [label_tile(image, label) for label, image in tiles]
    if not labeled:
        labeled = [label_tile(Image.new("RGB", (128, 128), (30, 34, 42)), "empty")]
    tile_width, tile_height = labeled[0].size
    rows = int(math.ceil(len(labeled) / columns))
    header_height = 86
    canvas = Image.new("RGB", (columns * tile_width, header_height + rows * tile_height), (8, 12, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 13), title[:180], fill=(244, 248, 255))
    draw.text((14, 43), subtitle[:220], fill=(174, 196, 214))
    for idx, tile in enumerate(labeled):
        canvas.paste(tile, ((idx % columns) * tile_width, header_height + (idx // columns) * tile_height))
    return canvas


def build_case_panel(
    *,
    case_row: dict[str, Any],
    batch: dict[str, Any],
    item_idx: int,
    view_idx: int,
    baseline: torch.Tensor,
    target: torch.Tensor,
    refined: torch.Tensor,
    size: int,
    include_sampling_tiles: bool,
) -> Image.Image:
    view_uv = batch["view_uvs"][item_idx : item_idx + 1, view_idx : view_idx + 1]
    sampled_prior = sample_uv_maps_to_view(baseline[item_idx : item_idx + 1], view_uv)[0, 0]
    sampled_target = sample_uv_maps_to_view(target[item_idx : item_idx + 1], view_uv)[0, 0]
    sampled_refined = sample_uv_maps_to_view(refined[item_idx : item_idx + 1], view_uv)[0, 0]
    stored_target = batch["view_targets"][item_idx, view_idx]
    mask = batch["view_masks"][item_idx, view_idx, 0]
    valid = view_uv_valid_map(batch["view_uvs"][item_idx, view_idx]).float()
    prior_error = (sampled_prior - stored_target).abs().mean(dim=0)
    pred_error = (sampled_refined - stored_target).abs().mean(dim=0)
    target_error = (sampled_target - stored_target).abs().mean(dim=0)

    tiles: list[tuple[str, Image.Image]] = [
        ("Input RGB", rgb_tile(input_rgb_image(batch, item_idx, view_idx), size)),
        ("Prior view rough", gray_tile(sampled_prior[0], size)),
        ("Prior view metal", gray_tile(sampled_prior[1], size)),
        ("Sampled GT rough", gray_tile(sampled_target[0], size)),
        ("Sampled GT metal", gray_tile(sampled_target[1], size)),
        ("Pred view rough", gray_tile(sampled_refined[0], size)),
        ("Pred view metal", gray_tile(sampled_refined[1], size)),
        ("abs Prior-GT view", heat_tile(prior_error, size)),
        ("abs Pred-GT view", heat_tile(pred_error, size)),
        ("View mask", gray_tile(mask, size)),
    ]
    if include_sampling_tiles:
        tiles.extend(
            [
                ("Prior UV rough", gray_tile(baseline[item_idx, 0], size)),
                ("Prior UV metal", gray_tile(baseline[item_idx, 1], size)),
                ("Target UV rough", gray_tile(target[item_idx, 0], size)),
                ("Target UV metal", gray_tile(target[item_idx, 1], size)),
                ("Pred UV rough", gray_tile(refined[item_idx, 0], size)),
                ("Pred UV metal", gray_tile(refined[item_idx, 1], size)),
                ("Confidence UV", gray_tile(batch["uv_target_confidence"][item_idx, 0], size)),
                ("Stored GT rough", gray_tile(stored_target[0], size)),
                ("Stored GT metal", gray_tile(stored_target[1], size)),
                ("abs target rough", heat_tile((sampled_target[0] - stored_target[0]).abs(), size)),
                ("abs target metal", heat_tile((sampled_target[1] - stored_target[1]).abs(), size)),
                ("abs target total", heat_tile(target_error, size)),
                ("view_uv valid", gray_tile(valid, size)),
            ]
        )
    uv_gain = finite_float(case_row.get("uv_gain"))
    view_gain = finite_float(case_row.get("view_gain"))
    uv_improved = uv_gain is not None and uv_gain > 0.0
    view_regressed = view_gain is not None and view_gain < 0.0
    title = (
        f"{case_row.get('object_id')} | {case_row.get('view_name')} | "
        f"uv_gain={fmt_float(uv_gain)} view_gain={fmt_float(view_gain)} "
        f"uv_improved={uv_improved} view_regressed={view_regressed}"
    )
    subtitle = (
        f"source={case_row.get('source_name')} material={case_row.get('material_family')} "
        f"prior={case_row.get('prior_label')}/{case_row.get('prior_source_type')} "
        f"target_identity={fmt_float(case_row.get('target_prior_identity'))}"
    )
    return grid_panel(tiles, title=title, subtitle=subtitle, columns=6)


def fmt_float(value: Any) -> str:
    number = finite_float(value)
    return "NA" if number is None else f"{number:.6f}"


def find_cases_in_dataset(
    *,
    selected: list[dict[str, Any]],
    dataset: CanonicalMaterialDataset,
    pipeline: MaterialRefinementPipeline,
    args: argparse.Namespace,
    output_dir: Path,
    include_sampling_tiles: bool,
) -> list[Path]:
    requested = {CaseKey(str(row["object_id"]), str(row["view_name"])): row for row in selected}
    if not requested:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or 1),
        shuffle=False,
        num_workers=int(args.num_workers or 0),
        pin_memory=False,
        collate_fn=collate_material_samples,
    )
    paths: list[Path] = []
    with torch.no_grad():
        for batch in loader:
            batch_keys = {
                CaseKey(str(object_id), str(view_name))
                for item_idx, object_id in enumerate(batch["object_id"])
                for view_name in batch["view_names"][item_idx]
            }
            if not (set(requested.keys()) & batch_keys):
                continue
            outputs = detach_outputs(pipeline.model(move_batch_to_device(batch, pipeline.device)))
            baseline = outputs.get("input_prior", outputs.get("baseline"))
            refined = outputs["refined"]
            target = torch.cat([batch["uv_target_roughness"], batch["uv_target_metallic"]], dim=1)
            for item_idx, object_id in enumerate(batch["object_id"]):
                for view_idx, view_name in enumerate(batch["view_names"][item_idx]):
                    key = CaseKey(str(object_id), str(view_name))
                    case_row = requested.get(key)
                    if case_row is None:
                        continue
                    panel = build_case_panel(
                        case_row=case_row,
                        batch=batch,
                        item_idx=item_idx,
                        view_idx=view_idx,
                        baseline=baseline,
                        target=target,
                        refined=refined,
                        size=int(args.panel_size),
                        include_sampling_tiles=include_sampling_tiles,
                    )
                    panel_path = output_dir / f"{safe_name(object_id)}__{safe_name(view_name)}.png"
                    panel.save(panel_path)
                    paths.append(panel_path)
                    del requested[key]
            if not requested:
                break
    return paths


def build_html_index(
    *,
    rows: list[dict[str, Any]],
    panel_paths: list[Path],
    output_path: Path,
    title: str,
    intro: str,
) -> None:
    cards = []
    path_by_name = {path.name: path for path in panel_paths}
    for row in rows:
        expected = f"{safe_name(row.get('object_id'))}__{safe_name(row.get('view_name'))}.png"
        panel_path = path_by_name.get(expected)
        image_html = (
            f"<img src='{html.escape(panel_path.name)}' alt='{html.escape(str(row.get('object_id')))} panel'>"
            if panel_path is not None
            else "<div class='missing'>panel missing</div>"
        )
        meta = " | ".join(
            [
                f"view={row.get('view_name')}",
                f"uv_gain={fmt_float(row.get('uv_gain'))}",
                f"view_gain={fmt_float(row.get('view_gain'))}",
                f"prior={row.get('prior_label')}/{row.get('prior_source_type')}",
                f"source={row.get('source_name')}",
                f"material={row.get('material_family')}",
                f"identity={fmt_float(row.get('target_prior_identity'))}",
                f"target_mismatch={fmt_float(row.get('sampled_target_stored_total_mae'))}",
            ]
        )
        cards.append(
            "\n".join(
                [
                    "<section class='card'>",
                    f"<h2>{html.escape(str(row.get('object_id')))}</h2>",
                    f"<div class='meta'>{html.escape(meta)}</div>",
                    image_html,
                    "</section>",
                ]
            )
        )
    output_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'>",
                f"<title>{html.escape(title)}</title>",
                "<style>",
                "body{margin:0;background:#081019;color:#eef5ff;font-family:Arial,sans-serif}",
                ".wrap{max-width:1500px;margin:0 auto;padding:28px}",
                ".card{background:#111b27;border:1px solid #263648;border-radius:8px;padding:18px;margin:18px 0}",
                ".meta{color:#a9c4d8;margin:6px 0 14px;line-height:1.35}",
                "img{width:100%;background:#090d13;border-radius:6px}",
                ".missing{padding:36px;background:#17202d;color:#fca5a5}",
                "table{border-collapse:collapse;width:100%;margin:12px 0}",
                "td,th{border-bottom:1px solid #334155;padding:7px;text-align:left}",
                "</style></head><body><main class='wrap'>",
                f"<h1>{html.escape(title)}</h1>",
                f"<p>{html.escape(intro)}</p>",
                *cards,
                "</main></body></html>",
            ]
        ),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def counter_table(rows: list[dict[str, Any]], key: str) -> list[str]:
    counts = Counter(str(row.get(key, "unknown")) for row in rows)
    return [f"- {key} `{name}`: `{count}`" for name, count in counts.most_common()]


def identity_notes(mode: str, row: dict[str, Any]) -> str:
    if mode == "prior_as_pred":
        if row.get("identity_pass"):
            return "PASS: forced refined=input_prior exactly reproduces the input-prior view metric."
        return "FAIL: forced refined=input_prior does not reproduce the input-prior view metric."
    if mode == "target_as_pred":
        if row.get("identity_pass"):
            return "PASS: sampled UV target is close to stored view target and beats input prior."
        return "FAIL: sampled UV target is not close enough to stored view target, or does not beat input prior."
    if mode == "bootstrap_as_pred":
        return "Diagnostic on without_prior rows only; shows whether bootstrap/rm_init is already view-regressive."
    if mode == "constant_as_pred":
        return "Sanity lower bound: roughness=0.5 metallic=0.0."
    return ""


def build_identity_summary(
    *,
    aggregates: dict[str, dict[str, list[float]]],
    summary_payload: dict[str, Any],
    args: argparse.Namespace,
    sampling_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    existing_view_input = (summary_payload.get("view_level") or {}).get("input_prior_total_mae")
    for mode in ["prior_as_pred", "target_as_pred", "bootstrap_as_pred", "constant_as_pred", "model_pred"]:
        item = aggregates[mode]
        uv_baseline = optional_mean(item["uv_baseline"])
        uv_refined = optional_mean(item["uv_refined"])
        view_baseline = optional_mean(item["view_baseline"])
        view_refined = optional_mean(item["view_refined"])
        rates = pair_direction_rates(item["view_baseline"], item["view_refined"])
        gain = None if view_baseline is None or view_refined is None else float(view_baseline - view_refined)
        uv_total = uv_refined
        view_total = view_refined
        identity_pass = None
        if mode == "prior_as_pred":
            max_abs = max([abs(a - b) for a, b in zip(item["view_baseline"], item["view_refined"], strict=False)] or [0.0])
            existing_delta = (
                None
                if existing_view_input is None or view_refined is None
                else abs(float(existing_view_input) - float(view_refined))
            )
            identity_pass = bool(max_abs <= args.prior_identity_atol and (existing_delta is None or existing_delta <= 1.0e-6))
        elif mode == "target_as_pred":
            target_mismatch = sampling_summary.get("sampled_target_stored_total_mae_mean")
            identity_pass = bool(
                gain is not None
                and gain > 0.0
                and target_mismatch is not None
                and float(target_mismatch) <= float(args.target_view_mae_threshold)
            )
        row = {
            "mode": mode,
            "uv_input_prior_total_mae": uv_baseline,
            "uv_total_mae": uv_total,
            "view_input_prior_total_mae": view_baseline,
            "view_total_mae": view_total,
            "gain_vs_input_prior": gain,
            "improvement_rate": rates["improvement_rate"],
            "regression_rate": rates["regression_rate"],
            "row_count": rates["count"],
            "identity_pass": identity_pass,
        }
        row["notes"] = identity_notes(mode, row)
        rows.append(row)
    return rows


def write_identity_markdown(path: Path, rows: list[dict[str, Any]], sampling_summary: dict[str, Any]) -> None:
    lines = [
        "# View Identity Test Summary",
        "",
        "| mode | uv_total_mae | view_total_mae | gain_vs_input_prior | improvement_rate | regression_rate | notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["mode"]),
                    fmt_float(row.get("uv_total_mae")),
                    fmt_float(row.get("view_total_mae")),
                    fmt_float(row.get("gain_vs_input_prior")),
                    fmt_float(row.get("improvement_rate")),
                    fmt_float(row.get("regression_rate")),
                    str(row.get("notes", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Sampling Sanity",
            "",
            f"- sampled_target_stored_total_mae_mean: `{fmt_float(sampling_summary.get('sampled_target_stored_total_mae_mean'))}`",
            f"- sampled_target_stored_total_mae_p95: `{fmt_float(sampling_summary.get('sampled_target_stored_total_mae_p95'))}`",
            f"- uv_valid_in_mask_rate_mean: `{fmt_float(sampling_summary.get('uv_valid_in_mask_rate_mean'))}`",
            f"- eval_y_flip_target_mismatch_mean: `{fmt_float(sampling_summary.get('target_mismatch_y_flip_align_true_mean'))}`",
            f"- no_y_flip_target_mismatch_mean: `{fmt_float(sampling_summary.get('target_mismatch_no_y_flip_align_true_mean'))}`",
            f"- align_corners_false_target_mismatch_mean: `{fmt_float(sampling_summary.get('target_mismatch_y_flip_align_false_mean'))}`",
            f"- channel_swapped_target_mismatch_mean: `{fmt_float(sampling_summary.get('target_mismatch_channel_swapped_mean'))}`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_with_metric_debug(
    computed_cases: list[dict[str, Any]],
    metric_debug_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not metric_debug_rows:
        return {
            "available": False,
            "baseline_abs_diff_mean": None,
            "baseline_abs_diff_max": None,
            "refined_abs_diff_mean": None,
            "refined_abs_diff_max": None,
        }
    computed = {
        (str(row.get("object_id")), str(row.get("view_name"))): row
        for row in computed_cases
    }
    baseline_diffs = []
    refined_diffs = []
    for row in metric_debug_rows:
        key = (str(row.get("object_id")), str(row.get("view_name")))
        current = computed.get(key)
        if current is None:
            continue
        debug_baseline = finite_float(row.get("input_prior_total_mae"))
        debug_refined = finite_float(row.get("refined_total_mae"))
        if debug_baseline is not None:
            baseline_diffs.append(abs(debug_baseline - float(current["view_prior_error"])))
        if debug_refined is not None:
            refined_diffs.append(abs(debug_refined - float(current["view_pred_error"])))
    return {
        "available": True,
        "matched_rows": len(baseline_diffs),
        "baseline_abs_diff_mean": optional_mean(baseline_diffs),
        "baseline_abs_diff_max": max(baseline_diffs) if baseline_diffs else None,
        "refined_abs_diff_mean": optional_mean(refined_diffs),
        "refined_abs_diff_max": max(refined_diffs) if refined_diffs else None,
    }


def select_conflict_cases(rows: list[dict[str, Any]], case_count: int) -> dict[str, list[dict[str, Any]]]:
    def uv_gain(row: dict[str, Any]) -> float:
        return finite_float(row.get("uv_gain")) or 0.0

    def view_gain(row: dict[str, Any]) -> float:
        return finite_float(row.get("view_gain")) or 0.0

    uv_improved_view_regressed = [
        row for row in rows if uv_gain(row) > 0.0 and view_gain(row) < 0.0
    ]
    without_prior_regressed = [
        row for row in rows if str(row.get("prior_label")) == "without_prior" and view_gain(row) < 0.0
    ]
    with_prior_regressed = [
        row for row in rows if str(row.get("prior_label")) == "with_prior" and view_gain(row) < 0.0
    ]
    return {
        "uv_improved_view_regressed_top": sorted(
            uv_improved_view_regressed,
            key=lambda row: view_gain(row),
        )[:case_count],
        "uv_improved_most_view_regressed_most": sorted(
            uv_improved_view_regressed,
            key=lambda row: (-(uv_gain(row) + abs(min(view_gain(row), 0.0))), view_gain(row)),
        )[:case_count],
        "without_prior_view_regressed_top": sorted(
            without_prior_regressed,
            key=lambda row: view_gain(row),
        )[:case_count],
        "with_prior_view_regressed_top": sorted(
            with_prior_regressed,
            key=lambda row: view_gain(row),
        )[:case_count],
    }


def write_conflict_markdown(
    *,
    path: Path,
    selected: dict[str, list[dict[str, Any]]],
    category_outputs: dict[str, dict[str, Path]],
    all_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# UV/View Conflict Cases",
        "",
        f"- total_view_rows: `{len(all_rows)}`",
        f"- uv_improved_view_regressed_rows: `{sum(1 for row in all_rows if (finite_float(row.get('uv_gain')) or 0.0) > 0.0 and (finite_float(row.get('view_gain')) or 0.0) < 0.0)}`",
        "",
    ]
    for name, rows in selected.items():
        outputs = category_outputs.get(name, {})
        lines.extend(
            [
                f"## {name}",
                "",
                f"- selected_rows: `{len(rows)}`",
                f"- csv: `{outputs.get('csv')}`",
                f"- html: `{outputs.get('html')}`",
                "",
                "### Distributions",
                "",
                *counter_table(rows, "source_name"),
                *counter_table(rows, "material_family"),
                *counter_table(rows, "prior_source_type"),
                *counter_table(
                    [
                        {**row, "target_prior_identity_bucket": target_identity_bucket(row.get("target_prior_identity"))}
                        for row in rows
                    ],
                    "target_prior_identity_bucket",
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def group_alignment_stats(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    output = {}
    for group, items in sorted(grouped.items()):
        mismatches = [finite_float(row.get("sampled_target_stored_total_mae")) for row in items]
        mismatches = [value for value in mismatches if value is not None]
        valid_rates = [finite_float(row.get("uv_valid_in_mask_rate")) for row in items]
        valid_rates = [value for value in valid_rates if value is not None]
        high_mismatch = sum(value > 0.25 for value in mismatches)
        output[group] = {
            "count": len(items),
            "mismatch_mean": optional_mean(mismatches),
            "mismatch_p95": percentile(mismatches, 95),
            "high_mismatch_rate": float(high_mismatch / max(len(mismatches), 1)),
            "uv_valid_in_mask_rate_mean": optional_mean(valid_rates),
        }
    return output


def source_recommendation(stats: dict[str, Any]) -> str:
    mismatch = finite_float(stats.get("mismatch_mean"))
    valid = finite_float(stats.get("uv_valid_in_mask_rate_mean"))
    if valid is not None and valid < 0.95:
        return "diagnostic-only until view_uv visibility is fixed"
    if mismatch is None:
        return "diagnostic-only until alignment can be measured"
    if mismatch > 0.25:
        return "do not train against stored view target; use sampled UV target or rebuild view targets"
    if mismatch > 0.05:
        return "review before paper use; prefer sampled UV target for view loss"
    return "alignment acceptable"


def write_target_view_alignment_audit(
    *,
    path: Path,
    json_path: Path,
    view_cases: list[dict[str, Any]],
    sampling_summary: dict[str, Any],
    debug_cases: list[dict[str, Any]],
) -> None:
    by_source = group_alignment_stats(view_cases, "source_name")
    by_material = group_alignment_stats(view_cases, "material_family")
    by_prior_source = group_alignment_stats(view_cases, "prior_source_type")
    payload = {
        "sampling_summary": sampling_summary,
        "by_source_name": by_source,
        "by_material_family": by_material,
        "by_prior_source_type": by_prior_source,
        "top_debug_cases": debug_cases,
    }
    write_json(json_path, payload)

    def table(stats: dict[str, dict[str, Any]]) -> list[str]:
        lines = [
            "| group | count | mismatch_mean | mismatch_p95 | high_mismatch_rate | uv_valid_in_mask_rate | recommendation |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for group, item in stats.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        group,
                        str(item.get("count")),
                        fmt_float(item.get("mismatch_mean")),
                        fmt_float(item.get("mismatch_p95")),
                        fmt_float(item.get("high_mismatch_rate")),
                        fmt_float(item.get("uv_valid_in_mask_rate_mean")),
                        source_recommendation(item),
                    ]
                )
                + " |"
            )
        return lines

    lines = [
        "# Target/View Alignment Audit",
        "",
        "## Verdict",
        "",
        f"- sampled_target_stored_total_mae_mean: `{fmt_float(sampling_summary.get('sampled_target_stored_total_mae_mean'))}`",
        f"- sampled_target_stored_total_mae_p95: `{fmt_float(sampling_summary.get('sampled_target_stored_total_mae_p95'))}`",
        f"- uv_valid_in_mask_rate_mean: `{fmt_float(sampling_summary.get('uv_valid_in_mask_rate_mean'))}`",
        f"- y_flip_better_than_no_flip: `{sampling_summary.get('y_flip_better_than_no_flip')}`",
        f"- align_corners_true_better_than_false: `{sampling_summary.get('align_corners_true_better_than_false')}`",
        f"- channel_order_consistent: `{sampling_summary.get('channel_order_consistent')}`",
        "",
        "Stored view targets are not aligned with UV targets on this subset. Training should not use stored view target PNGs.",
        "",
        "## Recommended Fix",
        "",
        "- Rebuild or invalidate stored per-view RM targets for affected sources.",
        "- For R-v2.1 view-aware loss, use sampled UV target through `view_uv` as the consistent target.",
        "- Keep affected high-mismatch sources diagnostic-only until a rebake proves alignment.",
        "- Rebuild the acceptance subset with `target_prior_identity_mean <= 0.30` before any paper-facing claim.",
        "",
        "## by source_name",
        "",
        *table(by_source),
        "",
        "## by material_family",
        "",
        *table(by_material),
        "",
        "## by prior_source_type",
        "",
        *table(by_prior_source),
        "",
        "## Top Alignment Mismatch Cases",
        "",
        "| object_id | view_name | source | material | prior_source_type | mismatch | uv_valid_in_mask |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in debug_cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("object_id")),
                    str(row.get("view_name")),
                    str(row.get("source_name")),
                    str(row.get("material_family")),
                    str(row.get("prior_source_type")),
                    fmt_float(row.get("sampled_target_stored_total_mae")),
                    fmt_float(row.get("uv_valid_in_mask_rate")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_sampling(
    *,
    view_cases: list[dict[str, Any]],
    sampling_values: dict[str, list[float]],
    metric_debug_compare: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "view_rows": len(view_cases),
        "uv_u_min": optional_mean(sampling_values["uv_u_min"]),
        "uv_u_max": optional_mean(sampling_values["uv_u_max"]),
        "uv_v_min": optional_mean(sampling_values["uv_v_min"]),
        "uv_v_max": optional_mean(sampling_values["uv_v_max"]),
        "uv_valid_rate_mean": optional_mean(sampling_values["uv_valid_rate"]),
        "uv_valid_in_mask_rate_mean": optional_mean(sampling_values["uv_valid_in_mask_rate"]),
        "mask_pixel_rate_mean": optional_mean(sampling_values["mask_pixel_rate"]),
        "sampled_target_stored_total_mae_mean": optional_mean(sampling_values["target_mismatch"]),
        "sampled_target_stored_total_mae_p95": percentile(sampling_values["target_mismatch"], 95),
        "target_mismatch_y_flip_align_true_mean": optional_mean(sampling_values["target_mismatch"]),
        "target_mismatch_no_y_flip_align_true_mean": optional_mean(sampling_values["target_mismatch_no_flip"]),
        "target_mismatch_y_flip_align_false_mean": optional_mean(sampling_values["target_mismatch_align_false"]),
        "target_mismatch_channel_swapped_mean": optional_mean(sampling_values["target_mismatch_swapped"]),
        "rough_channel_mismatch_mean": optional_mean(sampling_values["target_mismatch_rough"]),
        "metal_channel_mismatch_mean": optional_mean(sampling_values["target_mismatch_metal"]),
        "metric_debug_compare": metric_debug_compare,
    }
    y_flip = finite_float(summary["target_mismatch_y_flip_align_true_mean"])
    no_flip = finite_float(summary["target_mismatch_no_y_flip_align_true_mean"])
    align_false = finite_float(summary["target_mismatch_y_flip_align_false_mean"])
    swapped = finite_float(summary["target_mismatch_channel_swapped_mean"])
    summary["y_flip_better_than_no_flip"] = (
        None if y_flip is None or no_flip is None else bool(y_flip <= no_flip)
    )
    summary["align_corners_true_better_than_false"] = (
        None if y_flip is None or align_false is None else bool(y_flip <= align_false)
    )
    summary["channel_order_consistent"] = (
        None if y_flip is None or swapped is None else bool(y_flip <= swapped)
    )
    return summary


def percentile(values: list[Any], pct: float) -> float | None:
    finite_values = [finite_float(value) for value in values]
    finite_values = [value for value in finite_values if value is not None]
    if not finite_values:
        return None
    return float(np.percentile(finite_values, pct))


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = load_json(args.summary)
    metric_debug_rows = load_csv_rows(args.metric_debug_csv)
    eval_rows = load_json_rows(args.rows)

    if args.checkpoint is None or not args.checkpoint.exists():
        raise SystemExit(f"checkpoint_not_found:{args.checkpoint}")
    if args.manifest is None or not args.manifest.exists():
        raise SystemExit(f"manifest_not_found:{args.manifest}")

    pipeline = MaterialRefinementPipeline.from_checkpoint(
        args.checkpoint,
        device=args.device,
        cuda_device_index=args.cuda_device_index,
        model_cfg_overrides=build_model_cfg_overrides(args),
    )
    dataset = build_dataset(args, pipeline.atlas_size, pipeline.buffer_resolution)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or 1),
        shuffle=False,
        num_workers=int(args.num_workers or 0),
        pin_memory=False,
        collate_fn=collate_material_samples,
    )
    aggregates: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"uv_baseline": [], "uv_refined": [], "view_baseline": [], "view_refined": []}
    )
    sampling_values: dict[str, list[float]] = defaultdict(list)
    view_cases: list[dict[str, Any]] = []
    object_uv_metrics: dict[str, dict[str, float]] = {}
    prior_identity_row_diffs: list[float] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            outputs = detach_outputs(pipeline.model(move_batch_to_device(batch, pipeline.device)))
            baseline = outputs.get("input_prior", outputs.get("baseline"))
            refined = outputs["refined"]
            rm_init = outputs.get("rm_init", outputs.get("initial", baseline))
            prior_init_outputs = outputs.get("prior_init_outputs") or {}
            bootstrap = (
                prior_init_outputs.get("bootstrap_rm_uv")
                if isinstance(prior_init_outputs, dict)
                else None
            )
            if not isinstance(bootstrap, torch.Tensor):
                bootstrap = rm_init
            target = torch.cat([batch["uv_target_roughness"], batch["uv_target_metallic"]], dim=1)
            constant = torch.zeros_like(target)
            constant[:, 0:1] = 0.5
            pred_modes = {
                "prior_as_pred": baseline,
                "target_as_pred": target,
                "constant_as_pred": constant,
                "model_pred": refined,
            }
            if batch.get("view_targets") is None or batch.get("view_uvs") is None:
                continue
            sampled_modes = {
                mode: sample_uv_maps_to_view(pred_map, batch["view_uvs"])
                for mode, pred_map in pred_modes.items()
            }
            sampled_bootstrap = sample_uv_maps_to_view(bootstrap, batch["view_uvs"])
            sampled_baseline = sampled_modes["prior_as_pred"]
            sampled_target = sampled_modes["target_as_pred"]
            sampled_refined = sampled_modes["model_pred"]
            sampled_target_no_flip = sample_uv_maps_to_view_options(
                target,
                batch["view_uvs"],
                y_flip=False,
                align_corners=True,
            )
            sampled_target_align_false = sample_uv_maps_to_view_options(
                target,
                batch["view_uvs"],
                y_flip=True,
                align_corners=False,
            )

            print(
                f"[view-audit] batch={batch_index}/{len(loader)} records={len(batch['object_id'])}",
                flush=True,
            )
            for item_idx, object_id in enumerate(batch["object_id"]):
                metadata = batch_metadata(batch, item_idx)
                confidence = batch["uv_target_confidence"][item_idx]
                target_item = target[item_idx]
                baseline_uv = uv_total_error(baseline[item_idx], target_item, confidence)[2]
                refined_uv = uv_total_error(refined[item_idx], target_item, confidence)[2]
                bootstrap_uv = uv_total_error(bootstrap[item_idx], target_item, confidence)[2]
                target_uv = uv_total_error(target_item, target_item, confidence)[2]
                constant_uv = uv_total_error(constant[item_idx], target_item, confidence)[2]
                uv_gain = baseline_uv - refined_uv
                object_uv_metrics[str(object_id)] = {
                    "uv_prior_error": baseline_uv,
                    "uv_pred_error": refined_uv,
                    "uv_gain": uv_gain,
                    "uv_bootstrap_error": bootstrap_uv,
                    "uv_target_error": target_uv,
                    "uv_constant_error": constant_uv,
                }
                add_mode_value(
                    aggregates,
                    "prior_as_pred",
                    uv_baseline=baseline_uv,
                    uv_refined=baseline_uv,
                )
                add_mode_value(
                    aggregates,
                    "target_as_pred",
                    uv_baseline=baseline_uv,
                    uv_refined=target_uv,
                )
                add_mode_value(
                    aggregates,
                    "constant_as_pred",
                    uv_baseline=baseline_uv,
                    uv_refined=constant_uv,
                )
                add_mode_value(
                    aggregates,
                    "model_pred",
                    uv_baseline=baseline_uv,
                    uv_refined=refined_uv,
                )
                if metadata["prior_label"] == "without_prior":
                    add_mode_value(
                        aggregates,
                        "bootstrap_as_pred",
                        uv_baseline=baseline_uv,
                        uv_refined=bootstrap_uv,
                    )

                for view_idx, view_name in enumerate(batch["view_names"][item_idx]):
                    mask = batch["view_masks"][item_idx, view_idx, 0] > 0.5
                    if not bool(mask.any()):
                        continue
                    gt_view = batch["view_targets"][item_idx, view_idx]
                    view_uv = batch["view_uvs"][item_idx, view_idx]
                    valid_map = view_uv_valid_map(view_uv)
                    mask_count = float(mask.sum().item()) or 1.0
                    valid_in_mask = float((valid_map & mask).sum().item() / mask_count)
                    sampling_values["uv_valid_in_mask_rate"].append(valid_in_mask)
                    sampling_values["uv_valid_rate"].append(float(valid_map.float().mean().item()))
                    sampling_values["mask_pixel_rate"].append(float(mask.float().mean().item()))
                    finite_uv = torch.where(torch.isfinite(view_uv), view_uv, torch.zeros_like(view_uv))
                    sampling_values["uv_u_min"].append(float(finite_uv[..., 0].min().item()))
                    sampling_values["uv_u_max"].append(float(finite_uv[..., 0].max().item()))
                    sampling_values["uv_v_min"].append(float(finite_uv[..., 1].min().item()))
                    sampling_values["uv_v_max"].append(float(finite_uv[..., 1].max().item()))

                    baseline_view_error = masked_view_total_error(sampled_baseline[item_idx, view_idx], gt_view, mask)[2]
                    refined_view_error = masked_view_total_error(sampled_refined[item_idx, view_idx], gt_view, mask)[2]
                    target_view_error = masked_view_total_error(sampled_target[item_idx, view_idx], gt_view, mask)[2]
                    target_no_flip_error = masked_view_total_error(sampled_target_no_flip[item_idx, view_idx], gt_view, mask)[2]
                    target_align_false_error = masked_view_total_error(sampled_target_align_false[item_idx, view_idx], gt_view, mask)[2]
                    swapped_target = torch.stack(
                        [sampled_target[item_idx, view_idx, 1], sampled_target[item_idx, view_idx, 0]],
                        dim=0,
                    )
                    target_swapped_error = masked_view_total_error(swapped_target, gt_view, mask)[2]
                    target_rough_error = float(
                        (sampled_target[item_idx, view_idx, 0][mask] - gt_view[0][mask]).abs().mean().item()
                    )
                    target_metal_error = float(
                        (sampled_target[item_idx, view_idx, 1][mask] - gt_view[1][mask]).abs().mean().item()
                    )
                    sampling_values["target_mismatch"].append(target_view_error)
                    sampling_values["target_mismatch_no_flip"].append(target_no_flip_error)
                    sampling_values["target_mismatch_align_false"].append(target_align_false_error)
                    sampling_values["target_mismatch_swapped"].append(target_swapped_error)
                    sampling_values["target_mismatch_rough"].append(target_rough_error)
                    sampling_values["target_mismatch_metal"].append(target_metal_error)

                    for mode, sampled in sampled_modes.items():
                        mode_view_error = masked_view_total_error(sampled[item_idx, view_idx], gt_view, mask)[2]
                        add_mode_value(
                            aggregates,
                            mode,
                            view_baseline=baseline_view_error,
                            view_refined=mode_view_error,
                        )
                        if mode == "prior_as_pred":
                            prior_identity_row_diffs.append(abs(baseline_view_error - mode_view_error))
                    if metadata["prior_label"] == "without_prior":
                        bootstrap_view_error = masked_view_total_error(
                            sampled_bootstrap[item_idx, view_idx],
                            gt_view,
                            mask,
                        )[2]
                        add_mode_value(
                            aggregates,
                            "bootstrap_as_pred",
                            view_baseline=baseline_view_error,
                            view_refined=bootstrap_view_error,
                        )

                    case = {
                        "object_id": str(object_id),
                        "view_name": str(view_name),
                        **metadata,
                        "uv_gain": uv_gain,
                        "view_gain": baseline_view_error - refined_view_error,
                        "uv_prior_error": baseline_uv,
                        "uv_pred_error": refined_uv,
                        "view_prior_error": baseline_view_error,
                        "view_pred_error": refined_view_error,
                        "target_prior_identity_bucket": target_identity_bucket(metadata["target_prior_identity"]),
                        "sampled_target_stored_total_mae": target_view_error,
                        "sampled_target_stored_roughness_mae": target_rough_error,
                        "sampled_target_stored_metallic_mae": target_metal_error,
                        "target_mismatch_no_y_flip": target_no_flip_error,
                        "target_mismatch_align_corners_false": target_align_false_error,
                        "target_mismatch_channel_swapped": target_swapped_error,
                        "uv_valid_in_mask_rate": valid_in_mask,
                        "uv_valid_rate": float(valid_map.float().mean().item()),
                        "mask_pixel_rate": float(mask.float().mean().item()),
                    }
                    view_cases.append(case)

    metric_debug_compare = compare_with_metric_debug(view_cases, metric_debug_rows)
    sampling_summary = summarize_sampling(
        view_cases=view_cases,
        sampling_values=sampling_values,
        metric_debug_compare=metric_debug_compare,
    )
    identity_rows = build_identity_summary(
        aggregates=aggregates,
        summary_payload=summary_payload,
        args=args,
        sampling_summary=sampling_summary,
    )

    sampling_summary["prior_as_pred_max_row_abs_diff"] = max(prior_identity_row_diffs) if prior_identity_row_diffs else None
    sampling_summary["eval_rows_json_count"] = len(eval_rows)
    sampling_summary["metric_debug_csv_count"] = len(metric_debug_rows)
    sampling_summary["object_uv_metric_count"] = len(object_uv_metrics)

    write_json(args.output_dir / "view_sampling_audit_summary.json", sampling_summary)
    write_json(args.output_dir / "view_identity_test_summary.json", {"rows": identity_rows})
    write_identity_markdown(args.output_dir / "view_identity_test_summary.md", identity_rows, sampling_summary)
    write_csv(
        args.output_dir / "view_identity_test_summary.csv",
        identity_rows,
        [
            "mode",
            "uv_input_prior_total_mae",
            "uv_total_mae",
            "view_input_prior_total_mae",
            "view_total_mae",
            "gain_vs_input_prior",
            "improvement_rate",
            "regression_rate",
            "row_count",
            "identity_pass",
            "notes",
        ],
    )
    write_csv(args.output_dir / "view_case_metrics.csv", view_cases, CASE_COLUMNS)

    debug_cases = sorted(
        view_cases,
        key=lambda row: finite_float(row.get("sampled_target_stored_total_mae")) or -1.0,
        reverse=True,
    )[: int(args.debug_case_count)]
    sampling_panel_dir = args.output_dir / "view_sampling_debug_panels"
    sampling_panel_paths = find_cases_in_dataset(
        selected=debug_cases,
        dataset=dataset,
        pipeline=pipeline,
        args=args,
        output_dir=sampling_panel_dir,
        include_sampling_tiles=True,
    )
    for path in sampling_panel_paths:
        target = args.output_dir / path.name
        if not target.exists():
            target.symlink_to(path.relative_to(args.output_dir))
    build_html_index(
        rows=debug_cases,
        panel_paths=[args.output_dir / path.name for path in sampling_panel_paths],
        output_path=args.output_dir / "view_sampling_debug_index.html",
        title="View Sampling Debug Panels",
        intro=(
            "Panels are ranked by sampled UV target vs stored view target mismatch. "
            "Large error here points to target-space or UV-to-view sampling inconsistency."
        ),
    )

    selected_conflicts = select_conflict_cases(view_cases, int(args.case_count))
    category_outputs: dict[str, dict[str, Path]] = {}
    conflict_root = args.output_dir / "conflict_cases"
    for name, rows in selected_conflicts.items():
        category_dir = conflict_root / name
        csv_path = category_dir / f"{name}.csv"
        write_csv(csv_path, rows, CASE_COLUMNS)
        panel_dir = category_dir / "panels"
        panel_paths = find_cases_in_dataset(
            selected=rows,
            dataset=dataset,
            pipeline=pipeline,
            args=args,
            output_dir=panel_dir,
            include_sampling_tiles=False,
        )
        for panel_path in panel_paths:
            target = category_dir / panel_path.name
            if not target.exists():
                target.symlink_to(Path("panels") / panel_path.name)
        html_path = category_dir / f"{name}.html"
        build_html_index(
            rows=rows,
            panel_paths=[category_dir / path.name for path in panel_paths],
            output_path=html_path,
            title=name,
            intro="Top view regression cases with a view-space row: input RGB, sampled prior/GT/pred RM views, errors, and visibility.",
        )
        category_outputs[name] = {"csv": csv_path, "html": html_path}

    write_conflict_markdown(
        path=args.output_dir / "uv_view_conflict_cases.md",
        selected=selected_conflicts,
        category_outputs=category_outputs,
        all_rows=view_cases,
    )
    write_target_view_alignment_audit(
        path=args.output_dir / "target_view_alignment_audit.md",
        json_path=args.output_dir / "target_view_alignment_audit.json",
        view_cases=view_cases,
        sampling_summary=sampling_summary,
        debug_cases=debug_cases,
    )

    root_cause = build_root_cause_report(
        args=args,
        summary_payload=summary_payload,
        identity_rows=identity_rows,
        sampling_summary=sampling_summary,
        selected_conflicts=selected_conflicts,
        view_cases=view_cases,
    )
    write_json(args.output_dir / "view_degradation_root_cause.json", root_cause)
    (args.output_dir / "view_degradation_root_cause.md").write_text(
        root_cause["markdown"],
        encoding="utf-8",
    )
    result = {
        "summary": {
            "identity_rows": identity_rows,
            "sampling_summary": sampling_summary,
            "conflict_case_counts": {key: len(value) for key, value in selected_conflicts.items()},
            "root_cause_classification": root_cause["classification"],
        },
        "paths": {
            "view_identity_test_summary": str((args.output_dir / "view_identity_test_summary.md").resolve()),
            "view_sampling_debug_index": str((args.output_dir / "view_sampling_debug_index.html").resolve()),
            "target_view_alignment_audit": str((args.output_dir / "target_view_alignment_audit.md").resolve()),
            "uv_view_conflict_cases": str((args.output_dir / "uv_view_conflict_cases.md").resolve()),
            "view_degradation_root_cause": str((args.output_dir / "view_degradation_root_cause.md").resolve()),
        },
    }
    write_json(args.output_dir / "audit_run_summary.json", result)
    write_json(args.output_dir / "audit_summary.json", result)
    return result


def identity_row(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    for row in rows:
        if row.get("mode") == mode:
            return row
    return {}


def top_distribution(rows: list[dict[str, Any]], key: str, limit: int = 5) -> str:
    if not rows:
        return "none"
    counts = Counter(str(row.get(key, "unknown")) for row in rows)
    return ", ".join(f"{name}={count}" for name, count in counts.most_common(limit))


def classify_root_cause(
    identity_rows: list[dict[str, Any]],
    sampling_summary: dict[str, Any],
) -> str:
    prior_pass = bool(identity_row(identity_rows, "prior_as_pred").get("identity_pass"))
    target_pass = bool(identity_row(identity_rows, "target_as_pred").get("identity_pass"))
    if not prior_pass:
        return "view_metric_baseline_refined_identity_bug"
    if not target_pass:
        return "uv_view_target_or_sampling_mismatch"
    return "r_v2_output_view_regression"


def build_root_cause_report(
    *,
    args: argparse.Namespace,
    summary_payload: dict[str, Any],
    identity_rows: list[dict[str, Any]],
    sampling_summary: dict[str, Any],
    selected_conflicts: dict[str, list[dict[str, Any]]],
    view_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    prior = identity_row(identity_rows, "prior_as_pred")
    target = identity_row(identity_rows, "target_as_pred")
    bootstrap = identity_row(identity_rows, "bootstrap_as_pred")
    constant = identity_row(identity_rows, "constant_as_pred")
    model = identity_row(identity_rows, "model_pred")
    classification = classify_root_cause(identity_rows, sampling_summary)

    target_prior_identity_mean = None
    dataset_summary = summary_payload.get("dataset_summary") or {}
    manifest_summary = load_json(args.manifest).get("summary") if args.manifest and args.manifest.exists() else None
    if isinstance(manifest_summary, dict):
        target_prior_identity_mean = manifest_summary.get("target_prior_identity_mean")
    if target_prior_identity_mean is None:
        values = [finite_float(row.get("target_prior_identity")) for row in view_cases]
        values = [value for value in values if value is not None]
        target_prior_identity_mean = float(np.mean(values)) if values else None

    uv_gain = summary_payload.get("gain_total")
    view_gain = (summary_payload.get("view_level") or {}).get("gain_total")
    view_regression_rate = (summary_payload.get("view_level") or {}).get("regression_rate")
    constant_better_than_model = (
        finite_float(constant.get("view_total_mae")) is not None
        and finite_float(model.get("view_total_mae")) is not None
        and float(constant["view_total_mae"]) < float(model["view_total_mae"])
    )
    uv_target_view_consistent = bool(target.get("identity_pass"))
    prior_identity_pass = bool(prior.get("identity_pass"))
    target_identity_pass = bool(target.get("identity_pass"))
    conflict_rows = selected_conflicts.get("uv_improved_view_regressed_top", [])
    without_prior_rows = selected_conflicts.get("without_prior_view_regressed_top", [])
    with_prior_rows = selected_conflicts.get("with_prior_view_regressed_top", [])

    continue_training = bool(
        prior_identity_pass
        and target_identity_pass
        and finite_float(uv_gain) is not None
        and float(uv_gain) > 0.0
        and (
            finite_float(view_gain) is not None
            and (float(view_gain) >= 0.0 or (finite_float(view_regression_rate) or 1.0) < 0.50)
        )
        and finite_float(target_prior_identity_mean) is not None
        and float(target_prior_identity_mean) <= 0.30
    )
    if classification == "uv_view_target_or_sampling_mismatch":
        view_loss_recommendation = "No. Fix view eval/data target consistency before adding a training loss."
    elif classification == "r_v2_output_view_regression":
        view_loss_recommendation = "Yes, consider moving view_consistency_loss earlier after confirming identity tests stay green."
    else:
        view_loss_recommendation = "No. Fix the view metric identity bug first."

    rebuild_subset = bool(finite_float(target_prior_identity_mean) is not None and float(target_prior_identity_mean) > 0.30)
    markdown = "\n".join(
        [
            "# View Degradation Root Cause",
            "",
            "## Verdict",
            "",
            f"- classification: `{classification}`",
            f"- continue_training_allowed: `{continue_training}`",
            f"- uv_gain_total: `{fmt_float(uv_gain)}`",
            f"- view_gain_total: `{fmt_float(view_gain)}`",
            f"- view_regression_rate: `{fmt_float(view_regression_rate)}`",
            f"- target_prior_identity_mean: `{fmt_float(target_prior_identity_mean)}`",
            "",
            "## Required Answers",
            "",
            f"1. UV target and view target consistent? `{uv_target_view_consistent}`. "
            f"sampled_target_vs_stored_mean=`{fmt_float(sampling_summary.get('sampled_target_stored_total_mae_mean'))}`, "
            f"p95=`{fmt_float(sampling_summary.get('sampled_target_stored_total_mae_p95'))}`.",
            f"2. prior_as_pred reproduces input-prior view metric? `{prior_identity_pass}`. "
            f"view_total=`{fmt_float(prior.get('view_total_mae'))}`, "
            f"max_row_abs_diff=`{fmt_float(sampling_summary.get('prior_as_pred_max_row_abs_diff'))}`.",
            f"3. target_as_pred reaches near view-space optimum? `{target_identity_pass}`. "
            f"view_total=`{fmt_float(target.get('view_total_mae'))}`, "
            f"gain_vs_input_prior=`{fmt_float(target.get('gain_vs_input_prior'))}`.",
            f"4. Root cause: `{classification}`. "
            f"constant_better_than_model=`{constant_better_than_model}` "
            f"(constant_view=`{fmt_float(constant.get('view_total_mae'))}`, "
            f"model_view=`{fmt_float(model.get('view_total_mae'))}`).",
            f"5. Concentration: uv/view conflict source distribution `{top_distribution(conflict_rows, 'source_name')}`; "
            f"material distribution `{top_distribution(conflict_rows, 'material_family')}`; "
            f"without_prior_top_count=`{len(without_prior_rows)}`, with_prior_top_count=`{len(with_prior_rows)}`.",
            f"6. Allow continued training? `{continue_training}`. Phase4 must stay false until identity, view direction, and identity-mean gates pass.",
            f"7. Add view_consistency_loss now? {view_loss_recommendation}",
            f"8. Rebuild acceptance subset to lower target_prior_identity_mean? `{rebuild_subset}`.",
            "",
            "## Identity Tests",
            "",
            "| mode | uv_total_mae | view_total_mae | gain_vs_input_prior | improvement_rate | regression_rate | identity_pass |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                "| "
                + " | ".join(
                    [
                        str(row.get("mode")),
                        fmt_float(row.get("uv_total_mae")),
                        fmt_float(row.get("view_total_mae")),
                        fmt_float(row.get("gain_vs_input_prior")),
                        fmt_float(row.get("improvement_rate")),
                        fmt_float(row.get("regression_rate")),
                        str(row.get("identity_pass")),
                    ]
                )
                + " |"
                for row in identity_rows
            ],
            "",
            "## Sampling Checks",
            "",
            f"- y_flip_better_than_no_flip: `{sampling_summary.get('y_flip_better_than_no_flip')}`",
            f"- align_corners_true_better_than_false: `{sampling_summary.get('align_corners_true_better_than_false')}`",
            f"- channel_order_consistent: `{sampling_summary.get('channel_order_consistent')}`",
            f"- uv_valid_in_mask_rate_mean: `{fmt_float(sampling_summary.get('uv_valid_in_mask_rate_mean'))}`",
            f"- metric_debug_baseline_abs_diff_max: `{fmt_float((sampling_summary.get('metric_debug_compare') or {}).get('baseline_abs_diff_max'))}`",
            f"- metric_debug_refined_abs_diff_max: `{fmt_float((sampling_summary.get('metric_debug_compare') or {}).get('refined_abs_diff_max'))}`",
            "",
            "## Linked Artifacts",
            "",
            "- view_identity_test_summary.md",
            "- view_sampling_debug_index.html",
            "- uv_view_conflict_cases.md",
        ]
    )
    return {
        "classification": classification,
        "continue_training_allowed": continue_training,
        "target_prior_identity_mean": target_prior_identity_mean,
        "constant_better_than_model": constant_better_than_model,
        "view_consistency_loss_recommendation": view_loss_recommendation,
        "rebuild_acceptance_subset": rebuild_subset,
        "markdown": markdown,
    }


def main() -> None:
    args = build_parser().parse_args()
    eval_args = load_json(args.eval_dir / "eval_args.json")
    apply_eval_arg_defaults(args, eval_args)
    args.manifest = Path(args.manifest)
    args.checkpoint = Path(args.checkpoint)
    args.summary = Path(args.summary)
    args.rows = Path(args.rows)
    args.metric_debug_csv = Path(args.metric_debug_csv)
    result = run_audit(args)
    print(json.dumps(make_json_serializable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
