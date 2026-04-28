from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine import (  # noqa: E402
    CanonicalMaterialDataset,
    MaterialRefinementPipeline,
    collate_material_samples,
)
from sf3d.material_refine.data_utils import summarize_records  # noqa: E402
from sf3d.material_refine.experiment import (  # noqa: E402
    flatten_for_logging,
    log_path_artifact,
    make_json_serializable,
    maybe_init_wandb,
    sanitize_log_dict,
    wandb,
)
from sf3d.material_refine.io import save_atlas_bundle  # noqa: E402
from sf3d.material_refine.io import tensor_to_pil  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "material_refiner_eval"
DEFAULT_MANIFEST = REPO_ROOT / "output" / "material_refine" / "canonical_manifest_v1.json"
FAILURE_TAGS = [
    "over_smoothing",
    "metal_nonmetal_confusion",
    "local_highlight_misread",
    "boundary_bleed",
]
METAL_THRESHOLD = 0.20
GROUP_DIAGNOSTIC_MIN_COUNT = 16
MAIN_METRIC_NAMES = [
    "uv_rm_mae",
    "view_rm_mae",
    "proxy_render_mse",
    "proxy_render_psnr",
    "proxy_render_ssim",
    "proxy_render_lpips",
]
MATERIAL_SPECIFIC_METRIC_NAMES = [
    "boundary_bleed_score",
    "metal_nonmetal_confusion",
    "highlight_localization_error",
    "rm_gradient_preservation",
    "prior_residual_safety",
    "confidence_calibrated_error",
    "material_family_breakdown",
]
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
PAPER_MAIN_VARIANT_LABELS = {
    "ours_full": "Ours Full",
    "prior_smoothing": "Prior Smoothing",
    "scalar_broadcast": "Scalar Broadcast",
    "no_view_refiner": "Ours w/o View",
    "no_residual_refiner": "Ours w/o Residual",
}
PAPER_MAIN_METRIC_COLUMNS = ["uv_rm_mae", "view_rm_mae", "psnr", "ssim", "lpips"]


def first_nonempty(*values: Any, default: str = "unknown") -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return str(default)


def batch_item_value(
    batch: dict[str, Any],
    metadata: dict[str, Any],
    key: str,
    item_idx: int,
    default: Any = "unknown",
) -> Any:
    value = batch.get(key)
    if isinstance(value, (list, tuple)) and item_idx < len(value):
        item = value[item_idx]
        if item not in (None, ""):
            return item
    if isinstance(metadata, dict):
        item = metadata.get(key)
        if item not in (None, ""):
            return item
    return default


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


def material_case_id(
    *,
    object_id: Any,
    prior_variant_type: Any,
    pair_id: Any,
    prior_variant_id: Any,
    fallback_index: int | None = None,
) -> str:
    suffix = first_case_identity(pair_id, prior_variant_id, fallback_index)
    return "__".join(
        [
            safe_path_component(object_id),
            safe_path_component(prior_variant_type),
            safe_path_component(suffix, max_length=128),
        ]
    )


def paper_main_metric_row(
    *,
    method: str,
    side: str,
    metrics_main: dict[str, Any],
    metric_basis: str,
    note: str = "",
) -> dict[str, Any]:
    return {
        "method": method,
        "metric_side": side,
        "metric_basis": metric_basis,
        "note": note,
        "uv_rm_mae": (metrics_main.get("uv_rm_mae") or {}).get("total", {}).get(side),
        "view_rm_mae": (metrics_main.get("view_rm_mae") or {}).get("total", {}).get(side),
        "psnr": (metrics_main.get("proxy_render_psnr") or {}).get(side),
        "ssim": (metrics_main.get("proxy_render_ssim") or {}).get(side),
        "lpips": (metrics_main.get("proxy_render_lpips") or {}).get(side),
    }


def build_paper_main_table_entries(eval_variant: str, metrics_main: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if eval_variant == "ours_full":
        rows.append(
            paper_main_metric_row(
                method="Original Generator Asset",
                side="baseline",
                metrics_main=metrics_main,
                metric_basis="original_asset_proxy_if_available",
                note=(
                    "R-only metric values use the input prior proxy unless a raw original generator render is exported."
                ),
            )
        )
    rows.append(
        paper_main_metric_row(
            method="Input Prior",
            side="baseline",
            metrics_main=metrics_main,
            metric_basis="canonical_uv_prior",
            note="Canonical UV/view metric baseline from uv_prior_roughness and uv_prior_metallic.",
        )
    )
    variant_label = PAPER_MAIN_VARIANT_LABELS.get(eval_variant)
    if variant_label:
        rows.append(
            paper_main_metric_row(
                method=variant_label,
                side="refined",
                metrics_main=metrics_main,
                metric_basis="eval_variant_refined_output",
            )
        )
    return rows


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


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",")]
    items = [item for item in items if item]
    return items or None


def load_config_defaults(config_paths: list[Path]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for config_path in config_paths:
        payload = OmegaConf.to_container(
            OmegaConf.load(config_path),
            resolve=True,
        )
        if not isinstance(payload, dict):
            raise TypeError(f"unsupported_config_payload:{config_path}")
        for key, value in payload.items():
            if isinstance(value, dict):
                continue
            defaults[key] = value
    return defaults


def collect_config_paths(args: argparse.Namespace) -> list[Path]:
    config_paths = []
    for attr in ("method_config", "data_config", "config"):
        value = getattr(args, attr, None)
        if not value:
            continue
        if isinstance(value, list):
            config_paths.extend(value)
        else:
            config_paths.append(value)
    return config_paths


def build_parser(config_defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate the material refiner and export NG-style summary/report artifacts.",
    )
    parser.add_argument("--config", action="append", type=Path, default=[])
    parser.add_argument("--method-config", action="append", type=Path, default=[])
    parser.add_argument("--data-config", action="append", type=Path, default=[])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--split-strategy", choices=["auto", "manifest", "hash"], default="auto")
    parser.add_argument("--hash-val-ratio", type=float, default=0.1)
    parser.add_argument("--hash-test-ratio", type=float, default=0.1)
    parser.add_argument("--generator-ids", type=str, default=None)
    parser.add_argument("--source-names", type=str, default=None)
    parser.add_argument("--supervision-tiers", type=str, default=None)
    parser.add_argument("--supervision-roles", type=str, default=None)
    parser.add_argument("--license-buckets", type=str, default=None)
    parser.add_argument("--target-quality-tiers", type=str, default=None)
    parser.add_argument("--paper-splits", type=str, default=None)
    parser.add_argument("--material-families", type=str, default=None)
    parser.add_argument("--lighting-bank-ids", type=str, default=None)
    parser.add_argument("--require-prior", type=str, default="any")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--view-sample-count", type=int, default=0)
    parser.add_argument("--min-hard-views", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10, help="Print NG-style eval progress every N batches.")
    parser.add_argument("--cuda-device-index", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--eval-variant",
        choices=[
            "ours_full",
            "no_prior_refiner",
            "no_residual_refiner",
            "no_view_refiner",
            "scalar_broadcast",
            "prior_smoothing",
        ],
        default="ours_full",
    )
    parser.add_argument("--prior-smoothing-kernel", type=int, default=9)
    parser.add_argument("--enable-prior-source-embedding", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-no-prior-bootstrap", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-boundary-safety", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-change-gate", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-material-aux-head", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-render-proxy-loss", type=parse_optional_bool, default=None)
    parser.add_argument(
        "--render-metric-mode",
        choices=["proxy_uv_shading", "disabled"],
        default="proxy_uv_shading",
        help="How to obtain rendered RGB metrics before full Blender re-render integration.",
    )
    parser.add_argument(
        "--max-artifact-objects",
        type=int,
        default=24,
        help="Maximum number of objects for which atlas/view images are written. 0 keeps full legacy output.",
    )
    parser.add_argument(
        "--enable-lpips",
        type=parse_bool,
        default=True,
        help="Compute LPIPS when the optional lpips package is installed.",
    )
    parser.add_argument(
        "--diagnostic-min-group-count",
        type=int,
        default=GROUP_DIAGNOSTIC_MIN_COUNT,
        help="Groups smaller than this are marked diagnostic-only in reports.",
    )
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument(
        "--print-summary-json",
        type=parse_bool,
        default=False,
        help="Print the full summary JSON to stdout. Defaults off to keep W&B console usable.",
    )
    parser.add_argument("--report-to", choices=["none", "wandb"], default="wandb")
    parser.add_argument("--tracker-project-name", type=str, default="stable-fast-3d-material-refine")
    parser.add_argument("--tracker-run-name", type=str, default=None)
    parser.add_argument("--tracker-group", type=str, default="material-refine-eval")
    parser.add_argument("--tracker-tags", type=str, default="material-refine,eval")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-resume-id", type=str, default=None)
    parser.add_argument("--wandb-resume-mode", type=str, default="allow")
    parser.add_argument("--wandb-max-rows", type=int, default=128)
    parser.add_argument(
        "--wandb-log-top-cases",
        type=parse_bool,
        default=False,
        help="Upload the heavy eval/top_cases media table. Defaults off; cases stay local in diagnostic_cases.json.",
    )
    parser.add_argument(
        "--wandb-log-group-breakdowns",
        type=parse_bool,
        default=False,
        help=(
            "Upload compact material_family/prior_label diagnostics only. Defaults off; "
            "full group breakdowns stay in summary.json/report.html."
        ),
    )
    parser.add_argument(
        "--wandb-log-paper-main-table",
        type=parse_bool,
        default=False,
        help="Upload the compact paper main table to W&B. Defaults off; table stays in summary.json/report.html.",
    )
    parser.add_argument("--wandb-log-artifacts", type=parse_bool, default=True)
    parser.add_argument(
        "--wandb-artifact-policy",
        choices=["none", "summary", "full"],
        default="summary",
        help=(
            "Controls evaluation artifact upload volume. 'summary' uploads reports/JSON only; "
            "'full' also uploads the per-case artifact image tree."
        ),
    )
    parser.set_defaults(**config_defaults)
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", action="append", type=Path, default=[])
    pre_parser.add_argument("--method-config", action="append", type=Path, default=[])
    pre_parser.add_argument("--data-config", action="append", type=Path, default=[])
    pre_args, _ = pre_parser.parse_known_args()
    parser = build_parser(load_config_defaults(collect_config_paths(pre_args)))
    args = parser.parse_args()
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required either on the command line or in the config file")
    args.generator_ids = parse_csv_list(args.generator_ids)
    args.source_names = parse_csv_list(args.source_names)
    args.supervision_tiers = parse_csv_list(args.supervision_tiers)
    args.supervision_roles = parse_csv_list(args.supervision_roles)
    args.license_buckets = parse_csv_list(args.license_buckets)
    args.target_quality_tiers = parse_csv_list(args.target_quality_tiers)
    args.paper_splits = parse_csv_list(args.paper_splits)
    args.material_families = parse_csv_list(args.material_families)
    args.lighting_bank_ids = parse_csv_list(args.lighting_bank_ids)
    args.require_prior = parse_optional_bool(args.require_prior)
    return args


def format_duration(seconds: float | int | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "unknown"
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def safe_metric_key(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(":", "_")


def compact_eval_group_logs(
    summary_payload: dict[str, Any],
    *,
    max_groups_per_axis: int = 12,
) -> dict[str, Any]:
    logs: dict[str, Any] = {}
    group_axes = [
        ("material_family", "by_material_family"),
        ("prior_label", "by_prior_label"),
        ("prior_source_type", "by_prior_source_type"),
    ]
    metric_names = [
        "count",
        "refined_total_mae",
        "improvement_total",
        "refined_psnr",
        "refined_ssim",
        "refined_lpips",
        "refined_boundary_bleed_score",
        "refined_metal_confusion_rate",
        "prior_residual_safety_score",
    ]
    for axis_name, payload_key in group_axes:
        groups = summary_payload.get(payload_key) or {}
        if not isinstance(groups, dict):
            continue
        logs[f"eval/group/{axis_name}/group_count"] = len(groups)
        if len(groups) > max_groups_per_axis:
            logs[f"eval/group/{axis_name}/skipped_too_many_groups"] = 1.0
            continue
        for group_key, metrics in groups.items():
            if not isinstance(metrics, dict):
                continue
            safe_group = safe_metric_key(group_key)
            for metric_name in metric_names:
                value = metrics.get(metric_name)
                if value is not None:
                    logs[f"eval/group/{axis_name}/{safe_group}/{metric_name}"] = value
    return logs


def compact_eval_console_summary(summary_payload: dict[str, Any]) -> dict[str, Any]:
    main_metrics = summary_payload.get("metrics_main") or {}
    special_metrics = summary_payload.get("metrics_material_specific") or {}
    return {
        "eval_summary": {
            "rows": summary_payload.get("rows"),
            "objects": summary_payload.get("objects"),
            "baseline_uv_total_mae": summary_payload.get("baseline_total_mae"),
            "input_prior_uv_total_mae": summary_payload.get("input_prior_total_mae"),
            "refined_uv_total_mae": summary_payload.get("refined_total_mae"),
            "gain_total": summary_payload.get("gain_total"),
            "avg_improvement_total": summary_payload.get("avg_improvement_total"),
            "view_total_mae": (main_metrics.get("view_rm_mae") or {}).get("total"),
            "proxy_render_psnr": main_metrics.get("proxy_render_psnr"),
            "proxy_render_ssim": main_metrics.get("proxy_render_ssim"),
            "proxy_render_lpips": main_metrics.get("proxy_render_lpips"),
            "boundary_bleed_score": special_metrics.get("boundary_bleed_score"),
            "metal_nonmetal_view": (
                special_metrics.get("metal_nonmetal_confusion") or {}
            ).get("view_level"),
            "prior_residual_safety": special_metrics.get("prior_residual_safety"),
            "metric_disagreement": (
                summary_payload.get("metrics_diagnostics", {})
                .get("metric_disagreement", {})
                .get("has_disagreement")
            ),
            "warning_count": len(summary_payload.get("metric_warnings", [])),
            "runtime": summary_payload.get("runtime"),
            "summary_json": summary_payload.get("summary_json"),
        }
    }


def resolve_device(args: argparse.Namespace) -> str:
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return f"cuda:{args.cuda_device_index}"
    return "cpu"


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def masked_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    visible = values[mask]
    if visible.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "edge_mean": 0.0,
            "interior_mean": 0.0,
        }
    interior = np.asarray(
        Image.fromarray((mask.astype(np.uint8) * 255)).filter(ImageFilter.MinFilter(9))
    ) > 0
    edge = mask & ~interior
    return {
        "mean": float(visible.mean()),
        "std": float(visible.std()),
        "p10": float(np.quantile(visible, 0.10)),
        "p50": float(np.quantile(visible, 0.50)),
        "p90": float(np.quantile(visible, 0.90)),
        "edge_mean": float(values[edge].mean()) if edge.any() else float(visible.mean()),
        "interior_mean": float(values[interior].mean()) if interior.any() else float(visible.mean()),
    }


def classify_case(
    *,
    pred_roughness: float,
    pred_metallic: float,
    gt_roughness: dict[str, float],
    gt_metallic: dict[str, float],
    brightness_p99: float,
) -> tuple[list[str], str]:
    trigger_scores = {}
    rough_error = abs(pred_roughness - gt_roughness["mean"])
    metal_error = abs(pred_metallic - gt_metallic["mean"])
    rough_range = gt_roughness["p90"] - gt_roughness["p10"]
    rough_var = gt_roughness["std"] ** 2
    metal_mean = gt_metallic["mean"]

    if rough_range > 0.30 and rough_var > 0.01 and rough_error < 0.10:
        trigger_scores["over_smoothing"] = rough_range + rough_var * 4.0 - rough_error

    if metal_mean < 0.15 and pred_metallic > 0.35:
        trigger_scores["metal_nonmetal_confusion"] = pred_metallic - metal_mean
    elif metal_mean > 0.20 and pred_metallic < 0.08:
        trigger_scores["metal_nonmetal_confusion"] = metal_mean - pred_metallic

    if brightness_p99 > 0.82 and metal_mean < 0.30 and (rough_error > 0.12 or metal_error > 0.12):
        trigger_scores["local_highlight_misread"] = max(rough_error, metal_error) + (brightness_p99 - 0.82)

    metal_edge_gap = abs(gt_metallic["edge_mean"] - gt_metallic["interior_mean"])
    rough_edge_gap = abs(gt_roughness["edge_mean"] - gt_roughness["interior_mean"])
    if metal_edge_gap > 0.20:
        edge_dist = abs(pred_metallic - gt_metallic["edge_mean"])
        interior_dist = abs(pred_metallic - gt_metallic["interior_mean"])
        if edge_dist + 0.05 < interior_dist:
            trigger_scores["boundary_bleed"] = metal_edge_gap + (interior_dist - edge_dist)
    if rough_edge_gap > 0.20:
        edge_dist = abs(pred_roughness - gt_roughness["edge_mean"])
        interior_dist = abs(pred_roughness - gt_roughness["interior_mean"])
        if edge_dist + 0.05 < interior_dist:
            score = rough_edge_gap + (interior_dist - edge_dist)
            trigger_scores["boundary_bleed"] = max(trigger_scores.get("boundary_bleed", 0.0), score)

    tags = [tag for tag in FAILURE_TAGS if tag in trigger_scores]
    primary = max(tags, key=lambda item: trigger_scores[item]) if tags else "none"
    return tags, primary


def sample_uv_maps_to_view(uv_maps: torch.Tensor, view_uvs: torch.Tensor) -> torch.Tensor:
    grid = view_uvs.clone()
    grid[..., 0] = grid[..., 0] * 2.0 - 1.0
    grid[..., 1] = (1.0 - grid[..., 1]) * 2.0 - 1.0
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
        align_corners=True,
    )
    return sampled.view(batch, views, sampled.shape[1], height, width)


def view_uv_valid_mask(view_uv: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(view_uv).all(dim=-1)
    in_range = (
        (view_uv[..., 0] >= 0.0)
        & (view_uv[..., 0] <= 1.0)
        & (view_uv[..., 1] >= 0.0)
        & (view_uv[..., 1] <= 1.0)
    )
    return finite & in_range


def save_view_space_artifacts(
    view_dir: Path,
    safe_view_name: str,
    *,
    sampled_baseline: torch.Tensor,
    sampled_target: torch.Tensor,
    sampled_refined: torch.Tensor,
    stored_target: torch.Tensor,
    mask: torch.Tensor,
    view_uv: torch.Tensor,
) -> dict[str, str]:
    view_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sampled_input_prior_view_roughness": view_dir / f"{safe_view_name}_sampled_input_prior_roughness.png",
        "sampled_input_prior_view_metallic": view_dir / f"{safe_view_name}_sampled_input_prior_metallic.png",
        "sampled_gt_view_roughness": view_dir / f"{safe_view_name}_sampled_gt_roughness.png",
        "sampled_gt_view_metallic": view_dir / f"{safe_view_name}_sampled_gt_metallic.png",
        "sampled_pred_view_roughness": view_dir / f"{safe_view_name}_sampled_pred_roughness.png",
        "sampled_pred_view_metallic": view_dir / f"{safe_view_name}_sampled_pred_metallic.png",
        "stored_view_target_roughness": view_dir / f"{safe_view_name}_stored_target_roughness.png",
        "stored_view_target_metallic": view_dir / f"{safe_view_name}_stored_target_metallic.png",
        "prior_gt_view_error": view_dir / f"{safe_view_name}_prior_gt_view_error.png",
        "pred_gt_view_error": view_dir / f"{safe_view_name}_pred_gt_view_error.png",
        "view_mask": view_dir / f"{safe_view_name}_view_mask.png",
        "view_uv_valid": view_dir / f"{safe_view_name}_view_uv_valid.png",
    }
    tensor_to_pil(sampled_baseline[0:1], grayscale=True).save(paths["sampled_input_prior_view_roughness"])
    tensor_to_pil(sampled_baseline[1:2], grayscale=True).save(paths["sampled_input_prior_view_metallic"])
    tensor_to_pil(sampled_target[0:1], grayscale=True).save(paths["sampled_gt_view_roughness"])
    tensor_to_pil(sampled_target[1:2], grayscale=True).save(paths["sampled_gt_view_metallic"])
    tensor_to_pil(sampled_refined[0:1], grayscale=True).save(paths["sampled_pred_view_roughness"])
    tensor_to_pil(sampled_refined[1:2], grayscale=True).save(paths["sampled_pred_view_metallic"])
    tensor_to_pil(stored_target[0:1], grayscale=True).save(paths["stored_view_target_roughness"])
    tensor_to_pil(stored_target[1:2], grayscale=True).save(paths["stored_view_target_metallic"])
    save_error_heatmap(paths["prior_gt_view_error"], (sampled_baseline - sampled_target).abs())
    save_error_heatmap(paths["pred_gt_view_error"], (sampled_refined - sampled_target).abs())
    tensor_to_pil(mask[None].float(), grayscale=True).save(paths["view_mask"])
    tensor_to_pil(view_uv_valid_mask(view_uv)[None].float(), grayscale=True).save(paths["view_uv_valid"])
    return {key: str(value.resolve()) for key, value in paths.items()}


def confidence_weighted_mean(
    value: torch.Tensor,
    confidence: torch.Tensor,
) -> float:
    weight = float(confidence.sum().item())
    if weight <= 0.0:
        return float(value.mean().item())
    return float((value * confidence).sum().item() / weight)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def optional_mean(values: list[Any]) -> float | None:
    finite_values = [finite_or_none(value) for value in values]
    finite_values = [value for value in finite_values if value is not None]
    if not finite_values:
        return None
    return float(np.mean(finite_values))


def optional_delta(
    baseline: float | None,
    refined: float | None,
    *,
    higher_is_better: bool,
) -> float | None:
    if baseline is None or refined is None:
        return None
    return float(refined - baseline if higher_is_better else baseline - refined)


def pair_direction_rates(
    baseline_values: list[Any],
    refined_values: list[Any],
) -> dict[str, float | int | None]:
    improved = 0
    regressed = 0
    tied = 0
    count = 0
    for baseline_value, refined_value in zip(baseline_values, refined_values, strict=False):
        baseline = finite_or_none(baseline_value)
        refined = finite_or_none(refined_value)
        if baseline is None or refined is None:
            continue
        count += 1
        if refined < baseline:
            improved += 1
        elif refined > baseline:
            regressed += 1
        else:
            tied += 1
    if count <= 0:
        return {
            "improvement_rate": None,
            "regression_rate": None,
            "tie_rate": None,
            "count": 0,
        }
    return {
        "improvement_rate": float(improved / count),
        "regression_rate": float(regressed / count),
        "tie_rate": float(tied / count),
        "count": int(count),
    }


def metric_pair(
    *,
    baseline: float | None,
    refined: float | None,
    higher_is_better: bool,
    count: int,
    mode: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "baseline": baseline,
        "refined": refined,
        "delta": optional_delta(
            baseline,
            refined,
            higher_is_better=higher_is_better,
        ),
        "higher_is_better": higher_is_better,
        "available_count": int(count),
    }
    if mode:
        payload["mode"] = mode
    return payload


def masked_mean_np(values: np.ndarray, mask: np.ndarray) -> float | None:
    visible = values[mask]
    if visible.size == 0:
        return None
    return float(visible.mean())


def compute_masked_psnr(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    visible = mask.astype(bool)
    if prediction.shape != target.shape or visible.sum() == 0:
        return None
    diff = prediction - target
    if diff.ndim == 3:
        visible = np.broadcast_to(visible[..., None], diff.shape)
    mse = float(np.mean(np.square(diff[visible])))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def compute_masked_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    visible = mask.astype(bool)
    if prediction.shape != target.shape or visible.sum() == 0:
        return None
    diff = prediction - target
    if diff.ndim == 3:
        visible = np.broadcast_to(visible[..., None], diff.shape)
    return float(np.mean(np.square(diff[visible])))


def compute_masked_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    visible = mask.astype(bool)
    if prediction.shape != target.shape or visible.sum() == 0:
        return None
    if prediction.ndim == 3:
        pred_values = prediction[visible].reshape(-1, prediction.shape[-1])
        target_values = target[visible].reshape(-1, target.shape[-1])
    else:
        pred_values = prediction[visible].reshape(-1, 1)
        target_values = target[visible].reshape(-1, 1)
    if pred_values.size == 0:
        return None
    scores = []
    c1 = 0.01**2
    c2 = 0.03**2
    for channel_idx in range(pred_values.shape[1]):
        x = pred_values[:, channel_idx].astype(np.float64)
        y = target_values[:, channel_idx].astype(np.float64)
        mux = float(x.mean())
        muy = float(y.mean())
        vx = float(x.var())
        vy = float(y.var())
        covariance = float(((x - mux) * (y - muy)).mean())
        numerator = (2.0 * mux * muy + c1) * (2.0 * covariance + c2)
        denominator = (mux * mux + muy * muy + c1) * (vx + vy + c2)
        scores.append(numerator / denominator if denominator > 0.0 else 0.0)
    return float(np.clip(np.mean(scores), -1.0, 1.0))


def initialize_lpips_metric(enabled: bool, device: str) -> tuple[Any | None, dict[str, Any]]:
    if not enabled:
        return None, {
            "available": False,
            "reason": "disabled_by_config",
        }
    try:
        import lpips  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional environment.
        return None, {
            "available": False,
            "reason": f"lpips_import_failed:{type(exc).__name__}:{exc}",
        }
    try:
        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model, {
            "available": True,
            "reason": "ok",
            "net": "alex",
        }
    except Exception as exc:  # pragma: no cover - depends on optional environment.
        return None, {
            "available": False,
            "reason": f"lpips_init_failed:{type(exc).__name__}:{exc}",
        }


def compute_lpips_distance(
    model: Any | None,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    device: str,
) -> float | None:
    if model is None or prediction.shape != target.shape or mask.sum() == 0:
        return None
    pred = prediction.copy()
    ref = target.copy()
    visible = mask.astype(bool)
    if pred.ndim != 3 or pred.shape[-1] != 3:
        return None
    pred[~visible] = 0.0
    ref[~visible] = 0.0
    pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
    ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).float()
    pred_tensor = pred_tensor.to(device) * 2.0 - 1.0
    ref_tensor = ref_tensor.to(device) * 2.0 - 1.0
    with torch.no_grad():
        score = model(pred_tensor, ref_tensor)
    return float(score.detach().cpu().reshape(-1)[0].item())


def normalize_view_normal(normal: np.ndarray) -> np.ndarray:
    decoded = normal * 2.0 - 1.0
    norm = np.linalg.norm(decoded, axis=0, keepdims=True)
    return decoded / np.maximum(norm, 1e-6)


def proxy_render_from_uv_material(
    *,
    albedo: np.ndarray,
    normal: np.ndarray,
    rm_map: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    roughness = np.clip(rm_map[0], 0.02, 1.0)
    metallic = np.clip(rm_map[1], 0.0, 1.0)
    n = normalize_view_normal(np.clip(normal, 0.0, 1.0))
    light = np.asarray([0.35, -0.45, 0.82], dtype=np.float32)
    light = light / np.linalg.norm(light)
    view = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    half_vec = light + view
    half_vec = half_vec / np.linalg.norm(half_vec)
    ndotl = np.clip(np.sum(n * light[:, None, None], axis=0), 0.0, 1.0)
    ndoth = np.clip(np.sum(n * half_vec[:, None, None], axis=0), 0.0, 1.0)
    spec_power = 2.0 + (1.0 - roughness) * 64.0
    specular = np.power(ndoth, spec_power) * (0.04 * (1.0 - metallic) + metallic)
    diffuse = albedo * (0.18 + 0.82 * ndotl[None, :, :]) * (1.0 - 0.55 * metallic[None, :, :])
    color = diffuse + specular[None, :, :] * (0.35 + 0.65 * albedo)
    color = np.clip(color, 0.0, 1.0)
    color *= mask[None, :, :].astype(np.float32)
    return np.moveaxis(color, 0, -1)


def save_rgb_tensor_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(data, mode="RGB").save(path)


def save_error_heatmap(path: Path, error: torch.Tensor) -> None:
    value = error.detach().cpu()
    if value.ndim == 3:
        value = value.sum(dim=0, keepdim=True)
    value = value / max(float(value.max().item()), 1e-6)
    tensor_to_pil(value.clamp(0.0, 1.0), grayscale=True).save(path)


def gradient_magnitude_np(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        values = values[None, ...]
    grad_x = np.diff(values, axis=-1, append=values[..., -1:])
    grad_y = np.diff(values, axis=-2, append=values[..., -1:, :])
    return np.sqrt(np.square(grad_x) + np.square(grad_y)).mean(axis=0)


def make_edge_band(
    roughness: np.ndarray,
    metallic: np.ndarray,
    mask: np.ndarray,
    *,
    dilation: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    visible = mask.astype(bool)
    if not visible.any():
        return visible, visible
    grad = gradient_magnitude_np(np.stack([roughness, metallic], axis=0))
    visible_grad = grad[visible]
    threshold = max(float(np.quantile(visible_grad, 0.75)), 0.025)
    edge_seed = (grad >= threshold) & visible
    if not edge_seed.any():
        interior_mask = np.asarray(
            Image.fromarray((visible.astype(np.uint8) * 255)).filter(ImageFilter.MinFilter(9))
        ) > 0
        edge_seed = visible & ~interior_mask
    edge = np.asarray(
        Image.fromarray((edge_seed.astype(np.uint8) * 255)).filter(
            ImageFilter.MaxFilter(max(int(dilation), 1) | 1)
        )
    ) > 0
    edge &= visible
    interior = visible & ~edge
    if not interior.any():
        interior = visible
    return edge, interior


def compute_boundary_bleed_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    edge, interior = make_edge_band(target[0], target[1], mask)
    error = np.abs(prediction - target).sum(axis=0)
    edge_error = masked_mean_np(error, edge)
    interior_error = masked_mean_np(error, interior)
    score = None
    if edge_error is not None and interior_error is not None:
        score = float(edge_error - interior_error)
    return {
        "score": score,
        "edge_error": edge_error,
        "interior_error": interior_error,
        "edge_pixel_rate": float(edge.sum() / max(mask.sum(), 1)),
    }


def compute_gradient_preservation(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    pred_grad = gradient_magnitude_np(prediction)
    target_grad = gradient_magnitude_np(target)
    visible = mask.astype(bool)
    if visible.sum() < 2:
        return None
    x = pred_grad[visible].astype(np.float64)
    y = target_grad[visible].astype(np.float64)
    if float(x.std()) <= 1e-8 or float(y.std()) <= 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_highlight_localization(
    prediction: np.ndarray,
    target: np.ndarray,
    reference_rgb: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    visible = mask.astype(bool)
    if visible.sum() < 8:
        return {
            "error": None,
            "iou": None,
            "center_distance": None,
            "highlight_pixel_rate": 0.0,
            "rm_error": None,
        }
    luminance = (
        0.2126 * reference_rgb[..., 0]
        + 0.7152 * reference_rgb[..., 1]
        + 0.0722 * reference_rgb[..., 2]
    )
    visible_luminance = luminance[visible]
    threshold = max(float(np.quantile(visible_luminance, 0.90)), 0.72)
    highlight = (luminance >= threshold) & visible
    if highlight.sum() < 4:
        return {
            "error": None,
            "iou": None,
            "center_distance": None,
            "highlight_pixel_rate": float(highlight.sum() / max(visible.sum(), 1)),
            "rm_error": None,
        }
    response = (1.0 - prediction[0]) * (0.55 + 0.45 * prediction[1])
    response_values = response[visible]
    response_threshold = float(
        np.quantile(response_values, 1.0 - min(float(highlight.sum() / visible.sum()), 0.50))
    )
    predicted_highlight = (response >= response_threshold) & visible
    intersection = float((highlight & predicted_highlight).sum())
    union = float((highlight | predicted_highlight).sum())
    iou = intersection / max(union, 1.0)
    yy, xx = np.indices(mask.shape)
    highlight_weight = highlight.astype(np.float64)
    predicted_weight = predicted_highlight.astype(np.float64)
    h_sum = max(float(highlight_weight.sum()), 1.0)
    p_sum = max(float(predicted_weight.sum()), 1.0)
    h_center = np.asarray(
        [
            float((xx * highlight_weight).sum() / h_sum),
            float((yy * highlight_weight).sum() / h_sum),
        ]
    )
    p_center = np.asarray(
        [
            float((xx * predicted_weight).sum() / p_sum),
            float((yy * predicted_weight).sum() / p_sum),
        ]
    )
    diag = math.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2)
    center_distance = float(np.linalg.norm(h_center - p_center) / max(diag, 1.0))
    rm_error = float(np.abs(prediction[:, highlight] - target[:, highlight]).mean())
    return {
        "error": float((1.0 - iou) + center_distance),
        "iou": float(iou),
        "center_distance": center_distance,
        "highlight_pixel_rate": float(highlight.sum() / max(visible.sum(), 1)),
        "rm_error": rm_error,
    }


def compute_residual_safety(
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, float]:
    baseline_np = baseline.detach().cpu().numpy()
    refined_np = refined.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    confidence_np = confidence.detach().cpu().numpy()[0]
    visible = confidence_np > 0.01
    if not visible.any():
        visible = np.ones_like(confidence_np, dtype=bool)
    baseline_error = np.abs(baseline_np - target_np).sum(axis=0)
    refined_error = np.abs(refined_np - target_np).sum(axis=0)
    residual = np.abs(refined_np - baseline_np).sum(axis=0)
    target_gap = np.abs(target_np - baseline_np).sum(axis=0)
    changed = (residual > 0.03) & visible
    improvement = (refined_error + 1e-6 < baseline_error - 0.01) & visible
    regression = (refined_error > baseline_error + 0.01) & visible
    safe_region = ((target_gap > 0.05) | (confidence_np < 0.60)) & visible
    safe_improvement = changed & improvement & safe_region
    unnecessary_change = changed & (baseline_error < 0.03) & ~improvement
    changed_count = max(int(changed.sum()), 1)
    visible_count = max(int(visible.sum()), 1)
    safe_rate = float(safe_improvement.sum() / changed_count)
    unnecessary_rate = float(unnecessary_change.sum() / changed_count)
    regression_rate = float(regression.sum() / visible_count)
    return {
        "changed_pixel_rate": float(changed.sum() / visible_count),
        "safe_improvement_rate": safe_rate,
        "unnecessary_change_rate": unnecessary_rate,
        "regression_rate": regression_rate,
        "safety_score": float(safe_rate - unnecessary_rate - regression_rate),
    }


def compute_confidence_bins(
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, dict[str, float | None]]:
    baseline_error = (baseline.detach().cpu() - target.detach().cpu()).abs().sum(dim=0).numpy()
    refined_error = (refined.detach().cpu() - target.detach().cpu()).abs().sum(dim=0).numpy()
    conf = confidence.detach().cpu().numpy()[0]
    bins = {
        "low": conf < 0.34,
        "mid": (conf >= 0.34) & (conf < 0.67),
        "high": conf >= 0.67,
    }
    result = {}
    for name, mask in bins.items():
        if not mask.any():
            result[name] = {
                "pixel_count": 0.0,
                "baseline_total_mae": None,
                "refined_total_mae": None,
                "improvement_total": None,
            }
            continue
        baseline_mean = float(baseline_error[mask].mean())
        refined_mean = float(refined_error[mask].mean())
        result[name] = {
            "pixel_count": float(mask.sum()),
            "baseline_total_mae": baseline_mean,
            "refined_total_mae": refined_mean,
            "improvement_total": baseline_mean - refined_mean,
        }
    return result


def binary_confusion_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float,
) -> dict[str, float]:
    if not labels:
        return {
            "f1": 0.0,
            "auroc": 0.0,
            "balanced_accuracy": 0.0,
            "confusion_rate": 0.0,
            "positive_count": 0.0,
            "negative_count": 0.0,
        }
    tp = tn = fp = fn = 0
    for label, score in zip(labels, scores, strict=True):
        pred = int(score >= threshold)
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
        else:
            fn += 1
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "f1": compute_binary_f1(labels, scores, threshold),
        "auroc": compute_binary_auroc(labels, scores),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "confusion_rate": float((fp + fn) / max(len(labels), 1)),
        "positive_count": float(sum(labels)),
        "negative_count": float(len(labels) - sum(labels)),
    }


def compute_binary_f1(labels: list[int], scores: list[float], threshold: float) -> float:
    if not labels:
        return 0.0
    tp = fp = fn = 0
    for label, score in zip(labels, scores, strict=True):
        pred = int(score >= threshold)
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_binary_auroc(labels: list[int], scores: list[float]) -> float:
    if not labels:
        return 0.0
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(np.asarray(scores, dtype=np.float64))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    positive_rank_sum = float(sum(ranks[idx] for idx, label in enumerate(labels) if label == 1))
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def smooth_prior_maps(prior_maps: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    return F.avg_pool2d(prior_maps, kernel_size=kernel_size, stride=1, padding=padding)


def run_variant_model(
    *,
    pipeline: MaterialRefinementPipeline,
    device_batch: dict[str, Any],
    variant: str,
) -> dict[str, torch.Tensor]:
    model = pipeline.model
    original_flags = {
        "disable_prior_inputs": bool(model.cfg.disable_prior_inputs),
        "disable_residual_head": bool(model.cfg.disable_residual_head),
        "disable_view_fusion": bool(model.cfg.disable_view_fusion),
    }
    if variant == "no_prior_refiner":
        model.cfg.disable_prior_inputs = True
    elif variant == "no_residual_refiner":
        model.cfg.disable_residual_head = True
    elif variant == "no_view_refiner":
        model.cfg.disable_view_fusion = True
    with torch.no_grad():
        outputs = model(device_batch)
    model.cfg.disable_prior_inputs = original_flags["disable_prior_inputs"]
    model.cfg.disable_residual_head = original_flags["disable_residual_head"]
    model.cfg.disable_view_fusion = original_flags["disable_view_fusion"]
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in outputs.items()
    }


def resolve_rgba_path(rgba_path: Any) -> str | None:
    if rgba_path is None:
        return None
    if isinstance(rgba_path, (str, os.PathLike)):
        return os.fspath(rgba_path)
    if isinstance(rgba_path, dict):
        for key in ("rgba", "path", "file"):
            value = rgba_path.get(key)
            if isinstance(value, (str, os.PathLike)):
                return os.fspath(value)
        for value in rgba_path.values():
            resolved = resolve_rgba_path(value)
            if resolved:
                return resolved
    return None


def luminance_p99_from_rgba(rgba_path: Any) -> tuple[float, float]:
    resolved = resolve_rgba_path(rgba_path)
    if not resolved:
        return 0.0, 0.0
    path = Path(resolved)
    if not path.exists():
        return 0.0, 0.0
    arr = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    mask = arr[..., 3] > 0.01
    if not mask.any():
        return 0.0, 0.0
    luminance = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    visible = luminance[mask]
    return float(np.quantile(visible, 0.99)), float(((luminance > 0.92) & mask).sum() / mask.sum())


def summarize_group_rows(rows: list[dict[str, Any]], key_name: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "values": defaultdict(list),
            "baseline_metal_scores": [],
            "refined_metal_scores": [],
            "metal_labels": [],
        }
    )
    scalar_keys = [
        "baseline_total_mae",
        "input_prior_total_mae",
        "refined_total_mae",
        "gain_total",
        "improvement_total",
        "baseline_psnr",
        "refined_psnr",
        "baseline_ssim",
        "refined_ssim",
        "baseline_lpips",
        "refined_lpips",
        "baseline_boundary_bleed_score",
        "refined_boundary_bleed_score",
        "baseline_highlight_localization_error",
        "refined_highlight_localization_error",
        "baseline_rm_gradient_preservation",
        "refined_rm_gradient_preservation",
        "prior_residual_safety_score",
        "prior_residual_regression_rate",
        "prior_residual_unnecessary_change_rate",
        "prior_reliability_mean",
        "change_gate_mean",
        "mean_abs_delta",
        "boundary_delta_mean",
        "bootstrap_enabled",
    ]
    for row in rows:
        key = str(row.get(key_name, "unknown"))
        bucket = grouped[key]
        bucket["count"] += 1.0
        for scalar_key in scalar_keys:
            value = finite_or_none(row.get(scalar_key))
            if value is not None:
                bucket["values"][scalar_key].append(value)
        if "gt_is_metal" in row:
            bucket["metal_labels"].append(int(bool(row["gt_is_metal"])))
            bucket["baseline_metal_scores"].append(float(row.get("baseline_pred_metallic_mean", 0.0)))
            bucket["refined_metal_scores"].append(float(row.get("refined_pred_metallic_mean", 0.0)))
    finalized = {}
    for key, bucket in grouped.items():
        count = max(bucket["count"], 1.0)
        labels = [int(value) for value in bucket["metal_labels"]]
        baseline_scores = [float(value) for value in bucket["baseline_metal_scores"]]
        refined_scores = [float(value) for value in bucket["refined_metal_scores"]]
        baseline_confusion = binary_confusion_metrics(labels, baseline_scores, METAL_THRESHOLD)
        refined_confusion = binary_confusion_metrics(labels, refined_scores, METAL_THRESHOLD)
        item: dict[str, Any] = {
            "count": int(bucket["count"]),
            "baseline_metal_f1": baseline_confusion["f1"],
            "refined_metal_f1": refined_confusion["f1"],
            "baseline_metal_auroc": baseline_confusion["auroc"],
            "refined_metal_auroc": refined_confusion["auroc"],
            "baseline_metal_balanced_accuracy": baseline_confusion["balanced_accuracy"],
            "refined_metal_balanced_accuracy": refined_confusion["balanced_accuracy"],
            "baseline_metal_confusion_rate": baseline_confusion["confusion_rate"],
            "refined_metal_confusion_rate": refined_confusion["confusion_rate"],
        }
        for scalar_key in scalar_keys:
            item[scalar_key] = optional_mean(bucket["values"][scalar_key])
        finalized[key] = item
    return finalized


def aggregate_object_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("object_id", "unknown"))].append(row)
    object_rows = []
    metric_keys = [
        "baseline_roughness_mae",
        "baseline_metallic_mae",
        "refined_roughness_mae",
        "refined_metallic_mae",
        "baseline_total_mae",
        "input_prior_total_mae",
        "refined_total_mae",
        "gain_total",
        "improvement_total",
        "baseline_psnr",
        "refined_psnr",
        "baseline_ssim",
        "refined_ssim",
        "baseline_lpips",
        "refined_lpips",
        "baseline_boundary_bleed_score",
        "refined_boundary_bleed_score",
        "baseline_highlight_localization_error",
        "refined_highlight_localization_error",
        "baseline_rm_gradient_preservation",
        "refined_rm_gradient_preservation",
        "prior_residual_safety_score",
        "prior_residual_regression_rate",
        "prior_residual_unnecessary_change_rate",
        "prior_reliability_mean",
        "change_gate_mean",
        "mean_abs_delta",
        "boundary_delta_mean",
        "bootstrap_enabled",
        "gt_metallic_mean",
        "baseline_pred_metallic_mean",
        "refined_pred_metallic_mean",
        "highlight_fraction",
    ]
    passthrough_keys = [
        "generator_id",
        "source_name",
        "category_bucket",
        "prior_label",
        "prior_source_type",
        "prior_generation_mode",
        "prior_mode",
        "prior_variant_type",
        "prior_quality_bin",
        "prior_spatiality",
        "training_role",
        "upstream_model_id",
        "has_material_prior",
        "supervision_tier",
        "supervision_role",
        "license_bucket",
        "target_source_type",
        "target_prior_identity",
        "target_is_prior_copy",
        "target_quality_tier",
        "paper_split",
        "split",
        "material_family",
        "lighting_bank_id",
        "thin_boundary_flag",
        "eval_variant",
        "has_effective_view_supervision",
    ]
    for object_id, object_group in sorted(grouped.items()):
        first = object_group[0]
        out: dict[str, Any] = {"object_id": object_id, "view_rows": len(object_group)}
        for key in passthrough_keys:
            out[key] = first.get(key)
        for key in metric_keys:
            values = [finite_or_none(row.get(key)) for row in object_group if key in row]
            values = [value for value in values if value is not None]
            out[key] = float(np.mean(values)) if values else None
        out["gt_is_metal"] = bool((finite_or_none(out.get("gt_metallic_mean")) or 0.0) >= METAL_THRESHOLD)
        object_rows.append(out)
    return object_rows


def build_dataset(args: argparse.Namespace, atlas_size: int, buffer_resolution: int) -> CanonicalMaterialDataset:
    return CanonicalMaterialDataset(
        args.manifest,
        split=args.split,
        split_strategy=args.split_strategy,
        hash_val_ratio=args.hash_val_ratio,
        hash_test_ratio=args.hash_test_ratio,
        generator_ids=args.generator_ids,
        source_names=args.source_names,
        supervision_tiers=args.supervision_tiers,
        supervision_roles=args.supervision_roles,
        license_buckets=args.license_buckets,
        target_quality_tiers=args.target_quality_tiers,
        paper_splits=args.paper_splits,
        material_families=args.material_families,
        lighting_bank_ids=args.lighting_bank_ids,
        require_prior=args.require_prior,
        max_records=args.max_samples,
        atlas_size=atlas_size,
        buffer_resolution=buffer_resolution,
        max_views_per_sample=args.view_sample_count,
        min_hard_views_per_sample=args.min_hard_views,
        randomize_view_subset=False,
    )


def collect_optional_pair(
    rows: list[dict[str, Any]],
    baseline_key: str,
    refined_key: str,
    *,
    higher_is_better: bool,
    mode: str | None = None,
) -> dict[str, Any]:
    baseline_values = [row.get(baseline_key) for row in rows]
    refined_values = [row.get(refined_key) for row in rows]
    available_count = min(
        len([value for value in baseline_values if finite_or_none(value) is not None]),
        len([value for value in refined_values if finite_or_none(value) is not None]),
    )
    return metric_pair(
        baseline=optional_mean(baseline_values),
        refined=optional_mean(refined_values),
        higher_is_better=higher_is_better,
        count=available_count,
        mode=mode,
    )


def summarize_confidence_bins(
    bin_rows: list[dict[str, dict[str, float | None]]],
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for bin_name in ("low", "mid", "high"):
        baseline_values = []
        refined_values = []
        improvement_values = []
        pixel_counts = []
        for row in bin_rows:
            item = row.get(bin_name, {})
            baseline_values.append(item.get("baseline_total_mae"))
            refined_values.append(item.get("refined_total_mae"))
            improvement_values.append(item.get("improvement_total"))
            pixel_count = finite_or_none(item.get("pixel_count"))
            if pixel_count is not None:
                pixel_counts.append(pixel_count)
        summary[bin_name] = {
            "sample_count": len([value for value in baseline_values if finite_or_none(value) is not None]),
            "pixel_count": float(np.sum(pixel_counts)) if pixel_counts else 0.0,
            "baseline_total_mae": optional_mean(baseline_values),
            "refined_total_mae": optional_mean(refined_values),
            "improvement_total": optional_mean(improvement_values),
        }
    return summary


def build_metric_availability(
    *,
    rows: list[dict[str, Any]],
    dataset_size: int,
    render_metric_mode: str,
    lpips_status: dict[str, Any],
    effective_view_supervision_rate: float,
) -> dict[str, dict[str, Any]]:
    def count_pair(baseline_key: str, refined_key: str) -> int:
        return min(
            len([row for row in rows if finite_or_none(row.get(baseline_key)) is not None]),
            len([row for row in rows if finite_or_none(row.get(refined_key)) is not None]),
        )

    return {
        "uv_rm_mae": {
            "available": dataset_size > 0,
            "available_count": dataset_size,
            "reason": "uv_target_rm_and_confidence",
        },
        "view_rm_mae": {
            "available": count_pair("baseline_total_mae", "refined_total_mae") > 0,
            "available_count": count_pair("baseline_total_mae", "refined_total_mae"),
            "reason": "view_uv_and_view_rm_targets" if effective_view_supervision_rate > 0.0 else "missing_view_uv_or_view_rm_targets",
        },
        "proxy_render_psnr": {
            "available": count_pair("baseline_psnr", "refined_psnr") > 0,
            "available_count": count_pair("baseline_psnr", "refined_psnr"),
            "reason": render_metric_mode,
        },
        "proxy_render_mse": {
            "available": count_pair("baseline_mse", "refined_mse") > 0,
            "available_count": count_pair("baseline_mse", "refined_mse"),
            "reason": render_metric_mode,
        },
        "proxy_render_ssim": {
            "available": count_pair("baseline_ssim", "refined_ssim") > 0,
            "available_count": count_pair("baseline_ssim", "refined_ssim"),
            "reason": render_metric_mode,
        },
        "proxy_render_lpips": {
            "available": count_pair("baseline_lpips", "refined_lpips") > 0,
            "available_count": count_pair("baseline_lpips", "refined_lpips"),
            "reason": lpips_status.get("reason", "unknown"),
            "lpips_available": bool(lpips_status.get("available", False)),
        },
    }


def build_metric_warnings(
    *,
    summary_payload: dict[str, Any],
    diagnostic_min_group_count: int,
) -> list[str]:
    warnings = []
    availability = summary_payload.get("metric_availability", {})
    for name in MAIN_METRIC_NAMES:
        item = availability.get(name, {})
        if not item.get("available", False):
            warnings.append(f"metric_unavailable:{name}:{item.get('reason', 'unknown')}")
    metal = summary_payload.get("metrics_material_specific", {}).get("metal_nonmetal_confusion", {})
    for prefix in ("uv_level", "view_level", "object_level"):
        level_item = metal.get(prefix, {})
        for variant in ("baseline", "refined"):
            item = level_item.get(variant, {}) if isinstance(level_item, dict) else {}
            if item.get("f1") == 1.0 and item.get("auroc") == 0.0:
                warnings.append(f"metal_metric_degenerate:{prefix}:{variant}:f1=1.0_auroc=0.0")
            if item.get("positive_count", 0.0) == 0.0 or item.get("negative_count", 0.0) == 0.0:
                warnings.append(f"metal_metric_single_class:{prefix}:{variant}")
    by_group = summary_payload.get("metrics_by_group", {})
    for group_name, group_values in by_group.items():
        if not isinstance(group_values, dict):
            continue
        if group_name == "by_view_name":
            small_count = sum(
                1
                for group_item in group_values.values()
                if isinstance(group_item, dict)
                and int(group_item.get("count", 0)) < diagnostic_min_group_count
            )
            if small_count:
                warnings.append(
                    f"group_diagnostic_axis_only:{group_name}:small_groups={small_count};details_in_summary_json"
                )
            continue
        for group_id, group_item in group_values.items():
            if isinstance(group_item, dict) and int(group_item.get("count", 0)) < diagnostic_min_group_count:
                warnings.append(f"group_diagnostic_only:{group_name}:{group_id}:count={group_item.get('count', 0)}")
    disagreement = summary_payload.get("metrics_diagnostics", {}).get("metric_disagreement", {})
    if disagreement.get("has_disagreement"):
        warnings.extend(disagreement.get("warnings", []))
    return warnings


def build_metric_disagreement(
    *,
    uv_improvement: float,
    view_improvement: float | None,
    object_improvement: float | None,
) -> dict[str, Any]:
    values = {
        "uv_level_improvement_total": uv_improvement,
        "view_level_improvement_total": view_improvement,
        "object_level_improvement_total": object_improvement,
    }
    signs = {
        key: (None if value is None else (1 if value > 1e-6 else -1 if value < -1e-6 else 0))
        for key, value in values.items()
    }
    finite_signs = {key: value for key, value in signs.items() if value is not None}
    non_zero_signs = {key: value for key, value in finite_signs.items() if value != 0}
    has_disagreement = len(set(non_zero_signs.values())) > 1
    warnings = []
    if has_disagreement:
        warnings.append(
            "metric_disagreement:uv/view/object improvements have conflicting signs; do not use as paper-stage conclusion until audited"
        )
    return {
        "has_disagreement": bool(has_disagreement),
        "values": values,
        "signs": signs,
        "warnings": warnings,
    }


def write_metric_disagreement_report(
    output_dir: Path,
    disagreement: dict[str, Any],
) -> tuple[Path, Path]:
    json_path = output_dir / "metric_disagreement_report.json"
    html_path = output_dir / "metric_disagreement_report.html"
    json_path.write_text(
        json.dumps(make_json_serializable(disagreement), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = []
    for key, value in disagreement.get("values", {}).items():
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (
                key,
                "n/a" if value is None else f"{float(value):.6f}",
                disagreement.get("signs", {}).get(key),
            )
        )
    warning_html = "".join(
        f"<li>{str(warning)}</li>" for warning in disagreement.get("warnings", [])
    ) or "<li>none</li>"
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'>",
                "<title>Metric Disagreement Report</title>",
                "<style>body{font-family:Arial,sans-serif;background:#111827;color:#f8fafc;margin:24px;}table{border-collapse:collapse;width:100%;}td,th{border-bottom:1px solid #334155;padding:8px;text-align:left}.bad{color:#fca5a5}.good{color:#86efac}</style>",
                "</head><body>",
                "<h1>Metric Disagreement Report</h1>",
                f"<p>Status: <strong class='{'bad' if disagreement.get('has_disagreement') else 'good'}'>{'DISAGREEMENT' if disagreement.get('has_disagreement') else 'OK'}</strong></p>",
                "<table><thead><tr><th>Metric Level</th><th>Improvement</th><th>Sign</th></tr></thead><tbody>",
                *rows,
                "</tbody></table>",
                "<h2>Warnings</h2><ul>",
                warning_html,
                "</ul></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, html_path


def build_model_cfg_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Merge audited eval-only model toggles into checkpoint configs."""
    overrides: dict[str, Any] = {}
    for key in MODEL_CFG_OVERRIDE_KEYS:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                overrides[key] = value
    return overrides


def write_diagnostic_cases(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    top_k: int = 24,
) -> Path:
    def compact(row: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "object_id",
            "view_name",
            "generator_id",
            "source_name",
            "material_family",
            "paper_split",
            "baseline_total_mae",
            "refined_total_mae",
            "improvement_total",
            "baseline_primary_failure",
            "refined_primary_failure",
            "baseline_boundary_bleed_score",
            "refined_boundary_bleed_score",
            "baseline_highlight_localization_error",
            "refined_highlight_localization_error",
            "baseline_rm_gradient_preservation",
            "refined_rm_gradient_preservation",
            "prior_residual_safety_score",
        ]
        return {key: make_json_serializable(row.get(key)) for key in keys}

    improved = sorted(rows, key=lambda row: finite_or_none(row.get("improvement_total")) or -999.0, reverse=True)
    regressed = sorted(rows, key=lambda row: finite_or_none(row.get("improvement_total")) or 999.0)
    uncertain = sorted(
        rows,
        key=lambda row: (
            abs(finite_or_none(row.get("improvement_total")) or 0.0),
            -(finite_or_none(row.get("baseline_total_mae")) or 0.0),
        ),
    )
    by_failure = {}
    for tag in FAILURE_TAGS:
        tagged = [
            row for row in rows if tag in row.get("baseline_tags", []) or tag in row.get("refined_tags", [])
        ]
        by_failure[tag] = [compact(row) for row in sorted(tagged, key=lambda row: finite_or_none(row.get("refined_total_mae")) or 0.0, reverse=True)[:top_k]]
    payload = {
        "top_improved": [compact(row) for row in improved[:top_k]],
        "top_regressed": [compact(row) for row in regressed[:top_k]],
        "top_uncertain": [compact(row) for row in uncertain[:top_k]],
        "top_failure_cases": by_failure,
    }
    path = output_dir / "diagnostic_cases.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


from sf3d.material_refine.eval_metrics import (  # noqa: E402,F401,F811
    GROUP_DIAGNOSTIC_MIN_COUNT,
    MAIN_METRIC_NAMES,
    MATERIAL_SPECIFIC_METRIC_NAMES,
    METAL_THRESHOLD,
    binary_confusion_metrics,
    build_metric_availability,
    build_metric_disagreement,
    build_metric_warnings,
    collect_optional_pair,
    compute_binary_auroc,
    compute_binary_f1,
    compute_boundary_bleed_metrics,
    compute_confidence_bins,
    compute_gradient_preservation,
    compute_highlight_localization,
    compute_lpips_distance,
    compute_lpips_distance_batch,
    compute_masked_psnr,
    compute_masked_ssim,
    compute_residual_safety,
    confidence_weighted_mean,
    finite_or_none,
    initialize_lpips_metric,
    metric_pair,
    optional_delta,
    optional_mean,
    proxy_render_from_uv_material,
    save_error_heatmap,
    save_rgb_tensor_image,
    summarize_confidence_bins,
    write_diagnostic_cases,
    write_metric_disagreement_report,
)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "eval_args.json"
    save_path.write_text(
        json.dumps(make_json_serializable(vars(args)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.manifest.exists():
        shutil.copy2(args.manifest, output_dir / "manifest_snapshot.json")
    if not args.checkpoint.exists():
        resolved_checkpoint = args.checkpoint.resolve(strict=False)
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint} (resolved: {resolved_checkpoint}). "
            "If this is a symlink, verify that its target checkpoint was not pruned."
        )

    device = resolve_device(args)
    lpips_model, lpips_status = initialize_lpips_metric(
        args.enable_lpips and args.render_metric_mode != "disabled",
        device,
    )
    pipeline = MaterialRefinementPipeline.from_checkpoint(
        args.checkpoint,
        device=device,
        cuda_device_index=args.cuda_device_index,
        model_cfg_overrides=build_model_cfg_overrides(args),
    )
    dataset = build_dataset(args, pipeline.atlas_size, pipeline.buffer_resolution)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_material_samples,
    )

    use_wandb = args.report_to == "wandb"
    run_name = args.tracker_run_name or f"{args.checkpoint.stem}-{args.split}"
    tracker_config = make_json_serializable(dict(vars(args)))
    tracker_config["dataset_summary"] = summarize_records(dataset.records)
    run = maybe_init_wandb(
        enabled=use_wandb,
        project=args.tracker_project_name,
        job_type="eval",
        config=tracker_config,
        mode=args.wandb_mode,
        name=run_name,
        group=args.tracker_group,
        tags=args.tracker_tags,
        run_id=args.wandb_resume_id,
        resume=args.wandb_resume_mode,
        dir_path=args.wandb_dir,
    )

    rows = []
    summary = {
        "baseline_total_mae": [],
        "refined_total_mae": [],
        "baseline_roughness_mae": [],
        "baseline_metallic_mae": [],
        "refined_roughness_mae": [],
        "refined_metallic_mae": [],
        "effective_view_supervision_samples": 0,
        "baseline_metal_scores": [],
        "refined_metal_scores": [],
        "metal_labels": [],
        "batch_seconds": [],
        "confidence_bins": [],
        "residual_safety": [],
    }
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    eval_started_at = time.perf_counter()
    processed_records = 0
    total_batches = len(loader)
    total_records = len(dataset.records)
    for batch_index, batch in enumerate(loader, start=1):
        device_batch = move_batch_to_device(batch, pipeline.device)
        start_time = time.perf_counter()
        with torch.no_grad():
            default_outputs = pipeline.model(device_batch)
        default_outputs = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in default_outputs.items()
        }
        variant_outputs = default_outputs
        if args.eval_variant in {"no_prior_refiner", "no_residual_refiner", "no_view_refiner"}:
            variant_outputs = run_variant_model(
                pipeline=pipeline,
                device_batch=device_batch,
                variant=args.eval_variant,
            )
        outputs = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in variant_outputs.items()
        } if variant_outputs is not default_outputs else default_outputs
        batch_seconds = time.perf_counter() - start_time
        summary["batch_seconds"].append(float(batch_seconds))
        baseline = default_outputs.get("input_prior", default_outputs["baseline"])
        refined = outputs["refined"]
        diagnostics = outputs.get("diagnostics") or {}
        prior_init_outputs = outputs.get("prior_init_outputs") or {}
        rm_init = outputs.get("rm_init", outputs.get("initial", baseline))
        delta_rm = outputs.get("delta_rm", outputs.get("residual_delta", refined - baseline))
        change_gate = outputs.get("change_gate", outputs.get("residual_gate"))
        prior_reliability = outputs.get("prior_reliability")
        bootstrap_rm = (
            prior_init_outputs.get("bootstrap_rm_uv")
            if isinstance(prior_init_outputs, dict)
            else None
        )
        if args.eval_variant == "scalar_broadcast":
            prior_scalar_roughness = batch["uv_prior_roughness"].flatten(2).mean(dim=2, keepdim=True).view(-1, 1, 1, 1)
            prior_scalar_metallic = batch["uv_prior_metallic"].flatten(2).mean(dim=2, keepdim=True).view(-1, 1, 1, 1)
            refined = torch.cat(
                [
                    torch.full_like(batch["uv_prior_roughness"], 0.5),
                    torch.zeros_like(batch["uv_prior_metallic"]),
                ],
                dim=1,
            )
            refined[:, 0:1] = prior_scalar_roughness.expand_as(batch["uv_prior_roughness"])
            refined[:, 1:2] = prior_scalar_metallic.expand_as(batch["uv_prior_metallic"])
        elif args.eval_variant == "prior_smoothing":
            refined = smooth_prior_maps(baseline, args.prior_smoothing_kernel)
        target = torch.cat(
            [batch["uv_target_roughness"], batch["uv_target_metallic"]],
            dim=1,
        )
        confidence = batch["uv_target_confidence"]

        for item_idx, object_id in enumerate(batch["object_id"]):
            def tensor_item_mean(value: Any) -> float | None:
                if not isinstance(value, torch.Tensor):
                    return None
                tensor = value.detach().cpu()
                if tensor.ndim == 0:
                    return float(tensor.item())
                if tensor.shape[0] <= item_idx:
                    return None
                return float(tensor[item_idx].float().mean().item())

            def diag_item_value(key: str) -> float | None:
                value = diagnostics.get(key) if isinstance(diagnostics, dict) else None
                if isinstance(value, torch.Tensor):
                    tensor = value.detach().cpu().flatten()
                    if tensor.numel() > item_idx:
                        return float(tensor[item_idx].item())
                    if tensor.numel() == 1:
                        return float(tensor.item())
                return finite_or_none(value)

            object_id = str(object_id)
            metadata = batch["metadata"][item_idx]
            pair_id = str(batch_item_value(batch, metadata, "pair_id", item_idx, ""))
            target_bundle_id = str(batch_item_value(batch, metadata, "target_bundle_id", item_idx, ""))
            prior_variant_id = str(batch_item_value(batch, metadata, "prior_variant_id", item_idx, ""))
            generator_id = str(batch["generator_id"][item_idx])
            source_name = str(batch["source_name"][item_idx])
            category_bucket = str(batch["category_bucket"][item_idx])
            has_material_prior = bool(batch["has_material_prior"][item_idx])
            prior_label = "with_prior" if has_material_prior else "without_prior"
            prior_mode = str(batch["prior_mode"][item_idx])
            prior_generation_mode = first_nonempty(
                batch_item_value(batch, metadata, "prior_generation_mode", item_idx, ""),
                batch_item_value(batch, metadata, "prior_source_type", item_idx, ""),
                prior_mode,
            )
            prior_source_type = first_nonempty(
                batch_item_value(batch, metadata, "prior_source_type", item_idx, ""),
                prior_generation_mode,
                prior_mode,
            )
            prior_variant_type = first_nonempty(
                batch_item_value(batch, metadata, "prior_variant_type", item_idx, ""),
                "unknown",
            )
            prior_quality_bin = first_nonempty(
                batch_item_value(batch, metadata, "prior_quality_bin", item_idx, ""),
                "unknown",
            )
            prior_spatiality = first_nonempty(
                batch_item_value(batch, metadata, "prior_spatiality", item_idx, ""),
                "unknown",
            )
            training_role = first_nonempty(
                batch_item_value(batch, metadata, "training_role", item_idx, ""),
                "unknown",
            )
            upstream_model_id = first_nonempty(
                batch_item_value(batch, metadata, "upstream_model_id", item_idx, ""),
                "unknown",
            )
            sample_weight = float(batch_item_value(batch, metadata, "sample_weight", item_idx, 1.0))
            input_prior_baseline_source = (
                "model_no_prior_bootstrap_baseline"
                if prior_variant_type == "no_prior_bootstrap" or prior_spatiality == "no_prior"
                else "model_input_prior_from_provided_prior"
            )

            global_object_index = processed_records + item_idx
            write_artifacts = args.max_artifact_objects <= 0 or global_object_index < args.max_artifact_objects
            case_id = material_case_id(
                object_id=object_id,
                prior_variant_type=prior_variant_type,
                pair_id=pair_id,
                prior_variant_id=prior_variant_id,
                fallback_index=global_object_index,
            )
            object_dir = output_dir / "artifacts" / case_id
            atlas_paths = {}
            if write_artifacts:
                atlas_paths = save_atlas_bundle(
                    object_dir,
                    baseline_roughness=baseline[item_idx, 0],
                    baseline_metallic=baseline[item_idx, 1],
                    refined_roughness=refined[item_idx, 0],
                    refined_metallic=refined[item_idx, 1],
                    confidence=confidence[item_idx],
                )
                tensor_to_pil(rm_init[item_idx, 0:1], grayscale=True).save(object_dir / "rm_init_roughness.png")
                tensor_to_pil(rm_init[item_idx, 1:2], grayscale=True).save(object_dir / "rm_init_metallic.png")
                tensor_to_pil(delta_rm[item_idx].abs().mean(dim=0, keepdim=True).clamp(0.0, 1.0), grayscale=True).save(object_dir / "delta_abs.png")
                atlas_paths["rm_init_roughness"] = object_dir / "rm_init_roughness.png"
                atlas_paths["rm_init_metallic"] = object_dir / "rm_init_metallic.png"
                atlas_paths["delta_abs"] = object_dir / "delta_abs.png"
                if isinstance(change_gate, torch.Tensor):
                    gate = change_gate[item_idx].detach().cpu()
                    if gate.ndim == 3 and gate.shape[0] > 1:
                        gate = gate.mean(dim=0, keepdim=True)
                    tensor_to_pil(gate[:1], grayscale=True).save(object_dir / "change_gate.png")
                    atlas_paths["change_gate"] = object_dir / "change_gate.png"
                if isinstance(prior_reliability, torch.Tensor):
                    reliability = prior_reliability[item_idx].detach().cpu()
                    if reliability.ndim == 3 and reliability.shape[0] > 1:
                        reliability = reliability.mean(dim=0, keepdim=True)
                    tensor_to_pil(reliability[:1], grayscale=True).save(object_dir / "prior_reliability.png")
                    atlas_paths["prior_reliability"] = object_dir / "prior_reliability.png"
                if isinstance(bootstrap_rm, torch.Tensor):
                    bootstrap = bootstrap_rm.detach().cpu()
                    if bootstrap.shape[0] > item_idx and bootstrap.shape[1] >= 2:
                        tensor_to_pil(bootstrap[item_idx, 0:1], grayscale=True).save(object_dir / "bootstrap_roughness.png")
                        tensor_to_pil(bootstrap[item_idx, 1:2], grayscale=True).save(object_dir / "bootstrap_metallic.png")
                        atlas_paths["bootstrap_roughness"] = object_dir / "bootstrap_roughness.png"
                        atlas_paths["bootstrap_metallic"] = object_dir / "bootstrap_metallic.png"
            target_item = target[item_idx]
            prior_reliability_mean = tensor_item_mean(prior_reliability)
            change_gate_mean = tensor_item_mean(change_gate)
            mean_abs_delta = diag_item_value("mean_abs_delta")
            boundary_delta_mean = diag_item_value("boundary_delta_mean")
            bootstrap_enabled = diag_item_value("bootstrap_enabled")
            if write_artifacts:
                target_roughness_path = object_dir / "target_roughness.png"
                target_metallic_path = object_dir / "target_metallic.png"
                baseline_error_path = object_dir / "baseline_rm_error.png"
                refined_error_path = object_dir / "refined_rm_error.png"
                tensor_to_pil(target_item[0:1], grayscale=True).save(target_roughness_path)
                tensor_to_pil(target_item[1:2], grayscale=True).save(target_metallic_path)
                save_error_heatmap(baseline_error_path, (baseline[item_idx] - target_item).abs())
                save_error_heatmap(refined_error_path, (refined[item_idx] - target_item).abs())
                atlas_paths.update(
                    {
                        "target_roughness": target_roughness_path,
                        "target_metallic": target_metallic_path,
                        "baseline_rm_error": baseline_error_path,
                        "refined_rm_error": refined_error_path,
                    }
                )

            base_uv_mae = (baseline[item_idx] - target[item_idx]).abs() * confidence[item_idx]
            refined_uv_mae = (refined[item_idx] - target[item_idx]).abs() * confidence[item_idx]
            uv_weight = float(confidence[item_idx].sum().item()) or 1.0
            baseline_uv_roughness_mae = float(base_uv_mae[0].sum().item() / uv_weight)
            baseline_uv_metallic_mae = float(base_uv_mae[1].sum().item() / uv_weight)
            refined_uv_roughness_mae = float(refined_uv_mae[0].sum().item() / uv_weight)
            refined_uv_metallic_mae = float(refined_uv_mae[1].sum().item() / uv_weight)
            residual_safety = compute_residual_safety(
                baseline[item_idx],
                refined[item_idx],
                target_item,
                confidence[item_idx],
            )
            confidence_bins = compute_confidence_bins(
                baseline[item_idx],
                refined[item_idx],
                target_item,
                confidence[item_idx],
            )
            summary["residual_safety"].append(residual_safety)
            summary["confidence_bins"].append(confidence_bins)

            supervision_tier = str(batch["supervision_tier"][item_idx])
            supervision_role = str(batch["supervision_role"][item_idx])
            license_bucket = str(batch["license_bucket"][item_idx])
            target_source_type = str(batch["target_source_type"][item_idx])
            target_is_prior_copy = parse_bool(
                batch_item_value(batch, metadata, "target_is_prior_copy", item_idx, False)
            )
            target_prior_identity = float(batch["target_prior_identity"][item_idx].item())
            target_quality_tier = str(batch["target_quality_tier"][item_idx])
            paper_split = str(batch["paper_split"][item_idx])
            split = str(batch["split"][item_idx])
            material_family = str(batch["material_family"][item_idx])
            lighting_bank_id = str(batch["lighting_bank_id"][item_idx])
            thin_boundary_flag = bool(batch["thin_boundary_flag"][item_idx])

            summary["baseline_roughness_mae"].append(baseline_uv_roughness_mae)
            summary["baseline_metallic_mae"].append(baseline_uv_metallic_mae)
            summary["refined_roughness_mae"].append(refined_uv_roughness_mae)
            summary["refined_metallic_mae"].append(refined_uv_metallic_mae)
            summary["baseline_total_mae"].append(
                baseline_uv_roughness_mae + baseline_uv_metallic_mae
            )
            summary["refined_total_mae"].append(
                refined_uv_roughness_mae + refined_uv_metallic_mae
            )
            has_effective_view_supervision = bool(
                batch["has_effective_view_supervision"][item_idx]
            )
            summary["effective_view_supervision_samples"] += int(has_effective_view_supervision)
            uv_gt_metal_mean = confidence_weighted_mean(
                batch["uv_target_metallic"][item_idx],
                confidence[item_idx],
            )
            uv_baseline_metal_mean = confidence_weighted_mean(
                baseline[item_idx, 1:2],
                confidence[item_idx],
            )
            uv_refined_metal_mean = confidence_weighted_mean(
                refined[item_idx, 1:2],
                confidence[item_idx],
            )
            summary["metal_labels"].append(int(uv_gt_metal_mean >= METAL_THRESHOLD))
            summary["baseline_metal_scores"].append(float(uv_baseline_metal_mean))
            summary["refined_metal_scores"].append(float(uv_refined_metal_mean))

            view_targets = batch["view_targets"]
            view_uvs = batch["view_uvs"]
            input_prior_total_uv = baseline_uv_roughness_mae + baseline_uv_metallic_mae
            refined_total_uv = refined_uv_roughness_mae + refined_uv_metallic_mae
            gain_total_uv = input_prior_total_uv - refined_total_uv
            if view_uvs is None or not has_effective_view_supervision:
                rows.append(
                    {
                        "object_id": object_id,
                        "case_id": case_id,
                        "pair_id": pair_id,
                        "target_bundle_id": target_bundle_id,
                        "prior_variant_id": prior_variant_id,
                        "generator_id": generator_id,
                        "source_name": source_name,
                        "category_bucket": category_bucket,
                        "prior_label": prior_label,
                        "prior_source_type": prior_source_type,
                        "prior_generation_mode": prior_generation_mode,
                        "prior_mode": prior_mode,
                        "prior_variant_type": prior_variant_type,
                        "prior_quality_bin": prior_quality_bin,
                        "prior_spatiality": prior_spatiality,
                        "training_role": training_role,
                        "upstream_model_id": upstream_model_id,
                        "sample_weight": sample_weight,
                        "input_prior_baseline_source": input_prior_baseline_source,
                        "has_material_prior": has_material_prior,
                        "supervision_tier": supervision_tier,
                        "supervision_role": supervision_role,
                        "license_bucket": license_bucket,
                        "target_source_type": target_source_type,
                        "target_prior_identity": target_prior_identity,
                        "target_is_prior_copy": target_is_prior_copy,
                        "target_quality_tier": target_quality_tier,
                        "paper_split": paper_split,
                        "split": split,
                        "material_family": material_family,
                        "lighting_bank_id": lighting_bank_id,
                        "thin_boundary_flag": thin_boundary_flag,
                        "eval_variant": args.eval_variant,
                        "view_name": "uv_space",
                        "view_target_basis": "uv_space",
                        "stored_view_target_available": bool(view_targets is not None),
                        "baseline_roughness_mae": baseline_uv_roughness_mae,
                        "baseline_metallic_mae": baseline_uv_metallic_mae,
                        "refined_roughness_mae": refined_uv_roughness_mae,
                        "refined_metallic_mae": refined_uv_metallic_mae,
                        "baseline_total_mae": baseline_uv_roughness_mae + baseline_uv_metallic_mae,
                        "input_prior_total_mae": input_prior_total_uv,
                        "refined_total_mae": refined_uv_roughness_mae + refined_uv_metallic_mae,
                        "gain_total": gain_total_uv,
                        "uv_prior_error": input_prior_total_uv,
                        "uv_pred_error": refined_total_uv,
                        "uv_gain": gain_total_uv,
                        "uv_improved": bool(gain_total_uv > 0.0),
                        "view_prior_error": None,
                        "view_pred_error": None,
                        "view_gain": None,
                        "view_regressed": False,
                        "improvement_total": (baseline_uv_roughness_mae + baseline_uv_metallic_mae)
                        - (refined_uv_roughness_mae + refined_uv_metallic_mae),
                        "baseline_tags": [],
                        "refined_tags": [],
                        "baseline_primary_failure": "none",
                        "refined_primary_failure": "none",
                        "baseline_psnr": None,
                        "refined_psnr": None,
                        "baseline_ssim": None,
                        "refined_ssim": None,
                        "baseline_lpips": None,
                        "refined_lpips": None,
                        "baseline_boundary_bleed_score": None,
                        "refined_boundary_bleed_score": None,
                        "baseline_highlight_localization_error": None,
                        "refined_highlight_localization_error": None,
                        "baseline_rm_gradient_preservation": None,
                        "refined_rm_gradient_preservation": None,
                        "prior_residual_safety_score": residual_safety["safety_score"],
                        "prior_residual_safe_improvement_rate": residual_safety["safe_improvement_rate"],
                        "prior_residual_unnecessary_change_rate": residual_safety["unnecessary_change_rate"],
                        "prior_residual_regression_rate": residual_safety["regression_rate"],
                        "prior_residual_changed_pixel_rate": residual_safety["changed_pixel_rate"],
                        "prior_reliability_mean": prior_reliability_mean,
                        "change_gate_mean": change_gate_mean,
                        "mean_abs_delta": mean_abs_delta,
                        "boundary_delta_mean": boundary_delta_mean,
                        "bootstrap_enabled": bootstrap_enabled,
                        "gt_metallic_mean": uv_gt_metal_mean,
                        "baseline_pred_metallic_mean": uv_baseline_metal_mean,
                        "refined_pred_metallic_mean": uv_refined_metal_mean,
                        "gt_is_metal": bool(uv_gt_metal_mean >= METAL_THRESHOLD),
                        "has_effective_view_supervision": False,
                        "runtime_ms_per_object": float(batch_seconds * 1000.0 / max(len(batch["object_id"]), 1)),
                        "paths": {key: str(value.resolve()) for key, value in atlas_paths.items()},
                    }
                )
                continue

            sampled_baseline = sample_uv_maps_to_view(
                baseline[item_idx : item_idx + 1],
                view_uvs[item_idx : item_idx + 1],
            )[0]
            sampled_refined = sample_uv_maps_to_view(
                refined[item_idx : item_idx + 1],
                view_uvs[item_idx : item_idx + 1],
            )[0]
            sampled_target = sample_uv_maps_to_view(
                target[item_idx : item_idx + 1],
                view_uvs[item_idx : item_idx + 1],
            )[0]
            sampled_albedo = sample_uv_maps_to_view(
                batch["uv_albedo"][item_idx : item_idx + 1],
                view_uvs[item_idx : item_idx + 1],
            )[0]
            sampled_normal = sample_uv_maps_to_view(
                batch["uv_normal"][item_idx : item_idx + 1],
                view_uvs[item_idx : item_idx + 1],
            )[0]
            pending_lpips_rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

            for view_idx, view_name in enumerate(batch["view_names"][item_idx]):
                gt_view = sampled_target[view_idx]
                stored_target_view = (
                    view_targets[item_idx, view_idx]
                    if view_targets is not None
                    else sampled_target[view_idx]
                )
                valid_uv = view_uv_valid_mask(view_uvs[item_idx, view_idx])
                mask = (batch["view_masks"][item_idx, view_idx, 0] > 0.5) & valid_uv
                if not mask.any():
                    continue
                safe_view_name = "".join(
                    char if char.isalnum() or char in {"-", "_"} else "_"
                    for char in str(view_name)
                )
                gt_rough = gt_view[0].numpy()
                gt_metal = gt_view[1].numpy()
                baseline_view = sampled_baseline[view_idx].numpy()
                refined_view = sampled_refined[view_idx].numpy()
                mask_np = mask.numpy()

                gt_rough_stats = masked_stats(gt_rough, mask_np)
                gt_metal_stats = masked_stats(gt_metal, mask_np)
                brightness_p99 = 0.0
                highlight_fraction = 0.0
                rgba_lookup = metadata.get("view_rgba_paths", {})
                rgba_path = rgba_lookup.get(view_name) if isinstance(rgba_lookup, dict) else None
                if rgba_path:
                    brightness_p99, highlight_fraction = luminance_p99_from_rgba(rgba_path)

                base_pred_rough_mean = float(baseline_view[0][mask_np].mean())
                base_pred_metal_mean = float(baseline_view[1][mask_np].mean())
                refined_pred_rough_mean = float(refined_view[0][mask_np].mean())
                refined_pred_metal_mean = float(refined_view[1][mask_np].mean())
                baseline_tags, baseline_primary = classify_case(
                    pred_roughness=base_pred_rough_mean,
                    pred_metallic=base_pred_metal_mean,
                    gt_roughness=gt_rough_stats,
                    gt_metallic=gt_metal_stats,
                    brightness_p99=brightness_p99,
                )
                refined_tags, refined_primary = classify_case(
                    pred_roughness=refined_pred_rough_mean,
                    pred_metallic=refined_pred_metal_mean,
                    gt_roughness=gt_rough_stats,
                    gt_metallic=gt_metal_stats,
                    brightness_p99=brightness_p99,
                )
                baseline_rough_mae = float(np.abs(baseline_view[0][mask_np] - gt_rough[mask_np]).mean())
                baseline_metal_mae = float(np.abs(baseline_view[1][mask_np] - gt_metal[mask_np]).mean())
                refined_rough_mae = float(np.abs(refined_view[0][mask_np] - gt_rough[mask_np]).mean())
                refined_metal_mae = float(np.abs(refined_view[1][mask_np] - gt_metal[mask_np]).mean())
                input_prior_total_view = baseline_rough_mae + baseline_metal_mae
                refined_total_view = refined_rough_mae + refined_metal_mae
                gain_total_view = input_prior_total_view - refined_total_view
                view_paths = {}
                if write_artifacts:
                    view_paths.update(
                        save_view_space_artifacts(
                            object_dir / "views",
                            safe_view_name,
                            sampled_baseline=sampled_baseline[view_idx],
                            sampled_target=sampled_target[view_idx],
                            sampled_refined=sampled_refined[view_idx],
                            stored_target=stored_target_view,
                            mask=mask.float(),
                            view_uv=view_uvs[item_idx, view_idx],
                        )
                    )
                target_view = np.stack([gt_rough, gt_metal], axis=0)
                baseline_boundary = compute_boundary_bleed_metrics(
                    baseline_view,
                    target_view,
                    mask_np,
                )
                refined_boundary = compute_boundary_bleed_metrics(
                    refined_view,
                    target_view,
                    mask_np,
                )
                reference_rgb = batch["view_features"][item_idx, view_idx, 0:3].numpy()
                reference_rgb_hwc = np.moveaxis(reference_rgb, 0, -1)
                baseline_highlight = compute_highlight_localization(
                    baseline_view,
                    target_view,
                    reference_rgb_hwc,
                    mask_np,
                )
                refined_highlight = compute_highlight_localization(
                    refined_view,
                    target_view,
                    reference_rgb_hwc,
                    mask_np,
                )
                baseline_gradient_preservation = compute_gradient_preservation(
                    baseline_view,
                    target_view,
                    mask_np,
                )
                refined_gradient_preservation = compute_gradient_preservation(
                    refined_view,
                    target_view,
                    mask_np,
                )
                baseline_mse = refined_mse = None
                baseline_psnr = refined_psnr = None
                baseline_ssim = refined_ssim = None
                baseline_lpips = refined_lpips = None
                if args.render_metric_mode == "proxy_uv_shading":
                    baseline_proxy = proxy_render_from_uv_material(
                        albedo=sampled_albedo[view_idx].numpy(),
                        normal=sampled_normal[view_idx].numpy(),
                        rm_map=baseline_view,
                        mask=mask_np,
                    )
                    refined_proxy = proxy_render_from_uv_material(
                        albedo=sampled_albedo[view_idx].numpy(),
                        normal=sampled_normal[view_idx].numpy(),
                        rm_map=refined_view,
                        mask=mask_np,
                    )
                    baseline_mse = compute_masked_mse(baseline_proxy, reference_rgb_hwc, mask_np)
                    refined_mse = compute_masked_mse(refined_proxy, reference_rgb_hwc, mask_np)
                    baseline_psnr = compute_masked_psnr(baseline_proxy, reference_rgb_hwc, mask_np)
                    refined_psnr = compute_masked_psnr(refined_proxy, reference_rgb_hwc, mask_np)
                    baseline_ssim = compute_masked_ssim(baseline_proxy, reference_rgb_hwc, mask_np)
                    refined_ssim = compute_masked_ssim(refined_proxy, reference_rgb_hwc, mask_np)
                    if lpips_model is not None:
                        pending_lpips_rows.append(
                            (
                                len(rows),
                                baseline_proxy,
                                refined_proxy,
                                reference_rgb_hwc,
                                mask_np,
                            )
                        )
                    if write_artifacts:
                        view_dir = object_dir / "views"
                        reference_path = view_dir / f"{safe_view_name}_reference_rgb.png"
                        baseline_proxy_path = view_dir / f"{safe_view_name}_baseline_proxy_render.png"
                        refined_proxy_path = view_dir / f"{safe_view_name}_refined_proxy_render.png"
                        save_rgb_tensor_image(reference_path, reference_rgb_hwc * mask_np[..., None])
                        save_rgb_tensor_image(baseline_proxy_path, baseline_proxy)
                        save_rgb_tensor_image(refined_proxy_path, refined_proxy)
                        view_paths = {
                            **view_paths,
                            "reference_rgb": str(reference_path.resolve()),
                            "baseline_proxy_render": str(baseline_proxy_path.resolve()),
                            "refined_proxy_render": str(refined_proxy_path.resolve()),
                        }

                rows.append(
                    {
                        "object_id": object_id,
                        "case_id": case_id,
                        "pair_id": pair_id,
                        "target_bundle_id": target_bundle_id,
                        "prior_variant_id": prior_variant_id,
                        "generator_id": generator_id,
                        "source_name": source_name,
                        "category_bucket": category_bucket,
                        "prior_label": prior_label,
                        "prior_source_type": prior_source_type,
                        "prior_generation_mode": prior_generation_mode,
                        "prior_mode": prior_mode,
                        "prior_variant_type": prior_variant_type,
                        "prior_quality_bin": prior_quality_bin,
                        "prior_spatiality": prior_spatiality,
                        "training_role": training_role,
                        "upstream_model_id": upstream_model_id,
                        "sample_weight": sample_weight,
                        "input_prior_baseline_source": input_prior_baseline_source,
                        "has_material_prior": has_material_prior,
                        "supervision_tier": supervision_tier,
                        "supervision_role": supervision_role,
                        "license_bucket": license_bucket,
                        "target_source_type": target_source_type,
                        "target_prior_identity": target_prior_identity,
                        "target_is_prior_copy": target_is_prior_copy,
                        "target_quality_tier": target_quality_tier,
                        "paper_split": paper_split,
                        "split": split,
                        "material_family": material_family,
                        "lighting_bank_id": lighting_bank_id,
                        "thin_boundary_flag": thin_boundary_flag,
                        "eval_variant": args.eval_variant,
                        "view_name": view_name,
                        "view_target_basis": "sampled_uv_target",
                        "stored_view_target_available": bool(view_targets is not None),
                        "baseline_roughness_mae": baseline_rough_mae,
                        "baseline_metallic_mae": baseline_metal_mae,
                        "refined_roughness_mae": refined_rough_mae,
                        "refined_metallic_mae": refined_metal_mae,
                        "baseline_total_mae": baseline_rough_mae + baseline_metal_mae,
                        "input_prior_total_mae": input_prior_total_view,
                        "refined_total_mae": refined_rough_mae + refined_metal_mae,
                        "gain_total": gain_total_view,
                        "uv_prior_error": input_prior_total_uv,
                        "uv_pred_error": refined_total_uv,
                        "uv_gain": gain_total_uv,
                        "uv_improved": bool(gain_total_uv > 0.0),
                        "view_prior_error": input_prior_total_view,
                        "view_pred_error": refined_total_view,
                        "view_gain": gain_total_view,
                        "view_regressed": bool(gain_total_view < 0.0),
                        "improvement_total": (baseline_rough_mae + baseline_metal_mae)
                        - (refined_rough_mae + refined_metal_mae),
                        "baseline_psnr": baseline_psnr,
                        "refined_psnr": refined_psnr,
                        "baseline_mse": baseline_mse,
                        "refined_mse": refined_mse,
                        "baseline_ssim": baseline_ssim,
                        "refined_ssim": refined_ssim,
                        "baseline_lpips": baseline_lpips,
                        "refined_lpips": refined_lpips,
                        "baseline_boundary_bleed_score": baseline_boundary["score"],
                        "refined_boundary_bleed_score": refined_boundary["score"],
                        "baseline_boundary_edge_error": baseline_boundary["edge_error"],
                        "refined_boundary_edge_error": refined_boundary["edge_error"],
                        "baseline_boundary_interior_error": baseline_boundary["interior_error"],
                        "refined_boundary_interior_error": refined_boundary["interior_error"],
                        "boundary_edge_pixel_rate": refined_boundary["edge_pixel_rate"],
                        "baseline_highlight_localization_error": baseline_highlight["error"],
                        "refined_highlight_localization_error": refined_highlight["error"],
                        "baseline_highlight_iou": baseline_highlight["iou"],
                        "refined_highlight_iou": refined_highlight["iou"],
                        "baseline_highlight_rm_error": baseline_highlight["rm_error"],
                        "refined_highlight_rm_error": refined_highlight["rm_error"],
                        "highlight_pixel_rate": refined_highlight["highlight_pixel_rate"],
                        "baseline_rm_gradient_preservation": baseline_gradient_preservation,
                        "refined_rm_gradient_preservation": refined_gradient_preservation,
                        "prior_residual_safety_score": residual_safety["safety_score"],
                        "prior_residual_safe_improvement_rate": residual_safety["safe_improvement_rate"],
                        "prior_residual_unnecessary_change_rate": residual_safety["unnecessary_change_rate"],
                        "prior_residual_regression_rate": residual_safety["regression_rate"],
                        "prior_residual_changed_pixel_rate": residual_safety["changed_pixel_rate"],
                        "prior_reliability_mean": prior_reliability_mean,
                        "change_gate_mean": change_gate_mean,
                        "mean_abs_delta": mean_abs_delta,
                        "boundary_delta_mean": boundary_delta_mean,
                        "bootstrap_enabled": bootstrap_enabled,
                        "gt_roughness_mean": gt_rough_stats["mean"],
                        "gt_metallic_mean": gt_metal_stats["mean"],
                        "baseline_pred_roughness_mean": base_pred_rough_mean,
                        "baseline_pred_metallic_mean": base_pred_metal_mean,
                        "refined_pred_roughness_mean": refined_pred_rough_mean,
                        "refined_pred_metallic_mean": refined_pred_metal_mean,
                        "brightness_p99": brightness_p99,
                        "highlight_fraction": highlight_fraction,
                        "baseline_tags": baseline_tags,
                        "refined_tags": refined_tags,
                        "baseline_primary_failure": baseline_primary,
                        "refined_primary_failure": refined_primary,
                        "gt_is_metal": bool(gt_metal_stats["mean"] >= METAL_THRESHOLD),
                        "has_effective_view_supervision": True,
                        "runtime_ms_per_object": float(batch_seconds * 1000.0 / max(len(batch["object_id"]), 1)),
                        "paths": {
                            **{key: str(value.resolve()) for key, value in atlas_paths.items()},
                            **view_paths,
                        },
                    }
                )

            if pending_lpips_rows:
                row_indices = [item[0] for item in pending_lpips_rows]
                baseline_lpips_values = compute_lpips_distance_batch(
                    lpips_model,
                    [item[1] for item in pending_lpips_rows],
                    [item[3] for item in pending_lpips_rows],
                    [item[4] for item in pending_lpips_rows],
                    pipeline.device,
                )
                refined_lpips_values = compute_lpips_distance_batch(
                    lpips_model,
                    [item[2] for item in pending_lpips_rows],
                    [item[3] for item in pending_lpips_rows],
                    [item[4] for item in pending_lpips_rows],
                    pipeline.device,
                )
                for row_idx, baseline_lpips_value, refined_lpips_value in zip(
                    row_indices,
                    baseline_lpips_values,
                    refined_lpips_values,
                    strict=True,
                ):
                    rows[row_idx]["baseline_lpips"] = baseline_lpips_value
                    rows[row_idx]["refined_lpips"] = refined_lpips_value

        processed_records += len(batch["object_id"])
        if args.log_every > 0 and (
            batch_index == 1 or batch_index % args.log_every == 0 or batch_index == total_batches
        ):
            elapsed = time.perf_counter() - eval_started_at
            avg_batch_seconds = elapsed / max(batch_index, 1)
            eta_seconds = avg_batch_seconds * max(total_batches - batch_index, 0)
            print(
                "[eval] "
                f"batch={batch_index}/{total_batches} "
                f"records={processed_records}/{total_records} "
                f"rows={len(rows)} "
                f"elapsed={format_duration(elapsed)} "
                f"eta={format_duration(eta_seconds)} "
                f"avg_batch_s={avg_batch_seconds:.4f} "
                f"device={pipeline.device}"
            )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    object_rows = aggregate_object_rows(rows)
    object_metrics_path = output_dir / "object_metrics.json"
    object_metrics_path.write_text(
        json.dumps(object_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    baseline_tag_counts = Counter()
    refined_tag_counts = Counter()
    for row in rows:
        for tag in row.get("baseline_tags", []):
            baseline_tag_counts[tag] += 1
        for tag in row.get("refined_tags", []):
            refined_tag_counts[tag] += 1

    uv_baseline_roughness_mae = optional_mean(summary["baseline_roughness_mae"]) or 0.0
    uv_baseline_metallic_mae = optional_mean(summary["baseline_metallic_mae"]) or 0.0
    uv_refined_roughness_mae = optional_mean(summary["refined_roughness_mae"]) or 0.0
    uv_refined_metallic_mae = optional_mean(summary["refined_metallic_mae"]) or 0.0
    uv_baseline_total_mae = optional_mean(summary["baseline_total_mae"]) or 0.0
    uv_refined_total_mae = optional_mean(summary["refined_total_mae"]) or 0.0
    uv_improvement_total = float(uv_baseline_total_mae - uv_refined_total_mae)
    view_baseline_total_mae = optional_mean([row.get("baseline_total_mae") for row in rows])
    view_refined_total_mae = optional_mean([row.get("refined_total_mae") for row in rows])
    view_improvement_total = optional_delta(
        view_baseline_total_mae,
        view_refined_total_mae,
        higher_is_better=False,
    )
    object_baseline_total_mae = optional_mean([row.get("baseline_total_mae") for row in object_rows])
    object_refined_total_mae = optional_mean([row.get("refined_total_mae") for row in object_rows])
    object_improvement_total = optional_delta(
        object_baseline_total_mae,
        object_refined_total_mae,
        higher_is_better=False,
    )
    uv_direction_rates = pair_direction_rates(
        summary["baseline_total_mae"],
        summary["refined_total_mae"],
    )
    view_direction_rates = pair_direction_rates(
        [row.get("baseline_total_mae") for row in rows],
        [row.get("refined_total_mae") for row in rows],
    )
    object_direction_rates = pair_direction_rates(
        [row.get("baseline_total_mae") for row in object_rows],
        [row.get("refined_total_mae") for row in object_rows],
    )
    uv_confusion_baseline = binary_confusion_metrics(
        summary["metal_labels"],
        summary["baseline_metal_scores"],
        METAL_THRESHOLD,
    )
    uv_confusion_refined = binary_confusion_metrics(
        summary["metal_labels"],
        summary["refined_metal_scores"],
        METAL_THRESHOLD,
    )
    view_labels = [int(bool(row.get("gt_is_metal", False))) for row in rows]
    view_baseline_metal_scores = [float(row.get("baseline_pred_metallic_mean", 0.0)) for row in rows]
    view_refined_metal_scores = [float(row.get("refined_pred_metallic_mean", 0.0)) for row in rows]
    object_labels = [int(bool(row.get("gt_is_metal", False))) for row in object_rows]
    object_baseline_metal_scores = [
        float(finite_or_none(row.get("baseline_pred_metallic_mean")) or 0.0)
        for row in object_rows
    ]
    object_refined_metal_scores = [
        float(finite_or_none(row.get("refined_pred_metallic_mean")) or 0.0)
        for row in object_rows
    ]
    view_confusion_baseline = binary_confusion_metrics(
        view_labels,
        view_baseline_metal_scores,
        METAL_THRESHOLD,
    )
    view_confusion_refined = binary_confusion_metrics(
        view_labels,
        view_refined_metal_scores,
        METAL_THRESHOLD,
    )
    object_confusion_baseline = binary_confusion_metrics(
        object_labels,
        object_baseline_metal_scores,
        METAL_THRESHOLD,
    )
    object_confusion_refined = binary_confusion_metrics(
        object_labels,
        object_refined_metal_scores,
        METAL_THRESHOLD,
    )
    residual_safety_summary = {
        key: optional_mean([item.get(key) for item in summary["residual_safety"]])
        for key in [
            "changed_pixel_rate",
            "safe_improvement_rate",
            "unnecessary_change_rate",
            "regression_rate",
            "safety_score",
        ]
    }
    confidence_bin_summary = summarize_confidence_bins(summary["confidence_bins"])
    grouped_summaries = {
        "by_source_name": summarize_group_rows(rows, "source_name"),
        "by_generator_id": summarize_group_rows(rows, "generator_id"),
        "by_license_bucket": summarize_group_rows(rows, "license_bucket"),
        "by_category_bucket": summarize_group_rows(rows, "category_bucket"),
        "by_prior_label": summarize_group_rows(rows, "prior_label"),
        "by_prior_source_type": summarize_group_rows(rows, "prior_source_type"),
        "by_prior_mode": summarize_group_rows(rows, "prior_mode"),
        "by_prior_variant_type": summarize_group_rows(rows, "prior_variant_type"),
        "by_prior_quality_bin": summarize_group_rows(rows, "prior_quality_bin"),
        "by_prior_spatiality": summarize_group_rows(rows, "prior_spatiality"),
        "by_training_role": summarize_group_rows(rows, "training_role"),
        "by_split": summarize_group_rows(rows, "split"),
        "by_supervision_tier": summarize_group_rows(rows, "supervision_tier"),
        "by_supervision_role": summarize_group_rows(rows, "supervision_role"),
        "by_target_quality_tier": summarize_group_rows(rows, "target_quality_tier"),
        "by_paper_split": summarize_group_rows(rows, "paper_split"),
        "by_material_family": summarize_group_rows(rows, "material_family"),
        "by_lighting_bank_id": summarize_group_rows(rows, "lighting_bank_id"),
        "by_thin_boundary_flag": summarize_group_rows(rows, "thin_boundary_flag"),
        "by_view_name": summarize_group_rows(rows, "view_name"),
    }
    failure_tag_reduction = {
        tag: {
            "baseline": int(baseline_tag_counts.get(tag, 0)),
            "refined": int(refined_tag_counts.get(tag, 0)),
            "reduction": int(baseline_tag_counts.get(tag, 0) - refined_tag_counts.get(tag, 0)),
            "relative_reduction": (
                float((baseline_tag_counts.get(tag, 0) - refined_tag_counts.get(tag, 0)) / baseline_tag_counts[tag])
                if baseline_tag_counts.get(tag, 0)
                else None
            ),
        }
        for tag in FAILURE_TAGS
    }
    metric_disagreement = build_metric_disagreement(
        uv_improvement=uv_improvement_total,
        view_improvement=view_improvement_total,
        object_improvement=object_improvement_total,
    )
    disagreement_json_path, disagreement_html_path = write_metric_disagreement_report(
        output_dir,
        metric_disagreement,
    )
    diagnostic_cases_path = write_diagnostic_cases(output_dir, rows)
    effective_view_supervision_rate = float(
        summary["effective_view_supervision_samples"] / max(len(dataset.records), 1)
    )
    metrics_main = {
        "uv_rm_mae": {
            "roughness": metric_pair(
                baseline=uv_baseline_roughness_mae,
                refined=uv_refined_roughness_mae,
                higher_is_better=False,
                count=len(summary["baseline_roughness_mae"]),
            ),
            "metallic": metric_pair(
                baseline=uv_baseline_metallic_mae,
                refined=uv_refined_metallic_mae,
                higher_is_better=False,
                count=len(summary["baseline_metallic_mae"]),
            ),
            "total": metric_pair(
                baseline=uv_baseline_total_mae,
                refined=uv_refined_total_mae,
                higher_is_better=False,
                count=len(summary["baseline_total_mae"]),
            ),
        },
        "view_rm_mae": {
            "total": metric_pair(
                baseline=view_baseline_total_mae,
                refined=view_refined_total_mae,
                higher_is_better=False,
                count=len(rows),
            )
        },
        "proxy_render_psnr": collect_optional_pair(
            rows,
            "baseline_psnr",
            "refined_psnr",
            higher_is_better=True,
            mode=args.render_metric_mode,
        ),
        "proxy_render_mse": collect_optional_pair(
            rows,
            "baseline_mse",
            "refined_mse",
            higher_is_better=False,
            mode=args.render_metric_mode,
        ),
        "proxy_render_ssim": collect_optional_pair(
            rows,
            "baseline_ssim",
            "refined_ssim",
            higher_is_better=True,
            mode=args.render_metric_mode,
        ),
        "proxy_render_lpips": collect_optional_pair(
            rows,
            "baseline_lpips",
            "refined_lpips",
            higher_is_better=False,
            mode=args.render_metric_mode,
        ),
    }
    metrics_material_specific = {
        "boundary_bleed_score": collect_optional_pair(
            rows,
            "baseline_boundary_bleed_score",
            "refined_boundary_bleed_score",
            higher_is_better=False,
        ),
        "metal_nonmetal_confusion": {
            "threshold": METAL_THRESHOLD,
            "uv_level": {
                "baseline": uv_confusion_baseline,
                "refined": uv_confusion_refined,
            },
            "view_level": {
                "baseline": view_confusion_baseline,
                "refined": view_confusion_refined,
            },
            "object_level": {
                "baseline": object_confusion_baseline,
                "refined": object_confusion_refined,
            },
        },
        "highlight_localization_error": collect_optional_pair(
            rows,
            "baseline_highlight_localization_error",
            "refined_highlight_localization_error",
            higher_is_better=False,
        ),
        "rm_gradient_preservation": collect_optional_pair(
            rows,
            "baseline_rm_gradient_preservation",
            "refined_rm_gradient_preservation",
            higher_is_better=True,
        ),
        "prior_residual_safety": residual_safety_summary,
        "confidence_calibrated_error": confidence_bin_summary,
        "material_family_breakdown": grouped_summaries["by_material_family"],
    }
    metric_availability = build_metric_availability(
        rows=rows,
        dataset_size=len(dataset.records),
        render_metric_mode=args.render_metric_mode,
        lpips_status=lpips_status,
        effective_view_supervision_rate=effective_view_supervision_rate,
    )
    summary_payload = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "eval_variant": args.eval_variant,
        "rows": len(rows),
        "objects": len(object_rows),
        "dataset_summary": summarize_records(dataset.records),
        "effective_view_supervision_samples": int(summary["effective_view_supervision_samples"]),
        "effective_view_supervision_rate": effective_view_supervision_rate,
        "render_metric_mode": args.render_metric_mode,
        "metric_registry": {
            "main": MAIN_METRIC_NAMES,
            "material_specific": MATERIAL_SPECIFIC_METRIC_NAMES,
            "diagnostics": [
                "before_after_visuals",
                "stratified_reliability",
                "metric_disagreement",
                "failure_taxonomy",
            ],
        },
        "metric_availability": metric_availability,
        "baseline_roughness_mae": uv_baseline_roughness_mae,
        "baseline_metallic_mae": uv_baseline_metallic_mae,
        "refined_roughness_mae": uv_refined_roughness_mae,
        "refined_metallic_mae": uv_refined_metallic_mae,
        "baseline_total_mae": uv_baseline_total_mae,
        "input_prior_total_mae": uv_baseline_total_mae,
        "refined_total_mae": uv_refined_total_mae,
        "gain_total": uv_improvement_total,
        "avg_improvement_total": uv_improvement_total,
        "improvement_rate": uv_direction_rates["improvement_rate"],
        "regression_rate": uv_direction_rates["regression_rate"],
        "tie_rate": uv_direction_rates["tie_rate"],
        "metric_basis_note": (
            "Top-level improvement/regression rates are UV-object rates and share the same basis as "
            "top-level input_prior_total_mae/refined_total_mae. View/object rates are reported under "
            "view_level and object_level."
        ),
        "prior_reliability_mean": optional_mean([row.get("prior_reliability_mean") for row in rows]),
        "change_gate_mean": optional_mean([row.get("change_gate_mean") for row in rows]),
        "mean_abs_delta": optional_mean([row.get("mean_abs_delta") for row in rows]),
        "boundary_delta_mean": optional_mean([row.get("boundary_delta_mean") for row in rows]),
        "bootstrap_enabled_rate": optional_mean([row.get("bootstrap_enabled") for row in rows]),
        "metal_nonmetal": {
            "threshold": METAL_THRESHOLD,
            "baseline_f1": uv_confusion_baseline["f1"],
            "refined_f1": uv_confusion_refined["f1"],
            "baseline_auroc": uv_confusion_baseline["auroc"],
            "refined_auroc": uv_confusion_refined["auroc"],
            "baseline_balanced_accuracy": uv_confusion_baseline["balanced_accuracy"],
            "refined_balanced_accuracy": uv_confusion_refined["balanced_accuracy"],
            "baseline_confusion_rate": uv_confusion_baseline["confusion_rate"],
            "refined_confusion_rate": uv_confusion_refined["confusion_rate"],
        },
        "baseline_tag_counts": dict(baseline_tag_counts),
        "refined_tag_counts": dict(refined_tag_counts),
        "failure_tag_reduction": failure_tag_reduction,
        "runtime": {
            "avg_batch_seconds": float(np.mean(summary["batch_seconds"])) if summary["batch_seconds"] else 0.0,
            "avg_object_seconds": float(sum(summary["batch_seconds"]) / max(len(dataset.records), 1)),
        },
        "memory": {
            "peak_allocated_gb": float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024**3)) if device.startswith("cuda") else 0.0,
            "peak_reserved_gb": float(torch.cuda.max_memory_reserved(torch.device(device)) / (1024**3)) if device.startswith("cuda") else 0.0,
        },
        **grouped_summaries,
        "object_level": {
            "objects": len(object_rows),
            "metric_level": "object_mean_of_view_rows",
            "baseline_total_mae": object_baseline_total_mae,
            "input_prior_total_mae": object_baseline_total_mae,
            "refined_total_mae": object_refined_total_mae,
            "gain_total": object_improvement_total,
            "avg_improvement_total": object_improvement_total,
            "improvement_rate": object_direction_rates["improvement_rate"],
            "regression_rate": object_direction_rates["regression_rate"],
            "tie_rate": object_direction_rates["tie_rate"],
            "by_source_name": summarize_group_rows(object_rows, "source_name"),
            "by_generator_id": summarize_group_rows(object_rows, "generator_id"),
            "by_prior_label": summarize_group_rows(object_rows, "prior_label"),
            "by_prior_source_type": summarize_group_rows(object_rows, "prior_source_type"),
            "by_prior_mode": summarize_group_rows(object_rows, "prior_mode"),
            "by_prior_variant_type": summarize_group_rows(object_rows, "prior_variant_type"),
            "by_prior_quality_bin": summarize_group_rows(object_rows, "prior_quality_bin"),
            "by_split": summarize_group_rows(object_rows, "split"),
            "by_target_quality_tier": summarize_group_rows(object_rows, "target_quality_tier"),
            "by_material_family": summarize_group_rows(object_rows, "material_family"),
            "by_paper_split": summarize_group_rows(object_rows, "paper_split"),
        },
        "view_level": {
            "metric_level": "view_rows",
            "rows": len(rows),
            "baseline_total_mae": view_baseline_total_mae,
            "input_prior_total_mae": view_baseline_total_mae,
            "refined_total_mae": view_refined_total_mae,
            "gain_total": view_improvement_total,
            "avg_improvement_total": view_improvement_total,
            "improvement_rate": view_direction_rates["improvement_rate"],
            "regression_rate": view_direction_rates["regression_rate"],
            "tie_rate": view_direction_rates["tie_rate"],
        },
        "metrics_main": metrics_main,
        "metrics_material_specific": metrics_material_specific,
        "metrics_diagnostics": {
            "before_after_visuals": {
                "artifact_root": str((output_dir / "artifacts").resolve()),
                "diagnostic_cases_json": str(diagnostic_cases_path.resolve()),
                "contains_atlas_error_maps": True,
                "contains_proxy_render_views": args.render_metric_mode == "proxy_uv_shading",
                "max_artifact_objects": int(args.max_artifact_objects),
            },
            "stratified_reliability": {
                "min_group_count_for_paper": int(args.diagnostic_min_group_count),
                "groups": list(grouped_summaries.keys()),
            },
            "metric_disagreement": metric_disagreement,
            "failure_taxonomy": {
                "failure_tags": FAILURE_TAGS,
                "tag_reduction": failure_tag_reduction,
                "diagnostic_cases_json": str(diagnostic_cases_path.resolve()),
            },
        },
        "metrics_by_group": grouped_summaries,
        "diagnostic_reports": {
            "metric_disagreement_json": str(disagreement_json_path.resolve()),
            "metric_disagreement_html": str(disagreement_html_path.resolve()),
            "diagnostic_cases_json": str(diagnostic_cases_path.resolve()),
        },
    }
    summary_payload["metric_warnings"] = build_metric_warnings(
        summary_payload=summary_payload,
        diagnostic_min_group_count=args.diagnostic_min_group_count,
    )
    summary_payload["paper_main_table"] = {
        "method_order": [
            "Original Generator Asset",
            "Input Prior",
            "Prior Smoothing",
            "Scalar Broadcast",
            "Ours w/o View",
            "Ours w/o Residual",
            "Ours Full",
        ],
        "metric_columns": PAPER_MAIN_METRIC_COLUMNS,
        "entries": build_paper_main_table_entries(args.eval_variant, metrics_main),
    }
    summary_path = output_dir / "summary.json"
    summary_payload["summary_json"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(make_json_serializable(summary_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    by_variant_dir = output_dir / "eval"
    by_variant_dir.mkdir(parents=True, exist_ok=True)
    by_variant_payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "split": args.split,
        "overall": {
            "rows": len(rows),
            "objects": len(object_rows),
            "input_prior_total_mae": uv_baseline_total_mae,
            "refined_total_mae": uv_refined_total_mae,
            "gain_total": uv_improvement_total,
            "improvement_rate": uv_direction_rates["improvement_rate"],
            "regression_rate": uv_direction_rates["regression_rate"],
        },
        "by_prior_variant_type": grouped_summaries["by_prior_variant_type"],
        "by_prior_quality_bin": grouped_summaries["by_prior_quality_bin"],
        "by_split": grouped_summaries["by_split"],
        "object_level": {
            "by_prior_variant_type": summary_payload["object_level"]["by_prior_variant_type"],
            "by_prior_quality_bin": summary_payload["object_level"]["by_prior_quality_bin"],
            "by_split": summary_payload["object_level"]["by_split"],
        },
    }
    by_variant_summary_path = by_variant_dir / "by_variant_summary.json"
    by_variant_summary_path.write_text(
        json.dumps(make_json_serializable(by_variant_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_lines = [
        "# Eval By Variant Report",
        "",
        f"- generated_at_utc: `{by_variant_payload['generated_at_utc']}`",
        f"- rows: `{len(rows)}`",
        f"- objects: `{len(object_rows)}`",
        "",
        "## Prior Variant Type",
        "",
        "| variant | count | input_prior_total_mae | refined_total_mae | gain_total | regression_rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, metrics in sorted(grouped_summaries["by_prior_variant_type"].items()):
        report_lines.append(
            "| "
            + " | ".join(
                [
                    str(variant),
                    str(metrics.get("count")),
                    f"{(metrics.get('input_prior_total_mae') or 0.0):.6f}",
                    f"{(metrics.get('refined_total_mae') or 0.0):.6f}",
                    f"{(metrics.get('gain_total') or 0.0):.6f}",
                    f"{(metrics.get('prior_residual_regression_rate') or 0.0):.6f}",
                ]
            )
            + " |"
        )
    (by_variant_dir / "by_variant_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary_payload["by_variant_summary_json"] = str(by_variant_summary_path.resolve())
    if args.print_summary_json:
        print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(compact_eval_console_summary(summary_payload), indent=2, ensure_ascii=False))

    report_path = None
    if not args.skip_report:
        from export_refined_material_report import build_report

        report_path = build_report(metrics_path, output_dir)

    if run is not None:
        log_payload = {
            "eval/effective_view_supervision_rate": summary_payload["effective_view_supervision_rate"],
            "eval/runtime/avg_object_seconds": summary_payload["runtime"]["avg_object_seconds"],
            "eval/object_level/avg_improvement_total": summary_payload["object_level"]["avg_improvement_total"],
            "eval/memory/peak_allocated_gb": summary_payload["memory"]["peak_allocated_gb"],
        }
        main_metrics = summary_payload["metrics_main"]
        special_metrics = summary_payload["metrics_material_specific"]
        log_payload.update(
            {
                "eval/main/psnr": main_metrics["proxy_render_psnr"]["refined"],
                "eval/main/mse": main_metrics["proxy_render_mse"]["refined"],
                "eval/main/ssim": main_metrics["proxy_render_ssim"]["refined"],
                "eval/main/lpips": main_metrics["proxy_render_lpips"]["refined"],
                "eval/input_prior_total_mae": summary_payload["input_prior_total_mae"],
                "eval/refined_total_mae": summary_payload["refined_total_mae"],
                "eval/gain_total": summary_payload["gain_total"],
                "eval/improvement_rate": summary_payload.get("improvement_rate"),
                "eval/regression_rate": summary_payload.get("regression_rate"),
                "eval/view_level/improvement_rate": summary_payload.get("view_level", {}).get("improvement_rate"),
                "eval/view_level/regression_rate": summary_payload.get("view_level", {}).get("regression_rate"),
                "eval/object_level/improvement_rate": summary_payload.get("object_level", {}).get("improvement_rate"),
                "eval/object_level/regression_rate": summary_payload.get("object_level", {}).get("regression_rate"),
                "eval/diagnostics/prior_reliability_mean": summary_payload.get("prior_reliability_mean"),
                "eval/diagnostics/change_gate_mean": summary_payload.get("change_gate_mean"),
                "eval/diagnostics/mean_abs_delta": summary_payload.get("mean_abs_delta"),
                "eval/diagnostics/boundary_delta_mean": summary_payload.get("boundary_delta_mean"),
                "eval/diagnostics/bootstrap_enabled_rate": summary_payload.get("bootstrap_enabled_rate"),
                "eval/rm/uv_total_mae": main_metrics["uv_rm_mae"]["total"]["refined"],
                "eval/rm/view_total_mae": main_metrics["view_rm_mae"]["total"]["refined"],
                "eval/special/boundary_bleed_score": special_metrics["boundary_bleed_score"]["refined"],
                "eval/special/metal_confusion_rate": special_metrics["metal_nonmetal_confusion"]["uv_level"]["refined"]["confusion_rate"],
                "eval/special/highlight_localization_error": special_metrics["highlight_localization_error"]["refined"],
                "eval/special/rm_gradient_preservation": special_metrics["rm_gradient_preservation"]["refined"],
                "eval/special/prior_residual_safety": special_metrics["prior_residual_safety"]["safety_score"],
            }
        )
        if args.wandb_log_group_breakdowns:
            log_payload.update(compact_eval_group_logs(summary_payload))
        for tag in FAILURE_TAGS:
            tag_payload = (summary_payload.get("failure_tag_reduction") or {}).get(tag) or {}
            reduction = tag_payload.get("reduction")
            if reduction is not None:
                log_payload[f"eval/failure_tag_reduction/{tag}/reduction"] = reduction
        log_payload = {key: value for key, value in log_payload.items() if value is not None}
        sanitized_logs, skipped_logs = sanitize_log_dict(log_payload)
        if skipped_logs:
            print(json.dumps({"skipped_eval_logs": skipped_logs}, ensure_ascii=False))
        run.log(sanitized_logs)

        if wandb is not None and bool(args.wandb_log_paper_main_table):
            paper_table = wandb.Table(
                columns=[
                    "method",
                    *PAPER_MAIN_METRIC_COLUMNS,
                    "metric_basis",
                    "note",
                ]
            )
            for row in summary_payload["paper_main_table"]["entries"]:
                paper_table.add_data(
                    row.get("method"),
                    *[row.get(metric) for metric in PAPER_MAIN_METRIC_COLUMNS],
                    row.get("metric_basis"),
                    row.get("note"),
                )
            run.log({"eval/paper_main_table_entries": paper_table})

        if wandb is not None and bool(args.wandb_log_top_cases):
            preview_table = wandb.Table(
                columns=[
                    "object_id",
                    "view_name",
                    "generator_id",
                    "source_name",
                    "prior_label",
                    "prior_source_type",
                    "prior_mode",
                    "target_source_type",
                    "baseline_total_mae",
                    "input_prior_total_mae",
                    "refined_total_mae",
                    "gain_total",
                    "improvement_total",
                    "baseline_roughness",
                    "baseline_metallic",
                    "refined_roughness",
                    "refined_metallic",
                    "target_roughness",
                    "target_metallic",
                    "baseline_error",
                    "refined_error",
                    "reference_rgb",
                    "baseline_proxy_render",
                    "refined_proxy_render",
                ]
            )
            ranked_rows = sorted(rows, key=lambda row: row["improvement_total"], reverse=True)
            for row in ranked_rows[: args.wandb_max_rows]:
                paths = row.get("paths", {})
                preview_table.add_data(
                    row["object_id"],
                    row["view_name"],
                    row.get("generator_id", "unknown"),
                    row.get("source_name", "unknown"),
                    row.get("prior_label", "unknown"),
                    row.get("prior_source_type", "unknown"),
                    row.get("prior_mode", "unknown"),
                    row.get("target_source_type", "unknown"),
                    row["baseline_total_mae"],
                    row.get("input_prior_total_mae", row["baseline_total_mae"]),
                    row["refined_total_mae"],
                    row.get("gain_total", row["improvement_total"]),
                    row["improvement_total"],
                    wandb.Image(paths["baseline_roughness"]) if paths.get("baseline_roughness") else None,
                    wandb.Image(paths["baseline_metallic"]) if paths.get("baseline_metallic") else None,
                    wandb.Image(paths["refined_roughness"]) if paths.get("refined_roughness") else None,
                    wandb.Image(paths["refined_metallic"]) if paths.get("refined_metallic") else None,
                    wandb.Image(paths["target_roughness"]) if paths.get("target_roughness") else None,
                    wandb.Image(paths["target_metallic"]) if paths.get("target_metallic") else None,
                    wandb.Image(paths["baseline_rm_error"]) if paths.get("baseline_rm_error") else None,
                    wandb.Image(paths["refined_rm_error"]) if paths.get("refined_rm_error") else None,
                    wandb.Image(paths["reference_rgb"]) if paths.get("reference_rgb") else None,
                    wandb.Image(paths["baseline_proxy_render"]) if paths.get("baseline_proxy_render") else None,
                    wandb.Image(paths["refined_proxy_render"]) if paths.get("refined_proxy_render") else None,
                )
            run.log({"eval/top_cases": preview_table})

        if args.wandb_log_artifacts and args.wandb_artifact_policy != "none":
            artifact_paths = [
                metrics_path,
                object_metrics_path,
                summary_path,
                disagreement_json_path,
                disagreement_html_path,
                diagnostic_cases_path,
            ]
            if args.wandb_artifact_policy == "full":
                artifact_paths.append(output_dir / "artifacts")
            if report_path is not None:
                artifact_paths.append(report_path)
            paper_metric_summary_path = output_dir / "paper_metric_summary.html"
            if paper_metric_summary_path.exists():
                artifact_paths.append(paper_metric_summary_path)
            log_path_artifact(
                run,
                name=f"{run_name}-eval",
                artifact_type="evaluation",
                paths=artifact_paths,
            )
        run.finish()


if __name__ == "__main__":
    main()
