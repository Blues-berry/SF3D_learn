from __future__ import annotations

import argparse
import json
import sys
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

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "material_refiner_eval"
DEFAULT_MANIFEST = REPO_ROOT / "output" / "material_refine" / "canonical_manifest_v1.json"
FAILURE_TAGS = [
    "over_smoothing",
    "metal_nonmetal_confusion",
    "local_highlight_misread",
    "boundary_bleed",
]


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--split-strategy", choices=["auto", "manifest", "hash"], default="auto")
    parser.add_argument("--hash-val-ratio", type=float, default=0.1)
    parser.add_argument("--hash-test-ratio", type=float, default=0.1)
    parser.add_argument("--generator-ids", type=str, default=None)
    parser.add_argument("--source-names", type=str, default=None)
    parser.add_argument("--supervision-tiers", type=str, default=None)
    parser.add_argument("--license-buckets", type=str, default=None)
    parser.add_argument("--require-prior", type=str, default="any")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cuda-device-index", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-report", action="store_true")
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
    parser.add_argument("--wandb-log-artifacts", type=parse_bool, default=True)
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
    args.generator_ids = parse_csv_list(args.generator_ids)
    args.source_names = parse_csv_list(args.source_names)
    args.supervision_tiers = parse_csv_list(args.supervision_tiers)
    args.license_buckets = parse_csv_list(args.license_buckets)
    args.require_prior = parse_optional_bool(args.require_prior)
    return args


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


def luminance_p99_from_rgba(rgba_path: str | None) -> tuple[float, float]:
    if not rgba_path:
        return 0.0, 0.0
    path = Path(rgba_path)
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
            "baseline_total_mae": 0.0,
            "refined_total_mae": 0.0,
            "improvement_total": 0.0,
        }
    )
    for row in rows:
        key = str(row.get(key_name, "unknown"))
        bucket = grouped[key]
        bucket["count"] += 1.0
        bucket["baseline_total_mae"] += float(row["baseline_total_mae"])
        bucket["refined_total_mae"] += float(row["refined_total_mae"])
        bucket["improvement_total"] += float(row["improvement_total"])
    finalized = {}
    for key, bucket in grouped.items():
        count = max(bucket["count"], 1.0)
        finalized[key] = {
            "count": int(bucket["count"]),
            "baseline_total_mae": bucket["baseline_total_mae"] / count,
            "refined_total_mae": bucket["refined_total_mae"] / count,
            "improvement_total": bucket["improvement_total"] / count,
        }
    return finalized


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
        license_buckets=args.license_buckets,
        require_prior=args.require_prior,
        max_records=args.max_samples,
        atlas_size=atlas_size,
        buffer_resolution=buffer_resolution,
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

    device = resolve_device(args)
    pipeline = MaterialRefinementPipeline.from_checkpoint(
        args.checkpoint,
        device=device,
        cuda_device_index=args.cuda_device_index,
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
    }

    for batch in loader:
        device_batch = move_batch_to_device(batch, pipeline.device)
        outputs = pipeline.model(device_batch)
        outputs = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in outputs.items()
        }
        baseline = outputs["baseline"]
        refined = outputs["refined"]
        target = torch.cat(
            [batch["uv_target_roughness"], batch["uv_target_metallic"]],
            dim=1,
        )
        confidence = batch["uv_target_confidence"]

        for item_idx, object_id in enumerate(batch["object_id"]):
            object_dir = output_dir / "artifacts" / object_id
            atlas_paths = save_atlas_bundle(
                object_dir,
                baseline_roughness=baseline[item_idx, 0],
                baseline_metallic=baseline[item_idx, 1],
                refined_roughness=refined[item_idx, 0],
                refined_metallic=refined[item_idx, 1],
                confidence=confidence[item_idx],
            )

            base_uv_mae = (baseline[item_idx] - target[item_idx]).abs() * confidence[item_idx]
            refined_uv_mae = (refined[item_idx] - target[item_idx]).abs() * confidence[item_idx]
            uv_weight = float(confidence[item_idx].sum().item()) or 1.0
            baseline_uv_roughness_mae = float(base_uv_mae[0].sum().item() / uv_weight)
            baseline_uv_metallic_mae = float(base_uv_mae[1].sum().item() / uv_weight)
            refined_uv_roughness_mae = float(refined_uv_mae[0].sum().item() / uv_weight)
            refined_uv_metallic_mae = float(refined_uv_mae[1].sum().item() / uv_weight)

            metadata = batch["metadata"][item_idx]
            generator_id = str(batch["generator_id"][item_idx])
            source_name = str(batch["source_name"][item_idx])
            prior_label = "with_prior" if bool(batch["has_material_prior"][item_idx]) else "without_prior"
            supervision_tier = str(batch["supervision_tier"][item_idx])

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

            view_targets = batch["view_targets"]
            view_uvs = batch["view_uvs"]
            if view_targets is None or view_uvs is None:
                rows.append(
                    {
                        "object_id": object_id,
                        "generator_id": generator_id,
                        "source_name": source_name,
                        "prior_label": prior_label,
                        "supervision_tier": supervision_tier,
                        "view_name": "uv_space",
                        "baseline_roughness_mae": baseline_uv_roughness_mae,
                        "baseline_metallic_mae": baseline_uv_metallic_mae,
                        "refined_roughness_mae": refined_uv_roughness_mae,
                        "refined_metallic_mae": refined_uv_metallic_mae,
                        "baseline_total_mae": baseline_uv_roughness_mae + baseline_uv_metallic_mae,
                        "refined_total_mae": refined_uv_roughness_mae + refined_uv_metallic_mae,
                        "improvement_total": (baseline_uv_roughness_mae + baseline_uv_metallic_mae)
                        - (refined_uv_roughness_mae + refined_uv_metallic_mae),
                        "baseline_tags": [],
                        "refined_tags": [],
                        "baseline_primary_failure": "none",
                        "refined_primary_failure": "none",
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

            for view_idx, view_name in enumerate(batch["view_names"][item_idx]):
                gt_view = view_targets[item_idx, view_idx]
                mask = batch["view_masks"][item_idx, view_idx, 0] > 0.5
                if not mask.any():
                    continue
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

                rows.append(
                    {
                        "object_id": object_id,
                        "generator_id": generator_id,
                        "source_name": source_name,
                        "prior_label": prior_label,
                        "supervision_tier": supervision_tier,
                        "view_name": view_name,
                        "baseline_roughness_mae": baseline_rough_mae,
                        "baseline_metallic_mae": baseline_metal_mae,
                        "refined_roughness_mae": refined_rough_mae,
                        "refined_metallic_mae": refined_metal_mae,
                        "baseline_total_mae": baseline_rough_mae + baseline_metal_mae,
                        "refined_total_mae": refined_rough_mae + refined_metal_mae,
                        "improvement_total": (baseline_rough_mae + baseline_metal_mae)
                        - (refined_rough_mae + refined_metal_mae),
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
                        "paths": {key: str(value.resolve()) for key, value in atlas_paths.items()},
                    }
                )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    baseline_tag_counts = Counter()
    refined_tag_counts = Counter()
    for row in rows:
        for tag in row.get("baseline_tags", []):
            baseline_tag_counts[tag] += 1
        for tag in row.get("refined_tags", []):
            refined_tag_counts[tag] += 1

    summary_payload = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "rows": len(rows),
        "dataset_summary": summarize_records(dataset.records),
        "baseline_roughness_mae": float(np.mean(summary["baseline_roughness_mae"])) if summary["baseline_roughness_mae"] else 0.0,
        "baseline_metallic_mae": float(np.mean(summary["baseline_metallic_mae"])) if summary["baseline_metallic_mae"] else 0.0,
        "refined_roughness_mae": float(np.mean(summary["refined_roughness_mae"])) if summary["refined_roughness_mae"] else 0.0,
        "refined_metallic_mae": float(np.mean(summary["refined_metallic_mae"])) if summary["refined_metallic_mae"] else 0.0,
        "baseline_total_mae": float(np.mean(summary["baseline_total_mae"])) if summary["baseline_total_mae"] else 0.0,
        "refined_total_mae": float(np.mean(summary["refined_total_mae"])) if summary["refined_total_mae"] else 0.0,
        "avg_improvement_total": float(
            np.mean(np.asarray(summary["baseline_total_mae"]) - np.asarray(summary["refined_total_mae"]))
        ) if summary["baseline_total_mae"] else 0.0,
        "baseline_tag_counts": dict(baseline_tag_counts),
        "refined_tag_counts": dict(refined_tag_counts),
        "by_source_name": summarize_group_rows(rows, "source_name"),
        "by_generator_id": summarize_group_rows(rows, "generator_id"),
        "by_prior_label": summarize_group_rows(rows, "prior_label"),
        "by_supervision_tier": summarize_group_rows(rows, "supervision_tier"),
        "by_view_name": summarize_group_rows(rows, "view_name"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(make_json_serializable(summary_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))

    report_path = None
    if not args.skip_report:
        from export_refined_material_report import build_report

        report_path = build_report(metrics_path, output_dir)

    if run is not None:
        log_payload = {
            "eval/baseline_total_mae": summary_payload["baseline_total_mae"],
            "eval/refined_total_mae": summary_payload["refined_total_mae"],
            "eval/avg_improvement_total": summary_payload["avg_improvement_total"],
        }
        log_payload.update(flatten_for_logging(summary_payload["dataset_summary"], prefix="dataset"))
        log_payload.update(flatten_for_logging(summary_payload["by_generator_id"], prefix="eval/by_generator_id"))
        log_payload.update(flatten_for_logging(summary_payload["by_source_name"], prefix="eval/by_source_name"))
        log_payload.update(flatten_for_logging(summary_payload["by_prior_label"], prefix="eval/by_prior_label"))
        log_payload.update(flatten_for_logging(summary_payload["by_supervision_tier"], prefix="eval/by_supervision_tier"))
        log_payload.update(flatten_for_logging(summary_payload["by_view_name"], prefix="eval/by_view_name"))
        sanitized_logs, skipped_logs = sanitize_log_dict(log_payload)
        if skipped_logs:
            print(json.dumps({"skipped_eval_logs": skipped_logs}, ensure_ascii=False))
        run.log(sanitized_logs)

        if wandb is not None:
            preview_table = wandb.Table(
                columns=[
                    "object_id",
                    "view_name",
                    "generator_id",
                    "source_name",
                    "prior_label",
                    "baseline_total_mae",
                    "refined_total_mae",
                    "improvement_total",
                    "baseline_roughness",
                    "baseline_metallic",
                    "refined_roughness",
                    "refined_metallic",
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
                    row["baseline_total_mae"],
                    row["refined_total_mae"],
                    row["improvement_total"],
                    wandb.Image(paths["baseline_roughness"]) if paths.get("baseline_roughness") else None,
                    wandb.Image(paths["baseline_metallic"]) if paths.get("baseline_metallic") else None,
                    wandb.Image(paths["refined_roughness"]) if paths.get("refined_roughness") else None,
                    wandb.Image(paths["refined_metallic"]) if paths.get("refined_metallic") else None,
                )
            run.log({"eval/top_cases": preview_table})

        if args.wandb_log_artifacts:
            artifact_paths = [metrics_path, summary_path, output_dir / "artifacts"]
            if report_path is not None:
                artifact_paths.append(report_path)
            log_path_artifact(
                run,
                name=f"{run_name}-eval",
                artifact_type="evaluation",
                paths=artifact_paths,
            )
        run.finish()


if __name__ == "__main__":
    main()
