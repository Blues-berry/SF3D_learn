from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine import (  # noqa: E402
    CanonicalMaterialDataset,
    MaterialRefiner,
    collate_material_samples,
)
from sf3d.material_refine.data_utils import (  # noqa: E402
    summarize_records,
    write_manifest_snapshot,
)
from sf3d.material_refine.experiment import (  # noqa: E402
    flatten_for_logging,
    log_path_artifact,
    make_json_serializable,
    maybe_init_wandb,
    sanitize_log_dict,
    wandb,
)
from sf3d.material_refine.io import tensor_to_pil  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "material_refiner"
DEFAULT_MANIFEST = REPO_ROOT / "output" / "material_refine" / "canonical_manifest_v1.json"


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
        description="Train the SF3D material refinement module with NG-style config, validation, and W&B hooks.",
    )
    parser.add_argument("--config", action="append", type=Path, default=[])
    parser.add_argument("--method-config", action="append", type=Path, default=[])
    parser.add_argument("--data-config", action="append", type=Path, default=[])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--split-strategy", choices=["auto", "manifest", "hash"], default="auto")
    parser.add_argument("--hash-val-ratio", type=float, default=0.1)
    parser.add_argument("--hash-test-ratio", type=float, default=0.1)
    parser.add_argument("--freeze-val-manifest-to", type=Path, default=None)
    parser.add_argument("--reload-manifest-every", type=int, default=1)
    parser.add_argument("--allow-empty-val", type=parse_bool, default=True)
    parser.add_argument("--train-generator-ids", type=str, default=None)
    parser.add_argument("--val-generator-ids", type=str, default=None)
    parser.add_argument("--train-source-names", type=str, default=None)
    parser.add_argument("--val-source-names", type=str, default=None)
    parser.add_argument("--train-supervision-tiers", type=str, default=None)
    parser.add_argument("--val-supervision-tiers", type=str, default=None)
    parser.add_argument("--train-license-buckets", type=str, default=None)
    parser.add_argument("--val-license-buckets", type=str, default=None)
    parser.add_argument("--train-require-prior", type=str, default="any")
    parser.add_argument("--val-require-prior", type=str, default="any")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--adam-amsgrad", type=parse_bool, default=False)
    parser.add_argument("--lr-scheduler", choices=["constant", "plateau"], default="plateau")
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-threshold", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--atlas-size", type=int, default=512)
    parser.add_argument("--buffer-resolution", type=int, default=256)
    parser.add_argument("--cuda-device-index", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--allow-tf32", type=parse_bool, default=True)
    parser.add_argument(
        "--train-balance-mode",
        choices=[
            "none",
            "generator",
            "source",
            "prior",
            "tier",
            "generator_x_prior",
            "source_x_prior",
        ],
        default="source_x_prior",
    )
    parser.add_argument("--train-balance-power", type=float, default=1.0)
    parser.add_argument("--train-samples-per-epoch", type=int, default=0)
    parser.add_argument("--drop-last", type=parse_bool, default=False)
    parser.add_argument("--prior-dropout-prob", type=float, default=0.0)
    parser.add_argument("--prior-dropout-start-prob", type=float, default=None)
    parser.add_argument("--prior-dropout-end-prob", type=float, default=None)
    parser.add_argument("--prior-dropout-warmup-epochs", type=int, default=0)
    parser.add_argument("--refine-weight", type=float, default=1.0)
    parser.add_argument("--coarse-weight", type=float, default=0.35)
    parser.add_argument("--prior-consistency-weight", type=float, default=0.10)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--view-consistency-weight", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--validation-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--checkpointing-steps", type=int, default=0)
    parser.add_argument("--save-only-best-checkpoint", type=parse_bool, default=False)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-preview-samples", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="wandb")
    parser.add_argument("--tracker-project-name", type=str, default="stable-fast-3d-material-refine")
    parser.add_argument("--tracker-run-name", type=str, default=None)
    parser.add_argument("--tracker-group", type=str, default="material-refine")
    parser.add_argument("--tracker-tags", type=str, default="material-refine,sf3d")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-resume-id", type=str, default=None)
    parser.add_argument("--wandb-resume-mode", type=str, default="allow")
    parser.add_argument("--wandb-log-artifacts", type=parse_bool, default=True)
    parser.add_argument("--wandb-watch", type=parse_bool, default=False)
    parser.add_argument("--export-training-curves", type=parse_bool, default=True)
    parser.add_argument("--preflight-audit-records", type=int, default=256)
    parser.add_argument("--target-prior-identity-warning-threshold", type=float, default=0.95)
    parser.add_argument("--fail-on-target-prior-identity", type=parse_bool, default=False)
    parser.add_argument("--preflight-only", action="store_true")
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
    args.train_manifest = args.train_manifest or args.manifest
    args.val_manifest = args.val_manifest or args.manifest
    args.train_generator_ids = parse_csv_list(args.train_generator_ids)
    args.val_generator_ids = parse_csv_list(args.val_generator_ids)
    args.train_source_names = parse_csv_list(args.train_source_names)
    args.val_source_names = parse_csv_list(args.val_source_names)
    args.train_supervision_tiers = parse_csv_list(args.train_supervision_tiers)
    args.val_supervision_tiers = parse_csv_list(args.val_supervision_tiers)
    args.train_license_buckets = parse_csv_list(args.train_license_buckets)
    args.val_license_buckets = parse_csv_list(args.val_license_buckets)
    args.train_require_prior = parse_optional_bool(args.train_require_prior)
    args.val_require_prior = parse_optional_bool(args.val_require_prior)
    args.reload_manifest_every = max(int(args.reload_manifest_every), 1)
    args.grad_accumulation_steps = max(int(args.grad_accumulation_steps), 1)
    args.log_every = max(int(args.log_every), 1)
    args.eval_every = max(int(args.eval_every), 1)
    args.save_every = max(int(args.save_every), 1)
    args.prior_dropout_prob = max(0.0, min(1.0, float(args.prior_dropout_prob)))
    if args.prior_dropout_start_prob is not None:
        args.prior_dropout_start_prob = max(0.0, min(1.0, float(args.prior_dropout_start_prob)))
    if args.prior_dropout_end_prob is not None:
        args.prior_dropout_end_prob = max(0.0, min(1.0, float(args.prior_dropout_end_prob)))
    args.prior_dropout_warmup_epochs = max(int(args.prior_dropout_warmup_epochs), 0)
    return args


def resolve_device(args: argparse.Namespace) -> str:
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return f"cuda:{args.cuda_device_index}"
    return "cpu"


def resolve_amp_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def apply_prior_dropout(batch: dict[str, Any], probability: float) -> int:
    if probability <= 0.0:
        return 0
    prior_confidence = batch["uv_prior_confidence"]
    batch_size = prior_confidence.shape[0]
    has_prior = prior_confidence.flatten(1).amax(dim=1) > 0.0
    drop_mask = (torch.rand(batch_size, device=prior_confidence.device) < probability) & has_prior
    dropped = int(drop_mask.sum().detach().cpu().item())
    if dropped == 0:
        return 0

    view = drop_mask.view(batch_size, 1, 1, 1)
    batch["uv_prior_roughness"] = torch.where(
        view,
        torch.full_like(batch["uv_prior_roughness"], 0.5),
        batch["uv_prior_roughness"],
    )
    batch["uv_prior_metallic"] = torch.where(
        view,
        torch.zeros_like(batch["uv_prior_metallic"]),
        batch["uv_prior_metallic"],
    )
    batch["uv_prior_confidence"] = torch.where(
        view,
        torch.zeros_like(batch["uv_prior_confidence"]),
        batch["uv_prior_confidence"],
    )
    return dropped


def current_prior_dropout_probability(args: argparse.Namespace, epoch: int) -> float:
    if args.prior_dropout_start_prob is None and args.prior_dropout_end_prob is None:
        return args.prior_dropout_prob
    start = args.prior_dropout_prob if args.prior_dropout_start_prob is None else args.prior_dropout_start_prob
    end = start if args.prior_dropout_end_prob is None else args.prior_dropout_end_prob
    if args.prior_dropout_warmup_epochs <= 1:
        return end
    progress = min(max((epoch - 1) / float(args.prior_dropout_warmup_epochs - 1), 0.0), 1.0)
    return start + (end - start) * progress


def total_variation_loss(tensor: torch.Tensor) -> torch.Tensor:
    loss_x = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs().mean()
    loss_y = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs().mean()
    return loss_x + loss_y


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


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    target = torch.cat(
        [batch["uv_target_roughness"], batch["uv_target_metallic"]],
        dim=1,
    )
    confidence = batch["uv_target_confidence"].clamp(0.0, 1.0)
    refined = outputs["refined"]
    coarse = outputs["coarse"]
    baseline = outputs["baseline"]
    prior_confidence = batch["uv_prior_confidence"].clamp(0.0, 1.0)

    refine_l1 = ((refined - target).abs() * confidence).sum() / confidence.sum().clamp_min(1.0)
    coarse_l1 = ((coarse - target).abs() * confidence).sum() / confidence.sum().clamp_min(1.0)
    prior_consistency = (
        (refined - baseline).abs() * (1.0 - confidence) * prior_confidence
    ).mean()
    smoothness = total_variation_loss(refined)

    if batch.get("view_targets") is not None and batch.get("view_uvs") is not None:
        sampled_refined = sample_uv_maps_to_view(refined, batch["view_uvs"])
        view_mask = batch["view_masks"].clamp(0.0, 1.0)
        view_targets = batch["view_targets"]
        view_consistency = (
            (sampled_refined - view_targets).abs() * view_mask
        ).sum() / view_mask.sum().clamp_min(1.0)
    else:
        view_consistency = refined.new_zeros(())

    total = (
        args.refine_weight * refine_l1
        + args.coarse_weight * coarse_l1
        + args.prior_consistency_weight * prior_consistency
        + args.smoothness_weight * smoothness
        + args.view_consistency_weight * view_consistency
    )
    return {
        "total": total,
        "refine_l1": refine_l1.detach(),
        "coarse_l1": coarse_l1.detach(),
        "prior_consistency": prior_consistency.detach(),
        "smoothness": smoothness.detach(),
        "view_consistency": view_consistency.detach(),
    }


def sample_balance_key(record: Any, mode: str) -> str:
    generator_key = str(record.generator_id)
    source_name = str(record.metadata.get("source_name", record.generator_id))
    prior_key = "with_prior" if record.has_material_prior else "without_prior"
    if mode == "generator":
        return generator_key
    if mode == "source":
        return source_name
    if mode == "prior":
        return prior_key
    if mode == "tier":
        return record.supervision_tier
    if mode == "generator_x_prior":
        return f"{generator_key}|{prior_key}"
    if mode == "source_x_prior":
        return f"{source_name}|{prior_key}"
    return "all"


def build_train_sampler(
    dataset: CanonicalMaterialDataset,
    args: argparse.Namespace,
) -> WeightedRandomSampler | None:
    if not dataset.records:
        return None
    use_sampler = args.train_balance_mode != "none" or args.train_samples_per_epoch > 0
    if not use_sampler:
        return None

    if args.train_balance_mode == "none":
        weights = [1.0] * len(dataset.records)
    else:
        group_keys = [sample_balance_key(record, args.train_balance_mode) for record in dataset.records]
        counts = Counter(group_keys)
        power = max(float(args.train_balance_power), 0.0)
        weights = [pow(1.0 / counts[group_key], power) for group_key in group_keys]

    num_samples = args.train_samples_per_epoch if args.train_samples_per_epoch > 0 else len(dataset.records)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def build_loader(
    dataset: CanonicalMaterialDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_material_samples,
    )


def make_dataset(
    manifest_path: Path,
    *,
    split: str,
    args: argparse.Namespace,
    generator_ids: list[str] | None,
    source_names: list[str] | None,
    supervision_tiers: list[str] | None,
    license_buckets: list[str] | None,
    require_prior: bool | None,
    max_records: int | None,
) -> CanonicalMaterialDataset:
    return CanonicalMaterialDataset(
        manifest_path,
        split=split,
        split_strategy=args.split_strategy,
        hash_val_ratio=args.hash_val_ratio,
        hash_test_ratio=args.hash_test_ratio,
        generator_ids=generator_ids,
        source_names=source_names,
        supervision_tiers=supervision_tiers,
        license_buckets=license_buckets,
        require_prior=require_prior,
        max_records=max_records,
        atlas_size=args.atlas_size,
        buffer_resolution=args.buffer_resolution,
    )


def create_or_load_frozen_val_manifest(
    args: argparse.Namespace,
    *,
    base_output_dir: Path,
) -> tuple[Path, str]:
    if args.freeze_val_manifest_to is None:
        return args.val_manifest, args.val_split
    freeze_path = args.freeze_val_manifest_to
    if not freeze_path.is_absolute():
        if freeze_path.parts and freeze_path.parts[0] == "output":
            freeze_path = REPO_ROOT / freeze_path
        else:
            freeze_path = base_output_dir / freeze_path
    if freeze_path.exists():
        return freeze_path, "all"

    val_dataset = make_dataset(
        args.val_manifest,
        split=args.val_split,
        args=args,
        generator_ids=args.val_generator_ids,
        source_names=args.val_source_names,
        supervision_tiers=args.val_supervision_tiers,
        license_buckets=args.val_license_buckets,
        require_prior=args.val_require_prior,
        max_records=args.max_val_records,
    )
    write_manifest_snapshot(
        freeze_path,
        records=val_dataset.records,
        source_manifest=args.val_manifest,
        metadata={
            "frozen_from_split": args.val_split,
            "split_strategy": args.split_strategy,
            "hash_val_ratio": args.hash_val_ratio,
            "hash_test_ratio": args.hash_test_ratio,
        },
    )
    return freeze_path, "all"


def dataset_summary_payload(dataset: CanonicalMaterialDataset) -> dict[str, Any]:
    summary = summarize_records(dataset.records)
    summary["num_batches_at_batch_size"] = math.ceil(len(dataset.records) / 1)
    return summary


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def build_optimizer(
    model: MaterialRefiner,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
        "weight_decay": args.weight_decay,
        "amsgrad": args.adam_amsgrad,
    }
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)


def model_parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "parameters_total": int(total),
        "parameters_trainable": int(trainable),
    }


def cuda_memory_log(device: str) -> dict[str, float]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {}
    torch_device = torch.device(device)
    return {
        "memory/gpu_allocated_gb": torch.cuda.memory_allocated(torch_device) / (1024**3),
        "memory/gpu_reserved_gb": torch.cuda.memory_reserved(torch_device) / (1024**3),
        "memory/gpu_max_allocated_gb": torch.cuda.max_memory_allocated(torch_device) / (1024**3),
    }


def format_metric(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.{digits}f}"


def wandb_auth_hint() -> str:
    if os.environ.get("WANDB_API_KEY"):
        return "WANDB_API_KEY"
    if (Path.home() / ".netrc").exists():
        return "~/.netrc"
    return "missing"


def _resolve_manifest_record_path(
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


def _file_digest(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifest_target_prior_identity(
    manifest_path: Path,
    *,
    max_records: int,
) -> dict[str, Any]:
    if max_records == 0:
        return {"enabled": False}
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text())
    raw_records = payload.get("records") or payload.get("objects") or payload.get("rows") or []
    if not isinstance(raw_records, list):
        return {"enabled": True, "error": "unsupported_manifest_records"}
    records = raw_records if max_records < 0 else raw_records[:max_records]
    checked = 0
    complete_pairs = 0
    same_roughness = 0
    same_metallic = 0
    missing_pairs = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        checked += 1
        paths = {
            key: _resolve_manifest_record_path(manifest_path, payload, record, record.get(key))
            for key in [
                "uv_prior_roughness_path",
                "uv_target_roughness_path",
                "uv_prior_metallic_path",
                "uv_target_metallic_path",
            ]
        }
        digests = {key: _file_digest(path) for key, path in paths.items()}
        if any(value is None for value in digests.values()):
            missing_pairs += 1
            continue
        complete_pairs += 1
        same_roughness += int(
            digests["uv_prior_roughness_path"] == digests["uv_target_roughness_path"]
        )
        same_metallic += int(
            digests["uv_prior_metallic_path"] == digests["uv_target_metallic_path"]
        )
    paired_identity = 0.0
    if complete_pairs:
        paired_identity = min(same_roughness, same_metallic) / float(complete_pairs)
    return {
        "enabled": True,
        "manifest": str(manifest_path),
        "records_total": len(raw_records),
        "records_checked": checked,
        "complete_pairs": complete_pairs,
        "missing_pairs": missing_pairs,
        "same_roughness": same_roughness,
        "same_metallic": same_metallic,
        "paired_identity_rate": paired_identity,
    }


def run_preflight_checks(args: argparse.Namespace, device: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_paths = {
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
    }
    if args.manifest:
        manifest_paths["manifest"] = args.manifest
    for label, path in manifest_paths.items():
        path = Path(path)
        if not path.exists():
            errors.append(f"{label}_missing:{path}")

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            errors.append("cuda_requested_but_unavailable")
        elif args.cuda_device_index >= torch.cuda.device_count():
            errors.append(
                f"cuda_device_index_out_of_range:{args.cuda_device_index}/count={torch.cuda.device_count()}"
            )
    if args.batch_size <= 0:
        errors.append(f"invalid_batch_size:{args.batch_size}")
    if args.learning_rate <= 0:
        errors.append(f"invalid_learning_rate:{args.learning_rate}")
    if args.report_to == "wandb" and args.wandb_mode == "online" and wandb_auth_hint() == "missing":
        warnings.append("wandb_online_requested_without_visible_login")
    if args.prior_dropout_prob > 0 and args.train_require_prior is False:
        warnings.append("prior_dropout_enabled_but_train_filter_requires_no_prior")
    if args.save_only_best_checkpoint and args.validation_steps > 0:
        warnings.append("save_only_best_checkpoint_with_step_validation_keeps_only_improved_step_checkpoints")
    manifest_audit = None
    if Path(args.train_manifest).exists():
        try:
            manifest_audit = audit_manifest_target_prior_identity(
                Path(args.train_manifest),
                max_records=int(args.preflight_audit_records),
            )
            identity_rate = float(manifest_audit.get("paired_identity_rate", 0.0))
            if (
                manifest_audit.get("complete_pairs", 0) > 0
                and identity_rate >= float(args.target_prior_identity_warning_threshold)
            ):
                message = (
                    "target_prior_identity_high:"
                    f"{identity_rate:.3f};baseline_metrics_may_be_trivial"
                )
                warnings.append(message)
                if args.fail_on_target_prior_identity:
                    errors.append(message)
        except Exception as exc:
            warnings.append(f"manifest_identity_audit_failed:{type(exc).__name__}:{exc}")

    payload = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_index": args.cuda_device_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wandb_auth": wandb_auth_hint(),
        "wandb_mode": args.wandb_mode,
        "output_dir": str(args.output_dir),
        "errors": errors,
        "warnings": warnings,
        "manifest_identity_audit": manifest_audit,
    }
    print(json.dumps({"preflight": payload}, ensure_ascii=False))
    status = "ok" if not errors else "failed"
    warning_suffix = f" warnings={len(warnings)}" if warnings else ""
    print(
        "[preflight] "
        f"status={status} device={device} cuda_index={args.cuda_device_index} "
        f"wandb_mode={args.wandb_mode} wandb_auth={payload['wandb_auth']}{warning_suffix}"
    )
    for warning in warnings:
        print(f"[preflight:warning] {warning}")
    if manifest_audit is not None:
        print(
            "[preflight:audit] "
            f"checked={manifest_audit.get('records_checked')} "
            f"complete_pairs={manifest_audit.get('complete_pairs')} "
            f"target_prior_identity={format_metric(manifest_audit.get('paired_identity_rate'), 4)}"
        )
    if errors:
        for error in errors:
            print(f"[preflight:error] {error}")
        raise RuntimeError("material_refine_preflight_failed")
    return payload


def print_train_interval(payload: dict[str, Any], device: str) -> None:
    memory = payload.get("memory/gpu_max_allocated_gb")
    grad = payload.get("train/grad_norm")
    print(
        "[train] "
        f"epoch={payload.get('epoch')} step={payload.get('optimizer_step')} "
        f"global_batch={payload.get('global_batch_step')} lr={format_metric(payload.get('lr'), 8)} "
        f"loss={format_metric(payload.get('train/total'))} refine={format_metric(payload.get('train/refine_l1'))} "
        f"coarse={format_metric(payload.get('train/coarse_l1'))} grad={format_metric(grad)} "
        f"samples_s={format_metric(payload.get('train/samples_per_second'), 3)} "
        f"max_mem_gb={format_metric(memory, 3) if memory is not None else 'n/a'} device={device}"
    )


def print_epoch_summary(
    *,
    args: argparse.Namespace,
    epoch_payload: dict[str, Any],
    best_val_metric: float | None,
    best_epoch: int | None,
    epoch_improved: bool,
    checkpoint_path: Path | None,
) -> None:
    val_payload = epoch_payload.get("val") or {}
    val_mae = (val_payload.get("uv_mae") or {}).get("total")
    train_payload = epoch_payload.get("train") or {}
    checkpoint_label = str(checkpoint_path) if checkpoint_path is not None else "not_saved"
    print(
        "[epoch] "
        f"{epoch_payload.get('epoch')}/{args.epochs} step={epoch_payload.get('optimizer_step')} "
        f"train_total={format_metric(train_payload.get('total'))} "
        f"val_uv_total={format_metric(val_mae)} "
        f"best={format_metric(best_val_metric)} best_epoch={best_epoch} "
        f"improved={epoch_improved} checkpoint={checkpoint_label}"
    )


def save_training_visualizations(history: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    if not history:
        return {}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(json.dumps({"training_visualization_warning": repr(exc)}))
        return {}

    epochs = [int(item.get("epoch", index + 1)) for index, item in enumerate(history)]
    train_total = [float((item.get("train") or {}).get("total", np.nan)) for item in history]
    val_total = [
        float(((item.get("val") or {}).get("uv_mae") or {}).get("total", np.nan))
        for item in history
    ]
    val_roughness = [
        float(((item.get("val") or {}).get("uv_mae") or {}).get("roughness", np.nan))
        for item in history
    ]
    val_metallic = [
        float(((item.get("val") or {}).get("uv_mae") or {}).get("metallic", np.nan))
        for item in history
    ]
    samples_per_second = [
        float((item.get("train") or {}).get("samples_per_second", np.nan))
        for item in history
    ]

    figure_path = output_dir / "training_curves.png"
    html_path = output_dir / "training_summary.html"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Material Refiner Training Curves")
    axes[0, 0].plot(epochs, train_total, marker="o", label="train total")
    axes[0, 0].plot(epochs, val_total, marker="o", label="val uv total")
    axes[0, 0].set_title("Total Loss / UV MAE")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(epochs, val_roughness, marker="o", label="roughness")
    axes[0, 1].plot(epochs, val_metallic, marker="o", label="metallic")
    axes[0, 1].set_title("Validation UV MAE By Channel")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(epochs, samples_per_second, marker="o")
    axes[1, 0].set_title("Training Throughput")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("samples/sec")
    axes[1, 0].grid(alpha=0.25)

    best_index = int(np.nanargmin(val_total)) if not np.isnan(val_total).all() else len(epochs) - 1
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.92,
        "\n".join(
            [
                f"epochs: {len(history)}",
                f"best epoch: {epochs[best_index]}",
                f"best val total: {format_metric(val_total[best_index])}",
                f"last train total: {format_metric(train_total[-1])}",
                f"last val total: {format_metric(val_total[-1])}",
            ]
        ),
        va="top",
        family="monospace",
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    latest = history[-1]
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Material Refiner Training Summary</title>",
                "<style>body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1200px;margin:auto}.card{background:#18202b;border-radius:18px;padding:18px;margin:16px 0}img{max-width:100%;border-radius:14px;background:white}code{color:#b9e6ff}</style>",
                "</head><body><div class='wrap'>",
                "<h1>Material Refiner Training Summary</h1>",
                f"<div class='card'><img src='{figure_path.name}' alt='training curves'></div>",
                "<div class='card'>",
                f"<div>Last epoch: <code>{latest.get('epoch')}</code></div>",
                f"<div>Last optimizer step: <code>{latest.get('optimizer_step')}</code></div>",
                f"<div>Last train total: <code>{format_metric((latest.get('train') or {}).get('total'))}</code></div>",
                f"<div>Last val total: <code>{format_metric(((latest.get('val') or {}).get('uv_mae') or {}).get('total'))}</code></div>",
                "</div></div></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "training_curves": str(figure_path.resolve()),
        "training_summary": str(html_path.resolve()),
    }


def maybe_apply_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    base_learning_rate: float,
    optimizer_step: int,
    warmup_steps: int,
) -> None:
    if warmup_steps <= 0:
        return
    if optimizer_step <= warmup_steps:
        warmup_lr = base_learning_rate * (optimizer_step / float(warmup_steps))
        set_optimizer_lr(optimizer, warmup_lr)
    elif optimizer_step == warmup_steps + 1:
        set_optimizer_lr(optimizer, base_learning_rate)


def grayscale_rgb_image(tensor: torch.Tensor) -> Image.Image:
    return tensor_to_pil(tensor.detach().cpu(), grayscale=True).convert("RGB")


def labeled_panel(images: list[Image.Image], labels: list[str]) -> Image.Image:
    width, height = images[0].size
    header_height = 22
    panel = Image.new("RGB", size=(width * len(images), height + header_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)
    for index, (image, label) in enumerate(zip(images, labels)):
        x0 = index * width
        panel.paste(image, box=(x0, header_height))
        draw.text((x0 + 4, 4), label, fill=(0, 0, 0))
    return panel


def save_validation_preview(
    output_dir: Path,
    *,
    validation_label: str,
    object_id: str,
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target_roughness: torch.Tensor,
    target_metallic: torch.Tensor,
    confidence: torch.Tensor,
) -> Path:
    preview_dir = output_dir / "validation_previews" / validation_label
    preview_dir.mkdir(parents=True, exist_ok=True)
    rough_panel = labeled_panel(
        [
            grayscale_rgb_image(baseline[0:1]),
            grayscale_rgb_image(target_roughness),
            grayscale_rgb_image(refined[0:1]),
        ],
        ["baseline_r", "target_r", "refined_r"],
    )
    metal_panel = labeled_panel(
        [
            grayscale_rgb_image(baseline[1:2]),
            grayscale_rgb_image(target_metallic),
            grayscale_rgb_image(refined[1:2]),
            grayscale_rgb_image(confidence),
        ],
        ["baseline_m", "target_m", "refined_m", "confidence"],
    )
    canvas = Image.new(
        "RGB",
        size=(max(rough_panel.width, metal_panel.width), rough_panel.height + metal_panel.height),
        color=(255, 255, 255),
    )
    canvas.paste(rough_panel, (0, 0))
    canvas.paste(metal_panel, (0, rough_panel.height))
    output_path = preview_dir / f"{object_id}.png"
    canvas.save(output_path)
    return output_path


def update_group_metric_store(
    store: dict[str, dict[str, float]],
    *,
    key: str,
    total_mae: float,
    roughness_mae: float,
    metallic_mae: float,
) -> None:
    bucket = store.setdefault(
        key,
        {"count": 0.0, "total_mae": 0.0, "roughness_mae": 0.0, "metallic_mae": 0.0},
    )
    bucket["count"] += 1.0
    bucket["total_mae"] += total_mae
    bucket["roughness_mae"] += roughness_mae
    bucket["metallic_mae"] += metallic_mae


def finalize_group_metrics(store: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    finalized = {}
    for key, bucket in store.items():
        count = max(bucket["count"], 1.0)
        finalized[key] = {
            "count": int(bucket["count"]),
            "total_mae": bucket["total_mae"] / count,
            "roughness_mae": bucket["roughness_mae"] / count,
            "metallic_mae": bucket["metallic_mae"] / count,
        }
    return finalized


def evaluate(
    model: MaterialRefiner,
    loader: DataLoader,
    *,
    device: str,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
    output_dir: Path,
    epoch: int,
    validation_label: str,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {
        "total": 0.0,
        "refine_l1": 0.0,
        "coarse_l1": 0.0,
        "prior_consistency": 0.0,
        "smoothness": 0.0,
        "view_consistency": 0.0,
    }
    uv_mae = {"roughness": 0.0, "metallic": 0.0, "total": 0.0, "count": 0.0}
    group_store: dict[str, dict[str, float]] = {}
    steps = 0
    preview_paths: list[Path] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if device.startswith("cuda")
                else nullcontext()
            ):
                outputs = model(batch)
                losses = compute_losses(outputs, batch, args)
            steps += 1
            for key in totals:
                totals[key] += float(losses[key].detach().cpu().item())

            target = torch.cat(
                [batch["uv_target_roughness"], batch["uv_target_metallic"]],
                dim=1,
            )
            confidence = batch["uv_target_confidence"].clamp(0.0, 1.0)
            refined = outputs["refined"]
            baseline = outputs["baseline"]

            per_sample_weight = confidence.flatten(1).sum(dim=1).clamp_min(1.0)
            roughness_mae = ((refined[:, 0:1] - target[:, 0:1]).abs() * confidence).flatten(1).sum(dim=1) / per_sample_weight
            metallic_mae = ((refined[:, 1:2] - target[:, 1:2]).abs() * confidence).flatten(1).sum(dim=1) / per_sample_weight
            total_mae = roughness_mae + metallic_mae

            uv_mae["roughness"] += float(roughness_mae.sum().item())
            uv_mae["metallic"] += float(metallic_mae.sum().item())
            uv_mae["total"] += float(total_mae.sum().item())
            uv_mae["count"] += float(refined.shape[0])

            for item_index, object_id in enumerate(batch["object_id"]):
                generator_id = str(batch["generator_id"][item_index])
                source_name = str(batch["source_name"][item_index])
                prior_name = "with_prior" if bool(batch["has_material_prior"][item_index]) else "without_prior"
                supervision_tier = str(batch["supervision_tier"][item_index])
                metric_kwargs = {
                    "total_mae": float(total_mae[item_index].item()),
                    "roughness_mae": float(roughness_mae[item_index].item()),
                    "metallic_mae": float(metallic_mae[item_index].item()),
                }
                update_group_metric_store(group_store, key=f"generator/{generator_id}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"source/{source_name}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"prior/{prior_name}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"tier/{supervision_tier}", **metric_kwargs)

                if len(preview_paths) < args.val_preview_samples:
                    preview_paths.append(
                        save_validation_preview(
                            output_dir,
                            validation_label=validation_label,
                            object_id=object_id,
                            baseline=baseline[item_index].detach().cpu(),
                            refined=refined[item_index].detach().cpu(),
                            target_roughness=batch["uv_target_roughness"][item_index].detach().cpu(),
                            target_metallic=batch["uv_target_metallic"][item_index].detach().cpu(),
                            confidence=batch["uv_target_confidence"][item_index].detach().cpu(),
                        )
                    )

    mean_losses = {key: value / max(steps, 1) for key, value in totals.items()}
    mean_uv_mae = {
        "roughness": uv_mae["roughness"] / max(uv_mae["count"], 1.0),
        "metallic": uv_mae["metallic"] / max(uv_mae["count"], 1.0),
        "total": uv_mae["total"] / max(uv_mae["count"], 1.0),
        "count": int(uv_mae["count"]),
    }
    return {
        "loss": mean_losses,
        "uv_mae": mean_uv_mae,
        "group_metrics": finalize_group_metrics(group_store),
        "preview_paths": [str(path.resolve()) for path in preview_paths],
    }


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_serializable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_checkpoint(
    output_dir: Path,
    *,
    model: MaterialRefiner,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scaler: torch.amp.GradScaler,
    epoch: int,
    optimizer_step: int,
    global_batch_step: int,
    best_val_metric: float | None,
    best_epoch: int | None,
    args: argparse.Namespace,
    metrics: dict[str, Any] | None,
    checkpoint_label: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_label = checkpoint_label or f"epoch_{epoch:03d}"
    checkpoint_path = output_dir / f"{checkpoint_label}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "global_batch_step": global_batch_step,
            "best_val_metric": best_val_metric,
            "best_epoch": best_epoch,
            "model_cfg": {key: value for key, value in model.cfg.items()},
            "atlas_size": args.atlas_size,
            "buffer_resolution": args.buffer_resolution,
            "train_args": vars(args),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    latest_path = output_dir / "latest.pt"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(checkpoint_path.name)
    return checkpoint_path


def update_best_symlink(output_dir: Path, checkpoint_path: Path) -> Path:
    best_path = output_dir / "best.pt"
    if best_path.exists() or best_path.is_symlink():
        best_path.unlink()
    best_path.symlink_to(checkpoint_path.name)
    return best_path


def prune_old_checkpoints(output_dir: Path, *, keep_last: int) -> None:
    if keep_last <= 0:
        return
    best_target = None
    best_path = output_dir / "best.pt"
    if best_path.is_symlink():
        best_target = best_path.resolve()
    latest_target = None
    latest_path = output_dir / "latest.pt"
    if latest_path.is_symlink():
        latest_target = latest_path.resolve()
    checkpoints = sorted([*output_dir.glob("epoch_*.pt"), *output_dir.glob("step_*.pt")])
    to_keep = set(checkpoints[-keep_last:])
    if best_target is not None:
        to_keep.add(best_target)
    if latest_target is not None:
        to_keep.add(latest_target)
    for checkpoint_path in checkpoints:
        if checkpoint_path not in to_keep:
            checkpoint_path.unlink(missing_ok=True)


def reload_data_if_needed(
    epoch: int,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    frozen_val_manifest: Path | None,
    val_manifest_path: Path,
    val_split: str,
) -> tuple[CanonicalMaterialDataset, DataLoader, CanonicalMaterialDataset | None, DataLoader | None, dict[str, Any]]:
    train_dataset = make_dataset(
        args.train_manifest,
        split=args.train_split,
        args=args,
        generator_ids=args.train_generator_ids,
        source_names=args.train_source_names,
        supervision_tiers=args.train_supervision_tiers,
        license_buckets=args.train_license_buckets,
        require_prior=args.train_require_prior,
        max_records=args.max_train_records,
    )
    train_sampler = build_train_sampler(train_dataset, args)
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=args.drop_last,
    )

    val_dataset = None
    val_loader = None
    try:
        val_dataset = make_dataset(
            val_manifest_path,
            split=val_split,
            args=args,
            generator_ids=args.val_generator_ids if frozen_val_manifest is None else None,
            source_names=args.val_source_names if frozen_val_manifest is None else None,
            supervision_tiers=args.val_supervision_tiers if frozen_val_manifest is None else None,
            license_buckets=args.val_license_buckets if frozen_val_manifest is None else None,
            require_prior=args.val_require_prior if frozen_val_manifest is None else None,
            max_records=args.max_val_records if frozen_val_manifest is None else None,
        )
        val_loader = build_loader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
    except RuntimeError:
        if not args.allow_empty_val:
            raise

    data_state = {
        "epoch": epoch,
        "train": summarize_records(train_dataset.records),
        "val": summarize_records(val_dataset.records) if val_dataset is not None else {"records": 0},
        "train_batches": len(train_loader),
        "val_batches": len(val_loader) if val_loader is not None else 0,
        "train_manifest": str(args.train_manifest.resolve()),
        "val_manifest": str(val_manifest_path.resolve()),
        "val_split": val_split,
    }
    save_json(output_dir / "dataset_state" / f"epoch_{epoch:03d}.json", data_state)
    return train_dataset, train_loader, val_dataset, val_loader, data_state


def run_validation_cycle(
    *,
    model: MaterialRefiner,
    val_loader: DataLoader,
    device: str,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
    output_dir: Path,
    epoch: int,
    optimizer_step: int,
    run: Any | None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    learning_rate: float,
    train_total: float | None,
    best_val_metric: float | None,
    best_epoch: int | None,
    validation_label: str,
) -> tuple[dict[str, Any], float, int, bool]:
    val_payload = evaluate(
        model,
        val_loader,
        device=device,
        amp_dtype=amp_dtype,
        args=args,
        output_dir=output_dir,
        epoch=epoch,
        validation_label=validation_label,
    )
    save_json(output_dir / "validation" / f"{validation_label}.json", val_payload)

    monitored_metric = float(val_payload["uv_mae"]["total"])
    if scheduler is not None and optimizer_step > args.warmup_steps:
        scheduler.step(monitored_metric)

    improved = best_val_metric is None or monitored_metric < (
        best_val_metric - args.early_stopping_min_delta
    )
    if improved:
        best_val_metric = monitored_metric
        best_epoch = epoch

    print(
        "[val] "
        f"label={validation_label} epoch={epoch} step={optimizer_step} "
        f"uv_total={format_metric(val_payload['uv_mae']['total'])} "
        f"roughness={format_metric(val_payload['uv_mae']['roughness'])} "
        f"metallic={format_metric(val_payload['uv_mae']['metallic'])} "
        f"best={format_metric(best_val_metric)} best_epoch={best_epoch} improved={improved}"
    )

    if run is not None:
        preview_images = []
        if wandb is not None:
            preview_images = [
                wandb.Image(path, caption=Path(path).stem)
                for path in val_payload.get("preview_paths", [])
            ]
        log_payload = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "lr": learning_rate,
            "best/val_uv_total_mae": best_val_metric,
            "best/epoch": best_epoch,
            "val/uv_total_mae": val_payload["uv_mae"]["total"],
            "val/uv_roughness_mae": val_payload["uv_mae"]["roughness"],
            "val/uv_metallic_mae": val_payload["uv_mae"]["metallic"],
        }
        if train_total is not None:
            log_payload["train/epoch_total"] = train_total
        log_payload.update(flatten_for_logging(val_payload["loss"], prefix="val/loss"))
        log_payload.update(flatten_for_logging(val_payload["group_metrics"], prefix="val/groups"))
        sanitized_logs, skipped_logs = sanitize_log_dict(log_payload)
        if skipped_logs:
            print(json.dumps({"validation_label": validation_label, "skipped_val_logs": skipped_logs}))
        if sanitized_logs:
            run.log(sanitized_logs, step=optimizer_step)
        if preview_images:
            run.log({"val/previews": preview_images}, step=optimizer_step)

    return val_payload, float(best_val_metric), int(best_epoch), improved


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None and args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            latest_path = output_dir / "latest.pt"
            if latest_path.exists():
                args.resume = latest_path
        else:
            args.resume = Path(args.resume_from_checkpoint)
    save_json(output_dir / "train_args.json", vars(args))

    set_seed(args.seed)
    device = resolve_device(args)
    amp_dtype = resolve_amp_dtype(args)
    preflight_payload = run_preflight_checks(args, device)
    if args.preflight_only:
        return
    torch.set_float32_matmul_precision(args.matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    use_wandb = args.report_to == "wandb"
    run_name = args.tracker_run_name or output_dir.name

    frozen_val_manifest = None
    val_manifest_path, val_split = create_or_load_frozen_val_manifest(args, base_output_dir=output_dir)
    if args.freeze_val_manifest_to is not None:
        frozen_val_manifest = val_manifest_path

    tracker_config = make_json_serializable(dict(vars(args)))
    tracker_config["device"] = device
    tracker_config["preflight"] = preflight_payload
    tracker_config["resolved_train_manifest"] = str(args.train_manifest.resolve())
    tracker_config["resolved_val_manifest"] = str(val_manifest_path.resolve())
    run = maybe_init_wandb(
        enabled=use_wandb,
        project=args.tracker_project_name,
        job_type="train",
        config=tracker_config,
        mode=args.wandb_mode,
        name=run_name,
        group=args.tracker_group,
        tags=args.tracker_tags,
        run_id=args.wandb_resume_id,
        resume=args.wandb_resume_mode,
        dir_path=args.wandb_dir,
    )

    model = MaterialRefiner()
    model.to(device)
    model_info = model_parameter_summary(model)
    runtime_info = {
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
        "amp_dtype": args.amp_dtype,
        "matmul_precision": args.matmul_precision,
        "allow_tf32": bool(args.allow_tf32),
        "optimizer": args.optimizer,
        **model_info,
    }
    print(json.dumps({"runtime": runtime_info}, ensure_ascii=False))
    if run is not None:
        run.config.update({"runtime": make_json_serializable(runtime_info)}, allow_val_change=True)

    optimizer = build_optimizer(model, args)
    if args.warmup_steps > 0:
        set_optimizer_lr(optimizer, 0.0)

    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=args.plateau_threshold,
            min_lr=args.min_learning_rate,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    history_path = output_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    state_path = output_dir / "train_state.json"
    train_state = json.loads(state_path.read_text()) if state_path.exists() else {}
    start_epoch = 1
    optimizer_step = int(train_state.get("optimizer_step", 0))
    global_batch_step = int(train_state.get("global_batch_step", 0))
    best_val_metric = train_state.get("best_val_metric")
    best_epoch = train_state.get("best_epoch")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.get("scheduler")
        if scheduler is not None and scheduler_state:
            scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.get("scaler")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        optimizer_step = int(checkpoint.get("optimizer_step", optimizer_step))
        global_batch_step = int(checkpoint.get("global_batch_step", global_batch_step))
        best_val_metric = checkpoint.get("best_val_metric", best_val_metric)
        best_epoch = checkpoint.get("best_epoch", best_epoch)
        print(
            json.dumps(
                {
                    "resume_checkpoint": str(args.resume.resolve()),
                    "resume_epoch": int(checkpoint.get("epoch", 0)),
                    "next_epoch": start_epoch,
                    "optimizer_step": optimizer_step,
                }
            )
        )

    train_dataset: CanonicalMaterialDataset | None = None
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    stale_epochs = 0
    stop_training = False

    if run is not None and args.wandb_watch and wandb is not None:
        wandb.watch(model, log="gradients", log_freq=max(args.log_every, 10))

    for epoch in range(start_epoch, args.epochs + 1):
        if train_loader is None or (epoch - start_epoch) % args.reload_manifest_every == 0:
            (
                train_dataset,
                train_loader,
                _val_dataset,
                val_loader,
                data_state,
            ) = reload_data_if_needed(
                epoch,
                args=args,
                output_dir=output_dir,
                frozen_val_manifest=frozen_val_manifest,
                val_manifest_path=val_manifest_path,
                val_split=val_split,
            )
            data_state_logs, skipped = sanitize_log_dict(
                flatten_for_logging(data_state, prefix="dataset")
            )
            if skipped:
                print(json.dumps({"epoch": epoch, "dataset_log_skipped": skipped}))
            if run is not None and data_state_logs:
                run.log(data_state_logs, step=optimizer_step)

        assert train_dataset is not None and train_loader is not None

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = defaultdict(float)
        interval = defaultdict(float)
        interval_steps = 0
        interval_examples = 0
        epoch_examples = 0
        batch_steps = 0
        prior_dropout_probability = current_prior_dropout_probability(args, epoch)
        epoch_start_time = time.perf_counter()
        interval_start_time = epoch_start_time
        last_grad_norm: float | None = None

        for batch_index, batch in enumerate(train_loader, start=1):
            global_batch_step += 1
            batch = move_batch_to_device(batch, device)
            prior_dropout_samples = apply_prior_dropout(batch, prior_dropout_probability)
            batch_examples = int(batch["uv_albedo"].shape[0])
            with (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if device.startswith("cuda")
                else nullcontext()
            ):
                outputs = model(batch)
                losses = compute_losses(outputs, batch, args)
                scaled_total = losses["total"] / args.grad_accumulation_steps

            scaler.scale(scaled_total).backward()
            batch_steps += 1
            interval_steps += 1
            interval_examples += batch_examples
            epoch_examples += batch_examples
            for key, value in losses.items():
                running[key] += float(value.detach().cpu().item())
                interval[key] += float(value.detach().cpu().item())
            running["prior_dropout_samples"] += float(prior_dropout_samples)
            interval["prior_dropout_samples"] += float(prior_dropout_samples)

            should_step = (
                batch_index % args.grad_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if not should_step:
                continue

            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                last_grad_norm = float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            maybe_apply_warmup(
                optimizer,
                base_learning_rate=args.learning_rate,
                optimizer_step=optimizer_step,
                warmup_steps=args.warmup_steps,
            )

            if optimizer_step % args.log_every == 0:
                now = time.perf_counter()
                interval_seconds = max(now - interval_start_time, 1e-9)
                interval_logs = {
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "global_batch_step": global_batch_step,
                    "lr": current_lr(optimizer),
                    "train/prior_dropout_probability": prior_dropout_probability,
                    "train/batches_per_log_window": interval_steps,
                    "train/samples_per_second": interval_examples / interval_seconds,
                    "train/seconds_per_batch": interval_seconds / max(interval_steps, 1),
                }
                if last_grad_norm is not None:
                    interval_logs["train/grad_norm"] = last_grad_norm
                for key, value in interval.items():
                    interval_logs[f"train/{key}"] = value / max(interval_steps, 1)
                interval_logs.update(cuda_memory_log(device))
                sanitized_logs, skipped_logs = sanitize_log_dict(interval_logs)
                if skipped_logs:
                    print(json.dumps({"optimizer_step": optimizer_step, "skipped_train_logs": skipped_logs}))
                if sanitized_logs:
                    print_train_interval(sanitized_logs, device)
                    print(json.dumps(sanitized_logs))
                    if run is not None:
                        run.log(sanitized_logs, step=optimizer_step)
                interval.clear()
                interval_steps = 0
                interval_examples = 0
                interval_start_time = now

            step_val_payload = None
            step_improved = False
            if (
                val_loader is not None
                and args.validation_steps > 0
                and optimizer_step % args.validation_steps == 0
            ):
                (
                    step_val_payload,
                    best_val_metric,
                    best_epoch,
                    step_improved,
                ) = run_validation_cycle(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    amp_dtype=amp_dtype,
                    args=args,
                    output_dir=output_dir,
                    epoch=epoch,
                    optimizer_step=optimizer_step,
                    run=run,
                    scheduler=scheduler,
                    learning_rate=current_lr(optimizer),
                    train_total=None,
                    best_val_metric=best_val_metric,
                    best_epoch=best_epoch,
                    validation_label=f"step_{optimizer_step:06d}",
                )
                stale_epochs = 0 if step_improved else stale_epochs + 1

            should_save_step_checkpoint = (
                args.checkpointing_steps > 0
                and optimizer_step % args.checkpointing_steps == 0
                and not args.save_only_best_checkpoint
            )
            should_save_best_step_checkpoint = step_improved
            if should_save_step_checkpoint or should_save_best_step_checkpoint:
                step_checkpoint = save_checkpoint(
                    output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    optimizer_step=optimizer_step,
                    global_batch_step=global_batch_step,
                    best_val_metric=best_val_metric,
                    best_epoch=best_epoch,
                    args=args,
                    metrics={
                        "train_total": None,
                        "val_uv_total_mae": None
                        if step_val_payload is None
                        else step_val_payload["uv_mae"]["total"],
                    },
                    checkpoint_label=f"step_{optimizer_step:06d}",
                )
                if step_improved:
                    update_best_symlink(output_dir, step_checkpoint)
                prune_old_checkpoints(output_dir, keep_last=args.keep_last_checkpoints)

            if args.max_train_steps > 0 and optimizer_step >= args.max_train_steps:
                stop_training = True
                break

        epoch_seconds = max(time.perf_counter() - epoch_start_time, 1e-9)
        train_metrics = {key: value / max(batch_steps, 1) for key, value in running.items()}
        train_metrics["prior_dropout_probability"] = prior_dropout_probability
        train_metrics["samples_per_second"] = epoch_examples / epoch_seconds
        train_metrics["epoch_seconds"] = epoch_seconds
        epoch_payload: dict[str, Any] = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "global_batch_step": global_batch_step,
            "lr": current_lr(optimizer),
            "train": train_metrics,
            "dataset": summarize_records(train_dataset.records),
        }

        val_payload = None
        epoch_improved = False
        if val_loader is not None and args.validation_steps <= 0 and epoch % args.eval_every == 0:
            (
                val_payload,
                best_val_metric,
                best_epoch,
                epoch_improved,
            ) = run_validation_cycle(
                model=model,
                val_loader=val_loader,
                device=device,
                amp_dtype=amp_dtype,
                args=args,
                output_dir=output_dir,
                epoch=epoch,
                optimizer_step=optimizer_step,
                run=run,
                scheduler=scheduler,
                learning_rate=current_lr(optimizer),
                train_total=train_metrics["total"],
                best_val_metric=best_val_metric,
                best_epoch=best_epoch,
                validation_label=f"epoch_{epoch:03d}",
            )
            epoch_payload["val"] = val_payload
            if epoch_improved:
                stale_epochs = 0
            else:
                stale_epochs += 1
        elif scheduler is not None and optimizer_step > args.warmup_steps:
            scheduler.step(train_metrics["total"])

        history.append(epoch_payload)
        save_json(history_path, history)
        visualization_paths = (
            save_training_visualizations(history, output_dir)
            if args.export_training_curves
            else {}
        )

        checkpoint_metrics = {
            "train_total": train_metrics["total"],
            "val_uv_total_mae": None if val_payload is None else val_payload["uv_mae"]["total"],
        }
        checkpoint_path = None
        should_save_epoch = epoch % args.save_every == 0 and not args.save_only_best_checkpoint
        should_save_best_epoch = epoch_improved and args.validation_steps <= 0
        if should_save_epoch or should_save_best_epoch:
            checkpoint_path = save_checkpoint(
                output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                optimizer_step=optimizer_step,
                global_batch_step=global_batch_step,
                best_val_metric=best_val_metric,
                best_epoch=best_epoch,
                args=args,
                metrics=checkpoint_metrics,
            )
            if should_save_best_epoch and checkpoint_path is not None:
                update_best_symlink(output_dir, checkpoint_path)
            prune_old_checkpoints(output_dir, keep_last=args.keep_last_checkpoints)

        train_state = {
            "last_epoch": epoch,
            "optimizer_step": optimizer_step,
            "global_batch_step": global_batch_step,
            "best_val_metric": best_val_metric,
            "best_epoch": best_epoch,
            "current_lr": current_lr(optimizer),
            "train_manifest": str(args.train_manifest.resolve()),
            "val_manifest": str(val_manifest_path.resolve()),
        }
        save_json(state_path, train_state)
        print_epoch_summary(
            args=args,
            epoch_payload=epoch_payload,
            best_val_metric=best_val_metric,
            best_epoch=best_epoch,
            epoch_improved=epoch_improved,
            checkpoint_path=checkpoint_path,
        )
        print(json.dumps(epoch_payload))

        if checkpoint_path is not None and run is not None and args.wandb_log_artifacts:
            artifact_paths = [
                checkpoint_path,
                history_path,
                state_path,
                output_dir / "train_args.json",
            ]
            if val_payload is not None:
                artifact_paths.append(output_dir / "validation" / f"epoch_{epoch:03d}.json")
            for path_value in visualization_paths.values():
                artifact_paths.append(Path(path_value))
            log_path_artifact(
                run,
                name=f"{run_name}-epoch-{epoch:03d}",
                artifact_type="checkpoint",
                paths=artifact_paths,
            )

        if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
            print(
                json.dumps(
                    {
                        "early_stop": True,
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_val_metric": best_val_metric,
                    }
                )
            )
            break
        if stop_training:
            print(json.dumps({"max_train_steps_reached": args.max_train_steps, "optimizer_step": optimizer_step}))
            break

    if run is not None:
        final_visualization_paths = (
            save_training_visualizations(history, output_dir)
            if args.export_training_curves
            else {}
        )
        final_artifacts = [
            history_path,
            state_path,
            output_dir / "latest.pt",
        ]
        for path_value in final_visualization_paths.values():
            final_artifacts.append(Path(path_value))
        best_path = output_dir / "best.pt"
        if best_path.exists():
            final_artifacts.append(best_path)
        log_path_artifact(
            run,
            name=f"{run_name}-final",
            artifact_type="training-run",
            paths=final_artifacts,
        )
        run.finish()
    print(
        "[final] "
        f"output_dir={output_dir} latest={output_dir / 'latest.pt'} "
        f"best={output_dir / 'best.pt'} history={history_path}"
    )


if __name__ == "__main__":
    main()
