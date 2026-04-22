from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

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
from sf3d.material_refine.manifest_quality import (  # noqa: E402
    DEFAULT_MAX_TARGET_PRIOR_IDENTITY_RATE_FOR_PAPER,
    DEFAULT_MIN_NONTRIVIAL_TARGET_COUNT_FOR_PAPER,
    audit_manifest,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "material_refiner"
DEFAULT_MANIFEST = REPO_ROOT / "output" / "material_refine" / "canonical_manifest_v1.json"
METALLIC_THRESHOLD = 0.5


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


def parse_weight_map(value: str | None) -> dict[str, float]:
    if value is None:
        return {}
    weights: dict[str, float] = {}
    for item in str(value).split(","):
        text = item.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"unsupported_weight_spec:{text}")
        key, raw_value = text.split("=", 1)
        weights[key.strip()] = float(raw_value.strip())
    return weights


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
    parser.add_argument("--train-supervision-roles", type=str, default=None)
    parser.add_argument("--val-supervision-roles", type=str, default=None)
    parser.add_argument("--train-license-buckets", type=str, default=None)
    parser.add_argument("--val-license-buckets", type=str, default=None)
    parser.add_argument("--train-target-quality-tiers", type=str, default=None)
    parser.add_argument("--val-target-quality-tiers", type=str, default=None)
    parser.add_argument("--train-paper-splits", type=str, default=None)
    parser.add_argument("--val-paper-splits", type=str, default=None)
    parser.add_argument("--train-material-families", type=str, default=None)
    parser.add_argument("--val-material-families", type=str, default=None)
    parser.add_argument("--train-lighting-bank-ids", type=str, default=None)
    parser.add_argument("--val-lighting-bank-ids", type=str, default=None)
    parser.add_argument("--train-require-prior", type=str, default="any")
    parser.add_argument("--val-require-prior", type=str, default="any")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--val-balance-key", type=str, default="none")
    parser.add_argument("--val-records-per-balance-group", type=int, default=0)
    parser.add_argument(
        "--val-expected-balance-groups",
        type=str,
        default="metal_dominant,ceramic_glossy,glass_metal,mixed_thin_boundary,glossy_non_metal",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=0, help="0 reuses --batch-size.")
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
    parser.add_argument("--train-view-sample-count", type=int, default=0)
    parser.add_argument("--train-min-hard-views", type=int, default=0)
    parser.add_argument("--train-randomize-view-subset", type=parse_bool, default=False)
    parser.add_argument("--val-view-sample-count", type=int, default=0)
    parser.add_argument("--val-min-hard-views", type=int, default=0)
    parser.add_argument("--disable-view-fusion", type=parse_bool, default=False)
    parser.add_argument("--disable-prior-inputs", type=parse_bool, default=False)
    parser.add_argument("--disable-residual-head", type=parse_bool, default=False)
    parser.add_argument("--enable-residual-gate", type=parse_bool, default=False)
    parser.add_argument("--residual-gate-bias", type=float, default=-0.5)
    parser.add_argument("--min-residual-gate", type=float, default=0.05)
    parser.add_argument("--prior-confidence-gate-strength", type=float, default=0.0)
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
            "material",
            "generator_x_prior",
            "source_x_prior",
            "material_x_source_x_prior",
            "material_x_generator_x_prior",
        ],
        default="source_x_prior",
    )
    parser.add_argument("--train-balance-power", type=float, default=1.0)
    parser.add_argument("--train-samples-per-epoch", type=int, default=0)
    parser.add_argument("--train-target-quality-weights", type=str, default=None)
    parser.add_argument("--train-difficulty-metadata-key", type=str, default=None)
    parser.add_argument("--train-difficulty-weights", type=str, default=None)
    parser.add_argument("--train-failure-tag-metadata-key", type=str, default=None)
    parser.add_argument("--train-failure-tag-weights", type=str, default=None)
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
    parser.add_argument("--edge-aware-weight", type=float, default=0.0)
    parser.add_argument("--edge-aware-epsilon", type=float, default=1e-3)
    parser.add_argument("--boundary-bleed-weight", type=float, default=0.0)
    parser.add_argument("--boundary-band-kernel", type=int, default=5)
    parser.add_argument("--gradient-preservation-weight", type=float, default=0.0)
    parser.add_argument("--metallic-classification-weight", type=float, default=0.0)
    parser.add_argument("--residual-safety-weight", type=float, default=0.0)
    parser.add_argument("--residual-safety-margin", type=float, default=0.03)
    parser.add_argument("--view-consistency-mode", choices=["auto", "required", "disabled"], default="auto")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--validation-steps", type=int, default=0)
    parser.add_argument(
        "--validation-progress-milestones",
        type=int,
        default=0,
        help="When >0, validate when total training progress crosses 1/N milestones.",
    )
    parser.add_argument(
        "--render-proxy-validation-milestone-interval",
        type=int,
        default=0,
        help="When >0, compute lightweight view-space RM proxy metrics every N progress-milestone validations.",
    )
    parser.add_argument(
        "--render-proxy-validation-max-batches",
        type=int,
        default=4,
        help="Maximum validation batches used for lightweight render-proxy metrics. 0 means full validation loader.",
    )
    parser.add_argument("--max-validation-batches", type=int, default=0, help="0 evaluates the full validation loader.")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--checkpointing-steps", type=int, default=0)
    parser.add_argument("--save-only-best-checkpoint", type=parse_bool, default=False)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--progress-bar",
        choices=["auto", "true", "false"],
        default="true",
        help="Use a compact tqdm progress bar for train steps. 'true' keeps it enabled even when piping through tee.",
    )
    parser.add_argument("--progress-bar-leave", type=parse_bool, default=False)
    parser.add_argument("--progress-bar-mininterval", type=float, default=0.5)
    parser.add_argument(
        "--train-line-logs",
        type=parse_bool,
        default=False,
        help="Print verbose [train] interval lines in addition to the progress bar.",
    )
    parser.add_argument("--val-preview-samples", type=int, default=4)
    parser.add_argument("--wandb-val-preview-max", type=int, default=8)
    parser.add_argument("--wandb-log-preview-grid", type=parse_bool, default=False)
    parser.add_argument("--save-preview-contact-sheet", type=parse_bool, default=False)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--init-from-checkpoint",
        type=Path,
        default=None,
        help="Load model weights only, then start a fresh optimizer/scheduler run.",
    )
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
    parser.add_argument(
        "--wandb-artifact-policy",
        choices=["none", "best", "final", "best_and_final", "all"],
        default="best_and_final",
        help=(
            "Controls checkpoint artifact upload volume. 'all' reproduces the old per-epoch behavior; "
            "'best_and_final' keeps W&B useful without flooding slow links."
        ),
    )
    parser.add_argument("--wandb-watch", type=parse_bool, default=False)
    parser.add_argument("--export-training-curves", type=parse_bool, default=True)
    parser.add_argument(
        "--terminal-json-logs",
        type=parse_bool,
        default=False,
        help="Also print full machine-readable JSON interval/epoch payloads to terminal.",
    )
    parser.add_argument(
        "--startup-probe-batches",
        type=int,
        default=2,
        help="Number of initial dataloader batches to read for startup throughput and tensor checks.",
    )
    parser.add_argument(
        "--startup-probe-device-transfer",
        type=parse_bool,
        default=True,
        help="During startup probing, also time host-to-device transfer for sampled batches.",
    )
    parser.add_argument(
        "--dataset-distribution-topk",
        type=int,
        default=6,
        help="How many distribution buckets to show in terminal dataset summaries.",
    )
    parser.add_argument("--preflight-audit-records", type=int, default=256)
    parser.add_argument("--target-prior-identity-warning-threshold", type=float, default=0.95)
    parser.add_argument(
        "--max-target-prior-identity-rate-for-paper",
        type=float,
        default=DEFAULT_MAX_TARGET_PRIOR_IDENTITY_RATE_FOR_PAPER,
    )
    parser.add_argument(
        "--min-nontrivial-target-count-for-paper",
        type=int,
        default=DEFAULT_MIN_NONTRIVIAL_TARGET_COUNT_FOR_PAPER,
    )
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
    if args.progress_bar is None:
        args.progress_bar = "true"
    elif isinstance(args.progress_bar, bool):
        args.progress_bar = "true" if args.progress_bar else "false"
    else:
        progress_bar = str(args.progress_bar).strip().lower()
        if progress_bar in {"1", "yes", "y", "on"}:
            progress_bar = "true"
        elif progress_bar in {"0", "no", "n", "off"}:
            progress_bar = "false"
        if progress_bar not in {"auto", "true", "false"}:
            raise ValueError(f"unsupported_progress_bar:{args.progress_bar}")
        args.progress_bar = progress_bar
    args.progress_bar_leave = parse_bool(args.progress_bar_leave) if args.progress_bar_leave is not None else False
    args.train_line_logs = parse_bool(args.train_line_logs) if args.train_line_logs is not None else False
    args.terminal_json_logs = parse_bool(args.terminal_json_logs) if args.terminal_json_logs is not None else False
    args.startup_probe_device_transfer = (
        parse_bool(args.startup_probe_device_transfer)
        if args.startup_probe_device_transfer is not None
        else True
    )
    args.train_manifest = args.train_manifest or args.manifest
    args.val_manifest = args.val_manifest or args.manifest
    args.train_generator_ids = parse_csv_list(args.train_generator_ids)
    args.val_generator_ids = parse_csv_list(args.val_generator_ids)
    args.train_source_names = parse_csv_list(args.train_source_names)
    args.val_source_names = parse_csv_list(args.val_source_names)
    args.train_supervision_tiers = parse_csv_list(args.train_supervision_tiers)
    args.val_supervision_tiers = parse_csv_list(args.val_supervision_tiers)
    args.train_supervision_roles = parse_csv_list(args.train_supervision_roles)
    args.val_supervision_roles = parse_csv_list(args.val_supervision_roles)
    args.train_license_buckets = parse_csv_list(args.train_license_buckets)
    args.val_license_buckets = parse_csv_list(args.val_license_buckets)
    args.train_target_quality_tiers = parse_csv_list(args.train_target_quality_tiers)
    args.val_target_quality_tiers = parse_csv_list(args.val_target_quality_tiers)
    args.train_paper_splits = parse_csv_list(args.train_paper_splits)
    args.val_paper_splits = parse_csv_list(args.val_paper_splits)
    args.train_material_families = parse_csv_list(args.train_material_families)
    args.val_material_families = parse_csv_list(args.val_material_families)
    args.train_lighting_bank_ids = parse_csv_list(args.train_lighting_bank_ids)
    args.val_lighting_bank_ids = parse_csv_list(args.val_lighting_bank_ids)
    args.val_expected_balance_groups = parse_csv_list(args.val_expected_balance_groups) or []
    args.train_require_prior = parse_optional_bool(args.train_require_prior)
    args.val_require_prior = parse_optional_bool(args.val_require_prior)
    args.reload_manifest_every = max(int(args.reload_manifest_every), 1)
    args.grad_accumulation_steps = max(int(args.grad_accumulation_steps), 1)
    args.log_every = max(int(args.log_every), 1)
    args.eval_every = max(int(args.eval_every), 1)
    args.validation_progress_milestones = max(int(args.validation_progress_milestones), 0)
    args.render_proxy_validation_milestone_interval = max(
        int(args.render_proxy_validation_milestone_interval), 0
    )
    args.render_proxy_validation_max_batches = max(int(args.render_proxy_validation_max_batches), 0)
    args.wandb_val_preview_max = max(int(args.wandb_val_preview_max), 0)
    args.val_records_per_balance_group = max(int(args.val_records_per_balance_group), 0)
    args.startup_probe_batches = max(int(args.startup_probe_batches), 0)
    args.dataset_distribution_topk = max(int(args.dataset_distribution_topk), 1)
    args.progress_bar_mininterval = max(float(args.progress_bar_mininterval), 0.0)
    args.boundary_band_kernel = max(int(args.boundary_band_kernel), 1)
    if args.boundary_band_kernel % 2 == 0:
        args.boundary_band_kernel += 1
    args.save_every = max(int(args.save_every), 1)
    args.prior_dropout_prob = max(0.0, min(1.0, float(args.prior_dropout_prob)))
    if args.prior_dropout_start_prob is not None:
        args.prior_dropout_start_prob = max(0.0, min(1.0, float(args.prior_dropout_start_prob)))
    if args.prior_dropout_end_prob is not None:
        args.prior_dropout_end_prob = max(0.0, min(1.0, float(args.prior_dropout_end_prob)))
    args.prior_dropout_warmup_epochs = max(int(args.prior_dropout_warmup_epochs), 0)
    args.min_nontrivial_target_count_for_paper = max(int(args.min_nontrivial_target_count_for_paper), 0)
    args.train_target_quality_weights = parse_weight_map(args.train_target_quality_weights)
    args.train_difficulty_weights = parse_weight_map(args.train_difficulty_weights)
    args.train_failure_tag_weights = parse_weight_map(args.train_failure_tag_weights)
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


def rm_gradient_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    grad_x = torch.zeros_like(tensor)
    grad_y = torch.zeros_like(tensor)
    grad_x[:, :, :, 1:] = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs()
    grad_y[:, :, 1:, :] = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs()
    return (grad_x + grad_y).amax(dim=1, keepdim=True)


def edge_aware_l1_loss(
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    edge_weight = rm_gradient_magnitude(target).detach()
    edge_weight = edge_weight / edge_weight.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(epsilon)
    edge_weight = edge_weight.clamp(0.0, 1.0) * confidence
    return ((refined - target).abs() * edge_weight).sum() / edge_weight.sum().clamp_min(1.0)


def boundary_bleed_loss(
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    kernel_size: int,
    epsilon: float,
) -> torch.Tensor:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    edge_weight = rm_gradient_magnitude(target).detach()
    edge_weight = edge_weight / edge_weight.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(epsilon)
    edge_band = F.max_pool2d(
        edge_weight.clamp(0.0, 1.0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).clamp(0.0, 1.0)
    boundary_weight = edge_band * confidence
    interior_weight = (1.0 - edge_band) * confidence
    error = (refined - target).abs()
    boundary_error = (error * boundary_weight).sum() / boundary_weight.sum().clamp_min(1.0)
    interior_error = (error * interior_weight).sum() / interior_weight.sum().clamp_min(1.0)
    return boundary_error + F.relu(boundary_error - interior_error)


def gradient_preservation_loss(
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    refined_dx = refined[:, :, :, 1:] - refined[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    weight_dx = torch.minimum(confidence[:, :, :, 1:], confidence[:, :, :, :-1])
    refined_dy = refined[:, :, 1:, :] - refined[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    weight_dy = torch.minimum(confidence[:, :, 1:, :], confidence[:, :, :-1, :])
    loss_dx = ((refined_dx - target_dx).abs() * weight_dx).sum()
    loss_dy = ((refined_dy - target_dy).abs() * weight_dy).sum()
    weight = weight_dx.sum() + weight_dy.sum()
    return (loss_dx + loss_dy) / weight.clamp_min(1.0)


def metallic_classification_loss(
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    metallic_pred = refined[:, 1:2].float().clamp(1e-4, 1.0 - 1e-4)
    metallic_target = (target[:, 1:2].float() >= METALLIC_THRESHOLD).to(metallic_pred.dtype)
    per_texel = -(
        metallic_target * metallic_pred.log()
        + (1.0 - metallic_target) * (1.0 - metallic_pred).log()
    )
    weight = confidence.float()
    return (per_texel * weight).sum() / weight.sum().clamp_min(1.0)


def residual_safety_loss(
    refined: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    target_delta = (target - baseline).abs().detach()
    unnecessary_region = (target_delta < margin).to(refined.dtype) * confidence
    residual = (refined - baseline).abs()
    return (residual * unnecessary_region).sum() / unnecessary_region.sum().clamp_min(1.0)


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


def extract_metadata_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


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
    edge_aware = (
        edge_aware_l1_loss(
            refined,
            target,
            confidence,
            epsilon=args.edge_aware_epsilon,
        )
        if args.edge_aware_weight > 0.0
        else refined.new_zeros(())
    )
    boundary_bleed = (
        boundary_bleed_loss(
            refined,
            target,
            confidence,
            kernel_size=args.boundary_band_kernel,
            epsilon=args.edge_aware_epsilon,
        )
        if args.boundary_bleed_weight > 0.0
        else refined.new_zeros(())
    )
    gradient_preservation = (
        gradient_preservation_loss(refined, target, confidence)
        if args.gradient_preservation_weight > 0.0
        else refined.new_zeros(())
    )
    metallic_classification = (
        metallic_classification_loss(refined, target, confidence)
        if args.metallic_classification_weight > 0.0
        else refined.new_zeros(())
    )
    residual_safety = (
        residual_safety_loss(
            refined,
            baseline,
            target,
            confidence,
            margin=args.residual_safety_margin,
        )
        if args.residual_safety_weight > 0.0
        else refined.new_zeros(())
    )
    residual_gate = outputs.get("residual_gate")
    residual_delta = outputs.get("residual_delta")
    residual_gate_mean = (
        residual_gate.mean() if residual_gate is not None else refined.new_zeros(())
    )
    residual_delta_abs = (
        residual_delta.abs().mean() if residual_delta is not None else refined.new_zeros(())
    )

    if (
        args.view_consistency_mode != "disabled"
        and batch.get("view_targets") is not None
        and batch.get("view_uvs") is not None
    ):
        sampled_refined = sample_uv_maps_to_view(refined, batch["view_uvs"])
        view_mask = batch["view_masks"].clamp(0.0, 1.0)
        supervision_mask = batch["has_effective_view_supervision"].to(view_mask.device)
        view_mask = view_mask * supervision_mask.view(-1, 1, 1, 1, 1)
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
        + args.edge_aware_weight * edge_aware
        + args.boundary_bleed_weight * boundary_bleed
        + args.gradient_preservation_weight * gradient_preservation
        + args.metallic_classification_weight * metallic_classification
        + args.residual_safety_weight * residual_safety
    )
    return {
        "total": total,
        "refine_l1": refine_l1.detach(),
        "coarse_l1": coarse_l1.detach(),
        "prior_consistency": prior_consistency.detach(),
        "smoothness": smoothness.detach(),
        "view_consistency": view_consistency.detach(),
        "edge_aware": edge_aware.detach(),
        "boundary_bleed": boundary_bleed.detach(),
        "gradient_preservation": gradient_preservation.detach(),
        "metallic_classification": metallic_classification.detach(),
        "residual_safety": residual_safety.detach(),
        "residual_gate_mean": residual_gate_mean.detach(),
        "residual_delta_abs": residual_delta_abs.detach(),
    }


def sample_balance_key(record: Any, mode: str) -> str:
    generator_key = str(record.generator_id)
    source_name = str(record.metadata.get("source_name", record.generator_id))
    prior_key = "with_prior" if record.has_material_prior else "without_prior"
    material_key = str(getattr(record, "material_family", None) or record.metadata.get("material_family", "unknown"))
    if mode == "generator":
        return generator_key
    if mode == "source":
        return source_name
    if mode == "prior":
        return prior_key
    if mode == "tier":
        return record.supervision_tier
    if mode == "material":
        return material_key
    if mode == "generator_x_prior":
        return f"{generator_key}|{prior_key}"
    if mode == "source_x_prior":
        return f"{source_name}|{prior_key}"
    if mode == "material_x_source_x_prior":
        return f"{material_key}|{source_name}|{prior_key}"
    if mode == "material_x_generator_x_prior":
        return f"{material_key}|{generator_key}|{prior_key}"
    return "all"


def sample_extra_weight(record: Any, args: argparse.Namespace) -> float:
    weight = 1.0
    if args.train_target_quality_weights:
        weight *= float(
            args.train_target_quality_weights.get(
                str(getattr(record, "target_quality_tier", "unknown")),
                1.0,
            )
        )
    if args.train_difficulty_metadata_key and args.train_difficulty_weights:
        difficulty_value = record.metadata.get(args.train_difficulty_metadata_key)
        if difficulty_value is not None:
            weight *= float(args.train_difficulty_weights.get(str(difficulty_value), 1.0))
    if args.train_failure_tag_metadata_key and args.train_failure_tag_weights:
        tags = extract_metadata_labels(record.metadata.get(args.train_failure_tag_metadata_key))
        if tags:
            tag_weight = max(
                float(args.train_failure_tag_weights.get(tag, 1.0))
                for tag in tags
            )
            weight *= tag_weight
    return max(weight, 1e-6)


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
    weights = [
        float(base_weight) * sample_extra_weight(record, args)
        for base_weight, record in zip(weights, dataset.records, strict=True)
    ]

    num_samples = args.train_samples_per_epoch if args.train_samples_per_epoch > 0 else len(dataset.records)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def record_field_value(record: Any, key: str) -> str:
    key = str(key or "none").strip()
    if not key or key.lower() in {"none", "off", "disabled"}:
        return "all"
    if hasattr(record, key):
        value = getattr(record, key)
    else:
        value = record.metadata.get(key, "unknown")
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value) or "unknown"
    return str(value)


def stable_record_order(records: list[Any], *, seed: int) -> list[Any]:
    rng = random.Random(seed)
    keyed_records = []
    for record in records:
        object_id = str(getattr(record, "object_id", ""))
        keyed_records.append((rng.random(), object_id, record))
    keyed_records.sort(key=lambda item: (item[0], item[1]))
    return [record for _random_key, _object_id, record in keyed_records]


def round_robin_groups(grouped_records: dict[str, list[Any]]) -> list[Any]:
    ordered_groups = sorted(grouped_records)
    selected: list[Any] = []
    depth = 0
    while True:
        added = False
        for group_key in ordered_groups:
            records = grouped_records[group_key]
            if depth < len(records):
                selected.append(records[depth])
                added = True
        if not added:
            return selected
        depth += 1


def balance_validation_records(
    records: list[Any],
    args: argparse.Namespace,
) -> tuple[list[Any], dict[str, Any]]:
    balance_key = str(getattr(args, "val_balance_key", "none") or "none").strip()
    max_records = args.max_val_records
    max_records = None if max_records is None or max_records <= 0 else int(max_records)
    expected_groups = [str(group) for group in getattr(args, "val_expected_balance_groups", [])]
    summary: dict[str, Any] = {
        "enabled": balance_key.lower() not in {"", "none", "off", "disabled"},
        "key": balance_key,
        "original_records": len(records),
        "max_records": max_records,
        "records_per_group": int(getattr(args, "val_records_per_balance_group", 0)),
        "expected_groups": expected_groups,
    }
    if not records:
        summary.update({"selected_records": 0, "groups": {}, "warnings": ["empty_validation_records"]})
        return records, summary

    if not summary["enabled"]:
        selected = records[:max_records] if max_records is not None else list(records)
        summary.update(
            {
                "selected_records": len(selected),
                "groups": {"all": len(selected)},
                "warnings": [],
            }
        )
        return selected, summary

    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[record_field_value(record, balance_key)].append(record)

    ordered_group_records = {
        group_key: stable_record_order(group_records, seed=int(args.seed) + index * 9973)
        for index, (group_key, group_records) in enumerate(sorted(grouped.items()))
    }
    group_count = max(len(ordered_group_records), 1)
    records_per_group = int(getattr(args, "val_records_per_balance_group", 0))
    if records_per_group <= 0:
        if max_records is not None:
            records_per_group = max(1, math.ceil(max_records / group_count))
        elif group_count > 1:
            records_per_group = min(len(group_records) for group_records in ordered_group_records.values())
        else:
            records_per_group = len(next(iter(ordered_group_records.values())))

    capped_groups = {
        group_key: group_records[:records_per_group]
        for group_key, group_records in ordered_group_records.items()
    }
    selected = round_robin_groups(capped_groups)
    if max_records is not None:
        selected = selected[:max_records]

    selected_counts = Counter(record_field_value(record, balance_key) for record in selected)
    available_counts = Counter({group_key: len(group_records) for group_key, group_records in grouped.items()})
    missing_expected = [
        group for group in expected_groups if group not in available_counts
    ]
    warnings = []
    if len(selected_counts) <= 1:
        warnings.append(f"validation_balance_single_group:{next(iter(selected_counts), 'none')}")
    if missing_expected:
        warnings.append("validation_balance_missing_expected_groups:" + ",".join(missing_expected))
    summary.update(
        {
            "selected_records": len(selected),
            "groups": dict(sorted(selected_counts.items())),
            "available_groups": dict(sorted(available_counts.items())),
            "missing_expected_groups": missing_expected,
            "warnings": warnings,
        }
    )
    return selected, summary


def compact_dataset_wandb_logs(data_state: dict[str, Any]) -> dict[str, Any]:
    val_balance = data_state.get("val_balance") or {}
    val_summary = data_state.get("val") or {}
    train_summary = data_state.get("train") or {}
    logs: dict[str, Any] = {
        "dataset/train_records": train_summary.get("records", 0),
        "dataset/val_records": val_summary.get("records", 0),
        "dataset/train_batches": data_state.get("train_batches", 0),
        "dataset/val_batches": data_state.get("val_batches", 0),
        "dataset/val_balance_group_count": len((val_balance.get("groups") or {})),
        "dataset/val_balance_warning_count": len((val_balance.get("warnings") or [])),
    }
    for group_key, count in (val_balance.get("groups") or {}).items():
        safe_group_key = str(group_key).replace("/", "_")
        logs[f"dataset/val_material_family/{safe_group_key}/count"] = count
    return logs


def filter_train_wandb_logs(logs: dict[str, Any]) -> dict[str, Any]:
    allowed_exact = {"epoch", "optimizer_step", "global_batch_step", "lr"}
    allowed_prefixes = (
        "train/",
        "memory/gpu_allocated_gb",
        "memory/gpu_reserved_gb",
        "memory/gpu_max_allocated_gb",
    )
    blocked_prefixes = ("progress/",)
    filtered: dict[str, Any] = {}
    for key, value in logs.items():
        if key.startswith(blocked_prefixes):
            continue
        if key in allowed_exact or key.startswith(allowed_prefixes):
            filtered[key] = value
    return filtered


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


def build_model_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "disable_view_fusion": bool(args.disable_view_fusion),
        "disable_prior_inputs": bool(args.disable_prior_inputs),
        "disable_residual_head": bool(args.disable_residual_head),
        "enable_residual_gate": bool(args.enable_residual_gate),
        "residual_gate_bias": float(args.residual_gate_bias),
        "min_residual_gate": float(args.min_residual_gate),
        "prior_confidence_gate_strength": float(args.prior_confidence_gate_strength),
    }


def make_dataset(
    manifest_path: Path,
    *,
    split: str,
    args: argparse.Namespace,
    generator_ids: list[str] | None,
    source_names: list[str] | None,
    supervision_tiers: list[str] | None,
    supervision_roles: list[str] | None,
    license_buckets: list[str] | None,
    target_quality_tiers: list[str] | None,
    paper_splits: list[str] | None,
    material_families: list[str] | None,
    lighting_bank_ids: list[str] | None,
    require_prior: bool | None,
    max_records: int | None,
    max_views_per_sample: int = 0,
    min_hard_views_per_sample: int = 0,
    randomize_view_subset: bool = False,
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
        supervision_roles=supervision_roles,
        license_buckets=license_buckets,
        target_quality_tiers=target_quality_tiers,
        paper_splits=paper_splits,
        material_families=material_families,
        lighting_bank_ids=lighting_bank_ids,
        require_prior=require_prior,
        max_records=max_records,
        atlas_size=args.atlas_size,
        buffer_resolution=args.buffer_resolution,
        max_views_per_sample=max_views_per_sample,
        min_hard_views_per_sample=min_hard_views_per_sample,
        randomize_view_subset=randomize_view_subset,
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
        supervision_roles=args.val_supervision_roles,
        license_buckets=args.val_license_buckets,
        target_quality_tiers=args.val_target_quality_tiers,
        paper_splits=args.val_paper_splits,
        material_families=args.val_material_families,
        lighting_bank_ids=args.val_lighting_bank_ids,
        require_prior=args.val_require_prior,
        max_records=None,
    )
    val_records, val_balance_summary = balance_validation_records(val_dataset.records, args)
    val_dataset.records = val_records
    write_manifest_snapshot(
        freeze_path,
        records=val_dataset.records,
        source_manifest=args.val_manifest,
        metadata={
            "frozen_from_split": args.val_split,
            "split_strategy": args.split_strategy,
            "hash_val_ratio": args.hash_val_ratio,
            "hash_test_ratio": args.hash_test_ratio,
            "val_balance": val_balance_summary,
        },
    )
    if val_balance_summary.get("warnings"):
        print_val_balance_summary(0, val_balance_summary)
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


def format_gb(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}GB"


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


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return "n/a"
    total_seconds = int(round(seconds))
    days, remainder = divmod(total_seconds, 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return "n/a"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    return format_duration(seconds)


def short_path(path: str | Path | None) -> str:
    if path is None:
        return "n/a"
    try:
        path = Path(path)
        if path.is_absolute():
            return str(path.relative_to(REPO_ROOT))
    except Exception:
        pass
    return str(path)


def top_counts_text(counts: dict[str, Any] | None, *, topk: int = 6) -> str:
    if not counts:
        return "none"
    items = sorted(
        ((str(key), int(value)) for key, value in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    shown = [f"{key}:{value}" for key, value in items[:topk]]
    remainder = len(items) - len(shown)
    if remainder > 0:
        shown.append(f"+{remainder} more")
    return ", ".join(shown)


def tensor_probe_stats(tensor: torch.Tensor | None, *, max_values: int = 262144) -> dict[str, Any]:
    if tensor is None:
        return {"available": False}
    detached = tensor.detach()
    stats: dict[str, Any] = {
        "available": True,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
    }
    if detached.numel() == 0:
        stats.update({"finite_rate": 1.0, "min": None, "max": None, "mean": None})
        return stats
    values = detached.float().reshape(-1)
    if values.numel() > max_values:
        stride = max(values.numel() // max_values, 1)
        values = values[::stride][:max_values]
    finite = torch.isfinite(values)
    finite_rate = float(finite.float().mean().item())
    values = values[finite]
    stats["finite_rate"] = finite_rate
    if values.numel() == 0:
        stats.update({"min": None, "max": None, "mean": None})
    else:
        stats.update(
            {
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "mean": float(values.mean().item()),
            }
        )
    return stats


def tensor_probe_text(stats: dict[str, Any]) -> str:
    if not stats.get("available"):
        return "missing"
    return (
        f"shape={tuple(stats.get('shape', []))} dtype={stats.get('dtype')} "
        f"range=[{format_metric(stats.get('min'), 4)}, {format_metric(stats.get('max'), 4)}] "
        f"mean={format_metric(stats.get('mean'), 4)} finite={format_metric(stats.get('finite_rate'), 4)}"
    )


def gpu_device_inventory(selected_device: str) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    selected_index = torch.device(selected_device).index if selected_device.startswith("cuda") else None
    if selected_index is None and selected_device.startswith("cuda"):
        selected_index = torch.cuda.current_device()
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free_bytes = None
        total_bytes = int(props.total_memory)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except Exception:
            free_bytes = None
        devices.append(
            {
                "index": index,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_gb": total_bytes / (1024**3),
                "free_gb": None if free_bytes is None else free_bytes / (1024**3),
                "used_gb": None
                if free_bytes is None
                else (total_bytes - free_bytes) / (1024**3),
                "selected": index == selected_index,
            }
        )
    return devices


def system_runtime_info(args: argparse.Namespace, device: str) -> dict[str, Any]:
    disk = shutil.disk_usage(args.output_dir.parent if args.output_dir.parent.exists() else REPO_ROOT)
    load_average = None
    try:
        load_average = os.getloadavg()
    except OSError:
        load_average = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu_count": os.cpu_count(),
        "load_average": load_average,
        "cwd": str(Path.cwd()),
        "output_dir": str(args.output_dir.resolve()),
        "disk_free_gb": disk.free / (1024**3),
        "disk_total_gb": disk.total / (1024**3),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": device,
        "gpus": gpu_device_inventory(device),
    }


def print_system_runtime_report(
    *,
    args: argparse.Namespace,
    device: str,
    runtime_info: dict[str, Any],
) -> None:
    load_average = runtime_info.get("load_average")
    load_text = (
        ",".join(format_metric(value, 2) for value in load_average)
        if load_average is not None
        else "n/a"
    )
    print(
        "[run] "
        f"name={args.tracker_run_name or args.output_dir.name} seed={args.seed} "
        f"output={short_path(args.output_dir)} cwd={short_path(Path.cwd())}"
    )
    print(
        "[system] "
        f"python={runtime_info.get('python')} torch={runtime_info.get('torch')} "
        f"cuda_runtime={runtime_info.get('cuda_runtime')} cudnn={runtime_info.get('cudnn')} "
        f"cpu={runtime_info.get('cpu_count')} load={load_text} "
        f"disk_free={format_gb(runtime_info.get('disk_free_gb'))}/{format_gb(runtime_info.get('disk_total_gb'))}"
    )
    print(
        "[device] "
        f"selected={device} amp={args.amp_dtype} matmul={args.matmul_precision} "
        f"tf32={bool(args.allow_tf32)} CUDA_VISIBLE_DEVICES={runtime_info.get('cuda_visible_devices')}"
    )
    if runtime_info.get("gpus"):
        for gpu in runtime_info["gpus"]:
            mark = "*" if gpu.get("selected") else " "
            print(
                "[device:gpu] "
                f"{mark}cuda:{gpu.get('index')} name={gpu.get('name')} "
                f"cap={gpu.get('capability')} total={format_gb(gpu.get('total_gb'))} "
                f"used={format_gb(gpu.get('used_gb'))} free={format_gb(gpu.get('free_gb'))}"
            )
    else:
        print("[device:gpu] cuda_unavailable using_cpu=true")


def sampler_description(loader: DataLoader) -> str:
    sampler = getattr(loader, "sampler", None)
    batch_sampler = getattr(loader, "batch_sampler", None)
    sampler_name = type(sampler).__name__ if sampler is not None else "none"
    if sampler is not None and hasattr(sampler, "num_samples"):
        sampler_name += f"(num_samples={getattr(sampler, 'num_samples')})"
    batch_sampler_name = type(batch_sampler).__name__ if batch_sampler is not None else "none"
    return f"sampler={sampler_name} batch_sampler={batch_sampler_name}"


def print_dataset_distribution(
    *,
    label: str,
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    topk = int(args.dataset_distribution_topk)
    print(
        f"[data:{label}] "
        f"records={summary.get('records', 0)} "
        f"default_split={top_counts_text(summary.get('default_split'), topk=topk)} "
        f"paper_split={top_counts_text(summary.get('paper_split'), topk=topk)}"
    )
    print(
        f"[data:{label}:dist] "
        f"source={top_counts_text(summary.get('source_name'), topk=topk)} | "
        f"generator={top_counts_text(summary.get('generator_id'), topk=topk)} | "
        f"material={top_counts_text(summary.get('material_family'), topk=topk)} | "
        f"prior={top_counts_text(summary.get('has_material_prior'), topk=topk)}"
    )
    print(
        f"[data:{label}:quality] "
        f"tier={top_counts_text(summary.get('supervision_tier'), topk=topk)} | "
        f"role={top_counts_text(summary.get('supervision_role'), topk=topk)} | "
        f"target_quality={top_counts_text(summary.get('target_quality_tier'), topk=topk)} | "
        f"license={top_counts_text(summary.get('license_bucket'), topk=topk)} | "
        f"view_sup={top_counts_text(summary.get('view_supervision_ready'), topk=topk)}"
    )


def count_existing_record_paths(records: list[Any], manifest_dir: Path, fields: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for field in fields:
        present = 0
        missing = 0
        empty = 0
        for record in records:
            value = getattr(record, field, None)
            if not value:
                empty += 1
                continue
            path = Path(value)
            if not path.is_absolute():
                if getattr(record, "bundle_root", None):
                    bundle_root = Path(record.bundle_root)
                    if not bundle_root.is_absolute():
                        bundle_root = manifest_dir / bundle_root
                    candidate = bundle_root / path
                    if candidate.exists():
                        present += 1
                        continue
                path = manifest_dir / path
            if path.exists():
                present += 1
            else:
                missing += 1
        counts[field] = {"present": present, "missing": missing, "empty": empty}
    return counts


def print_record_path_checks(
    *,
    label: str,
    dataset: CanonicalMaterialDataset,
) -> dict[str, dict[str, int]]:
    fields = [
        "uv_albedo_path",
        "uv_normal_path",
        "uv_prior_roughness_path",
        "uv_prior_metallic_path",
        "uv_target_roughness_path",
        "uv_target_metallic_path",
        "uv_target_confidence_path",
        "canonical_buffer_root",
    ]
    path_counts = count_existing_record_paths(dataset.records, dataset.manifest_dir, fields)
    compact = " | ".join(
        f"{field}:ok={counts['present']} miss={counts['missing']} empty={counts['empty']}"
        for field, counts in path_counts.items()
    )
    print(f"[data:{label}:paths] {compact}")
    return path_counts


def print_val_balance_summary(epoch: int, val_balance: dict[str, Any] | None) -> None:
    if not val_balance:
        return
    print(
        "[data:val_balance] "
        f"epoch={epoch} enabled={bool(val_balance.get('enabled'))} key={val_balance.get('key')} "
        f"selected={val_balance.get('selected_records')}/{val_balance.get('original_records')} "
        f"groups={top_counts_text(val_balance.get('groups'), topk=8)}"
    )
    for warning in val_balance.get("warnings") or []:
        print(f"[data:val_balance:warning] epoch={epoch} {warning}")


def print_data_reload_summary(
    *,
    epoch: int,
    data_state: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    train_summary = data_state.get("train") or {}
    val_summary = data_state.get("val") or {}
    print(
        "[data:reload] "
        f"epoch={epoch} train_records={train_summary.get('records', 0)} "
        f"val_records={val_summary.get('records', 0)} train_batches={data_state.get('train_batches')} "
        f"val_batches={data_state.get('val_batches')} "
        f"train_material={top_counts_text(train_summary.get('material_family'), topk=args.dataset_distribution_topk)} "
        f"val_material={top_counts_text(val_summary.get('material_family'), topk=args.dataset_distribution_topk)}"
    )
    print_val_balance_summary(epoch, data_state.get("val_balance"))


def batch_probe_payload(batch: dict[str, Any]) -> dict[str, Any]:
    sample_count = int(batch["uv_albedo"].shape[0])
    target = torch.cat([batch["uv_target_roughness"], batch["uv_target_metallic"]], dim=1)
    baseline = torch.cat([batch["uv_prior_roughness"], batch["uv_prior_metallic"]], dim=1)
    confidence = batch["uv_target_confidence"].clamp(0.0, 1.0)
    identity_delta = ((target - baseline).abs() * confidence).sum() / confidence.sum().clamp_min(1.0)
    prior_rate = float(sum(bool(item) for item in batch["has_material_prior"]) / max(sample_count, 1))
    effective_view_rate = float(batch["has_effective_view_supervision"].float().mean().item())
    return {
        "samples": sample_count,
        "object_ids": [str(item) for item in batch["object_id"][: min(sample_count, 4)]],
        "prior_rate": prior_rate,
        "effective_view_supervision_rate": effective_view_rate,
        "target_prior_weighted_delta": float(identity_delta.item()),
        "target_confidence_mean": float(confidence.mean().item()),
        "available_view_count_mean": float(batch["available_view_count"].float().mean().item()),
        "valid_view_count_mean": float(batch["valid_view_count"].float().mean().item()),
        "material_family": dict(Counter(str(item) for item in batch["material_family"])),
        "target_quality_tier": dict(Counter(str(item) for item in batch["target_quality_tier"])),
        "uv_albedo": tensor_probe_stats(batch.get("uv_albedo")),
        "uv_normal": tensor_probe_stats(batch.get("uv_normal")),
        "uv_prior_rm": tensor_probe_stats(baseline),
        "uv_target_rm": tensor_probe_stats(target),
        "uv_target_confidence": tensor_probe_stats(confidence),
        "view_features": tensor_probe_stats(batch.get("view_features")),
        "view_masks": tensor_probe_stats(batch.get("view_masks")),
        "view_uvs": tensor_probe_stats(batch.get("view_uvs")),
        "view_targets": tensor_probe_stats(batch.get("view_targets")),
    }


def probe_dataloader(
    *,
    label: str,
    loader: DataLoader,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    probe_batches = int(args.startup_probe_batches)
    payload: dict[str, Any] = {
        "label": label,
        "requested_batches": probe_batches,
        "read_batches": 0,
        "samples": 0,
        "read_seconds": 0.0,
        "device_transfer_seconds": 0.0,
        "first_batch": None,
    }
    if probe_batches <= 0:
        print(f"[data:{label}:probe] skipped startup_probe_batches=0")
        return payload
    iterator = iter(loader)
    for batch_index in range(1, probe_batches + 1):
        batch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        read_seconds = max(time.perf_counter() - batch_start, 0.0)
        payload["read_batches"] += 1
        payload["read_seconds"] += read_seconds
        payload["samples"] += int(batch["uv_albedo"].shape[0])
        if payload["first_batch"] is None:
            payload["first_batch"] = batch_probe_payload(batch)
        if args.startup_probe_device_transfer and device.startswith("cuda"):
            transfer_start = time.perf_counter()
            moved_batch = move_batch_to_device(batch, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize(torch.device(device))
            payload["device_transfer_seconds"] += max(time.perf_counter() - transfer_start, 0.0)
            del moved_batch
    read_batches = max(int(payload["read_batches"]), 1)
    samples = max(int(payload["samples"]), 1)
    payload["read_seconds_per_batch"] = float(payload["read_seconds"]) / read_batches
    payload["read_samples_per_second"] = samples / max(float(payload["read_seconds"]), 1e-9)
    payload["device_transfer_seconds_per_batch"] = float(payload["device_transfer_seconds"]) / read_batches
    first_batch = payload.get("first_batch") or {}
    print(
        f"[data:{label}:probe] "
        f"batches={payload['read_batches']}/{probe_batches} samples={payload['samples']} "
        f"read_batch={format_seconds(payload['read_seconds_per_batch'])} "
        f"read_samples_s={format_metric(payload['read_samples_per_second'], 3)} "
        f"h2d_batch={format_seconds(payload['device_transfer_seconds_per_batch'])} "
        f"prior_rate={format_metric(first_batch.get('prior_rate'), 4)} "
        f"view_sup_rate={format_metric(first_batch.get('effective_view_supervision_rate'), 4)} "
        f"target_prior_delta={format_metric(first_batch.get('target_prior_weighted_delta'), 6)} "
        f"conf_mean={format_metric(first_batch.get('target_confidence_mean'), 4)}"
    )
    if first_batch:
        print(
            f"[data:{label}:batch] "
            f"objects={','.join(first_batch.get('object_ids', []))} "
            f"material={top_counts_text(first_batch.get('material_family'), topk=4)} "
            f"target_quality={top_counts_text(first_batch.get('target_quality_tier'), topk=4)} "
            f"views={format_metric(first_batch.get('valid_view_count_mean'), 2)}/"
            f"{format_metric(first_batch.get('available_view_count_mean'), 2)}"
        )
        for key in (
            "uv_albedo",
            "uv_normal",
            "uv_prior_rm",
            "uv_target_rm",
            "uv_target_confidence",
            "view_features",
            "view_uvs",
            "view_targets",
        ):
            print(f"[data:{label}:tensor] {key} {tensor_probe_text(first_batch.get(key, {}))}")
    return payload


def print_training_plan(
    *,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    train_dataset: CanonicalMaterialDataset,
    model_info: dict[str, int],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    start_epoch: int,
) -> None:
    planned_samples = loader_planned_samples(train_loader, train_dataset)
    opt_steps_per_epoch = optimizer_steps_per_epoch(train_loader, args)
    session_epochs = max(args.epochs - start_epoch + 1, 0)
    if args.validation_progress_milestones > 0:
        validation_text = f"{args.validation_progress_milestones} progress milestones"
    elif args.validation_steps > 0:
        validation_text = f"every {args.validation_steps} optimizer steps"
    else:
        validation_text = f"every {args.eval_every} epoch(s)"
    print(
        "[model] "
        f"params_total={model_info.get('parameters_total')} "
        f"trainable={model_info.get('parameters_trainable')} cfg={build_model_cfg(args)}"
    )
    print(
        "[optimizer] "
        f"type={args.optimizer} lr={current_lr(optimizer):.8f}->{args.learning_rate:.8f} "
        f"min_lr={args.min_learning_rate:.8f} betas=({args.adam_beta1},{args.adam_beta2}) "
        f"wd={args.weight_decay} scheduler={type(scheduler).__name__ if scheduler else 'none'} "
        f"warmup_steps={args.warmup_steps} grad_clip={args.grad_clip_norm}"
    )
    print(
        "[schedule] "
        f"epochs={start_epoch}-{args.epochs} session_epochs={session_epochs} "
        f"batches_per_epoch={len(train_loader)} optimizer_steps_per_epoch={opt_steps_per_epoch} "
        f"total_optimizer_steps={args.max_train_steps if args.max_train_steps > 0 else opt_steps_per_epoch * args.epochs} "
        f"planned_samples_per_epoch={planned_samples} validation={validation_text}"
    )
    print(
        "[checkpoint] "
        f"output={short_path(args.output_dir)} save_every={args.save_every} "
        f"step_every={args.checkpointing_steps} best_only={bool(args.save_only_best_checkpoint)} "
        f"keep_last={args.keep_last_checkpoints} resume={short_path(args.resume)}"
    )
    print(
        "[loader] "
        f"train_batch={args.batch_size} val_batch={args.val_batch_size if args.val_batch_size > 0 else args.batch_size} "
        f"num_workers={args.num_workers} pin_memory={torch.cuda.is_available()} "
        f"persistent_workers={args.num_workers > 0} drop_last={bool(args.drop_last)} "
        f"{sampler_description(train_loader)}"
    )
    if val_loader is not None:
        print(
            "[loader:val] "
            f"batches={len(val_loader)} max_validation_batches={args.max_validation_batches} "
            f"{sampler_description(val_loader)}"
        )


def estimate_remaining_seconds(progress_fraction: float | None, elapsed_seconds: float | None) -> float | None:
    if progress_fraction is None or elapsed_seconds is None:
        return None
    progress_fraction = float(progress_fraction)
    elapsed_seconds = float(elapsed_seconds)
    if progress_fraction <= 0.0 or elapsed_seconds <= 0.0:
        return None
    progress_fraction = min(progress_fraction, 0.999999)
    return elapsed_seconds * (1.0 - progress_fraction) / progress_fraction


def loader_planned_samples(
    loader: DataLoader,
    dataset: CanonicalMaterialDataset,
) -> int:
    sampler = getattr(loader, "sampler", None)
    num_samples = getattr(sampler, "num_samples", None)
    if num_samples is not None:
        return int(num_samples)
    return int(len(dataset.records))


def optimizer_steps_per_epoch(loader: DataLoader, args: argparse.Namespace) -> int:
    return max(math.ceil(len(loader) / max(args.grad_accumulation_steps, 1)), 1)


def training_progress_fraction(
    *,
    args: argparse.Namespace,
    epoch: int,
    start_epoch: int,
    session_total_epochs: int,
    epoch_optimizer_step: int,
    epoch_optimizer_steps_total: int,
    optimizer_step: int,
) -> float:
    if args.max_train_steps > 0:
        return min(float(optimizer_step) / max(float(args.max_train_steps), 1.0), 1.0)
    epoch_progress = min(
        float(epoch_optimizer_step) / max(float(epoch_optimizer_steps_total), 1.0),
        1.0,
    )
    session_completed_epochs = (epoch - start_epoch) + epoch_progress
    return min(session_completed_epochs / max(float(session_total_epochs), 1.0), 1.0)


def wandb_auth_hint() -> str:
    if os.environ.get("WANDB_API_KEY"):
        return "WANDB_API_KEY"
    if (Path.home() / ".netrc").exists():
        return "~/.netrc"
    return "missing"


def run_preflight_checks(args: argparse.Namespace, device: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_paths = {
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
    }
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
    if args.val_batch_size < 0:
        errors.append(f"invalid_val_batch_size:{args.val_batch_size}")
    if args.max_validation_batches < 0:
        errors.append(f"invalid_max_validation_batches:{args.max_validation_batches}")
    if args.learning_rate <= 0:
        errors.append(f"invalid_learning_rate:{args.learning_rate}")
    if args.report_to == "wandb" and args.wandb_mode == "online" and wandb_auth_hint() == "missing":
        warnings.append("wandb_online_requested_without_visible_login")
    if args.prior_dropout_prob > 0 and args.train_require_prior is False:
        warnings.append("prior_dropout_enabled_but_train_filter_requires_no_prior")
    if args.save_only_best_checkpoint and args.validation_steps > 0:
        warnings.append("save_only_best_checkpoint_with_step_validation_keeps_only_improved_step_checkpoints")
    if args.validation_progress_milestones > 0 and args.validation_steps > 0:
        warnings.append("validation_progress_milestones_overrides_validation_steps_for_scheduled_validation")
    if args.validation_progress_milestones > 0 and args.eval_every > 0:
        warnings.append("validation_progress_milestones_enabled_epoch_validation_disabled")
    manifest_audit = None
    manifest_audit_path = None
    manifest_summary = None
    if Path(args.train_manifest).exists():
        try:
            manifest_audit = audit_manifest(
                Path(args.train_manifest),
                max_records=int(args.preflight_audit_records),
                identity_warning_threshold=float(args.target_prior_identity_warning_threshold),
                max_target_prior_identity_rate_for_paper=float(args.max_target_prior_identity_rate_for_paper),
                min_nontrivial_target_count_for_paper=int(args.min_nontrivial_target_count_for_paper),
            )
            manifest_summary = manifest_audit.get("summary") or {}
            manifest_audit_path = Path(args.output_dir) / "preflight_manifest_audit.json"
            save_json(manifest_audit_path, manifest_audit)
            identity_rate = float(manifest_summary.get("target_prior_identity_rate", 0.0))
            if (
                manifest_summary.get("records", 0) > 0
                and identity_rate >= float(args.target_prior_identity_warning_threshold)
            ):
                message = (
                    "target_prior_identity_high:"
                    f"{identity_rate:.3f};baseline_metrics_may_be_trivial"
                )
                warnings.append(message)
                if args.fail_on_target_prior_identity:
                    if identity_rate > float(args.max_target_prior_identity_rate_for_paper):
                        errors.append(message)
            paper_eligible_records = int(manifest_summary.get("paper_stage_eligible_records", 0))
            if (
                paper_eligible_records
                < int(args.min_nontrivial_target_count_for_paper)
            ):
                message = (
                    "paper_stage_nontrivial_targets_insufficient:"
                    f"{paper_eligible_records}<{int(args.min_nontrivial_target_count_for_paper)}"
                )
                warnings.append(message)
                if args.fail_on_target_prior_identity:
                    errors.append(message)
            effective_view_rate = float(
                manifest_summary.get("effective_view_supervision_record_rate", 0.0)
            )
            if args.view_consistency_mode == "required" and effective_view_rate <= 0.0:
                errors.append("view_consistency_required_but_effective_view_supervision_rate_is_zero")
            elif args.view_consistency_mode != "disabled" and args.view_consistency_weight > 0.0 and effective_view_rate <= 0.0:
                warnings.append(
                    "view_consistency_configured_without_effective_view_supervision:"
                    f"mode={args.view_consistency_mode};consider_weight_0_or_disabled"
                )
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
        "view_consistency_mode": args.view_consistency_mode,
        "view_consistency_weight": float(args.view_consistency_weight),
        "edge_aware_weight": float(args.edge_aware_weight),
        "boundary_bleed_weight": float(args.boundary_bleed_weight),
        "boundary_band_kernel": int(args.boundary_band_kernel),
        "gradient_preservation_weight": float(args.gradient_preservation_weight),
        "metallic_classification_weight": float(args.metallic_classification_weight),
        "residual_safety_weight": float(args.residual_safety_weight),
        "errors": errors,
        "warnings": warnings,
        "manifest_identity_audit": {
            "manifest": str(Path(args.train_manifest).resolve()),
            "audited_records": int((manifest_summary or {}).get("records", 0)),
            "summary": manifest_summary,
            "path": str(manifest_audit_path.resolve()) if manifest_audit_path is not None else None,
        },
    }
    if getattr(args, "terminal_json_logs", False):
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
    if manifest_summary is not None:
        print(
            "[preflight:audit] "
            f"records={manifest_summary.get('records')} "
            f"paper_eligible={manifest_summary.get('paper_stage_eligible_records')} "
            f"target_prior_identity={format_metric(manifest_summary.get('target_prior_identity_rate'), 4)} "
            f"effective_view_rate={format_metric(manifest_summary.get('effective_view_supervision_record_rate'), 4)}"
        )
    if errors:
        for error in errors:
            print(f"[preflight:error] {error}")
        raise RuntimeError("material_refine_preflight_failed")
    return payload


def print_train_interval(payload: dict[str, Any], device: str) -> None:
    memory = payload.get("memory/gpu_max_allocated_gb")
    grad = payload.get("train/grad_norm")
    view_rate = payload.get("train/effective_view_supervision_rate")
    epoch_index = int(payload.get("progress/epoch_index", payload.get("epoch", 0)) or 0)
    epoch_total = int(payload.get("progress/epoch_total", epoch_index) or epoch_index)
    epoch_step = int(payload.get("progress/epoch_step", 0) or 0)
    epoch_steps_total = int(payload.get("progress/epoch_steps_total", epoch_step) or max(epoch_step, 1))
    batch_step = int(payload.get("progress/batch_step", 0) or 0)
    batch_steps_total = int(payload.get("progress/batch_steps_total", batch_step) or max(batch_step, 1))
    global_step = int(payload.get("optimizer_step", 0) or 0)
    global_step_total = payload.get("progress/global_step_total")
    session_progress = payload.get("progress/session_progress")
    progress_text = (
        f"{float(session_progress) * 100.0:.1f}%"
        if session_progress is not None
        else "n/a"
    )
    global_step_text = (
        f"{global_step}/{int(global_step_total)}"
        if global_step_total is not None
        else str(global_step)
    )
    print(
        "[train] "
        f"epoch={epoch_index}/{epoch_total} "
        f"step={epoch_step}/{epoch_steps_total} "
        f"batch={batch_step}/{batch_steps_total} "
        f"global_step={global_step_text} "
        f"progress={progress_text} "
        f"elapsed={format_duration(payload.get('progress/session_elapsed_seconds'))} "
        f"eta_epoch={format_duration(payload.get('progress/eta_epoch_seconds'))} "
        f"eta_total={format_duration(payload.get('progress/eta_total_seconds'))} "
        f"global_batch={payload.get('global_batch_step')} lr={format_metric(payload.get('lr'), 8)} "
        f"loss={format_metric(payload.get('train/total'))} refine={format_metric(payload.get('train/refine_l1'))} "
        f"coarse={format_metric(payload.get('train/coarse_l1'))} "
        f"edge={format_metric(payload.get('train/edge_aware'))} "
        f"bnd={format_metric(payload.get('train/boundary_bleed'))} "
        f"grad_pres={format_metric(payload.get('train/gradient_preservation'))} "
        f"metal_cls={format_metric(payload.get('train/metallic_classification'))} "
        f"res_safe={format_metric(payload.get('train/residual_safety'))} "
        f"grad={format_metric(grad)} "
        f"view_sup_rate={format_metric(view_rate, 4) if view_rate is not None else 'n/a'} "
        f"samples_s={format_metric(payload.get('train/samples_per_second'), 3)} "
        f"max_mem_gb={format_metric(memory, 3) if memory is not None else 'n/a'} device={device}"
    )


def should_use_progress_bar(args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "progress_bar", "auto")).strip().lower()
    if mode == "true":
        return True
    if mode == "false":
        return False
    return bool(sys.stdout.isatty() or sys.stderr.isatty())


def make_epoch_progress_bar(
    *,
    args: argparse.Namespace,
    epoch: int,
    total_steps: int,
    initial_step: int = 0,
):
    if not should_use_progress_bar(args):
        return None
    return tqdm(
        total=max(int(total_steps), 1),
        initial=max(int(initial_step), 0),
        desc=f"epoch {epoch}/{args.epochs}",
        unit="step",
        dynamic_ncols=True,
        leave=bool(args.progress_bar_leave),
        mininterval=float(args.progress_bar_mininterval),
        ascii=True,
        smoothing=0.05,
    )


def update_epoch_progress_bar(
    progress_bar: Any | None,
    *,
    epoch_optimizer_step: int,
    payload: dict[str, Any] | None = None,
) -> None:
    if progress_bar is None:
        return
    target_step = max(int(epoch_optimizer_step), 0)
    delta = target_step - int(progress_bar.n)
    if delta > 0:
        progress_bar.update(delta)
    if not payload:
        return
    memory = payload.get("memory/gpu_max_allocated_gb")
    progress_bar.set_postfix_str(
        " ".join(
            [
                f"loss={format_metric(payload.get('train/total'), 4)}",
                f"lr={format_metric(payload.get('lr'), 7)}",
                f"ref={format_metric(payload.get('train/refine_l1'), 4)}",
                f"bnd={format_metric(payload.get('train/boundary_bleed'), 4)}",
                f"gn={format_metric(payload.get('train/grad_norm'), 3)}",
                f"v={format_metric(payload.get('train/effective_view_supervision_rate'), 2)}",
                f"sps={format_metric(payload.get('train/samples_per_second'), 2)}",
                f"mem={format_metric(memory, 2) if memory is not None else 'n/a'}G",
            ]
        )
    )


def clear_progress_bar(progress_bar: Any | None) -> None:
    if progress_bar is not None:
        progress_bar.clear()


def refresh_progress_bar(progress_bar: Any | None) -> None:
    if progress_bar is not None:
        progress_bar.refresh()


def print_epoch_start(
    *,
    args: argparse.Namespace,
    epoch: int,
    train_records: int,
    val_records: int,
    planned_samples: int,
    batch_steps_total: int,
    optimizer_steps_total: int,
) -> None:
    print(
        "[epoch:start] "
        f"{epoch}/{args.epochs} "
        f"train_records={train_records} val_records={val_records} "
        f"planned_samples={planned_samples} "
        f"batch_size={args.batch_size} grad_accum={args.grad_accumulation_steps} "
        f"batches={batch_steps_total} optimizer_steps={optimizer_steps_total}"
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
        f"epoch_time={format_duration(train_payload.get('epoch_seconds'))} "
        f"best={format_metric(best_val_metric)} best_epoch={best_epoch} "
        f"improved={epoch_improved} checkpoint={checkpoint_label}"
    )


def save_training_visualizations(history: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    if not history:
        return {}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[visualization:warning] export_failed={type(exc).__name__}:{exc}")
        return {}

    epochs = [int(item.get("epoch", index + 1)) for index, item in enumerate(history)]
    train_total = [float((item.get("train") or {}).get("total", np.nan)) for item in history]
    val_total = [
        float(((item.get("val") or {}).get("uv_mae") or {}).get("total", np.nan))
        for item in history
    ]
    baseline_val_total = [
        float(((item.get("val") or {}).get("baseline_uv_mae") or {}).get("total", np.nan))
        for item in history
    ]
    improvement_val_total = [
        float(((item.get("val") or {}).get("improvement_uv_mae") or {}).get("total", np.nan))
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
    if not np.isnan(baseline_val_total).all():
        axes[0, 0].plot(epochs, baseline_val_total, marker="o", label="SF3D baseline")
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

    if not np.isnan(improvement_val_total).all():
        axes[1, 0].plot(epochs, improvement_val_total, marker="o")
        axes[1, 0].axhline(0.0, color="black", linewidth=1, alpha=0.35)
        axes[1, 0].set_title("Refined Improvement Over SF3D")
        axes[1, 0].set_ylabel("baseline MAE - refined MAE")
    else:
        axes[1, 0].plot(epochs, samples_per_second, marker="o")
        axes[1, 0].set_title("Training Throughput")
        axes[1, 0].set_ylabel("samples/sec")
    axes[1, 0].set_xlabel("epoch")
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
                f"best baseline total: {format_metric(baseline_val_total[best_index]) if not np.isnan(baseline_val_total).all() else 'n/a'}",
                f"best improvement: {format_metric(improvement_val_total[best_index]) if not np.isnan(improvement_val_total).all() else 'n/a'}",
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
                f"<div>Last SF3D baseline total: <code>{format_metric(((latest.get('val') or {}).get('baseline_uv_mae') or {}).get('total'))}</code></div>",
                f"<div>Last improvement: <code>{format_metric(((latest.get('val') or {}).get('improvement_uv_mae') or {}).get('total'))}</code></div>",
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


def rgb_tensor_image(
    tensor: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> Image.Image:
    image = tensor_to_pil(tensor.detach().cpu()[:3]).convert("RGB")
    if mask is None:
        return image
    mask_np = mask.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()[..., None]
    image_np = np.asarray(image).astype(np.float32)
    background = np.array([20.0, 24.0, 30.0], dtype=np.float32)
    blended = image_np * mask_np + background * (1.0 - mask_np)
    return Image.fromarray(blended.clip(0.0, 255.0).round().astype(np.uint8), mode="RGB")


def delta_heatmap_image(
    reference: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> Image.Image:
    ref_np = reference.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()
    tgt_np = target.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()
    delta = np.abs(ref_np - tgt_np)
    if mask is not None:
        delta = delta * mask.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()
    delta = np.clip(delta * 3.5, 0.0, 1.0)
    heat = np.zeros((*delta.shape, 3), dtype=np.uint8)
    heat[..., 0] = (delta * 255.0).round().astype(np.uint8)
    heat[..., 1] = (np.sqrt(delta) * 170.0).round().astype(np.uint8)
    heat[..., 2] = ((1.0 - delta) * 45.0).round().astype(np.uint8)
    return Image.fromarray(heat, mode="RGB")


def preview_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
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


def draw_text_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (14, 18, 24),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    draw.text(
        (x0 + (x1 - x0 - text_width) / 2, y0 + (y1 - y0 - text_height) / 2),
        text,
        font=font,
        fill=fill,
    )


def compact_labeled_image(
    image: Image.Image,
    label: str,
    *,
    size: int = 176,
    header_height: int = 34,
) -> Image.Image:
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + header_height), (247, 249, 252))
    tile.paste(image, (0, header_height))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size - 1, header_height - 1), fill=(239, 243, 248))
    draw_text_centered(
        draw,
        (0, 0, size, header_height),
        label,
        font=preview_font(18, bold=True),
    )
    return tile


def labeled_tile(
    image: Image.Image,
    label: str,
    detail: str | None = None,
    *,
    size: int = 224,
) -> Image.Image:
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    header_height = 36 if detail else 24
    tile = Image.new("RGB", (image.width, image.height + header_height), (245, 247, 250))
    tile.paste(image, (0, header_height))
    draw = ImageDraw.Draw(tile)
    draw.text((6, 4), label, fill=(16, 18, 22))
    if detail:
        draw.text((6, 18), detail, fill=(82, 90, 102))
    return tile


def select_preview_view_indices(view_names: list[str], *, max_views: int = 4) -> list[int]:
    preferred_tokens = [
        "front_mid",
        "three_quarter_mid",
        "grazing_left",
        "thin_boundary_close",
        "front_high",
        "grazing_right",
        "top_oblique",
        "side_low",
    ]
    selected: list[int] = []
    for token in preferred_tokens:
        for index, name in enumerate(view_names):
            if index in selected:
                continue
            if str(name).startswith(token):
                selected.append(index)
                break
        if len(selected) >= max_views:
            return selected[:max_views]
    for index, _name in enumerate(view_names):
        if index not in selected:
            selected.append(index)
        if len(selected) >= max_views:
            break
    return selected[:max_views]


def preview_view_label(view_name: str) -> str:
    return str(view_name).split("__", 1)[0].replace("_", " ")


def save_preview_contact_sheet(
    output_dir: Path,
    *,
    validation_label: str,
    preview_paths: list[str],
) -> Path | None:
    if not preview_paths:
        return None
    images = [Image.open(path).convert("RGB") for path in preview_paths]
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    columns = 2 if len(images) > 1 else 1
    rows = math.ceil(len(images) / columns)
    header_height = 42
    gutter = 14
    canvas = Image.new(
        "RGB",
        (
            columns * tile_width + gutter * (columns + 1),
            header_height + rows * tile_height + gutter * (rows + 1),
        ),
        (10, 15, 22),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), f"{validation_label} validation previews", fill=(236, 242, 248))
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        x0 = gutter + col * (tile_width + gutter)
        y0 = header_height + gutter + row * (tile_height + gutter)
        if image.size != (tile_width, tile_height):
            image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        canvas.paste(image, (x0, y0))
    preview_dir = output_dir / "validation_previews" / validation_label
    preview_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet_path = preview_dir / "_contact_sheet.png"
    canvas.save(contact_sheet_path)
    return contact_sheet_path


def save_validation_preview(
    output_dir: Path,
    *,
    validation_label: str,
    object_id: str,
    baseline: torch.Tensor,
    refined: torch.Tensor,
    uv_albedo: torch.Tensor,
    uv_normal: torch.Tensor,
    target_roughness: torch.Tensor,
    target_metallic: torch.Tensor,
    confidence: torch.Tensor,
    view_features: torch.Tensor,
    view_masks: torch.Tensor,
    view_names: list[str],
    baseline_roughness_mae: float,
    baseline_metallic_mae: float,
    refined_roughness_mae: float,
    refined_metallic_mae: float,
) -> Path:
    preview_dir = output_dir / "validation_previews" / validation_label
    preview_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = select_preview_view_indices(view_names, max_views=1)
    preview_view_index = selected_indices[0] if selected_indices else 0
    preview_view = rgb_tensor_image(
        view_features[preview_view_index, 0:3],
        mask=view_masks[preview_view_index],
    )

    tile_size = 176
    row_label_width = 96
    gutter = 10
    title_height = 70
    columns = 4
    tile_width = tile_size
    tile_height = tile_size + 34
    canvas_width = row_label_width + columns * tile_width + (columns + 1) * gutter
    rows = [
        (
            "Input",
            [
                compact_labeled_image(preview_view, "SF3D", size=tile_size),
                compact_labeled_image(rgb_tensor_image(uv_albedo), "Albedo", size=tile_size),
                compact_labeled_image(rgb_tensor_image(uv_normal), "Normal", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(confidence), "Conf", size=tile_size),
            ],
        ),
        (
            "Rough",
            [
                compact_labeled_image(grayscale_rgb_image(baseline[0:1]), "SF3D", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(target_roughness), "GT", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(refined[0:1]), "Pred", size=tile_size),
                compact_labeled_image(delta_heatmap_image(refined[0:1], target_roughness), "Error", size=tile_size),
            ],
        ),
        (
            "Metal",
            [
                compact_labeled_image(grayscale_rgb_image(baseline[1:2]), "SF3D", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(target_metallic), "GT", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(refined[1:2]), "Pred", size=tile_size),
                compact_labeled_image(delta_heatmap_image(refined[1:2], target_metallic), "Error", size=tile_size),
            ],
        ),
    ]
    canvas_height = title_height + len(rows) * tile_height + (len(rows) + 1) * gutter
    canvas = Image.new("RGB", size=(canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    baseline_total = baseline_roughness_mae + baseline_metallic_mae
    refined_total = refined_roughness_mae + refined_metallic_mae
    improvement = baseline_total - refined_total
    title_font = preview_font(21, bold=True)
    detail_font = preview_font(15)
    draw.text((12, 10), str(object_id), font=title_font, fill=(8, 12, 18))
    draw.text(
        (12, 38),
        f"SF3D {baseline_total:.4f} | Pred {refined_total:.4f} | gain {improvement:+.4f}",
        font=detail_font,
        fill=(68, 76, 88),
    )
    row_font = preview_font(20, bold=True)
    for row_index, (row_label, row_tiles) in enumerate(rows):
        y0 = title_height + gutter + row_index * (tile_height + gutter)
        draw_text_centered(
            draw,
            (0, y0, row_label_width, y0 + tile_height),
            row_label,
            font=row_font,
            fill=(24, 32, 44),
        )
        for col_index, tile in enumerate(row_tiles):
            x0 = row_label_width + gutter + col_index * (tile_width + gutter)
            canvas.paste(tile, (x0, y0))
    safe_object_id = str(object_id).replace("/", "_").replace("\\", "_")
    output_path = preview_dir / f"{safe_object_id}.png"
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


def validation_milestone_index(validation_label: str) -> int | None:
    parts = str(validation_label).split("_")
    if len(parts) >= 2 and parts[0] == "progress":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def should_compute_render_proxy_validation(
    args: argparse.Namespace,
    validation_label: str,
) -> bool:
    interval = int(getattr(args, "render_proxy_validation_milestone_interval", 0) or 0)
    if interval <= 0:
        return False
    milestone = validation_milestone_index(validation_label)
    if milestone is None:
        return False
    return milestone > 0 and milestone % interval == 0


def psnr_from_mse(mse: float) -> float | None:
    if mse < 0.0 or math.isnan(mse):
        return None
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def finalize_render_proxy_metrics(store: dict[str, float]) -> dict[str, Any]:
    denom = max(float(store.get("denom", 0.0)), 1.0)
    baseline_mae = float(store.get("baseline_abs", 0.0) / denom)
    refined_mae = float(store.get("refined_abs", 0.0) / denom)
    baseline_mse = float(store.get("baseline_sq", 0.0) / denom)
    refined_mse = float(store.get("refined_sq", 0.0) / denom)
    return {
        "enabled": True,
        "available": bool(store.get("available_batches", 0.0) > 0.0),
        "batches": int(store.get("batches", 0.0)),
        "available_batches": int(store.get("available_batches", 0.0)),
        "samples": int(store.get("samples", 0.0)),
        "baseline_view_rm_mae": baseline_mae,
        "refined_view_rm_mae": refined_mae,
        "view_rm_mae_delta": baseline_mae - refined_mae,
        "baseline_proxy_rm_psnr": psnr_from_mse(baseline_mse),
        "refined_proxy_rm_psnr": psnr_from_mse(refined_mse),
        "proxy_rm_psnr_delta": (
            None
            if psnr_from_mse(baseline_mse) is None or psnr_from_mse(refined_mse) is None
            else psnr_from_mse(refined_mse) - psnr_from_mse(baseline_mse)
        ),
        "mode": "view_projected_rm_proxy",
    }


def update_render_proxy_metrics(
    store: dict[str, float],
    *,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    refined: torch.Tensor,
) -> None:
    if batch.get("view_targets") is None or batch.get("view_uvs") is None:
        return
    view_mask = batch["view_masks"].clamp(0.0, 1.0)
    supervision_mask = batch["has_effective_view_supervision"].to(view_mask.device)
    view_mask = view_mask * supervision_mask.view(-1, 1, 1, 1, 1)
    denom = float(view_mask.sum().detach().cpu().item()) * float(refined.shape[1])
    if denom <= 0.0:
        return
    target_views = batch["view_targets"]
    baseline_views = sample_uv_maps_to_view(baseline, batch["view_uvs"])
    refined_views = sample_uv_maps_to_view(refined, batch["view_uvs"])
    baseline_error = baseline_views - target_views
    refined_error = refined_views - target_views
    store["baseline_abs"] += float((baseline_error.abs() * view_mask).sum().detach().cpu().item())
    store["refined_abs"] += float((refined_error.abs() * view_mask).sum().detach().cpu().item())
    store["baseline_sq"] += float(((baseline_error ** 2) * view_mask).sum().detach().cpu().item())
    store["refined_sq"] += float(((refined_error ** 2) * view_mask).sum().detach().cpu().item())
    store["denom"] += denom
    store["available_batches"] += 1.0
    store["samples"] += float(refined.shape[0])


def finalize_residual_gate_diagnostics(store: dict[str, float]) -> dict[str, Any]:
    denom = max(float(store.get("denom", 0.0)), 1.0)
    case_count = max(float(store.get("cases", 0.0)), 1.0)
    return {
        "changed_pixel_rate": float(store.get("changed", 0.0) / denom),
        "unnecessary_change_rate": float(store.get("unnecessary", 0.0) / denom),
        "regression_rate": float(store.get("regression", 0.0) / denom),
        "safe_improvement_rate": float(store.get("safe_improvement", 0.0) / denom),
        "mean_residual_abs": float(store.get("residual_abs", 0.0) / denom),
        "case_count": int(store.get("cases", 0.0)),
        "mean_case_changed_pixel_rate": float(store.get("case_changed_rate", 0.0) / case_count),
        "mean_case_regression_rate": float(store.get("case_regression_rate", 0.0) / case_count),
    }


def update_residual_gate_diagnostics(
    store: dict[str, float],
    cases: list[dict[str, Any]],
    *,
    object_id: str,
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    margin: float,
) -> None:
    residual = (refined - baseline).abs()
    target_delta = (target - baseline).abs()
    baseline_error = (baseline - target).abs()
    refined_error = (refined - target).abs()
    confidence = confidence.clamp(0.0, 1.0)
    denom = float(confidence.sum().detach().cpu().item()) * float(refined.shape[0])
    if denom <= 0.0:
        return
    changed = (residual > margin).to(refined.dtype) * confidence
    unnecessary = ((target_delta < margin).to(refined.dtype) * changed)
    regression = ((refined_error > (baseline_error + 1e-6)).to(refined.dtype) * confidence)
    safe_improvement = ((refined_error < (baseline_error - 1e-6)).to(refined.dtype) * confidence)
    changed_sum = float(changed.sum().detach().cpu().item())
    regression_sum = float(regression.sum().detach().cpu().item())
    store["denom"] += denom
    store["changed"] += changed_sum
    store["unnecessary"] += float(unnecessary.sum().detach().cpu().item())
    store["regression"] += regression_sum
    store["safe_improvement"] += float(safe_improvement.sum().detach().cpu().item())
    store["residual_abs"] += float((residual * confidence).sum().detach().cpu().item())
    store["cases"] += 1.0
    store["case_changed_rate"] += changed_sum / denom
    store["case_regression_rate"] += regression_sum / denom
    cases.append(
        {
            "object_id": object_id,
            "changed_pixel_rate": changed_sum / denom,
            "unnecessary_change_rate": float(unnecessary.sum().detach().cpu().item()) / denom,
            "regression_rate": regression_sum / denom,
            "safe_improvement_rate": float(safe_improvement.sum().detach().cpu().item()) / denom,
            "mean_residual_abs": float((residual * confidence).sum().detach().cpu().item()) / denom,
        }
    )


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
        "edge_aware": 0.0,
        "boundary_bleed": 0.0,
        "gradient_preservation": 0.0,
        "metallic_classification": 0.0,
        "residual_safety": 0.0,
        "residual_gate_mean": 0.0,
        "residual_delta_abs": 0.0,
    }
    uv_mae = {"roughness": 0.0, "metallic": 0.0, "total": 0.0, "count": 0.0}
    baseline_uv_mae = {"roughness": 0.0, "metallic": 0.0, "total": 0.0}
    improvement = {
        "roughness": 0.0,
        "metallic": 0.0,
        "total": 0.0,
        "improved_samples": 0.0,
        "regressed_samples": 0.0,
        "unchanged_samples": 0.0,
    }
    group_store: dict[str, dict[str, float]] = {}
    baseline_group_store: dict[str, dict[str, float]] = {}
    improvement_group_store: dict[str, dict[str, float]] = {}
    steps = 0
    preview_paths: list[Path] = []
    preview_items: list[dict[str, Any]] = []
    render_proxy_enabled = should_compute_render_proxy_validation(args, validation_label)
    render_proxy_store: dict[str, float] = defaultdict(float)
    residual_gate_store: dict[str, float] = defaultdict(float)
    residual_gate_cases: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            if args.max_validation_batches > 0 and steps >= args.max_validation_batches:
                break
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
            if render_proxy_enabled and (
                args.render_proxy_validation_max_batches <= 0
                or render_proxy_store["batches"] < args.render_proxy_validation_max_batches
            ):
                render_proxy_store["batches"] += 1.0
                update_render_proxy_metrics(
                    render_proxy_store,
                    batch=batch,
                    baseline=baseline,
                    refined=refined,
                )

            per_sample_weight = confidence.flatten(1).sum(dim=1).clamp_min(1.0)
            baseline_roughness_mae = ((baseline[:, 0:1] - target[:, 0:1]).abs() * confidence).flatten(1).sum(dim=1) / per_sample_weight
            baseline_metallic_mae = ((baseline[:, 1:2] - target[:, 1:2]).abs() * confidence).flatten(1).sum(dim=1) / per_sample_weight
            roughness_mae = ((refined[:, 0:1] - target[:, 0:1]).abs() * confidence).flatten(1).sum(dim=1) / per_sample_weight
            metallic_mae = ((refined[:, 1:2] - target[:, 1:2]).abs() * confidence).flatten(1).sum(dim=1) / per_sample_weight
            baseline_total_mae = baseline_roughness_mae + baseline_metallic_mae
            total_mae = roughness_mae + metallic_mae
            improvement_roughness = baseline_roughness_mae - roughness_mae
            improvement_metallic = baseline_metallic_mae - metallic_mae
            improvement_total = baseline_total_mae - total_mae

            uv_mae["roughness"] += float(roughness_mae.sum().item())
            uv_mae["metallic"] += float(metallic_mae.sum().item())
            uv_mae["total"] += float(total_mae.sum().item())
            uv_mae["count"] += float(refined.shape[0])
            baseline_uv_mae["roughness"] += float(baseline_roughness_mae.sum().item())
            baseline_uv_mae["metallic"] += float(baseline_metallic_mae.sum().item())
            baseline_uv_mae["total"] += float(baseline_total_mae.sum().item())
            improvement["roughness"] += float(improvement_roughness.sum().item())
            improvement["metallic"] += float(improvement_metallic.sum().item())
            improvement["total"] += float(improvement_total.sum().item())
            improvement["improved_samples"] += float((improvement_total > 1e-6).sum().item())
            improvement["regressed_samples"] += float((improvement_total < -1e-6).sum().item())
            improvement["unchanged_samples"] += float((improvement_total.abs() <= 1e-6).sum().item())
            effective_view_samples = int(batch["has_effective_view_supervision"].sum().item())
            uv_mae.setdefault("effective_view_supervision_samples", 0.0)
            uv_mae["effective_view_supervision_samples"] += float(effective_view_samples)

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
                baseline_metric_kwargs = {
                    "total_mae": float(baseline_total_mae[item_index].item()),
                    "roughness_mae": float(baseline_roughness_mae[item_index].item()),
                    "metallic_mae": float(baseline_metallic_mae[item_index].item()),
                }
                improvement_metric_kwargs = {
                    "total_mae": float(improvement_total[item_index].item()),
                    "roughness_mae": float(improvement_roughness[item_index].item()),
                    "metallic_mae": float(improvement_metallic[item_index].item()),
                }
                update_group_metric_store(group_store, key=f"generator/{generator_id}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"source/{source_name}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"prior/{prior_name}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"tier/{supervision_tier}", **metric_kwargs)
                update_group_metric_store(baseline_group_store, key=f"generator/{generator_id}", **baseline_metric_kwargs)
                update_group_metric_store(baseline_group_store, key=f"source/{source_name}", **baseline_metric_kwargs)
                update_group_metric_store(baseline_group_store, key=f"prior/{prior_name}", **baseline_metric_kwargs)
                update_group_metric_store(baseline_group_store, key=f"tier/{supervision_tier}", **baseline_metric_kwargs)
                update_group_metric_store(improvement_group_store, key=f"generator/{generator_id}", **improvement_metric_kwargs)
                update_group_metric_store(improvement_group_store, key=f"source/{source_name}", **improvement_metric_kwargs)
                update_group_metric_store(improvement_group_store, key=f"prior/{prior_name}", **improvement_metric_kwargs)
                update_group_metric_store(improvement_group_store, key=f"tier/{supervision_tier}", **improvement_metric_kwargs)
                update_residual_gate_diagnostics(
                    residual_gate_store,
                    residual_gate_cases,
                    object_id=str(object_id),
                    baseline=baseline[item_index].detach(),
                    refined=refined[item_index].detach(),
                    target=target[item_index].detach(),
                    confidence=confidence[item_index].detach(),
                    margin=float(args.residual_safety_margin),
                )

                if len(preview_paths) < args.val_preview_samples:
                    preview_path = save_validation_preview(
                        output_dir,
                        validation_label=validation_label,
                        object_id=object_id,
                        baseline=baseline[item_index].detach().cpu(),
                        refined=refined[item_index].detach().cpu(),
                        uv_albedo=batch["uv_albedo"][item_index].detach().cpu(),
                        uv_normal=batch["uv_normal"][item_index].detach().cpu(),
                        target_roughness=batch["uv_target_roughness"][item_index].detach().cpu(),
                        target_metallic=batch["uv_target_metallic"][item_index].detach().cpu(),
                        confidence=batch["uv_target_confidence"][item_index].detach().cpu(),
                        view_features=batch["view_features"][item_index].detach().cpu(),
                        view_masks=batch["view_masks"][item_index].detach().cpu(),
                        view_names=list(batch["view_names"][item_index]),
                        baseline_roughness_mae=float(baseline_roughness_mae[item_index].item()),
                        baseline_metallic_mae=float(baseline_metallic_mae[item_index].item()),
                        refined_roughness_mae=float(roughness_mae[item_index].item()),
                        refined_metallic_mae=float(metallic_mae[item_index].item()),
                    )
                    preview_paths.append(preview_path)
                    baseline_total = float(baseline_total_mae[item_index].item())
                    refined_total = float(total_mae[item_index].item())
                    preview_items.append(
                        {
                            "path": str(preview_path.resolve()),
                            "object_id": object_id,
                            "generator_id": generator_id,
                            "source_name": source_name,
                            "prior_label": prior_name,
                            "baseline_total_mae": baseline_total,
                            "refined_total_mae": refined_total,
                            "improvement_total": baseline_total - refined_total,
                        }
                    )

    mean_losses = {key: value / max(steps, 1) for key, value in totals.items()}
    mean_uv_mae = {
        "roughness": uv_mae["roughness"] / max(uv_mae["count"], 1.0),
            "metallic": uv_mae["metallic"] / max(uv_mae["count"], 1.0),
        "total": uv_mae["total"] / max(uv_mae["count"], 1.0),
        "count": int(uv_mae["count"]),
    }
    sample_count = max(uv_mae["count"], 1.0)
    mean_baseline_uv_mae = {
        "roughness": baseline_uv_mae["roughness"] / sample_count,
        "metallic": baseline_uv_mae["metallic"] / sample_count,
        "total": baseline_uv_mae["total"] / sample_count,
        "count": int(uv_mae["count"]),
    }
    mean_improvement = {
        "roughness": improvement["roughness"] / sample_count,
        "metallic": improvement["metallic"] / sample_count,
        "total": improvement["total"] / sample_count,
        "improved_samples": int(improvement["improved_samples"]),
        "regressed_samples": int(improvement["regressed_samples"]),
        "unchanged_samples": int(improvement["unchanged_samples"]),
        "improvement_rate": improvement["improved_samples"] / sample_count,
        "regression_rate": improvement["regressed_samples"] / sample_count,
    }
    return {
        "loss": mean_losses,
        "uv_mae": mean_uv_mae,
        "baseline_uv_mae": mean_baseline_uv_mae,
        "improvement_uv_mae": mean_improvement,
        "group_metrics": finalize_group_metrics(group_store),
        "baseline_group_metrics": finalize_group_metrics(baseline_group_store),
        "improvement_group_metrics": finalize_group_metrics(improvement_group_store),
        "batches": int(steps),
        "max_validation_batches": int(args.max_validation_batches),
        "effective_view_supervision_samples": int(uv_mae.get("effective_view_supervision_samples", 0.0)),
        "effective_view_supervision_rate": float(uv_mae.get("effective_view_supervision_samples", 0.0) / max(uv_mae["count"], 1.0)),
        "view_consistency_enabled": bool(args.view_consistency_mode != "disabled" and args.view_consistency_weight > 0.0),
        "render_proxy_validation": (
            finalize_render_proxy_metrics(render_proxy_store)
            if render_proxy_enabled
            else {
                "enabled": False,
                "available": False,
                "reason": "disabled_or_not_a_render_proxy_milestone",
                "mode": "view_projected_rm_proxy",
            }
        ),
        "residual_gate_diagnostics": finalize_residual_gate_diagnostics(residual_gate_store),
        "residual_gate_cases": residual_gate_cases,
        "preview_paths": [str(path.resolve()) for path in preview_paths],
        "preview_items": preview_items,
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
    best_target: Path | None = None
    best_path = output_dir / "best.pt"
    if best_path.is_symlink():
        best_target = best_path.resolve()
    latest_target: Path | None = None
    latest_path = output_dir / "latest.pt"
    if latest_path.is_symlink():
        latest_target = latest_path.resolve()
    checkpoints = sorted([*output_dir.glob("epoch_*.pt"), *output_dir.glob("step_*.pt")])
    to_keep = {checkpoint_path.resolve() for checkpoint_path in checkpoints[-keep_last:]}
    if best_target is not None:
        to_keep.add(best_target)
    if latest_target is not None:
        to_keep.add(latest_target)
    for checkpoint_path in checkpoints:
        if checkpoint_path.resolve() not in to_keep:
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
        supervision_roles=args.train_supervision_roles,
        license_buckets=args.train_license_buckets,
        target_quality_tiers=args.train_target_quality_tiers,
        paper_splits=args.train_paper_splits,
        material_families=args.train_material_families,
        lighting_bank_ids=args.train_lighting_bank_ids,
        require_prior=args.train_require_prior,
        max_records=args.max_train_records,
        max_views_per_sample=args.train_view_sample_count,
        min_hard_views_per_sample=args.train_min_hard_views,
        randomize_view_subset=args.train_randomize_view_subset,
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
            supervision_roles=args.val_supervision_roles if frozen_val_manifest is None else None,
            license_buckets=args.val_license_buckets if frozen_val_manifest is None else None,
            target_quality_tiers=args.val_target_quality_tiers if frozen_val_manifest is None else None,
            paper_splits=args.val_paper_splits if frozen_val_manifest is None else None,
            material_families=args.val_material_families if frozen_val_manifest is None else None,
            lighting_bank_ids=args.val_lighting_bank_ids if frozen_val_manifest is None else None,
            require_prior=args.val_require_prior if frozen_val_manifest is None else None,
            max_records=None if frozen_val_manifest is None else None,
            max_views_per_sample=args.val_view_sample_count,
            min_hard_views_per_sample=args.val_min_hard_views,
            randomize_view_subset=False,
        )
        val_records, val_balance_summary = balance_validation_records(val_dataset.records, args)
        val_balance_summary["source"] = (
            "frozen_manifest" if frozen_val_manifest is not None else "live_manifest"
        )
        val_dataset.records = val_records
        val_loader = build_loader(
            val_dataset,
            batch_size=args.val_batch_size if args.val_batch_size > 0 else args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
    except RuntimeError:
        if not args.allow_empty_val:
            raise
        val_balance_summary = {
            "enabled": False,
            "key": str(args.val_balance_key),
            "original_records": 0,
            "selected_records": 0,
            "groups": {},
            "warnings": ["empty_validation_loader"],
        }

    data_state = {
        "epoch": epoch,
        "train": summarize_records(train_dataset.records),
        "val": summarize_records(val_dataset.records) if val_dataset is not None else {"records": 0},
        "val_balance": val_balance_summary,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader) if val_loader is not None else 0,
        "train_view_sample_count": int(args.train_view_sample_count),
        "val_view_sample_count": int(args.val_view_sample_count),
        "val_batch_size": int(args.val_batch_size if args.val_batch_size > 0 else args.batch_size),
        "max_validation_batches": int(args.max_validation_batches),
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
        f"batches={val_payload.get('batches')} samples={val_payload['uv_mae'].get('count')} "
        f"baseline={format_metric((val_payload.get('baseline_uv_mae') or {}).get('total'))} "
        f"uv_total={format_metric(val_payload['uv_mae']['total'])} "
        f"improve={format_metric((val_payload.get('improvement_uv_mae') or {}).get('total'))} "
        f"improve_rate={format_metric((val_payload.get('improvement_uv_mae') or {}).get('improvement_rate'), 4)} "
        f"roughness={format_metric(val_payload['uv_mae']['roughness'])} "
        f"metallic={format_metric(val_payload['uv_mae']['metallic'])} "
        f"proxy_view_delta={format_metric((val_payload.get('render_proxy_validation') or {}).get('view_rm_mae_delta'))} "
        f"res_change={format_metric((val_payload.get('residual_gate_diagnostics') or {}).get('changed_pixel_rate'), 4)} "
        f"res_reg={format_metric((val_payload.get('residual_gate_diagnostics') or {}).get('regression_rate'), 4)} "
        f"best={format_metric(best_val_metric)} best_epoch={best_epoch} improved={improved}"
    )

    if run is not None:
        preview_items = list(val_payload.get("preview_items", []))
        preview_contact_sheet = (
            save_preview_contact_sheet(
                output_dir,
                validation_label=validation_label,
                preview_paths=[str(item.get("path")) for item in preview_items if item.get("path")],
            )
            if bool(args.save_preview_contact_sheet)
            else None
        )
        if preview_contact_sheet is not None:
            val_payload["preview_contact_sheet"] = str(preview_contact_sheet.resolve())
        preview_images = []
        if wandb is not None:
            preview_images = [
                wandb.Image(
                    item["path"],
                    caption=(
                        f"{item.get('object_id')} | "
                        f"{item.get('baseline_total_mae', 0.0):.4f}"
                        f" -> {item.get('refined_total_mae', 0.0):.4f} "
                        f"(gain={item.get('improvement_total', 0.0):+.4f})"
                    ),
                )
                for item in preview_items[: args.wandb_val_preview_max]
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
            "val/baseline_uv_total_mae": val_payload["baseline_uv_mae"]["total"],
            "val/baseline_uv_roughness_mae": val_payload["baseline_uv_mae"]["roughness"],
            "val/baseline_uv_metallic_mae": val_payload["baseline_uv_mae"]["metallic"],
            "val/improvement_uv_total_mae": val_payload["improvement_uv_mae"]["total"],
            "val/improvement_uv_roughness_mae": val_payload["improvement_uv_mae"]["roughness"],
            "val/improvement_uv_metallic_mae": val_payload["improvement_uv_mae"]["metallic"],
            "val/improvement_rate": val_payload["improvement_uv_mae"]["improvement_rate"],
            "val/regression_rate": val_payload["improvement_uv_mae"]["regression_rate"],
            "val/effective_view_supervision_rate": val_payload.get("effective_view_supervision_rate", 0.0),
            "val/view_consistency_enabled": bool(val_payload.get("view_consistency_enabled", False)),
        }
        render_proxy = val_payload.get("render_proxy_validation") or {}
        if bool(render_proxy.get("enabled")):
            log_payload.update(
                {
                    "val/render_proxy/baseline_view_rm_mae": render_proxy.get("baseline_view_rm_mae"),
                    "val/render_proxy/refined_view_rm_mae": render_proxy.get("refined_view_rm_mae"),
                    "val/render_proxy/view_rm_mae_delta": render_proxy.get("view_rm_mae_delta"),
                    "val/render_proxy/baseline_rm_psnr": render_proxy.get("baseline_proxy_rm_psnr"),
                    "val/render_proxy/refined_rm_psnr": render_proxy.get("refined_proxy_rm_psnr"),
                    "val/render_proxy/rm_psnr_delta": render_proxy.get("proxy_rm_psnr_delta"),
                }
            )
        residual_diag = val_payload.get("residual_gate_diagnostics") or {}
        log_payload.update(
            {
                "val/residual_gate/changed_pixel_rate": residual_diag.get("changed_pixel_rate"),
                "val/residual_gate/unnecessary_change_rate": residual_diag.get("unnecessary_change_rate"),
                "val/residual_gate/regression_rate": residual_diag.get("regression_rate"),
                "val/residual_gate/safe_improvement_rate": residual_diag.get("safe_improvement_rate"),
                "val/residual_gate/mean_residual_abs": residual_diag.get("mean_residual_abs"),
            }
        )
        if train_total is not None:
            log_payload["train/epoch_total"] = train_total
        log_payload.update(flatten_for_logging(val_payload["loss"], prefix="val/loss"))
        sanitized_logs, skipped_logs = sanitize_log_dict(log_payload)
        if skipped_logs:
            print(f"[wandb:skip] validation_label={validation_label} keys={','.join(skipped_logs)}")
        if sanitized_logs:
            run.log(sanitized_logs, step=optimizer_step)
        if preview_images:
            preview_log_payload: dict[str, Any] = {"val/comparison_panels": preview_images}
            if (
                wandb is not None
                and preview_contact_sheet is not None
                and bool(args.wandb_log_preview_grid)
            ):
                preview_log_payload["val/preview_grid"] = wandb.Image(
                    str(preview_contact_sheet),
                    caption=f"{validation_label} preview grid",
                )
            run.log(preview_log_payload, step=optimizer_step)

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

    model_cfg = build_model_cfg(args)
    model = MaterialRefiner(model_cfg)
    model.to(device)
    if args.init_from_checkpoint is not None and args.resume is not None:
        raise ValueError("--init-from-checkpoint is model-only initialization and cannot be combined with --resume.")
    if args.init_from_checkpoint is not None:
        init_path = args.init_from_checkpoint
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            "[init] "
            f"checkpoint={short_path(init_path.resolve())} "
            f"source_epoch={checkpoint.get('epoch', 'n/a') if isinstance(checkpoint, dict) else 'n/a'} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            print(f"[init:warning] missing_keys={','.join(missing[:12])}")
        if unexpected:
            print(f"[init:warning] unexpected_keys={','.join(unexpected[:12])}")
    model_info = model_parameter_summary(model)
    runtime_info = system_runtime_info(args, device)
    runtime_info.update(
        {
            "amp_dtype": args.amp_dtype,
            "matmul_precision": args.matmul_precision,
            "allow_tf32": bool(args.allow_tf32),
            "optimizer": args.optimizer,
            "model_cfg": model_cfg,
            **model_info,
        }
    )
    print_system_runtime_report(args=args, device=device, runtime_info=runtime_info)
    if args.terminal_json_logs:
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
            "[resume] "
            f"checkpoint={short_path(args.resume.resolve())} "
            f"resume_epoch={int(checkpoint.get('epoch', 0))} "
            f"next_epoch={start_epoch} optimizer_step={optimizer_step}"
        )
        if args.terminal_json_logs:
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
    session_start_time = time.perf_counter()
    session_total_epochs = max(args.epochs - start_epoch + 1, 1)
    next_validation_milestone = max(int(train_state.get("next_validation_milestone", 1)), 1)
    startup_report_written = False

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
                compact_dataset_wandb_logs(data_state)
            )
            if skipped:
                print(f"[wandb:skip] epoch={epoch} dataset_keys={','.join(skipped)}")
            if run is not None and data_state_logs:
                run.log(data_state_logs, step=optimizer_step)
            if not startup_report_written:
                assert train_dataset is not None and train_loader is not None
                print_dataset_distribution(label="train", summary=data_state["train"], args=args)
                if _val_dataset is not None:
                    print_dataset_distribution(label="val", summary=data_state["val"], args=args)
                print_val_balance_summary(epoch, data_state.get("val_balance"))
                train_path_checks = print_record_path_checks(label="train", dataset=train_dataset)
                val_path_checks = (
                    print_record_path_checks(label="val", dataset=_val_dataset)
                    if _val_dataset is not None
                    else {}
                )
                train_probe = probe_dataloader(
                    label="train",
                    loader=train_loader,
                    args=args,
                    device=device,
                )
                val_probe = (
                    probe_dataloader(
                        label="val",
                        loader=val_loader,
                        args=args,
                        device=device,
                    )
                    if val_loader is not None
                    else {}
                )
                startup_checks = {
                    "runtime": runtime_info,
                    "model": model_info,
                    "data_state": data_state,
                    "train_path_checks": train_path_checks,
                    "val_path_checks": val_path_checks,
                    "train_probe": train_probe,
                    "val_probe": val_probe,
                }
                save_json(output_dir / "startup_checks.json", startup_checks)
                print_training_plan(
                    args=args,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    train_dataset=train_dataset,
                    model_info=model_info,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    start_epoch=start_epoch,
                )
                if run is not None:
                    run.config.update(
                        {"startup_checks": make_json_serializable(startup_checks)},
                        allow_val_change=True,
                    )
                startup_report_written = True
            else:
                print_data_reload_summary(epoch=epoch, data_state=data_state, args=args)

        assert train_dataset is not None and train_loader is not None
        train_records = len(train_dataset.records)
        val_records = 0
        if val_loader is not None:
            val_dataset = getattr(val_loader, "dataset", None)
            if val_dataset is not None and hasattr(val_dataset, "records"):
                val_records = len(val_dataset.records)
            elif val_dataset is not None:
                val_records = len(val_dataset)
        epoch_planned_samples = loader_planned_samples(train_loader, train_dataset)
        epoch_batch_steps_total = max(len(train_loader), 1)
        epoch_optimizer_steps_total = optimizer_steps_per_epoch(train_loader, args)
        print_epoch_start(
            args=args,
            epoch=epoch,
            train_records=train_records,
            val_records=val_records,
            planned_samples=epoch_planned_samples,
            batch_steps_total=epoch_batch_steps_total,
            optimizer_steps_total=epoch_optimizer_steps_total,
        )
        progress_bar = make_epoch_progress_bar(
            args=args,
            epoch=epoch,
            total_steps=epoch_optimizer_steps_total,
        )

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
            effective_view_samples = int(batch["has_effective_view_supervision"].sum().item())
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
            running["effective_view_supervision_samples"] += float(effective_view_samples)
            interval["effective_view_supervision_samples"] += float(effective_view_samples)

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
            epoch_optimizer_step = min(
                math.ceil(batch_index / max(args.grad_accumulation_steps, 1)),
                epoch_optimizer_steps_total,
            )
            update_epoch_progress_bar(
                progress_bar,
                epoch_optimizer_step=epoch_optimizer_step,
            )
            maybe_apply_warmup(
                optimizer,
                base_learning_rate=args.learning_rate,
                optimizer_step=optimizer_step,
                warmup_steps=args.warmup_steps,
            )

            if optimizer_step % args.log_every == 0:
                now = time.perf_counter()
                interval_seconds = max(now - interval_start_time, 1e-9)
                epoch_progress = min(epoch_optimizer_step / max(epoch_optimizer_steps_total, 1), 1.0)
                session_elapsed_seconds = max(now - session_start_time, 1e-9)
                session_progress = training_progress_fraction(
                    args=args,
                    epoch=epoch,
                    start_epoch=start_epoch,
                    session_total_epochs=session_total_epochs,
                    epoch_optimizer_step=epoch_optimizer_step,
                    epoch_optimizer_steps_total=epoch_optimizer_steps_total,
                    optimizer_step=optimizer_step,
                )
                interval_logs = {
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "global_batch_step": global_batch_step,
                    "lr": current_lr(optimizer),
                    "train/prior_dropout_probability": prior_dropout_probability,
                    "train/view_consistency_enabled": bool(args.view_consistency_mode != "disabled" and args.view_consistency_weight > 0.0),
                    "train/batches_per_log_window": interval_steps,
                    "train/samples_per_second": interval_examples / interval_seconds,
                    "train/seconds_per_batch": interval_seconds / max(interval_steps, 1),
                    "progress/epoch_index": epoch,
                    "progress/epoch_total": args.epochs,
                    "progress/batch_step": batch_index,
                    "progress/batch_steps_total": epoch_batch_steps_total,
                    "progress/epoch_step": epoch_optimizer_step,
                    "progress/epoch_steps_total": epoch_optimizer_steps_total,
                    "progress/global_step_total": args.max_train_steps
                    if args.max_train_steps > 0
                    else epoch_optimizer_steps_total * args.epochs,
                    "progress/epoch_progress": epoch_progress,
                    "progress/session_progress": session_progress,
                    "progress/session_elapsed_seconds": session_elapsed_seconds,
                    "progress/epoch_elapsed_seconds": max(now - epoch_start_time, 1e-9),
                    "progress/eta_epoch_seconds": estimate_remaining_seconds(
                        epoch_progress,
                        max(now - epoch_start_time, 1e-9),
                    ),
                    "progress/eta_total_seconds": estimate_remaining_seconds(
                        session_progress,
                        session_elapsed_seconds,
                    ),
                    "progress/train_records": train_records,
                    "progress/val_records": val_records,
                    "progress/planned_samples": epoch_planned_samples,
                }
                if interval_examples > 0:
                    interval_logs["train/effective_view_supervision_rate"] = interval["effective_view_supervision_samples"] / interval_examples
                if last_grad_norm is not None:
                    interval_logs["train/grad_norm"] = last_grad_norm
                for key, value in interval.items():
                    interval_logs[f"train/{key}"] = value / max(interval_steps, 1)
                interval_logs.update(cuda_memory_log(device))
                sanitized_logs, skipped_logs = sanitize_log_dict(interval_logs)
                if skipped_logs:
                    print(f"[log:skip] optimizer_step={optimizer_step} train_keys={','.join(skipped_logs)}")
                if sanitized_logs:
                    update_epoch_progress_bar(
                        progress_bar,
                        epoch_optimizer_step=epoch_optimizer_step,
                        payload=sanitized_logs,
                    )
                    if args.train_line_logs or progress_bar is None:
                        print_train_interval(sanitized_logs, device)
                    if args.terminal_json_logs:
                        print(json.dumps(sanitized_logs))
                    if run is not None:
                        wandb_logs, wandb_skipped_logs = sanitize_log_dict(
                            filter_train_wandb_logs(sanitized_logs)
                        )
                        if wandb_skipped_logs:
                            print(f"[wandb:skip] optimizer_step={optimizer_step} train_keys={','.join(wandb_skipped_logs)}")
                        if wandb_logs:
                            run.log(wandb_logs, step=optimizer_step)
                interval.clear()
                interval_steps = 0
                interval_examples = 0
                interval_start_time = now

            step_val_payload = None
            step_improved = False
            validation_label = None
            if val_loader is not None and args.validation_progress_milestones > 0:
                epoch_optimizer_step = min(
                    math.ceil(batch_index / max(args.grad_accumulation_steps, 1)),
                    epoch_optimizer_steps_total,
                )
                session_progress = training_progress_fraction(
                    args=args,
                    epoch=epoch,
                    start_epoch=start_epoch,
                    session_total_epochs=session_total_epochs,
                    epoch_optimizer_step=epoch_optimizer_step,
                    epoch_optimizer_steps_total=epoch_optimizer_steps_total,
                    optimizer_step=optimizer_step,
                )
                target_progress = next_validation_milestone / max(args.validation_progress_milestones, 1)
                if (
                    next_validation_milestone <= args.validation_progress_milestones
                    and session_progress + 1e-12 >= target_progress
                ):
                    validation_label = (
                        f"progress_{next_validation_milestone:03d}"
                        f"_of_{args.validation_progress_milestones:03d}"
                    )
                    next_validation_milestone += 1
            elif (
                val_loader is not None
                and args.validation_steps > 0
                and optimizer_step % args.validation_steps == 0
            ):
                validation_label = f"step_{optimizer_step:06d}"

            if validation_label is not None:
                clear_progress_bar(progress_bar)
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
                    validation_label=validation_label,
                )
                refresh_progress_bar(progress_bar)
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

        if progress_bar is not None:
            progress_bar.close()

        epoch_seconds = max(time.perf_counter() - epoch_start_time, 1e-9)
        train_metrics = {key: value / max(batch_steps, 1) for key, value in running.items()}
        train_metrics["prior_dropout_probability"] = prior_dropout_probability
        if epoch_examples > 0:
            train_metrics["effective_view_supervision_rate"] = running["effective_view_supervision_samples"] / epoch_examples
        train_metrics["view_consistency_enabled"] = bool(args.view_consistency_mode != "disabled" and args.view_consistency_weight > 0.0)
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
        if (
            val_loader is not None
            and args.validation_progress_milestones <= 0
            and args.validation_steps <= 0
            and epoch % args.eval_every == 0
        ):
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
            "next_validation_milestone": next_validation_milestone,
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
        if args.terminal_json_logs:
            print(json.dumps(epoch_payload))

        should_log_epoch_artifact = args.wandb_log_artifacts and args.wandb_artifact_policy in {"all"} or (
            args.wandb_log_artifacts
            and args.wandb_artifact_policy in {"best", "best_and_final"}
            and epoch_improved
        )
        if checkpoint_path is not None and run is not None and should_log_epoch_artifact:
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
                "[early_stop] "
                f"epoch={epoch} best_epoch={best_epoch} "
                f"best_val_metric={format_metric(best_val_metric)} "
                f"patience={args.early_stopping_patience}"
            )
            if args.terminal_json_logs:
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
            print(
                "[max_train_steps] "
                f"target={args.max_train_steps} optimizer_step={optimizer_step}"
            )
            if args.terminal_json_logs:
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
        if args.wandb_log_artifacts and args.wandb_artifact_policy in {"all", "final", "best_and_final"}:
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
