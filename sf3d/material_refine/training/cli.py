from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from sf3d.material_refine.manifest_quality import (
    DEFAULT_MAX_TARGET_PRIOR_IDENTITY_RATE_FOR_PAPER,
    DEFAULT_MIN_NONTRIVIAL_TARGET_COUNT_FOR_PAPER,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
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
    parser.add_argument("--disable-prior-source-embedding", type=parse_bool, default=False)
    parser.add_argument("--disable-no-prior-bootstrap", type=parse_bool, default=False)
    parser.add_argument("--disable-boundary-safety", type=parse_bool, default=False)
    parser.add_argument("--disable-change-gate", type=parse_bool, default=False)
    parser.add_argument("--disable-prior-safe-loss", type=parse_bool, default=False)
    parser.add_argument("--enable-residual-gate", type=parse_bool, default=False)
    parser.add_argument("--residual-gate-bias", type=float, default=-0.5)
    parser.add_argument("--min-residual-gate", type=float, default=0.05)
    parser.add_argument("--max-residual-gate", type=float, default=1.0)
    parser.add_argument("--prior-confidence-gate-strength", type=float, default=0.0)
    parser.add_argument("--enable-boundary-context", type=parse_bool, default=False)
    parser.add_argument("--boundary-context-strength", type=float, default=0.25)
    parser.add_argument("--enable-material-context", type=parse_bool, default=False)
    parser.add_argument("--material-context-classes", type=int, default=3)
    parser.add_argument("--material-delta-scale", type=float, default=0.06)
    parser.add_argument("--enable-render-consistency-gate", type=parse_bool, default=False)
    parser.add_argument("--render-gate-strength", type=float, default=0.25)
    parser.add_argument("--enable-prior-source-embedding", type=parse_bool, default=True)
    parser.add_argument("--enable-no-prior-bootstrap", type=parse_bool, default=True)
    parser.add_argument("--enable-boundary-safety", type=parse_bool, default=True)
    parser.add_argument("--enable-change-gate", type=parse_bool, default=True)
    parser.add_argument("--enable-material-aux-head", type=parse_bool, default=False)
    parser.add_argument("--enable-render-proxy-loss", type=parse_bool, default=False)
    parser.add_argument("--enable-dual-path-prior-init", type=parse_bool, default=False)
    parser.add_argument("--enable-domain-prior-calibration", type=parse_bool, default=False)
    parser.add_argument("--domain-feature-channels", type=int, default=8)
    parser.add_argument("--prior-feature-channels", type=int, default=16)
    parser.add_argument("--max-generator-embeddings", type=int, default=64)
    parser.add_argument("--max-source-embeddings", type=int, default=128)
    parser.add_argument("--enable-material-sensitive-view-encoder", type=parse_bool, default=False)
    parser.add_argument("--enable-hard-view-routing", type=parse_bool, default=False)
    parser.add_argument("--enable-tri-branch-fusion", type=parse_bool, default=False)
    parser.add_argument("--enable-boundary-safety-module", type=parse_bool, default=False)
    parser.add_argument("--boundary-safety-strength", type=float, default=0.35)
    parser.add_argument("--boundary-residual-suppression-strength", type=float, default=0.0)
    parser.add_argument("--view-uncertainty-residual-suppression-strength", type=float, default=0.0)
    parser.add_argument("--bleed-risk-residual-suppression-strength", type=float, default=0.0)
    parser.add_argument("--enable-material-topology-reasoning", type=parse_bool, default=False)
    parser.add_argument("--topology-feature-channels", type=int, default=16)
    parser.add_argument("--topology-patch-size", type=int, default=16)
    parser.add_argument("--topology-layers", type=int, default=2)
    parser.add_argument("--topology-heads", type=int, default=2)
    parser.add_argument("--topology-residual-suppression-strength", type=float, default=0.0)
    parser.add_argument("--enable-confidence-gated-trunk", type=parse_bool, default=False)
    parser.add_argument("--uncertainty-gate-strength", type=float, default=0.25)
    parser.add_argument("--residual-delta-init-std", type=float, default=1.0e-3)
    parser.add_argument("--trunk-uncertainty-init-bias", type=float, default=-1.0)
    parser.add_argument("--trunk-boundary-stability-init-bias", type=float, default=1.0)
    parser.add_argument("--enable-inverse-material-check", type=parse_bool, default=False)
    parser.add_argument("--inverse-check-strength", type=float, default=0.25)
    parser.add_argument("--enable-material-evidence-calibration", type=parse_bool, default=False)
    parser.add_argument("--material-evidence-channels", type=int, default=16)
    parser.add_argument("--material-evidence-strength", type=float, default=0.75)
    parser.add_argument("--enable-evidence-update-budget", type=parse_bool, default=False)
    parser.add_argument("--evidence-update-budget-strength", type=float, default=0.70)
    parser.add_argument("--evidence-update-budget-floor", type=float, default=0.04)
    parser.add_argument("--cuda-device-index", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--allow-tf32", type=parse_bool, default=True)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--torch-num-interop-threads", type=int, default=0)
    parser.add_argument(
        "--train-balance-mode",
        choices=[
            "auto",
            "none",
            "generator",
            "source",
            "prior",
            "prior_variant",
            "prior_quality",
            "training_role",
            "tier",
            "material",
            "generator_x_prior",
            "source_x_prior",
            "material_x_source_x_prior",
            "material_x_generator_x_prior",
        ],
        default="auto",
    )
    parser.add_argument("--train-balance-power", type=float, default=1.0)
    parser.add_argument("--train-samples-per-epoch", type=int, default=0)
    parser.add_argument(
        "--train-prior-variant-weights",
        type=str,
        default="near_gt_prior=1.0,mild_gap_prior=1.0,medium_gap_prior=1.0,large_gap_prior=1.0,no_prior_bootstrap=0.75",
    )
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
    parser.add_argument(
        "--enable-sampled-view-rm-loss",
        type=parse_bool,
        default=False,
        help="Alias for requiring sampled UV-to-view RM supervision without relying on stored view target PNGs.",
    )
    parser.add_argument(
        "--sampled-view-rm-loss-weight",
        type=float,
        default=0.0,
        help="Optional explicit weight for sampled UV-to-view RM loss; falls back to --view-consistency-weight.",
    )
    parser.add_argument("--edge-aware-weight", type=float, default=0.0)
    parser.add_argument("--edge-aware-epsilon", type=float, default=1e-3)
    parser.add_argument("--boundary-bleed-weight", type=float, default=0.0)
    parser.add_argument("--boundary-band-kernel", type=int, default=5)
    parser.add_argument("--gradient-preservation-weight", type=float, default=0.0)
    parser.add_argument("--metallic-classification-weight", type=float, default=0.0)
    parser.add_argument("--material-context-weight", type=float, default=0.0)
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
    parser.add_argument("--val-enable-lpips", type=parse_bool, default=True)
    parser.add_argument("--val-lpips-max-images", type=int, default=12)
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
    parser.add_argument("--val-preview-samples", type=int, default=30)
    parser.add_argument(
        "--val-preview-selection",
        choices=["balanced_by_variant", "effect_showcase", "first", "balanced"],
        default="balanced_by_variant",
        help="How validation preview examples are selected from each validation pass.",
    )
    parser.add_argument("--wandb-val-preview-max", type=int, default=30)
    parser.add_argument("--wandb-log-preview-grid", type=parse_bool, default=False)
    parser.add_argument("--save-preview-contact-sheet", type=parse_bool, default=False)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stopping-scope", choices=["epoch", "validation_event"], default="epoch")
    parser.add_argument(
        "--validation-selection-metric",
        choices=["uv_total", "uv_render_guarded", "gain_render_guarded"],
        default="gain_render_guarded",
    )
    parser.add_argument("--selection-view-rm-penalty", type=float, default=0.5)
    parser.add_argument("--selection-mse-penalty", type=float, default=0.5)
    parser.add_argument("--selection-psnr-penalty", type=float, default=0.25)
    parser.add_argument("--selection-residual-regression-penalty", type=float, default=0.1)
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
    args.domain_feature_channels = max(int(args.domain_feature_channels), 1)
    args.prior_feature_channels = max(int(args.prior_feature_channels), 1)
    args.max_generator_embeddings = max(int(args.max_generator_embeddings), 2)
    args.max_source_embeddings = max(int(args.max_source_embeddings), 2)
    if args.disable_prior_source_embedding:
        args.enable_prior_source_embedding = False
        args.enable_domain_prior_calibration = False
    if args.disable_no_prior_bootstrap:
        args.enable_no_prior_bootstrap = False
        args.enable_dual_path_prior_init = False
    if args.disable_boundary_safety:
        args.enable_boundary_safety = False
        args.enable_boundary_safety_module = False
    if args.disable_change_gate:
        args.enable_change_gate = False
        args.enable_residual_gate = False
    if args.enable_prior_source_embedding:
        args.enable_domain_prior_calibration = True
    if args.enable_no_prior_bootstrap:
        args.enable_dual_path_prior_init = True
    if args.enable_boundary_safety:
        args.enable_boundary_safety_module = True
    if args.enable_change_gate:
        args.enable_residual_gate = True
    if args.disable_prior_safe_loss:
        args.prior_consistency_weight = 0.0
        args.residual_safety_weight = 0.0
    args.max_residual_gate = max(0.0, min(1.0, float(args.max_residual_gate)))
    args.boundary_safety_strength = max(0.0, min(1.0, float(args.boundary_safety_strength)))
    args.boundary_residual_suppression_strength = max(
        0.0,
        min(1.0, float(args.boundary_residual_suppression_strength)),
    )
    args.topology_feature_channels = max(int(args.topology_feature_channels), 8)
    args.topology_patch_size = max(int(args.topology_patch_size), 4)
    args.topology_layers = max(int(args.topology_layers), 1)
    args.topology_heads = max(int(args.topology_heads), 1)
    args.uncertainty_gate_strength = max(0.0, min(1.0, float(args.uncertainty_gate_strength)))
    args.inverse_check_strength = max(0.0, min(1.0, float(args.inverse_check_strength)))
    args.material_evidence_channels = max(int(args.material_evidence_channels), 8)
    args.material_evidence_strength = max(0.0, min(1.0, float(args.material_evidence_strength)))
    args.evidence_update_budget_strength = max(
        0.0,
        min(1.0, float(args.evidence_update_budget_strength)),
    )
    args.evidence_update_budget_floor = max(
        0.0,
        min(0.5, float(args.evidence_update_budget_floor)),
    )
    args.save_every = max(int(args.save_every), 1)
    args.prior_dropout_prob = max(0.0, min(1.0, float(args.prior_dropout_prob)))
    if args.prior_dropout_start_prob is not None:
        args.prior_dropout_start_prob = max(0.0, min(1.0, float(args.prior_dropout_start_prob)))
    if args.prior_dropout_end_prob is not None:
        args.prior_dropout_end_prob = max(0.0, min(1.0, float(args.prior_dropout_end_prob)))
    args.prior_dropout_warmup_epochs = max(int(args.prior_dropout_warmup_epochs), 0)
    args.torch_num_threads = max(int(args.torch_num_threads), 0)
    args.torch_num_interop_threads = max(int(args.torch_num_interop_threads), 0)
    args.min_nontrivial_target_count_for_paper = max(int(args.min_nontrivial_target_count_for_paper), 0)
    args.train_target_quality_weights = parse_weight_map(args.train_target_quality_weights)
    args.train_difficulty_weights = parse_weight_map(args.train_difficulty_weights)
    args.train_failure_tag_weights = parse_weight_map(args.train_failure_tag_weights)
    args.train_prior_variant_weights = parse_weight_map(args.train_prior_variant_weights)
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
