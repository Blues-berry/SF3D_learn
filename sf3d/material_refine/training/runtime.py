from __future__ import annotations

import argparse
import base64
import html
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

# ---- pathlib checkpoint compatibility shim ----
# Some checkpoints were saved with newer pathlib internals, e.g. pathlib._local.
# Python 3.10 stdlib pathlib is a module, not a package, so torch.load may fail
# with: ModuleNotFoundError: No module named 'pathlib._local'.
import sys as _sys_pathlib_compat
import pathlib as _pathlib_compat
_sys_pathlib_compat.modules.setdefault("pathlib._local", _pathlib_compat)
# ---- end pathlib checkpoint compatibility shim ----

import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

_LPIPS_MODEL: Any | None = None
_LPIPS_DEVICE: str | None = None
_LPIPS_FAILURE: str | None = None

REPO_ROOT = Path(__file__).resolve().parents[3]
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
MATERIAL_CONTEXT_GLOSSY_THRESHOLD = 0.45


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
    parser.add_argument("--train-variant-loss-weights", type=str, default=None)
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
    parser.add_argument("--roughness-channel-weight", type=float, default=1.0)
    parser.add_argument("--metallic-channel-weight", type=float, default=1.0)
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
    parser.add_argument("--val-preview-samples", type=int, default=4)
    parser.add_argument(
        "--val-preview-selection",
        choices=["effect_showcase", "first", "balanced", "balanced_by_variant"],
        default="effect_showcase",
        help="How validation preview examples are selected from each validation pass.",
    )
    parser.add_argument(
        "--realrender-upload-policy",
        choices=["grouped_30_case", "summary_only", "disabled"],
        default="grouped_30_case",
    )
    parser.add_argument("--wandb-val-preview-max", type=int, default=8)
    parser.add_argument("--wandb-log-preview-grid", type=parse_bool, default=False)
    parser.add_argument("--save-preview-contact-sheet", type=parse_bool, default=False)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stopping-scope", choices=["epoch", "validation_event"], default="epoch")
    parser.add_argument(
        "--validation-selection-metric",
        choices=["uv_total", "uv_render_guarded", "gain_render_guarded", "variant_balanced_gain_render_guarded"],
        default="gain_render_guarded",
    )
    parser.add_argument("--selection-view-rm-penalty", type=float, default=0.5)
    parser.add_argument("--selection-mse-penalty", type=float, default=0.5)
    parser.add_argument("--selection-psnr-penalty", type=float, default=0.25)
    parser.add_argument("--selection-residual-regression-penalty", type=float, default=0.1)
    parser.add_argument("--selection-metric-near-gt-regression-multiplier", type=float, default=1.0)
    parser.add_argument("--selection-metric-withprior-regression-multiplier", type=float, default=1.0)
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
    args.roughness_channel_weight = max(float(args.roughness_channel_weight), 0.1)
    args.metallic_channel_weight = max(float(args.metallic_channel_weight), 0.1)
    args.selection_metric_near_gt_regression_multiplier = max(
        float(args.selection_metric_near_gt_regression_multiplier),
        1.0,
    )
    args.selection_metric_withprior_regression_multiplier = max(
        float(args.selection_metric_withprior_regression_multiplier),
        1.0,
    )
    args.train_target_quality_weights = parse_weight_map(args.train_target_quality_weights)
    args.train_difficulty_weights = parse_weight_map(args.train_difficulty_weights)
    args.train_failure_tag_weights = parse_weight_map(args.train_failure_tag_weights)
    args.train_prior_variant_weights = parse_weight_map(args.train_prior_variant_weights)
    args.train_variant_loss_weights = parse_weight_map(args.train_variant_loss_weights)
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


def material_context_loss(
    material_logits: torch.Tensor | None,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    if material_logits is None:
        return target.new_zeros(())
    class_count = int(material_logits.shape[1])
    if class_count < 2:
        return target.new_zeros(())
    roughness = target[:, 0:1].float()
    metallic = target[:, 1:2].float()
    labels = torch.zeros_like(roughness, dtype=torch.long)
    glossy_label = 1 if class_count > 1 else 0
    metal_label = min(2, class_count - 1)
    labels = torch.where(
        (metallic >= METALLIC_THRESHOLD),
        torch.full_like(labels, metal_label),
        labels,
    )
    labels = torch.where(
        (metallic < METALLIC_THRESHOLD) & (roughness < MATERIAL_CONTEXT_GLOSSY_THRESHOLD),
        torch.full_like(labels, glossy_label),
        labels,
    )
    per_texel = F.cross_entropy(material_logits.float(), labels[:, 0], reduction="none")
    weight = confidence[:, 0].float()
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
    grid = torch.where(torch.isfinite(view_uvs), view_uvs, torch.zeros_like(view_uvs)).clone()
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


def view_uv_valid_mask(view_uvs: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(view_uvs).all(dim=-1)
    in_range = (
        (view_uvs[..., 0] >= 0.0)
        & (view_uvs[..., 0] <= 1.0)
        & (view_uvs[..., 1] >= 0.0)
        & (view_uvs[..., 1] <= 1.0)
    )
    return (finite & in_range).to(view_uvs.dtype).unsqueeze(2)


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


def rm_channel_weight_tensor(
    args: argparse.Namespace,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    return like.new_tensor(
        [
            float(getattr(args, "roughness_channel_weight", 1.0)),
            float(getattr(args, "metallic_channel_weight", 1.0)),
        ]
    ).view(1, 2, 1, 1)


def variant_loss_weight_tensor(
    batch: dict[str, Any],
    args: argparse.Namespace,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    weights = getattr(args, "train_variant_loss_weights", {}) or {}
    variants = list(batch.get("prior_variant_type") or [])
    if not weights or not variants:
        size = int(batch["uv_albedo"].shape[0])
        return torch.ones(size, device=device, dtype=torch.float32)
    values = [
        float(weights.get(str(variant or "unknown"), 1.0))
        for variant in variants
    ]
    return torch.tensor(values, device=device, dtype=torch.float32).clamp_min(1.0e-6)


def weighted_total_variation_loss(
    tensor: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    loss_x = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    loss_y = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    per_sample = loss_x + loss_y
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)


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
    baseline = outputs.get("input_prior", outputs["baseline"])
    prior_confidence = batch.get("input_prior_confidence", batch["uv_prior_confidence"]).clamp(0.0, 1.0)

    refine_l1 = ((refined - target).abs() * confidence).sum() / confidence.sum().clamp_min(1.0)
    coarse_l1 = ((coarse - target).abs() * confidence).sum() / confidence.sum().clamp_min(1.0)
    target_prior_delta = (target - baseline).abs().mean(dim=1, keepdim=True)
    prior_safe_mask = (
        confidence
        * prior_confidence
        * (target_prior_delta <= float(args.residual_safety_margin)).to(refined.dtype)
    )
    prior_consistency = (
        (refined - baseline).abs() * prior_safe_mask
    ).sum() / prior_safe_mask.sum().clamp_min(1.0)
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
    material_context = (
        material_context_loss(outputs.get("material_logits"), target, confidence)
        if args.material_context_weight > 0.0
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
    residual_gate = outputs.get("change_gate", outputs.get("residual_gate"))
    residual_delta = outputs.get("delta_rm", outputs.get("residual_delta"))
    residual_gate_mean = (
        residual_gate.mean() if residual_gate is not None else refined.new_zeros(())
    )
    residual_delta_abs = (
        residual_delta.abs().mean() if residual_delta is not None else refined.new_zeros(())
    )
    view_uncertainty_gate = outputs.get("view_uncertainty_residual_gate_uv")
    bleed_risk_gate = outputs.get("bleed_risk_residual_gate_uv")
    topology_residual_gate = outputs.get("topology_residual_gate_uv")
    render_support_gate = outputs.get("render_support_gate")
    inverse_material_gate = outputs.get("inverse_material_gate_uv")
    residual_channel_gate = outputs.get("residual_channel_gate_uv")
    roughness_safety_gate = outputs.get("roughness_safety_gate_uv")
    metallic_safety_gate = outputs.get("metallic_safety_gate_uv")
    metallic_evidence = outputs.get("metallic_evidence_uv")
    metallic_cap_strength = outputs.get("metallic_cap_strength_uv")
    evidence_update_budget = outputs.get("evidence_update_budget_uv")
    evidence_update_support = outputs.get("evidence_update_support_uv")
    view_uncertainty_gate_mean = (
        view_uncertainty_gate.mean() if view_uncertainty_gate is not None else refined.new_ones(())
    )
    bleed_risk_gate_mean = (
        bleed_risk_gate.mean() if bleed_risk_gate is not None else refined.new_ones(())
    )
    topology_residual_gate_mean = (
        topology_residual_gate.mean() if topology_residual_gate is not None else refined.new_ones(())
    )
    render_support_gate_mean = (
        render_support_gate.mean() if render_support_gate is not None else refined.new_ones(())
    )
    inverse_material_gate_mean = (
        inverse_material_gate.mean() if inverse_material_gate is not None else refined.new_ones(())
    )
    residual_channel_gate_mean = (
        residual_channel_gate.mean() if residual_channel_gate is not None else refined.new_ones(())
    )
    roughness_safety_gate_mean = (
        roughness_safety_gate.mean() if roughness_safety_gate is not None else refined.new_ones(())
    )
    metallic_safety_gate_mean = (
        metallic_safety_gate.mean() if metallic_safety_gate is not None else refined.new_ones(())
    )
    metallic_evidence_mean = (
        metallic_evidence.mean() if metallic_evidence is not None else refined.new_zeros(())
    )
    metallic_cap_strength_mean = (
        metallic_cap_strength.mean() if metallic_cap_strength is not None else refined.new_zeros(())
    )
    evidence_update_budget_mean = (
        evidence_update_budget.mean()
        if evidence_update_budget is not None
        else refined.new_ones(())
    )
    evidence_update_support_mean = (
        evidence_update_support.mean()
        if evidence_update_support is not None
        else refined.new_ones(())
    )
    diagnostics = outputs.get("diagnostics") or {}

    def diagnostic_mean(name: str, fallback: torch.Tensor) -> torch.Tensor:
        value = diagnostics.get(name)
        if isinstance(value, torch.Tensor):
            return value.mean()
        return fallback

    change_gate_mean = diagnostic_mean("change_gate_mean", residual_gate_mean)
    mean_abs_delta = diagnostic_mean("mean_abs_delta", residual_delta_abs)
    prior_reliability_tensor = outputs.get("prior_reliability")
    prior_reliability_mean = diagnostic_mean(
        "prior_reliability_mean",
        prior_reliability_tensor.mean()
        if isinstance(prior_reliability_tensor, torch.Tensor)
        else prior_confidence.mean(),
    )
    boundary_delta_mean = diagnostic_mean(
        "boundary_delta_mean",
        refined.new_zeros(()),
    )

    sampled_view_rm_loss_enabled = bool(args.enable_sampled_view_rm_loss) or args.view_consistency_mode != "disabled"
    if sampled_view_rm_loss_enabled and batch.get("view_uvs") is not None:
        sampled_refined = sample_uv_maps_to_view(refined, batch["view_uvs"])
        view_mask = batch["view_masks"].clamp(0.0, 1.0)
        view_mask = view_mask * view_uv_valid_mask(batch["view_uvs"]).to(view_mask.device)
        supervision_mask = batch["has_effective_view_supervision"].to(view_mask.device)
        view_mask = view_mask * supervision_mask.view(-1, 1, 1, 1, 1)
        # Some generated canonical bundles store per-view RM PNGs as RGBA masks
        # with 0/1 byte RGB values, which become nearly black after normalization.
        # The reliable supervision source is the UV target plus UV correspondence.
        view_targets = sample_uv_maps_to_view(target, batch["view_uvs"])
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
        + max(float(args.view_consistency_weight), float(args.sampled_view_rm_loss_weight)) * view_consistency
        + args.edge_aware_weight * edge_aware
        + args.boundary_bleed_weight * boundary_bleed
        + args.gradient_preservation_weight * gradient_preservation
        + args.metallic_classification_weight * metallic_classification
        + args.material_context_weight * material_context
        + args.residual_safety_weight * residual_safety
    )
    return {
        "total": total,
        "loss_uv": refine_l1.detach(),
        "loss_prior_safe": (prior_consistency + residual_safety).detach(),
        "loss_boundary": boundary_bleed.detach(),
        "loss_gradient": (gradient_preservation + edge_aware).detach(),
        "refine_l1": refine_l1.detach(),
        "coarse_l1": coarse_l1.detach(),
        "prior_consistency": prior_consistency.detach(),
        "smoothness": smoothness.detach(),
        "view_consistency": view_consistency.detach(),
        "edge_aware": edge_aware.detach(),
        "boundary_bleed": boundary_bleed.detach(),
        "gradient_preservation": gradient_preservation.detach(),
        "metallic_classification": metallic_classification.detach(),
        "material_context": material_context.detach(),
        "residual_safety": residual_safety.detach(),
        "residual_gate_mean": residual_gate_mean.detach(),
        "residual_delta_abs": residual_delta_abs.detach(),
        "change_gate_mean": change_gate_mean.detach(),
        "mean_abs_delta": mean_abs_delta.detach(),
        "prior_reliability_mean": prior_reliability_mean.detach(),
        "boundary_delta_mean": boundary_delta_mean.detach(),
        "view_uncertainty_gate_mean": view_uncertainty_gate_mean.detach(),
        "bleed_risk_gate_mean": bleed_risk_gate_mean.detach(),
        "topology_residual_gate_mean": topology_residual_gate_mean.detach(),
        "render_support_gate_mean": render_support_gate_mean.detach(),
        "inverse_material_gate_mean": inverse_material_gate_mean.detach(),
        "residual_channel_gate_mean": residual_channel_gate_mean.detach(),
        "roughness_safety_gate_mean": roughness_safety_gate_mean.detach(),
        "metallic_safety_gate_mean": metallic_safety_gate_mean.detach(),
        "metallic_evidence_mean": metallic_evidence_mean.detach(),
        "metallic_cap_strength_mean": metallic_cap_strength_mean.detach(),
        "evidence_update_budget_mean": evidence_update_budget_mean.detach(),
        "evidence_update_support_mean": evidence_update_support_mean.detach(),
    }


def sample_balance_key(record: Any, mode: str) -> str:
    prior_variant_type = str(record.metadata.get("prior_variant_type", "unknown"))
    prior_quality_bin = str(record.metadata.get("prior_quality_bin", "unknown"))
    training_role = str(record.metadata.get("training_role", "unknown"))
    if mode == "auto":
        mode = "prior_variant" if prior_variant_type != "unknown" else "source_x_prior"
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
    if mode == "prior_variant":
        return prior_variant_type
    if mode == "prior_quality":
        return prior_quality_bin
    if mode == "training_role":
        return training_role
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
    prior_variant_type = str(record.metadata.get("prior_variant_type", "unknown"))
    if args.train_prior_variant_weights:
        weight *= float(args.train_prior_variant_weights.get(prior_variant_type, 1.0))
    weight *= float(record.metadata.get("sample_weight", 1.0) or 1.0)
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
    _ = data_state
    # Dataset constants and balance breakdowns are saved under dataset_state/*.json.
    # Keeping them out of W&B avoids dashboard clutter and makes the main charts
    # reflect model behavior instead of static manifest metadata.
    return {}


def filter_train_wandb_logs(logs: dict[str, Any]) -> dict[str, Any]:
    allowed_exact = {
        "optim/lr",
        "throughput/samples_per_second",
        "throughput/seconds_per_batch",
        "train/total",
        "train/refine_l1",
        "train/coarse_l1",
        "train/prior_consistency",
        "train/view_consistency",
        "train/grad_norm",
        "train/effective_view_supervision_rate",
    }
    filtered: dict[str, Any] = {}
    for key, value in logs.items():
        if key in allowed_exact:
            filtered[key] = value
    return filtered


def filter_validation_wandb_logs(logs: dict[str, Any]) -> dict[str, Any]:
    allowed_exact = {
        "best/selection_metric",
        "best/epoch",
        "val/gain_total",
        "val/object_level/regression_rate",
        "val/case_level/regression_rate",
    }
    return {
        key: value
        for key, value in logs.items()
        if key in allowed_exact
        or key in {
            "val/rm_proxy/view_mae/delta",
            "val/rm_proxy/view_mse/delta",
            "val/rm_proxy/view_psnr/delta",
        }
        or (key.startswith("val/by_variant/") and key.endswith("/gain_total"))
    }


def add_step_context(
    logs: dict[str, Any],
    *,
    epoch: int,
    optimizer_step: int,
    global_batch_step: int | None = None,
    learning_rate: float | None = None,
    progress_fraction: float | None = None,
) -> dict[str, Any]:
    enriched = dict(logs)
    if learning_rate is not None:
        enriched["optim/lr"] = float(learning_rate)
    return enriched


def configure_wandb_step_metrics(run: Any | None) -> None:
    if run is None or wandb is None:
        return
    try:
        for metric_key in (
            "optim/lr",
            "throughput/samples_per_second",
            "throughput/seconds_per_batch",
            "train/total",
            "train/refine_l1",
            "train/coarse_l1",
            "train/prior_consistency",
            "train/view_consistency",
            "train/grad_norm",
            "train/effective_view_supervision_rate",
            "val/input_prior_total_mae",
            "val/refined_total_mae",
            "val/gain_total",
            "val/rm_proxy/view_mae/baseline",
            "val/rm_proxy/view_mae/refined",
            "val/rm_proxy/view_mae/delta",
            "val/rm_proxy/view_mse/baseline",
            "val/rm_proxy/view_mse/refined",
            "val/rm_proxy/view_mse/delta",
            "val/rm_proxy/view_psnr/baseline",
            "val/rm_proxy/view_psnr/refined",
            "val/rm_proxy/view_psnr/delta",
            "val/object_level/avg_improvement_total",
            "val/object_level/regression_rate",
            "val/case_level/avg_improvement_total",
            "val/case_level/regression_rate",
            "best/selection_metric",
            "best/epoch",
        ):
            wandb.define_metric(metric_key)
    except Exception as exc:  # noqa: BLE001 - W&B metric setup should not block training.
        print(f"[wandb:warning] define_metric_failed={type(exc).__name__}: {exc}")


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
        "enable_prior_source_embedding": bool(args.enable_prior_source_embedding),
        "enable_no_prior_bootstrap": bool(args.enable_no_prior_bootstrap),
        "enable_boundary_safety": bool(args.enable_boundary_safety),
        "enable_change_gate": bool(args.enable_change_gate),
        "enable_material_aux_head": bool(args.enable_material_aux_head),
        "enable_render_proxy_loss": bool(args.enable_render_proxy_loss),
        "enable_residual_gate": bool(args.enable_residual_gate),
        "residual_gate_bias": float(args.residual_gate_bias),
        "min_residual_gate": float(args.min_residual_gate),
        "max_residual_gate": float(args.max_residual_gate),
        "prior_confidence_gate_strength": float(args.prior_confidence_gate_strength),
        "enable_boundary_context": bool(args.enable_boundary_context),
        "boundary_context_strength": float(args.boundary_context_strength),
        "enable_material_context": bool(args.enable_material_context),
        "material_context_classes": int(args.material_context_classes),
        "material_delta_scale": float(args.material_delta_scale),
        "enable_render_consistency_gate": bool(args.enable_render_consistency_gate),
        "render_gate_strength": float(args.render_gate_strength),
        "enable_dual_path_prior_init": bool(args.enable_dual_path_prior_init),
        "enable_domain_prior_calibration": bool(args.enable_domain_prior_calibration),
        "domain_feature_channels": int(args.domain_feature_channels),
        "prior_feature_channels": int(args.prior_feature_channels),
        "max_generator_embeddings": int(args.max_generator_embeddings),
        "max_source_embeddings": int(args.max_source_embeddings),
        "enable_material_sensitive_view_encoder": bool(args.enable_material_sensitive_view_encoder),
        "enable_hard_view_routing": bool(args.enable_hard_view_routing),
        "enable_tri_branch_fusion": bool(args.enable_tri_branch_fusion),
        "enable_boundary_safety_module": bool(args.enable_boundary_safety_module),
        "boundary_safety_strength": float(args.boundary_safety_strength),
        "boundary_residual_suppression_strength": float(
            args.boundary_residual_suppression_strength
        ),
        "view_uncertainty_residual_suppression_strength": float(
            args.view_uncertainty_residual_suppression_strength
        ),
        "bleed_risk_residual_suppression_strength": float(
            args.bleed_risk_residual_suppression_strength
        ),
        "enable_material_topology_reasoning": bool(args.enable_material_topology_reasoning),
        "topology_feature_channels": int(args.topology_feature_channels),
        "topology_patch_size": int(args.topology_patch_size),
        "topology_layers": int(args.topology_layers),
        "topology_heads": int(args.topology_heads),
        "topology_residual_suppression_strength": float(
            args.topology_residual_suppression_strength
        ),
        "enable_confidence_gated_trunk": bool(args.enable_confidence_gated_trunk),
        "uncertainty_gate_strength": float(args.uncertainty_gate_strength),
        "residual_delta_init_std": float(args.residual_delta_init_std),
        "trunk_uncertainty_init_bias": float(args.trunk_uncertainty_init_bias),
        "trunk_boundary_stability_init_bias": float(args.trunk_boundary_stability_init_bias),
        "enable_inverse_material_check": bool(args.enable_inverse_material_check),
        "inverse_check_strength": float(args.inverse_check_strength),
        "enable_material_evidence_calibration": bool(args.enable_material_evidence_calibration),
        "material_evidence_channels": int(args.material_evidence_channels),
        "material_evidence_strength": float(args.material_evidence_strength),
        "enable_evidence_update_budget": bool(args.enable_evidence_update_budget),
        "evidence_update_budget_strength": float(args.evidence_update_budget_strength),
        "evidence_update_budget_floor": float(args.evidence_update_budget_floor),
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
    projected_view_targets = None
    if batch.get("view_uvs") is not None:
        try:
            projected_view_targets = sample_uv_maps_to_view(target, batch["view_uvs"])
        except Exception:
            projected_view_targets = None
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
        "view_targets_png_raw": tensor_probe_stats(batch.get("view_targets")),
        "view_targets_projected_uv": tensor_probe_stats(projected_view_targets),
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
            "view_targets_png_raw",
            "view_targets_projected_uv",
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
    print(
        "[preflight:start] "
        f"device={device} train_manifest={args.train_manifest} "
        f"audit_records={args.preflight_audit_records}",
        flush=True,
    )
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
            print(
                "[preflight:audit:start] "
                f"manifest={Path(args.train_manifest)} "
                f"max_records={int(args.preflight_audit_records)}",
                flush=True,
            )
            manifest_audit = audit_manifest(
                Path(args.train_manifest),
                max_records=int(args.preflight_audit_records),
                identity_warning_threshold=float(args.target_prior_identity_warning_threshold),
                max_target_prior_identity_rate_for_paper=float(args.max_target_prior_identity_rate_for_paper),
                min_nontrivial_target_count_for_paper=int(args.min_nontrivial_target_count_for_paper),
                fast=True,
            )
            manifest_summary = manifest_audit.get("summary") or {}
            manifest_audit_path = Path(args.output_dir) / "preflight_manifest_audit.json"
            save_json(manifest_audit_path, manifest_audit)
            print(
                "[preflight:audit:done] "
                f"records={manifest_summary.get('records')} "
                f"paper_eligible={manifest_summary.get('paper_stage_eligible_records')}",
                flush=True,
            )
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
            elif (
                (args.view_consistency_mode != "disabled" or args.enable_sampled_view_rm_loss)
                and max(float(args.view_consistency_weight), float(args.sampled_view_rm_loss_weight)) > 0.0
                and effective_view_rate <= 0.0
            ):
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
        "enable_sampled_view_rm_loss": bool(args.enable_sampled_view_rm_loss),
        "sampled_view_rm_loss_weight": float(args.sampled_view_rm_loss_weight),
        "edge_aware_weight": float(args.edge_aware_weight),
        "boundary_bleed_weight": float(args.boundary_bleed_weight),
        "boundary_band_kernel": int(args.boundary_band_kernel),
        "gradient_preservation_weight": float(args.gradient_preservation_weight),
        "metallic_classification_weight": float(args.metallic_classification_weight),
        "material_context_weight": float(args.material_context_weight),
        "residual_safety_weight": float(args.residual_safety_weight),
        "r_module_v2": {
            "enable_dual_path_prior_init": bool(args.enable_dual_path_prior_init),
            "enable_domain_prior_calibration": bool(args.enable_domain_prior_calibration),
            "enable_boundary_context": bool(args.enable_boundary_context),
            "boundary_context_strength": float(args.boundary_context_strength),
            "enable_material_context": bool(args.enable_material_context),
            "material_context_classes": int(args.material_context_classes),
            "material_delta_scale": float(args.material_delta_scale),
            "enable_render_consistency_gate": bool(args.enable_render_consistency_gate),
            "render_gate_strength": float(args.render_gate_strength),
            "enable_material_sensitive_view_encoder": bool(args.enable_material_sensitive_view_encoder),
            "enable_hard_view_routing": bool(args.enable_hard_view_routing),
            "enable_tri_branch_fusion": bool(args.enable_tri_branch_fusion),
            "enable_boundary_safety_module": bool(args.enable_boundary_safety_module),
            "max_residual_gate": float(args.max_residual_gate),
            "boundary_residual_suppression_strength": float(
                args.boundary_residual_suppression_strength
            ),
            "view_uncertainty_residual_suppression_strength": float(
                args.view_uncertainty_residual_suppression_strength
            ),
            "bleed_risk_residual_suppression_strength": float(
                args.bleed_risk_residual_suppression_strength
            ),
            "enable_material_topology_reasoning": bool(args.enable_material_topology_reasoning),
            "topology_residual_suppression_strength": float(
                args.topology_residual_suppression_strength
            ),
            "enable_confidence_gated_trunk": bool(args.enable_confidence_gated_trunk),
            "enable_inverse_material_check": bool(args.enable_inverse_material_check),
        },
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
        f"loss={format_metric(payload.get('train/total'))} "
        f"refine={format_metric(payload.get('train/refine_l1'))} "
        f"coarse={format_metric(payload.get('train/coarse_l1'))} "
        f"prior={format_metric(payload.get('train/prior_consistency'))} "
        f"view={format_metric(payload.get('train/view_consistency'))} "
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
                f"coarse={format_metric(payload.get('train/coarse_l1'), 4)}",
                f"prior={format_metric(payload.get('train/prior_consistency'), 4)}",
                f"view={format_metric(payload.get('train/view_consistency'), 4)}",
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
    latest_validation = epoch_payload.get("latest_validation") or {}
    val_mae = (
        (val_payload.get("uv_mae") or {}).get("total")
        if val_payload
        else latest_validation.get("val_total")
    )
    selection_metric = (
        (val_payload.get("selection_metric") or {}).get("selection_metric")
        if val_payload
        else latest_validation.get("selection_metric")
    )
    train_payload = epoch_payload.get("train") or {}
    checkpoint_label = str(checkpoint_path) if checkpoint_path is not None else "not_saved"
    print(
        "[epoch] "
        f"{epoch_payload.get('epoch')}/{args.epochs} step={epoch_payload.get('optimizer_step')} "
        f"train_total={format_metric(train_payload.get('total'))} "
        f"val_uv_total={format_metric(val_mae)} "
        f"val_select={format_metric(selection_metric)} "
        f"epoch_time={format_duration(train_payload.get('epoch_seconds'))} "
        f"best_select={format_metric(best_val_metric)} best_epoch={best_epoch} "
        f"improved={epoch_improved} checkpoint={checkpoint_label}"
    )


def validation_event_sort_key(path: Path) -> tuple[int, int, str]:
    label = path.stem
    parts = label.split("_")
    if len(parts) >= 2 and parts[0] == "progress":
        try:
            return (0, int(parts[1]), label)
        except ValueError:
            pass
    if len(parts) >= 2 and parts[0] == "epoch":
        try:
            return (1, int(parts[1]), label)
        except ValueError:
            pass
    return (2, 0, label)


def load_validation_events(output_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    validation_dir = output_dir / "validation"
    if not validation_dir.exists():
        return events
    for path in sorted(validation_dir.glob("*.json"), key=validation_event_sort_key):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report export must not block training.
            print(f"[visualization:warning] validation_event_read_failed={path.name}:{type(exc).__name__}:{exc}")
            continue
        render_proxy = payload.get("render_proxy_validation") or {}
        improvement = payload.get("improvement_uv_mae") or {}
        case_level = payload.get("case_level") or {}
        selection = payload.get("selection_metric") or {}
        events.append(
            {
                "label": payload.get("validation_label", path.stem),
                "epoch": payload.get("epoch"),
                "optimizer_step": payload.get("optimizer_step"),
                "uv_total": (payload.get("uv_mae") or {}).get("total"),
                "uv_roughness": (payload.get("uv_mae") or {}).get("roughness"),
                "uv_metallic": (payload.get("uv_mae") or {}).get("metallic"),
                "input_prior_total": (
                    payload.get("input_prior_uv_mae")
                    or payload.get("baseline_uv_mae")
                    or {}
                ).get("total"),
                "uv_gain": improvement.get("total"),
                "sample_regression_rate": improvement.get("regression_rate"),
                "case_avg_gain": case_level.get("avg_improvement_total"),
                "case_regression_rate": case_level.get("regression_rate"),
                "rm_proxy_view_mae_delta": render_proxy.get("view_rm_mae_delta"),
                "rm_proxy_view_mse_delta": render_proxy.get("proxy_rm_mse_delta"),
                "rm_proxy_view_psnr_delta": render_proxy.get("proxy_rm_psnr_delta"),
                "selection_metric": selection.get("selection_metric"),
                "selection_mode": selection.get("mode"),
                "path": str(path.resolve()),
            }
        )
    return events


def maybe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def maybe_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def inline_png_data_uri(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_latest_validation_payload(validation_events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not validation_events:
        return None
    latest_path = validation_events[-1].get("path")
    if not latest_path:
        return None
    try:
        return json.loads(Path(str(latest_path)).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report export must not block training.
        print(f"[visualization:warning] latest_validation_read_failed={type(exc).__name__}:{exc}")
        return None


def summarize_latest_validation(
    history: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
) -> dict[str, Any]:
    latest_epoch = history[-1] if history else {}
    latest_epoch_val = latest_epoch.get("val") or {}
    if latest_epoch_val:
        improvement_payload = latest_epoch_val.get("improvement_uv_mae") or {}
        baseline_payload = (
            latest_epoch_val.get("input_prior_uv_mae")
            or latest_epoch_val.get("baseline_uv_mae")
            or {}
        )
        return {
            "source": "epoch_validation",
            "label": f"epoch_{int(latest_epoch.get('epoch', 0)):03d}",
            "epoch": latest_epoch.get("epoch"),
            "optimizer_step": latest_epoch.get("optimizer_step"),
            "val_total": (latest_epoch_val.get("uv_mae") or {}).get("total"),
            "input_prior_total": baseline_payload.get("total"),
            "improvement_total": improvement_payload.get("total"),
            "selection_metric": (latest_epoch_val.get("selection_metric") or {}).get("selection_metric"),
        }
    if validation_events:
        latest_event = validation_events[-1]
        return {
            "source": "validation_event",
            "label": latest_event.get("label"),
            "epoch": latest_event.get("epoch"),
            "optimizer_step": latest_event.get("optimizer_step"),
            "val_total": latest_event.get("uv_total"),
            "input_prior_total": latest_event.get("input_prior_total"),
            "improvement_total": latest_event.get("uv_gain"),
            "selection_metric": latest_event.get("selection_metric"),
        }
    return {
        "source": "missing",
        "label": None,
        "epoch": latest_epoch.get("epoch"),
        "optimizer_step": latest_epoch.get("optimizer_step"),
        "val_total": None,
        "input_prior_total": None,
        "improvement_total": None,
        "selection_metric": None,
    }


def build_variant_summary_rows(
    validation_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not validation_payload:
        return []
    group_metrics = validation_payload.get("group_metrics") or {}
    baseline_group_metrics = validation_payload.get("baseline_group_metrics") or {}
    improvement_group_metrics = validation_payload.get("improvement_group_metrics") or {}
    rows = []
    for group_key, metrics in sorted(group_metrics.items()):
        if not str(group_key).startswith("prior_variant_type/"):
            continue
        variant = str(group_key).split("/", 1)[1]
        rows.append(
            {
                "variant": variant,
                "input_prior_total_mae": (
                    (baseline_group_metrics.get(group_key) or {}).get("total_mae")
                ),
                "refined_total_mae": metrics.get("total_mae"),
                "gain_total": (
                    (improvement_group_metrics.get(group_key) or {}).get("total_mae")
                ),
            }
        )
    return rows


def write_training_overview(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    train_state: dict[str, Any],
    validation_events: list[dict[str, Any]],
    visualization_paths: dict[str, str],
) -> Path | None:
    if not history and not validation_events:
        return None
    latest_train = (history[-1].get("train") or {}) if history else {}
    latest_validation = summarize_latest_validation(history, validation_events)
    latest_validation_payload = load_latest_validation_payload(validation_events)
    latest_render_proxy = (latest_validation_payload or {}).get("render_proxy_validation") or {}
    latest_object_level = (latest_validation_payload or {}).get("object_level") or {}
    latest_case_level = (latest_validation_payload or {}).get("case_level") or {}
    variant_rows = build_variant_summary_rows(latest_validation_payload)
    evidence_curve_uri = inline_png_data_uri(
        Path(str(visualization_paths["training_evidence_curves"]))
        if "training_evidence_curves" in visualization_paths
        else None
    )
    train_curve_uri = inline_png_data_uri(
        Path(str(visualization_paths["training_curves"]))
        if "training_curves" in visualization_paths
        else None
    )
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    best_event = None
    if validation_events:
        best_event = max(
            validation_events,
            key=lambda item: maybe_float(item.get("uv_gain")),
        )
    variant_rows_html = [
        "<tr>"
        f"<td>{html.escape(str(row.get('variant', 'unknown')))}</td>"
        f"<td>{format_metric(row.get('input_prior_total_mae'))}</td>"
        f"<td>{format_metric(row.get('refined_total_mae'))}</td>"
        f"<td>{format_metric(row.get('gain_total'))}</td>"
        "</tr>"
        for row in variant_rows
    ]
    overview_path = output_dir / "training_overview.html"
    overview_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Material Refiner Training Overview</title>",
                (
                    "<style>"
                    "body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}"
                    ".wrap{max-width:1320px;margin:auto}"
                    ".card{background:#18202b;border-radius:12px;padding:18px;margin:16px 0}"
                    ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}"
                    ".metric{font-size:13px;color:#b9e6ff;margin-bottom:6px}"
                    "img{max-width:100%;border-radius:8px;background:#fff}"
                    "table{border-collapse:collapse;width:100%;font-size:13px}"
                    "th,td{border-bottom:1px solid #2d3748;padding:8px;text-align:left}"
                    "th{color:#b9e6ff}"
                    "code{color:#b9e6ff}"
                    "ul{margin:8px 0 0 20px;padding:0}"
                    "</style>"
                ),
                "</head><body><div class='wrap'>",
                "<h1>Material Refiner Training Overview</h1>",
                "<div class='card'><div class='grid'>",
                f"<div><div class='metric'>Run</div><div><code>{html.escape(str(args.tracker_run_name or output_dir.name))}</code></div></div>",
                f"<div><div class='metric'>Project / Group</div><div><code>{html.escape(str(args.tracker_project_name))}</code> / <code>{html.escape(str(args.tracker_group))}</code></div></div>",
                f"<div><div class='metric'>Best Checkpoint</div><div><code>{html.escape(str(best_path.resolve() if best_path.exists() else best_path))}</code></div></div>",
                f"<div><div class='metric'>Latest Checkpoint</div><div><code>{html.escape(str(latest_path.resolve() if latest_path.exists() else latest_path))}</code></div></div>",
                f"<div><div class='metric'>Best Epoch</div><div><code>{html.escape(str(train_state.get('best_epoch')))}</code></div></div>",
                f"<div><div class='metric'>Best Selection Metric</div><div><code>{format_metric(train_state.get('best_val_metric'))}</code></div></div>",
                "</div></div>",
                "<div class='card'><h2>Core Summary</h2><div class='grid'>",
                f"<div><div class='metric'>Latest Train Total</div><div><code>{format_metric(latest_train.get('total'))}</code></div></div>",
                f"<div><div class='metric'>Latest Validation Source</div><div><code>{html.escape(str(latest_validation.get('source')))}</code></div></div>",
                f"<div><div class='metric'>Latest Validation Label</div><div><code>{html.escape(str(latest_validation.get('label')))}</code></div></div>",
                f"<div><div class='metric'>Latest UV Total</div><div><code>{format_metric(latest_validation.get('val_total'))}</code></div></div>",
                f"<div><div class='metric'>Latest Input Prior Total</div><div><code>{format_metric(latest_validation.get('input_prior_total'))}</code></div></div>",
                f"<div><div class='metric'>Latest UV Gain</div><div><code>{format_metric(latest_validation.get('improvement_total'))}</code></div></div>",
                f"<div><div class='metric'>Latest RM Proxy View MAE Delta</div><div><code>{format_metric(latest_render_proxy.get('view_rm_mae_delta'))}</code></div></div>",
                f"<div><div class='metric'>Latest RM Proxy View MSE Delta</div><div><code>{format_metric(latest_render_proxy.get('proxy_rm_mse_delta'))}</code></div></div>",
                f"<div><div class='metric'>Latest RM Proxy View PSNR Delta</div><div><code>{format_metric(latest_render_proxy.get('proxy_rm_psnr_delta'))}</code></div></div>",
                f"<div><div class='metric'>Object Regression Rate</div><div><code>{format_metric(latest_object_level.get('regression_rate'), 4)}</code></div></div>",
                f"<div><div class='metric'>Case Regression Rate</div><div><code>{format_metric(latest_case_level.get('regression_rate'), 4)}</code></div></div>",
                f"<div><div class='metric'>Validation Events</div><div><code>{len(validation_events)}</code></div></div>",
                "</div></div>",
                (
                    "<div class='card'><h2>Best Gain Event</h2>"
                    f"<div><code>{html.escape(str(best_event.get('label')))}</code> | "
                    f"UV gain <code>{format_metric(best_event.get('uv_gain'))}</code> | "
                    f"RM PSNR delta <code>{format_metric(best_event.get('rm_proxy_view_psnr_delta'))}</code> | "
                    f"case regression <code>{format_metric(best_event.get('case_regression_rate'), 4)}</code></div>"
                    "</div>"
                )
                if best_event is not None
                else "",
                (
                    "<div class='card'><h2>By Variant</h2><table><thead><tr>"
                    "<th>variant</th><th>input prior total MAE</th><th>refined total MAE</th><th>UV gain</th>"
                    "</tr></thead><tbody>"
                    + "".join(variant_rows_html)
                    + "</tbody></table></div>"
                )
                if variant_rows_html
                else "<div class='card'><h2>By Variant</h2><div>No validation payload with by-variant metrics was available at overview export time.</div></div>",
                (
                    f"<div class='card'><h2>Validation Evidence Curves</h2><img src='{evidence_curve_uri}' alt='training evidence curves'></div>"
                )
                if evidence_curve_uri is not None
                else "",
                (
                    f"<div class='card'><h2>Training Curves</h2><img src='{train_curve_uri}' alt='training curves'></div>"
                )
                if train_curve_uri is not None
                else "",
                "<div class='card'><h2>Metric Semantics</h2><ul>"
                "<li><b>RM proxy</b>: view-projected roughness/metallic target metrics computed from UV RM maps through <code>view_uvs</code>.</li>"
                "<li><b>RGB proxy</b>: eval-only <code>proxy_uv_shading</code> metrics, not the same as RM proxy.</li>"
                "<li><b>Real render</b>: reserved for future renderer/Blender re-render metrics.</li>"
                "<li>This run is an A-track prior-gap validation on <code>ABO_locked_core</code> / <code>glossy_non_metal</code>, not a broad generalization claim.</li>"
                "</ul></div>",
                "</div></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return overview_path


def write_training_evidence_report(
    *,
    output_dir: Path,
    validation_events: list[dict[str, Any]],
    figure_path: Path | None,
) -> dict[str, str]:
    if not validation_events:
        return {}
    report_json_path = output_dir / "training_evidence_report.json"
    report_html_path = output_dir / "training_evidence_report.html"
    latest = validation_events[-1]
    best_gain = max(
        validation_events,
        key=lambda item: maybe_float(item.get("uv_gain")),
    )
    payload = {
        "metric_families": {
            "rm_proxy": "View-projected roughness/metallic metrics computed from UV RM maps and view_uvs.",
            "rgb_proxy": "Evaluation-only lightweight proxy_uv_shading metrics.",
            "real_render": "Reserved for future renderer/Blender re-render metrics.",
        },
        "latest": latest,
        "best_uv_gain": best_gain,
        "events": validation_events,
    }
    save_json(report_json_path, payload)
    rows = []
    for item in validation_events[-32:]:
        rows.append(
            "<tr>"
            f"<td>{item.get('label')}</td>"
            f"<td>{format_metric(item.get('uv_gain'))}</td>"
            f"<td>{format_metric(item.get('rm_proxy_view_mae_delta'))}</td>"
            f"<td>{format_metric(item.get('rm_proxy_view_mse_delta'))}</td>"
            f"<td>{format_metric(item.get('rm_proxy_view_psnr_delta'))}</td>"
            f"<td>{format_metric(item.get('case_regression_rate'), 4)}</td>"
            f"<td>{format_metric(item.get('selection_metric'))}</td>"
            "</tr>"
        )
    report_html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Material Refiner Training Evidence</title>",
                "<style>body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1220px;margin:auto}.card{background:#18202b;border-radius:10px;padding:18px;margin:16px 0}img{max-width:100%;border-radius:8px;background:white}code{color:#b9e6ff}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #2d3748;padding:7px;text-align:left}th{color:#b9e6ff}</style>",
                "</head><body><div class='wrap'>",
                "<h1>Material Refiner Training Evidence</h1>",
                "<div class='card'>",
                "<div><b>RM proxy</b>: view-projected roughness/metallic target metrics from UV maps and view_uvs.</div>",
                "<div><b>RGB proxy</b>: eval proxy_uv_shading. <b>Real render</b>: future renderer/Blender metrics.</div>",
                f"<div>Latest UV gain: <code>{format_metric(latest.get('uv_gain'))}</code>; RM proxy PSNR delta: <code>{format_metric(latest.get('rm_proxy_view_psnr_delta'))}</code>; case regression: <code>{format_metric(latest.get('case_regression_rate'), 4)}</code></div>",
                f"<div>Best UV gain event: <code>{best_gain.get('label')}</code> = <code>{format_metric(best_gain.get('uv_gain'))}</code></div>",
                "</div>",
                f"<div class='card'><img src='{figure_path.name}' alt='training evidence curves'></div>" if figure_path is not None else "",
                "<div class='card'><table><thead><tr><th>event</th><th>UV gain</th><th>RM MAE delta</th><th>RM MSE delta</th><th>RM PSNR delta</th><th>case regression</th><th>selection</th></tr></thead><tbody>",
                *rows,
                "</tbody></table></div></div></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "training_evidence_report": str(report_html_path.resolve()),
        "training_evidence_json": str(report_json_path.resolve()),
    }


def save_training_visualizations(history: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    if not history:
        return {}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[visualization:warning] export_failed={type(exc).__name__}:{exc}")
        return {}

    validation_events = load_validation_events(output_dir)
    epochs = [int(item.get("epoch", index + 1)) for index, item in enumerate(history)]
    train_total = [float((item.get("train") or {}).get("total", np.nan)) for item in history]
    val_total_epoch = [
        float(((item.get("val") or {}).get("uv_mae") or {}).get("total", np.nan))
        for item in history
    ]
    baseline_val_total_epoch = [
        float(((item.get("val") or {}).get("baseline_uv_mae") or {}).get("total", np.nan))
        for item in history
    ]
    improvement_val_total_epoch = [
        float(((item.get("val") or {}).get("improvement_uv_mae") or {}).get("total", np.nan))
        for item in history
    ]
    val_roughness_epoch = [
        float(((item.get("val") or {}).get("uv_mae") or {}).get("roughness", np.nan))
        for item in history
    ]
    val_metallic_epoch = [
        float(((item.get("val") or {}).get("uv_mae") or {}).get("metallic", np.nan))
        for item in history
    ]
    samples_per_second = [
        float((item.get("train") or {}).get("samples_per_second", np.nan))
        for item in history
    ]
    use_event_validation = bool(validation_events) and np.isnan(val_total_epoch).all()
    if use_event_validation:
        val_x = list(range(1, len(validation_events) + 1))
        val_x_labels = [str(item.get("label")) for item in validation_events]
        val_total = [maybe_float(item.get("uv_total")) for item in validation_events]
        baseline_val_total = [maybe_float(item.get("input_prior_total")) for item in validation_events]
        improvement_val_total = [maybe_float(item.get("uv_gain")) for item in validation_events]
        val_roughness = [maybe_float(item.get("uv_roughness")) for item in validation_events]
        val_metallic = [maybe_float(item.get("uv_metallic")) for item in validation_events]
    else:
        val_x = epochs
        val_x_labels = [str(epoch) for epoch in epochs]
        val_total = val_total_epoch
        baseline_val_total = baseline_val_total_epoch
        improvement_val_total = improvement_val_total_epoch
        val_roughness = val_roughness_epoch
        val_metallic = val_metallic_epoch

    figure_path = output_dir / "training_curves.png"
    html_path = output_dir / "training_summary.html"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Material Refiner Training Curves")
    if use_event_validation:
        axes[0, 0].plot(val_x, val_total, marker="o", label="val uv total")
    else:
        axes[0, 0].plot(epochs, train_total, marker="o", label="train total")
        axes[0, 0].plot(val_x, val_total, marker="o", label="val uv total")
    if not np.isnan(baseline_val_total).all():
        axes[0, 0].plot(val_x, baseline_val_total, marker="o", label="input prior")
    axes[0, 0].set_title("Validation Event UV MAE" if use_event_validation else "Total Loss / UV MAE")
    axes[0, 0].set_xlabel("validation event" if use_event_validation else "epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(val_x, val_roughness, marker="o", label="roughness")
    axes[0, 1].plot(val_x, val_metallic, marker="o", label="metallic")
    axes[0, 1].set_title("Validation UV MAE By Channel")
    axes[0, 1].set_xlabel("validation event" if use_event_validation else "epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    if not np.isnan(improvement_val_total).all():
        axes[1, 0].plot(val_x, improvement_val_total, marker="o")
        axes[1, 0].axhline(0.0, color="black", linewidth=1, alpha=0.35)
        axes[1, 0].set_title("Refined Gain Over Input Prior")
        axes[1, 0].set_ylabel("baseline MAE - refined MAE")
    else:
        axes[1, 0].plot(epochs, samples_per_second, marker="o")
        axes[1, 0].set_title("Training Throughput")
        axes[1, 0].set_ylabel("samples/sec")
    axes[1, 0].set_xlabel("validation event" if not np.isnan(improvement_val_total).all() and use_event_validation else "epoch")
    axes[1, 0].grid(alpha=0.25)

    best_index = int(np.nanargmin(val_total)) if not np.isnan(val_total).all() else len(val_total) - 1
    latest_validation = summarize_latest_validation(history, validation_events)
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.92,
        "\n".join(
            [
                f"epochs: {len(history)}",
                (
                    f"best validation event: {val_x_labels[best_index]}"
                    if use_event_validation
                    else f"best epoch: {epochs[best_index]}"
                ),
                f"best val total: {format_metric(val_total[best_index])}",
                f"best baseline total: {format_metric(baseline_val_total[best_index]) if not np.isnan(baseline_val_total).all() else 'n/a'}",
                f"best improvement: {format_metric(improvement_val_total[best_index]) if not np.isnan(improvement_val_total).all() else 'n/a'}",
                f"last train total: {format_metric(train_total[-1])}",
                f"last val total: {format_metric(latest_validation.get('val_total'))}",
                f"latest validation source: {latest_validation.get('source')}",
                f"latest validation label: {latest_validation.get('label')}",
            ]
        ),
        va="top",
        family="monospace",
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    evidence_figure_path: Path | None = None
    if validation_events:
        evidence_figure_path = output_dir / "training_evidence_curves.png"
        event_x = list(range(1, len(validation_events) + 1))
        event_labels = [str(item.get("label")) for item in validation_events]
        uv_gain = [maybe_float(item.get("uv_gain")) for item in validation_events]
        rm_mae_delta = [maybe_float(item.get("rm_proxy_view_mae_delta")) for item in validation_events]
        rm_mse_delta = [maybe_float(item.get("rm_proxy_view_mse_delta")) for item in validation_events]
        rm_psnr_delta = [maybe_float(item.get("rm_proxy_view_psnr_delta")) for item in validation_events]
        case_regression = [maybe_float(item.get("case_regression_rate")) for item in validation_events]
        fig2, axes2 = plt.subplots(2, 2, figsize=(13, 8))
        fig2.suptitle("Validation Evidence: Baseline vs Refined")
        axes2[0, 0].plot(event_x, uv_gain, marker="o")
        axes2[0, 0].axhline(0.0, color="black", linewidth=1, alpha=0.35)
        axes2[0, 0].set_title("UV Gain (input prior MAE - refined MAE)")
        axes2[0, 0].grid(alpha=0.25)
        axes2[0, 1].plot(event_x, rm_psnr_delta, marker="o", label="PSNR delta")
        axes2[0, 1].axhline(0.0, color="black", linewidth=1, alpha=0.35)
        axes2[0, 1].set_title("RM Proxy View PSNR Delta")
        axes2[0, 1].grid(alpha=0.25)
        axes2[1, 0].plot(event_x, rm_mae_delta, marker="o", label="MAE delta")
        axes2[1, 0].plot(event_x, rm_mse_delta, marker="o", label="MSE delta")
        axes2[1, 0].axhline(0.0, color="black", linewidth=1, alpha=0.35)
        axes2[1, 0].set_title("RM Proxy Positive Delta Means Refined Is Better")
        axes2[1, 0].legend()
        axes2[1, 0].grid(alpha=0.25)
        axes2[1, 1].plot(event_x, case_regression, marker="o")
        axes2[1, 1].set_title("Case-Level Regression Rate")
        axes2[1, 1].grid(alpha=0.25)
        for axis in axes2.flat:
            axis.set_xlabel("validation event")
            if len(event_labels) <= 12:
                axis.set_xticks(event_x)
                axis.set_xticklabels(event_labels, rotation=30, ha="right")
        fig2.tight_layout()
        fig2.savefig(evidence_figure_path, dpi=160)
        plt.close(fig2)

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
                f"<div>Latest validation source: <code>{html.escape(str(latest_validation.get('source')))}</code></div>",
                f"<div>Latest validation label: <code>{html.escape(str(latest_validation.get('label')))}</code></div>",
                f"<div>Last val total: <code>{format_metric(latest_validation.get('val_total'))}</code></div>",
                f"<div>Last input prior total: <code>{format_metric(latest_validation.get('input_prior_total'))}</code></div>",
                f"<div>Last improvement: <code>{format_metric(latest_validation.get('improvement_total'))}</code></div>",
                "</div></div></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    evidence_paths = write_training_evidence_report(
        output_dir=output_dir,
        validation_events=validation_events,
        figure_path=evidence_figure_path,
    )
    return {
        "training_curves": str(figure_path.resolve()),
        "training_summary": str(html_path.resolve()),
        **(
            {"training_evidence_curves": str(evidence_figure_path.resolve())}
            if evidence_figure_path is not None
            else {}
        ),
        **evidence_paths,
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


def enhanced_grayscale_rgb_image(
    tensor: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    gamma: float = 0.65,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> Image.Image:
    value = tensor.detach().cpu().float().clamp(0.0, 1.0).squeeze(0).numpy()
    valid = np.ones_like(value, dtype=bool)
    if mask is not None:
        valid = mask.detach().cpu().float().squeeze(0).numpy() > 0.05
    visible_values = value[valid]
    if visible_values.size >= 8:
        low = float(np.percentile(visible_values, lower_percentile))
        high = float(np.percentile(visible_values, upper_percentile))
        if high > low + 1e-5:
            value = (value - low) / (high - low)
    value = np.clip(value, 0.0, 1.0)
    value = np.power(value, max(float(gamma), 1e-4))
    if mask is not None:
        value = value * valid.astype(np.float32)
    data = (value * 255.0).round().astype(np.uint8)
    return Image.fromarray(np.stack([data, data, data], axis=-1), mode="RGB")


def enhanced_rgb_tensor_image(
    tensor: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    gamma: float = 0.65,
) -> Image.Image:
    image = tensor.detach().cpu().float()[:3].clamp(0.0, 1.0).numpy()
    valid = np.ones(image.shape[1:], dtype=bool)
    if mask is not None:
        valid = mask.detach().cpu().float().squeeze(0).numpy() > 0.05
    for channel in range(image.shape[0]):
        channel_values = image[channel]
        visible_values = channel_values[valid]
        if visible_values.size >= 8:
            low = float(np.percentile(visible_values, 2.0))
            high = float(np.percentile(visible_values, 98.0))
            if high > low + 1e-5:
                image[channel] = (channel_values - low) / (high - low)
    image = np.power(np.clip(image, 0.0, 1.0), max(float(gamma), 1e-4))
    if mask is not None:
        image = image * valid[None, :, :].astype(np.float32)
    data = (np.moveaxis(image, 0, -1) * 255.0).round().astype(np.uint8)
    return Image.fromarray(data, mode="RGB")


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


def sanitize_preview_filename_component(value: Any, *, default: str = "unknown", max_length: int = 96) -> str:
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


def first_preview_identity(*values: Any) -> str:
    for value in values:
        text = "" if value is None else str(value).strip()
        if text and text.lower() not in {"none", "null", "unknown"}:
            return text
    return "unknown"


def preview_effect_bucket(gain_total: float) -> str:
    if gain_total > 0.02:
        return "positive_gain"
    if gain_total < -0.02:
        return "regression"
    return "near_zero"


def should_select_validation_preview(
    *,
    mode: str,
    selected_count: int,
    max_count: int,
    prior_variant_type: str,
    effect_bucket: str,
    seen_variants: set[str],
    seen_effects: set[str],
) -> bool:
    if selected_count >= max_count:
        return False
    if mode == "first":
        return True
    if selected_count < max(1, min(max_count, 2)):
        return True
    if mode == "balanced":
        return prior_variant_type not in seen_variants
    return effect_bucket not in seen_effects or prior_variant_type not in seen_variants


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
    pair_id: str = "",
    target_bundle_id: str = "",
    prior_variant_id: str = "",
    prior_variant_type: str = "unknown",
    prior_quality_bin: str = "unknown",
    training_role: str = "unknown",
    view_rm_mae_delta: float | None = None,
    preview_slot: int = 0,
    source_name: str = "unknown_source",
    material_family: str = "unknown_material",
    prior_label: str = "unknown_prior",
) -> Path:
    preview_dir = output_dir / "validation_previews" / validation_label
    preview_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = select_preview_view_indices(view_names, max_views=1)
    preview_view_index = selected_indices[0] if selected_indices else 0
    preview_view = rgb_tensor_image(
        view_features[preview_view_index, 0:3],
        mask=view_masks[preview_view_index],
    )
    preview_view_enhanced = enhanced_rgb_tensor_image(
        view_features[preview_view_index, 0:3],
        mask=view_masks[preview_view_index],
    )

    tile_size = 128
    row_label_width = 90
    gutter = 8
    title_height = 94
    columns = 7
    tile_width = tile_size
    tile_height = tile_size + 34
    canvas_width = row_label_width + columns * tile_width + (columns + 1) * gutter
    rows = [
        (
            "Input",
            [
                compact_labeled_image(preview_view, "View", size=tile_size),
                compact_labeled_image(preview_view_enhanced, "View γ", size=tile_size),
                compact_labeled_image(rgb_tensor_image(uv_albedo), "Albedo", size=tile_size),
                compact_labeled_image(enhanced_rgb_tensor_image(uv_albedo), "Albedo γ", size=tile_size),
                compact_labeled_image(rgb_tensor_image(uv_normal), "Normal", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(confidence), "Conf", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(confidence), "Conf γ", size=tile_size),
            ],
        ),
        (
            "Rough",
            [
                compact_labeled_image(grayscale_rgb_image(baseline[0:1]), "Input Prior", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(baseline[0:1], mask=confidence), "Prior γ", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(target_roughness), "GT", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(target_roughness, mask=confidence), "GT γ", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(refined[0:1]), "Pred", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(refined[0:1], mask=confidence), "Pred γ", size=tile_size),
                compact_labeled_image(delta_heatmap_image(refined[0:1], target_roughness), "Err", size=tile_size),
            ],
        ),
        (
            "Metal",
            [
                compact_labeled_image(grayscale_rgb_image(baseline[1:2]), "Input Prior", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(baseline[1:2], mask=confidence), "Prior γ", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(target_metallic), "GT", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(target_metallic, mask=confidence), "GT γ", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(refined[1:2]), "Pred", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(refined[1:2], mask=confidence), "Pred γ", size=tile_size),
                compact_labeled_image(delta_heatmap_image(refined[1:2], target_metallic), "Err", size=tile_size),
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
    id_font = preview_font(12)
    draw.text((12, 10), str(object_id), font=title_font, fill=(8, 12, 18))
    draw.text(
        (12, 38),
        (
            f"Input Prior {baseline_total:.4f} | Pred {refined_total:.4f} | "
            f"UV gain {improvement:+.4f} | view RM delta {format_metric(view_rm_mae_delta, 4)} | {prior_label} | "
            f"variant={prior_variant_type} | quality={prior_quality_bin} | role={training_role}"
        ),
        font=detail_font,
        fill=(68, 76, 88),
    )
    identity_line = (
        f"pair={pair_id or 'unknown'} | target={target_bundle_id or 'unknown'} | "
        f"prior_variant={prior_variant_id or 'unknown'}"
    )
    identity_detail_line = f"{identity_line} | {material_family} | {source_name}"
    title_line_width = canvas_width - 24
    detail_bbox = draw.textbbox((12, 62), identity_detail_line, font=id_font)
    id_bbox = draw.textbbox((12, 62), identity_line, font=id_font)
    if detail_bbox[2] - detail_bbox[0] <= title_line_width:
        draw.text((12, 62), identity_detail_line, font=id_font, fill=(94, 104, 118))
    elif id_bbox[2] - id_bbox[0] <= title_line_width:
        draw.text((12, 62), identity_line, font=id_font, fill=(94, 104, 118))
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
    pair_or_prior_variant_id = first_preview_identity(pair_id, prior_variant_id)
    safe_preview_slot = max(int(preview_slot), 0)
    safe_object_id = sanitize_preview_filename_component(object_id)
    safe_prior_variant_type = sanitize_preview_filename_component(prior_variant_type)
    safe_pair_or_prior_variant_id = sanitize_preview_filename_component(pair_or_prior_variant_id, max_length=128)
    output_path = preview_dir / (
        f"{safe_preview_slot:03d}__{safe_object_id}__{safe_prior_variant_type}__"
        f"{safe_pair_or_prior_variant_id}.png"
    )
    canvas.save(output_path)
    return output_path


def build_preview_integrity_report(preview_items: list[dict[str, Any]]) -> dict[str, Any]:
    required_metadata = [
        "case_id",
        "object_id",
        "pair_id",
        "target_bundle_id",
        "prior_variant_id",
        "prior_variant_type",
        "prior_quality_bin",
        "training_role",
    ]
    paths = [str(item.get("path") or "") for item in preview_items]
    path_counts = Counter(path for path in paths if path)
    duplicate_paths = {
        path: count
        for path, count in sorted(path_counts.items())
        if count > 1
    }
    missing_files = [
        path
        for path in paths
        if path and not Path(path).exists()
    ]
    missing_metadata_items = []
    for index, item in enumerate(preview_items):
        missing_fields = [
            field
            for field in required_metadata
            if field not in item or str(item.get(field) or "").strip() == ""
        ]
        if missing_fields:
            missing_metadata_items.append(
                {
                    "index": index,
                    "path": str(item.get("path") or ""),
                    "object_id": str(item.get("object_id") or ""),
                    "missing_fields": missing_fields,
                }
            )
    return {
        "enabled": True,
        "ok": not duplicate_paths and not missing_files and not missing_metadata_items,
        "items": len(preview_items),
        "unique_paths": len(path_counts),
        "duplicate_paths": duplicate_paths,
        "missing_files": missing_files,
        "required_metadata_fields": required_metadata,
        "missing_metadata_items": missing_metadata_items,
    }


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


def update_object_improvement_store(
    store: dict[str, dict[str, float]],
    *,
    object_id: str,
    total_improvement: float,
    roughness_improvement: float,
    metallic_improvement: float,
) -> None:
    bucket = store.setdefault(
        object_id,
        {"count": 0.0, "total": 0.0, "roughness": 0.0, "metallic": 0.0},
    )
    bucket["count"] += 1.0
    bucket["total"] += total_improvement
    bucket["roughness"] += roughness_improvement
    bucket["metallic"] += metallic_improvement


def make_material_case_key(
    *,
    object_id: str,
    pair_id: str,
    prior_variant_id: str,
    prior_variant_type: str,
) -> str:
    identity = pair_id if pair_id and pair_id != "unknown" else prior_variant_id
    if not identity or identity == "unknown":
        identity = "no_pair_or_variant_id"
    return f"{object_id}|{identity}|{prior_variant_type or 'unknown'}"


def finalize_improvement_level_metrics(
    store: dict[str, dict[str, float]],
    *,
    level_name: str,
) -> dict[str, Any]:
    values = []
    id_key = f"{level_name}_id"
    for item_id, bucket in sorted(store.items()):
        count = max(float(bucket.get("count", 0.0)), 1.0)
        values.append(
            {
                id_key: item_id,
                "count": int(bucket.get("count", 0.0)),
                "avg_improvement_total": float(bucket.get("total", 0.0) / count),
                "avg_improvement_roughness": float(bucket.get("roughness", 0.0) / count),
                "avg_improvement_metallic": float(bucket.get("metallic", 0.0) / count),
            }
        )
    level_count = max(len(values), 1)
    improved = sum(1 for item in values if item["avg_improvement_total"] > 1e-6)
    regressed = sum(1 for item in values if item["avg_improvement_total"] < -1e-6)
    unchanged = len(values) - improved - regressed
    plural = f"{level_name}s"
    return {
        f"{level_name}_count": len(values),
        "avg_improvement_total": (
            sum(item["avg_improvement_total"] for item in values) / level_count
        ),
        "avg_improvement_roughness": (
            sum(item["avg_improvement_roughness"] for item in values) / level_count
        ),
        "avg_improvement_metallic": (
            sum(item["avg_improvement_metallic"] for item in values) / level_count
        ),
        f"improved_{plural}": improved,
        f"regressed_{plural}": regressed,
        f"unchanged_{plural}": unchanged,
        "improvement_rate": improved / level_count,
        "regression_rate": regressed / level_count,
        "unchanged_rate": unchanged / level_count,
        "entries": values,
    }


def finalize_object_improvement_metrics(store: dict[str, dict[str, float]]) -> dict[str, Any]:
    return finalize_improvement_level_metrics(store, level_name="object")


def finalize_case_improvement_metrics(store: dict[str, dict[str, float]]) -> dict[str, Any]:
    return finalize_improvement_level_metrics(store, level_name="case")


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


def masked_global_ssim_torch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
    if prediction.numel() == 0 or target.shape != prediction.shape:
        return None
    weight = mask.expand_as(prediction).to(prediction.dtype)
    denom = weight.sum().clamp_min(1.0)
    mu_x = (prediction * weight).sum() / denom
    mu_y = (target * weight).sum() / denom
    var_x = ((prediction - mu_x).square() * weight).sum() / denom
    var_y = ((target - mu_y).square() * weight).sum() / denom
    cov_xy = ((prediction - mu_x) * (target - mu_y) * weight).sum() / denom
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return float(ssim.detach().clamp(-1.0, 1.0).cpu().item())


def get_lpips_model(enabled: bool, device: str) -> Any | None:
    global _LPIPS_DEVICE, _LPIPS_FAILURE, _LPIPS_MODEL
    if not enabled:
        return None
    if _LPIPS_MODEL is not None and _LPIPS_DEVICE == device:
        return _LPIPS_MODEL
    if _LPIPS_FAILURE is not None:
        return None
    try:
        import lpips  # type: ignore

        model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:  # noqa: BLE001 - optional metric should not block training.
        _LPIPS_FAILURE = f"{type(exc).__name__}: {exc}"
        print(f"[metric:warning] lpips_unavailable={_LPIPS_FAILURE}")
        return None
    _LPIPS_MODEL = model
    _LPIPS_DEVICE = device
    return _LPIPS_MODEL


def rm_to_lpips_rgb(rm_view: torch.Tensor) -> torch.Tensor:
    if rm_view.shape[1] == 1:
        rgb = rm_view.expand(-1, 3, -1, -1)
    else:
        rough = rm_view[:, 0:1]
        metal = rm_view[:, 1:2]
        rgb = torch.cat([rough, metal, rough], dim=1)
    return rgb.clamp(0.0, 1.0) * 2.0 - 1.0


def compute_lpips_rm_proxy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: str,
    enabled: bool,
    max_images: int,
) -> float | None:
    model = get_lpips_model(enabled, device)
    if model is None or max_images <= 0:
        return None
    pred = prediction.reshape(-1, prediction.shape[2], prediction.shape[3], prediction.shape[4])
    tgt = target.reshape(-1, target.shape[2], target.shape[3], target.shape[4])
    m = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
    valid = m.flatten(1).sum(dim=1) > 8
    if not bool(valid.any()):
        return None
    pred = pred[valid][:max_images] * m[valid][:max_images]
    tgt = tgt[valid][:max_images] * m[valid][:max_images]
    with torch.no_grad():
        distances = model(rm_to_lpips_rgb(pred), rm_to_lpips_rgb(tgt))
    return float(distances.mean().detach().cpu().item())


def safe_wandb_key(value: Any) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value or "unknown")
    )
    return safe.strip("_") or "unknown"


def finalize_render_proxy_metrics(store: dict[str, float]) -> dict[str, Any]:
    denom = max(float(store.get("denom", 0.0)), 1.0)
    baseline_mae = float(store.get("baseline_abs", 0.0) / denom)
    refined_mae = float(store.get("refined_abs", 0.0) / denom)
    baseline_mse = float(store.get("baseline_sq", 0.0) / denom)
    refined_mse = float(store.get("refined_sq", 0.0) / denom)
    ssim_count = max(float(store.get("ssim_count", 0.0)), 1.0)
    lpips_count = max(float(store.get("lpips_count", 0.0)), 1.0)
    return {
        "enabled": True,
        "available": bool(store.get("available_batches", 0.0) > 0.0),
        "batches": int(store.get("batches", 0.0)),
        "available_batches": int(store.get("available_batches", 0.0)),
        "samples": int(store.get("samples", 0.0)),
        "baseline_view_rm_mae": baseline_mae,
        "refined_view_rm_mae": refined_mae,
        "view_rm_mae_delta": baseline_mae - refined_mae,
        "baseline_proxy_rm_mse": baseline_mse,
        "refined_proxy_rm_mse": refined_mse,
        "proxy_rm_mse_delta": baseline_mse - refined_mse,
        "baseline_proxy_rm_psnr": psnr_from_mse(baseline_mse),
        "refined_proxy_rm_psnr": psnr_from_mse(refined_mse),
        "baseline_proxy_rm_ssim": float(store.get("baseline_ssim", 0.0) / ssim_count)
        if store.get("ssim_count", 0.0) > 0.0
        else None,
        "refined_proxy_rm_ssim": float(store.get("refined_ssim", 0.0) / ssim_count)
        if store.get("ssim_count", 0.0) > 0.0
        else None,
        "baseline_proxy_rm_lpips": float(store.get("baseline_lpips", 0.0) / lpips_count)
        if store.get("lpips_count", 0.0) > 0.0
        else None,
        "refined_proxy_rm_lpips": float(store.get("refined_lpips", 0.0) / lpips_count)
        if store.get("lpips_count", 0.0) > 0.0
        else None,
        "proxy_rm_psnr_delta": (
            None
            if psnr_from_mse(baseline_mse) is None or psnr_from_mse(refined_mse) is None
            else psnr_from_mse(refined_mse) - psnr_from_mse(baseline_mse)
        ),
        "lpips_status": "available" if store.get("lpips_count", 0.0) > 0.0 else (_LPIPS_FAILURE or "not_computed"),
        "mode": "view_projected_rm_proxy",
    }


def update_render_proxy_metrics(
    store: dict[str, float],
    *,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    device: str,
) -> None:
    if batch.get("view_uvs") is None:
        return
    view_mask = batch["view_masks"].clamp(0.0, 1.0)
    view_mask = view_mask * view_uv_valid_mask(batch["view_uvs"]).to(view_mask.device)
    supervision_mask = batch["has_effective_view_supervision"].to(view_mask.device)
    view_mask = view_mask * supervision_mask.view(-1, 1, 1, 1, 1)
    denom = float(view_mask.sum().detach().cpu().item()) * float(refined.shape[1])
    if denom <= 0.0:
        return
    target_views = sample_uv_maps_to_view(target, batch["view_uvs"])
    baseline_views = sample_uv_maps_to_view(baseline, batch["view_uvs"])
    refined_views = sample_uv_maps_to_view(refined, batch["view_uvs"])
    baseline_ssim = masked_global_ssim_torch(baseline_views, target_views, view_mask)
    refined_ssim = masked_global_ssim_torch(refined_views, target_views, view_mask)
    if baseline_ssim is not None and refined_ssim is not None:
        store["baseline_ssim"] += baseline_ssim
        store["refined_ssim"] += refined_ssim
        store["ssim_count"] += 1.0
    if bool(getattr(args, "val_enable_lpips", True)):
        baseline_lpips = compute_lpips_rm_proxy(
            baseline_views,
            target_views,
            view_mask,
            device=device,
            enabled=True,
            max_images=int(getattr(args, "val_lpips_max_images", 12)),
        )
        refined_lpips = compute_lpips_rm_proxy(
            refined_views,
            target_views,
            view_mask,
            device=device,
            enabled=True,
            max_images=int(getattr(args, "val_lpips_max_images", 12)),
        )
        if baseline_lpips is not None and refined_lpips is not None:
            store["baseline_lpips"] += baseline_lpips
            store["refined_lpips"] += refined_lpips
            store["lpips_count"] += 1.0
    baseline_error = baseline_views - target_views
    refined_error = refined_views - target_views
    store["baseline_abs"] += float((baseline_error.abs() * view_mask).sum().detach().cpu().item())
    store["refined_abs"] += float((refined_error.abs() * view_mask).sum().detach().cpu().item())
    store["baseline_sq"] += float(((baseline_error ** 2) * view_mask).sum().detach().cpu().item())
    store["refined_sq"] += float(((refined_error ** 2) * view_mask).sum().detach().cpu().item())
    store["denom"] += denom
    store["available_batches"] += 1.0
    store["samples"] += float(refined.shape[0])


def per_sample_view_rm_mae_delta(
    *,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor | None:
    if batch.get("view_uvs") is None:
        return None
    view_mask = batch["view_masks"].clamp(0.0, 1.0)
    view_mask = view_mask * view_uv_valid_mask(batch["view_uvs"]).to(view_mask.device)
    supervision_mask = batch["has_effective_view_supervision"].to(view_mask.device)
    view_mask = view_mask * supervision_mask.view(-1, 1, 1, 1, 1)
    denom = view_mask.flatten(1).sum(dim=1).clamp_min(1.0) * float(refined.shape[1])
    target_views = sample_uv_maps_to_view(target, batch["view_uvs"])
    baseline_views = sample_uv_maps_to_view(baseline, batch["view_uvs"])
    refined_views = sample_uv_maps_to_view(refined, batch["view_uvs"])
    baseline_mae = ((baseline_views - target_views).abs() * view_mask).flatten(1).sum(dim=1) / denom
    refined_mae = ((refined_views - target_views).abs() * view_mask).flatten(1).sum(dim=1) / denom
    has_support = view_mask.flatten(1).sum(dim=1) > 0.0
    return torch.where(has_support, baseline_mae - refined_mae, torch.full_like(baseline_mae, float("nan")))


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


def _metric_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def update_validation_special_metrics(
    store: dict[str, float],
    *,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    _ = batch, baseline
    confidence = confidence.float().clamp(0.0, 1.0)
    error = (refined.float() - target.float()).abs()
    channel_weight = confidence.expand_as(error)

    target_edge = rm_gradient_magnitude(target.float()).detach()
    edge_max = target_edge.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    edge_norm = (target_edge / edge_max).clamp(0.0, 1.0)
    edge_band = F.max_pool2d(edge_norm, kernel_size=7, stride=1, padding=3).clamp(0.0, 1.0)
    boundary_weight = edge_band.expand_as(error) * channel_weight
    interior_weight = (1.0 - edge_band).expand_as(error) * channel_weight
    boundary_denom = boundary_weight.sum()
    interior_denom = interior_weight.sum()
    if _metric_float(boundary_denom) > 0.0 and _metric_float(interior_denom) > 0.0:
        boundary_error = (error * boundary_weight).sum() / boundary_denom.clamp_min(1.0)
        interior_error = (error * interior_weight).sum() / interior_denom.clamp_min(1.0)
        store["boundary_bleed_score"] += _metric_float(boundary_error - interior_error)
        store["boundary_bleed_count"] += 1.0

    metallic_target = (target[:, 1:2].float() >= METALLIC_THRESHOLD).to(refined.dtype)
    metallic_pred = (refined[:, 1:2].float() >= METALLIC_THRESHOLD).to(refined.dtype)
    metal_denom = confidence.sum()
    if _metric_float(metal_denom) > 0.0:
        confusion = ((metallic_pred != metallic_target).to(refined.dtype) * confidence).sum()
        store["metal_confusion_rate"] += _metric_float(confusion / metal_denom.clamp_min(1.0))
        store["metal_confusion_count"] += 1.0

    highlight_mask = (target[:, 0:1].float() <= 0.35).to(refined.dtype) * confidence
    highlight_denom = highlight_mask.sum()
    if _metric_float(highlight_denom) > 0.0:
        highlight_error = (error.mean(dim=1, keepdim=True) * highlight_mask).sum()
        store["highlight_localization_error"] += _metric_float(highlight_error / highlight_denom.clamp_min(1.0))
        store["highlight_count"] += 1.0

    refined_grad = rm_gradient_magnitude(refined.float())
    target_grad = rm_gradient_magnitude(target.float())
    grad_weight = confidence
    grad_denom = grad_weight.sum()
    if _metric_float(grad_denom) > 0.0:
        refined_mean = (refined_grad * grad_weight).sum() / grad_denom.clamp_min(1.0)
        target_mean = (target_grad * grad_weight).sum() / grad_denom.clamp_min(1.0)
        refined_centered = refined_grad - refined_mean
        target_centered = target_grad - target_mean
        var_refined = (refined_centered.square() * grad_weight).sum() / grad_denom.clamp_min(1.0)
        var_target = (target_centered.square() * grad_weight).sum() / grad_denom.clamp_min(1.0)
        if _metric_float(var_refined) > 1e-8 and _metric_float(var_target) > 1e-8:
            corr = (
                (refined_centered * target_centered * grad_weight).sum()
                / grad_denom.clamp_min(1.0)
                / (var_refined.sqrt() * var_target.sqrt()).clamp_min(1e-6)
            )
            store["rm_gradient_preservation"] += _metric_float(corr.clamp(-1.0, 1.0))
            store["rm_gradient_count"] += 1.0


def finalize_validation_special_metrics(store: dict[str, float]) -> dict[str, Any]:
    def mean(name: str, count_name: str) -> float | None:
        count = float(store.get(count_name, 0.0))
        if count <= 0.0:
            return None
        return float(store.get(name, 0.0) / count)

    return {
        "boundary_bleed_score": mean("boundary_bleed_score", "boundary_bleed_count"),
        "metal_confusion_rate": mean("metal_confusion_rate", "metal_confusion_count"),
        "highlight_localization_error": mean("highlight_localization_error", "highlight_count"),
        "rm_gradient_preservation": mean("rm_gradient_preservation", "rm_gradient_count"),
        "metric_availability": {
            "boundary_bleed_score_batches": int(store.get("boundary_bleed_count", 0.0)),
            "metal_confusion_rate_batches": int(store.get("metal_confusion_count", 0.0)),
            "highlight_localization_error_batches": int(store.get("highlight_count", 0.0)),
            "rm_gradient_preservation_batches": int(store.get("rm_gradient_count", 0.0)),
        },
    }


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
        "material_context": 0.0,
        "residual_safety": 0.0,
        "residual_gate_mean": 0.0,
        "residual_delta_abs": 0.0,
        "view_uncertainty_gate_mean": 0.0,
        "bleed_risk_gate_mean": 0.0,
        "topology_residual_gate_mean": 0.0,
        "render_support_gate_mean": 0.0,
        "inverse_material_gate_mean": 0.0,
        "residual_channel_gate_mean": 0.0,
        "roughness_safety_gate_mean": 0.0,
        "metallic_safety_gate_mean": 0.0,
        "metallic_evidence_mean": 0.0,
        "metallic_cap_strength_mean": 0.0,
        "evidence_update_budget_mean": 0.0,
        "evidence_update_support_mean": 0.0,
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
    object_improvement_store: dict[str, dict[str, float]] = {}
    case_improvement_store: dict[str, dict[str, float]] = {}
    variant_view_store: dict[str, dict[str, float]] = {}
    steps = 0
    preview_candidates: list[dict[str, Any]] = []
    render_proxy_enabled = should_compute_render_proxy_validation(args, validation_label)
    render_proxy_store: dict[str, float] = defaultdict(float)
    residual_gate_store: dict[str, float] = defaultdict(float)
    residual_gate_cases: list[dict[str, Any]] = []
    special_metric_store: dict[str, float] = defaultdict(float)

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
            baseline = outputs.get("input_prior", outputs["baseline"])
            update_validation_special_metrics(
                special_metric_store,
                batch=batch,
                baseline=baseline,
                refined=refined,
                target=target,
                confidence=confidence,
            )
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
                    target=target,
                    args=args,
                    device=device,
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
            view_rm_delta = (
                per_sample_view_rm_mae_delta(
                    batch=batch,
                    baseline=baseline,
                    refined=refined,
                    target=target,
                )
                if int(args.val_preview_samples) > 0
                else None
            )

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

            batch_size = len(batch["object_id"])
            for item_index, object_id in enumerate(batch["object_id"]):
                object_id = str(object_id)
                pair_id = str(batch.get("pair_id", ["unknown"] * batch_size)[item_index] or "unknown")
                target_bundle_id = str(batch.get("target_bundle_id", ["unknown"] * batch_size)[item_index] or "unknown")
                prior_variant_id = str(batch.get("prior_variant_id", ["unknown"] * batch_size)[item_index] or "unknown")
                generator_id = str(batch["generator_id"][item_index])
                source_name = str(batch["source_name"][item_index])
                material_family = str(batch["material_family"][item_index])
                prior_name = "with_prior" if bool(batch["has_material_prior"][item_index]) else "without_prior"
                prior_variant_type = str(batch.get("prior_variant_type", ["unknown"] * batch_size)[item_index] or "unknown")
                prior_quality_bin = str(batch.get("prior_quality_bin", ["unknown"] * batch_size)[item_index] or "unknown")
                training_role = str(batch.get("training_role", ["unknown"] * batch_size)[item_index] or "unknown")
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
                update_object_improvement_store(
                    object_improvement_store,
                    object_id=object_id,
                    total_improvement=improvement_metric_kwargs["total_mae"],
                    roughness_improvement=improvement_metric_kwargs["roughness_mae"],
                    metallic_improvement=improvement_metric_kwargs["metallic_mae"],
                )
                case_key = make_material_case_key(
                    object_id=object_id,
                    pair_id=pair_id,
                    prior_variant_id=prior_variant_id,
                    prior_variant_type=prior_variant_type,
                )
                update_object_improvement_store(
                    case_improvement_store,
                    object_id=case_key,
                    total_improvement=improvement_metric_kwargs["total_mae"],
                    roughness_improvement=improvement_metric_kwargs["roughness_mae"],
                    metallic_improvement=improvement_metric_kwargs["metallic_mae"],
                )
                update_group_metric_store(group_store, key=f"generator/{generator_id}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"source/{source_name}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"prior/{prior_name}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"prior_variant_type/{prior_variant_type}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"prior_quality_bin/{prior_quality_bin}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"training_role/{training_role}", **metric_kwargs)
                update_group_metric_store(group_store, key=f"tier/{supervision_tier}", **metric_kwargs)
                update_group_metric_store(baseline_group_store, key=f"generator/{generator_id}", **baseline_metric_kwargs)
                update_group_metric_store(baseline_group_store, key=f"source/{source_name}", **baseline_metric_kwargs)
                update_group_metric_store(baseline_group_store, key=f"prior/{prior_name}", **baseline_metric_kwargs)
                update_group_metric_store(baseline_group_store, key=f"prior_variant_type/{prior_variant_type}", **baseline_metric_kwargs)
                update_group_metric_store(baseline_group_store, key=f"prior_quality_bin/{prior_quality_bin}", **baseline_metric_kwargs)
                update_group_metric_store(baseline_group_store, key=f"training_role/{training_role}", **baseline_metric_kwargs)
                update_group_metric_store(baseline_group_store, key=f"tier/{supervision_tier}", **baseline_metric_kwargs)
                update_group_metric_store(improvement_group_store, key=f"generator/{generator_id}", **improvement_metric_kwargs)
                update_group_metric_store(improvement_group_store, key=f"source/{source_name}", **improvement_metric_kwargs)
                update_group_metric_store(improvement_group_store, key=f"prior/{prior_name}", **improvement_metric_kwargs)
                update_group_metric_store(improvement_group_store, key=f"prior_variant_type/{prior_variant_type}", **improvement_metric_kwargs)
                update_group_metric_store(improvement_group_store, key=f"prior_quality_bin/{prior_quality_bin}", **improvement_metric_kwargs)
                update_group_metric_store(improvement_group_store, key=f"training_role/{training_role}", **improvement_metric_kwargs)
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

                view_rm_mae_delta_value = None
                if view_rm_delta is not None:
                    value = float(view_rm_delta[item_index].detach().cpu().item())
                    view_rm_mae_delta_value = None if math.isnan(value) else value
                effect_bucket = preview_effect_bucket(improvement_metric_kwargs["total_mae"])
                variant_view_bucket = variant_view_store.setdefault(
                    prior_variant_type,
                    {
                        "count": 0.0,
                        "effective_view_supervision_samples": 0.0,
                        "view_rm_delta_sum": 0.0,
                        "view_rm_delta_count": 0.0,
                    },
                )
                variant_view_bucket["count"] += 1.0
                variant_view_bucket["effective_view_supervision_samples"] += float(
                    bool(batch["has_effective_view_supervision"][item_index].item())
                )
                if view_rm_mae_delta_value is not None:
                    variant_view_bucket["view_rm_delta_sum"] += float(view_rm_mae_delta_value)
                    variant_view_bucket["view_rm_delta_count"] += 1.0
                if int(args.val_preview_samples) > 0:
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
                        pair_id=pair_id,
                        target_bundle_id=target_bundle_id,
                        prior_variant_id=prior_variant_id,
                        prior_variant_type=prior_variant_type,
                        prior_quality_bin=prior_quality_bin,
                        training_role=training_role,
                        view_rm_mae_delta=view_rm_mae_delta_value,
                        preview_slot=None,
                        source_name=source_name,
                        material_family=material_family,
                        prior_label=prior_name,
                    )
                    baseline_total = float(baseline_total_mae[item_index].item())
                    refined_total = float(total_mae[item_index].item())
                    preview_candidates.append(
                        {
                            "path": str(preview_path.resolve()),
                            "case_id": case_key,
                            "object_id": object_id,
                            "pair_id": pair_id,
                            "target_bundle_id": target_bundle_id,
                            "prior_variant_id": prior_variant_id,
                            "generator_id": generator_id,
                            "source_name": source_name,
                            "material_family": material_family,
                            "prior_label": prior_name,
                            "prior_variant_type": prior_variant_type,
                            "prior_quality_bin": prior_quality_bin,
                            "training_role": training_role,
                            "effect_bucket": effect_bucket,
                            "baseline_total_mae": baseline_total,
                            "input_prior_total_mae": baseline_total,
                            "refined_total_mae": refined_total,
                            "gain_total": baseline_total - refined_total,
                            "improvement_total": baseline_total - refined_total,
                            "view_rm_mae_delta": view_rm_mae_delta_value,
                        }
                    )

    preview_items = finalize_selected_preview_items(
        output_dir,
        validation_label=validation_label,
        preview_items=preview_candidates,
        mode=str(getattr(args, "val_preview_selection", "balanced_by_variant")),
        max_count=int(args.val_preview_samples),
    )
    preview_paths = [Path(str(item.get("path") or "")) for item in preview_items if item.get("path")]
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
    view_stats_by_variant = {
        variant: {
            "count": int(bucket.get("count", 0.0)),
            "effective_view_supervision_rate": float(
                bucket.get("effective_view_supervision_samples", 0.0) / max(bucket.get("count", 0.0), 1.0)
            ),
            "sampled_view_rm_proxy_delta": (
                float(bucket.get("view_rm_delta_sum", 0.0) / max(bucket.get("view_rm_delta_count", 0.0), 1.0))
                if bucket.get("view_rm_delta_count", 0.0) > 0.0
                else None
            ),
        }
        for variant, bucket in sorted(variant_view_store.items())
    }
    return {
        "loss": mean_losses,
        "uv_mae": mean_uv_mae,
        "baseline_uv_mae": mean_baseline_uv_mae,
        "input_prior_uv_mae": mean_baseline_uv_mae,
        "improvement_uv_mae": mean_improvement,
        "group_metrics": finalize_group_metrics(group_store),
        "baseline_group_metrics": finalize_group_metrics(baseline_group_store),
        "improvement_group_metrics": finalize_group_metrics(improvement_group_store),
        "object_level": finalize_object_improvement_metrics(object_improvement_store),
        "case_level": finalize_case_improvement_metrics(case_improvement_store),
        "batches": int(steps),
        "max_validation_batches": int(args.max_validation_batches),
        "effective_view_supervision_samples": int(uv_mae.get("effective_view_supervision_samples", 0.0)),
        "effective_view_supervision_rate": float(uv_mae.get("effective_view_supervision_samples", 0.0) / max(uv_mae["count"], 1.0)),
        "view_stats_by_variant": view_stats_by_variant,
        "view_consistency_enabled": bool(
            (args.view_consistency_mode != "disabled" or args.enable_sampled_view_rm_loss)
            and max(float(args.view_consistency_weight), float(args.sampled_view_rm_loss_weight)) > 0.0
        ),
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
        "special_metrics": finalize_validation_special_metrics(special_metric_store),
        "residual_gate_diagnostics": finalize_residual_gate_diagnostics(residual_gate_store),
        "residual_gate_cases": residual_gate_cases,
        "preview_paths": [str(path.resolve()) for path in preview_paths],
        "preview_items": preview_items,
        "preview_integrity": build_preview_integrity_report(preview_items),
    }


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_serializable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def compute_validation_selection_metric(
    val_payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    uv_total = float((val_payload.get("uv_mae") or {}).get("total", 0.0) or 0.0)
    input_prior_total = float(
        (
            val_payload.get("input_prior_uv_mae")
            or val_payload.get("baseline_uv_mae")
            or {}
        ).get("total", 0.0)
        or 0.0
    )
    uv_gain = float((val_payload.get("improvement_uv_mae") or {}).get("total", input_prior_total - uv_total) or 0.0)
    if args.validation_selection_metric == "uv_total":
        return uv_total, {
            "uv_total": uv_total,
            "input_prior_total": input_prior_total,
            "uv_gain": uv_gain,
            "selection_metric": uv_total,
            "lower_is_better": True,
        }

    render_proxy = val_payload.get("render_proxy_validation") or {}
    residual_diag = val_payload.get("residual_gate_diagnostics") or {}
    view_penalty = 0.0
    mse_penalty = 0.0
    psnr_penalty = 0.0
    render_guard_available = bool(render_proxy.get("available"))
    if render_guard_available:
        view_penalty = max(0.0, -float(render_proxy.get("view_rm_mae_delta", 0.0) or 0.0))
        mse_penalty = max(0.0, -float(render_proxy.get("proxy_rm_mse_delta", 0.0) or 0.0))
        psnr_penalty = max(0.0, -float(render_proxy.get("proxy_rm_psnr_delta", 0.0) or 0.0))
    regression_penalty = max(
        0.0,
        float(residual_diag.get("regression_rate", 0.0) or 0.0),
    )
    penalty_total = (
        float(args.selection_view_rm_penalty) * view_penalty
        + float(getattr(args, "selection_mse_penalty", 0.5)) * mse_penalty
        + float(args.selection_psnr_penalty) * psnr_penalty
        + float(args.selection_residual_regression_penalty) * regression_penalty
    )
    if args.validation_selection_metric == "gain_render_guarded":
        selection_metric = -uv_gain + penalty_total
    else:
        selection_metric = uv_total + penalty_total
    return selection_metric, {
        "uv_total": uv_total,
        "input_prior_total": input_prior_total,
        "uv_gain": uv_gain,
        "view_penalty": view_penalty,
        "mse_penalty": mse_penalty,
        "psnr_penalty": psnr_penalty,
        "regression_penalty": regression_penalty,
        "penalty_total": penalty_total,
        "render_guard_available": render_guard_available,
        "selection_metric": selection_metric,
        "lower_is_better": True,
    }


def load_compatible_model_state(
    model: MaterialRefiner,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    model_state = model.state_dict()
    compatible_state = {}
    skipped_shape = []
    unexpected = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        compatible_state[key] = value
    missing, load_unexpected = model.load_state_dict(compatible_state, strict=False)
    unexpected.extend(load_unexpected)
    return list(missing), list(unexpected), skipped_shape


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
        if frozen_val_manifest is not None:
            # The frozen manifest already encodes the audited object-level
            # validation order.  Re-balancing again can silently move with-prior
            # samples out of the first validation window and make previews look
            # single-source even when the frozen set is mixed.
            val_balance_summary = {
                "enabled": True,
                "key": str(args.val_balance_key),
                "source": "frozen_manifest_preserved_order",
                "original_records": len(val_dataset.records),
                "selected_records": len(val_dataset.records),
                "max_records": None,
                "records_per_group": int(getattr(args, "val_records_per_balance_group", 0)),
                "expected_groups": list(getattr(args, "val_expected_balance_groups", []) or []),
                "groups": dict(
                    sorted(
                        Counter(
                            record_field_value(record, str(args.val_balance_key))
                            for record in val_dataset.records
                        ).items()
                    )
                ),
                "warnings": [],
            }
        else:
            val_records, val_balance_summary = balance_validation_records(val_dataset.records, args)
            val_balance_summary["source"] = "live_manifest"
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
    global_batch_step: int,
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
    monitored_metric, selection_components = compute_validation_selection_metric(val_payload, args)
    val_payload["selection_metric"] = {
        "mode": str(args.validation_selection_metric),
        **selection_components,
    }
    val_payload["validation_label"] = validation_label
    val_payload["epoch"] = int(epoch)
    val_payload["optimizer_step"] = int(optimizer_step)
    val_payload["global_batch_step"] = int(global_batch_step)
    dataset_record_count = len(getattr(val_loader.dataset, "records", []) or [])
    val_payload["record_count"] = int((val_payload.get("uv_mae") or {}).get("count") or 0)
    val_payload["dataset_record_count"] = int(dataset_record_count)
    val_payload["evaluation_basis"] = (
        f"monitor_val_balanced_{dataset_record_count}"
        if dataset_record_count > 0
        else "monitor_val_balanced_unknown"
    )
    save_json(output_dir / "validation" / f"{validation_label}.json", val_payload)

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
        f"input_prior={format_metric((val_payload.get('input_prior_uv_mae') or val_payload.get('baseline_uv_mae') or {}).get('total'))} "
        f"uv_total={format_metric(val_payload['uv_mae']['total'])} "
        f"improve={format_metric((val_payload.get('improvement_uv_mae') or {}).get('total'))} "
        f"improve_rate={format_metric((val_payload.get('improvement_uv_mae') or {}).get('improvement_rate'), 4)} "
        f"roughness={format_metric(val_payload['uv_mae']['roughness'])} "
        f"metallic={format_metric(val_payload['uv_mae']['metallic'])} "
        f"proxy_view_delta={format_metric((val_payload.get('render_proxy_validation') or {}).get('view_rm_mae_delta'))} "
        f"select={format_metric(monitored_metric)} "
        f"select_mode={args.validation_selection_metric} "
        f"res_change={format_metric((val_payload.get('residual_gate_diagnostics') or {}).get('changed_pixel_rate'), 4)} "
        f"res_reg={format_metric((val_payload.get('residual_gate_diagnostics') or {}).get('regression_rate'), 4)} "
        f"best={format_metric(best_val_metric)} best_epoch={best_epoch} improved={improved}"
    )

    if run is not None:
        preview_items = list(val_payload.get("preview_items", []))
        preview_images = []
        if wandb is not None:
            preview_images = [
                wandb.Image(
                    item["path"],
                    caption=(
                        f"{item.get('object_id')} | "
                        f"pair={item.get('pair_id', 'unknown')} | "
                        f"variant={item.get('prior_variant_type', 'unknown')} | "
                        f"quality={item.get('prior_quality_bin', 'unknown')} | "
                        f"Prior -> Pred MAE "
                        f"{item.get('input_prior_total_mae', item.get('baseline_total_mae', 0.0)):.4f}"
                        f" -> {item.get('refined_total_mae', 0.0):.4f} "
                        f"(UV gain={item.get('gain_total', item.get('improvement_total', 0.0)):+.4f}, "
                        f"view RM delta={format_metric(item.get('view_rm_mae_delta'), 4)}, "
                        f"{item.get('effect_bucket', 'unknown')})"
                    ),
                )
                for item in preview_items[: args.wandb_val_preview_max]
            ]
        improvement_payload = val_payload.get("improvement_uv_mae") or {}
        object_level_payload = val_payload.get("object_level") or {}
        case_level_payload = val_payload.get("case_level") or {}
        log_payload = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "global_batch_step": global_batch_step,
            "lr": learning_rate,
            "best/selection_metric": best_val_metric,
            "best/epoch": best_epoch,
            "val/input_prior_total_mae": (val_payload.get("input_prior_uv_mae") or {}).get("total"),
            "val/refined_total_mae": val_payload["uv_mae"]["total"],
            "val/gain_total": improvement_payload.get("total"),
            "val/effective_view_supervision_rate": val_payload.get("effective_view_supervision_rate", 0.0),
            "val/object_level/avg_improvement_total": object_level_payload.get("avg_improvement_total"),
            "val/object_level/regression_rate": object_level_payload.get("regression_rate"),
            "val/case_level/avg_improvement_total": case_level_payload.get("avg_improvement_total"),
            "val/case_level/regression_rate": case_level_payload.get("regression_rate"),
        }
        group_metrics = val_payload.get("group_metrics") or {}
        baseline_group_metrics = val_payload.get("baseline_group_metrics") or {}
        improvement_group_metrics = val_payload.get("improvement_group_metrics") or {}
        for group_key, metrics in group_metrics.items():
            if not str(group_key).startswith("prior_variant_type/"):
                continue
            variant = safe_wandb_key(str(group_key).split("/", 1)[1])
            baseline_metrics = baseline_group_metrics.get(group_key) or {}
            improvement_metrics = improvement_group_metrics.get(group_key) or {}
            log_payload[f"val/by_variant/{variant}/refined_total_mae"] = metrics.get("total_mae")
            log_payload[f"val/by_variant/{variant}/input_prior_total_mae"] = baseline_metrics.get("total_mae")
            log_payload[f"val/by_variant/{variant}/gain_total"] = improvement_metrics.get("total_mae")
        render_proxy = val_payload.get("render_proxy_validation") or {}
        if bool(render_proxy.get("available")):
            log_payload["val/rm_proxy/view_mae/baseline"] = render_proxy.get("baseline_view_rm_mae")
            log_payload["val/rm_proxy/view_mae/refined"] = render_proxy.get("refined_view_rm_mae")
            log_payload["val/rm_proxy/view_mae/delta"] = render_proxy.get("view_rm_mae_delta")
            log_payload["val/rm_proxy/view_mse/baseline"] = render_proxy.get("baseline_proxy_rm_mse")
            log_payload["val/rm_proxy/view_mse/refined"] = render_proxy.get("refined_proxy_rm_mse")
            log_payload["val/rm_proxy/view_mse/delta"] = render_proxy.get("proxy_rm_mse_delta")
            log_payload["val/rm_proxy/view_psnr/baseline"] = render_proxy.get("baseline_proxy_rm_psnr")
            log_payload["val/rm_proxy/view_psnr/refined"] = render_proxy.get("refined_proxy_rm_psnr")
            log_payload["val/rm_proxy/view_psnr/delta"] = render_proxy.get("proxy_rm_psnr_delta")
        log_payload = add_step_context(
            log_payload,
            epoch=epoch,
            optimizer_step=optimizer_step,
            global_batch_step=global_batch_step,
            learning_rate=learning_rate,
        )
        log_payload = filter_validation_wandb_logs(log_payload)
        sanitized_logs, skipped_logs = sanitize_log_dict(log_payload)
        if skipped_logs:
            print(f"[wandb:skip] validation_label={validation_label} keys={','.join(skipped_logs)}")
        if sanitized_logs:
            run.log(sanitized_logs, step=optimizer_step)
        if preview_images:
            preview_log_payload: dict[str, Any] = {}
            preview_log_payload[f"val_preview/{safe_wandb_key(validation_label)}/cases"] = preview_images
            run.log(preview_log_payload, step=optimizer_step)

    return val_payload, float(best_val_metric), int(best_epoch), improved


def main() -> None:
    print("[startup] imports_ready parse_args_start", flush=True)
    args = parse_args()
    deprecated_noop_flags = []
    if bool(getattr(args, "save_preview_contact_sheet", False)):
        deprecated_noop_flags.append("save_preview_contact_sheet")
        args.save_preview_contact_sheet = False
    if bool(getattr(args, "wandb_log_preview_grid", False)):
        deprecated_noop_flags.append("wandb_log_preview_grid")
        args.wandb_log_preview_grid = False
    if deprecated_noop_flags:
        print(
            "[config:warning] deprecated_noop_flags="
            + ",".join(deprecated_noop_flags)
            + " active_path_uses_single_preview_images_only"
        )
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
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads > 0:
        try:
            torch.set_num_interop_threads(args.torch_num_interop_threads)
        except RuntimeError as exc:
            print(f"[system:warning] set_num_interop_threads_failed={type(exc).__name__}: {exc}")
    device = resolve_device(args)
    amp_dtype = resolve_amp_dtype(args)
    gradient_scaler_enabled = device.startswith("cuda") and amp_dtype == torch.float16
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
    configure_wandb_step_metrics(run)

    model_cfg = build_model_cfg(args)
    model = MaterialRefiner(model_cfg)
    model.to(device)
    if args.init_from_checkpoint is not None and args.resume is not None:
        raise ValueError("--init-from-checkpoint is model-only initialization and cannot be combined with --resume.")
    if args.init_from_checkpoint is not None:
        init_path = args.init_from_checkpoint
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        missing, unexpected, skipped_shape = load_compatible_model_state(model, state_dict)
        print(
            "[init] "
            f"checkpoint={short_path(init_path.resolve())} "
            f"source_epoch={checkpoint.get('epoch', 'n/a') if isinstance(checkpoint, dict) else 'n/a'} "
            f"missing={len(missing)} unexpected={len(unexpected)} skipped_shape={len(skipped_shape)}"
        )
        if missing:
            print(f"[init:warning] missing_keys={','.join(missing[:12])}")
        if unexpected:
            print(f"[init:warning] unexpected_keys={','.join(unexpected[:12])}")
        if skipped_shape:
            print(f"[init:warning] shape_mismatch_skipped={','.join(skipped_shape[:12])}")
    model_info = model_parameter_summary(model)
    runtime_info = system_runtime_info(args, device)
    runtime_info.update(
        {
            "amp_dtype": args.amp_dtype,
            "gradient_scaler_enabled": gradient_scaler_enabled,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
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

    # BF16 has a wide exponent range and should not use loss scaling.  Leaving
    # GradScaler enabled under BF16 produced non-finite grad-norm diagnostics in
    # clean cold-start runs and made it hard to tell whether optimizer steps were
    # actually effective.  Keep scaling for FP16 only.
    scaler = torch.amp.GradScaler("cuda", enabled=gradient_scaler_enabled)
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
                data_state_logs = add_step_context(
                    data_state_logs,
                    epoch=epoch,
                    optimizer_step=optimizer_step,
                    global_batch_step=global_batch_step,
                    learning_rate=current_lr(optimizer),
                )
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
        epoch_had_validation = False
        epoch_validation_improved = False

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
                if not torch.isfinite(losses["total"]).all():
                    object_preview = ",".join(str(item) for item in batch.get("object_id", [])[:4])
                    raise FloatingPointError(
                        "nonfinite_training_loss:"
                        f"epoch={epoch};batch_index={batch_index};objects={object_preview}"
                    )
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
                if not math.isfinite(last_grad_norm):
                    object_preview = ",".join(str(item) for item in batch.get("object_id", [])[:4])
                    raise FloatingPointError(
                        "nonfinite_gradient_norm:"
                        f"epoch={epoch};batch_index={batch_index};optimizer_step={optimizer_step + 1};"
                        f"objects={object_preview}"
                    )

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
                    "train/view_consistency_enabled": bool(
                        (args.view_consistency_mode != "disabled" or args.enable_sampled_view_rm_loss)
                        and max(float(args.view_consistency_weight), float(args.sampled_view_rm_loss_weight)) > 0.0
                    ),
                    "train/batches_per_log_window": interval_steps,
                    "train/samples_per_second": interval_examples / interval_seconds,
                    "train/seconds_per_batch": interval_seconds / max(interval_steps, 1),
                    "throughput/samples_per_second": interval_examples / interval_seconds,
                    "throughput/seconds_per_batch": interval_seconds / max(interval_steps, 1),
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
                interval_logs = add_step_context(
                    interval_logs,
                    epoch=epoch,
                    optimizer_step=optimizer_step,
                    global_batch_step=global_batch_step,
                    learning_rate=current_lr(optimizer),
                    progress_fraction=session_progress,
                )
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
                epoch_had_validation = True
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
                    global_batch_step=global_batch_step,
                    run=run,
                    scheduler=scheduler,
                    learning_rate=current_lr(optimizer),
                    train_total=None,
                    best_val_metric=best_val_metric,
                    best_epoch=best_epoch,
                    validation_label=validation_label,
                )
                refresh_progress_bar(progress_bar)
                epoch_validation_improved = epoch_validation_improved or step_improved
                if args.early_stopping_scope == "validation_event":
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
                        "val_selection_metric": None
                        if step_val_payload is None
                        else (step_val_payload.get("selection_metric") or {}).get("selection_metric"),
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
        train_metrics["view_consistency_enabled"] = bool(
            (args.view_consistency_mode != "disabled" or args.enable_sampled_view_rm_loss)
            and max(float(args.view_consistency_weight), float(args.sampled_view_rm_loss_weight)) > 0.0
        )
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
            epoch_had_validation = True
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
                global_batch_step=global_batch_step,
                run=run,
                scheduler=scheduler,
                learning_rate=current_lr(optimizer),
                train_total=train_metrics["total"],
                best_val_metric=best_val_metric,
                best_epoch=best_epoch,
                validation_label=f"epoch_{epoch:03d}",
            )
            epoch_payload["val"] = val_payload
            epoch_validation_improved = epoch_validation_improved or epoch_improved
            if args.early_stopping_scope == "validation_event":
                if epoch_improved:
                    stale_epochs = 0
                else:
                    stale_epochs += 1
        elif scheduler is not None and optimizer_step > args.warmup_steps:
            scheduler.step(train_metrics["total"])

        if (
            val_loader is not None
            and epoch_had_validation
            and args.early_stopping_scope == "epoch"
        ):
            stale_epochs = 0 if epoch_validation_improved else stale_epochs + 1
            epoch_improved = epoch_validation_improved

        epoch_payload["latest_validation"] = summarize_latest_validation(
            history + [epoch_payload],
            load_validation_events(output_dir),
        )
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
            "val_selection_metric": None
            if val_payload is None
            else (val_payload.get("selection_metric") or {}).get("selection_metric"),
        }
        checkpoint_path = None
        should_save_epoch = epoch % args.save_every == 0 and not args.save_only_best_checkpoint
        should_save_best_epoch = (
            epoch_improved
            and args.validation_steps <= 0
            and args.validation_progress_milestones <= 0
        )
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
        validation_events = load_validation_events(output_dir)
        training_overview_path = write_training_overview(
            output_dir=output_dir,
            args=args,
            history=history,
            train_state=train_state,
            validation_events=validation_events,
            visualization_paths=final_visualization_paths,
        )
        final_artifacts = [
            history_path,
            state_path,
            output_dir / "latest.pt",
        ]
        for path_value in final_visualization_paths.values():
            final_artifacts.append(Path(path_value))
        if training_overview_path is not None:
            final_artifacts.append(training_overview_path)
        best_path = output_dir / "best.pt"
        if best_path.exists():
            final_artifacts.append(best_path)
        if wandb is not None and training_overview_path is not None and training_overview_path.exists():
            run.log(
                {
                    "overview/training": wandb.Html(
                        training_overview_path.read_text(encoding="utf-8")
                    )
                },
                step=optimizer_step,
            )
        if args.wandb_log_artifacts and args.wandb_artifact_policy in {"all", "final", "best_and_final"}:
            log_path_artifact(
                run,
                name=f"{run_name}-final",
                artifact_type="training-run",
                paths=final_artifacts,
            )
        run.finish()
    final_visualization_paths = (
        save_training_visualizations(history, output_dir)
        if args.export_training_curves
        else {}
    )
    training_overview_path = write_training_overview(
        output_dir=output_dir,
        args=args,
        history=history,
        train_state=train_state,
        validation_events=load_validation_events(output_dir),
        visualization_paths=final_visualization_paths,
    )
    print(
        "[final] "
        f"output_dir={output_dir} latest={output_dir / 'latest.pt'} "
        f"best={output_dir / 'best.pt'} history={history_path} "
        f"overview={training_overview_path if training_overview_path is not None else 'n/a'}"
    )


from .cli import parse_args, resolve_amp_dtype, resolve_device, set_seed
from .metrics import (
    add_step_context,
    compute_losses,
    compute_validation_selection_metric,
    configure_wandb_step_metrics,
    filter_train_wandb_logs,
    filter_validation_wandb_logs,
    finalize_render_proxy_metrics,
    finalize_residual_gate_diagnostics,
    finalize_validation_special_metrics,
    get_lpips_model,
    masked_global_ssim_torch,
    per_sample_view_rm_mae_delta,
    psnr_from_mse,
    rm_to_lpips_rgb,
    sample_uv_maps_to_view,
    safe_wandb_key,
    update_render_proxy_metrics,
    update_residual_gate_diagnostics,
    update_validation_special_metrics,
    view_uv_valid_mask,
)
from .preview import (
    build_preview_integrity_report,
    finalize_selected_preview_items,
    preview_effect_bucket,
    save_validation_preview,
    should_select_validation_preview,
)
from .reports import (
    build_variant_summary_rows,
    load_latest_validation_payload,
    load_validation_events,
    save_training_visualizations,
    summarize_latest_validation,
    write_training_evidence_report,
    write_training_overview,
)


if __name__ == "__main__":
    main()
