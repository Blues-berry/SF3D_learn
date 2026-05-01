from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "material_refine_train_r_v2_1_trainV5_plus_a_full.yaml"
DEFAULT_OUTPUT_CONFIG = REPO_ROOT / "configs" / "material_refine_train_r_v2_1_trainV5_plus_a_withprior_iter1.yaml"
DEFAULT_OUTPUT_LAUNCHER = REPO_ROOT / "train" / "trainV5_plus_a_track" / "r_v2_1_trainV5_plus_a_withprior_iter1.sh"
DEFAULT_OUTPUT_RATIONALE = REPO_ROOT / "train" / "trainV5_plus_a_track" / "iter1_rationale.json"
DEFAULT_ITER1_OUTPUT_DIR = "output/material_refine_trainV5_plus_a_track_withprior_iter1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build the next A-track with-prior-focused iter1 config and launcher.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--output-launcher", type=Path, default=DEFAULT_OUTPUT_LAUNCHER)
    parser.add_argument("--output-rationale", type=Path, default=DEFAULT_OUTPUT_RATIONALE)
    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    print(
        "DEPRECATED: keep A-track training config changes in configs/ or TrainV5 training parameters; do not add new iterN dataset scripts.",
        file=sys.stderr,
    )
    args = parse_args()
    cfg = OmegaConf.load(args.base_config)
    fallback_reason: list[str] = []

    benchmark_val = load_json_if_exists(args.run_dir / "post_train_suite" / "val_ours_full_metrics" / "summary.json")
    benchmark_test = load_json_if_exists(args.run_dir / "post_train_suite" / "test_ours_full_metrics" / "summary.json")
    benchmark_val_realrender = load_json_if_exists(args.run_dir / "post_train_suite" / "val_ours_full_realrender" / "summary.json")
    benchmark_test_realrender = load_json_if_exists(args.run_dir / "post_train_suite" / "test_ours_full_realrender" / "summary.json")
    if benchmark_val is None:
        benchmark_val = load_json_if_exists(args.run_dir / "post_train_suite" / "val_ours_full" / "summary.json")
    if benchmark_test is None:
        fallback_reason.append("missing_benchmark_test_full")
    if benchmark_val_realrender is None:
        fallback_reason.append("missing_benchmark_val_realrender_30")
    if benchmark_test_realrender is None:
        fallback_reason.append("missing_benchmark_test_realrender_30")
    if benchmark_val is None:
        raise FileNotFoundError(f"missing_benchmark_val_summary_under:{args.run_dir}")

    updates = {
        "cuda_device_index": 1,
        "output_dir": DEFAULT_ITER1_OUTPUT_DIR,
        "validation_selection_metric": "hybrid_potential_gain_render_guarded",
        "selection_metric_near_gt_regression_multiplier": 2.0,
        "selection_metric_withprior_regression_multiplier": 1.5,
        "prior_confidence_gate_strength": 0.50,
        "prior_confidence_gate_strength_roughness": 0.15,
        "prior_confidence_gate_strength_metallic": 0.50,
        "prior_dropout_start_prob": 0.02,
        "prior_dropout_end_prob": 0.08,
        "prior_dropout_warmup_epochs": 6,
        "train_variant_loss_weights": "near_gt_prior=1.8,mild_gap_prior=1.6,medium_gap_prior=1.4,large_gap_prior=1.2,no_prior_bootstrap=0.7",
        "train_view_sample_count": 10,
        "train_min_hard_views": 5,
        "val_view_sample_count": 12,
        "residual_gate_bias": -1.25,
        "min_residual_gate": 0.02,
        "max_residual_gate": 0.55,
        "boundary_residual_suppression_strength": 0.35,
        "view_uncertainty_residual_suppression_strength": 0.25,
        "bleed_risk_residual_suppression_strength": 0.35,
        "topology_residual_suppression_strength": 0.15,
        "roughness_suppression_scale": 0.5,
        "withprior_gate_floor": 0.10,
        "withprior_roughness_gate_floor": 0.18,
        "near_gt_gate_floor": 0.06,
        "render_gate_blend_floor": 0.35,
        "monitor_val_target_records": 160,
        "hybrid_prior_percentile_fixed_weight": 0.7,
        "hybrid_prior_percentile_empirical_weight": 0.3,
        "hybrid_prior_min_potential": 0.10,
        "previous_baseline_uv_total": 0.25475716814398763,
        "previous_baseline_uv_gain": 0.1101572948275134,
        "previous_baseline_rm_proxy_view_mae_delta": 0.060564438506714946,
        "previous_baseline_rm_proxy_view_mse_delta": 0.07566712377925434,
        "previous_baseline_rm_proxy_view_psnr_delta": 5.038672034913205,
        "roughness_channel_weight": 1.35,
        "metallic_channel_weight": 1.0,
        "edge_aware_weight": 0.32,
        "gradient_preservation_weight": 0.50,
        "tracker_group": "r-v2-1-withprior-iter1",
        "tracker_run_name": args.output_config.stem,
    }
    for key, value in updates.items():
        cfg[key] = value

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(args.output_config))

    args.output_launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {REPO_ROOT}",
            f"python scripts/train_material_refiner.py --config {args.output_config}",
            "",
        ]
    )
    args.output_launcher.write_text(launcher, encoding="utf-8")
    args.output_launcher.chmod(0o755)

    rationale = {
        "run_dir": str(args.run_dir.resolve()),
        "base_config": str(args.base_config.resolve()),
        "output_config": str(args.output_config.resolve()),
        "output_launcher": str(args.output_launcher.resolve()),
        "generated_at_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "fallback_reason": fallback_reason,
        "benchmarks_used": {
            "benchmark_val_full": benchmark_val.get("evaluation_basis"),
            "benchmark_test_full": (benchmark_test or {}).get("evaluation_basis"),
            "benchmark_val_realrender": (benchmark_val_realrender or {}).get("evaluation_basis"),
            "benchmark_test_realrender": (benchmark_test_realrender or {}).get("evaluation_basis"),
        },
        "current_results_snapshot": {
            "benchmark_val_gain_total": benchmark_val.get("gain_total"),
            "benchmark_val_avg_improvement_roughness": benchmark_val.get("avg_improvement_roughness"),
            "benchmark_val_avg_improvement_metallic": benchmark_val.get("avg_improvement_metallic"),
            "benchmark_val_roughness_to_metallic_gain_ratio": benchmark_val.get("roughness_to_metallic_gain_ratio"),
            "benchmark_val_regression_rate": benchmark_val.get("regression_rate"),
            "benchmark_val_object_regression_rate": (benchmark_val.get("object_level") or {}).get("regression_rate"),
        },
        "iter1_updates": updates,
        "objective": {
            "with_prior_gain": "increase mild/medium/large gains without regressing near_gt_prior",
            "roughness_balance": "raise roughness improvement so it no longer trails metallic improvement by a large margin",
            "selection": "choose checkpoints using equal-weight prior-aware potential-normalized gain with render and regression penalties",
        },
    }
    save_json(args.output_rationale, rationale)
    print(json.dumps({"iter1_builder": rationale}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
