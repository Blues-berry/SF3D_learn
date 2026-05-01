from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.experiment import make_json_serializable, maybe_init_wandb, sanitize_log_dict, wandb

DEFAULT_BLENDER_BIN = "/4T/CXY/Neural_Gaffer_original/scripts/Objavarse_rendering/blender-3.2.2-linux-x64/blender"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the resumable post-train ablation and benchmark suite.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cuda-device-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--max-artifact-objects", type=int, default=24)
    parser.add_argument("--enable-lpips", type=str, default="true")
    parser.add_argument("--lpips-max-images", type=int, default=64)
    parser.add_argument("--blender-bin", type=str, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--real-render-cases", type=int, default=30)
    parser.add_argument("--wandb-benchmark-upload", type=str, default="false")
    parser.add_argument("--realrender-upload-policy", choices=["none", "grouped_30_case"], default="grouped_30_case")
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_serializable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_command(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, check=True, cwd=str(cwd))


def stage_status_path(stage_dir: Path) -> Path:
    return stage_dir / "stage_status.json"


def is_stage_complete(stage_dir: Path) -> bool:
    status_path = stage_status_path(stage_dir)
    if not status_path.exists():
        return False
    try:
        payload = load_json(status_path)
    except Exception:
        return False
    return payload.get("status") == "completed"


def write_stage_status(stage_dir: Path, *, status: str, stage_name: str, failure_reason: str | None = None) -> None:
    payload = {
        "stage_name": stage_name,
        "status": status,
        "failure_reason": failure_reason,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json(stage_status_path(stage_dir), payload)


def resolve_checkpoint(run_dir: Path) -> Path:
    for path in (run_dir / "best.pt", run_dir / "latest.pt"):
        if path.exists():
            return path
    raise FileNotFoundError(f"missing_checkpoint_under:{run_dir}")


def load_train_args(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "train_args.json"
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def run_eval_stage(
    args: argparse.Namespace,
    suite_dir: Path,
    checkpoint: Path,
    *,
    split: str,
    variant: str,
    label: str,
    extra: list[str] | None = None,
    artifact_objects_override: int | None = None,
    enable_lpips_override: bool | None = None,
    lpips_max_images_override: int | None = None,
) -> None:
    stage_dir = suite_dir / f"{split}_{label}"
    if is_stage_complete(stage_dir):
        return
    write_stage_status(stage_dir, status="running", stage_name=f"{split}_{label}")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_material_refiner.py"),
        "--config",
        str(args.config),
        "--manifest",
        str(args.manifest),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(stage_dir),
        "--split",
        split,
        "--cuda-device-index",
        str(args.cuda_device_index),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--eval-variant",
        variant,
        "--report-to",
        "none",
        "--log-every",
        "10",
        "--max-artifact-objects",
        str(args.max_artifact_objects if artifact_objects_override is None else artifact_objects_override),
        "--enable-lpips",
        "true" if (parse_bool(args.enable_lpips) if enable_lpips_override is None else bool(enable_lpips_override)) else "false",
        "--lpips-max-images",
        str(max(int(args.lpips_max_images if lpips_max_images_override is None else lpips_max_images_override), 0)),
    ]
    if extra:
        command.extend(extra)
    if int(args.eval_max_samples) > 0:
        command.extend(["--max-samples", str(int(args.eval_max_samples))])
    try:
        run_command(command, cwd=REPO_ROOT)
        write_stage_status(stage_dir, status="completed", stage_name=f"{split}_{label}")
    except subprocess.CalledProcessError as exc:
        write_stage_status(stage_dir, status="failed", stage_name=f"{split}_{label}", failure_reason=f"returncode={exc.returncode}")


def run_panel_stage(args: argparse.Namespace, suite_dir: Path, *, split: str, metrics_label: str, label: str) -> None:
    metrics_dir = suite_dir / f"{split}_{metrics_label}"
    stage_dir = suite_dir / f"{split}_{label}"
    metrics_json = metrics_dir / "metrics.json"
    if not metrics_json.exists():
        write_stage_status(stage_dir, status="skipped", stage_name=stage_dir.name, failure_reason="missing_metrics_json")
        return
    if is_stage_complete(stage_dir):
        return
    write_stage_status(stage_dir, status="running", stage_name=stage_dir.name)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_material_validation_comparison_panels.py"),
        "--manifest",
        str(args.manifest),
        "--metrics",
        str(metrics_json),
        "--output-dir",
        str(stage_dir),
        "--selection-mode",
        "balanced_by_variant",
        "--max-panels",
        "30",
        "--panel-size",
        "192",
        "--report-to",
        "none",
    ]
    try:
        run_command(command, cwd=REPO_ROOT)
        write_stage_status(stage_dir, status="completed", stage_name=stage_dir.name)
    except subprocess.CalledProcessError as exc:
        write_stage_status(stage_dir, status="failed", stage_name=stage_dir.name, failure_reason=f"returncode={exc.returncode}")


def run_real_render_stage(args: argparse.Namespace, suite_dir: Path, *, split: str, label: str, metrics_label: str) -> None:
    metrics_dir = suite_dir / f"{split}_{metrics_label}"
    metrics_json = metrics_dir / "metrics.json"
    stage_dir = suite_dir / f"{split}_{label}"
    if not metrics_json.exists():
        write_stage_status(stage_dir, status="skipped", stage_name=stage_dir.name, failure_reason="missing_metrics_json")
        return
    if is_stage_complete(stage_dir):
        return
    write_stage_status(stage_dir, status="running", stage_name=stage_dir.name)
    selection_mode = "random_balanced_by_variant" if args.realrender_upload_policy == "grouped_30_case" else "balanced_by_variant"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_material_refiner_real_render.py"),
        "--manifest",
        str(args.manifest),
        "--metrics",
        str(metrics_json),
        "--output-dir",
        str(stage_dir),
        "--split",
        split,
        "--selection-mode",
        selection_mode,
        "--max-cases",
        str(args.real_render_cases),
        "--hdri-preset",
        "hdri1",
        "--blender-bin",
        args.blender_bin,
        "--cuda-device-index",
        str(args.cuda_device_index),
        "--enable-lpips",
        "true" if parse_bool(args.enable_lpips) else "false",
    ]
    try:
        run_command(command, cwd=REPO_ROOT)
        write_stage_status(stage_dir, status="completed", stage_name=stage_dir.name)
    except subprocess.CalledProcessError as exc:
        write_stage_status(stage_dir, status="failed", stage_name=stage_dir.name, failure_reason=f"returncode={exc.returncode}")


def aggregate_suite(suite_dir: Path) -> dict[str, Any]:
    eval_rows = []
    real_render_rows = []
    stage_rows = []
    for stage_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        status_file = stage_status_path(stage_dir)
        if status_file.exists():
            stage_rows.append(load_json(status_file))
        summary_path = stage_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        if "real_render_metrics" in summary:
            metrics = summary.get("real_render_metrics") or {}
            real_render_rows.append(
                {
                    "stage": stage_dir.name,
                    "evaluation_basis": summary.get("evaluation_basis"),
                    "completed_cases": summary.get("completed_cases"),
                    "failed_cases": summary.get("failed_cases"),
                    "psnr_delta": summary.get("psnr_delta", ((metrics.get("psnr") or {}).get("delta"))),
                    "mse_delta": summary.get("mse_delta", ((metrics.get("mse") or {}).get("delta"))),
                    "ssim_delta": summary.get("ssim_delta", ((metrics.get("ssim") or {}).get("delta"))),
                    "lpips_delta": summary.get("lpips_delta", ((metrics.get("lpips") or {}).get("delta"))),
                    "by_prior_variant_type": summary.get("by_prior_variant_type"),
                    "summary_json": str(summary_path.resolve()),
                }
            )
            continue
        eval_rows.append(
            {
                "stage": stage_dir.name,
                "evaluation_basis": summary.get("evaluation_basis"),
                "gain_total": summary.get("gain_total"),
                "roughness_gain": summary.get("avg_improvement_roughness"),
                "metallic_gain": summary.get("avg_improvement_metallic"),
                "roughness_to_metallic_gain_ratio": summary.get("roughness_to_metallic_gain_ratio"),
                "case_regression_rate": summary.get("regression_rate"),
                "object_regression_rate": (summary.get("object_level") or {}).get("regression_rate"),
                "by_prior_variant_type": (summary.get("metrics_by_group") or {}).get("by_prior_variant_type"),
                "rgb_proxy_by_prior_variant_type": summary.get("rgb_proxy_by_prior_variant_type"),
                "summary_json": str(summary_path.resolve()),
            }
        )
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": stage_rows,
        "eval_summaries": eval_rows,
        "real_render_summaries": real_render_rows,
    }
    save_json(suite_dir / "ablation_matrix.json", payload)
    lines = [
        "# Material Refine Ablation Suite",
        "",
        "## Eval Summaries",
        "",
        "| stage | evaluation_basis | gain_total | roughness_gain | metallic_gain | gain_ratio | case_regression | object_regression |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in eval_rows:
        lines.append(
            "| {stage} | {evaluation_basis} | {gain_total:.6f} | {roughness_gain:.6f} | {metallic_gain:.6f} | {gain_ratio:.6f} | {case_regression_rate:.6f} | {object_regression_rate:.6f} |".format(
                stage=row["stage"],
                evaluation_basis=row.get("evaluation_basis") or "n/a",
                gain_total=float(row.get("gain_total") or 0.0),
                roughness_gain=float(row.get("roughness_gain") or 0.0),
                metallic_gain=float(row.get("metallic_gain") or 0.0),
                gain_ratio=float(row.get("roughness_to_metallic_gain_ratio") or 0.0),
                case_regression_rate=float(row.get("case_regression_rate") or 0.0),
                object_regression_rate=float(row.get("object_regression_rate") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Real Render Summaries",
            "",
            "| stage | evaluation_basis | completed_cases | failed_cases | psnr_delta | mse_delta | ssim_delta | lpips_delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in real_render_rows:
        lines.append(
            "| {stage} | {evaluation_basis} | {completed_cases} | {failed_cases} | {psnr_delta:.6f} | {mse_delta:.6f} | {ssim_delta:.6f} | {lpips_delta:.6f} |".format(
                stage=row["stage"],
                evaluation_basis=row.get("evaluation_basis") or "n/a",
                completed_cases=int(row.get("completed_cases") or 0),
                failed_cases=int(row.get("failed_cases") or 0),
                psnr_delta=float(row.get("psnr_delta") or 0.0),
                mse_delta=float(row.get("mse_delta") or 0.0),
                ssim_delta=float(row.get("ssim_delta") or 0.0),
                lpips_delta=float(row.get("lpips_delta") or 0.0),
            )
        )
    (suite_dir / "ablation_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(suite_dir / "suite_state.json", payload)
    return payload


def _rgb_proxy_variant_logs(summary: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    logs: dict[str, Any] = {}
    for variant, metrics in sorted((summary.get("rgb_proxy_by_prior_variant_type") or {}).items()):
        safe_variant = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(variant))
        metric_map = metrics.get("metrics_main") or {}
        for metric_name, field in (
            ("psnr", "proxy_render_psnr"),
            ("mse", "proxy_render_mse"),
            ("ssim", "proxy_render_ssim"),
            ("lpips", "proxy_render_lpips"),
        ):
            pair = metric_map.get(field) or {}
            logs[f"{prefix}/rgb_proxy/{safe_variant}/{metric_name}_delta"] = pair.get("delta")
    return logs


def _build_disagreement_html(summary: dict[str, Any], *, title: str) -> Any | None:
    if wandb is None:
        return None
    cases = ((summary.get("metrics_diagnostics") or {}).get("rgb_proxy_disagreement_cases") or [])[:40]
    if not cases:
        return None
    rows = []
    for item in cases:
        rows.append(
            "<tr>"
            f"<td>{item.get('object_id')}</td>"
            f"<td>{item.get('pair_id')}</td>"
            f"<td>{item.get('prior_variant_type')}</td>"
            f"<td>{item.get('rm_proxy_gain_total')}</td>"
            f"<td>{item.get('rgb_proxy_psnr_delta')}</td>"
            f"<td>{item.get('rgb_proxy_ssim_delta')}</td>"
            f"<td>{item.get('rgb_proxy_lpips_delta')}</td>"
            "</tr>"
        )
    html_doc = "\n".join(
        [
            "<html><body>",
            f"<h2>{title}</h2>",
            "<table border='1' cellspacing='0' cellpadding='4'>",
            "<tr><th>object</th><th>pair</th><th>variant</th><th>rm_gain</th><th>rgb_psnr_delta</th><th>rgb_ssim_delta</th><th>rgb_lpips_delta</th></tr>",
            *rows,
            "</table></body></html>",
        ]
    )
    return wandb.Html(html_doc)


def _safe_wandb_component(value: Any) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "unknown"))


def _metric_delta(prior_metrics: dict[str, Any], ours_metrics: dict[str, Any], name: str, *, higher_is_better: bool) -> float | None:
    prior_value = prior_metrics.get(name)
    ours_value = ours_metrics.get(name)
    if prior_value is None or ours_value is None:
        return None
    return float(ours_value - prior_value) if higher_is_better else float(prior_value - ours_value)


def init_benchmark_wandb(args: argparse.Namespace, *, run_dir: Path) -> Any | None:
    if not parse_bool(args.wandb_benchmark_upload):
        return None
    train_args = load_train_args(run_dir)
    project = str(train_args.get("tracker_project_name") or "stable-fast-3d-material-refine-trainv5-a-track")
    group = str(train_args.get("tracker_group") or "material-refine") + "-benchmark"
    name = f"{run_dir.name}-benchmark"
    config = {
        "run_dir": str(run_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "config": str(args.config.resolve()),
        "cuda_device_index": int(args.cuda_device_index),
        "real_render_cases": int(args.real_render_cases),
        "lpips_max_images": int(args.lpips_max_images),
        "realrender_upload_policy": str(args.realrender_upload_policy),
    }
    return maybe_init_wandb(
        enabled=True,
        project=project,
        job_type="benchmark",
        config=config,
        mode=str(train_args.get("wandb_mode") or "auto"),
        name=name,
        group=group,
        tags=["material-refine", "benchmark", "trainv5-a-track"],
        dir_path=train_args.get("wandb_dir"),
    )


def upload_benchmark_outputs(run: Any | None, *, suite_dir: Path) -> None:
    if run is None or wandb is None:
        return
    scalar_logs: dict[str, Any] = {}
    for split in ("val", "test"):
        metrics_stage = suite_dir / f"{split}_ours_full_metrics" / "summary.json"
        if metrics_stage.exists():
            summary = load_json(metrics_stage)
            scalar_logs[f"benchmark/{split}_full/gain_total"] = summary.get("gain_total")
            scalar_logs[f"benchmark/{split}_full/avg_improvement_roughness"] = summary.get("avg_improvement_roughness")
            scalar_logs[f"benchmark/{split}_full/avg_improvement_metallic"] = summary.get("avg_improvement_metallic")
            scalar_logs[f"benchmark/{split}_full/roughness_to_metallic_gain_ratio"] = summary.get("roughness_to_metallic_gain_ratio")
            scalar_logs[f"benchmark/{split}_full/regression_rate"] = summary.get("regression_rate")
            scalar_logs[f"benchmark/{split}_full/object_regression_rate"] = (summary.get("object_level") or {}).get("regression_rate")
            scalar_logs.update(_rgb_proxy_variant_logs(summary, prefix=f"benchmark/{split}_full"))
            disagreement_html = _build_disagreement_html(summary, title=f"{split} rgb_proxy disagreement cases")
            if disagreement_html is not None:
                run.log({f"benchmark/{split}_full/rgb_proxy_disagreement_cases": disagreement_html})
        realrender_stage = suite_dir / f"{split}_ours_full_realrender" / "summary.json"
        if realrender_stage.exists():
            summary = load_json(realrender_stage)
            metrics = summary.get("real_render_metrics") or {}
            prior_distribution = summary.get("selected_prior_distribution") or {}
            selected_distribution_total = max(sum(int(count or 0) for count in prior_distribution.values()), 1)
            for metric_name in ("psnr", "mse", "ssim", "lpips"):
                scalar_logs[f"benchmark/{split}_realrender_30/{metric_name}_delta"] = summary.get(
                    f"{metric_name}_delta",
                    ((metrics.get(metric_name) or {}).get("delta")),
                )
            for variant, count in prior_distribution.items():
                safe_variant = _safe_wandb_component(variant)
                scalar_logs[f"benchmark/{split}_realrender_30/prior_distribution/{safe_variant}_count"] = count
                scalar_logs[f"benchmark/{split}_realrender_30/prior_distribution/{safe_variant}_rate"] = float(count) / float(selected_distribution_total)
            if str(summary.get("selection_mode")) == "random_balanced_by_variant" and parse_bool(True):
                grouped_images: dict[str, list[Any]] = {}
                comparison_table = wandb.Table(
                    columns=[
                        "slot",
                        "object_id",
                        "pair_id",
                        "prior_variant_type",
                        "prior_quality_bin",
                        "case_gain_total",
                        "psnr_delta",
                        "ssim_delta",
                        "mse_delta",
                        "lpips_delta",
                        "prior_distribution",
                        "comparison_panel",
                        "gt",
                        "prior",
                        "ours",
                        "prior_error",
                        "ours_error",
                    ]
                )
                for case in summary.get("cases") or []:
                    variant = _safe_wandb_component(case.get("prior_variant_type") or "unknown")
                    paths = case.get("paths") or {}
                    distribution_text = str(
                        case.get("selected_prior_distribution_text")
                        or summary.get("selected_prior_distribution_text")
                        or "unknown"
                    )
                    prior_metrics = case.get("prior_metrics") or {}
                    ours_metrics = case.get("ours_metrics") or {}
                    psnr_delta = _metric_delta(prior_metrics, ours_metrics, "psnr", higher_is_better=True)
                    ssim_delta = _metric_delta(prior_metrics, ours_metrics, "ssim", higher_is_better=True)
                    mse_delta = _metric_delta(prior_metrics, ours_metrics, "mse", higher_is_better=False)
                    lpips_delta = _metric_delta(prior_metrics, ours_metrics, "lpips", higher_is_better=False)
                    comparison_table.add_data(
                        case.get("selection_slot"),
                        case.get("object_id"),
                        case.get("pair_id"),
                        case.get("prior_variant_type"),
                        case.get("prior_quality_bin"),
                        case.get("case_gain_total"),
                        psnr_delta,
                        ssim_delta,
                        mse_delta,
                        lpips_delta,
                        distribution_text,
                        wandb.Image(paths["comparison_panel"]) if paths.get("comparison_panel") else None,
                        wandb.Image(paths["gt"]) if paths.get("gt") else None,
                        wandb.Image(paths["prior"]) if paths.get("prior") else None,
                        wandb.Image(paths["ours"]) if paths.get("ours") else None,
                        wandb.Image(paths["prior_error"]) if paths.get("prior_error") else None,
                        wandb.Image(paths["ours_error"]) if paths.get("ours_error") else None,
                    )
                    caption = (
                        f"{case.get('object_id')} | {case.get('pair_id')} | "
                        f"gain={float(case.get('case_gain_total') or 0.0):+.4f} | "
                        f"selected prior distribution: {distribution_text}"
                    )
                    for label in ("comparison_panel", "gt", "prior", "ours", "prior_error", "ours_error"):
                        image_path = paths.get(label)
                        if not image_path:
                            continue
                        grouped_images.setdefault(
                            f"benchmark/{split}_realrender_30/{variant}/{label}",
                            [],
                        ).append(wandb.Image(image_path, caption=caption))
                run.log({f"benchmark/{split}_realrender_30/blender_rgb_comparisons": comparison_table})
                for key, images in grouped_images.items():
                    run.log({key: images})
    scalar_logs, skipped = sanitize_log_dict(scalar_logs)
    if skipped:
        print(f"[wandb:skip] benchmark_keys={','.join(sorted(skipped.keys()))}")
    if scalar_logs:
        run.log(scalar_logs)


def main() -> None:
    args = parse_args()
    suite_dir = args.output_dir or (args.run_dir / "post_train_suite")
    suite_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(args.run_dir)

    run_eval_stage(
        args,
        suite_dir,
        checkpoint,
        split="val",
        variant="ours_full",
        label="ours_full_metrics",
        enable_lpips_override=False,
        lpips_max_images_override=0,
    )
    run_eval_stage(
        args,
        suite_dir,
        checkpoint,
        split="val",
        variant="ours_full",
        label="ours_full_lpips",
        artifact_objects_override=0,
        enable_lpips_override=True,
    )
    run_eval_stage(
        args,
        suite_dir,
        checkpoint,
        split="test",
        variant="ours_full",
        label="ours_full_metrics",
        enable_lpips_override=False,
        lpips_max_images_override=0,
    )
    run_eval_stage(
        args,
        suite_dir,
        checkpoint,
        split="test",
        variant="ours_full",
        label="ours_full_lpips",
        artifact_objects_override=0,
        enable_lpips_override=True,
    )
    run_eval_stage(
        args,
        suite_dir,
        checkpoint,
        split="val",
        variant="ours_full",
        label="ours_full_renderbasis",
        artifact_objects_override=0,
        enable_lpips_override=False,
        lpips_max_images_override=0,
    )
    run_eval_stage(
        args,
        suite_dir,
        checkpoint,
        split="test",
        variant="ours_full",
        label="ours_full_renderbasis",
        artifact_objects_override=0,
        enable_lpips_override=False,
        lpips_max_images_override=0,
    )
    for variant in ("no_prior_refiner", "no_residual_refiner", "no_view_refiner", "scalar_broadcast"):
        run_eval_stage(args, suite_dir, checkpoint, split="val", variant=variant, label=variant, enable_lpips_override=False, lpips_max_images_override=0)
        run_eval_stage(args, suite_dir, checkpoint, split="test", variant=variant, label=variant, enable_lpips_override=False, lpips_max_images_override=0)
    for kernel in (3, 5, 9, 15):
        run_eval_stage(
            args,
            suite_dir,
            checkpoint,
            split="val",
            variant="prior_smoothing",
            label=f"prior_smoothing_k{kernel}",
            extra=["--prior-smoothing-kernel", str(kernel)],
            enable_lpips_override=False,
            lpips_max_images_override=0,
        )
        run_eval_stage(
            args,
            suite_dir,
            checkpoint,
            split="test",
            variant="prior_smoothing",
            label=f"prior_smoothing_k{kernel}",
            extra=["--prior-smoothing-kernel", str(kernel)],
            enable_lpips_override=False,
            lpips_max_images_override=0,
        )
    run_panel_stage(args, suite_dir, split="val", metrics_label="ours_full_metrics", label="ours_full_panels")
    run_panel_stage(args, suite_dir, split="test", metrics_label="ours_full_metrics", label="ours_full_panels")
    run_real_render_stage(args, suite_dir, split="val", label="ours_full_realrender", metrics_label="ours_full_renderbasis")
    run_real_render_stage(args, suite_dir, split="test", label="ours_full_realrender", metrics_label="ours_full_renderbasis")
    aggregate_suite(suite_dir)
    benchmark_run = init_benchmark_wandb(args, run_dir=args.run_dir)
    try:
        upload_benchmark_outputs(benchmark_run, suite_dir=suite_dir)
    finally:
        if benchmark_run is not None:
            benchmark_run.finish()


if __name__ == "__main__":
    main()
