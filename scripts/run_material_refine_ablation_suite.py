from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--blender-bin", type=str, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--real-render-cases", type=int, default=30)
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
        payload = json.loads(status_path.read_text(encoding="utf-8"))
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
    best_path = run_dir / "best.pt"
    latest_path = run_dir / "latest.pt"
    for path in (best_path, latest_path):
        if path.exists():
            return path
    raise FileNotFoundError(f"missing_checkpoint_under:{run_dir}")


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
        "true" if parse_bool(args.enable_lpips) else "false",
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


def run_panel_stage(args: argparse.Namespace, suite_dir: Path, *, split: str, label: str) -> None:
    metrics_dir = suite_dir / f"{split}_{label}"
    stage_dir = suite_dir / f"{split}_{label}_panels"
    metrics_json = metrics_dir / "metrics.json"
    if not metrics_json.exists():
        write_stage_status(stage_dir, status="skipped", stage_name=f"{split}_{label}_panels", failure_reason="missing_metrics_json")
        return
    if is_stage_complete(stage_dir):
        return
    write_stage_status(stage_dir, status="running", stage_name=f"{split}_{label}_panels")
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
        write_stage_status(stage_dir, status="completed", stage_name=f"{split}_{label}_panels")
    except subprocess.CalledProcessError as exc:
        write_stage_status(stage_dir, status="failed", stage_name=f"{split}_{label}_panels", failure_reason=f"returncode={exc.returncode}")


def run_real_render_stage(args: argparse.Namespace, suite_dir: Path, *, split: str, label: str, metrics_label: str | None = None) -> None:
    metrics_dir = suite_dir / f"{split}_{(metrics_label or label)}"
    metrics_json = metrics_dir / "metrics.json"
    stage_dir = suite_dir / f"{split}_{label}_realrender"
    if not metrics_json.exists():
        write_stage_status(stage_dir, status="skipped", stage_name=f"{split}_{label}_realrender", failure_reason="missing_metrics_json")
        return
    if is_stage_complete(stage_dir):
        return
    write_stage_status(stage_dir, status="running", stage_name=f"{split}_{label}_realrender")
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
        "balanced_by_variant",
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
        write_stage_status(stage_dir, status="completed", stage_name=f"{split}_{label}_realrender")
    except subprocess.CalledProcessError as exc:
        write_stage_status(stage_dir, status="failed", stage_name=f"{split}_{label}_realrender", failure_reason=f"returncode={exc.returncode}")


def aggregate_suite(suite_dir: Path) -> None:
    eval_rows = []
    real_render_rows = []
    stage_rows = []
    for stage_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        status_file = stage_status_path(stage_dir)
        if status_file.exists():
            stage_rows.append(json.loads(status_file.read_text(encoding="utf-8")))
        summary_path = stage_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "real_render_metrics" in summary:
            real_render_rows.append(
                {
                    "stage": stage_dir.name,
                    "evaluation_basis": summary.get("evaluation_basis"),
                    "completed_cases": summary.get("completed_cases"),
                    "failed_cases": summary.get("failed_cases"),
                    "psnr_delta": ((summary.get("real_render_metrics") or {}).get("psnr") or {}).get("delta"),
                    "mse_delta": ((summary.get("real_render_metrics") or {}).get("mse") or {}).get("delta"),
                    "ssim_delta": ((summary.get("real_render_metrics") or {}).get("ssim") or {}).get("delta"),
                    "lpips_delta": ((summary.get("real_render_metrics") or {}).get("lpips") or {}).get("delta"),
                    "summary_json": str(summary_path.resolve()),
                }
            )
        else:
            baseline_roughness = safe_float(summary.get("baseline_roughness_mae"))
            refined_roughness = safe_float(summary.get("refined_roughness_mae"))
            baseline_metallic = safe_float(summary.get("baseline_metallic_mae"))
            refined_metallic = safe_float(summary.get("refined_metallic_mae"))
            eval_rows.append(
                {
                    "stage": stage_dir.name,
                    "evaluation_basis": summary.get("evaluation_basis"),
                    "gain_total": summary.get("gain_total"),
                    "roughness_gain": (
                        None
                        if baseline_roughness is None or refined_roughness is None
                        else baseline_roughness - refined_roughness
                    ),
                    "metallic_gain": (
                        None
                        if baseline_metallic is None or refined_metallic is None
                        else baseline_metallic - refined_metallic
                    ),
                    "case_regression_rate": summary.get("regression_rate"),
                    "object_regression_rate": (summary.get("object_level") or {}).get("regression_rate"),
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
        "| stage | evaluation_basis | gain_total | roughness_gain | metallic_gain | case_regression | object_regression |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in eval_rows:
        lines.append(
            "| {stage} | {evaluation_basis} | {gain_total:.6f} | {roughness_gain:.6f} | {metallic_gain:.6f} | {case_regression_rate:.6f} | {object_regression_rate:.6f} |".format(
                stage=row["stage"],
                evaluation_basis=row.get("evaluation_basis") or "n/a",
                gain_total=float(row.get("gain_total") or 0.0),
                roughness_gain=float(row.get("roughness_gain") or 0.0),
                metallic_gain=float(row.get("metallic_gain") or 0.0),
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


def main() -> None:
    args = parse_args()
    suite_dir = args.output_dir or (args.run_dir / "post_train_suite")
    suite_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(args.run_dir)
    run_eval_stage(args, suite_dir, checkpoint, split="val", variant="ours_full", label="ours_full")
    run_eval_stage(args, suite_dir, checkpoint, split="test", variant="ours_full", label="ours_full")
    run_eval_stage(
        args,
        suite_dir,
        checkpoint,
        split="val",
        variant="ours_full",
        label="ours_full_renderbasis",
        artifact_objects_override=0,
    )
    run_eval_stage(
        args,
        suite_dir,
        checkpoint,
        split="test",
        variant="ours_full",
        label="ours_full_renderbasis",
        artifact_objects_override=0,
    )
    for variant in ("no_prior_refiner", "no_residual_refiner", "no_view_refiner", "scalar_broadcast"):
        run_eval_stage(args, suite_dir, checkpoint, split="val", variant=variant, label=variant)
        run_eval_stage(args, suite_dir, checkpoint, split="test", variant=variant, label=variant)
    for kernel in (3, 5, 9, 15):
        run_eval_stage(
            args,
            suite_dir,
            checkpoint,
            split="val",
            variant="prior_smoothing",
            label=f"prior_smoothing_k{kernel}",
            extra=["--prior-smoothing-kernel", str(kernel)],
        )
        run_eval_stage(
            args,
            suite_dir,
            checkpoint,
            split="test",
            variant="prior_smoothing",
            label=f"prior_smoothing_k{kernel}",
            extra=["--prior-smoothing-kernel", str(kernel)],
        )
    run_panel_stage(args, suite_dir, split="val", label="ours_full")
    run_panel_stage(args, suite_dir, split="test", label="ours_full")
    run_real_render_stage(args, suite_dir, split="val", label="ours_full", metrics_label="ours_full_renderbasis")
    run_real_render_stage(args, suite_dir, split="test", label="ours_full", metrics_label="ours_full_renderbasis")
    aggregate_suite(suite_dir)


if __name__ == "__main__":
    main()
