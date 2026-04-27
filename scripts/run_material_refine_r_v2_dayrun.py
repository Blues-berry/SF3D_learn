from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
PHASE_ORDER = ["phase0", "phase1", "phase2", "phase3", "phase4", "phase5"]
PROJECT_NAME = "stable-fast-3d-material-refine-r-v2-dayrun"
GROUP_NAME = "r-v2-dayrun"
TAGS = "r-only,input-prior,acceptance,quick-ablation,rehearsal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the gated R-v2 one-day acceptance + rehearsal pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/material_refine_r_v2_dayrun"))
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--start-phase", choices=PHASE_ORDER, default="phase0")
    parser.add_argument("--max-phase", choices=PHASE_ORDER + ["all"], default="all")
    parser.add_argument("--acceptance-epochs", type=int, default=3)
    parser.add_argument("--ablation-epochs", type=int, default=2)
    parser.add_argument("--rehearsal-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--atlas-size", type=int, default=256)
    parser.add_argument("--buffer-resolution", type=int, default=128)
    parser.add_argument("--view-sample-count", type=int, default=4)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="wandb")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def phase_enabled(args: argparse.Namespace, phase: str) -> bool:
    start = PHASE_ORDER.index(args.start_phase)
    end = len(PHASE_ORDER) - 1 if args.max_phase == "all" else PHASE_ORDER.index(args.max_phase)
    idx = PHASE_ORDER.index(phase)
    return start <= idx <= end


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(
    cmd: list[str],
    *,
    log_path: Path,
    cwd: Path = REPO_ROOT,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_line = shlex.join(cmd)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {command_line}\n")
        handle.flush()
        print(f"[cmd] {command_line}", flush=True)
        if dry_run:
            handle.write("[dry-run] skipped\n")
            return 0
        process_env = os.environ.copy()
        process_env["PYTHONUNBUFFERED"] = "1"
        if env:
            process_env.update(env)
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            handle.write(line)
        return int(proc.wait())


def command_output(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return int(proc.returncode), proc.stdout


def run_search(pattern: str, paths: list[str]) -> tuple[int, str, str]:
    if shutil.which("rg"):
        cmd = ["rg", pattern, *paths]
        tool = "rg"
    else:
        cmd = ["grep", "-R", "-nE", pattern, *paths]
        tool = "grep_fallback"
    code, output = command_output(cmd)
    if code == 1:
        code = 0
    return code, output, tool


def phase0(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root
    logs = output_root / "logs"
    compile_files = [
        "sf3d/material_refine/dataset.py",
        "sf3d/material_refine/model.py",
        "scripts/train_material_refiner.py",
        "scripts/eval_material_refiner.py",
        "scripts/export_material_validation_comparison_panels.py",
        "scripts/export_material_attribute_comparison.py",
        "scripts/run_material_refine_ablation_suite.py",
        "scripts/build_material_refine_r_v2_acceptance_subset.py",
        "scripts/run_material_refine_r_v2_dayrun.py",
    ]
    py_compile_code = run_cmd(
        [PYTHON, "-m", "py_compile", *compile_files],
        log_path=logs / "phase0_py_compile.log",
        dry_run=args.dry_run,
    )
    forbidden_pattern = "|".join(["SF3D " "baseline", "SF3D/" "Prior", "SF3D" "="])
    fixed_code, fixed_output, fixed_tool = run_search(
        forbidden_pattern,
        ["sf3d", "scripts", "docs"],
    )
    leak_code, leak_output, leak_tool = run_search(
        r"uv_target_roughness|uv_target_metallic",
        ["sf3d/material_refine/model.py"],
    )
    fields_code, fields_output, fields_tool = run_search(
        r"input_prior_total_mae|baseline_total_mae|prior_source_type|prior_reliability|change_gate",
        ["scripts", "sf3d/material_refine", "docs"],
    )
    passed = (
        py_compile_code == 0
        and fixed_code == 0
        and not fixed_output.strip()
        and leak_code == 0
        and not leak_output.strip()
        and fields_code == 0
        and bool(fields_output.strip())
    )
    report = {
        "phase": "phase0",
        "passed": passed,
        "py_compile_code": py_compile_code,
        "fixed_string_tool": fixed_tool,
        "fixed_string_matches": fixed_output.strip().splitlines(),
        "target_leak_tool": leak_tool,
        "target_leak_matches": leak_output.strip().splitlines(),
        "field_check_tool": fields_tool,
        "field_check_match_count": len(fields_output.strip().splitlines()) if fields_output.strip() else 0,
    }
    md = [
        "# Phase0 Semantic Check",
        "",
        f"- passed: `{passed}`",
        f"- py_compile_code: `{py_compile_code}`",
        f"- fixed_string_tool: `{fixed_tool}`",
        f"- target_leak_tool: `{leak_tool}`",
        f"- field_check_tool: `{fields_tool}`",
        "",
        "## Fixed SF3D Baseline Strings",
        "",
        "```text",
        fixed_output.strip() or "none",
        "```",
        "",
        "## Target Leakage Search In Model",
        "",
        "```text",
        leak_output.strip() or "none",
        "```",
        "",
        "## Required Field Search Count",
        "",
        f"`{report['field_check_match_count']}` matches",
        "",
    ]
    (output_root / "phase0_semantic_check.md").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "phase0_semantic_check.md").write_text("\n".join(md), encoding="utf-8")
    write_json(output_root / "phase0_semantic_check.json", report)
    return report


def phase1(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.output_root / "subsets" / "r_v2_acceptance_128_manifest.json"
    cmd = [
        PYTHON,
        "scripts/build_material_refine_r_v2_acceptance_subset.py",
        "--output-root",
        str(args.output_root / "subsets"),
        "--output-manifest",
        str(manifest_path),
    ]
    code = run_cmd(cmd, log_path=args.output_root / "logs" / "phase1_subset.log", dry_run=args.dry_run)
    payload = load_json(manifest_path) if manifest_path.exists() else {}
    summary = payload.get("summary") or {}
    passed = code == 0 and bool(summary.get("diagnostic_ready")) and not summary.get("acceptance_blockers")
    report = {
        "phase": "phase1",
        "passed": passed,
        "code": code,
        "manifest": str(manifest_path.resolve()),
        "audit_md": str((args.output_root / "subsets" / "r_v2_acceptance_128_audit.md").resolve()),
        "summary": summary,
    }
    write_json(args.output_root / "phase1_acceptance_subset.json", report)
    return report


def train_command(
    *,
    manifest: Path,
    output_dir: Path,
    run_name: str,
    epochs: int,
    args: argparse.Namespace,
    extra_flags: list[str] | None = None,
    val_milestones: int = 8,
    max_validation_batches: int = 4,
) -> list[str]:
    flags = extra_flags or []
    return [
        PYTHON,
        "scripts/train_material_refiner.py",
        "--manifest",
        str(manifest),
        "--train-manifest",
        str(manifest),
        "--val-manifest",
        str(manifest),
        "--split-strategy",
        "manifest",
        "--train-split",
        "train",
        "--val-split",
        "val",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--val-batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--grad-accumulation-steps",
        "1",
        "--atlas-size",
        str(args.atlas_size),
        "--buffer-resolution",
        str(args.buffer_resolution),
        "--train-view-sample-count",
        str(args.view_sample_count),
        "--val-view-sample-count",
        str(args.view_sample_count),
        "--cuda-device-index",
        str(args.gpu_index),
        "--view-consistency-weight",
        "0.0",
        "--view-consistency-mode",
        "disabled",
        "--boundary-bleed-weight",
        "0.05",
        "--gradient-preservation-weight",
        "0.02",
        "--residual-safety-weight",
        "0.02",
        "--enable-prior-source-embedding",
        "true",
        "--enable-no-prior-bootstrap",
        "true",
        "--enable-boundary-safety",
        "true",
        "--enable-change-gate",
        "true",
        "--enable-material-aux-head",
        "false",
        "--enable-render-proxy-loss",
        "false",
        "--validation-progress-milestones",
        str(val_milestones),
        "--render-proxy-validation-milestone-interval",
        "4",
        "--render-proxy-validation-max-batches",
        "4",
        "--eval-every",
        "0",
        "--max-validation-batches",
        str(max_validation_batches),
        "--val-preview-samples",
        "4",
        "--wandb-val-preview-max",
        "4",
        "--wandb-log-preview-grid",
        "false",
        "--save-preview-contact-sheet",
        "false",
        "--save-only-best-checkpoint",
        "true",
        "--keep-last-checkpoints",
        "1",
        "--log-every",
        "10",
        "--progress-bar",
        "true",
        "--train-line-logs",
        "false",
        "--fail-on-target-prior-identity",
        "false",
        "--min-nontrivial-target-count-for-paper",
        "0",
        "--preflight-audit-records",
        "128",
        "--output-dir",
        str(output_dir),
        "--report-to",
        args.report_to,
        "--tracker-project-name",
        PROJECT_NAME,
        "--tracker-run-name",
        run_name,
        "--tracker-group",
        GROUP_NAME,
        "--tracker-tags",
        TAGS,
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-log-artifacts",
        "true",
        "--wandb-artifact-policy",
        "best_and_final",
        *flags,
    ]


def find_checkpoint(train_dir: Path) -> Path | None:
    for name in ("best.pt", "latest.pt"):
        path = train_dir / name
        if path.exists():
            return path
    checkpoints = sorted(train_dir.glob("epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def eval_command(
    *,
    manifest: Path,
    checkpoint: Path,
    output_dir: Path,
    run_name: str,
    args: argparse.Namespace,
    split: str = "all",
    paper_splits: str | None = None,
    max_samples: int | None = None,
    enable_flags: dict[str, bool] | None = None,
) -> list[str]:
    enable_flags = enable_flags or {}
    cmd = [
        PYTHON,
        "scripts/eval_material_refiner.py",
        "--manifest",
        str(manifest),
        "--checkpoint",
        str(checkpoint),
        "--split",
        split,
        "--split-strategy",
        "manifest",
        "--batch-size",
        "2",
        "--num-workers",
        str(max(1, min(args.num_workers, 4))),
        "--view-sample-count",
        str(args.view_sample_count),
        "--cuda-device-index",
        str(args.gpu_index),
        "--max-artifact-objects",
        "16",
        "--output-dir",
        str(output_dir),
        "--report-to",
        args.report_to,
        "--tracker-project-name",
        PROJECT_NAME,
        "--tracker-run-name",
        run_name,
        "--tracker-group",
        GROUP_NAME,
        "--tracker-tags",
        TAGS + ",eval",
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-log-top-cases",
        "false",
        "--wandb-log-group-breakdowns",
        "false",
        "--wandb-log-paper-main-table",
        "false",
        "--wandb-log-artifacts",
        "true",
        "--wandb-artifact-policy",
        "summary",
        "--print-summary-json",
        "false",
    ]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    if paper_splits:
        cmd += ["--paper-splits", paper_splits]
    for key, value in enable_flags.items():
        cmd += [f"--{key.replace('_', '-')}", "true" if value else "false"]
    return cmd


def export_panels(
    *,
    manifest: Path,
    metrics: Path,
    output_dir: Path,
    run_name: str,
    args: argparse.Namespace,
    log_path: Path,
) -> int:
    code = run_cmd(
        [
            PYTHON,
            "scripts/export_material_validation_comparison_panels.py",
            "--manifest",
            str(manifest),
            "--metrics",
            str(metrics),
            "--output-dir",
            str(output_dir / "validation_comparison_panels"),
            "--max-panels",
            "16",
            "--panel-size",
            "160",
            "--report-to",
            args.report_to,
            "--tracker-project-name",
            PROJECT_NAME,
            "--tracker-run-name",
            run_name + "-panels",
            "--tracker-group",
            GROUP_NAME,
            "--tracker-tags",
            TAGS + ",panels",
            "--wandb-mode",
            args.wandb_mode,
            "--wandb-max-panel-images",
            "4",
            "--wandb-log-panel-table",
            "false",
            "--wandb-log-full-panel-artifact",
            "false",
        ],
        log_path=log_path,
        dry_run=args.dry_run,
    )
    if code != 0:
        return code
    return run_cmd(
        [
            PYTHON,
            "scripts/export_material_attribute_comparison.py",
            "--metrics-json",
            str(metrics),
            "--output-dir",
            str(output_dir / "material_attribute_comparison"),
        ],
        log_path=log_path,
        dry_run=args.dry_run,
    )


def run_metric_consistency_audit(args: argparse.Namespace, eval_dir: Path, log_path: Path) -> dict[str, Any]:
    metrics_path = eval_dir / "metrics.json"
    summary_path = eval_dir / "summary.json"
    audit_path = eval_dir / "metric_consistency_audit.json"
    if not metrics_path.exists() or not summary_path.exists():
        return {
            "code": None,
            "metric_consistency_pass": False,
            "phase4_rehearsal_gate_pass": False,
            "reason": "missing_metrics_or_summary",
        }
    code = run_cmd(
        [
            PYTHON,
            "scripts/audit_material_refine_metric_consistency.py",
            "--eval-dir",
            str(eval_dir),
        ],
        log_path=log_path,
        dry_run=args.dry_run,
    )
    payload = load_json(audit_path) if audit_path.exists() else {}
    payload["code"] = code
    if "metric_consistency_pass" not in payload:
        payload["metric_consistency_pass"] = False
    if "phase4_rehearsal_gate_pass" not in payload:
        payload["phase4_rehearsal_gate_pass"] = False
    return payload


def write_case_indexes(eval_dir: Path) -> None:
    path = eval_dir / "diagnostic_cases.json"
    if not path.exists():
        return
    payload = load_json(path)
    mapping = {
        "top_improved": payload.get("top_improved", []),
        "top_regressed": payload.get("top_regressed", []),
        "top_uncertain": payload.get("top_uncertain", []),
        "top_boundary_failures": (payload.get("top_failure_cases") or {}).get("boundary_bleed", []),
    }
    for name, rows in mapping.items():
        out_dir = eval_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# {name}", ""]
        for row in rows[:24]:
            lines.append(
                "- "
                f"`{row.get('object_id')}` view=`{row.get('view_name')}` "
                f"prior={row.get('baseline_total_mae')} refined={row.get('refined_total_mae')} "
                f"improvement={row.get('improvement_total')} "
                f"failure={row.get('refined_primary_failure')}"
            )
        (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_train_eval_panel(
    *,
    manifest: Path,
    train_dir: Path,
    eval_dir: Path,
    run_name: str,
    epochs: int,
    args: argparse.Namespace,
    extra_train_flags: list[str] | None = None,
    enable_eval_flags: dict[str, bool] | None = None,
    val_milestones: int = 8,
    max_samples: int | None = None,
) -> dict[str, Any]:
    log_path = args.output_root / "logs" / f"{run_name}.log"
    train_code = run_cmd(
        train_command(
            manifest=manifest,
            output_dir=train_dir,
            run_name=run_name,
            epochs=epochs,
            args=args,
            extra_flags=extra_train_flags,
            val_milestones=val_milestones,
        ),
        log_path=log_path,
        dry_run=args.dry_run,
    )
    checkpoint = find_checkpoint(train_dir)
    if train_code != 0 or checkpoint is None:
        return {"passed": False, "train_code": train_code, "checkpoint": None}
    eval_code = run_cmd(
        eval_command(
            manifest=manifest,
            checkpoint=checkpoint,
            output_dir=eval_dir,
            run_name=run_name + "-eval",
            args=args,
            split="all",
            max_samples=max_samples,
            enable_flags=enable_eval_flags,
        ),
        log_path=log_path,
        dry_run=args.dry_run,
    )
    summary_path = eval_dir / "summary.json"
    metrics_path = eval_dir / "metrics.json"
    metric_audit: dict[str, Any] = {}
    audit_code = 0
    if eval_code == 0:
        metric_audit = run_metric_consistency_audit(args, eval_dir, log_path)
        audit_code = int(metric_audit.get("code") or 0)
    panel_code = 0
    if eval_code == 0 and metrics_path.exists():
        panel_code = export_panels(
            manifest=manifest,
            metrics=metrics_path,
            output_dir=eval_dir,
            run_name=run_name,
            args=args,
            log_path=log_path,
        )
        write_case_indexes(eval_dir)
    summary = load_json(summary_path) if summary_path.exists() else {}
    return {
        "passed": train_code == 0 and eval_code == 0 and panel_code == 0 and bool(summary),
        "train_code": train_code,
        "eval_code": eval_code,
        "audit_code": audit_code,
        "panel_code": panel_code,
        "checkpoint": str(checkpoint.resolve()),
        "train_dir": str(train_dir.resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "summary": summary,
        "metric_consistency_audit": metric_audit,
    }


def phase2(args: argparse.Namespace, manifest: Path) -> dict[str, Any]:
    result = run_train_eval_panel(
        manifest=manifest,
        train_dir=args.output_root / "acceptance_128_train",
        eval_dir=args.output_root / "acceptance_128_eval",
        run_name="r-v2-acceptance-128",
        epochs=args.acceptance_epochs,
        args=args,
        val_milestones=8,
        max_samples=128,
        enable_eval_flags={
            "enable_prior_source_embedding": True,
            "enable_no_prior_bootstrap": True,
            "enable_boundary_safety": True,
            "enable_change_gate": True,
            "enable_material_aux_head": False,
            "enable_render_proxy_loss": False,
        },
    )
    summary = result.get("summary") or {}
    by_prior_label = summary.get("by_prior_label") or {}
    with_prior_reliability = (by_prior_label.get("with_prior") or {}).get("prior_reliability_mean")
    without_prior_reliability = (by_prior_label.get("without_prior") or {}).get("prior_reliability_mean")
    checks = {
        "has_required_summary_fields": all(
            key in summary
            for key in [
                "input_prior_total_mae",
                "refined_total_mae",
                "gain_total",
                "improvement_rate",
                "regression_rate",
                "prior_reliability_mean",
                "change_gate_mean",
                "mean_abs_delta",
                "boundary_delta_mean",
                "by_prior_mode",
                "by_prior_source_type",
                "by_material_family",
                "by_source_name",
            ]
        ),
        "prior_reliability_group_order_observable": with_prior_reliability is not None
        and without_prior_reliability is not None,
        "prior_reliability_with_gt_without": (
            with_prior_reliability > without_prior_reliability
            if with_prior_reliability is not None and without_prior_reliability is not None
            else None
        ),
        "change_gate_not_collapsed": (
            summary.get("change_gate_mean") is not None
            and 0.01 < float(summary["change_gate_mean"]) < 0.99
        ),
        "bootstrap_observable": (summary.get("bootstrap_enabled_rate") or 0.0) > 0.0,
        "regression_not_above_improvement": (
            (summary.get("regression_rate") or 0.0) <= (summary.get("improvement_rate") or 0.0)
        ),
    }
    result["checks"] = checks
    result["passed"] = bool(result.get("passed")) and checks["has_required_summary_fields"] and checks["bootstrap_observable"]
    metric_audit = result.get("metric_consistency_audit") or {}
    write_json(args.output_root / "r_v2_acceptance_128_summary.json", result)
    note = [
        "# R-v2 Acceptance 128 Note",
        "",
        f"- passed: `{result['passed']}`",
        f"- train_dir: `{rel(Path(result.get('train_dir', args.output_root)))}`",
        f"- eval_dir: `{rel(Path(result.get('eval_dir', args.output_root)))}`",
        f"- input_prior_total_mae: `{summary.get('input_prior_total_mae')}`",
        f"- refined_total_mae: `{summary.get('refined_total_mae')}`",
        f"- gain_total: `{summary.get('gain_total')}`",
        f"- improvement_rate: `{summary.get('improvement_rate')}`",
        f"- regression_rate: `{summary.get('regression_rate')}`",
        f"- prior_reliability_mean: `{summary.get('prior_reliability_mean')}`",
        f"- change_gate_mean: `{summary.get('change_gate_mean')}`",
        "",
        "## Checks",
        "",
        *[f"- `{key}`: `{value}`" for key, value in checks.items()],
        "",
        "## Metric Consistency Audit",
        "",
        f"- metric_consistency_pass: `{metric_audit.get('metric_consistency_pass')}`",
        f"- phase4_rehearsal_gate_pass: `{metric_audit.get('phase4_rehearsal_gate_pass')}`",
        f"- view_row_regression_dominates: `{metric_audit.get('view_row_regression_dominates')}`",
        f"- computed_improvement_rate_debug: `{metric_audit.get('computed_improvement_rate_debug')}`",
        f"- computed_regression_rate_debug: `{metric_audit.get('computed_regression_rate_debug')}`",
        f"- summary_improvement_rate: `{metric_audit.get('summary_improvement_rate')}`",
        f"- summary_regression_rate: `{metric_audit.get('summary_regression_rate')}`",
        f"- audit: `{rel(args.output_root / 'acceptance_128_eval' / 'metric_consistency_audit.md')}`",
        f"- debug_csv: `{rel(args.output_root / 'acceptance_128_eval' / 'metric_consistency_debug.csv')}`",
        "",
    ]
    (args.output_root / "r_v2_acceptance_128_note.md").write_text("\n".join(note), encoding="utf-8")
    return result


def summarize_eval(summary: dict[str, Any]) -> dict[str, Any]:
    by_prior = summary.get("by_prior_label") or {}
    view_level = summary.get("view_level") or {}
    return {
        "input_prior_total_mae": summary.get("input_prior_total_mae"),
        "refined_total_mae": summary.get("refined_total_mae"),
        "gain_total": summary.get("gain_total"),
        "improvement_rate": summary.get("improvement_rate"),
        "regression_rate": summary.get("regression_rate"),
        "view_gain_total": view_level.get("gain_total"),
        "view_improvement_rate": view_level.get("improvement_rate"),
        "view_regression_rate": view_level.get("regression_rate"),
        "prior_reliability_mean": summary.get("prior_reliability_mean"),
        "change_gate_mean": summary.get("change_gate_mean"),
        "mean_abs_delta": summary.get("mean_abs_delta"),
        "boundary_delta_mean": summary.get("boundary_delta_mean"),
        "without_prior_gain": (by_prior.get("without_prior") or {}).get("gain_total"),
        "with_prior_gain": (by_prior.get("with_prior") or {}).get("gain_total"),
    }


def phase3(args: argparse.Namespace, manifest: Path, phase2_result: dict[str, Any]) -> dict[str, Any]:
    base_summary = phase2_result.get("summary") or {}
    runs: dict[str, dict[str, Any]] = {
        "A0_input_prior_only": {
        "enabled_modules": "none; evaluates Input Prior vs GT",
            "summary": {
                "input_prior_total_mae": base_summary.get("input_prior_total_mae"),
                "refined_total_mae": base_summary.get("input_prior_total_mae"),
                "gain_total": 0.0,
                "improvement_rate": 0.0,
                "regression_rate": 0.0,
                "view_level": {
                    "gain_total": 0.0,
                    "improvement_rate": 0.0,
                    "regression_rate": 0.0,
                },
            },
            "passed": True,
            "notes": "No training run. Baseline row derived from A1 eval input-prior metrics.",
        },
        "A1_ours_full": {
            "enabled_modules": "prior_source_embedding,no_prior_bootstrap,boundary_safety,change_gate",
            "summary": base_summary,
            "passed": bool(phase2_result.get("passed")),
            "notes": "Reuses Phase2 acceptance run.",
        },
    }
    ablations = {
        "A2_wo_no_prior_bootstrap": (["--disable-no-prior-bootstrap", "true"], {"enable_no_prior_bootstrap": False}),
        "A3_wo_change_gate": (["--disable-change-gate", "true"], {"enable_change_gate": False}),
        "A4_wo_boundary_safety": (["--disable-boundary-safety", "true"], {"enable_boundary_safety": False}),
        "A5_wo_prior_source_embedding": (["--disable-prior-source-embedding", "true"], {"enable_prior_source_embedding": False}),
    }
    for run_name, (train_flags, eval_flag_overrides) in ablations.items():
        enable_flags = {
            "enable_prior_source_embedding": True,
            "enable_no_prior_bootstrap": True,
            "enable_boundary_safety": True,
            "enable_change_gate": True,
            "enable_material_aux_head": False,
            "enable_render_proxy_loss": False,
        }
        enable_flags.update(eval_flag_overrides)
        result = run_train_eval_panel(
            manifest=manifest,
            train_dir=args.output_root / "ablation_quick" / run_name / "train",
            eval_dir=args.output_root / "ablation_quick" / run_name / "eval",
            run_name="r-v2-" + run_name.lower(),
            epochs=args.ablation_epochs,
            args=args,
            extra_train_flags=train_flags,
            enable_eval_flags=enable_flags,
            val_milestones=4,
            max_samples=128,
        )
        runs[run_name] = {
            "enabled_modules": run_name.replace("_", " "),
            "summary": result.get("summary") or {},
            "passed": bool(result.get("passed")),
            "notes": "quick diagnostic ablation",
            "paths": {
                "train_dir": result.get("train_dir"),
                "eval_dir": result.get("eval_dir"),
            },
        }
    table_lines = [
        "# R-v2 Quick Ablation Table",
        "",
        "Top-level rates are UV-object rates. View-row direction is shown separately because Phase4 is gated on it.",
        "",
        "| run_name | enabled_modules | uv_input_prior_mae | uv_refined_mae | uv_gain | uv_improve_rate | uv_regress_rate | view_gain | view_improve_rate | view_regress_rate | boundary_delta_mean | without_prior_gain | with_prior_gain | notes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run_name, result in runs.items():
        metrics = summarize_eval(result.get("summary") or {})
        table_lines.append(
            "| "
            + " | ".join(
                [
                    run_name,
                    str(result.get("enabled_modules")),
                    str(metrics["input_prior_total_mae"]),
                    str(metrics["refined_total_mae"]),
                    str(metrics["gain_total"]),
                    str(metrics["improvement_rate"]),
                    str(metrics["regression_rate"]),
                    str(metrics["view_gain_total"]),
                    str(metrics["view_improvement_rate"]),
                    str(metrics["view_regression_rate"]),
                    str(metrics["boundary_delta_mean"]),
                    str(metrics["without_prior_gain"]),
                    str(metrics["with_prior_gain"]),
                    str(result.get("notes", "")),
                ]
            )
            + " |"
        )
    out_dir = args.output_root / "ablation_quick"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r_v2_quick_ablation_table.md").write_text("\n".join(table_lines), encoding="utf-8")
    phase_report = {"phase": "phase3", "runs": runs, "table": str((out_dir / "r_v2_quick_ablation_table.md").resolve())}
    full_gain = base_summary.get("gain_total")
    base_view = base_summary.get("view_level") or {}
    view_regression_dominates = (
        base_view.get("regression_rate") is not None
        and base_view.get("improvement_rate") is not None
        and float(base_view["regression_rate"]) > float(base_view["improvement_rate"])
    )
    phase_report["view_regression_dominates"] = view_regression_dominates
    phase_report["passed"] = (
        bool(phase2_result.get("passed"))
        and full_gain is not None
        and float(full_gain) >= -1.0e-4
        and not view_regression_dominates
    )
    if not phase_report["passed"]:
        (out_dir / "r_v2_design_risk_note.md").write_text(
            "# R-v2 Design Risk Note\n\n"
            f"- phase2_passed: `{phase2_result.get('passed')}`\n"
            f"- ours_full_gain_total: `{full_gain}`\n"
            "- Quick ablation did not justify paper-stage rehearsal. Do not launch full training.\n",
            encoding="utf-8",
        )
    write_json(out_dir / "r_v2_quick_ablation_summary.json", phase_report)
    return phase_report


def rehearsal_manifest() -> Path | None:
    candidates = [
        Path(
            "output/material_refine_pipeline_20260418T091559Z/"
            "paper_stage1_pipeline_auto_select/readiness/stage1_subset/paper_stage1_subset_manifest.json"
        ),
        Path("output/material_refine_paper/dataset_sync_check_20260422/longrun_monitor_stage1_subset_654/paper_stage1_subset_manifest.json"),
        Path("output/material_refine_paper/latest_dataset_check_20260421/stage1_subset_merged490/paper_stage1_subset_manifest.json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def phase4_metric_gate(args: argparse.Namespace, phase2_result: dict[str, Any]) -> dict[str, Any]:
    audit = phase2_result.get("metric_consistency_audit") or {}
    audit_path = args.output_root / "acceptance_128_eval" / "metric_consistency_audit.json"
    if not audit and audit_path.exists():
        audit = load_json(audit_path)
    if not audit:
        return {
            "passed": False,
            "reason": "metric_consistency_audit_missing",
            "audit_path": str(audit_path.resolve()),
        }
    if not audit.get("metric_consistency_pass"):
        return {
            "passed": False,
            "reason": "metric_consistency_audit_failed",
            "audit_path": str(audit_path.resolve()),
            "audit": audit,
        }
    if audit.get("view_row_regression_dominates"):
        return {
            "passed": False,
            "reason": "view_level_regression_dominates",
            "audit_path": str(audit_path.resolve()),
            "audit": audit,
        }
    if not audit.get("phase4_rehearsal_gate_pass", False):
        return {
            "passed": False,
            "reason": "phase4_rehearsal_gate_failed",
            "audit_path": str(audit_path.resolve()),
            "audit": audit,
        }
    return {
        "passed": True,
        "reason": "metric_consistency_and_view_direction_passed",
        "audit_path": str(audit_path.resolve()),
        "audit": audit,
    }


def phase4(args: argparse.Namespace, phase2_result: dict[str, Any], phase3_result: dict[str, Any]) -> dict[str, Any]:
    if not phase2_result.get("passed") or not phase3_result.get("passed"):
        result = {
            "phase": "phase4",
            "passed": False,
            "skipped": True,
            "reason": "phase2_or_phase3_gate_failed",
        }
        write_json(args.output_root / "paper_stage_rehearsal_210" / "phase4_skipped.json", result)
        return result
    metric_gate = phase4_metric_gate(args, phase2_result)
    if not metric_gate.get("passed"):
        result = {
            "phase": "phase4",
            "passed": False,
            "skipped": True,
            "reason": metric_gate.get("reason"),
            "metric_gate": metric_gate,
        }
        write_json(args.output_root / "paper_stage_rehearsal_210" / "phase4_skipped.json", result)
        return result
    manifest = rehearsal_manifest()
    if manifest is None:
        result = {
            "phase": "phase4",
            "passed": False,
            "skipped": True,
            "reason": "no_rehearsal_manifest_found",
        }
        write_json(args.output_root / "paper_stage_rehearsal_210" / "phase4_skipped.json", result)
        return result
    result = run_train_eval_panel(
        manifest=manifest,
        train_dir=args.output_root / "paper_stage_rehearsal_210" / "train",
        eval_dir=args.output_root / "paper_stage_rehearsal_210" / "eval_all",
        run_name="r-v2-paper-stage-rehearsal-210",
        epochs=args.rehearsal_epochs,
        args=args,
        val_milestones=16,
        max_samples=None,
        enable_eval_flags={
            "enable_prior_source_embedding": True,
            "enable_no_prior_bootstrap": True,
            "enable_boundary_safety": True,
            "enable_change_gate": True,
            "enable_material_aux_head": False,
            "enable_render_proxy_loss": False,
        },
    )
    result["phase"] = "phase4"
    result["manifest"] = str(manifest.resolve())
    summary = result.get("summary") or {}
    result["passed"] = bool(result.get("passed")) and (
        summary.get("gain_total") is not None
        and float(summary.get("gain_total")) > 0.0
        and (summary.get("improvement_rate") or 0.0) > (summary.get("regression_rate") or 0.0)
    )
    note = [
        "# R-v2 Paper-stage Rehearsal Note",
        "",
        f"- passed: `{result['passed']}`",
        f"- manifest: `{rel(manifest)}`",
        f"- train_dir: `{rel(Path(result.get('train_dir', args.output_root)))}`",
        f"- eval_dir: `{rel(Path(result.get('eval_dir', args.output_root)))}`",
        f"- input_prior_total_mae: `{summary.get('input_prior_total_mae')}`",
        f"- refined_total_mae: `{summary.get('refined_total_mae')}`",
        f"- gain_total: `{summary.get('gain_total')}`",
        f"- improvement_rate: `{summary.get('improvement_rate')}`",
        f"- regression_rate: `{summary.get('regression_rate')}`",
        "",
    ]
    out_dir = args.output_root / "paper_stage_rehearsal_210"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r_v2_paper_stage_rehearsal_note.md").write_text("\n".join(note), encoding="utf-8")
    write_json(out_dir / "phase4_rehearsal_summary.json", result)
    return result


def phase5(
    args: argparse.Namespace,
    *,
    phase0_result: dict[str, Any] | None,
    phase1_result: dict[str, Any] | None,
    phase2_result: dict[str, Any] | None,
    phase3_result: dict[str, Any] | None,
    phase4_result: dict[str, Any] | None,
) -> dict[str, Any]:
    acceptance_summary = (phase2_result or {}).get("summary") or {}
    subset_summary = ((phase1_result or {}).get("summary") or {})
    rehearsal_summary = (phase4_result or {}).get("summary") or {}
    approve_round9 = bool((phase4_result or {}).get("passed"))
    approve_stage1 = bool((phase1_result or {}).get("passed")) and bool(subset_summary.get("paper_acceptance_ready"))
    approve_input_prior_claim = approve_round9 and float(rehearsal_summary.get("gain_total") or 0.0) > 0.0
    decision = {
        "phase": "phase5",
        "approve_round9_full_training": approve_round9,
        "approve_stage1_v2_expansion": approve_stage1,
        "approve_paper_claim_input_prior_vs_ours": approve_input_prior_claim,
        "approve_claim_sf3d_vs_ours": False,
        "phase_status": {
            "phase0": bool((phase0_result or {}).get("passed")),
            "phase1": bool((phase1_result or {}).get("passed")),
            "phase2": bool((phase2_result or {}).get("passed")),
            "phase3": bool((phase3_result or {}).get("passed")),
            "phase4": bool((phase4_result or {}).get("passed")),
        },
    }
    lines = [
        "# R-v2 Dayrun Decision",
        "",
        "## Summary",
        "",
        f"- phase0_semantics_passed: `{decision['phase_status']['phase0']}`",
        f"- acceptance_subset_passed: `{decision['phase_status']['phase1']}`",
        f"- acceptance_128_train_eval_passed: `{decision['phase_status']['phase2']}`",
        f"- quick_ablation_passed: `{decision['phase_status']['phase3']}`",
        f"- paper_rehearsal_passed: `{decision['phase_status']['phase4']}`",
        "",
        "## Core Metrics",
        "",
        "| scope | input_prior_total_mae | refined_total_mae | gain_total | improvement_rate | regression_rate |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| acceptance_128 | {acceptance_summary.get('input_prior_total_mae')} | "
            f"{acceptance_summary.get('refined_total_mae')} | {acceptance_summary.get('gain_total')} | "
            f"{acceptance_summary.get('improvement_rate')} | {acceptance_summary.get('regression_rate')} |"
        ),
        (
            f"| rehearsal_210 | {rehearsal_summary.get('input_prior_total_mae')} | "
            f"{rehearsal_summary.get('refined_total_mae')} | {rehearsal_summary.get('gain_total')} | "
            f"{rehearsal_summary.get('improvement_rate')} | {rehearsal_summary.get('regression_rate')} |"
        ),
        "",
        "## Key Links",
        "",
        f"- acceptance panels: `{rel(args.output_root / 'acceptance_128_eval' / 'validation_comparison_panels' / 'validation_comparison_index.html')}`",
        f"- acceptance attribute comparison: `{rel(args.output_root / 'acceptance_128_eval' / 'material_attribute_comparison' / 'material_attribute_comparison.html')}`",
        f"- ablation table: `{rel(args.output_root / 'ablation_quick' / 'r_v2_quick_ablation_table.md')}`",
        f"- rehearsal panels: `{rel(args.output_root / 'paper_stage_rehearsal_210' / 'eval_all' / 'validation_comparison_panels' / 'validation_comparison_index.html')}`",
        "",
        "## Required Answers",
        "",
        f"- R-v2 是否比 Input Prior 更接近 GT: `{acceptance_summary.get('gain_total')}` on acceptance, `{rehearsal_summary.get('gain_total')}` on rehearsal.",
        "- 提升主要来自 with_prior 还是 without_prior: see `by_prior_label` / `by_prior_source_type` in summary.json.",
        f"- no-prior bootstrap 是否有效: `bootstrap_enabled_rate={acceptance_summary.get('bootstrap_enabled_rate')}`.",
        f"- change_gate 是否降低 regression: `change_gate_mean={acceptance_summary.get('change_gate_mean')}`, ablation table required for direction.",
        "- boundary_safety 是否降低边界问题: see `boundary_delta_mean` and boundary_bleed metrics in eval summaries.",
        "- prior_source_embedding 是否有必要: see A5 ablation row.",
        "- 是否存在 source/material_family 拖后腿: see grouped summaries in eval summary.json.",
        f"- target/prior 过近风险: `paper_identity_ready={subset_summary.get('paper_identity_ready')}`, identity_mean=`{subset_summary.get('target_prior_identity_mean')}`.",
        "- panel 是否可解释: local HTML panels include Input Prior, GT, Pred, delta, reliability, gate and bootstrap maps.",
        "",
        "## Final Booleans",
        "",
        *[f"- {key}: `{value}`" for key, value in decision.items() if key.startswith("approve_")],
        "",
    ]
    path = args.output_root / "r_v2_dayrun_decision.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(args.output_root / "r_v2_dayrun_decision.json", decision)
    return decision


def main() -> None:
    args = parse_args()
    args.output_root = (REPO_ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_root / "dayrun_args.json",
        {
            **jsonable_args(args),
            "output_root": str(args.output_root),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )

    results: dict[str, dict[str, Any] | None] = {phase: None for phase in PHASE_ORDER}
    manifest = args.output_root / "subsets" / "r_v2_acceptance_128_manifest.json"

    if phase_enabled(args, "phase0"):
        results["phase0"] = phase0(args)
        if not results["phase0"].get("passed"):
            print("[gate] phase0 failed; stopping before training.", flush=True)
            phase5(args, phase0_result=results["phase0"], phase1_result=None, phase2_result=None, phase3_result=None, phase4_result=None)
            raise SystemExit(1)
    if phase_enabled(args, "phase1"):
        results["phase1"] = phase1(args)
        if not results["phase1"].get("passed"):
            print("[gate] phase1 hard blockers; stopping before training.", flush=True)
            phase5(args, phase0_result=results["phase0"], phase1_result=results["phase1"], phase2_result=None, phase3_result=None, phase4_result=None)
            raise SystemExit(1)
    elif manifest.exists():
        results["phase1"] = {"passed": True, "manifest": str(manifest.resolve()), "summary": (load_json(manifest).get("summary") or {})}

    if phase_enabled(args, "phase2"):
        results["phase2"] = phase2(args, manifest)
        if not results["phase2"].get("passed"):
            print("[gate] phase2 engineering failure; stopping before ablation.", flush=True)
            phase5(args, phase0_result=results["phase0"], phase1_result=results["phase1"], phase2_result=results["phase2"], phase3_result=None, phase4_result=None)
            raise SystemExit(1)
    if phase_enabled(args, "phase3"):
        results["phase3"] = phase3(args, manifest, results["phase2"] or {})
    if phase_enabled(args, "phase4"):
        results["phase4"] = phase4(args, results["phase2"] or {}, results["phase3"] or {})
    if phase_enabled(args, "phase5") or args.max_phase == "all":
        results["phase5"] = phase5(
            args,
            phase0_result=results["phase0"],
            phase1_result=results["phase1"],
            phase2_result=results["phase2"],
            phase3_result=results["phase3"],
            phase4_result=results["phase4"],
        )
    print(json.dumps({key: value.get("passed") if isinstance(value, dict) else None for key, value in results.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
