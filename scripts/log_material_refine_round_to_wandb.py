from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.experiment import (  # noqa: E402
    flatten_for_logging,
    log_path_artifact,
    make_json_serializable,
    maybe_init_wandb,
    sanitize_log_dict,
    wandb,
)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Log a complete material-refine round summary to W&B.",
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--round-analysis-dir", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--tracker-project-name", type=str, default="stable-fast-3d-material-refine")
    parser.add_argument("--tracker-run-name", type=str, default="material-refine-round-summary")
    parser.add_argument("--tracker-group", type=str, default="material-refine-round")
    parser.add_argument("--tracker-tags", type=str, default="material-refine,round-summary")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-log-artifacts", type=parse_bool, default=True)
    parser.add_argument("--max-panel-images", type=int, default=12)
    return parser.parse_args()


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect_scalar_logs(args: argparse.Namespace) -> dict[str, Any]:
    train_state = load_json_if_exists(args.train_dir / "train_state.json") or {}
    eval_summary = load_json_if_exists(args.eval_dir / "summary.json") or {}
    audit_payload = load_json_if_exists(args.audit_dir / "manifest_audit_summary.json") or {}
    round_analysis = load_json_if_exists(args.round_analysis_dir / "round_analysis.json") or {}

    logs: dict[str, Any] = {
        "round/best_epoch": train_state.get("best_epoch"),
        "round/best_val_metric": train_state.get("best_val_metric"),
        "round/last_epoch": train_state.get("last_epoch"),
        "round/optimizer_step": train_state.get("optimizer_step"),
        "eval/baseline_total_mae": eval_summary.get("baseline_total_mae"),
        "eval/refined_total_mae": eval_summary.get("refined_total_mae"),
        "eval/avg_improvement_total": eval_summary.get("avg_improvement_total"),
        "eval/rows": eval_summary.get("rows"),
    }
    if isinstance(eval_summary.get("by_prior_label"), dict):
        logs.update(flatten_for_logging(eval_summary["by_prior_label"], prefix="eval/by_prior_label"))
    if isinstance(eval_summary.get("by_generator_id"), dict):
        logs.update(flatten_for_logging(eval_summary["by_generator_id"], prefix="eval/by_generator_id"))
    audit_summary = audit_payload.get("summary") if isinstance(audit_payload, dict) else None
    if isinstance(audit_summary, dict):
        logs.update(flatten_for_logging(audit_summary, prefix="audit"))
    if isinstance(round_analysis.get("attribute"), dict):
        logs.update(flatten_for_logging(round_analysis["attribute"], prefix="attribute"))
    sanitized, skipped = sanitize_log_dict(logs)
    if skipped:
        print(json.dumps({"skipped_round_summary_logs": skipped}, ensure_ascii=False))
    return sanitized


def collect_artifact_paths(args: argparse.Namespace) -> list[Path]:
    paths = [
        args.train_dir / "train_args.json",
        args.train_dir / "history.json",
        args.train_dir / "train_state.json",
        args.train_dir / "best.pt",
        args.train_dir / "latest.pt",
        args.train_dir / "training_curves.png",
        args.train_dir / "training_summary.html",
        args.eval_dir / "summary.json",
        args.eval_dir / "metrics.json",
        args.eval_dir / "report.html",
        args.eval_dir / "material_attribute_summary.json",
        args.eval_dir / "material_attribute_comparison.png",
        args.eval_dir / "material_attribute_comparison.html",
        args.panel_dir / "validation_comparison_index.html",
        args.panel_dir / "validation_comparison_summary.json",
        args.audit_dir / "manifest_audit_summary.json",
        args.audit_dir / "manifest_audit.html",
        args.audit_dir / "manifest_audit.png",
        args.round_analysis_dir / "round_analysis.json",
        args.round_analysis_dir / "round_analysis.html",
    ]
    return [path for path in paths if path.exists()]


def log_media(run: Any, args: argparse.Namespace) -> None:
    if wandb is None:
        return
    media = {}
    training_curves = args.train_dir / "training_curves.png"
    attr_plot = args.eval_dir / "material_attribute_comparison.png"
    audit_plot = args.audit_dir / "manifest_audit.png"
    if training_curves.exists():
        media["round/training_curves"] = wandb.Image(str(training_curves))
    if attr_plot.exists():
        media["round/material_attribute_comparison"] = wandb.Image(str(attr_plot))
    if audit_plot.exists():
        media["round/manifest_audit"] = wandb.Image(str(audit_plot))
    if media:
        run.log(media)

    panels = sorted(args.panel_dir.glob("*.png"))[: args.max_panel_images]
    if panels:
        table = wandb.Table(columns=["panel_name", "panel"])
        for panel in panels:
            table.add_data(panel.stem, wandb.Image(str(panel)))
        run.log({"round/validation_comparison_panels": table})


def main() -> None:
    args = parse_args()
    run = maybe_init_wandb(
        enabled=True,
        project=args.tracker_project_name,
        job_type="round-summary",
        config=make_json_serializable(vars(args)),
        mode=args.wandb_mode,
        name=args.tracker_run_name,
        group=args.tracker_group,
        tags=args.tracker_tags,
        dir_path=args.wandb_dir,
    )
    logs = collect_scalar_logs(args)
    if logs:
        run.log(logs)
    log_media(run, args)
    if args.wandb_log_artifacts:
        log_path_artifact(
            run,
            name=f"{args.tracker_run_name}-artifacts",
            artifact_type="material-refine-round",
            paths=collect_artifact_paths(args),
        )
    run.finish()
    print(json.dumps({"wandb_round_summary_logged": args.tracker_run_name}, ensure_ascii=False))


if __name__ == "__main__":
    main()
