from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LEARNED_ABLATIONS = {
    "no_prior_refiner": REPO_ROOT / "configs" / "material_refine_ablation_no_prior_override.yaml",
    "no_view_refiner": REPO_ROOT / "configs" / "material_refine_ablation_no_view_override.yaml",
    "no_residual_refiner": REPO_ROOT / "configs" / "material_refine_ablation_no_residual_override.yaml",
}
EVAL_ONLY_ABLATIONS = {
    "scalar_broadcast": "scalar_broadcast",
    "prior_smoothing": "prior_smoothing",
    "ours_full": "ours_full",
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the paper ablation suite on the fixed stage1 manifest protocol.",
    )
    parser.add_argument("--stage1-manifest", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, default=REPO_ROOT / "configs" / "material_refine_train_paper_stage1.yaml")
    parser.add_argument("--eval-config", type=Path, default=REPO_ROOT / "configs" / "material_refine_eval_paper_benchmark.yaml")
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output" / "material_refine_paper" / "ablation_suite")
    parser.add_argument("--python-bin", type=Path, default=Path("/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python"))
    parser.add_argument("--run-learned-ablations", type=parse_bool, default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="online")
    parser.add_argument("--tracker-group", type=str, default="paper-ablation")
    return parser.parse_args()


def run_command(cmd: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"command_failed:{process.returncode}:{' '.join(cmd)}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_root / "logs"
    results: dict[str, Any] = {}

    for name, eval_variant in EVAL_ONLY_ABLATIONS.items():
        eval_dir = args.output_root / f"eval_{name}"
        cmd = [
            str(args.python_bin),
            "scripts/eval_material_refiner.py",
            "--config",
            str(args.eval_config),
            "--manifest",
            str(args.stage1_manifest),
            "--checkpoint",
            str(args.reference_checkpoint),
            "--split",
            "test",
            "--split-strategy",
            "manifest",
            "--eval-variant",
            eval_variant,
            "--output-dir",
            str(eval_dir),
            "--tracker-run-name",
            f"material-refine-paper-ablation-{name}",
            "--tracker-group",
            args.tracker_group,
            "--tracker-tags",
            f"material-refine,paper,ablation,{name}",
            "--wandb-mode",
            args.wandb_mode,
            "--num-workers",
            str(args.num_workers),
        ]
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])
        if args.device:
            cmd.extend(["--device", str(args.device)])
        run_command(cmd, log_path=logs_dir / f"eval_{name}.log")
        results[name] = {
            "mode": "eval_only",
            "eval_variant": eval_variant,
            "eval_dir": str(eval_dir),
        }

    if args.run_learned_ablations:
        for name, override_config in LEARNED_ABLATIONS.items():
            train_dir = args.output_root / f"train_{name}"
            eval_dir = args.output_root / f"eval_{name}"
            train_cmd = [
                str(args.python_bin),
                "scripts/train_material_refiner.py",
                "--config",
                str(args.train_config),
                "--config",
                str(override_config),
                "--manifest",
                str(args.stage1_manifest),
                "--train-manifest",
                str(args.stage1_manifest),
                "--val-manifest",
                str(args.stage1_manifest),
                "--split-strategy",
                "manifest",
                "--train-split",
                "train",
                "--val-split",
                "val",
                "--output-dir",
                str(train_dir),
                "--tracker-run-name",
                f"material-refine-paper-ablation-{name}",
                "--tracker-group",
                args.tracker_group,
                "--tracker-tags",
                f"material-refine,paper,ablation,{name}",
                "--wandb-mode",
                args.wandb_mode,
            ]
            run_command(train_cmd, log_path=logs_dir / f"train_{name}.log")
            checkpoint = train_dir / "best.pt"
            if not checkpoint.exists():
                checkpoint = train_dir / "latest.pt"
            if not checkpoint.exists():
                raise RuntimeError(f"checkpoint_not_found:{train_dir}")
            eval_cmd = [
                str(args.python_bin),
                "scripts/eval_material_refiner.py",
                "--config",
                str(args.eval_config),
                "--manifest",
                str(args.stage1_manifest),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--split-strategy",
                "manifest",
                "--output-dir",
                str(eval_dir),
                "--tracker-run-name",
                f"material-refine-paper-ablation-{name}-eval",
                "--tracker-group",
                args.tracker_group,
                "--tracker-tags",
                f"material-refine,paper,ablation,{name}",
                "--wandb-mode",
                args.wandb_mode,
                "--num-workers",
                str(args.num_workers),
            ]
            if args.max_samples is not None:
                eval_cmd.extend(["--max-samples", str(args.max_samples)])
            if args.device:
                eval_cmd.extend(["--device", str(args.device)])
            run_command(eval_cmd, log_path=logs_dir / f"eval_{name}_trained.log")
            results[name] = {
                "mode": "train_and_eval",
                "override_config": str(override_config),
                "train_dir": str(train_dir),
                "eval_dir": str(eval_dir),
                "checkpoint": str(checkpoint),
            }

    write_json(args.output_root / "ablation_suite_results.json", results)
    print(json.dumps({"output_root": str(args.output_root), "ablations": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
