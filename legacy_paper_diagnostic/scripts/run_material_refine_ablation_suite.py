from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


ABLATIONS: dict[str, dict[str, Any]] = {
    "ours_full": {},
    "without_view_fusion": {"disable_view_fusion": True},
    "without_no_prior_bootstrap": {
        "disable_no_prior_bootstrap": True,
        "enable_no_prior_bootstrap": False,
    },
    "without_boundary_safety": {
        "disable_boundary_safety": True,
        "enable_boundary_safety": False,
    },
    "without_change_gate": {
        "disable_change_gate": True,
        "enable_change_gate": False,
    },
    "without_prior_source_conditioning": {
        "disable_prior_source_embedding": True,
        "enable_prior_source_embedding": False,
    },
    "without_prior_safe_loss": {"disable_prior_safe_loss": True},
    "without_residual_head": {"disable_residual_head": True},
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Material-refine R-v2 ablation launcher/config generator.",
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "material_refine_ablations")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--train-script", type=Path, default=REPO_ROOT / "scripts" / "train_material_refiner.py")
    parser.add_argument("--dry-run", type=parse_bool, default=True)
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args()


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def write_override(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{key}: {yaml_scalar(value)}" for key, value in sorted(payload.items())) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    for name, override in ABLATIONS.items():
        override_path = args.output_dir / "configs" / f"{name}.yaml"
        run_dir = args.output_dir / "runs" / name
        payload = {
            "tracker_group": f"material-refine-ablation-{args.output_dir.name}",
            "tracker_run_name": name,
            "output_dir": str(run_dir),
            **override,
        }
        write_override(override_path, payload)
        command = [
            args.python,
            str(args.train_script),
            "--config",
            str(args.base_config),
            "--method-config",
            str(override_path),
            *args.extra_arg,
        ]
        commands.append(command)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)

    summary = {
        "base_config": str(args.base_config),
        "output_dir": str(args.output_dir.resolve()),
        "dry_run": bool(args.dry_run),
        "ablations": [
            {
                "name": name,
                "override": ABLATIONS[name],
                "config": str((args.output_dir / "configs" / f"{name}.yaml").resolve()),
                "command": commands[index],
            }
            for index, name in enumerate(ABLATIONS)
        ],
        "required_metrics": [
            "input_prior_total_mae",
            "refined_total_mae",
            "gain_total",
            "regression_rate",
        ],
    }
    summary_path = args.output_dir / "ablation_suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
