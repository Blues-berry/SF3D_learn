from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_csv(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in str(value).split(",") if item.strip()}
    return items or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Launch the Round9 boundary-band smoke ablation matrix with resolved configs and logs.",
    )
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=REPO_ROOT / "configs" / "material_refine_round9_boundary_ablation_matrix.yaml",
    )
    parser.add_argument("--python-bin", type=Path, default=Path("/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python"))
    parser.add_argument("--only", type=str, default=None, help="Comma-separated variant names to run.")
    parser.add_argument("--max-variants", type=int, default=0, help="0 means all selected variants.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override variant output dirs under this root.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cuda-device-index", type=int, default=None)
    parser.add_argument("--report-to", choices=["none", "wandb"], default=None)
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default=None)
    parser.add_argument("--dry-run", type=parse_bool, default=False)
    parser.add_argument("--continue-on-error", type=parse_bool, default=False)
    return parser.parse_args()


def plain_dict(value: Any) -> dict[str, Any]:
    payload = OmegaConf.to_container(value, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("expected_mapping")
    return payload


def merge_variant_config(matrix: dict[str, Any], variant: dict[str, Any], *, output_root: Path | None) -> dict[str, Any]:
    base_config = REPO_ROOT / str(matrix["base_config"])
    base = plain_dict(OmegaConf.load(base_config))
    defaults = dict(matrix.get("default_smoke_overrides") or {})
    variant_overrides = dict(variant)
    variant_name = str(variant_overrides.pop("name"))

    merged = OmegaConf.merge(base, defaults, variant_overrides)
    payload = plain_dict(merged)
    payload["round9_ablation_variant"] = variant_name
    payload["round9_ablation_matrix_config"] = str(Path(matrix["matrix_config_path"]).resolve())
    if output_root is not None:
        payload["output_dir"] = str(output_root / variant_name)
    if "tracker_run_name" not in payload or not payload["tracker_run_name"]:
        payload["tracker_run_name"] = f"material-refine-round9-{variant_name}"
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_variant(
    *,
    args: argparse.Namespace,
    variant_name: str,
    train_config: dict[str, Any],
    logs_dir: Path,
) -> dict[str, Any]:
    output_dir = REPO_ROOT / str(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = output_dir / "resolved_round9_ablation_config.yaml"
    OmegaConf.save(config=OmegaConf.create(train_config), f=resolved_config)

    cmd = [
        str(args.python_bin),
        "scripts/train_material_refiner.py",
        "--config",
        str(resolved_config),
    ]
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.cuda_device_index is not None:
        cmd.extend(["--cuda-device-index", str(args.cuda_device_index)])
    if args.report_to is not None:
        cmd.extend(["--report-to", args.report_to])
    if args.wandb_mode is not None:
        cmd.extend(["--wandb-mode", args.wandb_mode])

    result = {
        "variant": variant_name,
        "output_dir": str(output_dir),
        "resolved_config": str(resolved_config),
        "command": cmd,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        return result

    log_path = logs_dir / f"{variant_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    result.update(
        {
            "returncode": process.returncode,
            "log_path": str(log_path),
            "elapsed_seconds": time.time() - started_at,
            "best_checkpoint": str(output_dir / "best.pt"),
            "latest_checkpoint": str(output_dir / "latest.pt"),
        }
    )
    if process.returncode != 0 and not args.continue_on_error:
        raise RuntimeError(f"round9_ablation_failed:{variant_name}:returncode={process.returncode}:log={log_path}")
    return result


def main() -> None:
    args = parse_args()
    matrix_path = args.matrix_config.resolve()
    matrix = plain_dict(OmegaConf.load(matrix_path))
    matrix["matrix_config_path"] = str(matrix_path)
    selected_names = parse_csv(args.only)
    variants = list(matrix.get("variants") or [])
    if selected_names is not None:
        variants = [variant for variant in variants if str(variant.get("name")) in selected_names]
    if args.max_variants > 0:
        variants = variants[: args.max_variants]
    if not variants:
        raise ValueError("no_round9_ablation_variants_selected")

    output_root = args.output_root.resolve() if args.output_root is not None else None
    logs_dir = (output_root or (REPO_ROOT / "output" / "material_refine_paper" / "round9_boundary_ablation")) / "logs"
    results: list[dict[str, Any]] = []
    for variant in variants:
        variant_dict = dict(variant)
        variant_name = str(variant_dict.get("name"))
        train_config = merge_variant_config(matrix, variant_dict, output_root=output_root)
        print(
            f"[round9_ablation] variant={variant_name} output={train_config['output_dir']} dry_run={args.dry_run}",
            flush=True,
        )
        results.append(
            run_variant(
                args=args,
                variant_name=variant_name,
                train_config=train_config,
                logs_dir=logs_dir,
            )
        )

    summary_path = (output_root or (REPO_ROOT / "output" / "material_refine_paper" / "round9_boundary_ablation")) / "round9_ablation_launcher_summary.json"
    write_json(summary_path, {"matrix_config": str(matrix_path), "results": results})
    print(
        json.dumps({"summary": str(summary_path), "variants": [item["variant"] for item in results]}, indent=2, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
