from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the full paper-stage material-refine pipeline once the manifest passes P0 readiness gates.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "output" / "material_refine_pipeline_20260418T091559Z" / "prepared" / "full" / "canonical_manifest_full.json",
    )
    parser.add_argument("--candidate-manifest", action="append", type=Path, default=[])
    parser.add_argument("--manifest-glob", action="append", type=str, default=[])
    parser.add_argument(
        "--train-config",
        type=Path,
        default=REPO_ROOT / "configs" / "material_refine_train_paper_stage1.yaml",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=REPO_ROOT / "configs" / "material_refine_eval_paper_benchmark.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output" / "material_refine_paper" / "paper_stage1_pipeline",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python"),
    )
    parser.add_argument("--wait-for-ready", type=parse_bool, default=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-polls", type=int, default=0, help="0 means no explicit limit.")
    parser.add_argument("--skip-train", type=parse_bool, default=False)
    parser.add_argument("--skip-panels", type=parse_bool, default=False)
    parser.add_argument("--skip-round-summary", type=parse_bool, default=False)
    parser.add_argument("--tracker-group", type=str, default="paper-stage1")
    parser.add_argument("--tracker-tags", type=str, default="material-refine,paper,stage1,pipeline")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="online")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(cmd: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
    if process.returncode != 0:
        raise RuntimeError(f"command_failed:{process.returncode}:{' '.join(cmd)}")


def count_manifest_records(path: Path) -> int:
    if not path.exists():
        return 0
    payload = load_json(path)
    records = payload.get("records")
    return len(records) if isinstance(records, list) else 0


def resolve_selected_manifest(args: argparse.Namespace, readiness_root: Path) -> Path:
    if not args.candidate_manifest and not args.manifest_glob:
        return args.manifest.resolve()
    selection_root = readiness_root / "manifest_selection"
    selection_output = selection_root / "best_manifest_selection.json"
    selection_cache = selection_root / "manifest_selection_cache.json"
    cmd = [
        str(args.python_bin),
        "scripts/select_material_refine_best_manifest.py",
        "--output-json",
        str(selection_output),
        "--cache-json",
        str(selection_cache),
    ]
    for manifest in args.candidate_manifest:
        cmd.extend(["--manifest", str(manifest)])
    for pattern in args.manifest_glob:
        cmd.extend(["--manifest-glob", pattern])
    run_command(cmd, log_path=selection_root / "manifest_selection.log")
    selection_payload = load_json(selection_output)
    selected = selection_payload.get("selected_manifest")
    if not selected:
        raise RuntimeError("no_candidate_manifest_found")
    return Path(str(selected)).resolve()


def ensure_readiness(args: argparse.Namespace, readiness_root: Path) -> dict[str, Any]:
    poll_count = 0
    while True:
        audit_dir = readiness_root / "manifest_audit"
        buffer_dir = readiness_root / "buffer_validation"
        stage1_root = readiness_root / "stage1_subset"
        logs_dir = readiness_root / "logs"
        try:
            selected_manifest = resolve_selected_manifest(args, readiness_root)
        except RuntimeError as exc:
            if not str(exc).startswith("no_candidate_manifest_found"):
                raise
            readiness_state = {
                "manifest": None,
                "paper_stage_ready": False,
                "stage1_subset_records": 0,
                "ood_eval_records": 0,
                "readiness_blockers": [str(exc)],
                "poll_count": poll_count,
                "updated_at_unix": time.time(),
            }
            write_json(readiness_root / "readiness_state.json", readiness_state)
            poll_count += 1
            if not args.wait_for_ready:
                raise
            if args.max_polls > 0 and poll_count >= args.max_polls:
                raise RuntimeError("paper_stage_not_ready_max_polls_exceeded")
            time.sleep(max(int(args.poll_seconds), 1))
            continue

        run_command(
            [
                str(args.python_bin),
                "scripts/audit_material_refine_manifest.py",
                "--manifest",
                str(selected_manifest),
                "--output-dir",
                str(audit_dir),
            ],
            log_path=logs_dir / "manifest_audit.log",
        )
        run_command(
            [
                str(args.python_bin),
                "scripts/validate_material_refine_buffers.py",
                "--manifest",
                str(selected_manifest),
                "--output-dir",
                str(buffer_dir),
            ],
            log_path=logs_dir / "buffer_validation.log",
        )
        run_command(
            [
                str(args.python_bin),
                "scripts/build_material_refine_paper_stage1_subset.py",
                "--manifest",
                str(selected_manifest),
                "--output-root",
                str(stage1_root),
            ],
            log_path=logs_dir / "stage1_subset.log",
        )

        readiness_summary_path = stage1_root / "paper_stage1_readiness_summary.json"
        readiness_summary = load_json(readiness_summary_path)
        readiness_state = {
            "manifest": str(selected_manifest),
            "paper_stage_ready": bool(readiness_summary.get("paper_stage_ready")),
            "stage1_subset_records": int(readiness_summary.get("stage1_subset_records", 0)),
            "ood_eval_records": int(readiness_summary.get("ood_eval_records", 0)),
            "readiness_blockers": readiness_summary.get("readiness_blockers", []),
            "poll_count": poll_count,
            "updated_at_unix": time.time(),
        }
        write_json(readiness_root / "readiness_state.json", readiness_state)
        if readiness_state["paper_stage_ready"] and readiness_state["stage1_subset_records"] > 0:
            return readiness_summary

        poll_count += 1
        if not args.wait_for_ready:
            raise RuntimeError("paper_stage_not_ready")
        if args.max_polls > 0 and poll_count >= args.max_polls:
            raise RuntimeError("paper_stage_not_ready_max_polls_exceeded")
        time.sleep(max(int(args.poll_seconds), 1))


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "pipeline_args.json", make_json_serializable(vars(args)))

    readiness_root = args.output_root / "readiness"
    try:
        readiness_summary = ensure_readiness(args, readiness_root)
    except RuntimeError as exc:
        readiness_state_path = readiness_root / "readiness_state.json"
        readiness_state = load_json(readiness_state_path) if readiness_state_path.exists() else {}
        blocked_payload = {
            "status": "blocked",
            "reason": str(exc),
            "output_root": str(args.output_root),
            "readiness_state": readiness_state,
        }
        write_json(args.output_root / "pipeline_result.json", blocked_payload)
        print(json.dumps(blocked_payload, indent=2, ensure_ascii=False))
        sys.exit(3)
    stage1_root = readiness_root / "stage1_subset"
    stage1_manifest = stage1_root / "paper_stage1_subset_manifest.json"
    ood_manifest = stage1_root / "paper_stage1_ood_manifest.json"
    selected_manifest = Path(str(readiness_summary["manifest"])).resolve()

    train_dir = args.output_root / "train_stage1_main"
    eval_dir = args.output_root / "eval_stage1_test"
    ood_eval_dir = args.output_root / "eval_stage1_ood"
    panel_dir = eval_dir / "validation_comparison_panels"
    round_analysis_dir = args.output_root / "round_analysis"
    logs_dir = args.output_root / "logs"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_base_cmd = [
        str(args.python_bin),
        "scripts/train_material_refiner.py",
        "--config",
        str(args.train_config),
        "--manifest",
        str(stage1_manifest),
        "--train-manifest",
        str(stage1_manifest),
        "--val-manifest",
        str(stage1_manifest),
        "--split-strategy",
        "manifest",
        "--train-split",
        "train",
        "--val-split",
        "val",
        "--freeze-val-manifest-to",
        str(train_dir / "stage1_frozen_val_manifest.json"),
        "--output-dir",
        str(train_dir),
    ]

    run_command(
        train_base_cmd + ["--preflight-only"],
        log_path=logs_dir / "train_preflight.log",
    )

    if not args.skip_train:
        run_command(
            train_base_cmd,
            log_path=logs_dir / "train.log",
        )

    checkpoint = train_dir / "best.pt"
    if not checkpoint.exists():
        checkpoint = train_dir / "latest.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"checkpoint_not_found:{train_dir}")

    eval_base_cmd = [
        str(args.python_bin),
        "scripts/eval_material_refiner.py",
        "--config",
        str(args.eval_config),
        "--checkpoint",
        str(checkpoint),
        "--tracker-group",
        args.tracker_group,
        "--tracker-tags",
        args.tracker_tags,
        "--wandb-mode",
        args.wandb_mode,
    ]
    run_command(
        eval_base_cmd
        + [
            "--manifest",
            str(stage1_manifest),
            "--split",
            "test",
            "--split-strategy",
            "manifest",
            "--paper-splits",
            "paper_test_iid,paper_test_material_holdout,paper_test_real_lighting",
            "--output-dir",
            str(eval_dir),
            "--tracker-run-name",
            "material-refine-paper-stage1-test",
        ],
        log_path=logs_dir / "eval_test.log",
    )

    if count_manifest_records(ood_manifest) > 0:
        run_command(
            eval_base_cmd
            + [
                "--manifest",
                str(ood_manifest),
                "--split",
                "all",
                "--split-strategy",
                "manifest",
                "--paper-splits",
                "paper_test_ood_object",
                "--output-dir",
                str(ood_eval_dir),
                "--tracker-run-name",
                "material-refine-paper-stage1-ood",
            ],
            log_path=logs_dir / "eval_ood.log",
        )

    run_command(
        [
            str(args.python_bin),
            "scripts/export_material_attribute_comparison.py",
            "--metrics-json",
            str(eval_dir / "metrics.json"),
            "--output-dir",
            str(eval_dir),
        ],
        log_path=logs_dir / "attribute_comparison.log",
    )

    if not args.skip_panels:
        run_command(
            [
                str(args.python_bin),
                "scripts/export_material_validation_comparison_panels.py",
                "--manifest",
                str(stage1_manifest),
                "--metrics",
                str(eval_dir / "metrics.json"),
                "--output-dir",
                str(panel_dir),
                "--max-panels",
                "64",
                "--report-to",
                "wandb",
                "--tracker-run-name",
                "material-refine-paper-stage1-panels",
                "--tracker-group",
                args.tracker_group,
                "--tracker-tags",
                args.tracker_tags,
                "--wandb-mode",
                args.wandb_mode,
            ],
            log_path=logs_dir / "validation_panels.log",
        )

    run_command(
        [
            str(args.python_bin),
            "scripts/export_material_refine_round_analysis.py",
            "--train-dir",
            str(train_dir),
            "--eval-dir",
            str(eval_dir),
            "--audit-dir",
            str(readiness_root / "manifest_audit"),
            "--output-dir",
            str(round_analysis_dir),
        ],
        log_path=logs_dir / "round_analysis.log",
    )

    if not args.skip_round_summary:
        run_command(
            [
                str(args.python_bin),
                "scripts/log_material_refine_round_to_wandb.py",
                "--train-dir",
                str(train_dir),
                "--eval-dir",
                str(eval_dir),
                "--audit-dir",
                str(readiness_root / "manifest_audit"),
                "--round-analysis-dir",
                str(round_analysis_dir),
                "--panel-dir",
                str(panel_dir),
                "--tracker-run-name",
                "material-refine-paper-stage1-summary",
                "--tracker-group",
                args.tracker_group,
                "--tracker-tags",
                args.tracker_tags,
                "--wandb-mode",
                args.wandb_mode,
            ],
            log_path=logs_dir / "wandb_round_summary.log",
        )

    write_json(
        args.output_root / "pipeline_result.json",
        {
            "manifest": str(selected_manifest),
            "stage1_manifest": str(stage1_manifest.resolve()),
            "ood_manifest": str(ood_manifest.resolve()),
            "readiness_summary": readiness_summary,
            "checkpoint": str(checkpoint.resolve()),
            "train_dir": str(train_dir.resolve()),
            "eval_dir": str(eval_dir.resolve()),
            "ood_eval_dir": str(ood_eval_dir.resolve()) if ood_eval_dir.exists() else None,
            "panel_dir": str(panel_dir.resolve()),
            "round_analysis_dir": str(round_analysis_dir.resolve()),
        },
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "output_root": str(args.output_root),
                "train_dir": str(train_dir),
                "eval_dir": str(eval_dir),
                "checkpoint": str(checkpoint),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
