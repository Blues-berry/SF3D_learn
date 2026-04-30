#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPANSION_ROOT = REPO_ROOT / "output/material_refine_expansion_candidates"
DEFAULT_MERGED_MANIFEST = EXPANSION_ROOT / "merged_expansion_candidate_manifest.json"
DEFAULT_OBJAVERSE_ROOT = REPO_ROOT / "output/highlight_pool_a_8k/aux_sources/objaverse_xl"
DEFAULT_OBJAVERSE_STAGE_ROOT = REPO_ROOT / "output/highlight_pool_a_8k/objaverse_cached_increment_material_priority"
DEFAULT_OBJAVERSE_SOURCE_DIR = EXPANSION_ROOT / "material_priority_objaverse_increment"
DEFAULT_STAGE_ROOT = EXPANSION_ROOT / "material_priority_stage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Stage material-priority source downloads and connect them to the active TrainV5 queue flow.",
    )
    parser.add_argument("--expansion-root", type=Path, default=EXPANSION_ROOT)
    parser.add_argument("--merged-manifest", type=Path, default=DEFAULT_MERGED_MANIFEST)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--objaverse-root", type=Path, default=DEFAULT_OBJAVERSE_ROOT)
    parser.add_argument("--objaverse-stage-root", type=Path, default=DEFAULT_OBJAVERSE_STAGE_ROOT)
    parser.add_argument("--objaverse-source-dir", type=Path, default=DEFAULT_OBJAVERSE_SOURCE_DIR)
    parser.add_argument("--objaverse-target", type=int, default=1200)
    parser.add_argument("--objaverse-sources", type=str, default="smithsonian,thingiverse,sketchfab")
    parser.add_argument(
        "--priority-material-families",
        type=str,
        default="mixed_thin_boundary,glass_metal,ceramic_glazed_lacquer,metal_dominant,glossy_non_metal",
    )
    parser.add_argument(
        "--target-material-family-ratios",
        type=str,
        default="mixed_thin_boundary=0.28,glass_metal=0.24,ceramic_glazed_lacquer=0.22,metal_dominant=0.18,glossy_non_metal=0.08",
    )
    parser.add_argument("--source-priority", type=str, default="smithsonian,thingiverse,sketchfab")
    parser.add_argument("--download-objaverse", action="store_true")
    parser.add_argument("--download-mode", choices=("direct", "proxy-probe", "mirror-probe", "off"), default="direct")
    parser.add_argument("--download-probe-size", type=int, default=100)
    parser.add_argument("--min-download-success-rate", type=float, default=0.20)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-second-pass", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def file_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def summarize_source_candidate_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    records = [row for row in rows if isinstance(row, dict)]
    return {
        "records": len(records),
        "source_name": dict(Counter(str(row.get("source_name") or "unknown") for row in records)),
        "license_bucket": dict(Counter(str(row.get("license_bucket") or "unknown") for row in records)),
        "material_family": dict(
            Counter(
                str(row.get("material_family") or row.get("highlight_material_class") or "unknown")
                for row in records
            )
        ),
    }


def write_source_sidecars(source_dir: Path, summary: dict[str, Any]) -> None:
    write_text(
        source_dir / "source_progress.md",
        "\n".join(
            [
                "# Source Progress",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- records: `{summary['records']}`",
                f"- source_name: `{json.dumps(summary['source_name'], ensure_ascii=False)}`",
            ]
        ),
    )
    write_text(
        source_dir / "source_license_summary.md",
        "\n".join(
            [
                "# Source License Summary",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- license_bucket: `{json.dumps(summary['license_bucket'], ensure_ascii=False)}`",
            ]
        ),
    )
    write_text(
        source_dir / "source_material_family_guess.md",
        "\n".join(
            [
                "# Source Material Family Guess",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- material_family: `{json.dumps(summary['material_family'], ensure_ascii=False)}`",
            ]
        ),
    )


def main() -> None:
    args = parse_args()
    args.stage_root.mkdir(parents=True, exist_ok=True)
    args.objaverse_stage_root.mkdir(parents=True, exist_ok=True)
    args.objaverse_source_dir.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = []
    effective_download_mode = str(args.download_mode)
    proxy_env_present = any(os.environ.get(key) for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))
    effective_target = int(args.objaverse_target)
    should_download = bool(args.download_objaverse) and effective_download_mode != "off"
    objaverse_plan_root = args.stage_root / "objaverse_1200_serial"
    objaverse_download_status = objaverse_plan_root / "objaverse_1200_download_status.json"
    objaverse_increment_manifest = objaverse_plan_root / "objaverse_1200_aggregate_increment_manifest.json"
    download_attempted = 0
    download_succeeded = 0
    download_failure_reason = ""
    download_success_rate = 0.0
    source_summary = {"records": 0, "source_name": {}, "license_bucket": {}, "material_family": {}}
    source_manifest_path = args.objaverse_source_dir / "source_candidate_manifest.json"
    objaverse_plan_status: dict[str, Any] = {}

    if effective_target > 0:
        objaverse_stage = run_cmd(
            [
                sys.executable,
                "scripts/manage_trainv5_objaverse_1200_serial_plan.py",
                "--objaverse-root",
                str(args.objaverse_root),
                "--plan-root",
                str(objaverse_plan_root),
                "--source-dir",
                str(args.objaverse_source_dir),
                "--sources",
                args.objaverse_sources,
                "--target-total",
                str(effective_target),
                "--priority-material-families",
                args.priority_material_families,
                "--target-material-family-ratios",
                args.target_material_family_ratios,
                "--source-priority",
                args.source_priority,
                *(["--download-objaverse"] if should_download else []),
                "--download-mode",
                effective_download_mode,
            ],
            cwd=REPO_ROOT,
        )
        command_log.append(objaverse_stage)
        if objaverse_stage["returncode"] != 0:
            raise SystemExit(objaverse_stage["stderr"] or objaverse_stage["stdout"] or "objaverse_stage_failed")

        if file_exists(objaverse_download_status):
            objaverse_plan_status = read_json(objaverse_download_status)
            download_attempted = int(objaverse_plan_status.get("download_attempted_latest") or 0)
            download_succeeded = int(objaverse_plan_status.get("downloaded_latest") or 0)
            download_failure_reason = str(objaverse_plan_status.get("download_failure_reason_latest") or "")
            download_success_rate = float(objaverse_plan_status.get("download_success_rate_latest") or 0.0)
            source_manifest_path = Path(
                str((objaverse_plan_status.get("source_manifest_files") or {}).get("compat_json") or source_manifest_path)
            )

        if file_exists(source_manifest_path):
            source_summary = summarize_source_candidate_manifest(source_manifest_path)
            write_source_sidecars(args.objaverse_source_dir, source_summary)

    merge_result = None
    if not args.skip_merge:
        merge_result = run_cmd(
            [
                sys.executable,
                "scripts/merge_material_refine_expansion_candidates.py",
                "--expansion-root",
                str(args.expansion_root),
                "--output-manifest",
                str(args.merged_manifest),
            ],
            cwd=REPO_ROOT,
        )
        command_log.append(merge_result)
        if merge_result["returncode"] != 0:
            raise SystemExit(merge_result["stderr"] or merge_result["stdout"] or "merge_expansion_candidates_failed")

    second_pass_result = None
    if not args.skip_second_pass:
        second_pass_result = run_cmd(
            [
                sys.executable,
                "scripts/build_material_refine_trainV5_expansion_second_pass.py",
                "--input-manifest",
                str(args.merged_manifest),
            ],
            cwd=REPO_ROOT,
        )
        command_log.append(second_pass_result)
        if second_pass_result["returncode"] != 0:
            raise SystemExit(second_pass_result["stderr"] or second_pass_result["stdout"] or "second_pass_failed")

    repair_result = None
    if not args.skip_repair:
        repair_result = run_cmd(
            [
                sys.executable,
                "scripts/build_material_refine_trainV5_repair_and_expansion_plan.py",
                "--second-pass-dir",
                str(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass"),
            ],
            cwd=REPO_ROOT,
        )
        command_log.append(repair_result)
        if repair_result["returncode"] != 0:
            raise SystemExit(repair_result["stderr"] or repair_result["stdout"] or "repair_and_queue_build_failed")

    objaverse_artifacts_result = run_cmd(
        [
            sys.executable,
            "scripts/build_trainv5_objaverse_1200_serial_artifacts.py",
            "--source-manifest",
            str(source_manifest_path),
            "--second-pass-dir",
            str(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass"),
            "--target-total",
            str(args.objaverse_target),
        ],
        cwd=REPO_ROOT,
    )
    command_log.append(objaverse_artifacts_result)
    if objaverse_artifacts_result["returncode"] != 0:
        raise SystemExit(objaverse_artifacts_result["stderr"] or objaverse_artifacts_result["stdout"] or "objaverse_1200_artifact_build_failed")

    summary = {
        "generated_at_utc": utc_now(),
        "objaverse_target": args.objaverse_target,
        "effective_objaverse_target": effective_target,
        "objaverse_sources": [item.strip() for item in args.objaverse_sources.split(",") if item.strip()],
        "priority_material_families": [item.strip() for item in args.priority_material_families.split(",") if item.strip()],
        "target_material_family_rats": args.target_material_family_ratios,
        "download_objaverse": should_download,
        "download_mode": effective_download_mode,
        "download_transport": effective_download_mode,
        "download_probe_size": int(args.download_probe_size),
        "min_download_success_rate": float(args.min_download_success_rate),
        "proxy_env_present": proxy_env_present,
        "download_attempted": int(download_attempted),
        "download_succeeded": int(download_succeeded),
        "download_success_rate": float(download_success_rate),
        "download_failure_reason": download_failure_reason,
        "download_below_threshold": bool(
            should_download and download_attempted > 0 and download_success_rate < float(args.min_download_success_rate)
        ),
        "source_candidate_manifest": str(source_manifest_path.resolve()) if source_manifest_path.exists() else str(source_manifest_path),
        "objaverse_1200_download_status": str(objaverse_download_status.resolve()) if objaverse_download_status.exists() else str(objaverse_download_status),
        "objaverse_1200_ready_exact_1200": bool(objaverse_plan_status.get("ready_exact_1200")),
        "objaverse_1200_missing_total": int(objaverse_plan_status.get("missing_total") or 0),
        "objaverse_1200_retry_round": int(objaverse_plan_status.get("retry_round") or 0),
        "objaverse_1200_topup_needed": bool(objaverse_plan_status.get("topup_needed")),
        "merged_manifest": str(args.merged_manifest.resolve()),
        "queue_latest": str((REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/trainV5_plus_rebake_queue_latest.json").resolve()),
        "objaverse_1200_second_pass_summary": str((REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/objaverse_1200_serial/objaverse_1200_second_pass_summary.json").resolve()),
        "objaverse_1200_repair_summary": str((REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/objaverse_1200_serial/objaverse_1200_repair_summary.json").resolve()),
        "objaverse_1200_rebake_input_manifest": str((REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/objaverse_1200_serial/objaverse_1200_rebake_input_manifest.json").resolve()),
        "command_log_count": len(command_log),
        "source_summary": source_summary,
    }
    write_json(args.stage_root / "material_priority_stage_summary.json", {"summary": summary, "commands": command_log})
    write_text(
        args.stage_root / "material_priority_stage_summary.md",
        "\n".join(
            [
                "# Material Priority Stage Summary",
                "",
                f"- generated_at_utc: `{summary['generated_at_utc']}`",
                f"- objaverse_target: `{summary['objaverse_target']}`",
                f"- effective_objaverse_target: `{summary['effective_objaverse_target']}`",
                f"- objaverse_sources: `{json.dumps(summary['objaverse_sources'], ensure_ascii=False)}`",
                f"- priority_material_families: `{json.dumps(summary['priority_material_families'], ensure_ascii=False)}`",
                f"- download_objaverse: `{str(summary['download_objaverse']).lower()}`",
                f"- download_mode: `{summary['download_mode']}`",
                f"- download_transport: `{summary['download_transport']}`",
                f"- proxy_env_present: `{str(summary['proxy_env_present']).lower()}`",
                f"- download_attempted: `{summary['download_attempted']}`",
                f"- download_succeeded: `{summary['download_succeeded']}`",
                f"- download_success_rate: `{summary['download_success_rate']}`",
                f"- download_failure_reason: `{summary['download_failure_reason']}`",
                f"- download_below_threshold: `{str(summary['download_below_threshold']).lower()}`",
                f"- objaverse_1200_download_status: `{summary['objaverse_1200_download_status']}`",
                f"- objaverse_1200_ready_exact_1200: `{str(summary['objaverse_1200_ready_exact_1200']).lower()}`",
                f"- objaverse_1200_missing_total: `{summary['objaverse_1200_missing_total']}`",
                f"- objaverse_1200_retry_round: `{summary['objaverse_1200_retry_round']}`",
                f"- objaverse_1200_topup_needed: `{str(summary['objaverse_1200_topup_needed']).lower()}`",
                f"- source_candidate_manifest: `{summary['source_candidate_manifest']}`",
                f"- merged_manifest: `{summary['merged_manifest']}`",
                f"- queue_latest: `{summary['queue_latest']}`",
                f"- objaverse_1200_second_pass_summary: `{summary['objaverse_1200_second_pass_summary']}`",
                f"- objaverse_1200_repair_summary: `{summary['objaverse_1200_repair_summary']}`",
                f"- objaverse_1200_rebake_input_manifest: `{summary['objaverse_1200_rebake_input_manifest']}`",
                f"- source_summary: `{json.dumps(summary['source_summary'], ensure_ascii=False)}`",
                "",
                "This stage does not launch GPU rebake or training. It only stages sources, merges candidates, and refreshes CPU-side queue artifacts.",
            ]
        ),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
