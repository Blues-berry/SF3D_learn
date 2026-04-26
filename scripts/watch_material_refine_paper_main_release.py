#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "material_refine_dataset_factory_gpu0.json"
DEFAULT_OK_MD = REPO_ROOT / "paper 主训练OK.md"
PYTHON_BIN = Path("/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def manifest_record_count(path: Path) -> int:
    try:
        payload = read_json(path)
    except Exception:
        return 0
    records = payload.get("records") or []
    return len(records) if isinstance(records, list) else 0


def resolve_globs(patterns: list[str], *, min_records: int) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        raw_pattern = str(repo_path(pattern)) if not Path(pattern).is_absolute() else pattern
        for value in glob.glob(raw_pattern, recursive=True):
            path = Path(value)
            key = str(path.resolve())
            if key in seen or not path.exists():
                continue
            if manifest_record_count(path) < min_records:
                continue
            seen.add(key)
            paths.append(path)
    return sorted(paths, key=lambda item: (item.stat().st_mtime, str(item)))


def process_is_running(script_name: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", script_name],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return False
    current = str(Path(__file__).name)
    return any(current not in line for line in result.stdout.splitlines())


def run_capture(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-80:],
    }


def maybe_promote(config: dict[str, Any], *, force: bool) -> dict[str, Any]:
    promotion = config.get("promotion", {})
    if not bool(promotion.get("enabled", True)):
        return {"action": "promotion_disabled"}
    manifests = resolve_globs(
        [str(item) for item in promotion.get("input_manifest_globs", [])],
        min_records=int(promotion.get("min_input_records", 8)),
    )
    if not manifests:
        return {"action": "no_promotion_inputs"}
    output_manifest = repo_path(
        promotion.get(
            "output_manifest",
            "output/material_refine_paper/reworked_candidates/factory_promoted/latest/canonical_manifest_promoted.json",
        )
    )
    newest_input_mtime = max(path.stat().st_mtime for path in manifests)
    if (
        not force
        and output_manifest.exists()
        and output_manifest.stat().st_mtime >= newest_input_mtime
    ):
        return {"action": "promotion_up_to_date", "inputs": len(manifests)}
    min_input_age_seconds = float(promotion.get("min_input_mtime_age_seconds", 0.0))
    newest_input_age_seconds = max(0.0, time.time() - newest_input_mtime)
    if min_input_age_seconds > 0.0 and newest_input_age_seconds < min_input_age_seconds:
        return {
            "action": "input_manifest_still_updating",
            "inputs": len(manifests),
            "newest_input_age_seconds": round(newest_input_age_seconds, 1),
            "min_input_mtime_age_seconds": min_input_age_seconds,
        }
    if process_is_running("promote_material_refine_targets.py"):
        return {"action": "promotion_already_running", "inputs": len(manifests)}
    command = [
        str(PYTHON_BIN),
        "scripts/promote_material_refine_targets.py",
        "--output-manifest",
        str(output_manifest),
        "--report-json",
        str(repo_path(promotion.get("report_json", output_manifest.with_suffix(".promotion_report.json")))),
        "--report-md",
        str(repo_path(promotion.get("report_md", output_manifest.with_suffix(".promotion_report.md")))),
        "--report-html",
        str(repo_path(promotion.get("report_html", output_manifest.with_suffix(".promotion_report.html")))),
        "--allowed-license-buckets",
        str(promotion.get("allowed_license_buckets", "")),
        "--promotable-source-names",
        str(promotion.get("promotable_source_names", "")),
        "--min-confidence-mean",
        str(promotion.get("min_confidence_mean", 0.70)),
        "--min-confidence-nonzero-rate",
        str(promotion.get("min_confidence_nonzero_rate", 0.50)),
        "--min-target-coverage",
        str(promotion.get("min_target_coverage", 0.50)),
        "--max-target-prior-identity",
        str(promotion.get("max_target_prior_identity", 0.95)),
        "--min-valid-view-count",
        str(promotion.get("min_valid_view_count", 1)),
        "--min-strict-complete-view-rate",
        str(promotion.get("min_strict_complete_view_rate", 0.80)),
    ]
    if not bool(promotion.get("promote_auxiliary_to_paper_main", True)):
        command.append("--no-promote-auxiliary-to-paper-main")
    if not bool(promotion.get("keep_unpromoted_audit_fields", True)):
        command.append("--no-keep-unpromoted-audit-fields")
    for manifest in manifests:
        command.extend(["--manifest", str(manifest)])
    result = run_capture(command)
    result["action"] = "promoted" if result["returncode"] == 0 else "promotion_failed"
    result["inputs"] = len(manifests)
    return result


def maybe_build_stage1_v3(config: dict[str, Any], *, force: bool) -> dict[str, Any]:
    stage = config.get("stage1_v3", {})
    if not bool(stage.get("enabled", True)):
        return {"action": "stage1_v3_disabled"}
    input_manifest = repo_path(
        stage.get(
            "input_manifest",
            config.get("promotion", {}).get(
                "output_manifest",
                "output/material_refine_paper/reworked_candidates/factory_promoted/latest/canonical_manifest_promoted.json",
            ),
        )
    )
    if not input_manifest.exists():
        return {"action": "missing_stage1_input", "input_manifest": str(input_manifest)}
    output_root = repo_path(stage.get("output_root", "output/material_refine_paper/stage1_v3_dataset_latest"))
    report_json = output_root / "stage1_v3_dataset_audit.json"
    if (
        not force
        and report_json.exists()
        and report_json.stat().st_mtime >= input_manifest.stat().st_mtime
    ):
        return {"action": "stage1_v3_up_to_date", "report_json": str(report_json)}
    if process_is_running("build_material_refine_stage1_v3_subsets.py"):
        return {"action": "stage1_v3_already_running", "report_json": str(report_json)}
    command = [
        str(PYTHON_BIN),
        "scripts/build_material_refine_stage1_v3_subsets.py",
        "--manifest",
        str(input_manifest),
        "--output-root",
        str(output_root),
        "--target-records",
        str(stage.get("target_records", 900)),
        "--min-paper-eligible",
        str(stage.get("min_paper_eligible", 800)),
        "--min-material-family-records",
        str(stage.get("min_material_family_records", 64)),
        "--max-material-family-ratio",
        str(stage.get("max_material_family_ratio", 0.40)),
        "--min-no-prior-records",
        str(stage.get("min_no_prior_records", 100)),
        "--min-secondary-source-records",
        str(stage.get("min_secondary_source_records", 100)),
        "--min-confidence-mean",
        str(stage.get("min_confidence_mean", 0.70)),
        "--target-confidence-mean",
        str(stage.get("target_confidence_mean", 0.75)),
        "--min-confidence-nonzero-rate",
        str(stage.get("min_confidence_nonzero_rate", 0.50)),
        "--min-target-coverage",
        str(stage.get("min_target_coverage", 0.50)),
        "--identity-like-threshold",
        str(stage.get("identity_like_threshold", 0.999)),
        "--min-valid-view-count",
        str(stage.get("min_valid_view_count", 1)),
        "--max-diagnostic-records",
        str(stage.get("max_diagnostic_records", 1024)),
        "--max-ood-records",
        str(stage.get("max_ood_records", 512)),
        "--diagnostic-min-per-material-family",
        str(stage.get("diagnostic_min_per_material_family", 32)),
        "--ood-min-per-material-family",
        str(stage.get("ood_min_per_material_family", 24)),
        "--paper-license-buckets",
        str(stage.get("paper_license_buckets", config.get("promotion", {}).get("allowed_license_buckets", ""))),
        "--material-quotas",
        str(stage.get("material_quotas", "")),
        "--main-train-source-names",
        str(stage.get("main_train_source_names", "ABO_locked_core,3D-FUTURE_highlight_local_8k")),
        "--train-ratio",
        str(stage.get("train_ratio", 0.70)),
        "--val-ratio",
        str(stage.get("val_ratio", 0.12)),
        "--iid-test-ratio",
        str(stage.get("iid_test_ratio", 0.10)),
        "--material-holdout-ratio",
        str(stage.get("material_holdout_ratio", 0.08)),
    ]
    command.append("--fill-deficits" if bool(stage.get("fill_deficits", False)) else "--no-fill-deficits")
    result = run_capture(command)
    result["action"] = "stage1_v3_built" if result["returncode"] == 0 else "stage1_v3_failed"
    result["report_json"] = str(report_json)
    return result


def readiness_report(config: dict[str, Any]) -> dict[str, Any]:
    stage = config.get("stage1_v3", {})
    output_root = repo_path(stage.get("output_root", "output/material_refine_paper/stage1_v3_dataset_latest"))
    audit_path = output_root / "stage1_v3_dataset_audit.json"
    if not audit_path.exists():
        return {"ready": False, "reason": "missing_stage1_v3_audit", "audit_path": str(audit_path)}
    audit = read_json(audit_path)
    ready = bool(audit.get("stage1_v3_ready"))
    return {
        "ready": ready,
        "audit_path": str(audit_path),
        "recommendation": audit.get("recommendation"),
        "blockers": audit.get("blockers") or [],
        "subset_paths": audit.get("subset_paths") or {},
        "subset_summaries": audit.get("subset_summaries") or {},
    }


def write_ok_file(path: Path, report: dict[str, Any]) -> None:
    summaries = report.get("subset_summaries") or {}
    balanced = summaries.get("balanced_paper") or {}
    strict = summaries.get("strict_paper_candidates") or {}
    lines = [
        "# Paper 主训练 OK",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- readiness_audit: `{report.get('audit_path')}`",
        f"- recommendation: `{report.get('recommendation')}`",
        f"- strict_paper_candidates: `{strict.get('records')}`",
        f"- balanced_paper_records: `{balanced.get('records')}`",
        f"- balanced_material_family: `{json.dumps(balanced.get('material_family', {}), ensure_ascii=False)}`",
        f"- balanced_target_quality_tier: `{json.dumps(balanced.get('target_quality_tier', {}), ensure_ascii=False)}`",
        f"- balanced_target_source_type: `{json.dumps(balanced.get('target_source_type', {}), ensure_ascii=False)}`",
        f"- balanced_has_material_prior: `{json.dumps(balanced.get('has_material_prior', {}), ensure_ascii=False)}`",
        "",
        "该文件只表示数据侧 paper-stage 主训练门禁已通过；本 watcher 不会启动训练。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    report = readiness_report(config)
    actions: dict[str, Any] = {"readiness_before": report}
    if report.get("ready"):
        write_ok_file(args.ok_md, report)
        actions["ok_written"] = str(args.ok_md)
        return actions
    actions["promotion"] = maybe_promote(config, force=bool(args.force))
    actions["stage1_v3"] = maybe_build_stage1_v3(config, force=bool(args.force))
    report = readiness_report(config)
    actions["readiness_after"] = report
    if report.get("ready"):
        write_ok_file(args.ok_md, report)
        actions["ok_written"] = str(args.ok_md)
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data-only watcher that writes paper 主训练OK.md when Stage1-v3 readiness passes."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ok-md", type=Path, default=DEFAULT_OK_MD)
    parser.add_argument("--status-json", type=Path, default=REPO_ROOT / "output" / "material_refine_paper" / "paper_main_release_watch_status.json")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = repo_path(args.config)
    args.ok_md = repo_path(args.ok_md)
    args.status_json = repo_path(args.status_json)
    while True:
        status = {"updated_at_utc": utc_now(), **run_once(args)}
        write_json(args.status_json, status)
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
        if status.get("ok_written") or not args.loop:
            break
        time.sleep(max(30, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
