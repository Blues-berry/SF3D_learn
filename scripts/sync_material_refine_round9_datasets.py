from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT_MAIN_MANIFEST = Path(
    "output/material_refine_paper/latest_dataset_check_20260421/"
    "stage1_subset_merged490/paper_stage1_subset_manifest.json"
)
DEFAULT_OUTPUT_BASE = Path("output/material_refine_paper")
DEFAULT_FACTORY_ROOT = Path("output/material_refine_dataset_factory")
DEFAULT_PAPER_LICENSE_BUCKETS = (
    "cc_by_nc_4_0,"
    "cc_by_nc_4_0_pending_reconcile,"
    "custom_tianchi_research_noncommercial_no_redistribution"
)
ACTIVE_CONFIGS = [
    "configs/material_refine_train_paper_stage1_round9_conservative_boundary.yaml",
    "configs/material_refine_train_paper_stage1_round9_conservative_boundary_resume_latest.yaml",
    "configs/material_refine_eval_paper_stage1_round9_conservative_boundary.yaml",
    "configs/material_refine_eval_stage1_v2_diagnostic_20260423.yaml",
    "configs/material_refine_eval_stage1_v2_ood_20260423.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Discover growing material-refine manifests, rebuild Stage1-v2 subsets, "
            "and refresh the Round9 latest readiness index."
        ),
    )
    parser.add_argument("--factory-root", type=Path, default=DEFAULT_FACTORY_ROOT)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--current-main-manifest", type=Path, default=DEFAULT_CURRENT_MAIN_MANIFEST)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument(
        "--latest-index",
        type=Path,
        default=Path("output/material_refine_paper/round9_dataset_latest.json"),
    )
    parser.add_argument(
        "--latest-md",
        type=Path,
        default=Path("output/material_refine_paper/round9_dataset_latest.md"),
    )
    parser.add_argument("--paper-license-buckets", type=str, default=DEFAULT_PAPER_LICENSE_BUCKETS)
    parser.add_argument("--max-diagnostic-records", type=int, default=512)
    parser.add_argument("--max-ood-records", type=int, default=256)
    parser.add_argument("--min-strict-records-to-replace", type=int, default=384)
    parser.add_argument("--min-paper-records-per-new-material", type=int, default=16)
    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def discover_factory_manifests(factory_root: Path) -> list[Path]:
    root = resolve(factory_root)
    if not root.exists():
        return []
    candidates = []
    patterns = [
        "*/canonical_manifest_supervisor_merged.json",
        "*/canonical_manifest_merged.json",
        "*/canonical_manifest.json",
    ]
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    return sorted({path.resolve() for path in candidates if path.is_file()})


def dedupe_paths(paths: list[Path]) -> list[Path]:
    selected: dict[str, Path] = {}
    for path in paths:
        resolved = resolve(path).resolve()
        if resolved.exists():
            selected[str(resolved)] = resolved
    return [selected[key] for key in sorted(selected)]


def run_command(command: list[str], *, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = " ".join(command)
    if dry_run:
        log_path.write_text(f"[dry-run] {text}\n", encoding="utf-8")
        print(f"[dry-run] {text}")
        return 0
    print(f"[run] {text}")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {text}\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def config_findings() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    active = {str(resolve(Path(path)).resolve()) for path in ACTIVE_CONFIGS}
    for path in sorted((REPO_ROOT / "configs").glob("material_refine*.yaml")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        reasons: list[str] = []
        if str(path.resolve()) in active:
            reasons.append("active_round9_entry")
        if "stage1_v2_dataset_20260422" in text:
            reasons.append("stale_stage1_v2_20260422_reference")
        if "diagnostic_smoke" in path.name or "loader-smoke" in text:
            reasons.append("historical_smoke_only")
        if "round1" in path.name or "round2" in path.name or "round3" in path.name:
            reasons.append("historical_round_not_active")
        if "round4" in path.name or "round5" in path.name or "round6" in path.name:
            reasons.append("historical_round_not_active")
        if "round7" in path.name or "round8" in path.name:
            reasons.append("historical_round_not_active")
        if "latest_dataset_check_20260421" in text and str(path.resolve()) not in active:
            reasons.append("frozen_locked346_reference_for_repro_only")
        if "default_smoke_overrides" in text:
            reasons.append("ablation_smoke_matrix_not_active_main")
        if reasons:
            findings.append({"path": display(path), "reasons": reasons})
    return findings


def summarize_latest(
    *,
    args: argparse.Namespace,
    timestamp: str,
    output_root: Path,
    manifests: list[Path],
    build_returncode: int,
    readiness_returncode: int | None,
) -> dict[str, Any]:
    report_path = output_root / "stage1_v2_dataset_sync_report.json"
    readiness_dir = args.output_base / f"round9_dataset_readiness_auto_{timestamp}"
    readiness_path = readiness_dir / "round9_dataset_readiness.json"
    report = load_json(report_path)
    readiness = load_json(readiness_path)
    return {
        "timestamp_utc": timestamp,
        "output_root": display(output_root),
        "current_main_manifest": display(resolve(args.current_main_manifest)),
        "discovered_manifests": [display(path) for path in manifests],
        "stage1_v2_report": display(report_path),
        "stage1_v2_subset_paths": report.get("subset_paths", {}),
        "stage1_v2_subset_summaries": report.get("subset_summaries", {}),
        "stage1_v2_recommendation": report.get("recommendation"),
        "stage1_v2_blockers": report.get("blockers", []),
        "readiness_json": display(readiness_path),
        "readiness_recommendation": readiness.get("recommendation"),
        "train_manifest_for_round9": (
            readiness.get("dataset", {}).get("train_manifest_for_round9")
            if readiness
            else None
        ),
        "diagnostic_manifest_for_round9": (
            readiness.get("dataset", {}).get("diagnostic_manifest_for_round9")
            if readiness
            else report.get("subset_paths", {}).get("diverse_diagnostic")
        ),
        "ood_manifest_for_round9": (
            readiness.get("dataset", {}).get("ood_manifest_for_round9")
            if readiness
            else report.get("subset_paths", {}).get("ood_eval")
        ),
        "build_returncode": build_returncode,
        "readiness_returncode": readiness_returncode,
        "active_configs": ACTIVE_CONFIGS,
        "config_findings": config_findings(),
        "not_adapted_to_incremental_data": [
            "历史 smoke 和早期 round 配置仍保留用于复现，但默认启动应只看 active_configs/latest index。",
            "Round9 paper-stage 主训练仍锁定 current_main；只有 strict subset 通过替换门禁后才切换。",
            "二次验证后的新增训练样本必须使用规范字段：target_quality_tier in {paper_pseudo,paper_strong} 且 supervision_role=paper_main。",
            "Diagnostic/OOD 子集可以包含 smoke_only 或 auxiliary_upgrade_queue，但只能 eval-only，不能混进 paper-stage train。",
            "新增来源的 license_bucket 必须先归一化到 paper license allowlist，否则 strict paper selection 会拦下。",
        ],
    }


def write_latest_md(path: Path, payload: dict[str, Any]) -> None:
    summaries = payload.get("stage1_v2_subset_summaries") or {}
    strict = summaries.get("strict_paper") or {}
    diagnostic = summaries.get("diverse_diagnostic") or {}
    ood = summaries.get("ood_eval") or {}
    lines = [
        "# Material Refine Round9 Dataset Latest",
        "",
        f"- timestamp_utc: `{payload['timestamp_utc']}`",
        f"- output_root: `{payload['output_root']}`",
        f"- recommendation: `{payload.get('readiness_recommendation') or payload.get('stage1_v2_recommendation')}`",
        f"- current_main_manifest: `{payload['current_main_manifest']}`",
        f"- train_manifest_for_round9: `{payload.get('train_manifest_for_round9')}`",
        f"- diagnostic_manifest_for_round9: `{payload.get('diagnostic_manifest_for_round9')}`",
        f"- ood_manifest_for_round9: `{payload.get('ood_manifest_for_round9')}`",
        "",
        "## 当前可用数据",
        "",
        f"- strict_paper: `{strict.get('records')}` records, material `{json.dumps(strict.get('material_family', {}), ensure_ascii=False)}`",
        f"- diverse_diagnostic: `{diagnostic.get('records')}` records, material `{json.dumps(diagnostic.get('material_family', {}), ensure_ascii=False)}`",
        f"- ood_eval: `{ood.get('records')}` records, material `{json.dumps(ood.get('material_family', {}), ensure_ascii=False)}`",
        "",
        "## 增量数据不适配点",
        "",
    ]
    for item in payload.get("not_adapted_to_incremental_data") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## 活跃配置", ""])
    for item in payload.get("active_configs") or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## 历史/归档候选配置", ""])
    for finding in payload.get("config_findings") or []:
        reasons = ",".join(finding.get("reasons") or [])
        lines.append(f"- `{finding['path']}`: `{reasons}`")
    lines.extend(["", "## 数据源 manifests", ""])
    for manifest in payload.get("discovered_manifests") or []:
        lines.append(f"- `{manifest}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_base = resolve(args.output_base)
    timestamp = utc_stamp()
    output_name = args.output_name or f"stage1_v2_dataset_auto_{timestamp}"
    output_root = args.output_base / output_name
    logs_dir = output_root / "logs"
    discovered = discover_factory_manifests(args.factory_root)
    manifests = dedupe_paths([resolve(args.current_main_manifest), *discovered, *args.manifest])
    if not manifests:
        raise SystemExit("no_manifests_discovered")

    build_command = [
        sys.executable,
        "scripts/build_material_refine_stage1_v2_subsets.py",
        "--output-root",
        display(output_root),
        "--paper-license-buckets",
        str(args.paper_license_buckets),
        "--max-diagnostic-records",
        str(args.max_diagnostic_records),
        "--max-ood-records",
        str(args.max_ood_records),
    ]
    for manifest in manifests:
        build_command.extend(["--manifest", display(manifest)])
    build_returncode = run_command(
        build_command,
        log_path=logs_dir / "build_stage1_v2.log",
        dry_run=bool(args.dry_run),
    )
    if build_returncode != 0:
        raise SystemExit(build_returncode)

    readiness_returncode: int | None = None
    if not args.skip_readiness:
        readiness_command = [
            sys.executable,
            "scripts/analyze_material_refine_round9_readiness.py",
            "--stage1-v2-report",
            display(output_root / "stage1_v2_dataset_sync_report.json"),
            "--current-main-manifest",
            display(resolve(args.current_main_manifest)),
            "--output-dir",
            display(args.output_base / f"round9_dataset_readiness_auto_{timestamp}"),
            "--min-strict-records-to-replace",
            str(args.min_strict_records_to_replace),
            "--min-paper-records-per-new-material",
            str(args.min_paper_records_per_new_material),
        ]
        readiness_returncode = run_command(
            readiness_command,
            log_path=logs_dir / "analyze_round9_readiness.log",
            dry_run=bool(args.dry_run),
        )
        if readiness_returncode != 0:
            raise SystemExit(readiness_returncode)

    payload = summarize_latest(
        args=args,
        timestamp=timestamp,
        output_root=output_root,
        manifests=manifests,
        build_returncode=build_returncode,
        readiness_returncode=readiness_returncode,
    )
    latest_index = resolve(args.latest_index)
    latest_md = resolve(args.latest_md)
    latest_index.parent.mkdir(parents=True, exist_ok=True)
    latest_index.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_latest_md(latest_md, payload)
    print(
        json.dumps(
            {
                "latest_index": display(latest_index),
                "latest_md": display(latest_md),
                "recommendation": payload.get("readiness_recommendation")
                or payload.get("stage1_v2_recommendation"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
