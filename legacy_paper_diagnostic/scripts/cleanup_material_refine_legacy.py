#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "material_refine_paper"

KEEP_NAMES = {
    "stage1_v3_dataset_latest",
    "stage1_locked346",
    "stage1_v3_round12_balanced_data_adaptation",
    "stage1_v3_round12_balanced_data_adaptation_eval_balanced_test",
    "stage1_v3_round12_balanced_data_adaptation_eval_locked346",
    "stage1_v3_round12_balanced_data_adaptation_eval_ood",
    "stage1_v3_round13_view_render_boundary_guard",
    "stage1_v3_round13_view_render_boundary_guard_eval_balanced_test",
    "stage1_v3_round13_view_render_boundary_guard_eval_locked346",
    "stage1_v3_round13_view_render_boundary_guard_eval_ood",
    "stage1_v3_round14_backbone_topology_render",
    "stage1_v3_round14_backbone_topology_render_eval_balanced_test",
    "stage1_v3_round14_backbone_topology_render_eval_locked346",
    "stage1_v3_round14_backbone_topology_render_eval_ood",
    "stage1_v3_round15_material_evidence_calibration",
    "stage1_v3_round15_material_evidence_calibration_eval_balanced_test",
    "stage1_v3_round15_material_evidence_calibration_eval_locked346",
    "stage1_v3_round15_material_evidence_calibration_eval_ood",
}

ACTIVE_PREFIXES = (
    "dataset_factory",
    "longrun",
    "paper_unlock",
    "scarce",
)

LEGACY_PATTERNS = (
    "smoke",
    "round1_",
    "round2_",
    "round3_",
    "round4_",
    "round5_",
    "round6_",
    "round7_",
    "round8_",
    "round9_",
    "stage1_round",
    "stage1_v1",
    "stage1_v2",
)


def dir_size(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.lstat().st_size
        except FileNotFoundError:
            continue
    return total


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f}{unit}"
        value /= 1024
    return f"{num_bytes}B"


def classify(path: Path) -> str | None:
    name = path.name
    if name in KEEP_NAMES:
        return None
    if name.startswith("_cleanup_trash"):
        return None
    if any(name.startswith(prefix) for prefix in ACTIVE_PREFIXES):
        return None
    if any(pattern in name for pattern in LEGACY_PATTERNS):
        return "legacy_smoke_or_superseded_round"
    return None


def collect_candidates(root: Path) -> list[dict[str, object]]:
    candidates = []
    if not root.exists():
        return candidates
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        reason = classify(child)
        if reason is None:
            continue
        size = dir_size(child)
        candidates.append(
            {
                "path": str(child.relative_to(REPO_ROOT)),
                "reason": reason,
                "size_bytes": size,
                "size": human_size(size),
            },
        )
    return candidates


def write_report(candidates: list[dict[str, object]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "dry_run_candidates",
        "candidate_count": len(candidates),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in candidates),
        "total_size": human_size(sum(int(item["size_bytes"]) for item in candidates)),
        "candidates": candidates,
        "keep_policy": sorted(KEEP_NAMES),
        "note": "Default action is non-destructive. Use --execute to move candidates into a timestamped trash directory.",
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def execute_move(candidates: list[dict[str, object]], trash_root: Path) -> list[dict[str, object]]:
    trash_root.mkdir(parents=True, exist_ok=True)
    moved = []
    for item in candidates:
        src = REPO_ROOT / str(item["path"])
        if not src.exists():
            item = {**item, "action": "missing_skipped"}
            moved.append(item)
            continue
        dst = trash_root / src.name
        suffix = 1
        while dst.exists():
            dst = trash_root / f"{src.name}__{suffix}"
            suffix += 1
        shutil.move(str(src), str(dst))
        item = {**item, "action": "moved", "trash_path": str(dst.relative_to(REPO_ROOT))}
        moved.append(item)
    return moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative cleanup for legacy material refine outputs.")
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "output" / "material_refine_cleanup_candidates.json")
    parser.add_argument("--execute", action="store_true", help="Move candidates into a timestamped trash directory.")
    parser.add_argument("--trash-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root if args.root.is_absolute() else REPO_ROOT / args.root
    report = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    candidates = collect_candidates(root)
    write_report(candidates, report)
    print(f"[cleanup] candidates={len(candidates)} total={human_size(sum(int(x['size_bytes']) for x in candidates))}")
    print(f"[cleanup] report={report}")
    for item in candidates[:80]:
        print(f"[cleanup:candidate] {item['size']} {item['path']} reason={item['reason']}")
    if len(candidates) > 80:
        print(f"[cleanup] ... {len(candidates) - 80} more candidates in report")
    if args.execute:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        trash_root = args.trash_root or (root / f"_cleanup_trash_{stamp}")
        trash_root = trash_root if trash_root.is_absolute() else REPO_ROOT / trash_root
        moved = execute_move(candidates, trash_root)
        moved_report = report.with_name(report.stem + "_executed.json")
        moved_report.write_text(
            json.dumps({"trash_root": str(trash_root), "moved": moved}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[cleanup] moved_report={moved_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
