from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CURRENT_LOCKED_RECORDS = 346


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize current dataset usability and Round9 ablation readiness.",
    )
    parser.add_argument(
        "--stage1-v2-report",
        type=Path,
        required=True,
        help="stage1_v2_dataset_sync_report.json produced by build_material_refine_stage1_v2_subsets.py.",
    )
    parser.add_argument(
        "--current-main-manifest",
        type=Path,
        default=Path(
            "output/material_refine_paper/latest_dataset_check_20260421/"
            "stage1_subset_merged490/paper_stage1_subset_manifest.json"
        ),
    )
    parser.add_argument(
        "--boundary-ablation-root",
        type=Path,
        default=Path("output/material_refine_paper/round9_boundary_ablation"),
    )
    parser.add_argument(
        "--rv2-ablation-root",
        type=Path,
        default=Path("output/material_refine_paper/round9_rv2_component_ablation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/material_refine_paper/round9_dataset_readiness"),
    )
    parser.add_argument("--min-strict-records-to-replace", type=int, default=384)
    parser.add_argument("--min-paper-records-per-new-material", type=int, default=16)
    parser.add_argument("--uv-tie-epsilon", type=float, default=2.0e-5)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def read_manifest_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    payload = load_json(path)
    records = payload.get("records") or []
    summary = payload.get("summary") or {}
    if summary:
        return {"exists": True, "path": str(path), **summary}
    counts: dict[str, dict[str, int]] = {}
    for key in ("source_name", "generator_id", "material_family", "license_bucket", "target_quality_tier"):
        bucket: dict[str, int] = {}
        for record in records:
            value = str(record.get(key, record.get("metadata", {}).get(key, "unknown")))
            bucket[value] = bucket.get(value, 0) + 1
        counts[key] = dict(sorted(bucket.items(), key=lambda item: (-item[1], item[0])))
    return {"exists": True, "path": str(path), "records": len(records), **counts}


def collect_ablation_metrics(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        validation_dir = variant_dir / "validation"
        if not validation_dir.exists():
            continue
        best_uv = None
        best_label = None
        last_payload: dict[str, Any] | None = None
        validation_count = 0
        for validation_path in sorted(validation_dir.glob("*.json")):
            payload = load_json(validation_path)
            uv_total = safe_float((payload.get("uv_mae") or {}).get("total"))
            if uv_total is None:
                continue
            validation_count += 1
            if best_uv is None or uv_total < best_uv:
                best_uv = uv_total
                best_label = validation_path.stem
            last_payload = payload
        if best_uv is None:
            continue
        render_proxy = (last_payload or {}).get("render_proxy_validation") or {}
        residual = (last_payload or {}).get("residual_gate_diagnostics") or {}
        loss = (last_payload or {}).get("loss") or {}
        rows.append(
            {
                "variant": variant_dir.name,
                "output_dir": str(variant_dir),
                "validation_count": validation_count,
                "best_uv_total_mae": best_uv,
                "best_label": best_label,
                "last_uv_total_mae": safe_float(((last_payload or {}).get("uv_mae") or {}).get("total")),
                "last_view_rm_mae_delta": safe_float(render_proxy.get("view_rm_mae_delta")),
                "last_proxy_rm_psnr_delta": safe_float(render_proxy.get("proxy_rm_psnr_delta")),
                "last_residual_regression_rate": safe_float(residual.get("regression_rate")),
                "last_residual_changed_pixel_rate": safe_float(residual.get("changed_pixel_rate")),
                "last_boundary_loss": safe_float(loss.get("boundary_bleed")),
                "last_material_context_loss": safe_float(loss.get("material_context")),
            }
        )
    return rows


def choose_boundary_variant(rows: list[dict[str, Any]], uv_tie_epsilon: float) -> dict[str, Any] | None:
    if not rows:
        return None
    best_uv = min(row["best_uv_total_mae"] for row in rows)
    candidates = [
        row
        for row in rows
        if row["best_uv_total_mae"] <= best_uv + uv_tie_epsilon
    ]
    candidates.sort(
        key=lambda row: (
            row.get("last_residual_regression_rate")
            if row.get("last_residual_regression_rate") is not None
            else 1.0,
            -(row.get("last_view_rm_mae_delta") or 0.0),
            row["best_uv_total_mae"],
        )
    )
    return candidates[0]


def choose_rv2_variant(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    rows = sorted(
        rows,
        key=lambda row: (
            row["best_uv_total_mae"],
            row.get("last_residual_regression_rate")
            if row.get("last_residual_regression_rate") is not None
            else 1.0,
        ),
    )
    return rows[0]


def strict_replacement_decision(
    strict_summary: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = int(strict_summary.get("records", 0))
    material_counts = strict_summary.get("material_family") or {}
    non_glossy_counts = {
        key: int(value)
        for key, value in material_counts.items()
        if key not in {"glossy_non_metal", "unknown"} and int(value) > 0
    }
    enough_total = records >= int(args.min_strict_records_to_replace)
    enough_new_material = any(
        count >= int(args.min_paper_records_per_new_material)
        for count in non_glossy_counts.values()
    )
    replace = bool(enough_total and enough_new_material)
    blockers = []
    if not enough_total:
        blockers.append(
            f"strict_records={records}<min_replace={int(args.min_strict_records_to_replace)}"
        )
    if not enough_new_material:
        blockers.append(
            "non_glossy_paper_material_count_below_"
            f"{int(args.min_paper_records_per_new_material)}:{non_glossy_counts or {}}"
        )
    return {
        "replace_current_main": replace,
        "records": records,
        "material_family": material_counts,
        "non_glossy_material_family": non_glossy_counts,
        "blockers": blockers,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    dataset = payload["dataset"]
    method = payload["method"]
    lines = [
        "# Round9 Dataset Readiness",
        "",
        f"- stage1_v2_report: `{payload['stage1_v2_report']}`",
        f"- current_main_manifest: `{dataset['current_main']['path']}`",
        f"- recommendation: `{payload['recommendation']}`",
        "",
        "## Dataset",
        "",
        f"- current_main_records: `{dataset['current_main'].get('records')}`",
        f"- strict_paper_records: `{dataset['strict_paper'].get('records')}`",
        f"- diagnostic_records: `{dataset['diverse_diagnostic'].get('records')}`",
        f"- ood_records: `{dataset['ood_eval'].get('records')}`",
        f"- strict_material_family: `{json.dumps(dataset['strict_paper'].get('material_family', {}), ensure_ascii=False)}`",
        f"- diagnostic_material_family: `{json.dumps(dataset['diverse_diagnostic'].get('material_family', {}), ensure_ascii=False)}`",
        f"- ood_material_family: `{json.dumps(dataset['ood_eval'].get('material_family', {}), ensure_ascii=False)}`",
        "",
        "## Strict Replacement Gate",
        "",
        f"- replace_current_main: `{dataset['strict_replacement']['replace_current_main']}`",
    ]
    for blocker in dataset["strict_replacement"].get("blockers") or ["none"]:
        lines.append(f"- blocker: `{blocker}`")
    lines.extend(
        [
            "",
            "## Method Ablation",
            "",
            f"- boundary_recommendation: `{(method.get('boundary_recommendation') or {}).get('variant')}`",
            f"- rv2_recommendation: `{(method.get('rv2_recommendation') or {}).get('variant')}`",
            f"- method_decision: `{method.get('decision')}`",
            "",
            "## Round9 Adaptation",
            "",
        ]
    )
    for item in payload["round9_adaptation"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = load_json(args.stage1_v2_report)
    subset_summaries = report.get("subset_summaries") or {}
    current_main = read_manifest_summary(args.current_main_manifest)
    strict_summary = subset_summaries.get("strict_paper") or {}
    diagnostic_summary = subset_summaries.get("diverse_diagnostic") or {}
    ood_summary = subset_summaries.get("ood_eval") or {}

    boundary_rows = collect_ablation_metrics(args.boundary_ablation_root)
    rv2_rows = collect_ablation_metrics(args.rv2_ablation_root)
    boundary_choice = choose_boundary_variant(boundary_rows, float(args.uv_tie_epsilon))
    rv2_choice = choose_rv2_variant(rv2_rows)
    strict_decision = strict_replacement_decision(strict_summary, args=args)

    use_current_main = not strict_decision["replace_current_main"]
    train_manifest = (
        str(args.current_main_manifest)
        if use_current_main
        else report.get("subset_paths", {}).get("strict_paper")
    )
    recommendation = (
        "KEEP_CURRENT_MAIN_AND_USE_STAGE1_V2_FOR_DIAGNOSTIC"
        if use_current_main
        else "STAGE1_V2_STRICT_READY_FOR_ROUND9_SMOKE"
    )
    method_decision = "boundary_loss_line_is_currently_safer_than_rv2_full"
    if rv2_choice and boundary_choice:
        rv2_uv = rv2_choice["best_uv_total_mae"]
        boundary_uv = boundary_choice["best_uv_total_mae"]
        if rv2_uv <= boundary_uv + 0.002:
            method_decision = "rv2_candidate_close_enough_for_additional_smoke_not_full_training"

    payload = {
        "stage1_v2_report": str(args.stage1_v2_report),
        "recommendation": recommendation,
        "dataset": {
            "current_main": current_main,
            "strict_paper": strict_summary,
            "diverse_diagnostic": diagnostic_summary,
            "ood_eval": ood_summary,
            "strict_replacement": strict_decision,
            "train_manifest_for_round9": train_manifest,
            "diagnostic_manifest_for_round9": report.get("subset_paths", {}).get("diverse_diagnostic"),
            "ood_manifest_for_round9": report.get("subset_paths", {}).get("ood_eval"),
        },
        "method": {
            "boundary_ablation_rows": boundary_rows,
            "rv2_ablation_rows": rv2_rows,
            "boundary_recommendation": boundary_choice,
            "rv2_recommendation": rv2_choice,
            "decision": method_decision,
        },
        "round9_adaptation": [
            "Use paper-stage training only from train_manifest_for_round9.",
            "Use diverse_diagnostic and ood manifests as eval-only gates; do not mix smoke_only into training.",
            "Prefer the boundary-loss line for full Round9 until R-v2 closes the UV gap.",
            "Use R-v2 no-render or conservative-low-delta variants only for short smoke and ablation.",
            "Keep validation-progress milestones at 40 and render-proxy snapshots every 10 milestones.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "round9_dataset_readiness.json"
    md_path = args.output_dir / "round9_dataset_readiness.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "recommendation": recommendation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
