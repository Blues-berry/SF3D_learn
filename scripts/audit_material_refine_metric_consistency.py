from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit consistency between UV aggregate metrics and per-row/view material-refine metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def rate(values: list[bool]) -> float | None:
    return sum(1 for value in values if value) / len(values) if values else None


def identity_bucket(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "unknown"
    if number <= 0.30:
        return "<=0.30"
    if number <= 0.70:
        return "0.30-0.70"
    return ">0.70"


def row_split(row: dict[str, Any]) -> str:
    return str(row.get("paper_split") or row.get("split") or row.get("default_split") or "unknown")


def debug_row(row: dict[str, Any]) -> dict[str, Any]:
    input_prior = finite_float(row.get("input_prior_total_mae"))
    baseline = finite_float(row.get("baseline_total_mae"))
    refined = finite_float(row.get("refined_total_mae"))
    if input_prior is None:
        input_prior = baseline
    computed_gain = None if input_prior is None or refined is None else input_prior - refined
    view_total_mae = refined if str(row.get("view_name", "")) != "uv_space" else None
    uv_total_mae = refined if str(row.get("view_name", "")) == "uv_space" else None
    return {
        "object_id": row.get("object_id"),
        "split": row_split(row),
        "prior_mode": row.get("prior_mode"),
        "prior_source_type": row.get("prior_source_type"),
        "has_material_prior": row.get("has_material_prior"),
        "target_source_type": row.get("target_source_type"),
        "target_prior_identity": row.get("target_prior_identity"),
        "input_prior_total_mae": input_prior,
        "baseline_total_mae": baseline,
        "refined_total_mae": refined,
        "improvement_total": finite_float(row.get("improvement_total")),
        "computed_gain_debug": computed_gain,
        "row_is_improved_debug": bool(refined < input_prior) if input_prior is not None and refined is not None else None,
        "row_is_regressed_debug": bool(refined > input_prior) if input_prior is not None and refined is not None else None,
        "view_total_mae": view_total_mae,
        "uv_total_mae": uv_total_mae,
        "view_name": row.get("view_name"),
        "material_family": row.get("material_family"),
        "source_name": row.get("source_name"),
        "prior_label": row.get("prior_label"),
        "target_prior_identity_bucket": identity_bucket(row.get("target_prior_identity")),
    }


def summarize_group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key, "unknown"))].append(row)
    output: dict[str, dict[str, Any]] = {}
    for bucket, items in sorted(buckets.items()):
        gains = [finite_float(item.get("computed_gain_debug")) for item in items]
        improved = [bool(item.get("row_is_improved_debug")) for item in items if item.get("row_is_improved_debug") is not None]
        regressed = [bool(item.get("row_is_regressed_debug")) for item in items if item.get("row_is_regressed_debug") is not None]
        output[bucket] = {
            "count": len(items),
            "mean_gain": mean(gains),
            "improvement_rate": rate(improved),
            "regression_rate": rate(regressed),
        }
    return output


def format_table(grouped: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| group | count | mean_gain | improvement_rate | regression_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, item in grouped.items():
        lines.append(
            f"| `{key}` | {item['count']} | {item['mean_gain']} | "
            f"{item['improvement_rate']} | {item['regression_rate']} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics_json or (args.eval_dir / "metrics.json")
    summary_path = args.summary_json or (args.eval_dir / "summary.json")
    output_csv = args.output_csv or (args.eval_dir / "metric_consistency_debug.csv")
    output_md = args.output_md or (args.eval_dir / "metric_consistency_audit.md")

    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    debug_rows = [debug_row(row) for row in rows]

    fieldnames = [
        "object_id",
        "split",
        "prior_mode",
        "prior_source_type",
        "has_material_prior",
        "target_source_type",
        "target_prior_identity",
        "input_prior_total_mae",
        "baseline_total_mae",
        "refined_total_mae",
        "improvement_total",
        "computed_gain_debug",
        "row_is_improved_debug",
        "row_is_regressed_debug",
        "view_total_mae",
        "uv_total_mae",
        "view_name",
        "material_family",
        "source_name",
        "prior_label",
        "target_prior_identity_bucket",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(debug_rows)

    input_mean = mean([finite_float(row.get("input_prior_total_mae")) for row in debug_rows])
    baseline_mean = mean([finite_float(row.get("baseline_total_mae")) for row in debug_rows])
    refined_mean = mean([finite_float(row.get("refined_total_mae")) for row in debug_rows])
    gain_mean = mean([finite_float(row.get("computed_gain_debug")) for row in debug_rows])
    improved_rate = rate([bool(row.get("row_is_improved_debug")) for row in debug_rows if row.get("row_is_improved_debug") is not None])
    regressed_rate = rate([bool(row.get("row_is_regressed_debug")) for row in debug_rows if row.get("row_is_regressed_debug") is not None])
    summary_improvement = finite_float(summary.get("improvement_rate"))
    summary_regression = finite_float(summary.get("regression_rate"))
    summary_rate_matches_debug = (
        summary_improvement is not None
        and summary_regression is not None
        and abs(summary_improvement - (improved_rate or 0.0)) < 1.0e-9
        and abs(summary_regression - (regressed_rate or 0.0)) < 1.0e-9
    )
    uv_summary = (summary.get("metrics_main") or {}).get("uv_rm_mae", {}).get("total", {})
    view_summary = (summary.get("metrics_main") or {}).get("view_rm_mae", {}).get("total", {})
    top_level_uses_uv_mean = (
        finite_float(summary.get("input_prior_total_mae")) == finite_float(uv_summary.get("baseline"))
        and finite_float(summary.get("refined_total_mae")) == finite_float(uv_summary.get("refined"))
    )
    top_level_rate_uses_view_rows = summary_rate_matches_debug
    metric_consistency_pass = bool(top_level_uses_uv_mean and not top_level_rate_uses_view_rows)
    view_row_regression_dominates = bool(
        improved_rate is not None
        and regressed_rate is not None
        and regressed_rate > improved_rate
    )
    phase4_rehearsal_gate_pass = bool(metric_consistency_pass and not view_row_regression_dominates)

    by_prior_mode = summarize_group(debug_rows, "prior_mode")
    by_prior_source_type = summarize_group(debug_rows, "prior_source_type")
    by_prior_label = summarize_group(debug_rows, "prior_label")
    by_material_family = summarize_group(debug_rows, "material_family")
    by_source_name = summarize_group(debug_rows, "source_name")
    by_identity_bucket = summarize_group(debug_rows, "target_prior_identity_bucket")

    audit_payload = {
        "rows": len(debug_rows),
        "input_prior_total_mae_mean": input_mean,
        "baseline_total_mae_mean": baseline_mean,
        "refined_total_mae_mean": refined_mean,
        "computed_gain_debug_mean": gain_mean,
        "computed_improvement_rate_debug": improved_rate,
        "computed_regression_rate_debug": regressed_rate,
        "summary_improvement_rate": summary_improvement,
        "summary_regression_rate": summary_regression,
        "summary_rate_matches_debug": summary_rate_matches_debug,
        "top_level_uses_uv_mean": top_level_uses_uv_mean,
        "top_level_rate_uses_view_rows": top_level_rate_uses_view_rows,
        "metric_consistency_pass": metric_consistency_pass,
        "view_row_regression_dominates": view_row_regression_dominates,
        "phase4_rehearsal_gate_pass": phase4_rehearsal_gate_pass,
        "uv_summary": uv_summary,
        "view_summary": view_summary,
        "groups": {
            "prior_mode": by_prior_mode,
            "prior_source_type": by_prior_source_type,
            "prior_label": by_prior_label,
            "material_family": by_material_family,
            "source_name": by_source_name,
            "target_prior_identity_bucket": by_identity_bucket,
        },
    }
    (args.eval_dir / "metric_consistency_audit.json").write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Metric Consistency Audit",
        "",
        f"- metrics_json: `{metrics_path.resolve()}`",
        f"- summary_json: `{summary_path.resolve()}`",
        f"- debug_csv: `{output_csv.resolve()}`",
        f"- metric_consistency_pass: `{metric_consistency_pass}`",
        "",
        "## Summary",
        "",
        f"1. rows 总数: `{len(debug_rows)}`",
        f"2. input_prior_total_mae mean: `{input_mean}`",
        f"3. baseline_total_mae mean: `{baseline_mean}`",
        f"4. refined_total_mae mean: `{refined_mean}`",
        f"5. computed_gain_debug mean: `{gain_mean}`",
        f"6. computed_improvement_rate_debug: `{improved_rate}`",
        f"7. computed_regression_rate_debug: `{regressed_rate}`",
        f"8. summary improvement_rate / regression_rate: `{summary_improvement}` / `{summary_regression}`",
        f"9. summary rates match per-row debug: `{summary_rate_matches_debug}`",
        f"10. top-level aggregate uses UV summary: `{top_level_uses_uv_mean}`",
        f"11. top-level rates use view rows: `{top_level_rate_uses_view_rows}`",
        f"12. view-row regression dominates: `{view_row_regression_dominates}`",
        f"13. phase4 rehearsal gate pass: `{phase4_rehearsal_gate_pass}`",
        "",
        "## Diagnosis",
        "",
        "- `improvement_total` rows are computed as `input_prior_total_mae - refined_total_mae`.",
        "- Current top-level `input_prior_total_mae/refined_total_mae/gain_total` are UV-level aggregate values.",
        (
            "- Current top-level `improvement_rate/regression_rate` are aligned with UV-object rates, "
            "not per-row/view debug rates."
            if metric_consistency_pass
            else "- Current top-level `improvement_rate/regression_rate` are still inconsistent with the top-level UV aggregate."
        ),
        (
            "- UV aggregate improves, but view/row-level metrics regress; keep the result diagnostic-only and do not enter Phase4 rehearsal."
            if view_row_regression_dominates
            else "- View/row-level direction does not dominate negatively under the current audit."
        ),
        "",
        "## Code Locations",
        "",
        "- `scripts/eval_material_refiner.py`: row `improvement_total` for UV fallback rows around row append block near `input_prior_total_uv`.",
        "- `scripts/eval_material_refiner.py`: row `improvement_total` for view rows around row append block near `input_prior_total_view`.",
        "- `scripts/eval_material_refiner.py`: top-level UV aggregate around `uv_baseline_total_mae`, `uv_refined_total_mae`, `uv_improvement_total`.",
        "- `scripts/eval_material_refiner.py`: current top-level rates around `summary_payload['improvement_rate']` and `summary_payload['regression_rate']` use `uv_direction_rates`; the pre-fix bug used view `rows`.",
        "",
        "## Group: prior_mode",
        "",
        *format_table(by_prior_mode),
        "",
        "## Group: prior_source_type",
        "",
        *format_table(by_prior_source_type),
        "",
        "## Group: with_prior vs without_prior",
        "",
        *format_table(by_prior_label),
        "",
        "## Group: material_family",
        "",
        *format_table(by_material_family),
        "",
        "## Group: source_name",
        "",
        *format_table(by_source_name),
        "",
        "## Group: target_prior_identity bucket",
        "",
        *format_table(by_identity_bucket),
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
