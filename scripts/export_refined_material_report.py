from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

FAILURE_TAGS = [
    "metal_nonmetal_confusion",
    "boundary_bleed",
    "local_highlight_misread",
    "over_smoothing",
]


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def metric_delta_class(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric > 0:
        return "good"
    if numeric < 0:
        return "bad"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(rows: list[dict]) -> dict:
    baseline_tag_counts = Counter()
    refined_tag_counts = Counter()
    by_generator = defaultdict(lambda: {"count": 0, "baseline_total_mae": 0.0, "refined_total_mae": 0.0, "improvement_total": 0.0})
    by_license_bucket = defaultdict(lambda: {"count": 0, "baseline_total_mae": 0.0, "refined_total_mae": 0.0, "improvement_total": 0.0})
    by_category_bucket = defaultdict(lambda: {"count": 0, "baseline_total_mae": 0.0, "refined_total_mae": 0.0, "improvement_total": 0.0})
    for row in rows:
        for tag in row.get("baseline_tags", []):
            baseline_tag_counts[tag] += 1
        for tag in row.get("refined_tags", []):
            refined_tag_counts[tag] += 1
        generator_id = str(row.get("generator_id", "unknown"))
        by_generator[generator_id]["count"] += 1
        by_generator[generator_id]["baseline_total_mae"] += float(row["baseline_total_mae"])
        by_generator[generator_id]["refined_total_mae"] += float(row["refined_total_mae"])
        by_generator[generator_id]["improvement_total"] += float(row["improvement_total"])
        license_bucket = str(row.get("license_bucket", "unknown"))
        by_license_bucket[license_bucket]["count"] += 1
        by_license_bucket[license_bucket]["baseline_total_mae"] += float(row["baseline_total_mae"])
        by_license_bucket[license_bucket]["refined_total_mae"] += float(row["refined_total_mae"])
        by_license_bucket[license_bucket]["improvement_total"] += float(row["improvement_total"])
        category_bucket = str(row.get("category_bucket", "unknown"))
        by_category_bucket[category_bucket]["count"] += 1
        by_category_bucket[category_bucket]["baseline_total_mae"] += float(row["baseline_total_mae"])
        by_category_bucket[category_bucket]["refined_total_mae"] += float(row["refined_total_mae"])
        by_category_bucket[category_bucket]["improvement_total"] += float(row["improvement_total"])
    generator_summary = {}
    for generator_id, item in by_generator.items():
        count = max(int(item["count"]), 1)
        generator_summary[generator_id] = {
            "count": int(item["count"]),
            "baseline_total_mae": item["baseline_total_mae"] / count,
            "refined_total_mae": item["refined_total_mae"] / count,
            "improvement_total": item["improvement_total"] / count,
        }
    license_summary = {}
    for license_bucket, item in by_license_bucket.items():
        count = max(int(item["count"]), 1)
        license_summary[license_bucket] = {
            "count": int(item["count"]),
            "baseline_total_mae": item["baseline_total_mae"] / count,
            "refined_total_mae": item["refined_total_mae"] / count,
            "improvement_total": item["improvement_total"] / count,
        }
    category_summary = {}
    for category_bucket, item in by_category_bucket.items():
        count = max(int(item["count"]), 1)
        category_summary[category_bucket] = {
            "count": int(item["count"]),
            "baseline_total_mae": item["baseline_total_mae"] / count,
            "refined_total_mae": item["refined_total_mae"] / count,
            "improvement_total": item["improvement_total"] / count,
        }
    return {
        "rows": len(rows),
        "baseline_total_mae": sum(row["baseline_total_mae"] for row in rows) / max(len(rows), 1),
        "refined_total_mae": sum(row["refined_total_mae"] for row in rows) / max(len(rows), 1),
        "avg_improvement": sum(row["improvement_total"] for row in rows) / max(len(rows), 1),
        "baseline_tag_counts": dict(baseline_tag_counts),
        "refined_tag_counts": dict(refined_tag_counts),
        "by_generator": generator_summary,
        "by_license_bucket": license_summary,
        "by_category_bucket": category_summary,
    }


def build_paper_metric_summary(summary: dict, output_dir: Path) -> Path:
    path = output_dir / "paper_metric_summary.html"
    metrics_main = summary.get("metrics_main", {})
    material_metrics = summary.get("metrics_material_specific", {})
    availability = summary.get("metric_availability", {})
    warnings = summary.get("metric_warnings", [])
    diagnostics = summary.get("diagnostic_reports", {})

    def pair_row(name: str, payload: dict, *, unit: str = "") -> str:
        return (
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{fmt(payload.get('baseline'))}{unit}</td>"
            f"<td>{fmt(payload.get('refined'))}{unit}</td>"
            f"<td class='{metric_delta_class(payload.get('delta'))}'>{fmt(payload.get('delta'))}{unit}</td>"
            f"<td>{'higher' if payload.get('higher_is_better') else 'lower'}</td>"
            f"<td>{payload.get('available_count', 0)}</td>"
            "</tr>"
        )

    uv_total = metrics_main.get("uv_rm_mae", {}).get("total", {})
    view_total = metrics_main.get("view_rm_mae", {}).get("total", {})
    main_rows = [
        pair_row("UV RM MAE", uv_total),
        pair_row("View RM MAE", view_total),
        pair_row("Proxy Render PSNR", metrics_main.get("proxy_render_psnr", {}), unit=" dB"),
        pair_row("Proxy Render SSIM", metrics_main.get("proxy_render_ssim", {})),
        pair_row("Proxy Render LPIPS", metrics_main.get("proxy_render_lpips", {})),
    ]
    special_rows = [
        pair_row("Boundary Bleed Score", material_metrics.get("boundary_bleed_score", {})),
        pair_row("Highlight Localization Error", material_metrics.get("highlight_localization_error", {})),
        pair_row("RM Gradient Preservation", material_metrics.get("rm_gradient_preservation", {})),
    ]
    residual = material_metrics.get("prior_residual_safety", {})
    confidence = material_metrics.get("confidence_calibrated_error", {})
    metal = material_metrics.get("metal_nonmetal_confusion", {})

    availability_rows = []
    for name, item in sorted(availability.items()):
        availability_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{'yes' if item.get('available') else 'no'}</td>"
            f"<td>{item.get('available_count', 0)}</td>"
            f"<td>{html.escape(str(item.get('reason', 'unknown')))}</td>"
            "</tr>"
        )

    warning_items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings) or "<li>none</li>"
    diagnostic_links = []
    for label, target in diagnostics.items():
        if not target:
            continue
        try:
            rel = os.path.relpath(str(target), output_dir)
        except ValueError:
            rel = str(target)
        diagnostic_links.append(f"<li><a href='{html.escape(rel)}'>{html.escape(label)}</a></li>")
    diagnostic_links_html = "".join(diagnostic_links) or "<li>none</li>"

    family_rows = []
    for family, item in sorted(material_metrics.get("material_family_breakdown", {}).items()):
        family_rows.append(
            "<tr>"
            f"<td>{html.escape(str(family))}</td>"
            f"<td>{item.get('count', 0)}</td>"
            f"<td>{fmt(item.get('baseline_total_mae'))}</td>"
            f"<td>{fmt(item.get('refined_total_mae'))}</td>"
            f"<td class='{metric_delta_class(item.get('improvement_total'))}'>{fmt(item.get('improvement_total'))}</td>"
            "</tr>"
        )

    def metal_level_rows(level: str) -> list[str]:
        item = metal.get(level, {})
        rows = []
        for variant in ("baseline", "refined"):
            metrics = item.get(variant, {})
            rows.append(
                "<tr>"
                f"<td>{html.escape(level)}</td>"
                f"<td>{html.escape(variant)}</td>"
                f"<td>{fmt(metrics.get('f1'))}</td>"
                f"<td>{fmt(metrics.get('auroc'))}</td>"
                f"<td>{fmt(metrics.get('balanced_accuracy'))}</td>"
                f"<td>{fmt(metrics.get('confusion_rate'))}</td>"
                "</tr>"
            )
        return rows

    html_text = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Paper Metric Summary</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;background:#101720;color:#eef2f8;margin:0}.wrap{max-width:1500px;margin:0 auto;padding:24px}.card{background:#17212f;border-radius:16px;padding:16px;margin:16px 0}table{width:100%;border-collapse:collapse}th,td{border-bottom:1px solid #334155;padding:8px;text-align:left}.good{color:#86efac}.bad{color:#fca5a5}a{color:#93c5fd}.pill{display:inline-block;background:#263244;border-radius:999px;padding:4px 10px;margin:2px}",
            "</style></head><body><div class='wrap'>",
            "<h1>Paper Metric Summary</h1>",
            f"<p><span class='pill'>mode: {html.escape(str(summary.get('render_metric_mode', 'unknown')))}</span><span class='pill'>rows: {summary.get('rows', 0)}</span><span class='pill'>objects: {summary.get('objects', 0)}</span></p>",
            "<div class='card'><h2>Main Metrics</h2><table><thead><tr><th>Metric</th><th>Baseline</th><th>Refined</th><th>Delta</th><th>Better</th><th>Count</th></tr></thead><tbody>",
            *main_rows,
            "</tbody></table></div>",
            "<div class='card'><h2>Material-Specific Metrics</h2><table><thead><tr><th>Metric</th><th>Baseline</th><th>Refined</th><th>Delta</th><th>Better</th><th>Count</th></tr></thead><tbody>",
            *special_rows,
            "</tbody></table>",
            "<h3>Prior Residual Safety</h3>",
            f"<p>score {fmt(residual.get('safety_score'))}, safe improvement {fmt(residual.get('safe_improvement_rate'))}, unnecessary change {fmt(residual.get('unnecessary_change_rate'))}, regression {fmt(residual.get('regression_rate'))}</p>",
            "<h3>Confidence-Calibrated Error</h3><table><thead><tr><th>Confidence Bin</th><th>Samples</th><th>Baseline</th><th>Refined</th><th>Improvement</th></tr></thead><tbody>",
            *[
                f"<tr><td>{html.escape(name)}</td><td>{item.get('sample_count', 0)}</td><td>{fmt(item.get('baseline_total_mae'))}</td><td>{fmt(item.get('refined_total_mae'))}</td><td class='{metric_delta_class(item.get('improvement_total'))}'>{fmt(item.get('improvement_total'))}</td></tr>"
                for name, item in confidence.items()
            ],
            "</tbody></table></div>",
            "<div class='card'><h2>Metal / Non-Metal Diagnostics</h2><table><thead><tr><th>Level</th><th>Variant</th><th>F1</th><th>AUROC</th><th>Balanced Acc.</th><th>Confusion Rate</th></tr></thead><tbody>",
            *(metal_level_rows("uv_level") + metal_level_rows("view_level") + metal_level_rows("object_level")),
            "</tbody></table></div>",
            "<div class='card'><h2>Material Family Breakdown</h2><table><thead><tr><th>Family</th><th>Count</th><th>Baseline</th><th>Refined</th><th>Improvement</th></tr></thead><tbody>",
            *(family_rows or ["<tr><td colspan='5'>No material family breakdown available.</td></tr>"]),
            "</tbody></table></div>",
            "<div class='card'><h2>Metric Availability</h2><table><thead><tr><th>Metric</th><th>Available</th><th>Count</th><th>Reason</th></tr></thead><tbody>",
            *availability_rows,
            "</tbody></table></div>",
            "<div class='card'><h2>Warnings</h2><ul>",
            warning_items,
            "</ul></div>",
            "<div class='card'><h2>Diagnostic Reports</h2><ul>",
            diagnostic_links_html,
            "</ul></div>",
            "</div></body></html>",
        ]
    )
    path.write_text(html_text, encoding="utf-8")
    return path


def build_report(metrics_json: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(metrics_json.read_text())
    rows = sorted(rows, key=lambda row: row["improvement_total"], reverse=True)
    summary = summarize(rows)
    structured_summary_path = output_dir / "summary.json"
    if structured_summary_path.exists():
        structured_summary = json.loads(structured_summary_path.read_text())
        build_paper_metric_summary(structured_summary, output_dir)
    html_path = output_dir / "report.html"

    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Material Refiner Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; background:#0e1116; color:#eef2f8; margin:0; }",
        ".wrap { max-width: 1600px; margin: 0 auto; padding: 24px; }",
        ".stats { display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; margin-bottom: 24px; }",
        ".stat, .case { background:#171c24; border-radius:18px; padding:16px; }",
        ".casegrid { display:grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap:18px; }",
        ".atlas { display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; margin: 12px 0; }",
        ".renders { display:grid; grid-template-columns: repeat(3, 1fr); gap:8px; margin: 12px 0; }",
        ".atlas img, .renders img { width:100%; border-radius:12px; background:#0b0f15; }",
        ".chips span { display:inline-block; margin:4px 6px 0 0; padding:4px 8px; border-radius:999px; background:#2b3340; }",
        "a { color:#93c5fd; }",
        "table { width:100%; border-collapse: collapse; margin-top: 20px; }",
        "th, td { padding: 8px; border-bottom: 1px solid #2d3543; text-align:left; }",
        "</style></head><body><div class='wrap'>",
        "<h1>Material Refiner Before/After Report</h1>",
        "<p><a href='paper_metric_summary.html'>Paper metric summary</a> | <a href='metric_disagreement_report.html'>Metric disagreement report</a></p>",
        "<div class='stats'>",
        f"<div class='stat'><div>Rows</div><h2>{summary['rows']}</h2></div>",
        f"<div class='stat'><div>Baseline Total MAE</div><h2>{summary['baseline_total_mae']:.4f}</h2></div>",
        f"<div class='stat'><div>Refined Total MAE</div><h2>{summary['refined_total_mae']:.4f}</h2></div>",
        f"<div class='stat'><div>Avg Improvement</div><h2>{summary['avg_improvement']:.4f}</h2></div>",
        "</div>",
        "<table><thead><tr><th>Tag</th><th>Baseline</th><th>Refined</th><th>Reduction</th></tr></thead><tbody>",
    ]
    for tag in FAILURE_TAGS:
        lines.append(
            "<tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td></tr>"
            % (
                html.escape(tag),
                summary["baseline_tag_counts"].get(tag, 0),
                summary["refined_tag_counts"].get(tag, 0),
                summary["baseline_tag_counts"].get(tag, 0) - summary["refined_tag_counts"].get(tag, 0),
            )
        )
    lines.extend(["</tbody></table>", "<h2>By Generator</h2>", "<table><thead><tr><th>Generator</th><th>Count</th><th>Baseline MAE</th><th>Refined MAE</th><th>Improvement</th></tr></thead><tbody>"])
    for generator_id, item in sorted(summary["by_generator"].items()):
        lines.append(
            f"<tr><td>{html.escape(generator_id)}</td><td>{item['count']}</td><td>{item['baseline_total_mae']:.4f}</td><td>{item['refined_total_mae']:.4f}</td><td>{item['improvement_total']:.4f}</td></tr>"
        )
    lines.extend(["</tbody></table>", "<h2>By License Bucket</h2>", "<table><thead><tr><th>License</th><th>Count</th><th>Baseline MAE</th><th>Refined MAE</th><th>Improvement</th></tr></thead><tbody>"])
    for license_bucket, item in sorted(summary["by_license_bucket"].items()):
        lines.append(
            f"<tr><td>{html.escape(license_bucket)}</td><td>{item['count']}</td><td>{item['baseline_total_mae']:.4f}</td><td>{item['refined_total_mae']:.4f}</td><td>{item['improvement_total']:.4f}</td></tr>"
        )
    lines.extend(["</tbody></table>", "<h2>By Category Bucket</h2>", "<table><thead><tr><th>Category</th><th>Count</th><th>Baseline MAE</th><th>Refined MAE</th><th>Improvement</th></tr></thead><tbody>"])
    for category_bucket, item in sorted(summary["by_category_bucket"].items()):
        lines.append(
            f"<tr><td>{html.escape(category_bucket)}</td><td>{item['count']}</td><td>{item['baseline_total_mae']:.4f}</td><td>{item['refined_total_mae']:.4f}</td><td>{item['improvement_total']:.4f}</td></tr>"
        )
    lines.extend(["</tbody></table>", "<h2>Top Improved Cases</h2>", "<div class='casegrid'>"])

    for row in rows[:24]:
        paths = row.get("paths", {})
        atlas_html = []
        for key in [
            "baseline_roughness",
            "baseline_metallic",
            "refined_roughness",
            "refined_metallic",
            "target_roughness",
            "target_metallic",
            "baseline_rm_error",
            "refined_rm_error",
        ]:
            path = paths.get(key)
            if path:
                atlas_html.append(
                    f"<img src='{html.escape(os.path.relpath(path, output_dir))}' alt='{html.escape(key)}'>"
                )
        render_html = []
        for key in [
            "reference_rgb",
            "baseline_proxy_render",
            "refined_proxy_render",
        ]:
            path = paths.get(key)
            if path:
                render_html.append(
                    f"<img src='{html.escape(os.path.relpath(path, output_dir))}' alt='{html.escape(key)}'>"
                )
        baseline_tags = ", ".join(row.get("baseline_tags", [])) or "none"
        refined_tags = ", ".join(row.get("refined_tags", [])) or "none"
        lines.append(
            "<div class='case'>"
            f"<h3>{html.escape(row['object_id'])} / {html.escape(row['view_name'])}</h3>"
            f"<div>Generator: {html.escape(str(row.get('generator_id', 'unknown')))}</div>"
            f"<div>License: {html.escape(str(row.get('license_bucket', 'unknown')))}</div>"
            f"<div>Category: {html.escape(str(row.get('category_bucket', 'unknown')))}</div>"
            f"<div>Variant: {html.escape(str(row.get('eval_variant', 'ours_full')))}</div>"
            f"<div>Baseline total MAE: {row['baseline_total_mae']:.4f}</div>"
            f"<div>Refined total MAE: {row['refined_total_mae']:.4f}</div>"
            f"<div>Improvement: {row['improvement_total']:.4f}</div>"
            f"<div>PSNR b/r: {fmt(row.get('baseline_psnr'))} / {fmt(row.get('refined_psnr'))}</div>"
            f"<div>SSIM b/r: {fmt(row.get('baseline_ssim'))} / {fmt(row.get('refined_ssim'))}</div>"
            f"<div>LPIPS b/r: {fmt(row.get('baseline_lpips'))} / {fmt(row.get('refined_lpips'))}</div>"
            f"<div>Boundary bleed b/r: {fmt(row.get('baseline_boundary_bleed_score'))} / {fmt(row.get('refined_boundary_bleed_score'))}</div>"
            f"<div class='chips'><span>baseline: {html.escape(baseline_tags)}</span><span>refined: {html.escape(refined_tags)}</span></div>"
            f"<div class='atlas'>{''.join(atlas_html)}</div>"
            f"<div class='renders'>{''.join(render_html)}</div>"
            "</div>"
        )
    lines.extend(["</div></div></body></html>"])
    html_path.write_text("\n".join(lines))
    return html_path


def main() -> None:
    args = parse_args()
    path = build_report(args.metrics_json, args.output_dir)
    print(f"Report HTML: {path}")


if __name__ == "__main__":
    main()
