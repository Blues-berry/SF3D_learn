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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(rows: list[dict]) -> dict:
    baseline_tag_counts = Counter()
    refined_tag_counts = Counter()
    by_generator = defaultdict(lambda: {"count": 0, "baseline_total_mae": 0.0, "refined_total_mae": 0.0, "improvement_total": 0.0})
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
    generator_summary = {}
    for generator_id, item in by_generator.items():
        count = max(int(item["count"]), 1)
        generator_summary[generator_id] = {
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
    }


def build_report(metrics_json: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(metrics_json.read_text())
    rows = sorted(rows, key=lambda row: row["improvement_total"], reverse=True)
    summary = summarize(rows)
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
        ".atlas img { width:100%; border-radius:12px; background:#0b0f15; }",
        ".chips span { display:inline-block; margin:4px 6px 0 0; padding:4px 8px; border-radius:999px; background:#2b3340; }",
        "table { width:100%; border-collapse: collapse; margin-top: 20px; }",
        "th, td { padding: 8px; border-bottom: 1px solid #2d3543; text-align:left; }",
        "</style></head><body><div class='wrap'>",
        "<h1>Material Refiner Before/After Report</h1>",
        "<div class='stats'>",
        f"<div class='stat'><div>Rows</div><h2>{summary['rows']}</h2></div>",
        f"<div class='stat'><div>Baseline Total MAE</div><h2>{summary['baseline_total_mae']:.4f}</h2></div>",
        f"<div class='stat'><div>Refined Total MAE</div><h2>{summary['refined_total_mae']:.4f}</h2></div>",
        f"<div class='stat'><div>Avg Improvement</div><h2>{summary['avg_improvement']:.4f}</h2></div>",
        "</div>",
        "<table><thead><tr><th>Tag</th><th>Baseline</th><th>Refined</th></tr></thead><tbody>",
    ]
    for tag in FAILURE_TAGS:
        lines.append(
            "<tr><td>%s</td><td>%d</td><td>%d</td></tr>"
            % (
                html.escape(tag),
                summary["baseline_tag_counts"].get(tag, 0),
                summary["refined_tag_counts"].get(tag, 0),
            )
        )
    lines.extend(["</tbody></table>", "<h2>By Generator</h2>", "<table><thead><tr><th>Generator</th><th>Count</th><th>Baseline MAE</th><th>Refined MAE</th><th>Improvement</th></tr></thead><tbody>"])
    for generator_id, item in sorted(summary["by_generator"].items()):
        lines.append(
            f"<tr><td>{html.escape(generator_id)}</td><td>{item['count']}</td><td>{item['baseline_total_mae']:.4f}</td><td>{item['refined_total_mae']:.4f}</td><td>{item['improvement_total']:.4f}</td></tr>"
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
        ]:
            path = paths.get(key)
            if path:
                atlas_html.append(
                    f"<img src='{html.escape(os.path.relpath(path, output_dir))}' alt='{html.escape(key)}'>"
                )
        baseline_tags = ", ".join(row.get("baseline_tags", [])) or "none"
        refined_tags = ", ".join(row.get("refined_tags", [])) or "none"
        lines.append(
            "<div class='case'>"
            f"<h3>{html.escape(row['object_id'])} / {html.escape(row['view_name'])}</h3>"
            f"<div>Generator: {html.escape(str(row.get('generator_id', 'unknown')))}</div>"
            f"<div>Baseline total MAE: {row['baseline_total_mae']:.4f}</div>"
            f"<div>Refined total MAE: {row['refined_total_mae']:.4f}</div>"
            f"<div>Improvement: {row['improvement_total']:.4f}</div>"
            f"<div class='chips'><span>baseline: {html.escape(baseline_tags)}</span><span>refined: {html.escape(refined_tags)}</span></div>"
            f"<div class='atlas'>{''.join(atlas_html)}</div>"
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
