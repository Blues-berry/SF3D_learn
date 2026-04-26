#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "objects",
    "rows",
    "baseline_total_mae",
    "refined_total_mae",
    "avg_improvement_total",
    "metal_nonmetal.baseline_confusion_rate",
    "metal_nonmetal.refined_confusion_rate",
    "metrics_main.proxy_render_psnr.delta",
    "metrics_main.proxy_render_ssim.delta",
    "metrics_main.proxy_render_lpips.delta",
    "metrics_material_specific.boundary_bleed_score.delta",
    "metrics_material_specific.prior_residual_safety.changed_pixel_rate",
    "metrics_material_specific.prior_residual_safety.safe_improvement_rate",
    "metrics_material_specific.prior_residual_safety.unnecessary_change_rate",
    "metrics_material_specific.prior_residual_safety.regression_rate",
    "metrics_diagnostics.metric_disagreement.has_disagreement",
]


def nested_get(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def load_summary(label_and_path: str) -> tuple[str, dict[str, Any]]:
    if "=" in label_and_path:
        label, path_text = label_and_path.split("=", 1)
    else:
        path = Path(label_and_path)
        label, path_text = path.parent.name, label_and_path
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return label, json.load(f)


def build_table(rows: list[dict[str, Any]], metrics: list[str]) -> str:
    header = ["metric", *[row["label"] for row in rows]]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for metric in metrics:
        values = [format_value(row["metrics"].get(metric)) for row in rows]
        lines.append("| " + " | ".join([metric, *values]) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare material refine eval summary.json files.")
    parser.add_argument("summaries", nargs="+", help="LABEL=path/to/summary.json or plain path.")
    parser.add_argument("--metric", action="append", default=None, help="Metric path to include. May repeat.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = args.metric or DEFAULT_METRICS
    rows = []
    for item in args.summaries:
        label, payload = load_summary(item)
        rows.append(
            {
                "label": label,
                "path": item.split("=", 1)[-1],
                "metrics": {metric: nested_get(payload, metric) for metric in metrics},
            },
        )
    result = {"metrics": metrics, "runs": rows}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    table = build_table(rows, metrics)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
