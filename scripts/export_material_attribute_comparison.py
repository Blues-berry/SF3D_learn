from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare input prior material attributes against refined RM atlas outputs.",
    )
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_gray(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    image_path = Path(path)
    if not image_path.exists():
        return None
    return np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0


def masked_values(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return values.reshape(-1)
    selected = values[mask]
    return selected if selected.size else values.reshape(-1)


def map_stats(values: np.ndarray, mask: np.ndarray | None) -> dict[str, float]:
    selected = masked_values(values, mask)
    return {
        "mean": float(selected.mean()),
        "std": float(selected.std()),
        "p10": float(np.quantile(selected, 0.10)),
        "p50": float(np.quantile(selected, 0.50)),
        "p90": float(np.quantile(selected, 0.90)),
    }


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "count": float(len(rows)),
        "baseline_roughness_mean": mean_or_zero([row["baseline_roughness_mean"] for row in rows]),
        "refined_roughness_mean": mean_or_zero([row["refined_roughness_mean"] for row in rows]),
        "roughness_delta_mean": mean_or_zero([row["roughness_delta_mean"] for row in rows]),
        "roughness_abs_delta_mean": mean_or_zero([row["roughness_abs_delta_mean"] for row in rows]),
        "baseline_metallic_mean": mean_or_zero([row["baseline_metallic_mean"] for row in rows]),
        "refined_metallic_mean": mean_or_zero([row["refined_metallic_mean"] for row in rows]),
        "metallic_delta_mean": mean_or_zero([row["metallic_delta_mean"] for row in rows]),
        "metallic_abs_delta_mean": mean_or_zero([row["metallic_abs_delta_mean"] for row in rows]),
    }


def build_attribute_rows(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for metric_row in metrics_rows:
        object_id = str(metric_row.get("object_id", "unknown"))
        if object_id in seen:
            continue
        seen.add(object_id)
        paths = metric_row.get("paths", {})
        baseline_roughness = load_gray(paths.get("baseline_roughness"))
        baseline_metallic = load_gray(paths.get("baseline_metallic"))
        refined_roughness = load_gray(paths.get("refined_roughness"))
        refined_metallic = load_gray(paths.get("refined_metallic"))
        if any(item is None for item in [baseline_roughness, baseline_metallic, refined_roughness, refined_metallic]):
            continue

        confidence = load_gray(paths.get("confidence"))
        mask = confidence > 0.01 if confidence is not None else None
        rough_delta = refined_roughness - baseline_roughness
        metal_delta = refined_metallic - baseline_metallic
        row = {
            "object_id": object_id,
            "generator_id": metric_row.get("generator_id", "unknown"),
            "source_name": metric_row.get("source_name", "unknown"),
            "prior_label": metric_row.get("prior_label", "unknown"),
            "supervision_tier": metric_row.get("supervision_tier", "unknown"),
            "baseline_roughness": map_stats(baseline_roughness, mask),
            "refined_roughness": map_stats(refined_roughness, mask),
            "baseline_metallic": map_stats(baseline_metallic, mask),
            "refined_metallic": map_stats(refined_metallic, mask),
            "roughness_delta": map_stats(rough_delta, mask),
            "roughness_abs_delta": map_stats(np.abs(rough_delta), mask),
            "metallic_delta": map_stats(metal_delta, mask),
            "metallic_abs_delta": map_stats(np.abs(metal_delta), mask),
        }
        row.update(
            {
                "baseline_roughness_mean": row["baseline_roughness"]["mean"],
                "refined_roughness_mean": row["refined_roughness"]["mean"],
                "roughness_delta_mean": row["roughness_delta"]["mean"],
                "roughness_abs_delta_mean": row["roughness_abs_delta"]["mean"],
                "baseline_metallic_mean": row["baseline_metallic"]["mean"],
                "refined_metallic_mean": row["refined_metallic"]["mean"],
                "metallic_delta_mean": row["metallic_delta"]["mean"],
                "metallic_abs_delta_mean": row["metallic_abs_delta"]["mean"],
            }
        )
        rows.append(row)
    return rows


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return {group: summarize_group(items) for group, items in sorted(groups.items())}


def save_plot(rows: list[dict[str, Any]], summary: dict[str, Any], output_path: Path) -> None:
    source_summary = summary["by_source_name"]
    labels = list(source_summary.keys()) or ["all"]
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Input Prior vs Material Refiner Attribute Comparison", fontsize=15)

    axes[0, 0].bar(x - width / 2, [source_summary[label]["baseline_roughness_mean"] for label in labels], width, label="Input prior")
    axes[0, 0].bar(x + width / 2, [source_summary[label]["refined_roughness_mean"] for label in labels], width, label="Refined")
    axes[0, 0].set_title("Roughness Mean")
    axes[0, 0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()

    axes[0, 1].bar(x - width / 2, [source_summary[label]["baseline_metallic_mean"] for label in labels], width, label="Input prior")
    axes[0, 1].bar(x + width / 2, [source_summary[label]["refined_metallic_mean"] for label in labels], width, label="Refined")
    axes[0, 1].set_title("Metallic Mean")
    axes[0, 1].set_xticks(x, labels, rotation=15, ha="right")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    axes[1, 0].bar(x - width / 2, [source_summary[label]["roughness_abs_delta_mean"] for label in labels], width, label="|Δ roughness|")
    axes[1, 0].bar(x + width / 2, [source_summary[label]["metallic_abs_delta_mean"] for label in labels], width, label="|Δ metallic|")
    axes[1, 0].set_title("Mean Absolute Attribute Shift")
    axes[1, 0].set_xticks(x, labels, rotation=15, ha="right")
    axes[1, 0].legend()

    rough_x = [row["baseline_roughness_mean"] for row in rows]
    rough_y = [row["refined_roughness_mean"] for row in rows]
    metal_x = [row["baseline_metallic_mean"] for row in rows]
    metal_y = [row["refined_metallic_mean"] for row in rows]
    axes[1, 1].scatter(rough_x, rough_y, label="roughness", alpha=0.72)
    axes[1, 1].scatter(metal_x, metal_y, label="metallic", alpha=0.72)
    axes[1, 1].plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    axes[1, 1].set_title("Per-Asset Input Prior vs Refined Mean")
    axes[1, 1].set_xlabel("Input prior mean")
    axes[1, 1].set_ylabel("Refined mean")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_html(summary: dict[str, Any], plot_path: Path, output_path: Path) -> None:
    rows = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Material Attribute Comparison</title>",
        "<style>body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1280px;margin:auto}.card{background:#18202b;border-radius:18px;padding:18px;margin:16px 0}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #334052;padding:8px;text-align:left}img{max-width:100%;border-radius:14px;background:white}</style>",
        "</head><body><div class='wrap'>",
        "<h1>Material Attribute Comparison</h1>",
        "<p>Baseline means the input prior roughness/metallic atlas used by the refiner. It may come from SF3D, an external asset prior, a scalar broadcast prior, or a fallback/no-prior default depending on prior_source_type.</p>",
        f"<div class='card'><img src='{html.escape(plot_path.name)}' alt='attribute comparison'></div>",
        "<div class='card'><h2>By Source</h2><table><thead><tr><th>Group</th><th>Count</th><th>Baseline R</th><th>Refined R</th><th>|ΔR|</th><th>Baseline M</th><th>Refined M</th><th>|ΔM|</th></tr></thead><tbody>",
    ]
    for group, item in summary["by_source_name"].items():
        rows.append(
            "<tr><td>%s</td><td>%d</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%.4f</td></tr>"
            % (
                html.escape(group),
                int(item["count"]),
                item["baseline_roughness_mean"],
                item["refined_roughness_mean"],
                item["roughness_abs_delta_mean"],
                item["baseline_metallic_mean"],
                item["refined_metallic_mean"],
                item["metallic_abs_delta_mean"],
            )
        )
    rows.extend(["</tbody></table></div>", "</div></body></html>"])
    output_path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = json.loads(args.metrics_json.read_text())
    rows = build_attribute_rows(metrics_rows)
    summary = {
        "rows": len(rows),
        "overall": summarize_group(rows),
        "by_generator_id": grouped_summary(rows, "generator_id"),
        "by_source_name": grouped_summary(rows, "source_name"),
        "by_prior_label": grouped_summary(rows, "prior_label"),
        "by_supervision_tier": grouped_summary(rows, "supervision_tier"),
        "records": rows,
    }
    summary_path = args.output_dir / "material_attribute_summary.json"
    plot_path = args.output_dir / "material_attribute_comparison.png"
    html_path = args.output_dir / "material_attribute_comparison.html"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_plot(rows, summary, plot_path)
    save_html(summary, plot_path, html_path)
    print(json.dumps({"summary": str(summary_path), "plot": str(plot_path), "html": str(html_path)}, indent=2))


if __name__ == "__main__":
    main()
