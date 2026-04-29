from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return f"{number:.{digits}f}"


def _variant_rows(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant, metrics in sorted((summary_payload.get("by_prior_variant_type") or {}).items()):
        rows.append(
            {
                "variant": str(variant),
                "count": metrics.get("count"),
                "input_prior_total_mae": metrics.get("input_prior_total_mae"),
                "refined_total_mae": metrics.get("refined_total_mae"),
                "gain_total": metrics.get("gain_total"),
                "regression_rate": metrics.get("prior_residual_regression_rate"),
            }
        )
    return rows


def build_report(metrics_path: Path, output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing_summary_json:{summary_path}")
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_main = summary_payload.get("metrics_main") or {}
    proxy_psnr = metrics_main.get("proxy_render_psnr") or {}
    proxy_mse = metrics_main.get("proxy_render_mse") or {}
    proxy_ssim = metrics_main.get("proxy_render_ssim") or {}
    proxy_lpips = metrics_main.get("proxy_render_lpips") or {}
    variant_rows = _variant_rows(summary_payload)

    report_path = output_dir / "evaluation_report.html"
    report_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Material Refiner Eval Report</title>",
                (
                    "<style>"
                    "body{font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}"
                    ".wrap{max-width:1280px;margin:auto}"
                    ".card{background:#172033;border-radius:12px;padding:18px;margin:16px 0}"
                    ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}"
                    ".metric{font-size:12px;color:#93c5fd;margin-bottom:6px}"
                    "table{border-collapse:collapse;width:100%}"
                    "th,td{padding:8px;border-bottom:1px solid #334155;text-align:left}"
                    "th{color:#93c5fd}"
                    "code{color:#bfdbfe}"
                    "</style>"
                ),
                "</head><body><div class='wrap'>",
                "<h1>Material Refiner Evaluation Report</h1>",
                "<div class='card'><div class='grid'>",
                f"<div><div class='metric'>Manifest</div><div><code>{html.escape(str(summary_payload.get('manifest')))}</code></div></div>",
                f"<div><div class='metric'>Checkpoint</div><div><code>{html.escape(str(summary_payload.get('checkpoint')))}</code></div></div>",
                f"<div><div class='metric'>Split</div><div><code>{html.escape(str(summary_payload.get('split')))}</code></div></div>",
                f"<div><div class='metric'>Evaluation Basis</div><div><code>{html.escape(str(summary_payload.get('evaluation_basis') or summary_payload.get('split') or 'unknown'))}</code></div></div>",
                f"<div><div class='metric'>Eval Variant</div><div><code>{html.escape(str(summary_payload.get('eval_variant')))}</code></div></div>",
                f"<div><div class='metric'>Rows / Objects</div><div><code>{html.escape(str(summary_payload.get('rows')))}</code> / <code>{html.escape(str(summary_payload.get('objects')))}</code></div></div>",
                f"<div><div class='metric'>Input Prior Total</div><div><code>{_fmt(summary_payload.get('input_prior_total_mae'))}</code></div></div>",
                f"<div><div class='metric'>Refined Total</div><div><code>{_fmt(summary_payload.get('refined_total_mae'))}</code></div></div>",
                f"<div><div class='metric'>Gain Total</div><div><code>{_fmt(summary_payload.get('gain_total'))}</code></div></div>",
                f"<div><div class='metric'>Case Regression</div><div><code>{_fmt(summary_payload.get('regression_rate'), 4)}</code></div></div>",
                f"<div><div class='metric'>Object Regression</div><div><code>{_fmt((summary_payload.get('object_level') or {}).get('regression_rate'), 4)}</code></div></div>",
                f"<div><div class='metric'>Metrics Rows JSON</div><div><code>{html.escape(str(metrics_path.resolve()))}</code></div></div>",
                "</div></div>",
                "<div class='card'><h2>Metric Families</h2><div class='grid'>",
                f"<div><div class='metric'>RM Proxy</div><div>{html.escape(str((summary_payload.get('metric_families') or {}).get('rm_proxy') or ''))}</div></div>",
                f"<div><div class='metric'>RGB Proxy</div><div>{html.escape(str((summary_payload.get('metric_families') or {}).get('rgb_proxy') or ''))}</div></div>",
                f"<div><div class='metric'>Real Render</div><div>{html.escape(str((summary_payload.get('metric_families') or {}).get('real_render') or ''))}</div></div>",
                "</div></div>",
                "<div class='card'><h2>RGB Proxy Diagnostics</h2><div class='grid'>",
                f"<div><div class='metric'>PSNR delta</div><div><code>{_fmt(proxy_psnr.get('delta'))}</code></div></div>",
                f"<div><div class='metric'>MSE delta</div><div><code>{_fmt(proxy_mse.get('delta'))}</code></div></div>",
                f"<div><div class='metric'>SSIM delta</div><div><code>{_fmt(proxy_ssim.get('delta'))}</code></div></div>",
                f"<div><div class='metric'>LPIPS delta</div><div><code>{_fmt(proxy_lpips.get('delta'))}</code></div></div>",
                "</div></div>",
                (
                    "<div class='card'><h2>By Prior Variant Type</h2><table><thead><tr>"
                    "<th>variant</th><th>count</th><th>input prior total MAE</th><th>refined total MAE</th><th>gain total</th><th>regression rate</th>"
                    "</tr></thead><tbody>"
                    + "".join(
                        [
                            "<tr>"
                            f"<td>{html.escape(str(row['variant']))}</td>"
                            f"<td>{html.escape(str(row['count']))}</td>"
                            f"<td>{_fmt(row['input_prior_total_mae'])}</td>"
                            f"<td>{_fmt(row['refined_total_mae'])}</td>"
                            f"<td>{_fmt(row['gain_total'])}</td>"
                            f"<td>{_fmt(row['regression_rate'], 4)}</td>"
                            "</tr>"
                            for row in variant_rows
                        ]
                    )
                    + "</tbody></table></div>"
                )
                if variant_rows
                else "",
                "</div></body></html>",
            ]
        ),
        encoding="utf-8",
    )

    paper_table = summary_payload.get("paper_main_table") or {}
    entries = paper_table.get("entries") or []
    if entries:
        paper_path = output_dir / "paper_metric_summary.html"
        headers = list((paper_table.get("metric_columns") or []))
        paper_path.write_text(
            "\n".join(
                [
                    "<!doctype html><html><head><meta charset='utf-8'><title>Paper Metric Summary</title>",
                    "<style>body{font-family:Arial,sans-serif;background:#ffffff;color:#111827;margin:24px}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #d1d5db;text-align:left}</style>",
                    "</head><body>",
                    "<h1>Paper Metric Summary</h1>",
                    "<table><thead><tr><th>method</th>"
                    + "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
                    + "<th>metric_basis</th><th>note</th></tr></thead><tbody>",
                    *[
                        "<tr>"
                        + f"<td>{html.escape(str(entry.get('method')))}</td>"
                        + "".join(f"<td>{_fmt(entry.get(header))}</td>" for header in headers)
                        + f"<td>{html.escape(str(entry.get('metric_basis')))}</td>"
                        + f"<td>{html.escape(str(entry.get('note')))}</td>"
                        + "</tr>"
                        for entry in entries
                    ],
                    "</tbody></table></body></html>",
                ]
            ),
            encoding="utf-8",
        )
    return report_path
