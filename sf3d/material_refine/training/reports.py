from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .common import format_metric, format_duration, maybe_float, maybe_float_or_none, save_json

def validation_event_sort_key(path: Path) -> tuple[int, int, str]:
    label = path.stem
    parts = label.split("_")
    if len(parts) >= 2 and parts[0] == "progress":
        try:
            return (0, int(parts[1]), label)
        except ValueError:
            pass
    if len(parts) >= 2 and parts[0] == "epoch":
        try:
            return (1, int(parts[1]), label)
        except ValueError:
            pass
    return (2, 0, label)


def load_validation_events(output_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    validation_dir = output_dir / "validation"
    if not validation_dir.exists():
        return events
    for path in sorted(validation_dir.glob("*.json"), key=validation_event_sort_key):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report export must not block training.
            print(f"[visualization:warning] validation_event_read_failed={path.name}:{type(exc).__name__}:{exc}")
            continue
        render_proxy = payload.get("render_proxy_validation") or {}
        improvement = payload.get("improvement_uv_mae") or {}
        case_level = payload.get("case_level") or {}
        selection = payload.get("selection_metric") or {}
        comparison = payload.get("comparison") or {}
        prior_aware = payload.get("prior_aware") or {}
        events.append(
            {
                "label": payload.get("validation_label", path.stem),
                "epoch": payload.get("epoch"),
                "optimizer_step": payload.get("optimizer_step"),
                "evaluation_basis": payload.get("evaluation_basis"),
                "record_count": payload.get("record_count"),
                "dataset_record_count": payload.get("dataset_record_count"),
                "uv_total": (payload.get("uv_mae") or {}).get("total"),
                "uv_roughness": (payload.get("uv_mae") or {}).get("roughness"),
                "uv_metallic": (payload.get("uv_mae") or {}).get("metallic"),
                "input_prior_total": (
                    payload.get("input_prior_uv_mae")
                    or payload.get("baseline_uv_mae")
                    or {}
                ).get("total"),
                "uv_gain": improvement.get("total"),
                "sample_regression_rate": improvement.get("regression_rate"),
                "case_avg_gain": case_level.get("avg_improvement_total"),
                "case_regression_rate": case_level.get("regression_rate"),
                "rm_proxy_view_mae_delta": render_proxy.get("view_rm_mae_delta"),
                "rm_proxy_view_mse_delta": render_proxy.get("proxy_rm_mse_delta"),
                "rm_proxy_view_psnr_delta": render_proxy.get("proxy_rm_psnr_delta"),
                "prior_aware_score": prior_aware.get("score"),
                "comparison": comparison,
                "delta_vs_prior": comparison.get("delta_vs_prior"),
                "delta_vs_previous_baseline_run": comparison.get("delta_vs_previous_baseline_run"),
                "vs_prior": comparison.get("vs_prior"),
                "vs_baseline_run": comparison.get("vs_baseline_run"),
                "selection_metric": selection.get("selection_metric"),
                "selection_mode": selection.get("mode"),
                "path": str(path.resolve()),
            }
        )
    return events


def maybe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def maybe_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def inline_png_data_uri(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_latest_validation_payload(validation_events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not validation_events:
        return None
    latest_path = validation_events[-1].get("path")
    if not latest_path:
        return None
    try:
        return json.loads(Path(str(latest_path)).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report export must not block training.
        print(f"[visualization:warning] latest_validation_read_failed={type(exc).__name__}:{exc}")
        return None


def summarize_latest_validation(
    history: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
) -> dict[str, Any]:
    latest_epoch = history[-1] if history else {}
    latest_epoch_val = latest_epoch.get("val") or {}
    if latest_epoch_val:
        improvement_payload = latest_epoch_val.get("improvement_uv_mae") or {}
        baseline_payload = (
            latest_epoch_val.get("input_prior_uv_mae")
            or latest_epoch_val.get("baseline_uv_mae")
            or {}
        )
        return {
            "source": "epoch_validation",
            "label": f"epoch_{int(latest_epoch.get('epoch', 0)):03d}",
            "epoch": latest_epoch.get("epoch"),
            "optimizer_step": latest_epoch.get("optimizer_step"),
            "val_total": (latest_epoch_val.get("uv_mae") or {}).get("total"),
            "input_prior_total": baseline_payload.get("total"),
            "improvement_total": improvement_payload.get("total"),
            "selection_metric": (latest_epoch_val.get("selection_metric") or {}).get("selection_metric"),
            "evaluation_basis": latest_epoch_val.get("evaluation_basis"),
            "record_count": latest_epoch_val.get("record_count"),
            "dataset_record_count": latest_epoch_val.get("dataset_record_count"),
        }
    if validation_events:
        latest_event = validation_events[-1]
        return {
            "source": "validation_event",
            "label": latest_event.get("label"),
            "epoch": latest_event.get("epoch"),
            "optimizer_step": latest_event.get("optimizer_step"),
            "val_total": latest_event.get("uv_total"),
            "input_prior_total": latest_event.get("input_prior_total"),
            "improvement_total": latest_event.get("uv_gain"),
            "selection_metric": latest_event.get("selection_metric"),
            "evaluation_basis": latest_event.get("evaluation_basis"),
            "record_count": latest_event.get("record_count"),
            "dataset_record_count": latest_event.get("dataset_record_count"),
        }
    return {
        "source": "missing",
        "label": None,
        "epoch": latest_epoch.get("epoch"),
        "optimizer_step": latest_epoch.get("optimizer_step"),
        "val_total": None,
        "input_prior_total": None,
        "improvement_total": None,
        "selection_metric": None,
        "evaluation_basis": None,
        "record_count": None,
        "dataset_record_count": None,
    }


def load_benchmark_summary(output_dir: Path) -> dict[str, Any] | None:
    candidate_paths = [
        output_dir / "post_train_optimization_suite" / "val_ours_full" / "summary.json",
        output_dir / "post_train_suite" / "val_ours_full" / "summary.json",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - overview export must not block training.
            print(f"[visualization:warning] benchmark_summary_read_failed={path.name}:{type(exc).__name__}:{exc}")
            return None
    return None


def build_variant_summary_rows(
    validation_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not validation_payload:
        return []
    group_metrics = validation_payload.get("group_metrics") or {}
    baseline_group_metrics = validation_payload.get("baseline_group_metrics") or {}
    improvement_group_metrics = validation_payload.get("improvement_group_metrics") or {}
    view_stats_by_variant = validation_payload.get("view_stats_by_variant") or {}
    rows = []
    for group_key, metrics in sorted(group_metrics.items()):
        if not str(group_key).startswith("prior_variant_type/"):
            continue
        variant = str(group_key).split("/", 1)[1]
        view_stats = view_stats_by_variant.get(variant) or {}
        rows.append(
            {
                "variant": variant,
                "input_prior_total_mae": (
                    (baseline_group_metrics.get(group_key) or {}).get("total_mae")
                ),
                "refined_total_mae": metrics.get("total_mae"),
                "gain_total": (
                    (improvement_group_metrics.get(group_key) or {}).get("total_mae")
                ),
                "effective_view_supervision_rate": view_stats.get("effective_view_supervision_rate"),
                "sampled_view_rm_proxy_delta": view_stats.get("sampled_view_rm_proxy_delta"),
            }
        )
    return rows


def write_training_overview(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    train_state: dict[str, Any],
    validation_events: list[dict[str, Any]],
    visualization_paths: dict[str, str],
) -> Path | None:
    if not history and not validation_events:
        return None
    latest_train = (history[-1].get("train") or {}) if history else {}
    latest_validation = summarize_latest_validation(history, validation_events)
    latest_validation_payload = load_latest_validation_payload(validation_events)
    latest_render_proxy = (latest_validation_payload or {}).get("render_proxy_validation") or {}
    latest_comparison = (latest_validation_payload or {}).get("comparison") or {}
    latest_delta_vs_prior = latest_comparison.get("delta_vs_prior") or {}
    latest_delta_vs_previous = latest_comparison.get("delta_vs_previous_baseline_run") or {}
    latest_object_level = (latest_validation_payload or {}).get("object_level") or {}
    latest_case_level = (latest_validation_payload or {}).get("case_level") or {}
    benchmark_summary = load_benchmark_summary(output_dir) or {}
    variant_rows = build_variant_summary_rows(latest_validation_payload)
    evidence_curve_uri = inline_png_data_uri(
        Path(str(visualization_paths["training_evidence_curves"]))
        if "training_evidence_curves" in visualization_paths
        else None
    )
    train_curve_uri = inline_png_data_uri(
        Path(str(visualization_paths["training_curves"]))
        if "training_curves" in visualization_paths
        else None
    )
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    best_event = None
    if validation_events:
        best_event = max(
            validation_events,
            key=lambda item: maybe_float(item.get("uv_gain")),
        )
    variant_rows_html = [
        "<tr>"
        f"<td>{html.escape(str(row.get('variant', 'unknown')))}</td>"
        f"<td>{format_metric(row.get('input_prior_total_mae'))}</td>"
        f"<td>{format_metric(row.get('refined_total_mae'))}</td>"
        f"<td>{format_metric(row.get('gain_total'))}</td>"
        f"<td>{format_metric(row.get('effective_view_supervision_rate'), 4)}</td>"
        f"<td>{format_metric(row.get('sampled_view_rm_proxy_delta'), 4)}</td>"
        "</tr>"
        for row in variant_rows
    ]
    overview_path = output_dir / "training_overview.html"
    overview_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Material Refiner Training Overview</title>",
                (
                    "<style>"
                    "body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}"
                    ".wrap{max-width:1320px;margin:auto}"
                    ".card{background:#18202b;border-radius:12px;padding:18px;margin:16px 0}"
                    ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}"
                    ".metric{font-size:13px;color:#b9e6ff;margin-bottom:6px}"
                    "img{max-width:100%;border-radius:8px;background:#fff}"
                    "table{border-collapse:collapse;width:100%;font-size:13px}"
                    "th,td{border-bottom:1px solid #2d3748;padding:8px;text-align:left}"
                    "th{color:#b9e6ff}"
                    "code{color:#b9e6ff}"
                    "ul{margin:8px 0 0 20px;padding:0}"
                    "</style>"
                ),
                "</head><body><div class='wrap'>",
                "<h1>Material Refiner Training Overview</h1>",
                "<div class='card'><div class='grid'>",
                f"<div><div class='metric'>Run</div><div><code>{html.escape(str(args.tracker_run_name or output_dir.name))}</code></div></div>",
                f"<div><div class='metric'>Project / Group</div><div><code>{html.escape(str(args.tracker_project_name))}</code> / <code>{html.escape(str(args.tracker_group))}</code></div></div>",
                f"<div><div class='metric'>Best Checkpoint</div><div><code>{html.escape(str(best_path.resolve() if best_path.exists() else best_path))}</code></div></div>",
                f"<div><div class='metric'>Latest Checkpoint</div><div><code>{html.escape(str(latest_path.resolve() if latest_path.exists() else latest_path))}</code></div></div>",
                f"<div><div class='metric'>Best Epoch</div><div><code>{html.escape(str(train_state.get('best_epoch')))}</code></div></div>",
                f"<div><div class='metric'>Best Selection Metric</div><div><code>{format_metric(train_state.get('best_val_metric'))}</code></div></div>",
                "</div></div>",
                "<div class='card'><h2>Core Summary</h2><div class='grid'>",
                f"<div><div class='metric'>Latest Train Total</div><div><code>{format_metric(latest_train.get('total'))}</code></div></div>",
                f"<div><div class='metric'>Latest Validation Source</div><div><code>{html.escape(str(latest_validation.get('source')))}</code></div></div>",
                f"<div><div class='metric'>Latest Validation Label</div><div><code>{html.escape(str(latest_validation.get('label')))}</code></div></div>",
                f"<div><div class='metric'>Selection Basis</div><div><code>{html.escape(str(latest_validation.get('evaluation_basis') or 'unknown'))}</code></div></div>",
                f"<div><div class='metric'>Selection Record Count</div><div><code>{html.escape(str(latest_validation.get('record_count') or 'unknown'))}</code> / dataset <code>{html.escape(str(latest_validation.get('dataset_record_count') or 'unknown'))}</code></div></div>",
                f"<div><div class='metric'>Latest UV Total</div><div><code>{format_metric(latest_validation.get('val_total'))}</code></div></div>",
                f"<div><div class='metric'>Latest Input Prior Total</div><div><code>{format_metric(latest_validation.get('input_prior_total'))}</code></div></div>",
                f"<div><div class='metric'>Latest UV Gain</div><div><code>{format_metric(latest_validation.get('improvement_total'))}</code></div></div>",
                f"<div><div class='metric'>Delta vs Prior UV Total</div><div><code>{format_metric(latest_delta_vs_prior.get('uv_total'))}</code></div></div>",
                f"<div><div class='metric'>Delta vs Previous Baseline UV Total</div><div><code>{format_metric(latest_delta_vs_previous.get('uv_total'))}</code></div></div>",
                f"<div><div class='metric'>Delta vs Previous Baseline UV Gain</div><div><code>{format_metric(latest_delta_vs_previous.get('uv_gain'))}</code></div></div>",
                f"<div><div class='metric'>Latest RM Proxy View MAE Delta</div><div><code>{format_metric(latest_render_proxy.get('view_rm_mae_delta'))}</code></div></div>",
                f"<div><div class='metric'>Latest RM Proxy View MSE Delta</div><div><code>{format_metric(latest_render_proxy.get('proxy_rm_mse_delta'))}</code></div></div>",
                f"<div><div class='metric'>Latest RM Proxy View PSNR Delta</div><div><code>{format_metric(latest_render_proxy.get('proxy_rm_psnr_delta'))}</code></div></div>",
                f"<div><div class='metric'>Delta vs Previous Baseline RM PSNR</div><div><code>{format_metric(latest_delta_vs_previous.get('rm_proxy_view_psnr_delta'))}</code></div></div>",
                f"<div><div class='metric'>Object Regression Rate</div><div><code>{format_metric(latest_object_level.get('regression_rate'), 4)}</code></div></div>",
                f"<div><div class='metric'>Case Regression Rate</div><div><code>{format_metric(latest_case_level.get('regression_rate'), 4)}</code></div></div>",
                f"<div><div class='metric'>Validation Events</div><div><code>{len(validation_events)}</code></div></div>",
                "</div></div>",
                (
                    "<div class='card'><h2>Best Gain Event</h2>"
                    f"<div><code>{html.escape(str(best_event.get('label')))}</code> | "
                    f"UV gain <code>{format_metric(best_event.get('uv_gain'))}</code> | "
                    f"RM PSNR delta <code>{format_metric(best_event.get('rm_proxy_view_psnr_delta'))}</code> | "
                    f"case regression <code>{format_metric(best_event.get('case_regression_rate'), 4)}</code></div>"
                    "</div>"
                )
                if best_event is not None
                else "",
                (
                    "<div class='card'><h2>Benchmark Summary</h2><div class='grid'>"
                    f"<div><div class='metric'>Benchmark Basis</div><div><code>{html.escape(str(benchmark_summary.get('evaluation_basis') or 'benchmark_val_full_190'))}</code></div></div>"
                    f"<div><div class='metric'>Benchmark Rows / Objects</div><div><code>{html.escape(str(benchmark_summary.get('rows') or 'n/a'))}</code> / <code>{html.escape(str((benchmark_summary.get('object_level') or {}).get('objects') or benchmark_summary.get('objects') or 'n/a'))}</code></div></div>"
                    f"<div><div class='metric'>Benchmark Input Prior Total</div><div><code>{format_metric(benchmark_summary.get('input_prior_total_mae'))}</code></div></div>"
                    f"<div><div class='metric'>Benchmark Refined Total</div><div><code>{format_metric(benchmark_summary.get('refined_total_mae'))}</code></div></div>"
                    f"<div><div class='metric'>Benchmark Gain</div><div><code>{format_metric(benchmark_summary.get('gain_total'))}</code></div></div>"
                    f"<div><div class='metric'>Benchmark Case Regression</div><div><code>{format_metric(benchmark_summary.get('regression_rate'), 4)}</code></div></div>"
                    f"<div><div class='metric'>Benchmark Object Regression</div><div><code>{format_metric((benchmark_summary.get('object_level') or {}).get('regression_rate'), 4)}</code></div></div>"
                    "</div></div>"
                )
                if benchmark_summary
                else "",
                (
                    "<div class='card'><h2>By Variant</h2><table><thead><tr>"
                    "<th>variant</th><th>input prior total MAE</th><th>refined total MAE</th><th>UV gain</th><th>view supervision rate</th><th>view RM delta</th>"
                    "</tr></thead><tbody>"
                    + "".join(variant_rows_html)
                    + "</tbody></table></div>"
                )
                if variant_rows_html
                else "<div class='card'><h2>By Variant</h2><div>No validation payload with by-variant metrics was available at overview export time.</div></div>",
                (
                    f"<div class='card'><h2>Validation Evidence Curves</h2><img src='{evidence_curve_uri}' alt='training evidence curves'></div>"
                )
                if evidence_curve_uri is not None
                else "",
                (
                    f"<div class='card'><h2>Training Curves</h2><img src='{train_curve_uri}' alt='training curves'></div>"
                )
                if train_curve_uri is not None
                else "",
                "<div class='card'><h2>Metric Semantics</h2><ul>"
                "<li><b>RM proxy</b>: view-projected roughness/metallic target metrics computed from UV RM maps through <code>view_uvs</code>.</li>"
                "<li><b>RGB proxy</b>: eval-only <code>proxy_uv_shading</code> metrics, not the same as RM proxy.</li>"
                "<li><b>Selection monitor</b>: <code>monitor_val_balanced_160</code> style balanced validation used for checkpoint selection.</li>"
                "<li><b>Benchmark summary</b>: full split benchmark written after training under post-train evaluation.</li>"
                "<li><b>Real render</b>: independent Blender re-render benchmark, separate from RM proxy and RGB proxy.</li>"
                "<li>This run is an A-track prior-gap validation on <code>ABO_locked_core</code> / <code>glossy_non_metal</code>, not a broad generalization claim.</li>"
                "</ul></div>",
                "</div></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return overview_path


def write_training_evidence_report(
    *,
    output_dir: Path,
    validation_events: list[dict[str, Any]],
    figure_path: Path | None,
) -> dict[str, str]:
    if not validation_events:
        return {}
    report_json_path = output_dir / "training_evidence_report.json"
    report_html_path = output_dir / "training_evidence_report.html"
    latest = validation_events[-1]
    best_gain = max(
        validation_events,
        key=lambda item: maybe_float(item.get("uv_gain")),
    )
    payload = {
        "metric_families": {
            "rm_proxy": "View-projected roughness/metallic metrics computed from UV RM maps and view_uvs.",
            "rgb_proxy": "Evaluation-only lightweight proxy_uv_shading metrics.",
            "real_render": "Independent Blender re-render benchmark.",
        },
        "evaluation_basis": latest.get("evaluation_basis"),
        "record_count": latest.get("record_count"),
        "dataset_record_count": latest.get("dataset_record_count"),
        "latest_validation_summary": {
            "label": latest.get("label"),
            "epoch": latest.get("epoch"),
            "optimizer_step": latest.get("optimizer_step"),
            "evaluation_basis": latest.get("evaluation_basis"),
            "record_count": latest.get("record_count"),
            "dataset_record_count": latest.get("dataset_record_count"),
            "uv_gain": latest.get("uv_gain"),
            "rm_proxy_view_mae_delta": latest.get("rm_proxy_view_mae_delta"),
            "rm_proxy_view_mse_delta": latest.get("rm_proxy_view_mse_delta"),
            "rm_proxy_view_psnr_delta": latest.get("rm_proxy_view_psnr_delta"),
            "prior_aware_score": latest.get("prior_aware_score"),
            "comparison": latest.get("comparison"),
            "delta_vs_prior": latest.get("delta_vs_prior"),
            "delta_vs_previous_baseline_run": latest.get("delta_vs_previous_baseline_run"),
            "vs_prior": latest.get("vs_prior"),
            "vs_baseline_run": latest.get("vs_baseline_run"),
            "case_regression_rate": latest.get("case_regression_rate"),
            "selection_metric": latest.get("selection_metric"),
        },
        "latest": latest,
        "best_uv_gain": best_gain,
        "events": validation_events,
    }
    save_json(report_json_path, payload)
    rows = []
    for item in validation_events[-32:]:
        rows.append(
            "<tr>"
            f"<td>{item.get('label')}</td>"
            f"<td>{format_metric(item.get('uv_gain'))}</td>"
            f"<td>{format_metric(item.get('rm_proxy_view_mae_delta'))}</td>"
            f"<td>{format_metric(item.get('rm_proxy_view_mse_delta'))}</td>"
            f"<td>{format_metric(item.get('rm_proxy_view_psnr_delta'))}</td>"
            f"<td>{format_metric(item.get('case_regression_rate'), 4)}</td>"
            f"<td>{format_metric(item.get('selection_metric'))}</td>"
            "</tr>"
        )
    report_html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Material Refiner Training Evidence</title>",
                "<style>body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1220px;margin:auto}.card{background:#18202b;border-radius:10px;padding:18px;margin:16px 0}img{max-width:100%;border-radius:8px;background:white}code{color:#b9e6ff}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #2d3748;padding:7px;text-align:left}th{color:#b9e6ff}</style>",
                "</head><body><div class='wrap'>",
                "<h1>Material Refiner Training Evidence</h1>",
                "<div class='card'>",
                "<div><b>RM proxy</b>: view-projected roughness/metallic target metrics from UV maps and view_uvs.</div>",
                "<div><b>RGB proxy</b>: eval proxy_uv_shading diagnostics. <b>Real render</b>: independent Blender re-render benchmark.</div>",
                f"<div>Selection basis: <code>{html.escape(str(latest.get('evaluation_basis') or 'unknown'))}</code>; record count: <code>{html.escape(str(latest.get('record_count') or 'unknown'))}</code> / dataset <code>{html.escape(str(latest.get('dataset_record_count') or 'unknown'))}</code></div>",
                f"<div>Latest UV gain: <code>{format_metric(latest.get('uv_gain'))}</code>; RM proxy PSNR delta: <code>{format_metric(latest.get('rm_proxy_view_psnr_delta'))}</code>; case regression: <code>{format_metric(latest.get('case_regression_rate'), 4)}</code></div>",
                f"<div>Delta vs prior UV total: <code>{format_metric((latest.get('delta_vs_prior') or {}).get('uv_total'))}</code>; delta vs previous baseline UV total: <code>{format_metric((latest.get('delta_vs_previous_baseline_run') or {}).get('uv_total'))}</code>; delta vs previous baseline RM PSNR: <code>{format_metric((latest.get('delta_vs_previous_baseline_run') or {}).get('rm_proxy_view_psnr_delta'))}</code></div>",
                f"<div>Best UV gain event: <code>{best_gain.get('label')}</code> = <code>{format_metric(best_gain.get('uv_gain'))}</code></div>",
                "</div>",
                f"<div class='card'><img src='{figure_path.name}' alt='training evidence curves'></div>" if figure_path is not None else "",
                "<div class='card'><table><thead><tr><th>event</th><th>UV gain</th><th>RM MAE delta</th><th>RM MSE delta</th><th>RM PSNR delta</th><th>case regression</th><th>selection</th></tr></thead><tbody>",
                *rows,
                "</tbody></table></div></div></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "training_evidence_report": str(report_html_path.resolve()),
        "training_evidence_json": str(report_json_path.resolve()),
    }

def save_training_visualizations(history: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    if not history:
        return {}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[visualization:warning] export_failed={type(exc).__name__}:{exc}")
        return {}

    validation_events = load_validation_events(output_dir)
    epochs = [int(item.get("epoch", index + 1)) for index, item in enumerate(history)]
    train_total = [float((item.get("train") or {}).get("total", np.nan)) for item in history]
    val_total_epoch = [
        float(((item.get("val") or {}).get("uv_mae") or {}).get("total", np.nan))
        for item in history
    ]
    baseline_val_total_epoch = [
        float(((item.get("val") or {}).get("baseline_uv_mae") or {}).get("total", np.nan))
        for item in history
    ]
    improvement_val_total_epoch = [
        float(((item.get("val") or {}).get("improvement_uv_mae") or {}).get("total", np.nan))
        for item in history
    ]
    val_roughness_epoch = [
        float(((item.get("val") or {}).get("uv_mae") or {}).get("roughness", np.nan))
        for item in history
    ]
    val_metallic_epoch = [
        float(((item.get("val") or {}).get("uv_mae") or {}).get("metallic", np.nan))
        for item in history
    ]
    samples_per_second = [
        float((item.get("train") or {}).get("samples_per_second", np.nan))
        for item in history
    ]
    use_event_validation = bool(validation_events)
    if use_event_validation:
        val_x = list(range(1, len(validation_events) + 1))
        val_x_labels = [str(item.get("label")) for item in validation_events]
        val_total = [maybe_float(item.get("uv_total")) for item in validation_events]
        baseline_val_total = [maybe_float(item.get("input_prior_total")) for item in validation_events]
        improvement_val_total = [maybe_float(item.get("uv_gain")) for item in validation_events]
        val_roughness = [maybe_float(item.get("uv_roughness")) for item in validation_events]
        val_metallic = [maybe_float(item.get("uv_metallic")) for item in validation_events]
    else:
        val_x = epochs
        val_x_labels = [str(epoch) for epoch in epochs]
        val_total = val_total_epoch
        baseline_val_total = baseline_val_total_epoch
        improvement_val_total = improvement_val_total_epoch
        val_roughness = val_roughness_epoch
        val_metallic = val_metallic_epoch

    figure_path = output_dir / "training_curves.png"
    html_path = output_dir / "training_summary.html"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Material Refiner Training Curves")
    if use_event_validation:
        axes[0, 0].plot(val_x, val_total, marker="o", label="val uv total")
    else:
        axes[0, 0].plot(epochs, train_total, marker="o", label="train total")
        axes[0, 0].plot(val_x, val_total, marker="o", label="val uv total")
    if not np.isnan(baseline_val_total).all():
        axes[0, 0].plot(val_x, baseline_val_total, marker="o", label="input prior")
    axes[0, 0].set_title("Validation Event UV MAE" if use_event_validation else "Total Loss / UV MAE")
    axes[0, 0].set_xlabel("validation event" if use_event_validation else "epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(val_x, val_roughness, marker="o", label="roughness")
    axes[0, 1].plot(val_x, val_metallic, marker="o", label="metallic")
    axes[0, 1].set_title("Validation UV MAE By Channel")
    axes[0, 1].set_xlabel("validation event" if use_event_validation else "epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    if not np.isnan(improvement_val_total).all():
        axes[1, 0].plot(val_x, improvement_val_total, marker="o")
        axes[1, 0].axhline(0.0, color="black", linewidth=1, alpha=0.35)
        axes[1, 0].set_title("Refined Gain Over Input Prior")
        axes[1, 0].set_ylabel("baseline MAE - refined MAE")
    else:
        axes[1, 0].plot(epochs, samples_per_second, marker="o")
        axes[1, 0].set_title("Training Throughput")
        axes[1, 0].set_ylabel("samples/sec")
    axes[1, 0].set_xlabel("validation event" if not np.isnan(improvement_val_total).all() and use_event_validation else "epoch")
    axes[1, 0].grid(alpha=0.25)

    best_index = int(np.nanargmin(val_total)) if not np.isnan(val_total).all() else len(val_total) - 1
    latest_validation = summarize_latest_validation(history, validation_events)
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.92,
        "\n".join(
            [
                f"epochs: {len(history)}",
                (
                    f"best validation event: {val_x_labels[best_index]}"
                    if use_event_validation
                    else f"best epoch: {epochs[best_index]}"
                ),
                f"best val total: {format_metric(val_total[best_index])}",
                f"best baseline total: {format_metric(baseline_val_total[best_index]) if not np.isnan(baseline_val_total).all() else 'n/a'}",
                f"best improvement: {format_metric(improvement_val_total[best_index]) if not np.isnan(improvement_val_total).all() else 'n/a'}",
                f"last train total: {format_metric(train_total[-1])}",
                f"last val total: {format_metric(latest_validation.get('val_total'))}",
                f"latest validation source: {latest_validation.get('source')}",
                f"latest validation label: {latest_validation.get('label')}",
            ]
        ),
        va="top",
        family="monospace",
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    evidence_figure_path: Path | None = None
    if validation_events:
        evidence_figure_path = output_dir / "training_evidence_curves.png"
        event_x = list(range(1, len(validation_events) + 1))
        event_labels = [str(item.get("label")) for item in validation_events]
        uv_gain = [maybe_float(item.get("uv_gain")) for item in validation_events]
        rm_mae_delta = [maybe_float(item.get("rm_proxy_view_mae_delta")) for item in validation_events]
        rm_mse_delta = [maybe_float(item.get("rm_proxy_view_mse_delta")) for item in validation_events]
        rm_psnr_delta = [maybe_float(item.get("rm_proxy_view_psnr_delta")) for item in validation_events]
        case_regression = [maybe_float(item.get("case_regression_rate")) for item in validation_events]
        fig2, axes2 = plt.subplots(2, 2, figsize=(13, 8))
        fig2.suptitle("Validation Evidence: Baseline vs Refined")
        axes2[0, 0].plot(event_x, uv_gain, marker="o")
        axes2[0, 0].axhline(0.0, color="black", linewidth=1, alpha=0.35)
        axes2[0, 0].set_title("UV Gain (input prior MAE - refined MAE)")
        axes2[0, 0].grid(alpha=0.25)
        axes2[0, 1].plot(event_x, rm_psnr_delta, marker="o", label="PSNR delta")
        axes2[0, 1].axhline(0.0, color="black", linewidth=1, alpha=0.35)
        axes2[0, 1].set_title("RM Proxy View PSNR Delta")
        axes2[0, 1].grid(alpha=0.25)
        axes2[1, 0].plot(event_x, rm_mae_delta, marker="o", label="MAE delta")
        axes2[1, 0].plot(event_x, rm_mse_delta, marker="o", label="MSE delta")
        axes2[1, 0].axhline(0.0, color="black", linewidth=1, alpha=0.35)
        axes2[1, 0].set_title("RM Proxy Positive Delta Means Refined Is Better")
        axes2[1, 0].legend()
        axes2[1, 0].grid(alpha=0.25)
        axes2[1, 1].plot(event_x, case_regression, marker="o")
        axes2[1, 1].set_title("Case-Level Regression Rate")
        axes2[1, 1].grid(alpha=0.25)
        for axis in axes2.flat:
            axis.set_xlabel("validation event")
            if len(event_labels) <= 12:
                axis.set_xticks(event_x)
                axis.set_xticklabels(event_labels, rotation=30, ha="right")
        fig2.tight_layout()
        fig2.savefig(evidence_figure_path, dpi=160)
        plt.close(fig2)

    latest = history[-1]
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Material Refiner Training Summary</title>",
                "<style>body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1200px;margin:auto}.card{background:#18202b;border-radius:18px;padding:18px;margin:16px 0}img{max-width:100%;border-radius:14px;background:white}code{color:#b9e6ff}</style>",
                "</head><body><div class='wrap'>",
                "<h1>Material Refiner Training Summary</h1>",
                f"<div class='card'><img src='{figure_path.name}' alt='training curves'></div>",
                "<div class='card'>",
                f"<div>Last epoch: <code>{latest.get('epoch')}</code></div>",
                f"<div>Last optimizer step: <code>{latest.get('optimizer_step')}</code></div>",
                f"<div>Last train total: <code>{format_metric((latest.get('train') or {}).get('total'))}</code></div>",
                f"<div>Latest validation source: <code>{html.escape(str(latest_validation.get('source')))}</code></div>",
                f"<div>Latest validation label: <code>{html.escape(str(latest_validation.get('label')))}</code></div>",
                f"<div>Last val total: <code>{format_metric(latest_validation.get('val_total'))}</code></div>",
                f"<div>Last input prior total: <code>{format_metric(latest_validation.get('input_prior_total'))}</code></div>",
                f"<div>Last improvement: <code>{format_metric(latest_validation.get('improvement_total'))}</code></div>",
                f"<div>Selection basis: <code>{html.escape(str(latest_validation.get('evaluation_basis') or 'unknown'))}</code></div>",
                f"<div>Selection record count: <code>{html.escape(str(latest_validation.get('record_count') or 'unknown'))}</code> / dataset <code>{html.escape(str(latest_validation.get('dataset_record_count') or 'unknown'))}</code></div>",
                "</div></div></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    evidence_paths = write_training_evidence_report(
        output_dir=output_dir,
        validation_events=validation_events,
        figure_path=evidence_figure_path,
    )
    return {
        "training_curves": str(figure_path.resolve()),
        "training_summary": str(html_path.resolve()),
        **(
            {"training_evidence_curves": str(evidence_figure_path.resolve())}
            if evidence_figure_path is not None
            else {}
        ),
        **evidence_paths,
    }
