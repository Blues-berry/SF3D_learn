from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a compact round-level analysis report for material refinement training/eval/audit outputs.",
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def metric(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.{digits}f}"


def rel(path: Path, base: Path) -> str:
    return html.escape(os.path.relpath(path, base))


def best_history_row(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in history if row.get("val")]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["val"]["uv_mae"]["total"])


def build_analysis(
    *,
    train_dir: Path,
    eval_dir: Path,
    audit_dir: Path,
) -> dict[str, Any]:
    train_state = load_json(train_dir / "train_state.json")
    history = load_json(train_dir / "history.json")
    eval_summary = load_json(eval_dir / "summary.json")
    attr_summary = load_json(eval_dir / "material_attribute_summary.json")
    audit_payload = load_json(audit_dir / "manifest_audit_summary.json")
    best_row = best_history_row(history)
    warnings = []
    audit_summary = audit_payload["summary"]
    if audit_summary.get("identity_warning"):
        warnings.append(
            "Target roughness/metallic maps are identical to input prior maps for most audited records; input-prior-vs-refined MAE is not a quality-improvement proof."
        )
    if eval_summary.get("baseline_total_mae", 0.0) == 0.0 and eval_summary.get("refined_total_mae", 0.0) > 0.0:
        warnings.append(
            "Input prior MAE is zero while refined MAE is non-zero, which indicates the current pseudo-GT favors the input prior exactly."
        )
    return {
        "train_state": train_state,
        "best_epoch": None if best_row is None else best_row.get("epoch"),
        "best_val_uv_total_mae": None
        if best_row is None
        else best_row["val"]["uv_mae"]["total"],
        "first_val_uv_total_mae": None
        if not history or not history[0].get("val")
        else history[0]["val"]["uv_mae"]["total"],
        "last_val_uv_total_mae": None
        if not history or not history[-1].get("val")
        else history[-1]["val"]["uv_mae"]["total"],
        "eval": {
            "rows": eval_summary.get("rows"),
            "baseline_total_mae": eval_summary.get("baseline_total_mae"),
            "refined_total_mae": eval_summary.get("refined_total_mae"),
            "by_generator_id": eval_summary.get("by_generator_id"),
            "by_prior_label": eval_summary.get("by_prior_label"),
            "by_source_name": eval_summary.get("by_source_name"),
        },
        "attribute": {
            "overall": attr_summary.get("overall"),
            "by_prior_label": attr_summary.get("by_prior_label"),
        },
        "manifest_audit": audit_summary,
        "warnings": warnings,
        "recommended_next_steps": [
            "Replace or augment pseudo-GT with non-trivial RM targets before claiming quality improvement over any input prior.",
            "Run manifest audit as a required gate before full training.",
            "Keep round1 checkpoint as a pipeline smoke reference, not as a production quality checkpoint.",
            "Use material-sensitive heldout sets for metal/non-metal confusion and boundary bleed once real targets are available.",
        ],
    }


def save_html(analysis: dict[str, Any], *, train_dir: Path, eval_dir: Path, audit_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "round_analysis.html"
    cards = [
        ("Best Val UV MAE", metric(analysis["best_val_uv_total_mae"])),
        ("Eval Refined MAE", metric(analysis["eval"]["refined_total_mae"])),
        ("Input Prior MAE", metric(analysis["eval"]["baseline_total_mae"])),
        ("Target==Prior", f"{analysis['manifest_audit']['same_rm_pair_rate']:.2%}"),
        ("Rendered RM Views", f"{analysis['manifest_audit'].get('rendered_rm_view_rate', 0.0):.2%}"),
    ]
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Material Refine Round Analysis</title>",
        "<style>body{font-family:Arial,sans-serif;background:#0f141c;color:#eef3f8;margin:0;padding:24px}.wrap{max-width:1400px;margin:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}.card{background:#18202b;border-radius:18px;padding:18px;margin:16px 0}.stat{background:#202a37;border-radius:16px;padding:16px}.warn{border-left:5px solid #ffba66}.viz{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}img{max-width:100%;border-radius:14px;background:white}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #334052;padding:8px;text-align:left}code{color:#b9e6ff}</style>",
        "</head><body><div class='wrap'>",
        "<h1>Material Refine Round Analysis</h1>",
        "<div class='grid'>",
    ]
    for label, value in cards:
        lines.append(f"<div class='stat'><div>{html.escape(label)}</div><h2>{html.escape(value)}</h2></div>")
    lines.append("</div>")
    if analysis["warnings"]:
        lines.append("<div class='card warn'><h2>Warnings</h2><ul>")
        for warning in analysis["warnings"]:
            lines.append(f"<li>{html.escape(warning)}</li>")
        lines.append("</ul></div>")
    lines.extend(
        [
            "<div class='card viz'>",
            f"<div><h2>Training Curves</h2><img src='{rel(train_dir / 'training_curves.png', output_dir)}'></div>",
            f"<div><h2>Material Attributes</h2><img src='{rel(eval_dir / 'material_attribute_comparison.png', output_dir)}'></div>",
            f"<div><h2>Manifest Audit</h2><img src='{rel(audit_dir / 'manifest_audit.png', output_dir)}'></div>",
            "</div>",
            "<div class='card'><h2>Eval By Prior</h2><table><thead><tr><th>Prior</th><th>Count</th><th>Input Prior MAE</th><th>Refined MAE</th><th>Gain</th></tr></thead><tbody>",
        ]
    )
    for prior, row in (analysis["eval"].get("by_prior_label") or {}).items():
        lines.append(
            f"<tr><td>{html.escape(prior)}</td><td>{row['count']}</td><td>{metric(row['baseline_total_mae'])}</td><td>{metric(row['refined_total_mae'])}</td><td>{metric(row['improvement_total'])}</td></tr>"
        )
    lines.extend(["</tbody></table></div>", "<div class='card'><h2>Recommended Next Steps</h2><ol>"])
    for step in analysis["recommended_next_steps"]:
        lines.append(f"<li>{html.escape(step)}</li>")
    lines.extend(["</ol></div>", "</div></body></html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")
    return html_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis = build_analysis(train_dir=args.train_dir, eval_dir=args.eval_dir, audit_dir=args.audit_dir)
    json_path = args.output_dir / "round_analysis.json"
    html_path = save_html(
        analysis,
        train_dir=args.train_dir.resolve(),
        eval_dir=args.eval_dir.resolve(),
        audit_dir=args.audit_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "html": str(html_path), "warnings": analysis["warnings"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
