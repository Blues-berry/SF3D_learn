from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.manifest_quality import (
    DEFAULT_MAX_TARGET_PRIOR_IDENTITY_RATE_FOR_PAPER,
    DEFAULT_MIN_NONTRIVIAL_TARGET_COUNT_FOR_PAPER,
    audit_manifest,
)


def parse_csv_list(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in str(value).split(",")]
    items = {part for part in parts if part}
    return items or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CanonicalAssetRecordV1 manifests for paper-stage readiness, target leakage, and view-supervision completeness.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=-1, help="-1 means audit all records.")
    parser.add_argument("--identity-warning-threshold", type=float, default=0.95)
    parser.add_argument(
        "--max-target-prior-identity-rate-for-paper",
        type=float,
        default=DEFAULT_MAX_TARGET_PRIOR_IDENTITY_RATE_FOR_PAPER,
    )
    parser.add_argument(
        "--min-nontrivial-target-count-for-paper",
        type=int,
        default=DEFAULT_MIN_NONTRIVIAL_TARGET_COUNT_FOR_PAPER,
    )
    parser.add_argument(
        "--paper-license-buckets",
        type=str,
        default=None,
        help="Optional CSV allowlist. When unset, all license buckets are treated as internal-research-allowed.",
    )
    return parser.parse_args()


def save_plot(summary: dict[str, Any], output_path: Path) -> None:
    by_source = summary.get("by_source") or {}
    source_labels = list(by_source.keys()) or ["all"]
    source_identity = [by_source[label]["target_prior_identity_rate"] for label in source_labels]
    source_paper = [by_source[label]["paper_stage_eligible_rate"] for label in source_labels]
    source_view = [by_source[label]["effective_view_supervision_record_rate"] for label in source_labels]
    source_x = np.arange(len(source_labels), dtype=np.float32)
    width = 0.25

    field_rates = summary.get("buffer_field_rates") or {}
    field_labels = [
        "rgba",
        "mask",
        "depth",
        "normal",
        "position",
        "uv",
        "visibility",
        "roughness",
        "metallic",
    ]
    field_values = [float(field_rates.get(label, 0.0)) for label in field_labels]

    target_source_counts = summary.get("target_source_type_counts") or {}
    target_source_labels = list(target_source_counts.keys()) or ["unknown"]
    target_source_values = [int(target_source_counts[label]) for label in target_source_labels]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].bar(source_x - width, source_identity, width, label="target==prior")
    axes[0, 0].bar(source_x, source_paper, width, label="paper-eligible")
    axes[0, 0].bar(source_x + width, source_view, width, label="effective view supervision")
    axes[0, 0].set_xticks(source_x, source_labels, rotation=15, ha="right")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("By Source")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(np.arange(len(field_labels)), field_values, color="#7cb8ff")
    axes[0, 1].set_xticks(np.arange(len(field_labels)), field_labels, rotation=20, ha="right")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("View Buffer Field Rates")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(np.arange(len(target_source_labels)), target_source_values, color="#c1e1c1")
    axes[1, 0].set_xticks(np.arange(len(target_source_labels)), target_source_labels, rotation=20, ha="right")
    axes[1, 0].set_title("Target Source Types")
    axes[1, 0].grid(axis="y", alpha=0.25)

    source_counts = summary.get("source_counts") or {"all": 1}
    axes[1, 1].pie(source_counts.values(), labels=source_counts.keys(), autopct="%1.0f%%")
    axes[1, 1].set_title("Source Mix")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def table_rows_from_counts(counts: dict[str, Any]) -> str:
    rows = []
    for key, value in sorted(counts.items()):
        rows.append(f"<tr><td>{html.escape(str(key))}</td><td>{int(value)}</td></tr>")
    return "\n".join(rows) if rows else "<tr><td>none</td><td>0</td></tr>"


def save_html(summary: dict[str, Any], plot_path: Path, output_path: Path) -> None:
    readiness_state = "READY" if summary.get("paper_stage_ready") else "BLOCKED"
    readiness_color = "#90ee90" if summary.get("paper_stage_ready") else "#ffba66"
    blockers = summary.get("readiness_blockers") or ["none"]
    by_source = summary.get("by_source") or {}
    field_rates = summary.get("buffer_field_rates") or {}

    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Material Refine Manifest Audit</title>",
        "<style>body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1320px;margin:auto}.card{background:#18202b;border-radius:18px;padding:18px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}.stat{background:#202a37;border-radius:16px;padding:16px}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #334052;padding:8px;text-align:left}img{max-width:100%;border-radius:14px;background:white}code{color:#b9e6ff}</style>",
        "</head><body><div class='wrap'>",
        "<h1>Material Refine Manifest Audit</h1>",
        f"<div class='card'><h2>Paper-Stage Readiness: <span style='color:{readiness_color}'>{readiness_state}</span></h2>",
        f"<p>paper_stage_eligible_records={summary['paper_stage_eligible_records']} | nontrivial_target_records={summary['nontrivial_target_records']} | target_prior_identity_rate={summary['target_prior_identity_rate']:.2%}</p>",
        "<ul>",
    ]
    for blocker in blockers:
        lines.append(f"<li>{html.escape(str(blocker))}</li>")
    lines.extend(
        [
            "</ul></div>",
            f"<div class='card'><img src='{html.escape(plot_path.name)}' alt='audit plot'></div>",
            "<div class='grid'>",
            f"<div class='stat'><div>Records</div><h2>{summary['records']}</h2></div>",
            f"<div class='stat'><div>Complete</div><h2>{summary['complete_records']} / {summary['records']}</h2></div>",
            f"<div class='stat'><div>High-Identity Targets</div><h2>{summary['target_prior_identity_rate']:.2%}</h2></div>",
            f"<div class='stat'><div>Paper Eligible</div><h2>{summary['paper_stage_eligible_records']} / {summary['records']}</h2></div>",
            f"<div class='stat'><div>View Ready</div><h2>{summary['view_supervision_ready_rate']:.2%}</h2></div>",
            f"<div class='stat'><div>Effective View Supervision</div><h2>{summary['effective_view_supervision_record_rate']:.2%}</h2></div>",
            f"<div class='stat'><div>Strict Buffer Complete</div><h2>{summary['strict_complete_record_rate']:.2%}</h2></div>",
            "</div>",
            "<div class='card'><h2>Target Audit</h2>",
            f"<p>target_is_prior_copy: {summary['target_is_prior_copy_count']} ({summary['target_is_prior_copy_rate']:.2%})</p>",
            f"<p>target_prior_identity_mean: {summary['target_prior_identity_mean']:.4f}</p>",
            f"<p>paper license allowed: {summary['paper_license_allowed_records']} ({summary['paper_license_allowed_rate']:.2%})</p>",
            f"<p>confidence mean(mean): {summary['confidence_summary_aggregate']['mean_of_mean']:.4f} | mean(nonzero_rate): {summary['confidence_summary_aggregate']['mean_nonzero_rate']:.4f}</p>",
            "</div>",
            "<div class='card'><h2>Target Source Types</h2><table><thead><tr><th>Type</th><th>Count</th></tr></thead><tbody>",
            table_rows_from_counts(summary.get("target_source_type_counts") or {}),
            "</tbody></table></div>",
            "<div class='card'><h2>Target Quality Tiers</h2><table><thead><tr><th>Tier</th><th>Count</th></tr></thead><tbody>",
            table_rows_from_counts(summary.get("target_quality_tier_counts") or {}),
            "</tbody></table></div>",
            "<div class='card'><h2>Supervision Roles</h2><table><thead><tr><th>Role</th><th>Count</th></tr></thead><tbody>",
            table_rows_from_counts(summary.get("supervision_role_counts") or {}),
            "</tbody></table></div>",
            "<div class='card'><h2>Material Families</h2><table><thead><tr><th>Family</th><th>Count</th></tr></thead><tbody>",
            table_rows_from_counts(summary.get("material_family_counts") or {}),
            "</tbody></table></div>",
            "<div class='card'><h2>Lighting Banks</h2><table><thead><tr><th>Bank</th><th>Count</th></tr></thead><tbody>",
            table_rows_from_counts(summary.get("lighting_bank_id_counts") or {}),
            "</tbody></table></div>",
            "<div class='card'><h2>View Buffer Rates</h2><table><thead><tr><th>Field</th><th>Rate</th></tr></thead><tbody>",
        ]
    )
    for field, rate in sorted(field_rates.items()):
        lines.append(f"<tr><td>{html.escape(str(field))}</td><td>{float(rate):.2%}</td></tr>")
    lines.extend(["</tbody></table></div>", "<div class='card'><h2>By Source</h2><table><thead><tr><th>Source</th><th>Count</th><th>Paper Eligible</th><th>Target==Prior</th><th>Effective View Supervision</th></tr></thead><tbody>"])
    for source, bucket in sorted(by_source.items()):
        lines.append(
            f"<tr><td>{html.escape(source)}</td><td>{bucket['count']}</td><td>{bucket['paper_stage_eligible_rate']:.2%}</td><td>{bucket['target_prior_identity_rate']:.2%}</td><td>{bucket['effective_view_supervision_record_rate']:.2%}</td></tr>"
        )
    lines.extend(["</tbody></table></div>", "</div></body></html>"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_payload = audit_manifest(
        args.manifest,
        max_records=args.max_records,
        identity_warning_threshold=args.identity_warning_threshold,
        max_target_prior_identity_rate_for_paper=args.max_target_prior_identity_rate_for_paper,
        min_nontrivial_target_count_for_paper=args.min_nontrivial_target_count_for_paper,
        allowed_paper_license_buckets=parse_csv_list(args.paper_license_buckets),
    )
    summary = output_payload["summary"]
    summary_path = args.output_dir / "manifest_audit_summary.json"
    plot_path = args.output_dir / "manifest_audit.png"
    html_path = args.output_dir / "manifest_audit.html"
    summary_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    save_plot(summary, plot_path)
    save_html(summary, plot_path, html_path)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "plot": str(plot_path),
                "html": str(html_path),
                "paper_stage_ready": summary["paper_stage_ready"],
                "paper_stage_eligible_records": summary["paper_stage_eligible_records"],
                "target_prior_identity_rate": summary["target_prior_identity_rate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
