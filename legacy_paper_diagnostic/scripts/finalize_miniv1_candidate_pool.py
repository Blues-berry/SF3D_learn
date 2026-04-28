#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DOCS_ROOT = PROJECT_ROOT / "docs" / "neural_gaffer_dataset_audit"
INPUT_POOL = DOCS_ROOT / "asset_supervision_audit_second_pass_ecommerce_glb_priority_500_probed.csv"
INPUT_SUBSET = DOCS_ROOT / "asset_supervision_semantic_validation_subset_24.csv"
INPUT_METRICS = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d/output/abo_material_probe_semantic24/metrics.json")

OUTPUT_CSV = DOCS_ROOT / "asset_supervision_miniv1_candidate_pool_500.csv"
OUTPUT_JSON = DOCS_ROOT / "asset_supervision_miniv1_candidate_pool_500.json"
OUTPUT_MD = DOCS_ROOT / "asset_supervision_miniv1_candidate_pool_500_summary.md"

POOL_VERSION = "mini_v1_candidate_pool_500"
POOL_STATUS = "primary_candidate"
SEMANTIC_SOURCE = "semantic_validation_subset_24_gt_render_only"


def mean(values: List[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def load_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_metrics(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    return data


def build_review_map(metrics_rows: List[dict]) -> Dict[str, dict]:
    gt_stats = defaultdict(
        lambda: {
            "rough_std": [],
            "metal_std": [],
            "metal_mean": [],
            "rough_span": [],
            "metal_span": [],
        }
    )
    for row in metrics_rows:
        stats = gt_stats[row["object_id"]]
        stats["rough_std"].append(float(row["gt_roughness_std"]))
        stats["metal_std"].append(float(row["gt_metallic_std"]))
        stats["metal_mean"].append(float(row["gt_metallic_mean"]))
        stats["rough_span"].append(float(row["gt_roughness_p90"]) - float(row["gt_roughness_p10"]))
        stats["metal_span"].append(float(row["gt_metallic_p90"]) - float(row["gt_metallic_p10"]))

    review_map: Dict[str, dict] = {}
    for object_id, stats in gt_stats.items():
        rough_std = mean(stats["rough_std"])
        metal_std = mean(stats["metal_std"])
        metal_mean = mean(stats["metal_mean"])
        rough_span = mean(stats["rough_span"])
        metal_span = mean(stats["metal_span"])

        labels: List[str] = []
        if rough_std < 0.03 and rough_span < 0.08:
            labels.append("review_constant_roughness")
        if metal_std < 0.03 and metal_span < 0.08 and metal_mean < 0.05:
            labels.append("review_constant_metallic_near_zero")
        if metal_std < 0.03 and metal_span < 0.08 and metal_mean > 0.95:
            labels.append("review_constant_metallic_near_one")
        if rough_std < 0.04 and metal_std < 0.04:
            labels.append("review_low_semantic_variation")

        if not labels:
            review_map[object_id] = {
                "review_status": "none",
                "review_priority": "none",
                "review_labels": "",
                "review_notes": "",
                "semantic_validation_status": "checked_clean",
            }
            continue

        high_risk_tags = {
            "review_constant_roughness",
            "review_low_semantic_variation",
            "review_constant_metallic_near_one",
        }
        priority = "high" if any(label in high_risk_tags for label in labels) else "medium"
        review_notes = (
            f"rough_std={rough_std:.4f}; rough_span={rough_span:.4f}; "
            f"metal_std={metal_std:.4f}; metal_span={metal_span:.4f}; metal_mean={metal_mean:.4f}"
        )
        review_map[object_id] = {
            "review_status": "needs_review",
            "review_priority": priority,
            "review_labels": ";".join(labels),
            "review_notes": review_notes,
            "semantic_validation_status": "flagged",
        }

    return review_map


def main():
    pool_rows = load_csv(INPUT_POOL)
    subset_rows = load_csv(INPUT_SUBSET)
    metrics_rows = load_metrics(INPUT_METRICS)

    subset_by_id = {row["object_id"]: row for row in subset_rows}
    review_map = build_review_map(metrics_rows)

    output_rows = []
    for row in pool_rows:
        object_id = row["object_id"]
        subset_row = subset_by_id.get(object_id)
        review = review_map.get(object_id)

        if review is None:
            review = {
                "review_status": "none",
                "review_priority": "none",
                "review_labels": "",
                "review_notes": "",
                "semantic_validation_status": "not_sampled",
            }

        out_row = dict(row)
        out_row.update(
            {
                "pool_version": POOL_VERSION,
                "pool_status": POOL_STATUS,
                "selection_decision": "keep_in_pool",
                "semantic_validation_source": SEMANTIC_SOURCE,
                "semantic_validation_status": review["semantic_validation_status"],
                "semantic_stratum": subset_row["semantic_stratum"] if subset_row else "",
                "review_status": review["review_status"],
                "review_priority": review["review_priority"],
                "review_labels": review["review_labels"],
                "review_notes": review["review_notes"],
            }
        )
        output_rows.append(out_row)

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    json_rows = [
        {
            "object_id": row["object_id"],
            "source_model_path": row["source_model_path"],
            "subset": row["subset"],
            "source": row["source"],
            "pool_version": row["pool_version"],
            "pool_status": row["pool_status"],
            "selection_decision": row["selection_decision"],
            "semantic_validation_status": row["semantic_validation_status"],
            "semantic_stratum": row["semantic_stratum"],
            "review_status": row["review_status"],
            "review_priority": row["review_priority"],
            "review_labels": row["review_labels"],
            "review_notes": row["review_notes"],
        }
        for row in output_rows
    ]
    OUTPUT_JSON.write_text(json.dumps(json_rows, indent=2), encoding="utf-8")

    review_rows = [row for row in output_rows if row["review_status"] == "needs_review"]
    status_counts = Counter(row["semantic_validation_status"] for row in output_rows)
    priority_counts = Counter(row["review_priority"] for row in output_rows)
    label_counts = Counter()
    for row in review_rows:
        for label in row["review_labels"].split(";"):
            if label:
                label_counts[label] += 1

    lines = [
        "# Mini-v1 Candidate Pool 500",
        "",
        f"- source_csv: {INPUT_POOL}",
        f"- semantic_subset_csv: {INPUT_SUBSET}",
        f"- semantic_metrics_json: {INPUT_METRICS}",
        f"- output_csv: {OUTPUT_CSV}",
        f"- output_json: {OUTPUT_JSON}",
        "",
        "## Decision",
        "",
        "- fix the current 500 ABO/ecommerce objects as the mini-v1 primary candidate supervision pool",
        "- keep the whole pool in place; do not reshuffle, replace, or downgrade the pool itself",
        "- attach review labels only to the small set surfaced by semantic calibration",
        "",
        "## Pool Counts",
        "",
        f"- total_objects: {len(output_rows)}",
        f"- primary_candidates: {sum(1 for row in output_rows if row['pool_status'] == POOL_STATUS)}",
        f"- keep_in_pool: {sum(1 for row in output_rows if row['selection_decision'] == 'keep_in_pool')}",
        "",
        "## Semantic Validation Status",
        "",
        f"- checked_clean: {status_counts.get('checked_clean', 0)}",
        f"- flagged: {status_counts.get('flagged', 0)}",
        f"- not_sampled: {status_counts.get('not_sampled', 0)}",
        "",
        "## Review Counts",
        "",
        f"- needs_review: {len(review_rows)}",
        f"- review_high: {priority_counts.get('high', 0)}",
        f"- review_medium: {priority_counts.get('medium', 0)}",
        "",
        "## Review Labels",
        "",
    ]
    for label, count in label_counts.most_common():
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "## Review Objects",
            "",
        ]
    )
    for row in review_rows:
        lines.append(
            "- "
            + f"{row['object_id']} | {Path(row['source_model_path']).stem} | "
            + f"{row['review_priority']} | {row['review_labels']}"
        )
    lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_MD}")
    print(
        {
            "total_objects": len(output_rows),
            "needs_review": len(review_rows),
            "review_high": priority_counts.get("high", 0),
            "review_medium": priority_counts.get("medium", 0),
        }
    )


if __name__ == "__main__":
    main()
