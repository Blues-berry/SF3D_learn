#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


INPUT_METRICS = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d/output/abo_material_probe_semantic24/metrics.json")
PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DOCS_ROOT = PROJECT_ROOT / "docs" / "neural_gaffer_dataset_audit"
INPUT_SUBSET = DOCS_ROOT / "asset_supervision_semantic_validation_subset_24.csv"
OUTPUT_MD = DOCS_ROOT / "semantic_validation_summary.md"


def mean(values):
    return 0.0 if not values else sum(values) / len(values)


def load_rows(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    return data


def main():
    metrics = load_rows(INPUT_METRICS)
    subset_rows = list(csv.DictReader(INPUT_SUBSET.open(newline="", encoding="utf-8")))
    subset_by_id = {row["object_id"]: row for row in subset_rows}

    rough_errors = [abs(row["pred_roughness"] - row["gt_roughness_mean"]) for row in metrics if row["pred_roughness"] is not None]
    metal_errors = [abs(row["pred_metallic"] - row["gt_metallic_mean"]) for row in metrics if row["pred_metallic"] is not None]
    inferred_mode = bool(rough_errors)

    tag_counts = Counter()
    object_tags = defaultdict(set)
    object_errors = defaultdict(list)
    gt_object_stats = defaultdict(lambda: {"rough_std": [], "metal_std": [], "metal_mean": [], "rough_p_span": [], "metal_p_span": []})
    for row in metrics:
        for tag in row.get("tags", []):
            tag_counts[tag] += 1
            object_tags[row["object_id"]].add(tag)
        if row["pred_roughness"] is not None:
            score = abs(row["pred_roughness"] - row["gt_roughness_mean"]) + abs(row["pred_metallic"] - row["gt_metallic_mean"])
            object_errors[row["object_id"]].append(score)
        gt_object_stats[row["object_id"]]["rough_std"].append(row["gt_roughness_std"])
        gt_object_stats[row["object_id"]]["metal_std"].append(row["gt_metallic_std"])
        gt_object_stats[row["object_id"]]["metal_mean"].append(row["gt_metallic_mean"])
        gt_object_stats[row["object_id"]]["rough_p_span"].append(row["gt_roughness_p90"] - row["gt_roughness_p10"])
        gt_object_stats[row["object_id"]]["metal_p_span"].append(row["gt_metallic_p90"] - row["gt_metallic_p10"])

    stratum_counts = Counter(row["semantic_stratum"] for row in subset_rows)
    objects_with_tags = sum(1 for tags in object_tags.values() if tags)
    inferred_views = sum(1 for row in metrics if row["pred_roughness"] is not None)

    gt_flags = {
        "constant_like_roughness": [],
        "constant_like_metallic": [],
        "low_semantic_variation": [],
    }
    constant_metal_near_zero = 0
    constant_metal_near_one = 0
    for object_id, stats in gt_object_stats.items():
        rough_std = mean(stats["rough_std"])
        metal_std = mean(stats["metal_std"])
        metal_mean = mean(stats["metal_mean"])
        rough_span = mean(stats["rough_p_span"])
        metal_span = mean(stats["metal_p_span"])
        if rough_std < 0.03 and rough_span < 0.08:
            gt_flags["constant_like_roughness"].append((object_id, rough_std, rough_span))
        if metal_std < 0.03 and metal_span < 0.08 and (metal_mean < 0.05 or metal_mean > 0.95):
            gt_flags["constant_like_metallic"].append((object_id, metal_std, metal_span))
            if metal_mean < 0.05:
                constant_metal_near_zero += 1
            elif metal_mean > 0.95:
                constant_metal_near_one += 1
        if rough_std < 0.04 and metal_std < 0.04:
            gt_flags["low_semantic_variation"].append((object_id, rough_std, metal_std))

    worst_objects = sorted(
        (
            (
                object_id,
                mean(scores),
                sorted(object_tags.get(object_id, [])),
                subset_by_id.get(object_id, {}),
            )
            for object_id, scores in object_errors.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:8]

    lines = [
        "# Semantic Validation Summary",
        "",
        f"- subset_csv: {INPUT_SUBSET}",
        f"- metrics_json: {INPUT_METRICS}",
        f"- output: {OUTPUT_MD}",
        "",
        "## Scope",
        "",
        f"- objects: {len(subset_rows)}",
        f"- views: {len(metrics)}",
        f"- inferred_views: {inferred_views}",
        "",
        "## Subset Breakdown",
        "",
    ]
    for key in ["dual_material", "single_material_texture_only", "single_material_factor_texture"]:
        lines.append(f"- {key}: {stratum_counts.get(key, 0)}")
    lines.extend(
        [
            "",
            "## Error Snapshot",
            "",
            f"- inference_mode: {'sf3d_baseline' if inferred_mode else 'gt_render_only'}",
            f"- mean_abs_roughness_error: {mean(rough_errors):.4f}" if inferred_mode else "- mean_abs_roughness_error: n/a (skip_inference)",
            f"- mean_abs_metallic_error: {mean(metal_errors):.4f}" if inferred_mode else "- mean_abs_metallic_error: n/a (skip_inference)",
            f"- objects_with_any_failure_tag: {objects_with_tags}/{len(subset_rows)}" if inferred_mode else "- objects_with_any_failure_tag: n/a (skip_inference)",
            "",
            "## GT Signal Checks",
            "",
            f"- constant_like_roughness_objects: {len(gt_flags['constant_like_roughness'])}/{len(subset_rows)}",
            f"- constant_like_metallic_objects: {len(gt_flags['constant_like_metallic'])}/{len(subset_rows)}",
            f"- constant_like_metallic_near_zero: {constant_metal_near_zero}",
            f"- constant_like_metallic_near_one: {constant_metal_near_one}",
            f"- low_semantic_variation_objects: {len(gt_flags['low_semantic_variation'])}/{len(subset_rows)}",
            "- note: constant-like metallic near 0 is often a valid non-metal signal and should be treated as a review flag, not an automatic reject.",
        ]
    )
    for key, title in [
        ("constant_like_roughness", "Constant-like Roughness Candidates"),
        ("constant_like_metallic", "Constant-like Metallic Candidates"),
        ("low_semantic_variation", "Low Semantic Variation Candidates"),
    ]:
        lines.extend(["", f"## {title}", ""])
        if not gt_flags[key]:
            lines.append("- none")
            continue
        for item in gt_flags[key][:8]:
            object_id = item[0]
            subset_row = subset_by_id.get(object_id, {})
            lines.append(f"- {object_id} | {subset_row.get('sku', '')} | {subset_row.get('semantic_stratum', '')} | stats={item[1:]}")
    if inferred_mode:
        lines.extend(
            [
                "",
                "## Failure Tags",
                "",
                "| tag | views |",
                "| --- | ---: |",
            ]
        )
        for tag in ["over_smoothing", "metal_nonmetal_confusion", "highlight_misread", "boundary_bleed"]:
            lines.append(f"| {tag} | {tag_counts.get(tag, 0)} |")
        lines.extend(
            [
                "",
                "## Highest-Error Objects",
                "",
                "| object_id | sku | stratum | mean_abs_error_sum | tags |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for object_id, score, tags, subset_row in worst_objects:
            lines.append(
                "| "
                + f"{object_id} | {subset_row.get('sku', '')} | {subset_row.get('semantic_stratum', '')} | "
                + f"{score:.4f} | {', '.join(tags) if tags else 'none'} |"
            )
    lines.extend(
        [
            "",
            "## Decision Read",
            "",
            "- This summary is a semantic calibration layer below the structural A_ready gate.",
            "- If failures cluster in a narrow mode, tighten rules for that mode instead of replacing the dataset.",
            "- If GT-only checks stay clean, the current ABO/ecommerce pool is strong enough to keep scaling before introducing new sources.",
            "",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
