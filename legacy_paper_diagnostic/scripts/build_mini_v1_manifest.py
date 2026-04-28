#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DOCS_ROOT = PROJECT_ROOT / "docs" / "neural_gaffer_dataset_audit"

DEFAULT_POOL_CSV = Path(
    DOCS_ROOT / "asset_supervision_miniv1_candidate_pool_500.csv"
)
DEFAULT_SEMANTIC_SUBSET_CSV = Path(
    DOCS_ROOT / "asset_supervision_semantic_validation_subset_24.csv"
)
DEFAULT_SEMANTIC_METRICS_JSON = Path(
    "/home/ubuntu/ssd_work/projects/stable-fast-3d/output/abo_material_probe_semantic24/metrics.json"
)
DEFAULT_OUTPUT_CSV = DOCS_ROOT / "mini_v1_manifest.csv"
DEFAULT_OUTPUT_JSON = DOCS_ROOT / "mini_v1_manifest.json"
DEFAULT_OUTPUT_MD = DOCS_ROOT / "mini_v1_manifest_summary.md"

MANIFEST_VERSION = "mini_v1_abo_ecommerce_v0.1"
POOL_NAME = "mini_v1_candidate_pool"
REVIEW_SPLIT_NAME = "semantic_review_split"

ROUGH_STD_THRESHOLD = 0.03
ROUGH_SPAN_THRESHOLD = 0.08
METAL_STD_THRESHOLD = 0.03
METAL_SPAN_THRESHOLD = 0.08
METALLIC_NEAR_ZERO_THRESHOLD = 0.05
LOW_VARIATION_ROUGH_STD_THRESHOLD = 0.04
LOW_VARIATION_METAL_STD_THRESHOLD = 0.04


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the mini-v1 ABO/ecommerce manifest with semantic review flags and stable splits."
    )
    parser.add_argument("--pool-csv", type=Path, default=DEFAULT_POOL_CSV)
    parser.add_argument("--semantic-subset-csv", type=Path, default=DEFAULT_SEMANTIC_SUBSET_CSV)
    parser.add_argument("--semantic-metrics-json", type=Path, default=DEFAULT_SEMANTIC_METRICS_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "rows" in payload:
        return payload["rows"]
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unsupported metrics payload type in {path}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bool_string(value: bool) -> str:
    return "true" if value else "false"


def derive_sku(row: dict) -> str:
    model_path = row.get("source_model_path", "")
    if not model_path:
        return ""
    return Path(model_path).stem


def stable_hash_key(object_id: str) -> str:
    return hashlib.sha1(object_id.encode("utf-8")).hexdigest()


def semantic_stats_by_object(metrics_rows: list[dict]) -> dict[str, dict[str, float]]:
    per_object: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "rough_std": [],
            "metal_std": [],
            "metal_mean": [],
            "rough_span": [],
            "metal_span": [],
        }
    )
    for row in metrics_rows:
        object_id = str(row["object_id"])
        per_object[object_id]["rough_std"].append(float(row["gt_roughness_std"]))
        per_object[object_id]["metal_std"].append(float(row["gt_metallic_std"]))
        per_object[object_id]["metal_mean"].append(float(row["gt_metallic_mean"]))
        per_object[object_id]["rough_span"].append(
            float(row["gt_roughness_p90"]) - float(row["gt_roughness_p10"])
        )
        per_object[object_id]["metal_span"].append(
            float(row["gt_metallic_p90"]) - float(row["gt_metallic_p10"])
        )
    reduced: dict[str, dict[str, float]] = {}
    for object_id, stats in per_object.items():
        reduced[object_id] = {
            "avg_gt_roughness_std": mean(stats["rough_std"]),
            "avg_gt_metallic_std": mean(stats["metal_std"]),
            "avg_gt_metallic_mean": mean(stats["metal_mean"]),
            "avg_gt_roughness_span": mean(stats["rough_span"]),
            "avg_gt_metallic_span": mean(stats["metal_span"]),
        }
    return reduced


def build_review_flags(stats: dict[str, float]) -> list[str]:
    flags: list[str] = []
    if (
        stats["avg_gt_roughness_std"] < ROUGH_STD_THRESHOLD
        and stats["avg_gt_roughness_span"] < ROUGH_SPAN_THRESHOLD
    ):
        flags.append("constant_like_roughness_review")
    if (
        stats["avg_gt_metallic_std"] < METAL_STD_THRESHOLD
        and stats["avg_gt_metallic_span"] < METAL_SPAN_THRESHOLD
        and stats["avg_gt_metallic_mean"] < METALLIC_NEAR_ZERO_THRESHOLD
    ):
        flags.append("metallic_near_zero_review")
    if (
        stats["avg_gt_roughness_std"] < LOW_VARIATION_ROUGH_STD_THRESHOLD
        and stats["avg_gt_metallic_std"] < LOW_VARIATION_METAL_STD_THRESHOLD
    ):
        flags.append("low_semantic_variation")
    return flags


def semantic_annotations(
    subset_rows: list[dict], metrics_rows: list[dict]
) -> tuple[dict[str, dict], dict[str, int]]:
    stats_by_object = semantic_stats_by_object(metrics_rows)
    annotations: dict[str, dict] = {}
    review_counter: Counter = Counter()
    for row in subset_rows:
        object_id = str(row["object_id"])
        stats = stats_by_object[object_id]
        review_flags = build_review_flags(stats)
        if review_flags:
            review_flags = ["semantic_probe_needed"] + review_flags
        annotations[object_id] = {
            "semantic_probe_sampled": True,
            "semantic_probe_status": "review_needed" if review_flags else "sampled_clean",
            "semantic_probe_needed": bool(review_flags),
            "semantic_stratum": row.get("semantic_stratum", ""),
            "selection_reason": row.get("selection_reason", ""),
            "sku": row.get("sku", ""),
            "review_flags": review_flags,
            **stats,
        }
        for flag in review_flags:
            review_counter[flag] += 1
    return annotations, dict(review_counter)


def target_holdout_count(stratum_size: int, ratio: float) -> int:
    if stratum_size < 3:
        return 0
    return max(1, round(stratum_size * ratio))


def build_split_map(non_review_rows: list[dict], val_ratio: float, test_ratio: float) -> dict[str, str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in non_review_rows:
        grouped[str(row["material_slot_count"])].append(row)

    split_map: dict[str, str] = {}
    for slot_count, rows in grouped.items():
        ranked = sorted(rows, key=lambda row: stable_hash_key(str(row["object_id"])))
        val_count = target_holdout_count(len(ranked), val_ratio)
        test_count = target_holdout_count(len(ranked), test_ratio)
        if val_count + test_count >= len(ranked):
            overflow = val_count + test_count - len(ranked) + 1
            while overflow > 0 and test_count > 0:
                test_count -= 1
                overflow -= 1
            while overflow > 0 and val_count > 0:
                val_count -= 1
                overflow -= 1

        for index, row in enumerate(ranked):
            object_id = str(row["object_id"])
            if index < val_count:
                split_map[object_id] = "val"
            elif index < val_count + test_count:
                split_map[object_id] = "test"
            else:
                split_map[object_id] = "train"
    return split_map


def serialize_record(row: dict, annotation: dict | None, split: str) -> dict:
    review_flags = list(annotation["review_flags"]) if annotation else []
    sampled = bool(annotation)
    semantic_probe_status = annotation["semantic_probe_status"] if annotation else "not_sampled"
    semantic_probe_needed = bool(annotation["semantic_probe_needed"]) if annotation else False
    sku = annotation["sku"] if annotation and annotation.get("sku") else derive_sku(row)
    review_priority = "high" if any(
        flag in {"constant_like_roughness_review", "low_semantic_variation"}
        for flag in review_flags
    ) else ("medium" if review_flags else "none")
    review_status = "needs_review" if review_flags else "none"
    split_reason = (
        "review_flagged_train_keep_policy"
        if review_flags and split == "train"
        else "deterministic_material_slot_stratified_holdout"
    )

    record = {
        "manifest_version": MANIFEST_VERSION,
        "pool_name": POOL_NAME,
        "object_id": row["object_id"],
        "sku": sku,
        "subset": row["subset"],
        "source": row["source"],
        "source_uid": row["source_uid"],
        "source_model_path": row["source_model_path"],
        "texture_root": row["texture_root"],
        "format": row["format"],
        "material_slot_count": int(row["material_slot_count"]),
        "audit_status": row["audit_status"],
        "notes": row.get("notes", ""),
        "auditor": row.get("auditor", ""),
        "audit_time": row.get("audit_time", ""),
        "semantic_probe_sampled": sampled,
        "semantic_probe_status": semantic_probe_status,
        "semantic_probe_needed": semantic_probe_needed,
        "semantic_stratum": annotation.get("semantic_stratum", "") if annotation else "",
        "selection_reason": annotation.get("selection_reason", "") if annotation else "",
        "review_flags": review_flags,
        "review_flag_count": len(review_flags),
        "review_status": review_status,
        "review_priority": review_priority,
        "default_split": split,
        "pool_role": "main_supervision",
        "include_in_default_train": split == "train",
        "include_in_default_eval": split in {"val", "test"},
        "include_in_main_supervision": True,
        "split_reason": split_reason,
        "avg_gt_roughness_std": annotation.get("avg_gt_roughness_std") if annotation else None,
        "avg_gt_roughness_span": annotation.get("avg_gt_roughness_span") if annotation else None,
        "avg_gt_metallic_std": annotation.get("avg_gt_metallic_std") if annotation else None,
        "avg_gt_metallic_span": annotation.get("avg_gt_metallic_span") if annotation else None,
        "avg_gt_metallic_mean": annotation.get("avg_gt_metallic_mean") if annotation else None,
    }
    return record


def csv_record(record: dict) -> dict:
    out = dict(record)
    out["review_flags"] = ";".join(record["review_flags"])
    for key in [
        "semantic_probe_sampled",
        "semantic_probe_needed",
        "include_in_default_train",
        "include_in_default_eval",
        "include_in_main_supervision",
    ]:
        out[key] = bool_string(bool(record[key]))
    for key in [
        "avg_gt_roughness_std",
        "avg_gt_roughness_span",
        "avg_gt_metallic_std",
        "avg_gt_metallic_span",
        "avg_gt_metallic_mean",
    ]:
        value = out[key]
        out[key] = "" if value is None else f"{float(value):.6f}"
    return out


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(csv_record(records[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(csv_record(record))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    records: list[dict],
    review_flag_counts: dict[str, int],
) -> None:
    split_counts = Counter(record["default_split"] for record in records)
    role_counts = Counter(record["pool_role"] for record in records)
    review_counts = Counter(record["review_status"] for record in records)
    review_priority_counts = Counter(record["review_priority"] for record in records)
    slot_by_split: dict[str, Counter] = defaultdict(Counter)
    flagged = [record for record in records if record["review_status"] == "needs_review"]
    for record in records:
        slot_by_split[record["default_split"]][record["material_slot_count"]] += 1

    lines = [
        "# mini_v1 Manifest Summary",
        "",
        f"- manifest_version: {MANIFEST_VERSION}",
        f"- pool_name: {POOL_NAME}",
        f"- generated_from: {args.pool_csv}",
        f"- semantic_subset: {args.semantic_subset_csv}",
        f"- semantic_metrics: {args.semantic_metrics_json}",
        f"- output_csv: {args.output_csv}",
        f"- output_json: {args.output_json}",
        "",
        "## Decision Lock",
        "",
        "- source_pool: ABO/ecommerce only",
        "- structural_gate: 500/500 A_ready",
        "- source_gate: 500/500 abo_selected",
        "- policy: keep the 500-object pool intact, add soft review flags, avoid automatic demotion in this stage",
        "",
        "## Split Counts",
        "",
        f"- train: {split_counts['train']}",
        f"- val: {split_counts['val']}",
        f"- test: {split_counts['test']}",
        f"- main_supervision_total: {role_counts['main_supervision']}",
        f"- review_flagged_total: {review_counts['needs_review']}",
        f"- review_flagged_train: {sum(1 for record in flagged if record['default_split'] == 'train')}",
        f"- review_high: {review_priority_counts['high']}",
        f"- review_medium: {review_priority_counts['medium']}",
        "",
        "## Review Flag Counts",
        "",
        f"- semantic_probe_needed: {review_flag_counts.get('semantic_probe_needed', 0)}",
        f"- constant_like_roughness_review: {review_flag_counts.get('constant_like_roughness_review', 0)}",
        f"- metallic_near_zero_review: {review_flag_counts.get('metallic_near_zero_review', 0)}",
        f"- low_semantic_variation: {review_flag_counts.get('low_semantic_variation', 0)}",
        "",
        "## Split Stratification",
        "",
        f"- train material_slot_count: {dict(slot_by_split['train'])}",
        f"- val material_slot_count: {dict(slot_by_split['val'])}",
        f"- test material_slot_count: {dict(slot_by_split['test'])}",
        "",
        "## Semantic Review Objects",
        "",
        "| object_id | sku | semantic_stratum | review_priority | review_flags |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in flagged:
        lines.append(
            "| "
            + f"{record['object_id']} | {record['sku']} | {record['semantic_stratum'] or 'n/a'} | {record['review_priority']} | "
            + f"{', '.join(record['review_flags'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- flagged objects stay inside the main supervision manifest and keep review labels instead of being split out.",
            "- `val` and `test` are deterministic holdouts built from the non-flagged pool so existing holdouts stay stable.",
            "- `metallic_near_zero_review` remains a review-only soft flag, not a rejection or demotion signal.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    pool_rows = read_csv_rows(args.pool_csv)
    subset_rows = read_csv_rows(args.semantic_subset_csv)
    metrics_rows = read_json_rows(args.semantic_metrics_json)

    if len(pool_rows) != 500:
        raise ValueError(f"Expected 500 pool rows, found {len(pool_rows)}")
    if any(row["audit_status"] != "A_ready" for row in pool_rows):
        raise ValueError("mini-v1 builder expects every pool row to be A_ready")
    if any(row["source"] != "abo_selected" for row in pool_rows):
        raise ValueError("mini-v1 builder expects every pool row to come from abo_selected")

    annotations, review_flag_counts = semantic_annotations(subset_rows, metrics_rows)
    flagged_ids = {
        object_id
        for object_id, annotation in annotations.items()
        if annotation["semantic_probe_needed"]
    }
    non_review_rows = [row for row in pool_rows if row["object_id"] not in flagged_ids]
    split_map = build_split_map(non_review_rows, args.val_ratio, args.test_ratio)

    records: list[dict] = []
    for row in pool_rows:
        object_id = str(row["object_id"])
        annotation = annotations.get(object_id)
        split = "train" if object_id in flagged_ids else split_map[object_id]
        records.append(serialize_record(row, annotation, split))

    records.sort(key=lambda record: record["object_id"])
    write_csv(args.output_csv, records)

    split_counts = Counter(record["default_split"] for record in records)
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "pool_name": POOL_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "pool_csv": str(args.pool_csv),
            "semantic_subset_csv": str(args.semantic_subset_csv),
            "semantic_metrics_json": str(args.semantic_metrics_json),
        },
        "source_assertions": {
            "expected_source": "abo_selected",
            "expected_audit_status": "A_ready",
            "pool_size": len(pool_rows),
        },
        "semantic_review_thresholds": {
            "constant_like_roughness": {
                "avg_gt_roughness_std_lt": ROUGH_STD_THRESHOLD,
                "avg_gt_roughness_span_lt": ROUGH_SPAN_THRESHOLD,
            },
            "metallic_near_zero_review": {
                "avg_gt_metallic_std_lt": METAL_STD_THRESHOLD,
                "avg_gt_metallic_span_lt": METAL_SPAN_THRESHOLD,
                "avg_gt_metallic_mean_lt": METALLIC_NEAR_ZERO_THRESHOLD,
            },
            "low_semantic_variation": {
                "avg_gt_roughness_std_lt": LOW_VARIATION_ROUGH_STD_THRESHOLD,
                "avg_gt_metallic_std_lt": LOW_VARIATION_METAL_STD_THRESHOLD,
            },
        },
        "split_strategy": {
            "type": "deterministic_material_slot_stratified_holdout_with_review_labels_kept_in_train",
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "semantic_review_policy": "flagged objects remain in the candidate pool and stay in the default train split with review labels",
        },
        "counts": {
            "train": split_counts["train"],
            "val": split_counts["val"],
            "test": split_counts["test"],
            "semantic_probe_needed": review_flag_counts.get("semantic_probe_needed", 0),
            "constant_like_roughness_review": review_flag_counts.get(
                "constant_like_roughness_review", 0
            ),
            "metallic_near_zero_review": review_flag_counts.get(
                "metallic_near_zero_review", 0
            ),
            "low_semantic_variation": review_flag_counts.get("low_semantic_variation", 0),
        },
        "records": records,
    }
    write_json(args.output_json, payload)
    write_summary(args.output_md, args=args, records=records, review_flag_counts=review_flag_counts)

    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
