#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Merge completed material-refine partial manifests and assign stable object-level IID splits.",
    )
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--split-salt", type=str, default="sf3d_material_refine_paper_stage1_v1")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def stable_unit_interval(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12 - 1)


def canonical_object_id(record: dict[str, Any]) -> str:
    return str(
        record.get("canonical_object_id")
        or record.get("object_id")
        or record.get("source_uid")
        or record.get("source_model_path")
        or ""
    )


def assign_default_split(object_id: str, *, train_ratio: float, val_ratio: float, split_salt: str) -> str:
    bucket = stable_unit_interval(f"{split_salt}:{object_id}")
    train_cut = max(0.0, min(1.0, train_ratio))
    val_cut = max(train_cut, min(1.0, train_cut + max(0.0, val_ratio)))
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def rebalance_splits(records: list[dict[str, Any]], *, split_salt: str) -> None:
    eligible_indices = [
        index
        for index, record in enumerate(records)
        if str(record.get("supervision_role") or "") == "paper_main"
        and str(record.get("target_quality_tier") or "") in {"paper_strong", "paper_pseudo"}
        and str(record.get("target_is_prior_copy")).lower() not in {"true", "1", "yes"}
    ]
    split_counts = Counter(str(records[index].get("default_split") or "train") for index in eligible_indices)
    if split_counts.get("val", 0) > 0 and split_counts.get("test", 0) > 0:
        return

    sorted_indices = sorted(
        eligible_indices,
        key=lambda index: hashlib.sha1(
            f"{split_salt}:rebalance:{canonical_object_id(records[index])}".encode("utf-8")
        ).hexdigest(),
    )
    total = len(sorted_indices)
    if total <= 0:
        return
    val_target = max(1, int(round(total * 0.10))) if total >= 10 else max(1, total // 5)
    test_target = max(1, int(round(total * 0.10))) if total >= 10 else max(1, total // 5)
    for offset, index in enumerate(sorted_indices):
        if offset < val_target:
            split = "val"
        elif offset < val_target + test_target:
            split = "test"
        else:
            split = "train"
        records[index]["default_split"] = split
        records[index]["paper_split"] = split


def main() -> None:
    args = parse_args()
    if args.train_ratio + args.val_ratio + args.test_ratio <= 0:
        raise ValueError("split ratios must have positive sum")
    ratio_sum = float(args.train_ratio + args.val_ratio + args.test_ratio)
    train_ratio = float(args.train_ratio) / ratio_sum
    val_ratio = float(args.val_ratio) / ratio_sum

    merged_by_object: dict[str, dict[str, Any]] = {}
    source_manifests = []
    skipped_records = []
    for manifest in args.manifest:
        payload = load_json(manifest)
        source_manifests.append(str(manifest.resolve()))
        skipped_records.extend(payload.get("skipped_records") or [])
        for record in payload.get("records") or []:
            object_id = canonical_object_id(record)
            if not object_id:
                continue
            existing = merged_by_object.get(object_id)
            if existing is None or int(record.get("valid_view_count") or 0) >= int(existing.get("valid_view_count") or 0):
                item = dict(record)
                split = assign_default_split(
                    object_id,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    split_salt=args.split_salt,
                )
                item["canonical_object_id"] = object_id
                item["default_split"] = split
                item["paper_split"] = split
                merged_by_object[object_id] = item

    records = sorted(merged_by_object.values(), key=lambda record: canonical_object_id(record))
    rebalance_splits(records, split_salt=args.split_salt)
    summary = {
        "records": len(records),
        "source_manifests": source_manifests,
        "default_split": dict(Counter(str(record.get("default_split") or "unknown") for record in records)),
        "paper_split": dict(Counter(str(record.get("paper_split") or "unknown") for record in records)),
        "target_quality_tier": dict(Counter(str(record.get("target_quality_tier") or "unknown") for record in records)),
        "target_source_type": dict(Counter(str(record.get("target_source_type") or "unknown") for record in records)),
        "supervision_role": dict(Counter(str(record.get("supervision_role") or "unknown") for record in records)),
        "skipped_records": len(skipped_records),
        "split_salt": args.split_salt,
    }
    payload = {
        "manifest_version": "canonical_asset_record_v1_merged_partial",
        "summary": summary,
        "records": records,
        "skipped_records": skipped_records,
    }
    write_json(args.output_manifest, payload)
    print(json.dumps({"output_manifest": str(args.output_manifest), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
