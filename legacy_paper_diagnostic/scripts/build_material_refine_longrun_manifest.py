#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MATERIAL_FAMILIES = {
    "metal_dominant",
    "ceramic_glazed_lacquer",
    "glass_metal",
    "mixed_thin_boundary",
    "glossy_non_metal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build a license-aware, object-split-safe manifest for long-running material-refine dataset production.",
    )
    parser.add_argument("--input-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-shard-dir", type=Path, default=None)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0, help="0 keeps all eligible records.")
    parser.add_argument("--paper-main-sources", type=str, default="ABO_locked_core")
    parser.add_argument("--auxiliary-sources", type=str, default="")
    parser.add_argument("--priority-material-families", type=str, default="")
    parser.add_argument("--target-material-family-ratios", type=str, default="")
    parser.add_argument("--paper-frontload-records", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.88)
    parser.add_argument("--val-ratio", type=float, default=0.06)
    parser.add_argument("--test-ratio", type=float, default=0.06)
    parser.add_argument("--prefer-paper-main-first", action="store_true")
    parser.add_argument(
        "--interleave-selection-keys",
        type=str,
        default="",
        help="Comma-separated keys used to round-robin selected records before sharding, e.g. material_family,source_name.",
    )
    return parser.parse_args()


def parse_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def canonical_material_family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text or text in {"none", "null", "unknown", "pending_abo_semantic_classification"}:
        return ""
    aliases = {
        "metal": "metal_dominant",
        "metallic": "metal_dominant",
        "metal_dominant": "metal_dominant",
        "ceramic": "ceramic_glazed_lacquer",
        "ceramic_glazed": "ceramic_glazed_lacquer",
        "glazed": "ceramic_glazed_lacquer",
        "lacquer": "ceramic_glazed_lacquer",
        "glass": "glass_metal",
        "glass_metal": "glass_metal",
        "mixed": "mixed_thin_boundary",
        "mixed_thin_boundary": "mixed_thin_boundary",
        "thin_boundary": "mixed_thin_boundary",
        "glossy": "glossy_non_metal",
        "glossy_non_metal": "glossy_non_metal",
        "nonmetal": "glossy_non_metal",
    }
    if text in aliases:
        return aliases[text]
    if "glass" in text:
        return "glass_metal"
    if any(token in text for token in ("ceramic", "glazed", "lacquer", "porcelain", "marble")):
        return "ceramic_glazed_lacquer"
    if any(token in text for token in ("metal", "metallic", "chrome", "steel", "brass", "copper", "aluminum", "aluminium")):
        return "metal_dominant"
    if any(token in text for token in ("thin", "boundary", "mixed", "frame", "wire", "lamp", "lighting")):
        return "mixed_thin_boundary"
    if any(token in text for token in ("gloss", "plastic", "leather", "wood", "paint")):
        return "glossy_non_metal"
    return ""


def parse_ratio_csv(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    ratios: dict[str, float] = {}
    for item in str(value).split(","):
        if not item.strip() or "=" not in item:
            continue
        key, raw = item.split("=", 1)
        family = canonical_material_family(key)
        if family not in MATERIAL_FAMILIES:
            continue
        try:
            ratio = float(raw)
        except ValueError:
            continue
        if ratio > 0.0:
            ratios[family] = ratio
    total = sum(ratios.values())
    if total > 0.0:
        ratios = {key: value / total for key, value in ratios.items()}
    return ratios


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def stable_unit(text: str) -> float:
    return stable_hash(text) / float(16**12 - 1)


def assign_split(object_id: str, train_ratio: float, val_ratio: float) -> str:
    value = stable_unit(object_id)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def failure_tags_for_family(family: str, *, thin_boundary: bool) -> list[str]:
    tags: set[str] = set()
    if family in {"metal_dominant", "glass_metal"}:
        tags.update({"metal_nonmetal_confusion", "local_highlight_misread"})
    if family in {"glass_metal", "mixed_thin_boundary"} or thin_boundary:
        tags.update({"boundary_bleed", "local_highlight_misread"})
    if family == "ceramic_glazed_lacquer":
        tags.add("local_highlight_misread")
    return sorted(tags)


def infer_distribution_tags(record: dict[str, Any]) -> tuple[str, bool, list[str]]:
    explicit_family = (
        canonical_material_family(record.get("material_family"))
        or canonical_material_family(record.get("highlight_material_class"))
        or canonical_material_family(record.get("material_bucket"))
    )
    text = " ".join(
        str(record.get(key) or "")
        for key in (
            "material_family",
            "highlight_material_class",
            "material_class_source",
            "highlight_priority_class",
            "material_bucket",
            "category_bucket",
            "label",
            "name",
            "category",
            "super_category",
            "material",
            "notes",
            "source_model_path",
            "source_texture_root",
        )
    ).lower()
    thin_boundary = False
    if explicit_family:
        family = explicit_family
    elif "unknown_pending_second_pass" in text or "pending_material_probe" in text:
        family = "unknown_pending_second_pass"
    elif "glass" in text:
        family = "glass_metal"
    elif any(token in text for token in ("ceramic", "marble", "granite", "lacquer", "glazed", "porcelain")):
        family = "ceramic_glazed_lacquer"
    elif any(token in text for token in ("steel", "chrome", "brass", "copper", "aluminum", "aluminium", "gold foil")) or bool(
        re.search(r"(?:material\s*=\s*metal|\bmetal\b|\bmetal frame\b|\bmetallic object\b)", text)
    ):
        family = "metal_dominant"
    elif any(token in text for token in ("lamp", "lighting", "pendant", "shelf", "rack", "frame", "wire", "handle", "chair", "table")):
        family = "mixed_thin_boundary"
        thin_boundary = True
    elif any(token in text for token in ("leather", "plastic", "wood", "cloth", "sofa")):
        family = "glossy_non_metal"
    elif "objaverse" in str(record.get("source_name") or record.get("generator_id") or "").lower():
        family = "unknown_pending_second_pass"
    else:
        family = "glossy_non_metal"
    if any(token in text for token in ("thin", "frame", "wire", "handle", "lamp", "pendant", "rack")):
        thin_boundary = True
    if family == "mixed_thin_boundary":
        thin_boundary = True
    return family, thin_boundary, failure_tags_for_family(family, thin_boundary=thin_boundary)


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            if not isinstance(record, dict):
                continue
            object_id = str(record.get("object_id") or record.get("source_uid") or "")
            if not object_id or object_id in seen:
                continue
            source_model_path = Path(str(record.get("source_model_path") or ""))
            if not source_model_path.exists():
                continue
            if not bool(record.get("source_asset_available", True)):
                continue
            seen.add(object_id)
            records.append(dict(record))
    return records


def normalize_record(
    record: dict[str, Any],
    *,
    paper_main_sources: set[str],
    auxiliary_sources: set[str],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, Any]:
    item = dict(record)
    object_id = str(item.get("object_id") or item.get("source_uid"))
    source_name = str(item.get("source_name") or item.get("generator_id") or "unknown")
    item["object_id"] = object_id
    item["include_in_full"] = True
    item["include_in_smoke"] = False
    item["default_split"] = assign_split(object_id, train_ratio, val_ratio)
    if source_name in paper_main_sources:
        item["supervision_role"] = "paper_main"
    elif not auxiliary_sources or source_name in auxiliary_sources:
        item["supervision_role"] = "auxiliary_upgrade_queue"
    else:
        item["supervision_role"] = str(item.get("supervision_role") or "unknown")
    item["target_quality_tier"] = ""
    item["target_is_prior_copy"] = False
    item["target_source_type"] = ""
    item["paper_split"] = ""
    item["view_supervision_ready"] = False
    item["valid_view_count"] = 0
    material_family, thin_boundary, failure_tags = infer_distribution_tags(item)
    item["material_family"] = material_family
    item["thin_boundary_flag"] = thin_boundary
    item["failure_tags"] = failure_tags
    item["sampling_bucket"] = f"{source_name}|{material_family}|{'no_prior' if not bool(item.get('has_material_prior')) else 'with_prior'}"
    return item


def stable_record_key(item: dict[str, Any], material_rank: dict[str, int]) -> tuple[int, int]:
    return (
        material_rank.get(str(item.get("material_family") or "unknown"), len(material_rank)),
        stable_hash(str(item["object_id"])),
    )


def select_quota_balanced_records(
    records: list[dict[str, Any]],
    *,
    max_records: int,
    target_ratios: dict[str, float],
    material_rank: dict[str, int],
    paper_frontload_records: int,
) -> list[dict[str, Any]]:
    if max_records <= 0:
        return sorted(records, key=lambda item: stable_record_key(item, material_rank))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    if paper_frontload_records > 0:
        paper_records = [item for item in records if item.get("supervision_role") == "paper_main"]
        paper_records.sort(key=lambda item: stable_record_key(item, material_rank))
        for item in paper_records[:paper_frontload_records]:
            selected.append(item)
            selected_ids.add(str(item["object_id"]))
            if len(selected) >= max_records:
                return selected

    remaining_slots = max_records - len(selected)
    remaining = [item for item in records if str(item["object_id"]) not in selected_ids]
    if remaining_slots <= 0:
        return selected
    if not target_ratios:
        selected.extend(sorted(remaining, key=lambda item: stable_record_key(item, material_rank))[:remaining_slots])
        return selected

    groups: dict[str, list[dict[str, Any]]] = {family: [] for family in target_ratios}
    overflow: list[dict[str, Any]] = []
    for item in remaining:
        family = str(item.get("material_family") or "unknown")
        if family in groups:
            groups[family].append(item)
        else:
            overflow.append(item)
    for bucket in groups.values():
        bucket.sort(key=lambda item: stable_hash(str(item["object_id"])))
    overflow.sort(key=lambda item: stable_record_key(item, material_rank))

    raw_quotas = {family: ratio * remaining_slots for family, ratio in target_ratios.items()}
    quotas = {family: int(raw) for family, raw in raw_quotas.items()}
    remainder = remaining_slots - sum(quotas.values())
    fractional = sorted(
        raw_quotas.items(),
        key=lambda pair: (pair[1] - int(pair[1]), -material_rank.get(pair[0], 999)),
        reverse=True,
    )
    for family, _raw in fractional[:remainder]:
        quotas[family] += 1

    for family in sorted(target_ratios, key=lambda fam: material_rank.get(fam, 999)):
        take = min(quotas.get(family, 0), len(groups.get(family, [])))
        for item in groups.get(family, [])[:take]:
            selected.append(item)
            selected_ids.add(str(item["object_id"]))

    if len(selected) < max_records:
        fill_pool: list[dict[str, Any]] = []
        for bucket in groups.values():
            fill_pool.extend(item for item in bucket if str(item["object_id"]) not in selected_ids)
        fill_pool.extend(item for item in overflow if str(item["object_id"]) not in selected_ids)
        fill_pool.sort(key=lambda item: stable_record_key(item, material_rank))
        selected.extend(fill_pool[: max_records - len(selected)])
    return selected[:max_records]


def interleave_records(records: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    if not records or not key_fields:
        return records
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for item in records:
        key = tuple(str(item.get(field) or "unknown") for field in key_fields)
        buckets.setdefault(key, []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: stable_hash(str(item["object_id"])))

    ordered_keys = sorted(
        buckets,
        key=lambda key: (
            -len(buckets[key]),
            stable_hash("|".join(key)),
        ),
    )
    interleaved: list[dict[str, Any]] = []
    while ordered_keys:
        next_keys: list[tuple[str, ...]] = []
        for key in ordered_keys:
            bucket = buckets[key]
            if bucket:
                interleaved.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        ordered_keys = next_keys
    return interleaved


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paper_main_sources = parse_csv(args.paper_main_sources)
    auxiliary_sources = parse_csv(args.auxiliary_sources)
    priority_material_families = [item.strip() for item in str(args.priority_material_families).split(",") if item.strip()]
    material_rank = {family: idx for idx, family in enumerate(priority_material_families)}
    target_material_ratios = parse_ratio_csv(args.target_material_family_ratios)
    train_ratio = max(0.0, min(1.0, float(args.train_ratio)))
    val_ratio = max(0.0, min(1.0 - train_ratio, float(args.val_ratio)))
    records = [
        normalize_record(
            record,
            paper_main_sources=paper_main_sources,
            auxiliary_sources=auxiliary_sources,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        for record in load_records(args.input_manifest)
    ]
    if args.prefer_paper_main_first:
        records.sort(
            key=lambda item: (
                0 if item.get("supervision_role") == "paper_main" else 1,
                material_rank.get(str(item.get("material_family") or "unknown"), len(material_rank)),
                stable_hash(str(item["object_id"])),
            )
        )
    else:
        records = select_quota_balanced_records(
            records,
            max_records=int(args.max_records),
            target_ratios=target_material_ratios,
            material_rank=material_rank,
            paper_frontload_records=max(0, int(args.paper_frontload_records)),
        )
    if args.max_records > 0:
        records = records[: int(args.max_records)]
    interleave_keys = [item.strip() for item in str(args.interleave_selection_keys).split(",") if item.strip()]
    records = interleave_records(records, interleave_keys)

    payload = {
        "manifest_version": "canonical_asset_record_v1_longrun_input",
        "source_manifests": [str(path.resolve()) for path in args.input_manifest],
        "split_policy": {
            "type": "object_id_sha1",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": max(0.0, 1.0 - train_ratio - val_ratio),
        },
        "selection_policy": {
            "prefer_paper_main_first": bool(args.prefer_paper_main_first),
            "paper_frontload_records": max(0, int(args.paper_frontload_records)),
            "target_material_family_ratios": target_material_ratios,
            "interleave_selection_keys": interleave_keys,
        },
        "counts": {
            "records": len(records),
            "source_name": dict(Counter(str(record.get("source_name") or "unknown") for record in records)),
            "supervision_role": dict(Counter(str(record.get("supervision_role") or "unknown") for record in records)),
            "default_split": dict(Counter(str(record.get("default_split") or "unknown") for record in records)),
            "license_bucket": dict(Counter(str(record.get("license_bucket") or "unknown") for record in records)),
            "material_family": dict(Counter(str(record.get("material_family") or "unknown") for record in records)),
            "has_material_prior": {
                "true": sum(bool(record.get("has_material_prior")) for record in records),
                "false": sum(not bool(record.get("has_material_prior")) for record in records),
            },
        },
        "records": records,
    }
    write_json(args.output_manifest, payload)

    if args.output_shard_dir is not None and args.shards >= 1:
        args.output_shard_dir.mkdir(parents=True, exist_ok=True)
        for shard_idx in range(args.shards):
            shard_records = [
                record
                for record in records
                if stable_hash(str(record["object_id"])) % int(args.shards) == shard_idx
            ]
            shard_payload = dict(payload)
            shard_payload["manifest_version"] = "canonical_asset_record_v1_longrun_input_shard"
            shard_payload["shard"] = {"index": shard_idx, "count": int(args.shards)}
            shard_payload["counts"] = dict(payload["counts"], records=len(shard_records))
            shard_payload["records"] = shard_records
            write_json(args.output_shard_dir / f"longrun_input_shard_{shard_idx:02d}.json", shard_payload)

    print(
        json.dumps(
            {
                "output_manifest": str(args.output_manifest),
                "records": len(records),
                "source_name": payload["counts"]["source_name"],
                "supervision_role": payload["counts"]["supervision_role"],
                "default_split": payload["counts"]["default_split"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
