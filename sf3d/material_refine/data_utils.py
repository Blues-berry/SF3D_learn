from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from sf3d.material_refine.types import CanonicalAssetRecordV1, parse_bool


def stable_hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def deterministic_holdout_split(
    object_id: str,
    *,
    val_ratio: float,
    test_ratio: float,
) -> str:
    val_ratio = max(0.0, min(1.0, float(val_ratio)))
    test_ratio = max(0.0, min(1.0, float(test_ratio)))
    bucket = int(stable_hash_key(object_id)[:8], 16) / 0xFFFFFFFF
    if bucket < val_ratio:
        return "val"
    if bucket < val_ratio + test_ratio:
        return "test"
    return "train"


def normalize_optional_values(values: Sequence[str] | str | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(",")]
    else:
        parts = [str(part).strip() for part in values]
    normalized = {part for part in parts if part}
    return normalized or None


def _has_explicit_holdouts(records: Sequence[CanonicalAssetRecordV1]) -> bool:
    counts = Counter(record.default_split for record in records)
    return counts.get("val", 0) > 0 or counts.get("test", 0) > 0


def select_split_records(
    records: Sequence[CanonicalAssetRecordV1],
    *,
    split: str,
    split_strategy: str = "auto",
    hash_val_ratio: float = 0.1,
    hash_test_ratio: float = 0.1,
) -> list[CanonicalAssetRecordV1]:
    if split == "all":
        return list(records)
    if split == "smoke":
        smoke_records = [
            record
            for record in records
            if parse_bool(record.metadata.get("is_smoke"))
            or str(record.metadata.get("execution_split", "")) == "smoke"
        ]
        return smoke_records or list(records[:16])

    if split_strategy not in {"auto", "manifest", "hash"}:
        raise ValueError(f"unsupported_split_strategy:{split_strategy}")

    effective_strategy = split_strategy
    if split_strategy == "auto":
        effective_strategy = "manifest" if _has_explicit_holdouts(records) else "hash"

    selected: list[CanonicalAssetRecordV1] = []
    for record in records:
        if effective_strategy == "manifest":
            record_split = record.default_split
        else:
            record_split = deterministic_holdout_split(
                record.object_id,
                val_ratio=hash_val_ratio,
                test_ratio=hash_test_ratio,
            )
        if split == "train":
            if record_split not in {"val", "test"}:
                selected.append(record)
        elif record_split == split:
            selected.append(record)
    return selected


def filter_records(
    records: Sequence[CanonicalAssetRecordV1],
    *,
    generator_ids: Sequence[str] | str | None = None,
    source_names: Sequence[str] | str | None = None,
    supervision_tiers: Sequence[str] | str | None = None,
    license_buckets: Sequence[str] | str | None = None,
    require_prior: bool | None = None,
) -> list[CanonicalAssetRecordV1]:
    allowed_generators = normalize_optional_values(generator_ids)
    allowed_sources = normalize_optional_values(source_names)
    allowed_tiers = normalize_optional_values(supervision_tiers)
    allowed_licenses = normalize_optional_values(license_buckets)

    filtered = []
    for record in records:
        if allowed_generators is not None and record.generator_id not in allowed_generators:
            continue
        source_name = str(record.metadata.get("source_name", record.generator_id))
        if allowed_sources is not None and source_name not in allowed_sources:
            continue
        if allowed_tiers is not None and record.supervision_tier not in allowed_tiers:
            continue
        if allowed_licenses is not None and record.license_bucket not in allowed_licenses:
            continue
        if require_prior is not None and bool(record.has_material_prior) != require_prior:
            continue
        filtered.append(record)
    return filtered


def summarize_records(records: Iterable[CanonicalAssetRecordV1]) -> dict[str, Any]:
    records = list(records)
    source_counts = Counter(str(record.metadata.get("source_name", record.generator_id)) for record in records)
    summary = {
        "records": len(records),
        "default_split": dict(Counter(record.default_split for record in records)),
        "generator_id": dict(Counter(record.generator_id for record in records)),
        "source_name": dict(source_counts),
        "supervision_tier": dict(Counter(record.supervision_tier for record in records)),
        "license_bucket": dict(Counter(record.license_bucket for record in records)),
        "prior_mode": dict(Counter(record.prior_mode for record in records)),
        "has_material_prior": {
            "true": sum(1 for record in records if record.has_material_prior),
            "false": sum(1 for record in records if not record.has_material_prior),
        },
    }
    return summary


def write_manifest_snapshot(
    output_path: Path,
    *,
    records: Sequence[CanonicalAssetRecordV1],
    source_manifest: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "manifest_version": "canonical_asset_record_v1",
        "source_manifest": str(Path(source_manifest).resolve()) if source_manifest else None,
        "summary": summarize_records(records),
        "records": [record.to_dict() for record in records],
    }
    if metadata:
        payload.update(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
