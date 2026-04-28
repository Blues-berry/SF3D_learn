from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MANIFESTS = [
    Path("output/material_refine_paper/stage1_v3_dataset_latest/stage1_v3_strict_paper_candidates.json"),
    Path("output/material_refine_paper/version1_dataset_20260426/stage1_v3_strict_paper_candidates.json"),
    Path("output/material_refine_paper/stage1_v3_dataset_20260423/stage1_v3_strict_paper_candidates.json"),
    Path("output/material_refine_r_v2_dayrun/subsets/r_v2_acceptance_128_manifest.json"),
    Path(
        "output/material_refine_pipeline_20260418T091559Z/"
        "paper_stage1_pipeline_auto_select/readiness/stage1_subset/paper_stage1_subset_manifest.json"
    ),
    Path("output/material_refine_paper/latest_dataset_check_20260421/stage1_subset_merged490/paper_stage1_subset_manifest.json"),
    Path("output/material_refine_paper/dataset_sync_check_20260422/longrun_monitor_stage1_subset_654/paper_stage1_subset_manifest.json"),
    Path("output/material_refine_longrun_stress24_hdri900_20260419T134158Z/canonical_manifest_monitor_merged.json"),
]
MATERIAL_PRIORITY = [
    "metal_dominant",
    "ceramic_glazed_lacquer",
    "glass_metal",
    "mixed_thin_boundary",
    "glossy_non_metal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build R-v2 7-day clean and hard diagnostic subsets.",
    )
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--metric-debug-csv", type=Path, default=Path("output/material_refine_r_v2_dayrun/acceptance_128_eval/metric_consistency_debug.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("output/material_refine_r_v2_7day/day3_subsets"))
    parser.add_argument("--clean-target-records", type=int, default=256)
    parser.add_argument("--clean-min-records", type=int, default=160)
    parser.add_argument("--hard-target-records", type=int, default=128)
    parser.add_argument("--preferred-identity", type=float, default=0.30)
    parser.add_argument("--require-existing-target-files", type=parse_bool, default=True)
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def to_float(value: Any, default: float = math.inf) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or payload.get("objects") or payload.get("rows") or []
    return [dict(record, source_manifest=str(path.resolve())) for record in records if isinstance(record, dict)]


def resolve_record_path(record: dict[str, Any], key: str, manifest_dir: Path) -> Path | None:
    value = record.get(key)
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    for root_key in ("bundle_root", "canonical_bundle_root", "canonical_buffer_root"):
        root_value = record.get(root_key)
        if not root_value:
            continue
        root = Path(str(root_value))
        if not root.is_absolute():
            root = manifest_dir / root
        candidate = root / path
        if candidate.exists():
            return candidate
    return manifest_dir / path


def infer_prior_source_type(record: dict[str, Any]) -> str:
    for key in ("prior_source_type", "prior_generation_mode", "prior_label"):
        value = record.get(key)
        if value:
            return str(value)
    prior_mode = str(record.get("prior_mode", "unknown") or "unknown")
    if prior_mode == "none" or not truthy(record.get("has_material_prior")):
        return "no_prior_placeholder"
    if prior_mode == "scalar_rm":
        return "input_scalar_prior"
    if prior_mode == "uv_rm":
        return "input_uv_prior"
    return "unknown"


def is_without_prior(record: dict[str, Any]) -> bool:
    return (
        str(record.get("prior_mode", "")) == "none"
        or not truthy(record.get("has_material_prior"))
        or infer_prior_source_type(record) in {"fallback_default", "no_prior_placeholder"}
    )


def clean_blockers(record: dict[str, Any], manifest_dir: Path, require_files: bool) -> list[str]:
    blockers: list[str] = []
    if str(record.get("target_source_type", "")) == "copied_from_prior":
        blockers.append("target_source_type=copied_from_prior")
    if truthy(record.get("target_is_prior_copy")):
        blockers.append("target_is_prior_copy=true")
    if not truthy(record.get("view_supervision_ready")):
        blockers.append("view_supervision_ready=false")
    for key in ("uv_target_roughness_path", "uv_target_metallic_path", "uv_target_confidence_path"):
        if not record.get(key):
            blockers.append(f"{key}=missing")
            continue
        if require_files:
            path = resolve_record_path(record, key, manifest_dir)
            if path is None or not path.exists():
                blockers.append(f"{key}=not_found")
    return blockers


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key, "unknown") or "unknown") for record in records))


def percentile(values: list[float], pct: float) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * pct
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return values[lo]
    return float(values[lo] * (hi - index) + values[hi] * (index - lo))


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("object_id") or record.get("canonical_object_id") or record.get("uid") or "")


def sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    identity = to_float(record.get("target_prior_identity"), 1.0)
    material_rank = MATERIAL_PRIORITY.index(str(record.get("material_family"))) if str(record.get("material_family")) in MATERIAL_PRIORITY else 99
    with_prior_rank = 0 if is_without_prior(record) else 1
    quality_rank = {"paper_strong": 0, "paper_pseudo": 1, "paper_weak": 2}.get(str(record.get("target_quality_tier")), 9)
    return (identity, quality_rank, material_rank, with_prior_rank, record_id(record))


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record_id(record)
        if not key:
            continue
        if key not in best or sort_key(record) < sort_key(best[key]):
            best[key] = record
    return sorted(best.values(), key=sort_key)


def append_until(selected: list[dict[str, Any]], pool: list[dict[str, Any]], target_len: int, seen: set[str]) -> None:
    for record in pool:
        key = record_id(record)
        if not key or key in seen:
            continue
        selected.append(record)
        seen.add(key)
        if len(selected) >= target_len:
            return


def assign_splits(records: list[dict[str, Any]], subset_name: str) -> list[dict[str, Any]]:
    out = []
    train_cut = int(round(len(records) * 0.70))
    val_cut = train_cut + int(round(len(records) * 0.15))
    for index, record in enumerate(records):
        item = dict(record)
        split = "train" if index < train_cut else "val" if index < val_cut else "test"
        item["default_split"] = split
        item["paper_split"] = f"{subset_name}_{split}"
        item["prior_source_type"] = infer_prior_source_type(item)
        out.append(item)
    return out


def select_clean(records: list[dict[str, Any]], target: int, preferred_identity: float) -> list[dict[str, Any]]:
    records = dedupe(records)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    low_identity = [record for record in records if to_float(record.get("target_prior_identity"), 1.0) <= preferred_identity]
    clean_pool = low_identity if low_identity else records
    for material in MATERIAL_PRIORITY:
        pool = [record for record in clean_pool if str(record.get("material_family")) == material]
        append_until(selected, pool, min(target, len(selected) + 24), seen)
    append_until(selected, [record for record in clean_pool if is_without_prior(record)], min(target, len(selected) + 64), seen)
    append_until(selected, [record for record in clean_pool if not is_without_prior(record)], min(target, len(selected) + 64), seen)
    append_until(selected, clean_pool, target, seen)
    return selected[:target]


def load_metric_debug(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    by_object: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            object_id = str(row.get("object_id", ""))
            if not object_id:
                continue
            current = by_object.get(object_id)
            gain = to_float(row.get("computed_gain_debug"), 0.0)
            if current is None or gain < to_float(current.get("computed_gain_debug"), 0.0):
                by_object[object_id] = row
    return by_object


def select_hard(records: list[dict[str, Any]], metric_debug: dict[str, dict[str, Any]], target: int) -> list[dict[str, Any]]:
    records = dedupe(records)

    def hard_score(record: dict[str, Any]) -> tuple[Any, ...]:
        metric = metric_debug.get(record_id(record), {})
        view_gain = to_float(metric.get("computed_gain_debug"), 0.0)
        identity = to_float(record.get("target_prior_identity"), 0.0)
        no_prior = 1 if is_without_prior(record) else 0
        thin = 1 if truthy(record.get("thin_boundary_flag")) else 0
        tags = str(record.get("failure_tags") or record.get("failure_tag") or "").lower()
        tag_hit = int(any(token in tags for token in ["boundary", "metal", "highlight", "confusion"]))
        return (-identity, view_gain, -no_prior, -thin, -tag_hit, record_id(record))

    return sorted(records, key=hard_score)[:target]


def summarize(records: list[dict[str, Any]], min_records: int, preferred_identity: float) -> dict[str, Any]:
    identities = [to_float(record.get("target_prior_identity"), math.nan) for record in records]
    identities = [value for value in identities if math.isfinite(value)]
    copied = sum(truthy(record.get("target_is_prior_copy")) or str(record.get("target_source_type")) == "copied_from_prior" for record in records)
    view_ready = sum(truthy(record.get("view_supervision_ready")) for record in records)
    with_prior = sum(not is_without_prior(record) for record in records)
    without_prior = len(records) - with_prior
    identity_mean = float(sum(identities) / len(identities)) if identities else None
    material_count = len(distribution(records, "material_family"))
    blockers = []
    if len(records) < min_records:
        blockers.append(f"records={len(records)} below min_records={min_records}")
    if identity_mean is None or identity_mean > preferred_identity:
        blockers.append(f"target_prior_identity_mean={identity_mean} above {preferred_identity}")
    if copied:
        blockers.append(f"copied_from_prior={copied}")
    if records and view_ready != len(records):
        blockers.append(f"view_supervision_ready_rate={view_ready / len(records):.4f} below 1.0")
    if material_count < 4:
        blockers.append(f"material_family_coverage={material_count} below 4")
    if with_prior == 0 or without_prior == 0:
        blockers.append(f"with_prior={with_prior} without_prior={without_prior}; both required")
    return {
        "records": len(records),
        "by_material_family": distribution(records, "material_family"),
        "by_source_name": distribution(records, "source_name"),
        "by_prior_source_type": distribution(records, "prior_source_type"),
        "by_prior_mode": distribution(records, "prior_mode"),
        "target_prior_identity_mean": identity_mean,
        "target_prior_identity_p50": percentile(identities, 0.50),
        "target_prior_identity_p95": percentile(identities, 0.95),
        "view_supervision_ready_rate": float(view_ready / max(len(records), 1)),
        "copied_from_prior": copied,
        "with_prior": with_prior,
        "without_prior": without_prior,
        "passed": not blockers,
        "blockers": blockers,
    }


def write_manifest(path: Path, records: list[dict[str, Any]], subset_name: str, summary: dict[str, Any], sources: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": "canonical_asset_record_v1_r_v2_7day_subset",
        "subset_name": subset_name,
        "source_manifests": sources,
        "summary": summary,
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_audit(path: Path, title: str, manifest_path: Path, summary: dict[str, Any]) -> None:
    def table(name: str) -> list[str]:
        values = summary.get(name) or {}
        return [f"- `{key}`: `{value}`" for key, value in sorted(values.items())] or ["- none"]

    lines = [
        f"# {title}",
        "",
        f"- manifest: `{manifest_path}`",
        f"- records: `{summary.get('records')}`",
        f"- passed: `{summary.get('passed')}`",
        f"- target_prior_identity_mean: `{summary.get('target_prior_identity_mean')}`",
        f"- target_prior_identity_p50: `{summary.get('target_prior_identity_p50')}`",
        f"- target_prior_identity_p95: `{summary.get('target_prior_identity_p95')}`",
        f"- view_supervision_ready_rate: `{summary.get('view_supervision_ready_rate')}`",
        f"- copied_from_prior: `{summary.get('copied_from_prior')}`",
        f"- with_prior: `{summary.get('with_prior')}`",
        f"- without_prior: `{summary.get('without_prior')}`",
        "",
        "## Blockers",
        "",
        *([f"- `{item}`" for item in summary.get("blockers", [])] or ["- none"]),
        "",
        "## by material_family",
        *table("by_material_family"),
        "",
        "## by source_name",
        *table("by_source_name"),
        "",
        "## by prior_source_type",
        *table("by_prior_source_type"),
        "",
        "## by prior_mode",
        *table("by_prior_mode"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifests = args.manifest or DEFAULT_MANIFESTS
    sources: list[str] = []
    clean_candidates: list[dict[str, Any]] = []
    hard_candidates: list[dict[str, Any]] = []
    rejected = Counter()
    for manifest in manifests:
        if not manifest.exists():
            continue
        sources.append(str(manifest.resolve()))
        manifest_dir = manifest.parent
        for record in load_records(manifest):
            record["prior_source_type"] = infer_prior_source_type(record)
            blockers = clean_blockers(record, manifest_dir, args.require_existing_target_files)
            if blockers:
                rejected.update(blockers)
            else:
                clean_candidates.append(record)
            if not truthy(record.get("target_is_prior_copy")):
                hard_candidates.append(record)

    clean = assign_splits(
        select_clean(clean_candidates, args.clean_target_records, args.preferred_identity),
        "r_v2_7day_clean",
    )
    hard = assign_splits(
        select_hard(hard_candidates, load_metric_debug(args.metric_debug_csv), args.hard_target_records),
        "r_v2_7day_hard_diagnostic",
    )
    clean_name = "clean_256_subset" if len(clean) >= args.clean_min_records else "clean_available_subset"
    clean_manifest = args.output_root / ("clean_256_manifest.json" if len(clean) >= args.clean_min_records else "clean_available_manifest.json")
    hard_manifest = args.output_root / "hard_diagnostic_128_manifest.json"
    clean_summary = summarize(clean, args.clean_min_records, args.preferred_identity)
    hard_summary = summarize(hard, 1, 1.0)
    clean_summary["available_clean_candidates"] = len(dedupe(clean_candidates))
    clean_summary["rejected_reasons"] = dict(rejected)
    hard_summary["available_hard_candidates"] = len(dedupe(hard_candidates))
    write_manifest(clean_manifest, clean, clean_name, clean_summary, sources)
    write_manifest(hard_manifest, hard, "hard_diagnostic_128_subset", hard_summary, sources)
    write_audit(args.output_root / ("clean_256_audit.md" if len(clean) >= args.clean_min_records else "clean_available_audit.md"), "R-v2 7-day Clean Subset Audit", clean_manifest, clean_summary)
    write_audit(args.output_root / "hard_diagnostic_128_audit.md", "R-v2 7-day Hard Diagnostic Subset Audit", hard_manifest, hard_summary)
    (args.output_root / "subset_build_summary.json").write_text(
        json.dumps({"clean_manifest": str(clean_manifest.resolve()), "hard_manifest": str(hard_manifest.resolve()), "clean_summary": clean_summary, "hard_summary": hard_summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"clean_summary": clean_summary, "hard_summary": hard_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
