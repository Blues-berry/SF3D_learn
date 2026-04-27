from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MANIFESTS = [
    Path("output/material_refine_paper/stage1_v3_dataset_latest/stage1_v3_strict_paper_candidates.json"),
    Path("output/material_refine_paper/version1_dataset_20260426/stage1_v3_strict_paper_candidates.json"),
    Path("output/material_refine_paper/stage1_v3_dataset_20260423/stage1_v3_strict_paper_candidates.json"),
    Path(
        "output/material_refine_pipeline_20260418T091559Z/"
        "paper_stage1_pipeline_auto_select/readiness/stage1_subset/paper_stage1_subset_manifest.json"
    ),
    Path("output/material_refine_paper/latest_dataset_check_20260421/stage1_subset_merged490/paper_stage1_subset_manifest.json"),
    Path("output/material_refine_paper/dataset_sync_check_20260422/longrun_monitor_stage1_subset_654/paper_stage1_subset_manifest.json"),
    Path("output/material_refine_longrun_stress24_hdri900_20260419T134158Z/canonical_manifest_monitor_merged.json"),
]

TARGET_QUALITY_PRIORITY = {"paper_strong": 0, "paper_pseudo": 1, "paper_weak": 2, "smoke_only": 5}
MATERIAL_PRIORITY = [
    "metal_dominant",
    "ceramic_glazed_lacquer",
    "glass_metal",
    "mixed_thin_boundary",
    "glossy_non_metal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean R-v2 one-day acceptance subset without touching G/C data production.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, default=Path("output/material_refine_r_v2_dayrun/subsets"))
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--target-records", type=int, default=128)
    parser.add_argument("--min-records", type=int, default=96)
    parser.add_argument("--max-target-prior-identity-preferred", type=float, default=0.30)
    parser.add_argument("--max-target-prior-identity-diagnostic", type=float, default=0.90)
    parser.add_argument("--require-existing-target-files", type=parse_bool, default=True)
    parser.add_argument("--train-records", type=int, default=80)
    parser.add_argument("--val-records", type=int, default=24)
    parser.add_argument("--test-records", type=int, default=24)
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid_bool:{value}")


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or payload.get("objects") or payload.get("rows") or []
    if not isinstance(records, list):
        raise TypeError(f"unsupported_manifest_records:{path}")
    return [dict(record, source_manifest=str(path.resolve())) for record in records if isinstance(record, dict)]


def to_float(value: Any, default: float = math.inf) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key, "unknown") or "unknown") for record in records))


def resolve_record_path(record: dict[str, Any], key: str, manifest_dir: Path) -> Path | None:
    value = record.get(key)
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    bundle_root = record.get("bundle_root") or record.get("canonical_bundle_root") or record.get("canonical_buffer_root")
    if bundle_root:
        bundle = Path(str(bundle_root))
        if not bundle.is_absolute():
            bundle = manifest_dir / bundle
        candidate = bundle / path
        if candidate.exists():
            return candidate
    return manifest_dir / path


def infer_prior_source_type(record: dict[str, Any]) -> str:
    for key in ("prior_source_type", "prior_generation_mode", "prior_label"):
        value = record.get(key)
        if value:
            return str(value)
    prior_mode = str(record.get("prior_mode", "unknown") or "unknown")
    has_prior = truthy(record.get("has_material_prior"))
    if prior_mode == "none" or not has_prior:
        return "no_prior_placeholder"
    if prior_mode == "scalar_rm":
        return "input_scalar_prior"
    if prior_mode == "uv_rm":
        return "input_uv_prior"
    return "unknown"


def is_without_prior(record: dict[str, Any]) -> bool:
    prior_mode = str(record.get("prior_mode", "") or "")
    prior_source = infer_prior_source_type(record)
    return (
        not truthy(record.get("has_material_prior"))
        or prior_mode == "none"
        or prior_source in {"fallback_default", "no_prior_placeholder"}
    )


def is_valid_clean_record(
    record: dict[str, Any],
    *,
    manifest_dir: Path,
    require_existing_target_files: bool,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if str(record.get("target_source_type", "unknown")) == "copied_from_prior":
        blockers.append("target_source_type=copied_from_prior")
    if truthy(record.get("target_is_prior_copy")):
        blockers.append("target_is_prior_copy=true")
    for key in ("uv_target_roughness_path", "uv_target_metallic_path", "uv_target_confidence_path"):
        if not record.get(key):
            blockers.append(f"{key}=missing")
            continue
        if require_existing_target_files:
            resolved = resolve_record_path(record, key, manifest_dir)
            if resolved is None or not resolved.exists():
                blockers.append(f"{key}=not_found")
    return not blockers, blockers


def record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    identity = to_float(record.get("target_prior_identity"), default=1.0)
    quality = TARGET_QUALITY_PRIORITY.get(str(record.get("target_quality_tier", "")), 10)
    view_ready = 0 if truthy(record.get("view_supervision_ready")) else 1
    confidence = -to_float(record.get("target_confidence_mean"), default=0.0)
    source_rank = 0 if str(record.get("source_name", "")).lower().startswith("3d-future") else 1
    return (quality, view_ready, identity, source_rank, confidence, str(record.get("object_id", "")))


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        object_id = str(record.get("object_id") or record.get("canonical_object_id") or record.get("uid") or "")
        if not object_id:
            object_id = f"record_{len(best)}"
        if object_id not in best or record_sort_key(record) < record_sort_key(best[object_id]):
            best[object_id] = record
    return sorted(best.values(), key=record_sort_key)


def append_unique(
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    count: int,
    selected_ids: set[str],
) -> None:
    if count <= 0:
        return
    for record in pool:
        object_id = str(record.get("object_id") or record.get("canonical_object_id") or "")
        if not object_id or object_id in selected_ids:
            continue
        selected.append(record)
        selected_ids.add(object_id)
        if count and len(selected) >= count:
            return


def select_acceptance_records(records: list[dict[str, Any]], target_records: int) -> list[dict[str, Any]]:
    records = dedupe(records)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def pool_for(predicate) -> list[dict[str, Any]]:
        return [record for record in records if predicate(record)]

    # First pass: guarantee material coverage where the data exists.
    for material_family in MATERIAL_PRIORITY:
        pool = pool_for(lambda item, mf=material_family: str(item.get("material_family", "unknown")) == mf)
        append_unique(selected, pool, count=len(selected) + 8, selected_ids=selected_ids)

    # Second pass: requested prior modes and difficult cases.
    with_prior = pool_for(lambda item: not is_without_prior(item))
    without_prior = pool_for(is_without_prior)
    fallback = pool_for(lambda item: infer_prior_source_type(item) in {"fallback_default", "no_prior_placeholder"} or str(item.get("prior_mode")) == "none")
    hard = pool_for(
        lambda item: truthy(item.get("thin_boundary_flag"))
        or "boundary" in str(item.get("failure_tags", "")).lower()
        or str(item.get("material_family", "")) in {"mixed_thin_boundary", "glass_metal", "metal_dominant"}
    )

    quotas = [
        (with_prior, 48),
        (without_prior, 32),
        (fallback, 24),
        (hard, 24),
    ]
    for pool, quota in quotas:
        append_unique(selected, pool, count=min(target_records, len(selected) + quota), selected_ids=selected_ids)

    # Final pass: fill with the best remaining clean examples.
    append_unique(selected, records, count=target_records, selected_ids=selected_ids)
    return selected[:target_records]


def assign_acceptance_splits(
    records: list[dict[str, Any]],
    *,
    train_records: int,
    val_records: int,
    test_records: int,
) -> list[dict[str, Any]]:
    total_requested = max(train_records, 0) + max(val_records, 0) + max(test_records, 0)
    if total_requested <= 0:
        train_records, val_records, test_records = 80, 24, 24
    output = []
    for index, record in enumerate(records):
        item = dict(record)
        if index < train_records:
            split = "train"
            paper_split = "r_v2_acceptance_train"
        elif index < train_records + val_records:
            split = "val"
            paper_split = "r_v2_acceptance_val"
        else:
            split = "test"
            paper_split = "r_v2_acceptance_test"
        item["default_split"] = split
        item["paper_split"] = paper_split
        item["prior_source_type"] = infer_prior_source_type(item)
        item["r_v2_acceptance_role"] = "without_prior" if is_without_prior(item) else "with_prior"
        item["selection_reason"] = "r_v2_one_day_acceptance_clean_diagnostic"
        output.append(item)
    return output


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return values[int(index)]
    return values[low] * (high - index) + values[high] * (index - low)


def summarize(records: list[dict[str, Any]], *, min_records: int, preferred_identity: float) -> dict[str, Any]:
    identities = [
        to_float(record.get("target_prior_identity"), default=math.nan)
        for record in records
        if math.isfinite(to_float(record.get("target_prior_identity"), default=math.nan))
    ]
    with_prior = sum(not is_without_prior(record) for record in records)
    without_prior = len(records) - with_prior
    material_count = len(distribution(records, "material_family"))
    copied = sum(str(record.get("target_source_type")) == "copied_from_prior" or truthy(record.get("target_is_prior_copy")) for record in records)
    identity_mean = sum(identities) / len(identities) if identities else None
    acceptance_blockers = []
    acceptance_warnings = []
    if len(records) < min_records:
        acceptance_blockers.append(f"records={len(records)} below min_records={min_records}")
    if copied:
        acceptance_blockers.append(f"copied_from_prior_records={copied}")
    if without_prior < 16:
        acceptance_blockers.append(f"without_prior={without_prior} below 16")
    if material_count < 3:
        acceptance_blockers.append(f"material_family_coverage={material_count} below 3")
    if identity_mean is None or identity_mean > preferred_identity:
        acceptance_warnings.append(
            f"target_prior_identity_mean={identity_mean} above preferred_threshold={preferred_identity}; "
            "continue as diagnostic acceptance only"
        )
    engineering_ready = not acceptance_blockers
    paper_identity_ready = identity_mean is not None and identity_mean <= preferred_identity
    return {
        "records": len(records),
        "by_default_split": distribution(records, "default_split"),
        "by_paper_split": distribution(records, "paper_split"),
        "by_prior_mode": distribution(records, "prior_mode"),
        "by_prior_source_type": distribution(records, "prior_source_type"),
        "by_material_family": distribution(records, "material_family"),
        "by_source_name": distribution(records, "source_name"),
        "by_target_source_type": distribution(records, "target_source_type"),
        "by_target_quality_tier": distribution(records, "target_quality_tier"),
        "with_prior": with_prior,
        "without_prior": without_prior,
        "view_supervision_ready": sum(truthy(record.get("view_supervision_ready")) for record in records),
        "target_prior_identity_mean": identity_mean,
        "target_prior_identity_p50": percentile(identities, 0.50),
        "target_prior_identity_p95": percentile(identities, 0.95),
        "copied_from_prior": copied,
        "acceptance_ready": engineering_ready,
        "paper_identity_ready": paper_identity_ready,
        "paper_acceptance_ready": engineering_ready and paper_identity_ready,
        "diagnostic_ready": len(records) >= min_records and copied == 0,
        "acceptance_blockers": acceptance_blockers,
        "acceptance_warnings": acceptance_warnings,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_audit_md(path: Path, summary: dict[str, Any], manifest_path: Path, source_manifests: list[str]) -> None:
    def table(counter: dict[str, int]) -> str:
        if not counter:
            return "- none\n"
        return "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(counter.items())) + "\n"

    lines = [
        "# R-v2 Acceptance 128 Audit",
        "",
        f"- output_manifest: `{manifest_path}`",
        f"- records: `{summary['records']}`",
        f"- acceptance_ready: `{summary['acceptance_ready']}`",
        f"- paper_identity_ready: `{summary['paper_identity_ready']}`",
        f"- paper_acceptance_ready: `{summary['paper_acceptance_ready']}`",
        f"- diagnostic_ready: `{summary['diagnostic_ready']}`",
        f"- with_prior: `{summary['with_prior']}`",
        f"- without_prior: `{summary['without_prior']}`",
        f"- view_supervision_ready: `{summary['view_supervision_ready']}`",
        f"- target_prior_identity_mean: `{summary['target_prior_identity_mean']}`",
        f"- target_prior_identity_p50: `{summary['target_prior_identity_p50']}`",
        f"- target_prior_identity_p95: `{summary['target_prior_identity_p95']}`",
        f"- copied_from_prior: `{summary['copied_from_prior']}`",
        "",
        "## Acceptance Blockers",
        "",
        *(f"- `{item}`" for item in summary["acceptance_blockers"]),
        "",
        "## Warnings",
        "",
        *(f"- `{item}`" for item in summary["acceptance_warnings"]),
        "",
        "## Source Manifests",
        "",
        *(f"- `{path}`" for path in source_manifests),
        "",
        "## prior_mode",
        table(summary["by_prior_mode"]),
        "## prior_source_type",
        table(summary["by_prior_source_type"]),
        "## material_family",
        table(summary["by_material_family"]),
        "## source_name",
        table(summary["by_source_name"]),
        "## target_source_type",
        table(summary["by_target_source_type"]),
        "## target_quality_tier",
        table(summary["by_target_quality_tier"]),
        "",
        "## Note",
        "",
        (
            "This subset is valid for R-v2 engineering/diagnostic acceptance. If the identity warning is present, "
            "do not use it as paper-stage evidence without a stricter low-identity subset."
        ),
        "",
    ]
    if not summary["acceptance_blockers"]:
        lines.insert(lines.index("## Acceptance Blockers") + 2, "- none")
    if not summary["acceptance_warnings"]:
        lines.insert(lines.index("## Warnings") + 2, "- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifests = args.manifest or DEFAULT_MANIFESTS
    source_manifests: list[str] = []
    clean_records: list[dict[str, Any]] = []
    rejected = Counter()

    for manifest in manifests:
        if not manifest.exists():
            continue
        source_manifests.append(str(manifest.resolve()))
        manifest_dir = manifest.parent
        for record in load_records(manifest):
            ok, blockers = is_valid_clean_record(
                record,
                manifest_dir=manifest_dir,
                require_existing_target_files=args.require_existing_target_files,
            )
            if not ok:
                rejected.update(blockers)
                continue
            if str(record.get("target_quality_tier", "")) not in {"paper_strong", "paper_pseudo", "paper_weak", ""}:
                rejected.update(["target_quality_not_paper"])
                continue
            clean_records.append(record)

    selected = select_acceptance_records(clean_records, args.target_records)
    selected = assign_acceptance_splits(
        selected,
        train_records=args.train_records,
        val_records=args.val_records,
        test_records=args.test_records,
    )
    summary = summarize(
        selected,
        min_records=args.min_records,
        preferred_identity=args.max_target_prior_identity_preferred,
    )
    summary["rejected_reasons"] = dict(rejected)
    summary["available_clean_records"] = len(dedupe(clean_records))
    summary["source_manifest_count"] = len(source_manifests)

    output_manifest = args.output_manifest or (args.output_root / "r_v2_acceptance_128_manifest.json")
    payload = {
        "manifest_version": "canonical_asset_record_v1_r_v2_acceptance_subset",
        "subset_name": "r_v2_acceptance_128",
        "experiment_role": "r_v2_one_day_acceptance_diagnostic",
        "source_manifests": source_manifests,
        "selection_policy": {
            "target_records": args.target_records,
            "min_records": args.min_records,
            "exclude_target_source_type": ["copied_from_prior"],
            "exclude_target_is_prior_copy": True,
            "require_target_paths": [
                "uv_target_roughness_path",
                "uv_target_metallic_path",
                "uv_target_confidence_path",
            ],
            "preferred_target_prior_identity": args.max_target_prior_identity_preferred,
            "diagnostic_identity_cap": args.max_target_prior_identity_diagnostic,
            "note": "If low-identity records are unavailable, the subset remains diagnostic-only.",
        },
        "summary": summary,
        "records": selected,
    }
    write_json(output_manifest, payload)
    write_json(args.output_root / "r_v2_acceptance_128_audit.json", {"summary": summary})
    write_audit_md(
        args.output_root / "r_v2_acceptance_128_audit.md",
        summary,
        output_manifest.resolve(),
        source_manifests,
    )
    print(json.dumps({"manifest": str(output_manifest.resolve()), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
