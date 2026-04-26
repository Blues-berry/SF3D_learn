from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.manifest_quality import AUDIT_SCHEMA_VERSION, audit_manifest


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Select the best available canonical manifest for paper-stage material refinement.",
    )
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--manifest-glob", action="append", type=str, default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--cache-json", type=Path, default=None)
    parser.add_argument("--paper-license-buckets", type=str, default=None)
    parser.add_argument("--min-records", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--prefer-diversity", type=parse_bool, default=True)
    parser.add_argument("--min-material-family-records-for-diversity", type=int, default=16)
    parser.add_argument("--min-source-records-for-diversity", type=int, default=16)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("records", [])
    else:
        rows = payload
    return [row for row in rows if isinstance(row, dict)]


def normalize_bool_label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "with_prior"}:
        return "true"
    if text in {"0", "false", "no", "n", "none", "no_prior", "without_prior"}:
        return "false"
    return text or "unknown"


def scan_manifest_metadata(path: Path) -> dict[str, Any]:
    try:
        rows = manifest_records(load_json(path))
    except Exception:
        return {}
    source_counts: Counter[str] = Counter()
    generator_counts: Counter[str] = Counter()
    material_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    prior_counts: Counter[str] = Counter()
    prior_mode_counts: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        source = row.get("source_name") or metadata.get("source_name") or row.get("generator_id") or "unknown"
        generator = row.get("generator_id") or metadata.get("generator_id") or source or "unknown"
        material = row.get("material_family") or metadata.get("material_family") or row.get("category_bucket") or "unknown"
        license_bucket = row.get("license_bucket") or metadata.get("license_bucket") or "unknown"
        prior_mode = row.get("prior_mode") or metadata.get("prior_mode") or "unknown"
        has_prior = row.get("has_material_prior")
        if has_prior is None:
            has_prior = metadata.get("has_material_prior")
        if has_prior is None and str(prior_mode).strip().lower() not in {"", "unknown"}:
            has_prior = str(prior_mode).strip().lower() not in {"none", "no_prior", "without_prior"}
        source_counts[str(source)] += 1
        generator_counts[str(generator)] += 1
        material_counts[str(material)] += 1
        license_counts[str(license_bucket)] += 1
        prior_counts[normalize_bool_label(has_prior)] += 1
        prior_mode_counts[str(prior_mode)] += 1
    return {
        "source_counts": dict(source_counts),
        "generator_counts": dict(generator_counts),
        "material_family_counts": dict(material_counts),
        "license_bucket_counts": dict(license_counts),
        "has_material_prior_counts": dict(prior_counts),
        "prior_mode_counts": dict(prior_mode_counts),
    }


def expand_candidates(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for manifest in args.manifest:
        paths.append(manifest if manifest.is_absolute() else REPO_ROOT / manifest)
    for pattern in args.manifest_glob:
        absolute_pattern = pattern if pattern.startswith("/") else str(REPO_ROOT / pattern)
        paths.extend(Path(item) for item in glob.glob(absolute_pattern, recursive=True))
    deduped: list[Path] = []
    seen: set[str] = set()
    existing_paths = [path for path in paths if path.exists() and path.is_file()]
    existing_paths = sorted(existing_paths, key=lambda item: item.stat().st_mtime_ns, reverse=True)
    for path in existing_paths:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path.resolve())
    return deduped[: max(int(args.max_candidates), 1)]


def manifest_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
    }


def count_diverse_groups(counts: dict[str, Any] | None, *, min_count: int) -> int:
    if not isinstance(counts, dict):
        return 0
    return sum(1 for value in counts.values() if int(value or 0) >= min_count)


def count_from_any(summary: dict[str, Any], keys: list[str], wanted: list[str]) -> int:
    for key in keys:
        counts = summary.get(key)
        if not isinstance(counts, dict):
            continue
        return sum(int(counts.get(item, 0) or 0) for item in wanted)
    return 0


def build_selection_features(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    material_family_counts = summary.get("material_family_counts") or {}
    source_counts = summary.get("source_name_counts") or summary.get("source_counts") or {}
    generator_counts = summary.get("generator_id_counts") or summary.get("generator_counts") or {}
    quality_counts = summary.get("target_quality_tier_counts") or {}
    source_type_counts = summary.get("target_source_type_counts") or {}
    prior_counts = summary.get("has_material_prior_counts") or summary.get("prior_mode_counts") or {}
    records = int(summary.get("records", 0) or 0)
    view_rate = float(summary.get("effective_view_supervision_record_rate", 0.0) or 0.0)
    no_prior_count = count_from_any(
        summary,
        ["has_material_prior_counts", "prior_mode_counts", "prior_label_counts"],
        ["False", "false", "0", "none", "no_prior", "without_prior"],
    )
    paper_strong_count = int(quality_counts.get("paper_strong", 0) or 0)
    paper_pseudo_count = int(quality_counts.get("paper_pseudo", 0) or 0)
    gt_render_count = int(source_type_counts.get("gt_render_baked", 0) or 0)
    pseudo_multiview_count = int(source_type_counts.get("pseudo_from_multiview", 0) or 0)
    return {
        "material_family_diversity": count_diverse_groups(
            material_family_counts,
            min_count=max(1, int(args.min_material_family_records_for_diversity)),
        ),
        "source_diversity": count_diverse_groups(
            source_counts,
            min_count=max(1, int(args.min_source_records_for_diversity)),
        ),
        "generator_diversity": count_diverse_groups(
            generator_counts,
            min_count=max(1, int(args.min_source_records_for_diversity)),
        ),
        "view_ready_records_estimate": int(round(records * view_rate)),
        "paper_strong_records": paper_strong_count,
        "paper_pseudo_records": paper_pseudo_count,
        "gt_render_baked_records": gt_render_count,
        "pseudo_from_multiview_records": pseudo_multiview_count,
        "no_prior_records_estimate": no_prior_count,
        "has_distribution_counts": {
            "material_family": bool(material_family_counts),
            "source": bool(source_counts),
            "generator": bool(generator_counts),
            "prior": bool(prior_counts),
        },
    }


def ranking_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    summary = candidate.get("summary") or {}
    features = candidate.get("selection_features") or {}
    ready = 1 if summary.get("paper_stage_ready") else 0
    eligible = int(summary.get("paper_stage_eligible_records", 0))
    records = int(summary.get("records", 0))
    view_rate = float(summary.get("effective_view_supervision_record_rate", 0.0))
    identity = float(summary.get("target_prior_identity_rate", 1.0))
    material_diversity = int(features.get("material_family_diversity", 0))
    source_diversity = int(features.get("source_diversity", 0))
    generator_diversity = int(features.get("generator_diversity", 0))
    paper_strong = int(features.get("paper_strong_records", 0))
    gt_render = int(features.get("gt_render_baked_records", 0))
    no_prior = int(features.get("no_prior_records_estimate", 0))
    return (
        ready,
        eligible,
        material_diversity,
        source_diversity,
        generator_diversity,
        min(no_prior, eligible),
        paper_strong,
        gt_render,
        view_rate,
        -identity,
        records,
    )


def main() -> None:
    args = parse_args()
    cache_path = args.cache_json or args.output_json.with_name("manifest_selection_cache.json")
    cache_payload = load_json(cache_path) if cache_path.exists() else {"entries": {}}
    cache_entries = cache_payload.get("entries") or {}

    allowed_license_buckets = set(parse_csv_list(args.paper_license_buckets)) or None
    candidates: list[dict[str, Any]] = []
    for manifest_path in expand_candidates(args):
        fingerprint = manifest_fingerprint(manifest_path)
        cache_entry = cache_entries.get(fingerprint["path"])
        if cache_entry and cache_entry.get("fingerprint") == fingerprint:
            candidate = dict(cache_entry)
            summary = candidate.get("summary") or {}
            if not summary.get("has_material_prior_counts") or not summary.get("prior_mode_counts"):
                metadata_counts = scan_manifest_metadata(manifest_path)
                for key, value in metadata_counts.items():
                    if not summary.get(key):
                        summary[key] = value
                candidate["summary"] = summary
        else:
            try:
                payload = audit_manifest(
                    manifest_path,
                    max_records=-1,
                    allowed_paper_license_buckets=allowed_license_buckets,
                )
                summary = payload["summary"]
                metadata_counts = scan_manifest_metadata(manifest_path)
                selected_summary = {
                    key: summary.get(key)
                    for key in [
                        "records",
                        "paper_stage_ready",
                        "paper_stage_eligible_records",
                        "target_prior_identity_rate",
                        "effective_view_supervision_record_rate",
                        "material_family_counts",
                        "source_name_counts",
                        "source_counts",
                        "generator_id_counts",
                        "generator_counts",
                        "license_bucket_counts",
                        "has_material_prior_counts",
                        "prior_mode_counts",
                        "target_source_type_counts",
                        "target_quality_tier_counts",
                        "readiness_blockers",
                        "audit_schema_version",
                    ]
                }
                for key, value in metadata_counts.items():
                    if not selected_summary.get(key):
                        selected_summary[key] = value
            except Exception as exc:
                selected_summary = {
                    "records": 0,
                    "paper_stage_ready": False,
                    "paper_stage_eligible_records": 0,
                    "target_prior_identity_rate": 1.0,
                    "effective_view_supervision_record_rate": 0.0,
                    "target_source_type_counts": {},
                    "target_quality_tier_counts": {},
                    "readiness_blockers": [f"audit_failed:{type(exc).__name__}:{exc}"],
                    "audit_schema_version": AUDIT_SCHEMA_VERSION,
                }
            candidate = {
                "manifest": fingerprint["path"],
                "fingerprint": fingerprint,
                "summary": selected_summary,
                "selection_features": build_selection_features(selected_summary, args),
            }
            cache_entries[fingerprint["path"]] = candidate
        if "selection_features" not in candidate:
            candidate["selection_features"] = build_selection_features(candidate.get("summary") or {}, args)
        if int((candidate.get("summary") or {}).get("records", 0)) < max(int(args.min_records), 0):
            continue
        candidates.append(candidate)

    ordered = sorted(candidates, key=ranking_key if args.prefer_diversity else lambda item: ranking_key({**item, "selection_features": {}}), reverse=True)
    best = ordered[0] if ordered else None
    payload = {
        "selected_manifest": None if best is None else best["manifest"],
        "selected_summary": None if best is None else best["summary"],
        "candidates": ordered,
    }
    write_json(args.output_json, payload)
    write_json(cache_path, {"entries": cache_entries})
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
