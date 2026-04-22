from __future__ import annotations

import argparse
import glob
import json
import sys
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
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def ranking_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    summary = candidate.get("summary") or {}
    ready = 1 if summary.get("paper_stage_ready") else 0
    eligible = int(summary.get("paper_stage_eligible_records", 0))
    records = int(summary.get("records", 0))
    view_rate = float(summary.get("effective_view_supervision_record_rate", 0.0))
    identity = float(summary.get("target_prior_identity_rate", 1.0))
    return (ready, eligible, view_rate, -identity, records)


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
        else:
            try:
                payload = audit_manifest(
                    manifest_path,
                    max_records=-1,
                    allowed_paper_license_buckets=allowed_license_buckets,
                )
                summary = payload["summary"]
                selected_summary = {
                    key: summary.get(key)
                    for key in [
                        "records",
                        "paper_stage_ready",
                        "paper_stage_eligible_records",
                        "target_prior_identity_rate",
                        "effective_view_supervision_record_rate",
                        "target_source_type_counts",
                        "target_quality_tier_counts",
                        "readiness_blockers",
                        "audit_schema_version",
                    ]
                }
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
            }
            cache_entries[fingerprint["path"]] = candidate
        if int((candidate.get("summary") or {}).get("records", 0)) < max(int(args.min_records), 0):
            continue
        candidates.append(candidate)

    ordered = sorted(candidates, key=ranking_key, reverse=True)
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
