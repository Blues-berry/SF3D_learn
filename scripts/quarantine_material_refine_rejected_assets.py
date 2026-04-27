#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Safely quarantine generated artifacts referenced only by material-refine "
            "reject manifests. The default candidate fields point at prepared bundle "
            "artifacts, not original source datasets."
        ),
    )
    parser.add_argument("--reject-manifest", type=Path, required=True)
    parser.add_argument("--protected-manifest", action="append", type=Path, default=[])
    parser.add_argument("--protected-manifest-glob", action="append", type=str, default=[])
    parser.add_argument("--candidate-path-field", action="append", type=str, default=[])
    parser.add_argument("--include-source-assets", action="store_true")
    parser.add_argument("--mode", choices=("report", "trash", "delete"), default="report")
    parser.add_argument("--trash-root", type=Path, default=REPO_ROOT / "output/material_refine_cleanup/reject_trash")
    parser.add_argument("--min-age-seconds", type=float, default=86400)
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or payload.get("objects") or payload.get("rows") or []
    return [record for record in records if isinstance(record, dict)]


def resolve_existing(path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = repo_path(Path(str(path_value))).resolve()
    if path.exists():
        return path
    return None


def path_fields(args: argparse.Namespace) -> list[str]:
    fields = list(args.candidate_path_field)
    if not fields:
        fields.extend(["canonical_bundle_root", "bundle_root"])
    if args.include_source_assets:
        fields.extend(["source_model_path", "canonical_mesh_path", "canonical_glb_path", "source_texture_root"])
    seen: set[str] = set()
    out: list[str] = []
    for field in fields:
        if field not in seen:
            seen.add(field)
            out.append(field)
    return out


def record_paths(record: dict[str, Any], fields: list[str]) -> set[Path]:
    paths: set[Path] = set()
    for field in fields:
        value = record.get(field)
        if isinstance(value, str):
            path = resolve_existing(value)
            if path is not None:
                paths.add(path)
    return paths


def protected_manifest_paths(args: argparse.Namespace) -> list[Path]:
    paths = [repo_path(path) for path in args.protected_manifest]
    for pattern in args.protected_manifest_glob:
        glob_pattern = str(repo_path(Path(pattern))) if not Path(pattern).is_absolute() else pattern
        paths.extend(Path(match) for match in glob.glob(glob_pattern, recursive=True))
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if path.exists() and key not in seen:
            seen.add(key)
            out.append(path)
    return out


def is_related(candidate: Path, protected: Path) -> bool:
    try:
        candidate.relative_to(protected)
        return True
    except ValueError:
        pass
    try:
        protected.relative_to(candidate)
        return True
    except ValueError:
        return False


def build_protected_paths(args: argparse.Namespace) -> set[Path]:
    fields = path_fields(argparse.Namespace(candidate_path_field=[], include_source_assets=True))
    protected: set[Path] = set()
    for manifest in protected_manifest_paths(args):
        try:
            records = load_records(manifest)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        for record in records:
            protected.update(record_paths(record, fields))
    return protected


def safe_trash_path(root: Path, candidate: Path) -> Path:
    digest = hashlib.sha1(str(candidate).encode("utf-8")).hexdigest()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{candidate.name}.{digest}.{stamp}"
    return root / name


def main() -> None:
    args = parse_args()
    reject_manifest = repo_path(args.reject_manifest)
    fields = path_fields(args)
    reject_records = load_records(reject_manifest)
    protected_paths = build_protected_paths(args)
    candidates: dict[Path, dict[str, Any]] = {}
    for record in reject_records:
        for path in record_paths(record, fields):
            candidates.setdefault(
                path,
                {
                    "path": str(path),
                    "object_id": record.get("canonical_object_id") or record.get("object_id"),
                    "source_name": record.get("source_name"),
                    "material_family": record.get("material_family"),
                },
            )

    now = time.time()
    actions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for candidate, meta in sorted(candidates.items(), key=lambda item: str(item[0])):
        if len(actions) >= int(args.max_actions):
            skipped.append({**meta, "reason": "max_actions_reached"})
            continue
        if any(is_related(candidate, protected) for protected in protected_paths):
            skipped.append({**meta, "reason": "protected_by_manifest"})
            continue
        age_seconds = max(0.0, now - candidate.stat().st_mtime)
        if age_seconds < float(args.min_age_seconds):
            skipped.append({**meta, "reason": "too_new", "age_seconds": round(age_seconds, 1)})
            continue
        size_bytes = candidate.stat().st_size if candidate.is_file() else 0
        if candidate.is_dir():
            try:
                size_bytes = sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file())
            except OSError:
                size_bytes = 0
        action = {**meta, "mode": args.mode, "age_seconds": round(age_seconds, 1), "size_bytes": size_bytes}
        if args.mode == "trash":
            trash_path = safe_trash_path(repo_path(args.trash_root), candidate)
            trash_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate), str(trash_path))
            action["trash_path"] = str(trash_path)
        elif args.mode == "delete":
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
        actions.append(action)

    report = {
        "updated_at_utc": utc_now(),
        "mode": args.mode,
        "reject_manifest": str(reject_manifest.resolve()),
        "candidate_path_fields": fields,
        "include_source_assets": bool(args.include_source_assets),
        "reject_records": len(reject_records),
        "protected_manifest_count": len(protected_manifest_paths(args)),
        "protected_path_count": len(protected_paths),
        "candidate_path_count": len(candidates),
        "action_count": len(actions),
        "skipped_count": len(skipped),
        "actions": actions,
        "skipped": skipped[:500],
    }
    output_json = repo_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
