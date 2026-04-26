#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_RECORD_PATH_KEYS = {
    "source_model_path",
    "source_texture_root",
    "canonical_mesh_path",
    "canonical_glb_path",
    "canonical_views_json",
    "canonical_buffer_root",
    "canonical_bundle_root",
    "uv_albedo_path",
    "uv_normal_path",
    "uv_prior_roughness_path",
    "uv_prior_metallic_path",
    "uv_target_roughness_path",
    "uv_target_metallic_path",
    "uv_target_confidence_path",
    "target_second_pass_source_manifest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Conservatively clean SF3D material-refine intermediate roots after "
            "processed assets have entered protected manifests."
        ),
    )
    parser.add_argument("--protected-manifest", action="append", default=[])
    parser.add_argument("--protected-manifest-glob", action="append", default=[])
    parser.add_argument("--candidate-root", action="append", default=[])
    parser.add_argument("--active-root", action="append", default=[])
    parser.add_argument("--active-root-glob", action="append", default=[])
    parser.add_argument("--trash-root", type=Path, default=REPO_ROOT / "output" / "material_refine_cleanup" / "trash")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "output" / "material_refine_cleanup" / "post_ingest_cleanup_report.json")
    parser.add_argument("--mode", choices=("report", "trash", "delete"), default="report")
    parser.add_argument("--min-age-seconds", type=float, default=86400.0)
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument(
        "--candidate-name-prefix",
        action="append",
        default=[],
        help="Only clean immediate child directories whose names start with one of these prefixes.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def normalize(path: Path) -> Path:
    # absolute() can still touch cwd but avoids resolve() walking large or
    # partially deleted source trees. Cleanup only needs lexical containment.
    return Path(os.path.abspath(path))


def resolve_glob(pattern: str) -> list[Path]:
    if not pattern:
        return []
    if Path(pattern).is_absolute():
        return sorted(Path(item) for item in glob.glob(pattern, recursive=True))
    return sorted(REPO_ROOT.glob(pattern))


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def key_looks_path_like(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("path", "root", "manifest", "json", "csv", "html", "bundle"))


def iter_path_values(value: Any, *, key: str = "") -> Iterable[str]:
    if isinstance(value, str):
        if key_looks_path_like(key):
            yield value
    elif isinstance(value, dict):
        for child_key, item in value.items():
            yield from iter_path_values(item, key=str(child_key))
    elif isinstance(value, list):
        for item in value:
            yield from iter_path_values(item, key=key)


def looks_path_like(value: str) -> bool:
    text = value.strip()
    if not text or text.startswith(("http://", "https://", "s3://", "wandb://")):
        return False
    if text.startswith(("/", "./", "../", "output/", "configs/", "data/", "assets/")):
        return True
    if text.startswith(("canonical_", "uv_", "view_", "render_")):
        return False
    return "/" in text and not any(character.isspace() for character in text)


def owner_under_candidate_roots(path: Path, candidate_roots: list[Path]) -> Path | None:
    item = normalize(path)
    item_text = str(item)
    for root in candidate_roots:
        root_text = str(root)
        if item_text == root_text:
            return root
        if item_text.startswith(root_text.rstrip(os.sep) + os.sep):
            relative_text = item_text[len(root_text.rstrip(os.sep)) + 1 :]
            first = relative_text.split(os.sep, 1)[0]
            return root / first if first else root
        if root_text.startswith(item_text.rstrip(os.sep) + os.sep):
            return item
    return None


def protected_paths_from_manifest(path: Path, candidate_roots: list[Path]) -> set[Path]:
    payload = load_json(path)
    if not payload:
        return set()
    protected: set[Path] = set()
    manifest_owner = owner_under_candidate_roots(path, candidate_roots)
    if manifest_owner is not None:
        protected.add(manifest_owner)

    path_values: list[str] = []
    for value in payload.get("output_files", {}).values() if isinstance(payload.get("output_files"), dict) else []:
        if isinstance(value, str):
            path_values.append(value)
    records = payload.get("records", [])
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in PROTECTED_RECORD_PATH_KEYS:
                value = record.get(key)
                if isinstance(value, str):
                    path_values.append(value)
            for key, value in record.items():
                if key in PROTECTED_RECORD_PATH_KEYS or not isinstance(value, str):
                    continue
                if key_looks_path_like(str(key)) and key.endswith(("_path", "_root", "_json", "_manifest")):
                    path_values.append(value)

    for text in path_values:
        if not looks_path_like(text):
            continue
        candidate = repo_path(text)
        owner = owner_under_candidate_roots(candidate, candidate_roots)
        if owner is not None:
            protected.add(owner)
    return protected


def path_overlaps(candidate: Path, protected: set[Path]) -> bool:
    candidate_resolved = normalize(candidate)
    for item in protected:
        try:
            item.relative_to(candidate_resolved)
            return True
        except ValueError:
            pass
        try:
            candidate_resolved.relative_to(item)
            return True
        except ValueError:
            pass
    return False


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(normalize(path))
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def safe_trash_path(trash_root: Path, path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    resolved = normalize(path)
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        relative = Path(*[part for part in resolved.parts if part not in ("/", "")])
    return trash_root / stamp / relative


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_paths: list[Path] = [repo_path(value) for value in args.protected_manifest]
    for pattern in args.protected_manifest_glob:
        manifest_paths.extend(resolve_glob(pattern))
    manifest_paths = unique_paths(path for path in manifest_paths if path.exists() and path.is_file())

    candidate_roots = [normalize(repo_path(value)) for value in args.candidate_root]
    protected: set[Path] = set()
    for manifest in manifest_paths:
        protected.update(protected_paths_from_manifest(manifest, candidate_roots))

    for value in args.active_root:
        path = repo_path(value)
        if path.exists():
            protected.add(normalize(path))
    for pattern in args.active_root_glob:
        for path in resolve_glob(pattern):
            if path.exists():
                protected.add(normalize(path))

    now = time.time()
    actions: list[dict[str, Any]] = []
    candidates: list[Path] = []
    for root in candidate_roots:
        if not root.exists() or not root.is_dir():
            actions.append({"root": str(root), "action": "missing_candidate_root"})
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if args.candidate_name_prefix and not any(
                child.name.startswith(prefix) for prefix in args.candidate_name_prefix
            ):
                actions.append({"path": str(child), "action": "kept_name_not_selected"})
                continue
            age_seconds = max(0.0, now - child.stat().st_mtime)
            if age_seconds < args.min_age_seconds:
                actions.append(
                    {
                        "path": str(child),
                        "action": "kept_recent",
                        "age_seconds": round(age_seconds, 1),
                    }
                )
                continue
            if path_overlaps(child, protected):
                actions.append({"path": str(child), "action": "kept_protected"})
                continue
            candidates.append(child)

    candidates = unique_paths(candidates)[: max(0, int(args.max_actions))]
    for path in candidates:
        if args.mode == "report":
            actions.append({"path": str(path), "action": "would_clean"})
            continue
        if args.mode == "trash":
            target = safe_trash_path(repo_path(args.trash_root), path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            actions.append({"path": str(path), "action": "moved_to_trash", "trash_path": str(target)})
            continue
        shutil.rmtree(path)
        actions.append({"path": str(path), "action": "deleted"})

    payload = {
        "generated_at_utc": utc_now(),
        "mode": args.mode,
        "min_age_seconds": args.min_age_seconds,
        "protected_manifest_count": len(manifest_paths),
        "protected_path_count": len(protected),
        "candidate_root_count": len(args.candidate_root),
        "clean_candidate_count": len(candidates),
        "actions": actions,
    }
    write_json(repo_path(args.output_json), payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "actions"}, indent=2))


if __name__ == "__main__":
    main()
