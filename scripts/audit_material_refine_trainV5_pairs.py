#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "train/trainV5_plus_a_track"
EXPECTED_VARIANTS = {
    "near_gt_prior",
    "mild_gap_prior",
    "medium_gap_prior",
    "large_gap_prior",
    "no_prior_bootstrap",
}
LARGE_FORBIDDEN_SUFFIXES = {".png", ".jpg", ".jpeg", ".exr", ".glb", ".npz", ".npy"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def records(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def path_exists(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).exists()


def resolved(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        return str(Path(value).resolve())
    except OSError:
        return str(value)


def sha256_file(value: Any) -> str | None:
    if not path_exists(value):
        return None
    digest = hashlib.sha256()
    with Path(str(value)).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Audit TrainV5 pair manifests for R-v2.1 pair-based training.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--expected-targets", type=int, default=322)
    parser.add_argument("--expected-pairs", type=int, default=1610)
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Write the JSON summary to this explicit path. Does not update the canonical Markdown report.",
    )
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="Update trainV5_pair_audit_report.json and .md under --input-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    target_bundles = records(input_dir / "trainV5_target_bundles.json")
    prior_variants = records(input_dir / "trainV5_prior_variants.json")
    training_pairs = records(input_dir / "trainV5_training_pairs.json")
    identity_controls_path = input_dir / "trainV5_identity_controls.json"
    identity_controls = records(identity_controls_path) if identity_controls_path.exists() else []

    blockers: list[str] = []
    if len(target_bundles) != args.expected_targets:
        blockers.append(f"target_bundles_expected_{args.expected_targets}_got_{len(target_bundles)}")
    if len(prior_variants) != args.expected_pairs:
        blockers.append(f"prior_variants_expected_{args.expected_pairs}_got_{len(prior_variants)}")
    if len(training_pairs) != args.expected_pairs:
        blockers.append(f"training_pairs_expected_{args.expected_pairs}_got_{len(training_pairs)}")

    variants_by_target: dict[str, list[str]] = defaultdict(list)
    pair_splits_by_object: dict[str, set[str]] = defaultdict(set)
    leakage_cases: list[dict[str, Any]] = []
    no_prior_path_cases: list[str] = []
    unresolved_cases: list[str] = []
    for pair in training_pairs:
        target_id = str(pair.get("target_bundle_id") or "")
        variant_type = str(pair.get("prior_variant_type") or "")
        variants_by_target[target_id].append(variant_type)
        pair_splits_by_object[str(pair.get("object_id") or "")].add(str(pair.get("split") or "unknown"))
        if not pair.get("path_resolved_ok"):
            unresolved_cases.append(str(pair.get("pair_id") or pair.get("training_pair_id") or "unknown"))
        if variant_type == "no_prior_bootstrap":
            if pair.get("uv_prior_roughness_path") or pair.get("uv_prior_metallic_path") or pair.get("scalar_prior_roughness") is not None or pair.get("scalar_prior_metallic") is not None:
                no_prior_path_cases.append(str(pair.get("pair_id") or pair.get("training_pair_id") or "unknown"))
        for channel in ("roughness", "metallic"):
            prior_path = pair.get(f"uv_prior_{channel}_path")
            target_path = pair.get(f"uv_target_{channel}_path")
            if not prior_path:
                continue
            same_path = bool(resolved(prior_path) and resolved(prior_path) == resolved(target_path))
            prior_hash = sha256_file(prior_path)
            target_hash = sha256_file(target_path)
            same_hash = bool(prior_hash and target_hash and prior_hash == target_hash)
            if same_path or same_hash:
                leakage_cases.append(
                    {
                        "pair_id": pair.get("pair_id") or pair.get("training_pair_id"),
                        "channel": channel,
                        "same_path": same_path,
                        "same_hash": same_hash,
                    }
                )

    bad_variant_targets = {
        target_id: sorted(types)
        for target_id, types in variants_by_target.items()
        if len(types) != 5 or set(types) != EXPECTED_VARIANTS
    }
    split_offenders = {
        object_id: sorted(splits)
        for object_id, splits in pair_splits_by_object.items()
        if len(splits) > 1
    }
    split_counts = Counter(str(pair.get("split") or "unknown") for pair in training_pairs)
    if bad_variant_targets:
        blockers.append("target_variant_set_mismatch")
    if split_offenders:
        blockers.append("object_level_split_leakage")
    if not all(split_counts.get(split, 0) > 0 for split in ("train", "val", "test")):
        blockers.append("split_empty")
    if leakage_cases:
        blockers.append("prior_target_path_or_hash_leakage")
    if no_prior_path_cases:
        blockers.append("no_prior_contains_prior_inputs")
    if unresolved_cases:
        blockers.append("unresolved_pair_paths")

    forbidden_train_files = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "train").rglob("*")
        if path.is_file() and path.suffix.lower() in LARGE_FORBIDDEN_SUFFIXES
    ]
    if forbidden_train_files:
        blockers.append("forbidden_large_asset_suffix_in_train_dir")

    summary = {
        "generated_at_utc": utc_now(),
        "input_dir": str(input_dir.resolve()),
        "audit_pass": not blockers,
        "blockers": blockers,
        "target_bundles": len(target_bundles),
        "prior_variants": len(prior_variants),
        "training_pairs": len(training_pairs),
        "identity_controls": len(identity_controls),
        "split": dict(split_counts),
        "prior_variant_type": dict(Counter(str(pair.get("prior_variant_type") or "unknown") for pair in training_pairs)),
        "prior_quality_bin": dict(Counter(str(pair.get("prior_quality_bin") or "unknown") for pair in training_pairs)),
        "prior_spatiality": dict(Counter(str(pair.get("prior_spatiality") or "unknown") for pair in training_pairs)),
        "bad_variant_targets": bad_variant_targets,
        "split_offenders": split_offenders,
        "leakage_cases": leakage_cases,
        "no_prior_path_cases": no_prior_path_cases,
        "unresolved_cases": unresolved_cases,
        "forbidden_train_files": forbidden_train_files,
    }
    if args.write_report is not None:
        write_json(args.write_report, summary)
    if args.update_report:
        write_json(input_dir / "trainV5_pair_audit_report.json", summary)
        md_lines = [
            "# TrainV5 Pair Audit Report",
            "",
            f"- generated_at_utc: `{summary['generated_at_utc']}`",
            f"- audit_pass: `{str(summary['audit_pass']).lower()}`",
            f"- target_bundles: `{summary['target_bundles']}`",
            f"- prior_variants: `{summary['prior_variants']}`",
            f"- training_pairs: `{summary['training_pairs']}`",
            f"- identity_controls: `{summary['identity_controls']}`",
            f"- split: `{json.dumps(summary['split'], ensure_ascii=False)}`",
            f"- prior_variant_type: `{json.dumps(summary['prior_variant_type'], ensure_ascii=False)}`",
            f"- prior_spatiality: `{json.dumps(summary['prior_spatiality'], ensure_ascii=False)}`",
            f"- blockers: `{json.dumps(blockers, ensure_ascii=False)}`",
        ]
        (input_dir / "trainV5_pair_audit_report.md").write_text(
            "\n".join(md_lines) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
