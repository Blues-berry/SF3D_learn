#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_material_refine_trainV5_plus_a_track import (  # noqa: E402
    VARIANT_SPECS,
    build_pair,
    build_prior_variant,
    build_target_bundle,
    summarize,
)
from sf3d.material_refine.trainv5_target_gate import (  # noqa: E402
    TARGET_GATE_VERSION,
    target_prior_relation_diagnostics,
    trainv5_target_truth_gate,
)


DEFAULT_B_ROOT = REPO_ROOT / "output/material_refine_trainV5_abc/B_track/full_1155_rebake"
DEFAULT_A_DIR = REPO_ROOT / "train/trainV5_plus_a_track"
DEFAULT_B_TRAIN_DIR = REPO_ROOT / "train/trainV5_plus_full"
DEFAULT_MERGED_DIR = REPO_ROOT / "train/trainV5_merged_ab"
VARIANT_ORDER = list(VARIANT_SPECS)


def batch_slug_from_root(path: Path) -> str:
    return path.name


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def records(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def skipped(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("skipped_records", [])
        return [row for row in rows if isinstance(row, dict)]
    return []


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def path_exists(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).exists()


def stable_split(object_id: str) -> str:
    bucket = int(hashlib.sha1(object_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 15:
        return "val"
    if bucket < 30:
        return "test"
    return "train"


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "unknown") for row in rows))


def numeric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = sorted(float(v) for row in rows if (v := finite_float(row.get(key))) is not None)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    p95_idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": values[len(values) // 2],
        "p95": values[p95_idx],
        "min": values[0],
        "max": values[-1],
    }


def license_allowed_for_engineering(record: dict[str, Any]) -> bool:
    explicit = record.get("license_allowed_for_training")
    if explicit is not None:
        return bool_value(explicit)
    status = str(record.get("license_status") or "").lower()
    bucket = str(record.get("license_bucket") or "").lower()
    blocked_tokens = ("blocked", "forbidden", "no_training", "hard_block")
    return not any(token in status or token in bucket for token in blocked_tokens)


def trainv5_gate(record: dict[str, Any]) -> tuple[bool, list[str]]:
    return trainv5_target_truth_gate(record)


def run_contract_audit(manifest: Path, output_root: Path) -> dict[str, Any]:
    audit_root = output_root / "contract_audit"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/audit_material_refine_rebake_v2_contract.py"),
        "--manifest",
        str(manifest),
        "--output-root",
        str(audit_root),
        "--mean-threshold",
        "0.08",
        "--p95-threshold",
        "0.20",
        "--min-balanced-records",
        "0",
        "--min-source-diversity",
        "1",
        "--max-source-ratio",
        "1.0",
        "--min-no-prior-ratio",
        "0.0",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"returncode": result.returncode, "stdout_tail": result.stdout.splitlines()[-80:], "audit_root": str(audit_root)}


def write_problem_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "object_id",
                "source_name",
                "material_family",
                "license_bucket",
                "target_view_alignment_mean",
                "target_view_alignment_p95",
                "target_as_pred_pass",
                "target_is_prior_copy",
                "target_prior_identity",
                "target_source_type",
                "blockers",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["blockers"] = ";".join(str(x) for x in row.get("target_truth_gate_blockers", []))
            writer.writerow(item)


def remove_blocked_markers(path: Path) -> None:
    if not path.exists():
        return
    for marker in path.glob("BLOCKED_*.md"):
        try:
            marker.unlink()
        except OSError:
            continue


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    return records(read_json(path, []))


def dedup_by_object(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        oid = str(row.get("object_id") or row.get("canonical_object_id") or "")
        if not oid:
            continue
        selected.setdefault(oid, row)
    return list(selected.values())


def dedup_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            continue
        selected.setdefault(value, row)
    return list(selected.values())


def write_rebake_reports(b_root: Path, manifest: Path, prepared: list[dict[str, Any]], skipped_rows: list[dict[str, Any]], gate_pass_rows: list[dict[str, Any]], gate_fail_rows: list[dict[str, Any]], contract_audit: dict[str, Any]) -> None:
    batch_slug = batch_slug_from_root(b_root)
    diagnostic = {
        "generated_at_utc": utc_now(),
        "source_manifest": str(manifest),
        "batch_name": batch_slug,
        "records": prepared,
        "skipped_records": skipped_rows,
        "summary": {
            "prepared_records": len(prepared),
            "skipped_records": len(skipped_rows),
            "target_gate_version": TARGET_GATE_VERSION,
            "target_truth_gate_pass": len(gate_pass_rows),
            "target_truth_gate_fail": len(gate_fail_rows),
            "pass_rate": len(gate_pass_rows) / len(prepared) if prepared else 0.0,
            "target_prior_relation_diagnostic": {
                "target_is_prior_copy": sum(bool_value(row.get("target_is_prior_copy")) for row in prepared),
                "target_not_prior_copy": sum(not bool_value(row.get("target_is_prior_copy")) for row in prepared),
            },
            "material_family": distribution(prepared, "material_family"),
            "source_name": distribution(prepared, "source_name"),
            "prior_mode": distribution(prepared, "prior_mode"),
            "target_view_alignment_mean": numeric_stats(prepared, "target_view_alignment_mean"),
            "target_view_alignment_p95": numeric_stats(prepared, "target_view_alignment_p95"),
        },
        "contract_audit": contract_audit,
    }
    write_json(b_root / f"{batch_slug}_diagnostic_manifest.json", diagnostic)
    target_truth_manifest = {
        "generated_at_utc": utc_now(),
        "source_manifest": str(manifest),
        "batch_name": batch_slug,
        "target_gate_version": TARGET_GATE_VERSION,
        "records": gate_pass_rows,
        "summary": summarize(gate_pass_rows),
    }
    write_json(b_root / f"{batch_slug}_target_truth_gate_pass_manifest.json", target_truth_manifest)
    write_problem_csv(b_root / f"{batch_slug}_problem_cases.csv", gate_fail_rows + skipped_rows)
    summary = diagnostic["summary"]
    write_text(
        b_root / f"{batch_slug}_path_audit.md",
        "\n".join(
            [
                f"# {batch_slug} Path Audit",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- prepared_records: `{len(prepared)}`",
                f"- skipped_records: `{len(skipped_rows)}`",
                f"- target_gate_version: `{TARGET_GATE_VERSION}`",
                f"- target_truth_gate_pass: `{len(gate_pass_rows)}`",
                f"- target_truth_gate_fail: `{len(gate_fail_rows)}`",
                f"- source_model_path_exists: `{sum(path_exists(row.get('source_model_path')) for row in prepared)}`",
                f"- canonical_buffer_root_exists: `{sum(path_exists(row.get('canonical_buffer_root')) for row in prepared)}`",
                f"- uv_target_paths_exist: `{sum(all(path_exists(row.get(k)) for k in ('uv_target_roughness_path','uv_target_metallic_path','uv_target_confidence_path')) for row in prepared)}`",
            ]
        ),
    )
    write_text(
        b_root / f"{batch_slug}_decision.md",
        "\n".join(
            [
                f"# {batch_slug} Rebake Decision",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                "- status: `completed`",
                "- full_rebake_launched: `true`",
                f"- prepared_records: `{len(prepared)}`",
                f"- skipped_records: `{len(skipped_rows)}`",
                f"- target_gate_version: `{TARGET_GATE_VERSION}`",
                f"- target_truth_gate_pass: `{len(gate_pass_rows)}`",
                f"- target_truth_gate_fail: `{len(gate_fail_rows)}`",
                f"- pass_rate: `{summary['pass_rate']}`",
                f"- target_prior_relation_diagnostic: `{json.dumps(summary['target_prior_relation_diagnostic'], ensure_ascii=False)}`",
                f"- material_family: `{json.dumps(summary['material_family'], ensure_ascii=False)}`",
                f"- source_name: `{json.dumps(summary['source_name'], ensure_ascii=False)}`",
                f"- target_view_alignment_mean: `{json.dumps(summary['target_view_alignment_mean'], ensure_ascii=False)}`",
                f"- target_view_alignment_p95: `{json.dumps(summary['target_view_alignment_p95'], ensure_ascii=False)}`",
                f"- contract_audit_returncode: `{contract_audit.get('returncode')}`",
                "",
                "This is a TrainV5 engineering gate decision, not a paper claim.",
            ]
        ),
    )
    write_json(
        b_root / f"{batch_slug}_decision.json",
        {
            "generated_at_utc": utc_now(),
            "batch_name": batch_slug,
            "status": "completed",
            "full_rebake_launched": True,
            "prepared_records": len(prepared),
            "skipped_records": len(skipped_rows),
            "target_gate_version": TARGET_GATE_VERSION,
            "target_truth_gate_pass": len(gate_pass_rows),
            "target_truth_gate_fail": len(gate_fail_rows),
            "pass_rate": summary["pass_rate"],
            "target_prior_relation_diagnostic": summary["target_prior_relation_diagnostic"],
            "contract_audit": contract_audit,
        },
    )
    write_text(
        b_root / "progress_final.md",
        "\n".join(
            [
                f"# {batch_slug} Progress Final",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                "- status: `completed`",
                f"- processed: `{len(prepared) + len(skipped_rows)}/1155`",
                f"- target_gate_version: `{TARGET_GATE_VERSION}`",
                f"- target_truth_gate_pass: `{len(gate_pass_rows)}`",
                f"- target_truth_gate_fail: `{len(gate_fail_rows)}`",
                f"- pass_rate: `{summary['pass_rate']}`",
                f"- material_family distribution: `{json.dumps(summary['material_family'], ensure_ascii=False)}`",
                f"- source distribution: `{json.dumps(summary['source_name'], ensure_ascii=False)}`",
                f"- prior hints: `{json.dumps(summary['prior_mode'], ensure_ascii=False)}`",
                f"- failure reason counts: `{json.dumps(dict(Counter(';'.join(row.get('target_truth_gate_blockers', [])) for row in gate_fail_rows)), ensure_ascii=False)}`",
            ]
        ),
    )


def finalize_rebake(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = read_json(args.full_manifest, {})
    prepared = records(payload)
    skipped_rows = skipped(payload)
    if not prepared and not skipped_rows:
        raise RuntimeError(f"full_rebake_manifest_has_no_records:{args.full_manifest}")
    contract_audit = run_contract_audit(args.full_manifest, args.b_root) if args.run_contract_audit else {"skipped": True}
    gate_pass_rows: list[dict[str, Any]] = []
    gate_fail_rows: list[dict[str, Any]] = []
    for row in prepared:
        gate_ok, blockers = trainv5_gate(row)
        item = dict(row)
        item["target_gate_version"] = TARGET_GATE_VERSION
        item["target_truth_gate_pass"] = gate_ok
        item["target_truth_gate_blockers"] = blockers
        item["target_prior_relation_diagnostic"] = target_prior_relation_diagnostics(item)
        if gate_ok:
            gate_pass_rows.append(item)
        else:
            gate_fail_rows.append(item)
    write_rebake_reports(args.b_root, args.full_manifest, prepared, skipped_rows, gate_pass_rows, gate_fail_rows, contract_audit)
    return gate_pass_rows, gate_fail_rows


def build_plus_full_records(full_manifest: Path, gate_pass_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bundles: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    identity_controls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in gate_pass_rows:
        oid = str(source.get("object_id") or source.get("canonical_object_id") or "")
        if not oid or oid in seen:
            continue
        seen.add(oid)
        item = dict(source)
        split = stable_split(oid)
        item["split"] = split
        item["default_split"] = split
        item["target_bundle_id"] = f"tb_b_full_{oid}"
        bundle = build_target_bundle(item)
        bundles.append(bundle)
        for variant_type in VARIANT_ORDER:
            variant, identity = build_prior_variant(bundle, variant_type, item)
            variants.append(variant)
            pairs.append(build_pair(bundle, variant))
            if identity is not None:
                identity_controls.append(identity)
    return bundles, variants, pairs, identity_controls


def write_plus_full_outputs(
    out_dir: Path,
    *,
    source_manifest: Path,
    bundles: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    identity_controls: list[dict[str, Any]],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    remove_blocked_markers(out_dir)
    write_json(
        out_dir / "trainV5_plus_target_bundles.json",
        {"generated_at_utc": utc_now(), "source_manifest": str(source_manifest), "records": bundles, "summary": summarize(bundles)},
    )
    write_json(
        out_dir / "trainV5_plus_prior_variants.json",
        {"generated_at_utc": utc_now(), "source_manifest": str(source_manifest), "records": variants, "identity_controls": identity_controls, "summary": summarize(variants)},
    )
    write_json(
        out_dir / "trainV5_plus_training_pairs.json",
        {"generated_at_utc": utc_now(), "source_manifest": str(source_manifest), "records": pairs, "summary": summarize(pairs)},
    )
    sampler = {
        "generated_at_utc": utc_now(),
        "sampler": "TrainV5_plus_full_pair_balanced_v1",
        "prior_variant_weights": {
            "near_gt_prior": 1.0,
            "mild_gap_prior": 1.0,
            "medium_gap_prior": 1.0,
            "large_gap_prior": 1.0,
            "no_prior_bootstrap": 0.75,
        },
        "balance_axes": ["material_family", "source_name", "prior_variant_type", "prior_quality_bin"],
    }
    write_json(out_dir / "trainV5_plus_sampler_config.json", sampler)
    split_counts = Counter(str(pair.get("split")) for pair in pairs)
    target_counts = Counter(str(pair.get("target_bundle_id")) for pair in pairs)
    readiness = {
        "generated_at_utc": utc_now(),
        "target_bundles": len(bundles),
        "prior_variants": len(variants),
        "training_pairs": len(pairs),
        "identity_controls": len(identity_controls),
        "expected_pairs": len(bundles) * 5,
        "ordinary_pairs_match_expected": len(pairs) == len(bundles) * 5,
        "each_target_has_five_ordinary_variants": all(count == 5 for count in target_counts.values()),
        "split_nonempty": all(split_counts.get(split, 0) > 0 for split in ("train", "val", "test")) if pairs else False,
        "split": dict(split_counts),
        "summary": summarize(pairs),
        "recommend_mixed_train": bool(pairs),
    }
    write_json(out_dir / "trainV5_plus_readiness_report.json", readiness)
    write_text(
        out_dir / "trainV5_plus_inventory.md",
        "\n".join(
            [
                "# TrainV5 Plus Full Inventory",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- target_bundles: `{len(bundles)}`",
                f"- prior_variants: `{len(variants)}`",
                f"- training_pairs: `{len(pairs)}`",
                f"- split: `{json.dumps(dict(split_counts), ensure_ascii=False)}`",
                f"- material_family: `{json.dumps(distribution(pairs, 'material_family'), ensure_ascii=False)}`",
                f"- source_name: `{json.dumps(distribution(pairs, 'source_name'), ensure_ascii=False)}`",
                f"- prior_variant_type: `{json.dumps(distribution(pairs, 'prior_variant_type'), ensure_ascii=False)}`",
                "",
                "Only lightweight manifests/configs are stored under train/. Large rebake assets remain in output.",
            ]
        ),
    )
    write_text(
        out_dir / "trainV5_plus_readiness_report.md",
        "\n".join(
            [
                "# TrainV5 Plus Full Readiness",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- ready: `{str(readiness['ordinary_pairs_match_expected'] and readiness['split_nonempty']).lower()}`",
                f"- ordinary_pairs_match_expected: `{str(readiness['ordinary_pairs_match_expected']).lower()}`",
                f"- each_target_has_five_ordinary_variants: `{str(readiness['each_target_has_five_ordinary_variants']).lower()}`",
                f"- split_nonempty: `{str(readiness['split_nonempty']).lower()}`",
                f"- recommend_mixed_train: `{str(readiness['recommend_mixed_train']).lower()}`",
            ]
        ),
    )
    return readiness


def build_plus_full(args: argparse.Namespace, gate_pass_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    batch_local_dir = args.b_root / "trainV5_plus_full_batch_local"
    bundles, variants, pairs, identity_controls = build_plus_full_records(args.full_manifest, gate_pass_rows)
    write_plus_full_outputs(
        batch_local_dir,
        source_manifest=args.full_manifest,
        bundles=bundles,
        variants=variants,
        pairs=pairs,
        identity_controls=identity_controls,
    )

    cumulative_bundles = load_manifest_records(args.b_train_dir / "trainV5_plus_target_bundles.json")
    cumulative_variants = load_manifest_records(args.b_train_dir / "trainV5_plus_prior_variants.json")
    cumulative_pairs = load_manifest_records(args.b_train_dir / "trainV5_plus_training_pairs.json")
    existing_objects = {str(row.get("object_id") or "") for row in cumulative_bundles}

    new_bundles = [row for row in bundles if str(row.get("object_id") or "") not in existing_objects]
    new_objects = {str(row.get("object_id") or "") for row in new_bundles}
    new_variants = [row for row in variants if str(row.get("object_id") or "") in new_objects]
    new_pairs = [row for row in pairs if str(row.get("object_id") or "") in new_objects]

    merged_bundles = dedup_by_object(cumulative_bundles + new_bundles)
    merged_variants = dedup_by_key(cumulative_variants + new_variants, "prior_variant_id")
    merged_pairs = dedup_by_key(cumulative_pairs + new_pairs, "pair_id")
    merged_identity_controls = [row for row in identity_controls if str(row.get("object_id") or "") in new_objects]
    write_plus_full_outputs(
        args.b_train_dir,
        source_manifest=args.full_manifest,
        bundles=merged_bundles,
        variants=merged_variants,
        pairs=merged_pairs,
        identity_controls=merged_identity_controls,
    )
    return merged_bundles, merged_variants, merged_pairs


def merge_ab(args: argparse.Namespace, b_bundles: list[dict[str, Any]], b_variants: list[dict[str, Any]], b_pairs: list[dict[str, Any]]) -> None:
    args.merged_dir.mkdir(parents=True, exist_ok=True)
    remove_blocked_markers(args.merged_dir)
    a_bundles = load_manifest_records(args.a_dir / "trainV5_target_bundles.json")
    a_variants = load_manifest_records(args.a_dir / "trainV5_prior_variants.json")
    a_pairs = load_manifest_records(args.a_dir / "trainV5_training_pairs.json")
    a_objects = {str(row.get("object_id")) for row in a_bundles}
    allowed_b_objects = {str(row.get("object_id")) for row in b_bundles if str(row.get("object_id")) not in a_objects}
    merged_bundles = a_bundles + [row for row in b_bundles if str(row.get("object_id")) in allowed_b_objects]
    merged_variants = a_variants + [row for row in b_variants if str(row.get("object_id")) in allowed_b_objects]
    merged_pairs = a_pairs + [row for row in b_pairs if str(row.get("object_id")) in allowed_b_objects]
    merged_bundles = dedup_by_object(merged_bundles)
    merged_variants = dedup_by_key(merged_variants, "prior_variant_id")
    merged_pairs = dedup_by_key(merged_pairs, "pair_id")
    pair_ids = Counter(str(row.get("pair_id")) for row in merged_pairs)
    target_ids = Counter(str(row.get("target_bundle_id")) for row in merged_bundles)
    split_by_object: dict[str, set[str]] = defaultdict(set)
    for row in merged_pairs:
        split_by_object[str(row.get("object_id"))].add(str(row.get("split")))
    split_offenders = {key: sorted(value) for key, value in split_by_object.items() if len(value) > 1}
    readiness = {
        "generated_at_utc": utc_now(),
        "target_bundles": len(merged_bundles),
        "prior_variants": len(merged_variants),
        "training_pairs": len(merged_pairs),
        "a_target_bundles": len(a_bundles),
        "b_target_bundles_used": len(allowed_b_objects),
        "b_duplicate_objects_suppressed": len(b_bundles) - len(allowed_b_objects),
        "pair_id_unique": all(count == 1 for count in pair_ids.values()),
        "target_bundle_id_unique": all(count == 1 for count in target_ids.values()),
        "split_leakage_offenders": split_offenders,
        "summary": summarize(merged_pairs),
        "recommend_mixed_train": bool(allowed_b_objects) and not split_offenders,
    }
    write_json(args.merged_dir / "trainV5_merged_target_bundles.json", {"generated_at_utc": utc_now(), "records": merged_bundles, "summary": summarize(merged_bundles)})
    write_json(args.merged_dir / "trainV5_merged_prior_variants.json", {"generated_at_utc": utc_now(), "records": merged_variants, "summary": summarize(merged_variants)})
    write_json(args.merged_dir / "trainV5_merged_training_pairs.json", {"generated_at_utc": utc_now(), "records": merged_pairs, "summary": summarize(merged_pairs)})
    write_json(
        args.merged_dir / "trainV5_merged_sampler_config.json",
        {
            "generated_at_utc": utc_now(),
            "sampler": "TrainV5_merged_ab_multi_axis_pair_sampler_v1",
            "balance_axes": ["material_family", "source_name", "prior_variant_type", "prior_quality_bin"],
            "prior_variant_weights": {
                "near_gt_prior": 1.0,
                "mild_gap_prior": 1.0,
                "medium_gap_prior": 1.0,
                "large_gap_prior": 1.0,
                "no_prior_bootstrap": 0.5,
            },
            "source_strategy": "cap_repeated_ABO_when_B_track_non_ABO_available",
        },
    )
    write_json(args.merged_dir / "trainV5_merged_readiness_report.json", readiness)
    summary = readiness["summary"]
    write_text(
        args.merged_dir / "trainV5_merged_inventory.md",
        "\n".join(
            [
                "# TrainV5 Merged AB Inventory",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- target_bundles: `{readiness['target_bundles']}`",
                f"- prior_variants: `{readiness['prior_variants']}`",
                f"- training_pairs: `{readiness['training_pairs']}`",
                f"- material_family: `{json.dumps(summary['material_family'], ensure_ascii=False)}`",
                f"- source_name: `{json.dumps(summary['source_name'], ensure_ascii=False)}`",
                f"- prior_variant_type: `{json.dumps(summary['prior_variant_type'], ensure_ascii=False)}`",
            ]
        ),
    )
    write_text(
        args.merged_dir / "trainV5_merged_readiness_report.md",
        "\n".join(
            [
                "# TrainV5 Merged AB Readiness",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- recommend_mixed_train: `{str(readiness['recommend_mixed_train']).lower()}`",
                f"- target_bundles: `{readiness['target_bundles']}`",
                f"- training_pairs: `{readiness['training_pairs']}`",
                f"- pair_id_unique: `{str(readiness['pair_id_unique']).lower()}`",
                f"- target_bundle_id_unique: `{str(readiness['target_bundle_id_unique']).lower()}`",
                f"- split_leakage_offenders: `{json.dumps(split_offenders, ensure_ascii=False)}`",
                f"- b_duplicate_objects_suppressed: `{readiness['b_duplicate_objects_suppressed']}`",
            ]
        ),
    )
    command = """#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/ssd_work/projects/stable-fast-3d

# Draft only. Do not run from B finalization.
CUDA_VISIBLE_DEVICES=1 python scripts/train_material_refiner.py \\
  --config configs/material_refine_train_r_v2_1_view_aware.yaml \\
  --train-manifest train/trainV5_merged_ab/trainV5_merged_training_pairs.json \\
  --val-manifest train/trainV5_merged_ab/trainV5_merged_training_pairs.json \\
  --split-strategy manifest \\
  --train-split train \\
  --val-split val \\
  --output-dir output/material_refine_trainV5_abc/B_track/merged_ab_engineering_train_draft \\
  --train-balance-mode prior_variant \\
  --train-prior-variant-weights near_gt_prior=1.0,mild_gap_prior=1.0,medium_gap_prior=1.0,large_gap_prior=1.0,no_prior_bootstrap=0.5 \\
  --wandb-mode disabled
"""
    cmd_path = args.merged_dir / "trainV5_merged_command_draft.sh"
    write_text(cmd_path, command)
    cmd_path.chmod(0o755)


def update_final_report(args: argparse.Namespace) -> None:
    batch_slug = batch_slug_from_root(args.b_root)
    decision = read_json(args.b_root / f"{batch_slug}_decision.json", {})
    plus = read_json(args.b_train_dir / "trainV5_plus_readiness_report.json", {})
    merged = read_json(args.merged_dir / "trainV5_merged_readiness_report.json", {})
    pending = read_json(REPO_ROOT / "output/material_refine_trainV5_abc/B_track/pending_repair/pending_repair_manifest.json", {"records": []})
    lines = [
        "# TrainV5 ABC Implementation Report",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        "",
        "## B Track Update",
        "",
        f"- batch_name: `{batch_slug}`",
        f"- batch_rebake_completed: `{decision.get('status') == 'completed'}`",
        f"- full_rebake_launched: `{decision.get('full_rebake_launched')}`",
        f"- prepared_records: `{decision.get('prepared_records')}`",
        f"- skipped_records: `{decision.get('skipped_records')}`",
        f"- target_gate_version: `{decision.get('target_gate_version')}`",
        f"- target_truth_gate_pass: `{decision.get('target_truth_gate_pass')}`",
        f"- target_truth_gate_fail: `{decision.get('target_truth_gate_fail')}`",
        f"- pass_rate: `{decision.get('pass_rate')}`",
        f"- target_prior_relation_diagnostic: `{json.dumps(decision.get('target_prior_relation_diagnostic', {}), ensure_ascii=False)}`",
        f"- TrainV5_plus_full_target_bundles: `{plus.get('target_bundles')}`",
        f"- TrainV5_plus_full_training_pairs: `{plus.get('training_pairs')}`",
        f"- merged_ab_target_bundles: `{merged.get('target_bundles')}`",
        f"- merged_ab_training_pairs: `{merged.get('training_pairs')}`",
        f"- merged_material_distribution: `{json.dumps(merged.get('summary', {}).get('material_family', {}), ensure_ascii=False)}`",
        f"- merged_source_distribution: `{json.dumps(merged.get('summary', {}).get('source_name', {}), ensure_ascii=False)}`",
        f"- merged_prior_distribution: `{json.dumps(merged.get('summary', {}).get('prior_variant_type', {}), ensure_ascii=False)}`",
        f"- recommend_mixed_train: `{merged.get('recommend_mixed_train')}`",
        f"- pending_repair_preserved: `{len(records(pending))}`",
        "- raw_assets_deleted: `false`",
        "- paper_claim_written: `false`",
    ]
    write_text(REPO_ROOT / "output/material_refine_trainV5_abc/TrainV5_ABC_final_report.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--b-root", type=Path, default=DEFAULT_B_ROOT)
    parser.add_argument("--full-manifest", type=Path, default=DEFAULT_B_ROOT / "full_1155_rebake_manifest.json")
    parser.add_argument("--a-dir", type=Path, default=DEFAULT_A_DIR)
    parser.add_argument("--b-train-dir", type=Path, default=DEFAULT_B_TRAIN_DIR)
    parser.add_argument("--merged-dir", type=Path, default=DEFAULT_MERGED_DIR)
    parser.add_argument("--run-contract-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-merge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate_pass_rows, _gate_fail_rows = finalize_rebake(args)
    b_bundles, b_variants, b_pairs = build_plus_full(args, gate_pass_rows)
    if not args.skip_merge:
        merge_ab(args, b_bundles, b_variants, b_pairs)
    update_final_report(args)


if __name__ == "__main__":
    main()
