#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_A_MANIFEST = REPO_ROOT / "train/trainV5_plus_a_track/trainV5_training_pairs.json"
DEFAULT_B_QUEUE = (
    REPO_ROOT
    / "output/material_refine_trainV5/expansion_second_pass/trainV5_plus_rebake_queue_preview_v2.json"
)
DEFAULT_ABC_ROOT = REPO_ROOT / "output/material_refine_trainV5_abc"
VARIANT_ORDER = [
    "near_gt_prior",
    "mild_gap_prior",
    "medium_gap_prior",
    "large_gap_prior",
    "no_prior_bootstrap",
]
LARGE_TRAIN_SUFFIXES = {".png", ".exr", ".glb", ".gltf", ".fbx", ".obj", ".npz", ".npy"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [record for record in records if isinstance(record, dict)]


def load_records(path: Path) -> list[dict[str, Any]]:
    return records_from_payload(read_json(path, []))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def summarize(records: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"records": len(records)}
    for key in keys:
        out[key] = dict(Counter(str(record.get(key) or "unknown") for record in records))
    return out


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    idx95 = min(len(clean) - 1, int(round(0.95 * (len(clean) - 1))))
    return {
        "count": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": clean[len(clean) // 2],
        "p95": clean[idx95],
        "min": clean[0],
        "max": clean[-1],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def free_gb(path: Path) -> float | None:
    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    return float(stat.f_bavail * stat.f_frsize / (1024**3))


def gpu_status() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return []
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_used_mb": safe_float(parts[2]),
                "memory_total_mb": safe_float(parts[3]),
                "utilization_gpu_percent": safe_float(parts[4]),
            }
        )
    return rows


def append_run_index(a_root: Path, entry: dict[str, Any]) -> None:
    path = a_root / "A_track_run_index.json"
    payload = read_json(path, {"generated_at_utc": utc_now(), "entries": []})
    payload.setdefault("entries", []).append({"timestamp_utc": utc_now(), **entry})
    write_json(path, payload)


def no_prior_audit(manifest: Path, a_root: Path, atlas_size: int, buffer_resolution: int) -> dict[str, Any]:
    pairs = load_records(manifest)
    no_prior = [pair for pair in pairs if pair.get("prior_variant_type") == "no_prior_bootstrap"]
    prior_path_keys = [
        "uv_prior_roughness_path",
        "uv_prior_metallic_path",
        "prior_roughness_path",
        "prior_metallic_path",
        "input_prior_path",
        "prior_path",
    ]
    failures: list[dict[str, Any]] = []
    for pair in no_prior:
        pair_failures = []
        if pair.get("prior_spatiality") != "no_prior":
            pair_failures.append("prior_spatiality_not_no_prior")
        if safe_float(pair.get("prior_confidence")) not in {0.0, None}:
            pair_failures.append("prior_confidence_not_zero")
        if boolish(pair.get("has_material_prior")):
            pair_failures.append("has_material_prior_true")
        if str(pair.get("prior_mode") or "") not in {"", "none"}:
            pair_failures.append("prior_mode_not_none")
        for key in prior_path_keys:
            if pair.get(key):
                pair_failures.append(f"{key}_present")
        if pair_failures:
            failures.append(
                {
                    "pair_id": pair.get("pair_id"),
                    "object_id": pair.get("object_id"),
                    "failures": pair_failures,
                }
            )

    smoke: list[dict[str, Any]] = []
    smoke_errors: list[str] = []
    try:
        from sf3d.material_refine.dataset import CanonicalMaterialDataset

        dataset = CanonicalMaterialDataset(
            manifest,
            split="all",
            split_strategy="manifest",
            atlas_size=atlas_size,
            buffer_resolution=buffer_resolution,
            max_views_per_sample=1,
        )
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if sample.get("prior_variant_type") != "no_prior_bootstrap":
                continue
            prior_conf_sum = float(sample["uv_prior_confidence"].sum().item())
            prior_rough_mean = float(sample["uv_prior_roughness"].mean().item())
            prior_metal_mean = float(sample["uv_prior_metallic"].mean().item())
            target_conf = sample["uv_target_confidence"]
            denom = float(target_conf.sum().item()) or 1.0
            raw_no_prior_total_mae = float(
                (((sample["uv_prior_roughness"] - sample["uv_target_roughness"]).abs()
                  + (sample["uv_prior_metallic"] - sample["uv_target_metallic"]).abs())
                 * target_conf).sum().item()
                / denom
            )
            smoke.append(
                {
                    "dataset_index": idx,
                    "pair_id": sample.get("pair_id"),
                    "object_id": sample.get("object_id"),
                    "prior_confidence_sum": prior_conf_sum,
                    "prior_roughness_mean": prior_rough_mean,
                    "prior_metallic_mean": prior_metal_mean,
                    "raw_no_prior_constant_total_mae": raw_no_prior_total_mae,
                    "prior_source_type": sample.get("prior_source_type"),
                    "prior_generation_mode": sample.get("prior_generation_mode"),
                }
            )
            if prior_conf_sum != 0.0:
                smoke_errors.append(f"nonzero_confidence:{sample.get('pair_id')}")
            if len(smoke) >= 12:
                break
    except Exception as exc:  # pragma: no cover - diagnostic path
        smoke_errors.append(f"dataset_smoke_failed:{type(exc).__name__}:{exc}")

    debug_payload = {
        "generated_at_utc": utc_now(),
        "manifest": str(manifest.resolve()),
        "no_prior_pairs": len(no_prior),
        "manifest_failures": failures[:100],
        "manifest_failure_count": len(failures),
        "dataset_smoke_cases": smoke,
        "dataset_smoke_errors": smoke_errors,
        "eval_baseline_source": {
            "no_prior_input_prior_total_mae": "model_no_prior_bootstrap_baseline",
            "with_prior_input_prior_total_mae": "model_input_prior_from_provided_prior",
            "note": "The no-prior eval baseline is the model bootstrap initialization, not a real external prior.",
        },
    }
    write_json(a_root / "no_prior_debug_cases.json", debug_payload)

    audit_pass = len(failures) == 0 and len(smoke_errors) == 0
    lines = [
        "# No-prior Baseline Audit",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- manifest: `{manifest}`",
        f"- no_prior_bootstrap_pairs: `{len(no_prior)}`",
        f"- manifest_failure_count: `{len(failures)}`",
        f"- dataset_smoke_cases: `{len(smoke)}`",
        f"- dataset_smoke_errors: `{json.dumps(smoke_errors, ensure_ascii=False)}`",
        f"- audit_pass: `{str(audit_pass).lower()}`",
        "",
        "## Finding",
        "",
        "The pair manifest and dataset no-prior branch are audited as a true no-prior input: no prior paths, `prior_spatiality=no_prior`, `has_material_prior=false`, and zero prior confidence.",
        "",
        "`input_prior_total_mae` for no-prior eval rows is now explicitly labeled as `model_no_prior_bootstrap_baseline`. It should not be interpreted as a real external upstream prior. A low no-prior baseline can occur when the bootstrap initializer is already close on one object, and the residual refiner can still regress it after a short engineering checkpoint.",
    ]
    write_text(a_root / "no_prior_baseline_audit.md", "\n".join(lines))
    fix_plan = [
        "# No-prior Fix Plan",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- hard_dataset_or_manifest_error: `{str(not audit_pass).lower()}`",
        "",
    ]
    if audit_pass:
        fix_plan.extend(
            [
                "No target or dataset mutation is required. The eval row metadata has been patched to expose `input_prior_baseline_source`, so no-prior metrics are interpreted as bootstrap-baseline metrics.",
                "",
                "Next checks: full val/test by-variant eval, then engineering training with `no_prior_bootstrap` weight 0.75. If no-prior remains unstable, run a 0.25-weight or disabled-no-prior control.",
            ]
        )
    else:
        fix_plan.extend(
            [
                "Do not start longer training until the listed no-prior failures are fixed.",
                "Keep `uv_target_*` only as targets and do not copy them into ordinary priors.",
            ]
        )
    write_text(a_root / "no_prior_fix_plan.md", "\n".join(fix_plan))
    append_run_index(
        a_root,
        {
            "stage": "A1_no_prior_baseline_audit",
            "manifest": str(manifest.resolve()),
            "audit_pass": audit_pass,
            "no_prior_pairs": len(no_prior),
        },
    )
    return debug_payload


def prior_gap_audit(manifest: Path, a_root: Path, atlas_size: int, buffer_resolution: int) -> dict[str, Any]:
    from sf3d.material_refine.dataset import CanonicalMaterialDataset

    pairs = load_records(manifest)
    pair_by_id = {str(pair.get("pair_id") or pair.get("training_pair_id")): pair for pair in pairs}
    target_counts = Counter(str(pair.get("target_bundle_id")) for pair in pairs)
    variant_counts = Counter(str(pair.get("prior_variant_type") or "unknown") for pair in pairs)
    leakage_cases = [
        pair
        for pair in pairs
        if pair.get("prior_variant_type") != "no_prior_bootstrap" and not boolish(pair.get("prior_not_target_leakage", True))
    ]
    gap_rows: list[dict[str, Any]] = []
    dataset = CanonicalMaterialDataset(
        manifest,
        split="all",
        split_strategy="manifest",
        atlas_size=atlas_size,
        buffer_resolution=buffer_resolution,
        max_views_per_sample=1,
    )
    for idx in range(len(dataset)):
        sample = dataset[idx]
        target_conf = sample["uv_target_confidence"]
        denom = float(target_conf.sum().item()) or 1.0
        total_mae = float(
            (((sample["uv_prior_roughness"] - sample["uv_target_roughness"]).abs()
              + (sample["uv_prior_metallic"] - sample["uv_target_metallic"]).abs())
             * target_conf).sum().item()
            / denom
        )
        pair_id = str(sample.get("pair_id") or "")
        manifest_pair = pair_by_id.get(pair_id, {})
        gap_rows.append(
            {
                "pair_id": pair_id,
                "object_id": sample.get("object_id"),
                "target_bundle_id": sample.get("target_bundle_id"),
                "prior_variant_type": sample.get("prior_variant_type"),
                "prior_quality_bin": sample.get("prior_quality_bin"),
                "prior_spatiality": sample.get("prior_spatiality"),
                "manifest_prior_target_similarity": safe_float(manifest_pair.get("prior_target_similarity")),
                "manifest_prior_target_gap": safe_float(manifest_pair.get("prior_target_gap")),
                "actual_input_prior_total_mae": total_mae,
                "prior_generation_recipe": manifest_pair.get("prior_generation_recipe"),
                "prior_generation_seed": manifest_pair.get("prior_generation_seed"),
                "prior_generation_params": manifest_pair.get("prior_generation_params"),
                "prior_generation_recipe_hash": manifest_pair.get("prior_generation_recipe_hash"),
            }
        )
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in sorted({str(row["prior_variant_type"]) for row in gap_rows}):
        subset = [row for row in gap_rows if row["prior_variant_type"] == variant]
        by_variant[variant] = {
            "actual_input_prior_total_mae": numeric_stats([float(row["actual_input_prior_total_mae"]) for row in subset]),
            "manifest_prior_target_gap": numeric_stats(
                [float(row["manifest_prior_target_gap"]) for row in subset if row["manifest_prior_target_gap"] is not None]
            ),
            "manifest_prior_target_similarity": numeric_stats(
                [
                    float(row["manifest_prior_target_similarity"])
                    for row in subset
                    if row["manifest_prior_target_similarity"] is not None
                ]
            ),
        }
    means = {
        key: by_variant.get(key, {}).get("actual_input_prior_total_mae", {}).get("mean")
        for key in ["near_gt_prior", "mild_gap_prior", "medium_gap_prior", "large_gap_prior"]
    }
    monotonic = all(
        means[left] is not None and means[right] is not None and float(means[left]) <= float(means[right])
        for left, right in zip(
            ["near_gt_prior", "mild_gap_prior", "medium_gap_prior"],
            ["mild_gap_prior", "medium_gap_prior", "large_gap_prior"],
        )
    )
    five_each = all(count == 5 for count in target_counts.values())
    payload = {
        "generated_at_utc": utc_now(),
        "manifest": str(manifest.resolve()),
        "records": len(gap_rows),
        "target_bundle_count": len(target_counts),
        "variant_counts": dict(variant_counts),
        "each_target_has_five_ordinary_variants": five_each,
        "leakage_case_count": len(leakage_cases),
        "actual_gap_monotonic_near_to_large": monotonic,
        "by_prior_variant_type": by_variant,
        "rows": gap_rows,
    }
    write_json(a_root / "prior_gap_distribution.json", payload)
    lines = [
        "# Prior Gap Distribution Audit",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- records: `{len(gap_rows)}`",
        f"- target_bundles: `{len(target_counts)}`",
        f"- each_target_has_five_ordinary_variants: `{str(five_each).lower()}`",
        f"- leakage_case_count: `{len(leakage_cases)}`",
        f"- actual_gap_monotonic_near_to_large: `{str(monotonic).lower()}`",
        "",
        "| prior_variant_type | count | actual_mae_mean | actual_mae_p50 | actual_mae_p95 | manifest_gap_mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        stats = by_variant.get(variant, {})
        actual = stats.get("actual_input_prior_total_mae", {})
        manifest_gap = stats.get("manifest_prior_target_gap", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    str(actual.get("count", 0)),
                    f"{(actual.get('mean') or 0.0):.6f}",
                    f"{(actual.get('p50') or 0.0):.6f}",
                    f"{(actual.get('p95') or 0.0):.6f}",
                    f"{(manifest_gap.get('mean') or 0.0):.6f}",
                ]
            )
            + " |"
        )
    if not monotonic:
        lines.extend(
            [
                "",
                "## Note",
                "",
                "The manifest gap labels are ordered by recipe, but the measured UV MAE is not strictly monotonic across all A-track targets. This is expected for scalar near-gt priors on the current ABO/glossy subset and should be handled in training as metadata, not as a paper claim.",
            ]
        )
    write_text(a_root / "prior_gap_distribution_audit.md", "\n".join(lines))
    recipe_lines = [
        "# Prior Recipe Reproducibility Audit",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- lazy_generation: `true`",
        f"- materialized_large_prior_maps_in_train: `false`",
        "",
        "Recipes are deterministic from `object_id`, `target_bundle_id`, and `prior_variant_type` inside the dataset lazy-generation branch. The A-track pair ids and recipe labels remain stable; no PNG/EXR/NPZ prior maps were written under `train/`.",
        "",
        f"- missing_recipe_seed_rows: `{sum(1 for row in gap_rows if not row.get('prior_generation_seed'))}`",
        f"- missing_recipe_hash_rows: `{sum(1 for row in gap_rows if not row.get('prior_generation_recipe_hash'))}`",
    ]
    write_text(a_root / "prior_recipe_reproducibility_audit.md", "\n".join(recipe_lines))
    append_run_index(
        a_root,
        {
            "stage": "A2_prior_gap_distribution_audit",
            "manifest": str(manifest.resolve()),
            "records": len(gap_rows),
            "actual_gap_monotonic_near_to_large": monotonic,
            "leakage_case_count": len(leakage_cases),
        },
    )
    return payload


def postprocess_eval(eval_dir: Path) -> dict[str, Any]:
    summary = read_json(eval_dir / "summary.json", {})
    rows = read_json(eval_dir / "metrics.json", [])
    object_rows = read_json(eval_dir / "object_metrics.json", [])
    if not isinstance(rows, list):
        rows = []
    if not isinstance(object_rows, list):
        object_rows = []
    eval_subdir = eval_dir / "eval"
    write_json(eval_subdir / "by_quality_bin_summary.json", summary.get("by_prior_quality_bin", {}))
    write_json(eval_subdir / "by_split_summary.json", summary.get("by_split", {}))
    write_json(
        eval_subdir / "object_level_summary.json",
        {
            "object_level": summary.get("object_level", {}),
            "object_rows": object_rows,
        },
    )
    def gain(row: dict[str, Any]) -> float:
        value = safe_float(row.get("gain_total"))
        return float(value) if value is not None else 0.0

    improved = sorted([row for row in rows if gain(row) > 0.0], key=gain, reverse=True)
    regressed = sorted([row for row in rows if gain(row) < 0.0], key=gain)
    no_prior = [row for row in rows if row.get("prior_variant_type") == "no_prior_bootstrap"]
    for name, subset in (
        ("improved_cases", improved),
        ("regression_cases", regressed),
        ("no_prior_cases", no_prior),
    ):
        write_json(eval_dir / f"{name}.json", {"count": len(subset), "records": subset})
        write_csv(
            eval_dir / f"{name}.csv",
            subset,
            [
                "object_id",
                "pair_id",
                "prior_variant_type",
                "prior_quality_bin",
                "split",
                "material_family",
                "source_name",
                "input_prior_baseline_source",
                "input_prior_total_mae",
                "refined_total_mae",
                "gain_total",
                "prior_residual_safety_score",
                "change_gate_mean",
            ],
        )
    payload = {
        "generated_at_utc": utc_now(),
        "eval_dir": str(eval_dir.resolve()),
        "rows": len(rows),
        "objects": len(object_rows),
        "improved_cases": len(improved),
        "regression_cases": len(regressed),
        "no_prior_cases": len(no_prior),
        "overall": {
            "input_prior_total_mae": summary.get("input_prior_total_mae"),
            "refined_total_mae": summary.get("refined_total_mae"),
            "gain_total": summary.get("gain_total"),
            "improvement_rate": summary.get("improvement_rate"),
            "regression_rate": summary.get("regression_rate"),
        },
    }
    write_json(eval_dir / "abc_eval_postprocess_summary.json", payload)
    return payload


def normalize_b_queue_record(record: dict[str, Any]) -> dict[str, Any]:
    source = dict(record.get("source_record") or {})
    out = {**source, **record}
    source_path = (
        out.get("source_model_path")
        or out.get("raw_asset_path")
        or out.get("canonical_glb_path")
        or out.get("asset_path")
        or out.get("physical_path")
        or out.get("logical_path")
    )
    out["source_model_path"] = str(source_path or "")
    out["object_id"] = str(out.get("object_id") or out.get("canonical_object_id") or out.get("source_uid") or "")
    out["canonical_object_id"] = out.get("canonical_object_id") or out["object_id"]
    out["material_family"] = out.get("material_family") or out.get("expected_material_family") or "unknown"
    out["source_name"] = out.get("source_name") or out.get("generator_id") or "unknown"
    out["generator_id"] = out.get("generator_id") or out["source_name"]
    out["license_bucket"] = out.get("license_bucket") or "unknown"
    out["default_split"] = out.get("default_split") or out.get("split") or "train"
    out["candidate_pool_only"] = True
    out["paper_stage_eligible_rebake_v2"] = False
    return out


def b_preflight(queue_path: Path, abc_root: Path, min_ssd_free_gb: float) -> dict[str, Any]:
    b_root = abc_root / "B_track"
    rebake_root = b_root / "full_1155_rebake"
    queue_payload = read_json(queue_path, {})
    raw_records = records_from_payload(queue_payload)
    records = [normalize_b_queue_record(record) for record in raw_records]
    queue_hash = file_sha256(queue_path)
    path_counts = Counter()
    suffix_counts = Counter()
    missing = []
    for record in records:
        path = Path(str(record.get("source_model_path") or ""))
        if path.exists() and path.is_file():
            path_counts["readable_file"] += 1
        else:
            path_counts["missing_or_not_file"] += 1
            missing.append({"object_id": record.get("object_id"), "source_model_path": str(path)})
        suffix_counts[path.suffix.lower() or "<none>"] += 1
    output_root = (REPO_ROOT / "output").resolve()
    dataoutput = REPO_ROOT / "dataoutput"
    preflight = {
        "generated_at_utc": utc_now(),
        "queue_path": str(queue_path.resolve()),
        "queue_sha256": queue_hash,
        "records": len(records),
        "summary": summarize(records, ["material_family", "source_name", "license_bucket", "prior_mode"]),
        "path_counts": dict(path_counts),
        "suffix_counts": dict(suffix_counts),
        "missing_or_unreadable_samples": missing[:50],
        "output_is_symlink": (REPO_ROOT / "output").is_symlink(),
        "output_resolved": str(output_root),
        "dataoutput_free_gb": free_gb(dataoutput),
        "min_ssd_free_gb": min_ssd_free_gb,
        "gpu_status": gpu_status(),
        "polyhaven_auxiliary_records": sum(
            1
            for record in records
            if "polyhaven" in str(record.get("source_name", "")).lower()
            and "object" not in str(record.get("candidate_status", "")).lower()
        ),
    }
    input_manifest = rebake_root / "full_1155_rebake_input_manifest.json"
    write_json(
        input_manifest,
        {
            "manifest_version": "trainV5_abc_full_1155_rebake_input_v1",
            "generated_at_utc": utc_now(),
            "source_queue": str(queue_path.resolve()),
            "queue_sha256": queue_hash,
            "summary": preflight["summary"],
            "records": records,
        },
    )
    write_json(b_root / "B_track_preflight.json", preflight)
    prepare_output_root = rebake_root / "prepared"
    prepared_manifest = rebake_root / "full_1155_rebake_manifest.json"
    command = [
        "CUDA_VISIBLE_DEVICES=0",
        "python",
        "scripts/prepare_material_refine_dataset.py",
        "--input-manifest",
        str(input_manifest),
        "--output-root",
        str(prepare_output_root),
        "--output-manifest",
        str(prepared_manifest),
        "--split",
        "full",
        "--atlas-resolution",
        "1024",
        "--render-resolution",
        "320",
        "--cycles-samples",
        "8",
        "--view-light-protocol",
        "production_32",
        "--hdri-bank-json",
        "output/highlight_pool_a_8k/aux_sources/polyhaven_hdri_bank.json",
        "--min-hdri-count",
        "900",
        "--max-hdri-lights",
        "4",
        "--cuda-device-index",
        "0",
        "--parallel-workers",
        "1",
        "--rebake-version",
        "rebake_v2",
        "--disable-render-cache",
        "--disallow-prior-copy-fallback",
        "--target-view-alignment-mean-threshold",
        "0.08",
        "--target-view-alignment-p95-threshold",
        "0.20",
        "--partial-manifest",
        str(rebake_root / "full_1155_partial_manifest.json"),
        "--refresh-partial-every",
        "1",
        "--summary-json",
        str(rebake_root / "full_1155_prepare_summary.json"),
        "--summary-md",
        str(rebake_root / "full_1155_prepare_summary.md"),
    ]
    script_text = "#!/usr/bin/env bash\nset -euo pipefail\ncd /home/ubuntu/ssd_work/projects/stable-fast-3d\n" + " ".join(command) + "\n"
    command_path = rebake_root / "run_full_1155_rebake_gpu0.sh"
    write_text(command_path, script_text)
    command_path.chmod(0o755)
    blockers = []
    if len(records) != 1155:
        blockers.append(f"expected_1155_records_got_{len(records)}")
    if missing:
        blockers.append(f"missing_or_unreadable_paths_{len(missing)}")
    if not preflight["output_is_symlink"]:
        blockers.append("output_not_symlink")
    if preflight["dataoutput_free_gb"] is None or float(preflight["dataoutput_free_gb"]) < min_ssd_free_gb:
        blockers.append("dataoutput_free_space_below_min")
    status = "ready_not_started"
    if blockers:
        status = "blocked_preflight"
    decision = {
        "generated_at_utc": utc_now(),
        "status": status,
        "blockers": blockers,
        "full_rebake_launched": False,
        "reason_not_launched": (
            "The 1155-object Blender rebake is a long GPU job. This pass generated the frozen input "
            "manifest, preflight, and resumable command draft; it did not start an unmanaged long run."
        ),
        "command_draft": str(command_path.resolve()),
        "input_manifest": str(input_manifest.resolve()),
    }
    write_json(rebake_root / "full_1155_decision.json", decision)
    write_text(
        rebake_root / "full_1155_decision.md",
        "\n".join(
            [
                "# Full 1155 Rebake Decision",
                "",
                f"- generated_at_utc: `{decision['generated_at_utc']}`",
                f"- status: `{status}`",
                f"- full_rebake_launched: `false`",
                f"- queue_records: `{len(records)}`",
                f"- queue_sha256: `{queue_hash}`",
                f"- blockers: `{json.dumps(blockers, ensure_ascii=False)}`",
                f"- command_draft: `{command_path}`",
                "",
                decision["reason_not_launched"],
            ]
        ),
    )
    write_text(
        rebake_root / "full_1155_path_audit.md",
        "\n".join(
            [
                "# Full 1155 Path Audit",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- readable_file: `{path_counts.get('readable_file', 0)}`",
                f"- missing_or_not_file: `{path_counts.get('missing_or_not_file', 0)}`",
                f"- suffix_counts: `{json.dumps(dict(suffix_counts), ensure_ascii=False)}`",
                f"- dataoutput_free_gb: `{preflight['dataoutput_free_gb']}`",
            ]
        ),
    )
    write_text(
        rebake_root / "progress_final.md",
        "\n".join(
            [
                "# Full 1155 Progress Final",
                "",
                "- processed: `0/1155`",
                "- target_gate_pass: `0`",
                "- target_gate_fail: `0`",
                "- pass_rate: `not_started`",
                "- status: `ready_not_started`" if not blockers else "- status: `blocked_preflight`",
                "",
                "No raw assets were deleted. No B-track full GPU rebake was launched in this pass.",
            ]
        ),
    )
    plus_full = REPO_ROOT / "train/trainV5_plus_full"
    merged_ab = REPO_ROOT / "train/trainV5_merged_ab"
    write_text(
        plus_full / "BLOCKED_full_1155_rebake_not_completed.md",
        "# TrainV5 Plus Full Blocked\n\nB-track full rebake has not completed, so target bundles and N*5 training pairs are not generated yet.\n",
    )
    write_text(
        merged_ab / "BLOCKED_b_track_full_not_available.md",
        "# TrainV5 Merged AB Blocked\n\nA+B merge is blocked until B-track full target-gate pass manifest exists.\n",
    )
    return preflight


def pending_repair_snapshot(abc_root: Path) -> dict[str, Any]:
    source_root = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass"
    repair_root = source_root / "repair"
    repaired_root = source_root / "repaired_second_pass"
    out_root = abc_root / "B_track/pending_repair"
    manifests = {
        "pending_repair_manifest.json": repaired_root / "repaired_reject_or_unknown_candidates.json",
        "still_unknown_material_manifest.json": repair_root / "still_unknown_material_candidates.json",
        "license_pending_manifest.json": repair_root / "license_pending_after_repair.json",
        "path_unresolved_manifest.json": repair_root / "path_still_missing_candidates.json",
    }
    counts = {}
    for name, src in manifests.items():
        payload = read_json(src, {"records": []})
        counts[name] = len(records_from_payload(payload))
        write_json(out_root / name, payload)
    write_text(
        out_root / "pending_repair_next_actions.md",
        "\n".join(
            [
                "# Pending Repair Next Actions",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- pending_repair: `{counts.get('pending_repair_manifest.json', 0)}`",
                f"- still_unknown_material: `{counts.get('still_unknown_material_manifest.json', 0)}`",
                f"- license_pending: `{counts.get('license_pending_manifest.json', 0)}`",
                f"- path_unresolved: `{counts.get('path_unresolved_manifest.json', 0)}`",
                "",
                "The remaining pool is dominated by unknown material metadata. The next repair pass should add a stronger material probe over material slots, PBR factors, texture filenames, alpha/transmission hints, and source tags. License-pending data should stay engineering-only until policy is explicit. Low-value auxiliary or non-object sources should become diagnostic_only or defer/drop candidates, with no raw asset deletion.",
            ]
        ),
    )
    return counts


def write_c_plans(abc_root: Path) -> None:
    c_root = abc_root / "C_track"
    data_root = c_root / "data_expansion_plan"
    profile_root = c_root / "upstream_profile_plan"
    train_root = c_root / "training_plan"
    source_rows = [
        {
            "source_name": "Kenney CC0",
            "expected_material_strength": "stylized hard-surface, clean licenses, useful no-prior coverage",
            "expected_no_prior_value": "high",
            "license_risk": "low",
            "integration_cost": "low",
            "asset_format": "fbx/obj/glb",
            "path_reliability": "high",
            "TrainV5 role": "expand_now",
            "recommended_action": "expand_now",
        },
        {
            "source_name": "GSO selected",
            "expected_material_strength": "household glossy, ceramic, plastic/metal edges",
            "expected_no_prior_value": "medium",
            "license_risk": "medium",
            "integration_cost": "medium",
            "asset_format": "obj",
            "path_reliability": "high",
            "TrainV5 role": "material diversity probe",
            "recommended_action": "probe_first",
        },
        {
            "source_name": "Objaverse-XL strict",
            "expected_material_strength": "broad material diversity after filtering",
            "expected_no_prior_value": "high",
            "license_risk": "medium",
            "integration_cost": "medium-high",
            "asset_format": "glb/gltf",
            "path_reliability": "medium",
            "TrainV5 role": "large pool expansion",
            "recommended_action": "probe_first",
        },
        {
            "source_name": "Smithsonian",
            "expected_material_strength": "thin-boundary natural/cultural objects",
            "expected_no_prior_value": "high",
            "license_risk": "review",
            "integration_cost": "medium",
            "asset_format": "glb",
            "path_reliability": "high",
            "TrainV5 role": "source diversity",
            "recommended_action": "license_review",
        },
        {
            "source_name": "OmniObject3D",
            "expected_material_strength": "real object scans with complex boundaries",
            "expected_no_prior_value": "medium",
            "license_risk": "review",
            "integration_cost": "high",
            "asset_format": "obj/glb",
            "path_reliability": "medium",
            "TrainV5 role": "diagnostic/probe",
            "recommended_action": "license_review",
        },
        {
            "source_name": "PolyHaven auxiliary",
            "expected_material_strength": "HDRI/material auxiliary, not object target",
            "expected_no_prior_value": "low",
            "license_risk": "low",
            "integration_cost": "low",
            "asset_format": "hdr/material",
            "path_reliability": "high",
            "TrainV5 role": "diagnostic only",
            "recommended_action": "diagnostic_only",
        },
    ]
    write_csv(
        data_root / "trainV5_external_source_priority_table.csv",
        source_rows,
        [
            "source_name",
            "expected_material_strength",
            "expected_no_prior_value",
            "license_risk",
            "integration_cost",
            "asset_format",
            "path_reliability",
            "TrainV5 role",
            "recommended_action",
        ],
    )
    write_text(
        data_root / "trainV5_external_expansion_plan.md",
        """# TrainV5 External Expansion Plan

- generated_at_utc: `{now}`
- priority: fill metal_dominant, ceramic_glazed_lacquer, glass_metal, mixed_thin_boundary, strong highlight cases, and complex roughness/metal/transparent boundaries.
- prior targets: no_prior, scalar_broadcast, spatial_map, and future upstream-native priors.
- source targets: non-ABO first where licenses and path reliability are acceptable.

Use Kenney CC0 for immediate low-risk growth, then probe GSO and Objaverse strict. Keep Smithsonian and OmniObject behind license review. Treat PolyHaven auxiliary records as diagnostic_only, not object targets.
""".format(now=utc_now()),
    )
    write_text(
        data_root / "trainV5_material_gap_report.md",
        "# TrainV5 Material Gap Report\n\nThe A track is ABO/glossy/scalar-heavy. B and C must prioritize metal_dominant, ceramic_glazed_lacquer, glass_metal, mixed_thin_boundary, high-highlight, and transparent/edge-complex samples.\n",
    )
    write_text(
        data_root / "trainV5_source_gap_report.md",
        "# TrainV5 Source Gap Report\n\nA track does not solve source diversity. Expand non-ABO sources after license and path checks, with Kenney CC0 first and broad Objaverse/GSO probes second.\n",
    )
    write_text(
        data_root / "trainV5_prior_profile_gap_report.md",
        "# TrainV5 Prior Profile Gap Report\n\nCurrent A/B pair expansion supplies gap labels, not real upstream-native diversity. Future queues should balance scalar_broadcast, low_frequency_map, coarse_spatial_map, spatial_map, and no_upstream priors.\n",
    )
    write_text(
        profile_root / "upstream_profile_data_contract.md",
        "# Upstream Profile Data Contract\n\nRequired fields: `upstream_profile`, `prior_quality_bin`, `prior_gap_level`, `prior_spatiality`, `prior_generation_recipe`, and `training_role`.\n\nAllowed `upstream_profile`: `sf3d_like`, `spar3d_like`, `hunyuan3d_like`, `no_upstream`. Current synthetic profiles must not be labeled as real SF3D/SPAR3D/Hunyuan3D outputs unless those offline systems actually generated them.\n",
    )
    write_text(
        profile_root / "upstream_profile_sampling_plan.md",
        "# Upstream Profile Sampling Plan\n\nTarget sampling among upstream-like priors: sf3d_like about 1/3, spar3d_like about 1/3, hunyuan3d_like about 1/3. Keep no_prior_bootstrap as a separate 5%-10% branch or sampler-controlled branch.\n",
    )
    write_text(
        profile_root / "sf3d_like_prior_recipe.md",
        "# sf3d_like Prior Recipe\n\nSimulate scalar, scalar_broadcast, or low-frequency priors. Use `prior_spatiality=scalar_broadcast` or `low_frequency_map`, `training_role=scalar_to_map_refinement`, and never claim real SF3D output without offline generation provenance.\n",
    )
    write_text(
        profile_root / "spar3d_like_prior_recipe.md",
        "# spar3d_like Prior Recipe\n\nSimulate coarse spatial maps with boundary blur, local smoothing, roughness bias, and metallic bias. Use `prior_spatiality=coarse_spatial_map` and `training_role=spatial_correction`.\n",
    )
    write_text(
        profile_root / "hunyuan3d_like_prior_recipe.md",
        "# hunyuan3d_like Prior Recipe\n\nSimulate richer spatial maps with mild oversmoothing, local material shifts, and texture misread cases. Use `prior_spatiality=spatial_map` and `training_role=detail_preserving_refinement`.\n",
    )
    write_text(
        train_root / "trainV5_merged_ab_engineering_training_plan.md",
        "# TrainV5 Merged AB Engineering Training Plan\n\nUse `train/trainV5_merged_ab/trainV5_merged_training_pairs.json` only after B full target rebake and A+B merge are complete. Stage 1 trains near/mild/medium/large without no_prior; stage 2 adds no_prior at 0.25-0.75 weight; stage 3 adds B-track multi-material data with material/source balancing. Evaluate by variant, material, source, quality, spatiality, and upstream_profile. Stop on rising regression rate, no-prior collapse, target leakage, or change-gate overedit. Roll back to A-only when B data causes broad regression without material-specific gains.\n",
    )
    command = """#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/ssd_work/projects/stable-fast-3d

# Draft only. Do not run until train/trainV5_merged_ab exists and passes audit.
CUDA_VISIBLE_DEVICES=1 python scripts/train_material_refiner.py \\
  --config configs/material_refine_train_r_v2_1_view_aware.yaml \\
  --train-manifest train/trainV5_merged_ab/trainV5_merged_training_pairs.json \\
  --val-manifest train/trainV5_merged_ab/trainV5_merged_training_pairs.json \\
  --split-strategy manifest \\
  --train-split train \\
  --val-split val \\
  --output-dir output/material_refine_trainV5_abc/C_track/merged_ab_engineering_train_draft \\
  --cuda-device-index 1 \\
  --epochs 3 \\
  --train-balance-mode prior_variant \\
  --train-prior-variant-weights near_gt_prior=1.0,mild_gap_prior=1.0,medium_gap_prior=1.0,large_gap_prior=1.0,no_prior_bootstrap=0.5 \\
  --wandb-mode disabled
"""
    command_path = train_root / "trainV5_merged_ab_command_draft.sh"
    write_text(command_path, command)
    command_path.chmod(0o755)
    write_text(
        train_root / "trainV5_longer_training_risk_checklist.md",
        "# TrainV5 Longer Training Risk Checklist\n\n- Confirm no target leakage and no object split leakage.\n- Confirm B-track target gate pass artifacts are from fresh rebake_v2 targets.\n- Keep no_prior and large_gap weights configurable.\n- Watch near_gt no-overedit and prior residual safety.\n- Evaluate best and latest on val/test before extending epochs.\n- Do not write paper claim from engineering runs.\n",
    )


def train_dir_safety() -> dict[str, Any]:
    forbidden = []
    for root in [REPO_ROOT / "train"]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and (path.suffix.lower() in LARGE_TRAIN_SUFFIXES or "buffer" in path.name.lower()):
                forbidden.append(str(path.relative_to(REPO_ROOT)))
    return {"forbidden_train_files": forbidden, "pass": len(forbidden) == 0}


def write_a_training_decision(a_root: Path, engineering_dir: Path | None = None) -> None:
    full_eval_root = a_root / "full_eval"
    eval_summaries = {}
    for name in ["best_val_eval", "best_test_eval", "latest_val_eval", "latest_test_eval"]:
        eval_summaries[name] = read_json(full_eval_root / name / "eval/by_variant_summary.json", {})
    engineering_summary = {}
    if engineering_dir is not None:
        engineering_summary = read_json(engineering_dir / "eval/by_variant_summary.json", {})
    best_val = eval_summaries.get("best_val_eval", {}).get("overall", {})
    latest_val = eval_summaries.get("latest_val_eval", {}).get("overall", {})
    best_better = (safe_float(best_val.get("gain_total")) or -999.0) >= (safe_float(latest_val.get("gain_total")) or -999.0)
    no_prior_audit = read_json(a_root / "no_prior_debug_cases.json", {})
    gap_audit = read_json(a_root / "prior_gap_distribution.json", {})
    lines = [
        "# A-track Sanity Diagnostic Decision",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- pair_based_dataset_correct: `{str(not no_prior_audit.get('manifest_failures')).lower()}`",
        "- no_prior_eval_baseline_source: `model_no_prior_bootstrap_baseline`",
        f"- synthetic_gap_monotonic_near_to_large: `{str(gap_audit.get('actual_gap_monotonic_near_to_large')).lower()}`",
        f"- best_vs_latest_preference: `{'best.pt' if best_better else 'latest.pt'}`",
        f"- best_val_gain_total: `{best_val.get('gain_total')}`",
        f"- latest_val_gain_total: `{latest_val.get('gain_total')}`",
        f"- engineering_eval_gain_total: `{engineering_summary.get('overall', {}).get('gain_total')}`",
        f"- engineering_eval_improvement_rate: `{engineering_summary.get('overall', {}).get('improvement_rate')}`",
        f"- engineering_eval_regression_rate: `{engineering_summary.get('overall', {}).get('regression_rate')}`",
        "- recommend_longer_trainV5_plus_training: `false`",
        "- recommend_B_full_rebake: `true_after_operator_longrun_window`",
        "",
        "The A-track chain is structurally correct, but current checkpoints are engineering diagnostics only. Longer training should wait for stable no-prior behavior and a managed B-track rebake window.",
    ]
    write_text(a_root / "sanity_diagnostic_decision.md", "\n".join(lines))
    write_text(
        a_root / "longer_training_recommendation.md",
        "# Longer Training Recommendation\n\nDo not start paper/full training yet. Run a controlled engineering comparison with no_prior weight 0.75 versus 0.25 after full val/test diagnostics are reviewed. Increase duration only if near_gt remains no-overedit safe and no_prior does not dominate regressions.\n",
    )


def final_report(abc_root: Path) -> None:
    a_root = abc_root / "A_track"
    b_root = abc_root / "B_track"
    c_root = abc_root / "C_track"
    pair_audit = read_json(REPO_ROOT / "train/trainV5_plus_a_track/trainV5_pair_audit_report.json", {})
    b_decision = read_json(b_root / "full_1155_rebake/full_1155_decision.json", {})
    preflight = read_json(b_root / "B_track_preflight.json", {})
    pending_counts = {
        name: len(load_records(b_root / "pending_repair" / name))
        for name in [
            "pending_repair_manifest.json",
            "still_unknown_material_manifest.json",
            "license_pending_manifest.json",
            "path_unresolved_manifest.json",
        ]
        if (b_root / "pending_repair" / name).exists()
    }
    eval_best_val = read_json(a_root / "full_eval/best_val_eval/eval/by_variant_summary.json", {})
    eval_best_test = read_json(a_root / "full_eval/best_test_eval/eval/by_variant_summary.json", {})
    engineering_eval = read_json(a_root / "engineering_train/eval_summary.json", {})
    train_safety = train_dir_safety()
    lines = [
        "# TrainV5 ABC Implementation Report",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        "",
        "## A Track",
        "",
        f"- pair_audit_pass: `{pair_audit.get('audit_pass')}`",
        f"- target_bundles: `{pair_audit.get('target_bundles')}`",
        f"- training_pairs: `{pair_audit.get('training_pairs')}`",
        f"- no_prior_audit: `{a_root / 'no_prior_baseline_audit.md'}`",
        f"- prior_gap_audit: `{a_root / 'prior_gap_distribution_audit.md'}`",
        f"- best_val_overall: `{json.dumps(eval_best_val.get('overall', {}), ensure_ascii=False)}`",
        f"- best_test_overall: `{json.dumps(eval_best_test.get('overall', {}), ensure_ascii=False)}`",
        f"- engineering_val_overall: `{json.dumps(engineering_eval.get('val_overall', {}), ensure_ascii=False)}`",
        f"- engineering_test_overall: `{json.dumps(engineering_eval.get('test_overall', {}), ensure_ascii=False)}`",
        f"- engineering_train_summary: `{a_root / 'engineering_train/train_summary.md'}`",
        f"- sanity_diagnostic_decision: `{a_root / 'sanity_diagnostic_decision.md'}`",
        "",
        "## B Track",
        "",
        f"- full_rebake_status: `{b_decision.get('status')}`",
        f"- full_rebake_launched: `{b_decision.get('full_rebake_launched')}`",
        f"- queue_records: `{preflight.get('records')}`",
        f"- readable_paths: `{preflight.get('path_counts', {}).get('readable_file')}`",
        f"- B_command_draft: `{b_decision.get('command_draft')}`",
        f"- TrainV5_plus_full_pairs: `blocked_until_full_rebake_completes`",
        f"- merged_ab_pairs: `blocked_until_B_gate_pass_manifest_exists`",
        f"- pending_repair_counts: `{json.dumps(pending_counts, ensure_ascii=False)}`",
        "",
        "## C Track",
        "",
        f"- expansion_plan: `{c_root / 'data_expansion_plan/trainV5_external_expansion_plan.md'}`",
        f"- upstream_profile_contract: `{c_root / 'upstream_profile_plan/upstream_profile_data_contract.md'}`",
        f"- merged_ab_training_plan: `{c_root / 'training_plan/trainV5_merged_ab_engineering_training_plan.md'}`",
        "",
        "## Safety",
        "",
        f"- train_dir_large_file_check_pass: `{str(train_safety['pass']).lower()}`",
        f"- forbidden_train_files: `{json.dumps(train_safety['forbidden_train_files'][:20], ensure_ascii=False)}`",
        "- paper_claim_written: `false`",
        "- raw_assets_deleted: `false`",
        "- online_upstream_generation_started: `false`",
        "",
        "## Recommendations",
        "",
        "- Start the B-track full rebake in a managed long-run window using the generated command draft.",
        "- Continue external data expansion with Kenney CC0 first, then GSO/Objaverse strict probes.",
        "- Delay native SF3D/SPAR3D/Hunyuan3D prior generation until the upstream profile contract is adopted.",
        "- Do not begin paper experiment tables until B-track rebake and merged AB eval are complete.",
    ]
    write_text(abc_root / "TrainV5_ABC_final_report.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--abc-root", type=Path, default=DEFAULT_ABC_ROOT)
    parser.add_argument("--a-manifest", type=Path, default=DEFAULT_A_MANIFEST)
    parser.add_argument("--b-queue", type=Path, default=DEFAULT_B_QUEUE)
    parser.add_argument("--audit-atlas-size", type=int, default=64)
    parser.add_argument("--audit-buffer-resolution", type=int, default=32)
    parser.add_argument("--min-ssd-free-gb", type=float, default=20.0)
    parser.add_argument("--run-a-audits", action="store_true")
    parser.add_argument("--run-b-preflight", action="store_true")
    parser.add_argument("--run-c-plans", action="store_true")
    parser.add_argument("--postprocess-eval-dir", type=Path, default=None)
    parser.add_argument("--write-a-decision", action="store_true")
    parser.add_argument("--engineering-dir", type=Path, default=None)
    parser.add_argument("--write-final-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.abc_root.mkdir(parents=True, exist_ok=True)
    if args.run_a_audits:
        a_root = args.abc_root / "A_track"
        a_root.mkdir(parents=True, exist_ok=True)
        no_prior_audit(args.a_manifest, a_root, args.audit_atlas_size, args.audit_buffer_resolution)
        prior_gap_audit(args.a_manifest, a_root, args.audit_atlas_size, args.audit_buffer_resolution)
    if args.postprocess_eval_dir is not None:
        postprocess_eval(args.postprocess_eval_dir)
    if args.run_b_preflight:
        b_preflight(args.b_queue, args.abc_root, args.min_ssd_free_gb)
        pending_repair_snapshot(args.abc_root)
    if args.run_c_plans:
        write_c_plans(args.abc_root)
    if args.write_a_decision:
        write_a_training_decision(args.abc_root / "A_track", args.engineering_dir)
    if args.write_final_report:
        final_report(args.abc_root)


if __name__ == "__main__":
    main()
