#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECOND_PASS_DIR = REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass"
REPAIRABLE_REASONS = {
    "unknown_material",
    "path_unresolved",
    "missing_asset",
    "license_blocked",
    "duplicate_object_id",
}
STRICT_ALLOWED_LICENSE_BUCKETS = {
    "Creative Commons Zero v1.0 Universal",
    "MIT License",
    "Apache License 2.0",
    'BSD 3-Clause "New" or "Revised" License',
    "The Unlicense",
    "Creative Commons - Attribution",
    "Creative Commons - Attribution - Share Alike",
}
PENDING_LICENSE_BUCKETS = {
    "cc_by_4_0_pending_reconcile",
    "smithsonian_open_access_pending_reconcile",
    "omniobject3d_license_pending_reconcile",
    "khronos_sample_license_pending_reconcile",
    "cc_by_nc_4_0_pending_reconcile",
    "cc_by_nc_4_0",
}
MATERIAL_PRIORITY = {
    "mixed_thin_boundary": 120.0,
    "glass_metal": 112.0,
    "ceramic_glazed_lacquer": 105.0,
    "metal_dominant": 100.0,
    "glossy_non_metal": 35.0,
}
SOURCE_DIVERSITY_BONUS = {
    "ABO_locked_core": 0.0,
    "Local_GSO_highlight_increment": 25.0,
    "Local_Kenney_CC0_increment": 18.0,
    "Local_Quaternius_CC0_increment": 16.0,
    "Local_Smithsonian_selected_increment": 28.0,
    "Local_OmniObject3D_increment": 24.0,
    "Khronos_highlight_reference_samples": 18.0,
}
MODEL_SUFFIXES = {".glb", ".gltf", ".obj", ".fbx"}
QUEUE_GENERATION = "trainV5_material_first_repair_v1"
MATERIAL_BATCH_PRIORITY = [
    "mixed_thin_boundary",
    "glass_metal",
    "ceramic_glazed_lacquer",
    "metal_dominant",
    "glossy_non_metal",
]
MATERIAL_BATCH_RATIOS = {
    "mixed_thin_boundary": 0.28,
    "glass_metal": 0.22,
    "ceramic_glazed_lacquer": 0.22,
    "metal_dominant": 0.18,
    "glossy_non_metal": 0.10,
}

MATERIAL_KEYWORDS = {
    "glass_metal": (
        "glass", "mirror", "transparent", "translucent", "transmission", "window",
        "lamp", "lantern", "bulb", "bottle", "jar", "crystal", "vitrine", "goblet",
        "metal", "chrome", "steel", "aluminum", "aluminium", "brass", "copper", "iron",
    ),
    "metal_dominant": (
        "metal", "metallic", "chrome", "steel", "aluminum", "aluminium", "brass", "copper",
        "iron", "silver", "gold", "tool", "gear", "bolt", "screw", "hinge", "hardware",
    ),
    "ceramic_glazed_lacquer": (
        "ceramic", "porcelain", "pottery", "stoneware", "tile", "glazed", "glaze", "lacquer",
        "enamel", "vase", "bowl", "plate", "dish", "cup", "mug", "teapot", "ramekin",
    ),
    "mixed_thin_boundary": (
        "thin", "frame", "wire", "grille", "mesh", "fence", "rail", "rack", "basket",
        "cage", "lattice", "net", "rope",
    ),
    "glossy_non_metal": (
        "glossy", "plastic", "polished", "painted", "acrylic", "resin", "varnish",
        "synthetic", "leather", "helmet", "case", "toy", "shoe", "coat",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def records(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Repair TrainV5 reject/unknown candidates with material-first GLB probing and queue-v4 generation.",
    )
    parser.add_argument("--second-pass-dir", type=Path, default=DEFAULT_SECOND_PASS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def object_id(record: dict[str, Any]) -> str:
    return str(record.get("object_id") or (record.get("source_record") or {}).get("object_id") or "")


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def path_exists(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.exists()


def resolved(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return str(path.resolve())
    except OSError:
        return str(value)


def storage_tier(value: Any) -> str:
    physical = resolved(value)
    if not physical:
        return "unknown"
    if physical.startswith("/4T/"):
        return "hdd_archive"
    if physical.startswith(str((REPO_ROOT / "dataoutput").resolve())):
        return "ssd_active"
    if physical.startswith(str(REPO_ROOT.resolve())):
        return "ssd_project_or_output_symlink"
    return "external_or_unknown"


def repo_logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        return str(Path(value).resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(value)


def candidate_text(record: dict[str, Any]) -> str:
    source = record.get("source_record") or {}
    keys = (
        "object_id",
        "source_name",
        "source_dataset",
        "generator_id",
        "asset_path",
        "physical_path",
        "logical_path",
        "source_model_path",
        "raw_asset_path",
        "canonical_glb_path",
        "canonical_mesh_path",
        "source_texture_root",
        "category",
        "category_bucket",
        "tags",
        "name",
        "title",
        "notes",
    )
    return " ".join(str(record.get(key) or source.get(key) or "") for key in keys).lower()


def current_material_family(record: dict[str, Any]) -> str:
    family = str(record.get("expected_material_family") or "")
    if family in {"", "unknown", "unknown_pending_second_pass", "pending_abo_semantic_classification", "still_unknown"}:
        return "unknown_material_pending_probe"
    return family


def keyword_probe(text: str) -> tuple[str, int]:
    scores = {
        family: sum(1 for token in tokens if token in text)
        for family, tokens in MATERIAL_KEYWORDS.items()
    }
    family, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return "unknown_material_pending_probe", 0
    return family, score


def load_glb_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12:
            raise ValueError("invalid_glb_header")
        magic, version, _length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise ValueError("not_glb_v2")
        chunk_header = handle.read(8)
        if len(chunk_header) < 8:
            raise ValueError("missing_json_chunk")
        chunk_len, chunk_type = struct.unpack("<I4s", chunk_header)
        if chunk_type != b"JSON":
            raise ValueError("first_chunk_not_json")
        chunk = handle.read(chunk_len)
    return json.loads(chunk.decode("utf-8"))


def load_gltf_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gltf_like_json(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".glb":
        return load_glb_json(path)
    if suffix == ".gltf":
        return load_gltf_json(path)
    raise ValueError(f"unsupported_probe_suffix:{suffix}")


def static_glb_probe(asset_path: str) -> tuple[str, float, str]:
    path = Path(asset_path)
    if path.suffix.lower() not in {".glb", ".gltf"}:
        return "unknown_material_pending_probe", 0.0, "non_glb_static_probe_skipped"
    try:
        doc = load_gltf_like_json(path)
    except Exception as exc:  # noqa: BLE001
        return "unknown_material_pending_probe", 0.0, f"static_glb_probe_failed:{type(exc).__name__}"

    materials = doc.get("materials", []) or []
    images = doc.get("images", []) or []
    material_text_bits: list[str] = []
    scores = Counter()
    metallic_values: list[float] = []
    roughness_values: list[float] = []
    transmission_detected = False
    alpha_detected = False
    double_sided_detected = False

    for material in materials:
        name = str(material.get("name") or "")
        material_text_bits.append(name)
        alpha_mode = str(material.get("alphaMode") or "").upper()
        if alpha_mode in {"MASK", "BLEND"}:
            alpha_detected = True
        if bool(material.get("doubleSided")):
            double_sided_detected = True
        pbr = material.get("pbrMetallicRoughness", {}) or {}
        metallic = pbr.get("metallicFactor")
        roughness = pbr.get("roughnessFactor")
        if isinstance(metallic, (int, float)):
            metallic_values.append(float(metallic))
        if isinstance(roughness, (int, float)):
            roughness_values.append(float(roughness))
        extensions = material.get("extensions", {}) or {}
        if "KHR_materials_transmission" in extensions:
            transmission_detected = True
            transmission = extensions["KHR_materials_transmission"].get("transmissionFactor")
            if isinstance(transmission, (int, float)) and float(transmission) > 0.05:
                transmission_detected = True

    for image in images:
        material_text_bits.append(str(image.get("name") or ""))
        material_text_bits.append(str(image.get("uri") or ""))

    family, keyword_score = keyword_probe(" ".join(material_text_bits).lower())
    if keyword_score > 0:
        scores[family] += keyword_score
    if transmission_detected:
        scores["glass_metal"] += 4
    if alpha_detected or double_sided_detected:
        scores["mixed_thin_boundary"] += 3
    if metallic_values and (sum(metallic_values) / len(metallic_values)) >= 0.55:
        scores["metal_dominant"] += 3
    if roughness_values and metallic_values:
        rough = sum(roughness_values) / len(roughness_values)
        metal = sum(metallic_values) / len(metallic_values)
        if rough <= 0.35 and metal <= 0.2:
            scores["glossy_non_metal"] += 2
        if rough <= 0.55 and metal <= 0.15 and not transmission_detected:
            scores["ceramic_glazed_lacquer"] += 1

    if not scores:
        return "unknown_material_pending_probe", 0.0, "static_glb_json_probe_no_signal"

    family, score = max(scores.items(), key=lambda item: item[1])
    confidence = min(0.92, 0.60 + 0.05 * float(score))
    return family, round(confidence, 4), "static_glb_json_probe"


def material_repair(record: dict[str, Any]) -> tuple[str, str, float, str]:
    current = current_material_family(record)
    if current != "unknown_material_pending_probe":
        return current, "known_material_preclassified", 0.95, "existing_manifest_label"

    family, score = keyword_probe(candidate_text(record))
    min_keyword_score = 2 if family in {"mixed_thin_boundary", "glossy_non_metal"} else 1
    if score >= min_keyword_score:
        confidence = min(0.88, 0.55 + 0.07 * float(score))
        return family, "metadata_keyword_probe", round(confidence, 4), "candidate_text_keyword"

    asset_path = str(record.get("asset_path") or record.get("physical_path") or "")
    probed_family, probe_confidence, probe_source = static_glb_probe(asset_path)
    if probed_family != "unknown_material_pending_probe":
        return probed_family, "static_glb_probe", probe_confidence, probe_source

    return "unknown_material_pending_probe", "blender_probe_recommended", 0.0, probe_source


def license_repair(record: dict[str, Any]) -> dict[str, Any]:
    bucket = str(record.get("license_bucket") or "")
    if bucket in STRICT_ALLOWED_LICENSE_BUCKETS:
        status = "allowed"
        research = True
        training = True
    elif bucket in PENDING_LICENSE_BUCKETS or "pending" in bucket.lower():
        status = "pending_review"
        research = True
        training = True
    else:
        status = "hard_blocked"
        research = False
        training = False
    return {
        "license_bucket": bucket,
        "license_status": status,
        "license_allowed_for_research": research,
        "license_allowed_for_training": training,
    }


def candidate_paths(record: dict[str, Any]) -> list[str]:
    source = record.get("source_record") or {}
    paths = []
    for key in ("asset_path", "physical_path", "source_model_path", "raw_asset_path", "canonical_glb_path", "canonical_mesh_path", "source_texture_root"):
        value = record.get(key) or source.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    return paths


def path_repair(record: dict[str, Any]) -> dict[str, Any]:
    for path in candidate_paths(record):
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if candidate.exists() and candidate.is_file():
            return {
                "path_resolved_ok": True,
                "physical_path": str(candidate.resolve()),
                "logical_path": repo_logical_path(str(candidate)),
                "storage_tier": storage_tier(str(candidate)),
                "asset_path": str(candidate.resolve()),
            }
    return {
        "path_resolved_ok": False,
        "physical_path": "",
        "logical_path": "",
        "storage_tier": "unknown",
        "asset_path": "",
    }


def asset_size_mb(path: str) -> float | None:
    if not path:
        return None
    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        return candidate.stat().st_size / (1024 * 1024)
    except OSError:
        return None


def estimated_cost_level(size_mb: float | None) -> str:
    if size_mb is None:
        return "unknown"
    if size_mb < 20:
        return "low"
    if size_mb < 120:
        return "medium"
    return "high"


def expected_prior_variant_types(record: dict[str, Any]) -> list[str]:
    has_prior = bool_value(record.get("has_material_prior"))
    prior_mode = str(record.get("prior_mode") or "").lower()
    if not has_prior or prior_mode in {"none", "no_prior", ""}:
        return ["no_prior_bootstrap", "synthetic_large_gap_prior"]
    if prior_mode == "scalar_rm":
        return ["scalar_broadcast_prior", "synthetic_mild_gap_prior", "synthetic_medium_gap_prior"]
    if prior_mode in {"uv_rm", "texture_rm", "spatial_map"}:
        return ["texture_rm_prior", "synthetic_mild_gap_prior", "synthetic_medium_gap_prior"]
    return ["existing_pipeline_prior", "synthetic_medium_gap_prior"]


def priority(record: dict[str, Any]) -> tuple[float, list[str]]:
    family = str(record.get("expected_material_family") or "unknown_material_pending_probe")
    score = MATERIAL_PRIORITY.get(family, 0.0)
    reasons = [f"material:{family}:{score:.0f}"]

    source = str(record.get("source_name") or "unknown")
    source_bonus = SOURCE_DIVERSITY_BONUS.get(source, 12.0 if source != "ABO_locked_core" else 0.0)
    score += source_bonus
    reasons.append(f"source_bonus:{source_bonus:.0f}")

    if bool_value(record.get("path_resolved_ok")):
        score += 6.0
        reasons.append("path_resolved:+6")

    license_status = str(record.get("license_status") or "")
    if license_status == "allowed":
        score += 8.0
        reasons.append("license_bonus:+8")
    elif license_status == "pending_review":
        score += 4.0
        reasons.append("license_bonus:+4")

    cost = estimated_cost_level(asset_size_mb(str(record.get("asset_path") or "")))
    if cost == "medium":
        score -= 2.0
        reasons.append("cost_penalty:-2")
    elif cost == "high":
        score -= 8.0
        reasons.append("cost_penalty:-8")

    return round(score, 4), reasons


def recommended_storage_tier(record: dict[str, Any]) -> str:
    family = str(record.get("expected_material_family") or "unknown_material_pending_probe")
    score = float(record.get("screening_priority_score", 0.0) or 0.0)
    size = asset_size_mb(str(record.get("asset_path") or ""))
    if family in {"mixed_thin_boundary", "glass_metal", "ceramic_glazed_lacquer", "metal_dominant"} and score >= 100.0:
        if size is None or size <= 300:
            return "ssd_active_for_rebake_cache_then_hdd_archive"
    return "hdd_archive"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "candidate_status": dict(Counter(str(row.get("candidate_status") or "unknown") for row in rows)),
        "blocked_reason": dict(Counter(reason for row in rows for reason in row.get("blocked_reason", []))),
        "material_family": dict(Counter(str(row.get("expected_material_family") or "unknown") for row in rows)),
        "source_name": dict(Counter(str(row.get("source_name") or "unknown") for row in rows)),
        "license_status": dict(Counter(str(row.get("license_status") or "unknown") for row in rows)),
        "prior_mode": dict(Counter(str(row.get("prior_mode") or "unknown") for row in rows)),
        "has_material_prior": {
            "true": sum(bool_value(row.get("has_material_prior")) for row in rows),
            "false": sum(not bool_value(row.get("has_material_prior")) for row in rows),
        },
        "material_probe_status": dict(Counter(str(row.get("material_probe_status") or "unknown") for row in rows)),
    }


def split_reason_buckets(rejected: list[dict[str, Any]], out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "repairable_candidates": [],
        "hard_reject_candidates": [],
        "pending_repair_candidates": [],
        "missing_asset_or_path_unresolved": [],
        "unknown_material_candidates": [],
        "license_pending_candidates": [],
        "license_hard_blocked_candidates": [],
        "duplicate_candidates": [],
        "source_policy_blocked_candidates": [],
        "polyhaven_auxiliary_records": [],
    }
    for row in rejected:
        reasons = set(row.get("blocked_reason") or [])
        license_info = license_repair(row)
        if "unknown_material" in reasons:
            buckets["unknown_material_candidates"].append(row)
        if reasons & {"missing_asset", "path_unresolved"}:
            buckets["missing_asset_or_path_unresolved"].append(row)
        if "duplicate_object_id" in reasons:
            buckets["duplicate_candidates"].append(row)
        if reasons & {"is_3dfuture", "polyhaven_not_object"}:
            buckets["source_policy_blocked_candidates"].append(row)
        if "polyhaven_not_object" in reasons:
            buckets["polyhaven_auxiliary_records"].append(row)
        if license_info["license_status"] == "pending_review":
            buckets["license_pending_candidates"].append(row)
        if license_info["license_status"] == "hard_blocked":
            buckets["license_hard_blocked_candidates"].append(row)
        if reasons and reasons <= REPAIRABLE_REASONS:
            buckets["repairable_candidates"].append(row)
        elif reasons:
            buckets["hard_reject_candidates"].append(row)
        else:
            buckets["pending_repair_candidates"].append(row)
    for name, rows in buckets.items():
        write_json(out_dir / f"{name}.json", {"generated_at_utc": utc_now(), "summary": summarize(rows), "records": rows})
    reason_counts = Counter(reason for row in rejected for reason in row.get("blocked_reason", []))
    write_md(
        out_dir / "reject_reason_breakdown.md",
        [
            "# Reject Reason Breakdown",
            "",
            f"- generated_at_utc: `{utc_now()}`",
            f"- records: `{len(rejected)}`",
            f"- blocked_reason: `{json.dumps(dict(reason_counts), ensure_ascii=False)}`",
            f"- repairable_candidates: `{len(buckets['repairable_candidates'])}`",
            f"- hard_reject_candidates: `{len(buckets['hard_reject_candidates'])}`",
            f"- pending_repair_candidates: `{len(buckets['pending_repair_candidates'])}`",
        ],
    )
    return buckets


def classify_repaired_candidate(item: dict[str, Any], original_reasons: list[str]) -> tuple[str, list[str], bool]:
    blockers = {
        reason
        for reason in original_reasons
        if reason in {"is_3dfuture", "polyhaven_not_object", "duplicate_object_id"}
    }
    if not bool_value(item.get("path_resolved_ok")):
        blockers.add("path_unresolved")
    if str(item.get("license_status") or "") == "hard_blocked":
        blockers.add("license_blocked")
    if str(item.get("expected_material_family") or "") == "unknown_material_pending_probe":
        blockers.add("unknown_material")

    hard = blockers & {"is_3dfuture", "polyhaven_not_object", "duplicate_object_id", "path_unresolved", "license_blocked"}
    if hard:
        return "hard_block_or_unusable", sorted(blockers), False
    if "unknown_material" in blockers:
        return "pending_material_probe", sorted(blockers), False
    return "target_rebake_candidate", [], True


def build_repaired_candidates(rejected: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    repaired: list[dict[str, Any]] = []
    material_repaired: list[dict[str, Any]] = []
    still_unknown: list[dict[str, Any]] = []
    license_repaired: list[dict[str, Any]] = []
    license_pending: list[dict[str, Any]] = []
    license_hard: list[dict[str, Any]] = []
    path_repaired: list[dict[str, Any]] = []
    path_missing: list[dict[str, Any]] = []
    for row in rejected:
        item = dict(row)
        material, probe_status, probe_confidence, probe_source = material_repair(item)
        item["expected_material_family"] = material
        item["material_probe_status"] = probe_status
        item["material_probe_confidence"] = probe_confidence
        item["material_probe_source"] = probe_source
        item["queue_generation"] = QUEUE_GENERATION
        if material == "unknown_material_pending_probe":
            still_unknown.append(item)
        else:
            material_repaired.append(item)

        license_info = license_repair(item)
        item.update(license_info)
        if license_info["license_status"] == "allowed":
            license_repaired.append(item)
        elif license_info["license_status"] == "pending_review":
            license_pending.append(item)
        else:
            license_hard.append(item)

        path_info = path_repair(item)
        item.update(path_info)
        if path_info["path_resolved_ok"]:
            path_repaired.append(item)
        else:
            path_missing.append(item)

        item["expected_prior_variant_types"] = expected_prior_variant_types(item)
        item["screening_priority_score"], item["screening_priority_reason"] = priority(item)
        item["priority_score"] = item["screening_priority_score"]
        item["priority_reason"] = item["screening_priority_reason"]
        item["estimated_cost_level"] = estimated_cost_level(asset_size_mb(str(item.get("asset_path") or "")))
        item["recommended_storage_tier"] = recommended_storage_tier(item)
        item["candidate_status"], item["blocked_reason"], item["rebake_ready"] = classify_repaired_candidate(
            item, list(row.get("blocked_reason") or [])
        )
        repaired.append(item)

    write_json(out_dir / "material_repaired_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(material_repaired), "records": material_repaired})
    write_json(out_dir / "still_unknown_material_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(still_unknown), "records": still_unknown})
    write_json(out_dir / "license_repaired_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(license_repaired), "records": license_repaired})
    write_json(out_dir / "license_hard_blocked_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(license_hard), "records": license_hard})
    write_json(out_dir / "license_pending_after_repair.json", {"generated_at_utc": utc_now(), "summary": summarize(license_pending), "records": license_pending})
    write_json(out_dir / "path_repaired_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(path_repaired), "records": path_repaired})
    write_json(out_dir / "path_still_missing_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(path_missing), "records": path_missing})
    write_json(out_dir / "path_remap.json", {"generated_at_utc": utc_now(), "records": [{"object_id": object_id(row), "physical_path": row.get("physical_path"), "logical_path": row.get("logical_path")} for row in path_repaired]})
    write_md(out_dir / "material_repair_report.md", ["# Material Repair Report", "", f"- material_repaired: `{len(material_repaired)}`", f"- still_unknown: `{len(still_unknown)}`", f"- distribution: `{json.dumps(summarize(material_repaired)['material_family'], ensure_ascii=False)}`"])
    write_md(out_dir / "license_repair_report.md", ["# License Repair Report", "", f"- allowed: `{len(license_repaired)}`", f"- pending_review: `{len(license_pending)}`", f"- hard_blocked: `{len(license_hard)}`", "", "Pending licenses remain engineering-only candidates and are not silently promoted to strict subsets."])
    write_md(out_dir / "path_repair_report.md", ["# Path Repair Report", "", f"- path_repaired: `{len(path_repaired)}`", f"- path_still_missing: `{len(path_missing)}`"])
    return repaired, {
        "path_repaired": len(path_repaired),
        "material_repaired": len(material_repaired),
        "still_unknown": len(still_unknown),
        "license_hard_blocked": len(license_hard),
        "license_pending": len(license_pending),
    }


def dedup_candidates(rows: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    suppressed: list[dict[str, Any]] = []

    def score(row: dict[str, Any]) -> tuple[int, int, int, int, float]:
        return (
            int(bool_value(row.get("rebake_ready"))),
            int(bool_value(row.get("path_resolved_ok"))),
            int(str(row.get("license_status")) == "allowed"),
            int(str(row.get("expected_material_family")) != "unknown_material_pending_probe"),
            float(row.get("screening_priority_score", 0.0) or 0.0),
        )

    for row in rows:
        oid = object_id(row)
        if not oid:
            suppressed.append(row)
            continue
        current = selected.get(oid)
        if current is None or score(row) > score(current):
            if current is not None:
                suppressed.append(current)
            selected[oid] = row
        else:
            suppressed.append(row)
    chosen = list(selected.values())
    write_json(out_dir / "dedup_selected_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(chosen), "records": chosen})
    write_json(out_dir / "duplicate_suppressed_candidates.json", {"generated_at_utc": utc_now(), "summary": summarize(suppressed), "records": suppressed})
    write_md(out_dir / "dedup_report.md", ["# Dedup Report", "", f"- selected: `{len(chosen)}`", f"- duplicate_suppressed: `{len(suppressed)}`"])
    return chosen, suppressed


def second_pass(rows: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    usable = [dict(row) for row in rows if bool_value(row.get("rebake_ready"))]
    target = sorted(
        usable,
        key=lambda row: (float(row.get("screening_priority_score", 0.0)), str(row.get("object_id") or "")),
        reverse=True,
    )
    pending = [
        dict(row)
        for row in rows
        if str(row.get("candidate_status") or "") == "pending_material_probe"
    ]
    rejected = [
        dict(row)
        for row in rows
        if str(row.get("candidate_status") or "") == "hard_block_or_unusable"
    ]

    base = {"generated_at_utc": utc_now(), "second_pass_version": "trainV5_repaired_second_pass_material_first_v2", "queue_generation": QUEUE_GENERATION}
    write_json(out_dir / "repaired_usable_candidates.json", {**base, "summary": summarize(usable), "records": usable})
    write_json(out_dir / "repaired_target_rebake_candidates.json", {**base, "summary": summarize(target), "records": target})
    write_json(out_dir / "repaired_pending_material_probe_candidates.json", {**base, "summary": summarize(pending), "records": pending})
    write_json(out_dir / "repaired_hard_block_or_unusable_candidates.json", {**base, "summary": summarize(rejected), "records": rejected})
    write_json(out_dir / "repaired_reject_or_unknown_candidates.json", {**base, "summary": summarize(pending + rejected), "records": pending + rejected})
    queue_payload = {**base, "summary": summarize(target), "records": target}
    write_json(out_dir / "repaired_trainV5_plus_rebake_queue_preview.json", queue_payload)
    write_json(out_dir / "repaired_trainV5_plus_rebake_queue_latest.json", queue_payload)
    write_md(out_dir / "repaired_second_pass_report.md", ["# Repaired Second-Pass Report", "", f"- repaired_target_rebake_candidates: `{len(target)}`", f"- repaired_pending_material_probe_candidates: `{len(pending)}`", f"- repaired_hard_block_or_unusable_candidates: `{len(rejected)}`", f"- material_family: `{json.dumps(summarize(target)['material_family'], ensure_ascii=False)}`"])
    return target, pending, rejected


def write_quota(path: Path, target: list[dict[str, Any]]) -> None:
    def batch_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "records": len(rows),
            "material_family": dict(Counter(str(row.get("expected_material_family") or "unknown") for row in rows)),
            "source_name": dict(Counter(str(row.get("source_name") or "unknown") for row in rows)),
            "has_material_prior": {"true": sum(bool_value(row.get("has_material_prior")) for row in rows), "false": sum(not bool_value(row.get("has_material_prior")) for row in rows)},
            "expected_prior_variant_types": dict(Counter(variant for row in rows for variant in row.get("expected_prior_variant_types", []))),
            "recommended_storage_tier": dict(Counter(str(row.get("recommended_storage_tier") or "unknown") for row in rows)),
        }
    lines = [
        "# TrainV5 Plus Quota Recommendation V4",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- full_sorted_target_rebake_candidates: `{len(target)}`",
        "",
        "## Batch Summaries",
        "",
        f"- batch_0_64: `{json.dumps(batch_summary(select_material_quota_batch(target, 64)), ensure_ascii=False)}`",
        f"- batch_1_256: `{json.dumps(batch_summary(select_material_quota_batch(target, 256)), ensure_ascii=False)}`",
        f"- batch_1_512: `{json.dumps(batch_summary(select_material_quota_batch(target, 512)), ensure_ascii=False)}`",
        f"- batch_2_1000: `{json.dumps(batch_summary(select_material_quota_batch(target, 1000)), ensure_ascii=False)}`",
        "",
        "Sorting is material-first. Prior-related fields are carried as finalize-time hints only.",
    ]
    write_md(path, lines)


def select_material_quota_batch(records: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    grouped = {
        family: [row for row in records if str(row.get("expected_material_family") or "") == family]
        for family in MATERIAL_BATCH_PRIORITY
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    raw_quotas = {
        family: MATERIAL_BATCH_RATIOS[family] * size
        for family in MATERIAL_BATCH_PRIORITY
    }
    quotas = {family: int(raw_quotas[family]) for family in MATERIAL_BATCH_PRIORITY}
    remainder = max(0, size - sum(quotas.values()))
    fractional = sorted(
        MATERIAL_BATCH_PRIORITY,
        key=lambda family: raw_quotas[family] - quotas[family],
        reverse=True,
    )
    for family in fractional[:remainder]:
        quotas[family] += 1

    for family in MATERIAL_BATCH_PRIORITY:
        rows = grouped[family]
        take = min(len(rows), quotas[family])
        for row in rows[:take]:
            oid = object_id(row)
            if oid and oid not in selected_ids:
                selected.append(row)
                selected_ids.add(oid)

    if len(selected) < size:
        for row in records:
            oid = object_id(row)
            if oid and oid in selected_ids:
                continue
            selected.append(row)
            if oid:
                selected_ids.add(oid)
            if len(selected) >= size:
                break
    return selected[: min(size, len(selected))]


def write_batch_manifests(out_dir: Path, records: list[dict[str, Any]]) -> None:
    for batch_name, batch_size in [
        ("batch_0_64_material_first", 64),
        ("batch_1_256_material_first", 256),
        ("batch_1_512_material_first", 512),
        ("batch_2_1000_material_first", 1000),
    ]:
        batch_records = select_material_quota_batch(records, batch_size)
        write_json(
            out_dir / f"{batch_name}.json",
            {
                "generated_at_utc": utc_now(),
                "queue_generation": QUEUE_GENERATION,
                "batch_name": batch_name,
                "batch_size": len(batch_records),
                "selection_policy": {
                    "type": "material_quota_round_robin",
                    "priority": MATERIAL_BATCH_PRIORITY,
                    "ratios": MATERIAL_BATCH_RATIOS,
                },
                "summary": summarize(batch_records),
                "records": batch_records,
            },
        )


def write_expansion_plan(base_dir: Path) -> None:
    out_dir = base_dir / "data_expansion_plan"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Smithsonian_OpenAccess", "mixed_thin_boundary/glass/ceramic", "high", "pending_policy", "medium", "glb/obj/usdz", "medium", "non_ABO_material_diversity", "expand_after_material_probe"),
        ("Kenney_CC0", "stylized glossy/nonmetal/thin", "medium", "low", "low", "glb/obj", "high", "thin_boundary_and_cheap_objects", "keep_as_fill_source"),
        ("Quaternius_CC0", "stylized hard-surface / metal / frame", "medium", "low", "low", "glb/obj/fbx", "high", "cheap_material_gap_fill", "stage_when_assets_exist"),
        ("GSO", "household ceramic/metal/plastic", "medium", "pending_policy", "medium", "obj/glb", "medium", "ceramic_and_gloss_household", "probe_first"),
        ("OmniObject3D", "broad real object coverage", "medium", "pending_policy", "high", "glb/obj", "medium", "future_real_object_fill", "license_review_then_probe"),
        ("Objaverse-XL_strict", "broad long tail", "medium", "medium", "medium", "glb", "medium", "primary_material_probe_pool", "continue_cached_increment_downloads"),
        ("PolyHaven_aux", "materials only/non-object", "low", "low", "low", "textures/hdr", "high", "diagnostics_only", "diagnostic_only"),
    ]
    with (out_dir / "trainV5_external_source_priority_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_name", "expected_material_strength", "expected_no_prior_value", "license_risk", "integration_cost", "asset_format", "expected_path_reliability", "expected_trainV5_role", "recommended_action"])
        writer.writerows(rows)
    write_md(
        out_dir / "trainV5_external_expansion_plan.md",
        [
            "# TrainV5 External Expansion Plan",
            "",
            f"- generated_at_utc: `{utc_now()}`",
            "",
            "## Goals",
            "",
            "- First fill material gaps: `mixed_thin_boundary`, `glass_metal`, `ceramic_glazed_lacquer`, `metal_dominant`.",
            "- Keep `glossy_non_metal` only as volume backfill, not as the main growth direction.",
            "- Treat prior-related fields as finalize-time hints instead of screening-time sort drivers.",
            "- Continue cached Objaverse/Smithsonian/Kenney-style source staging without starting GPU rebake here.",
        ],
    )


def main() -> None:
    args = parse_args()
    second_pass_dir = args.second_pass_dir
    repair_dir = args.output_dir or (second_pass_dir / "repair")
    repair_dir.mkdir(parents=True, exist_ok=True)
    rejected = records(second_pass_dir / "reject_or_unknown_candidates.json")
    original_target = records(second_pass_dir / "trainV5_plus_rebake_queue_preview.json")
    split_reason_buckets(rejected, repair_dir)
    repaired_rows, repair_counts = build_repaired_candidates(rejected, repair_dir)
    dedup_selected, duplicate_suppressed = dedup_candidates(repaired_rows, repair_dir)
    repaired_second_pass_dir = second_pass_dir / "repaired_second_pass"
    repaired_target, repaired_pending, repaired_rejected = second_pass(dedup_selected, repaired_second_pass_dir)

    merged: dict[str, dict[str, Any]] = {}
    for row in original_target + repaired_target:
        oid = object_id(row)
        if not oid:
            continue
        current = merged.get(oid)
        if current is None or float(row.get("screening_priority_score", 0.0) or 0.0) > float(current.get("screening_priority_score", 0.0) or 0.0):
            merged[oid] = row
    final_target = sorted(
        merged.values(),
        key=lambda row: (float(row.get("screening_priority_score", 0.0) or 0.0), str(row.get("object_id") or "")),
        reverse=True,
    )
    final_queue_payload = {
        "generated_at_utc": utc_now(),
        "queue_policy": "original_v2_plus_material_first_repaired_dedup_priority_sort",
        "queue_generation": QUEUE_GENERATION,
        "summary": summarize(final_target),
        "records": final_target,
    }
    write_json(second_pass_dir / "trainV5_plus_rebake_queue_preview_v4.json", final_queue_payload)
    write_json(second_pass_dir / "trainV5_plus_rebake_queue_latest.json", final_queue_payload)
    write_quota(second_pass_dir / "trainV5_plus_quota_recommendation_v4.md", final_target)
    write_batch_manifests(second_pass_dir, final_target)
    write_json(
        second_pass_dir / "pending_material_probe_by_source.json",
        {
            "generated_at_utc": utc_now(),
            "records": len(repaired_pending),
            "source_name": dict(Counter(str(row.get("source_name") or "unknown") for row in repaired_pending)),
            "material_probe_status": dict(Counter(str(row.get("material_probe_status") or "unknown") for row in repaired_pending)),
        },
    )
    final_counts = {
        "original_target_rebake_candidates": len(original_target),
        "repaired_target_rebake_candidates": len(repaired_target),
        "final_target_rebake_candidates": len(final_target),
        "pending_material_probe": len(repaired_pending),
        "hard_block_or_unusable": len(repaired_rejected),
        "still_unknown": repair_counts["still_unknown"],
        "duplicate_suppressed": len(duplicate_suppressed),
        "license_hard_blocked": repair_counts["license_hard_blocked"],
        "license_pending": repair_counts["license_pending"],
        "path_repaired": repair_counts["path_repaired"],
        "material_repaired": repair_counts["material_repaired"],
    }
    write_md(
        second_pass_dir / "final_candidate_pool_report.md",
        [
            "# TrainV5 Final Candidate Pool Report",
            "",
            f"- generated_at_utc: `{utc_now()}`",
            *[f"- {key}: `{value}`" for key, value in final_counts.items()],
            f"- material_family_before: `{json.dumps(summarize(original_target)['material_family'], ensure_ascii=False)}`",
            f"- material_family_after: `{json.dumps(summarize(final_target)['material_family'], ensure_ascii=False)}`",
            f"- source_distribution_before: `{json.dumps(summarize(original_target)['source_name'], ensure_ascii=False)}`",
            f"- source_distribution_after: `{json.dumps(summarize(final_target)['source_name'], ensure_ascii=False)}`",
            f"- pending_unknown_by_source: `{json.dumps(dict(Counter(str(row.get('source_name') or 'unknown') for row in repaired_pending)), ensure_ascii=False)}`",
            "- latest_queue: `output/material_refine_trainV5/expansion_second_pass/trainV5_plus_rebake_queue_latest.json`",
            "- quota_batch_64: `output/material_refine_trainV5/expansion_second_pass/batch_0_64_material_first.json`",
            "- quota_batch_256: `output/material_refine_trainV5/expansion_second_pass/batch_1_256_material_first.json`",
            "- quota_batch_512: `output/material_refine_trainV5/expansion_second_pass/batch_1_512_material_first.json`",
            "- recommended_next_rebake_batch_size: `64`",
            "",
            "No raw assets were deleted and no full GPU rebake was launched.",
        ],
    )
    write_expansion_plan(second_pass_dir.parent)
    print(json.dumps(final_counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
