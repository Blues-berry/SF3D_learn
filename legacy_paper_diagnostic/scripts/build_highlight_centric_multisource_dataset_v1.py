#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import requests
except Exception:  # pragma: no cover - requests is expected, but keep a fallback
    requests = None


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DOCS_ROOT = PROJECT_ROOT / "docs" / "neural_gaffer_dataset_audit"
NEURAL_GAFFER_ROOT = Path("/4T/CXY/Neural_Gaffer")
EXTERNAL_ROOT = NEURAL_GAFFER_ROOT / "external_data" / "neural_gaffer_original"
LEGACY_ROOT = Path("/4T/CXY/Neural_Gaffer_original")

ABO_CORE_CSV = DOCS_ROOT / "asset_supervision_miniv1_candidate_pool_500.csv"
THREE_FUTURE_INFO_JSON = (
    EXTERNAL_ROOT
    / "external_sources"
    / "downloads"
    / "3d_future"
    / "model_extracted"
    / "3D-FUTURE-model"
    / "model_info.json"
)
THREE_FUTURE_MODEL_ROOT = (
    EXTERNAL_ROOT
    / "external_sources"
    / "downloads"
    / "3d_future"
    / "model_extracted"
    / "3D-FUTURE-model"
)
OBJAVERSE_ECOMMERCE_JSON = LEGACY_ROOT / "objaverse_subsets" / "ecommerce_1000.json"

OUTPUT_SOURCE_REGISTRY = DOCS_ROOT / "source_registry_v2.csv"
OUTPUT_SOURCE_DECISION = DOCS_ROOT / "source_decision_v2.md"
OUTPUT_POOL_A_MANIFEST = DOCS_ROOT / "pool_A_candidate_manifest_v1.csv"
OUTPUT_POOL_B_MANIFEST = DOCS_ROOT / "pool_B_manifest_v1.csv"
OUTPUT_POOL_C_MANIFEST = DOCS_ROOT / "pool_C_manifest_v1.csv"
OUTPUT_POOL_D_MANIFEST = DOCS_ROOT / "pool_D_hdri_manifest_v1.csv"
OUTPUT_POOL_E_MANIFEST = DOCS_ROOT / "pool_E_material_manifest_v1.csv"
OUTPUT_POOL_F_MANIFEST = DOCS_ROOT / "pool_F_eval_manifest_v1.csv"
OUTPUT_LICENSE_RISK = DOCS_ROOT / "license_risk_report_v1.md"
OUTPUT_PILOT_QUEUE = DOCS_ROOT / "pilot_download_queue_v1.csv"
OUTPUT_POOL_A_3DFUTURE = DOCS_ROOT / "pool_A_pilot_3dfuture_30.csv"
OUTPUT_POOL_A_3DFUTURE_JSON = DOCS_ROOT / "pool_A_pilot_3dfuture_30.json"
OUTPUT_POOL_A_OBJAVERSE = DOCS_ROOT / "pool_A_pilot_objaverse_30.csv"
OUTPUT_POOL_A_SUMMARY = DOCS_ROOT / "pool_A_pilot_summary.md"
OUTPUT_POOL_B_PILOT = DOCS_ROOT / "pool_B_pilot_manifest.csv"
OUTPUT_POOL_B_SIGNAL = DOCS_ROOT / "pool_B_signal_check.md"
OUTPUT_HDRI_BANK = DOCS_ROOT / "hdri_bank_v1.csv"


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def slugify(text: str) -> str:
    parts = []
    for ch in text.lower():
        if ch.isalnum():
            parts.append(ch)
        else:
            parts.append("_")
    slug = "".join(parts)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "item"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            raise RuntimeError(f"cannot write empty csv without header: {path}")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


SOURCE_REGISTRY_FIELDS = [
    "source_name",
    "pool_name",
    "official_url",
    "license_bucket",
    "raw_license_text_or_page",
    "download_status",
    "format_main",
    "blender_compatible",
    "has_mesh_likely",
    "has_uv_likely",
    "has_pbr_likely",
    "has_controlled_lighting",
    "has_real_hdr_lighting",
    "has_material_prior",
    "fit_for_pool_A",
    "fit_for_pool_B",
    "fit_for_pool_C",
    "fit_for_pool_D",
    "fit_for_pool_E",
    "fit_for_pool_F",
    "risk_notes",
]


SOURCE_REGISTRY_ROWS: List[Dict[str, str]] = [
    {
        "source_name": "ABO_locked_core",
        "pool_name": "pool_A_direct_object_supervision",
        "official_url": "https://registry.opendata.aws/amazon-berkeley-objects/",
        "license_bucket": "cc_by_nc_4_0",
        "raw_license_text_or_page": "https://registry.opendata.aws/amazon-berkeley-objects/ (official registry lists CC BY-NC 4.0)",
        "download_status": "local_core_available",
        "format_main": "glb",
        "blender_compatible": "true",
        "has_mesh_likely": "true",
        "has_uv_likely": "true",
        "has_pbr_likely": "true",
        "has_controlled_lighting": "false",
        "has_real_hdr_lighting": "false",
        "has_material_prior": "false",
        "fit_for_pool_A": "true",
        "fit_for_pool_B": "false",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "false",
        "fit_for_pool_E": "false",
        "fit_for_pool_F": "false",
        "risk_notes": "non-commercial license; keep separate from freely redistributable release bundles",
    },
    {
        "source_name": "3D-FUTURE_candidate",
        "pool_name": "pool_A_direct_object_supervision",
        "official_url": "https://tianchi.aliyun.com/dataset/98063",
        "license_bucket": "custom_tianchi_terms",
        "raw_license_text_or_page": "https://tianchi.aliyun.com/specials/promotion/license",
        "download_status": "local_source_available",
        "format_main": "obj+mtl+texture",
        "blender_compatible": "true",
        "has_mesh_likely": "true",
        "has_uv_likely": "true",
        "has_pbr_likely": "false",
        "has_controlled_lighting": "false",
        "has_real_hdr_lighting": "false",
        "has_material_prior": "false",
        "fit_for_pool_A": "true",
        "fit_for_pool_B": "false",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "false",
        "fit_for_pool_E": "false",
        "fit_for_pool_F": "false",
        "risk_notes": "portal terms apply; object assets are present locally but mostly need deterministic conversion and likely material bake",
    },
    {
        "source_name": "Objaverse-XL_filtered_candidate",
        "pool_name": "pool_A_direct_object_supervision",
        "official_url": "https://objaverse.allenai.org/docs/objaverse-xl/",
        "license_bucket": "mixed_per_object_license",
        "raw_license_text_or_page": "https://github.com/allenai/objaverse-xl (annotations include license and source fields)",
        "download_status": "metadata_only",
        "format_main": "glb",
        "blender_compatible": "true",
        "has_mesh_likely": "true",
        "has_uv_likely": "true",
        "has_pbr_likely": "unknown",
        "has_controlled_lighting": "false",
        "has_real_hdr_lighting": "false",
        "has_material_prior": "false",
        "fit_for_pool_A": "true",
        "fit_for_pool_B": "false",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "false",
        "fit_for_pool_E": "false",
        "fit_for_pool_F": "false",
        "risk_notes": "do not bulk download before pilot; local asset chain missing, license is per-object and must remain split by bucket",
    },
    {
        "source_name": "OLATverse",
        "pool_name": "pool_B_controlled_highlight_supervision",
        "official_url": "https://vcai.mpi-inf.mpg.de/projects/OLATverse/",
        "license_bucket": "license_not_posted_project_page",
        "raw_license_text_or_page": "https://vcai.mpi-inf.mpg.de/projects/OLATverse/ (project page; full data release pending)",
        "download_status": "project_page_only",
        "format_main": "image sequences",
        "blender_compatible": "false",
        "has_mesh_likely": "false",
        "has_uv_likely": "false",
        "has_pbr_likely": "false",
        "has_controlled_lighting": "true",
        "has_real_hdr_lighting": "true",
        "has_material_prior": "false",
        "fit_for_pool_A": "false",
        "fit_for_pool_B": "true",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "false",
        "fit_for_pool_E": "false",
        "fit_for_pool_F": "false",
        "risk_notes": "excellent signal set on project page, but release terms and bulk access path still need confirmation",
    },
    {
        "source_name": "OpenIllumination",
        "pool_name": "pool_B_controlled_highlight_supervision",
        "official_url": "https://huggingface.co/datasets/OpenIllumination/OpenIllumination",
        "license_bucket": "cc_by_4_0",
        "raw_license_text_or_page": "https://huggingface.co/datasets/OpenIllumination/OpenIllumination",
        "download_status": "remote_pilot_only",
        "format_main": "image sequences",
        "blender_compatible": "false",
        "has_mesh_likely": "false",
        "has_uv_likely": "false",
        "has_pbr_likely": "false",
        "has_controlled_lighting": "true",
        "has_real_hdr_lighting": "false",
        "has_material_prior": "false",
        "fit_for_pool_A": "false",
        "fit_for_pool_B": "true",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "false",
        "fit_for_pool_E": "false",
        "fit_for_pool_F": "false",
        "risk_notes": "license is clear and favorable; masks are explicit, but normals/albedo/specular separation is not clearly advertised on the dataset card",
    },
    {
        "source_name": "ICTPolarReal",
        "pool_name": "pool_B_controlled_highlight_supervision",
        "official_url": "https://arxiv.org/abs/2501.16398",
        "license_bucket": "license_not_posted_preprint_project",
        "raw_license_text_or_page": "https://jingyangcarl.github.io/ICTPolarReal/ (project page referenced by preprint)",
        "download_status": "paper_project_only",
        "format_main": "polarized capture sequences",
        "blender_compatible": "false",
        "has_mesh_likely": "false",
        "has_uv_likely": "false",
        "has_pbr_likely": "false",
        "has_controlled_lighting": "true",
        "has_real_hdr_lighting": "false",
        "has_material_prior": "false",
        "fit_for_pool_A": "false",
        "fit_for_pool_B": "true",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "false",
        "fit_for_pool_E": "false",
        "fit_for_pool_F": "false",
        "risk_notes": "promising real polarized supervision, but access path and license still need project-side confirmation",
    },
    {
        "source_name": "3DCoMPaT++",
        "pool_name": "pool_C_part_material_highlight_semantics",
        "official_url": "https://3dcompat-dataset.org/v2/",
        "license_bucket": "custom_form_gated_license",
        "raw_license_text_or_page": "https://3dcompat-dataset.org/dl-dataset/ (download gated by license form)",
        "download_status": "not_started",
        "format_main": "mesh+part annotations",
        "blender_compatible": "true",
        "has_mesh_likely": "true",
        "has_uv_likely": "unknown",
        "has_pbr_likely": "false",
        "has_controlled_lighting": "false",
        "has_real_hdr_lighting": "false",
        "has_material_prior": "false",
        "fit_for_pool_A": "false",
        "fit_for_pool_B": "false",
        "fit_for_pool_C": "true",
        "fit_for_pool_D": "false",
        "fit_for_pool_E": "false",
        "fit_for_pool_F": "false",
        "risk_notes": "strong part/material semantics fit, but download is gated and not yet piloted locally",
    },
    {
        "source_name": "PolyHaven_HDRI",
        "pool_name": "pool_D_natural_light_hdri_bank",
        "official_url": "https://polyhaven.com/hdris",
        "license_bucket": "cc0",
        "raw_license_text_or_page": "https://polyhaven.com/license",
        "download_status": "api_curation_ready",
        "format_main": "hdr/exr",
        "blender_compatible": "true",
        "has_mesh_likely": "false",
        "has_uv_likely": "false",
        "has_pbr_likely": "false",
        "has_controlled_lighting": "false",
        "has_real_hdr_lighting": "true",
        "has_material_prior": "false",
        "fit_for_pool_A": "false",
        "fit_for_pool_B": "false",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "true",
        "fit_for_pool_E": "false",
        "fit_for_pool_F": "false",
        "risk_notes": "best low-risk HDRI source; direct API lets us curate pilot without downloading the full library",
    },
    {
        "source_name": "Laval_HDR",
        "pool_name": "pool_D_natural_light_hdri_bank",
        "official_url": "https://lvsn.github.io/fastindoorlight/",
        "license_bucket": "custom_research_project_terms",
        "raw_license_text_or_page": "https://indoor.hdrdb.com/ and project page above",
        "download_status": "pilot_slots_only",
        "format_main": "hdr panoramas",
        "blender_compatible": "true",
        "has_mesh_likely": "false",
        "has_uv_likely": "false",
        "has_pbr_likely": "false",
        "has_controlled_lighting": "false",
        "has_real_hdr_lighting": "true",
        "has_material_prior": "false",
        "fit_for_pool_A": "false",
        "fit_for_pool_B": "false",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "true",
        "fit_for_pool_E": "false",
        "fit_for_pool_F": "false",
        "risk_notes": "good indoor HDR complement, but naming/indexing and redistribution terms still need explicit confirmation",
    },
    {
        "source_name": "OpenSVBRDF",
        "pool_name": "pool_E_material_prior_bank",
        "official_url": "https://opensvbrdf.github.io/",
        "license_bucket": "custom_portal_terms_unstated",
        "raw_license_text_or_page": "https://opensvbrdf.github.io/",
        "download_status": "not_started",
        "format_main": "svbrdf maps",
        "blender_compatible": "false",
        "has_mesh_likely": "false",
        "has_uv_likely": "false",
        "has_pbr_likely": "true",
        "has_controlled_lighting": "false",
        "has_real_hdr_lighting": "false",
        "has_material_prior": "true",
        "fit_for_pool_A": "false",
        "fit_for_pool_B": "false",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "false",
        "fit_for_pool_E": "true",
        "fit_for_pool_F": "false",
        "risk_notes": "useful material prior bank, but license/download flow still need manual confirmation",
    },
    {
        "source_name": "MatSynth",
        "pool_name": "pool_E_material_prior_bank",
        "official_url": "https://huggingface.co/datasets/gvecchio/MatSynth",
        "license_bucket": "mixed_cc0_cc_by",
        "raw_license_text_or_page": "https://huggingface.co/datasets/gvecchio/MatSynth (dataset card lists CC0 and CC-BY assets)",
        "download_status": "remote_pilot_only",
        "format_main": "rendered materials + metadata",
        "blender_compatible": "false",
        "has_mesh_likely": "false",
        "has_uv_likely": "false",
        "has_pbr_likely": "true",
        "has_controlled_lighting": "false",
        "has_real_hdr_lighting": "false",
        "has_material_prior": "true",
        "fit_for_pool_A": "false",
        "fit_for_pool_B": "false",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "false",
        "fit_for_pool_E": "true",
        "fit_for_pool_F": "false",
        "risk_notes": "mixed permissive licensing is manageable if assets stay bucketed by license tag",
    },
    {
        "source_name": "Stanford-ORB",
        "pool_name": "pool_F_real_world_eval_holdout",
        "official_url": "https://stanfordorb.github.io/",
        "license_bucket": "benchmark_project_terms_unstated",
        "raw_license_text_or_page": "https://github.com/StanfordORB/Stanford-ORB",
        "download_status": "not_started",
        "format_main": "real-world benchmark captures",
        "blender_compatible": "false",
        "has_mesh_likely": "false",
        "has_uv_likely": "false",
        "has_pbr_likely": "false",
        "has_controlled_lighting": "false",
        "has_real_hdr_lighting": "false",
        "has_material_prior": "false",
        "fit_for_pool_A": "false",
        "fit_for_pool_B": "false",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "false",
        "fit_for_pool_E": "false",
        "fit_for_pool_F": "true",
        "risk_notes": "keep strictly as eval holdout until benchmark terms are confirmed",
    },
    {
        "source_name": "OmniObject3D",
        "pool_name": "pool_F_real_world_eval_holdout",
        "official_url": "https://omniobject3d.github.io/",
        "license_bucket": "portal_terms_plus_dataset_specific",
        "raw_license_text_or_page": "https://openxlab.org.cn/datasets/detail/omniobject3d/OmniObject3D-New",
        "download_status": "portal_login_likely_required",
        "format_main": "multi-view captures + models",
        "blender_compatible": "true",
        "has_mesh_likely": "true",
        "has_uv_likely": "unknown",
        "has_pbr_likely": "unknown",
        "has_controlled_lighting": "false",
        "has_real_hdr_lighting": "false",
        "has_material_prior": "false",
        "fit_for_pool_A": "false",
        "fit_for_pool_B": "false",
        "fit_for_pool_C": "false",
        "fit_for_pool_D": "false",
        "fit_for_pool_E": "false",
        "fit_for_pool_F": "true",
        "risk_notes": "good real-world eval complement, but portal download terms still need explicit review",
    },
]


def classify_3dfuture_material(row: Dict[str, object]) -> tuple[str, int, str]:
    material = (row.get("material") or "").strip()
    theme = (row.get("theme") or "").strip()
    category = (row.get("category") or "").strip()
    super_category = (row.get("super-category") or "").strip()

    reason_bits = []
    score = 0

    if material == "Metal" or theme in {"Gold Foil", "Wrought Iron"}:
        reason_bits.append("metal_surface")
        score += 5
        return "metal-dominant", score, ",".join(reason_bits)

    if material == "Glass":
        reason_bits.append("glass_surface")
        score += 5
        return "glass+metal composite", score, ",".join(reason_bits)

    if material in {"Smooth Leather", "Leather"}:
        reason_bits.append("specular_leather")
        score += 4
        if super_category in {"Chair", "Sofa", "Bed"}:
            score += 1
        return "glossy non-metal", score, ",".join(reason_bits)

    if super_category == "Lighting":
        reason_bits.append("lighting_fixture")
        score += 4
        return "mixed-material thin-boundary", score, ",".join(reason_bits)

    if material in {"Marble", "Stone"}:
        reason_bits.append("hard_polished_surface")
        score += 3
        return "ceramic/glazed", score, ",".join(reason_bits)

    if theme == "Smooth Net":
        reason_bits.append("thin_boundary_theme")
        score += 3
        return "mixed-material thin-boundary", score, ",".join(reason_bits)

    if category in {
        "Coffee Table",
        "Corner/Side Table",
        "Dining Table",
        "Pendant Lamp",
        "Floor Lamp",
        "Ceiling Lamp",
    }:
        reason_bits.append("hard_highlight_geometry")
        score += 2
        return "mixed-material thin-boundary", score, ",".join(reason_bits)

    return "glossy non-metal", 1, "fallback_generic_highlight_candidate"


def take_round_robin_buckets(grouped: Dict[str, List[dict]], targets: Dict[str, int], total: int) -> List[dict]:
    selected: List[dict] = []
    used = set()
    for bucket, want in targets.items():
        for row in grouped.get(bucket, [])[:want]:
            if row["model_id"] in used:
                continue
            used.add(row["model_id"])
            selected.append(row)
    if len(selected) >= total:
        return selected[:total]
    leftovers = []
    for rows in grouped.values():
        leftovers.extend(rows)
    leftovers.sort(key=lambda row: (-row["_score"], row["model_id"]))
    for row in leftovers:
        if row["model_id"] in used:
            continue
        used.add(row["model_id"])
        selected.append(row)
        if len(selected) >= total:
            break
    return selected[:total]


def build_3dfuture_candidates() -> List[Dict[str, str]]:
    rows = json.loads(THREE_FUTURE_INFO_JSON.read_text())
    candidates = []
    for row in rows:
        model_id = row["model_id"]
        obj_path = THREE_FUTURE_MODEL_ROOT / model_id / "raw_model.obj"
        texture_path = THREE_FUTURE_MODEL_ROOT / model_id / "texture.png"
        if not obj_path.exists():
            continue
        material_class, score, reason = classify_3dfuture_material(row)
        item = dict(row)
        item["_score"] = score
        item["_material_class"] = material_class
        item["_reason"] = reason
        item["_obj_path"] = str(obj_path)
        item["_texture_path"] = str(texture_path) if texture_path.exists() else ""
        candidates.append(item)

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in candidates:
        grouped[row["_material_class"]].append(row)
    for bucket_rows in grouped.values():
        bucket_rows.sort(
            key=lambda row: (
                -row["_score"],
                row.get("super-category") or "",
                row.get("category") or "",
                row["model_id"],
            )
        )

    targets = {
        "metal-dominant": 9,
        "ceramic/glazed": 6,
        "glass+metal composite": 4,
        "mixed-material thin-boundary": 6,
        "glossy non-metal": 5,
    }
    selected = take_round_robin_buckets(grouped, targets=targets, total=30)

    output_rows = []
    for index, row in enumerate(selected, start=1):
        model_id = row["model_id"]
        object_id = f"3dfuture_{model_id}"
        label = slugify(f"{row['_material_class']}_{row['category']}_{index}")
        name = f"{row['super-category']} | {row['category']} | {row.get('material') or 'unknown_material'}"
        output_rows.append(
            {
                "id": object_id,
                "object_id": object_id,
                "source_name": "3D-FUTURE_candidate",
                "pool_name": "pool_A_direct_object_supervision",
                "source_uid": model_id,
                "name": name,
                "label": label,
                "local_path": row["_obj_path"],
                "source_model_path": row["_obj_path"],
                "texture_root": row["_texture_path"],
                "format": "obj",
                "license_bucket": "custom_tianchi_terms",
                "download_status": "local_source_available",
                "local_asset_status": "available",
                "highlight_priority_class": row["_material_class"],
                "highlight_reason": row["_reason"],
                "super_category": row["super-category"],
                "category": row["category"],
                "style": row.get("style") or "",
                "theme": row.get("theme") or "",
                "material": row.get("material") or "",
                "blender_ready_guess": "true",
                "needs_conversion": "true",
                "needs_material_bake": "unknown",
                "audit_status": "pilot_pending_blender_audit",
                "reject_reason": "",
                "notes": "local_obj_source_present; deterministic_glb_conversion_required",
            }
        )
    return output_rows


def build_objaverse_candidates() -> List[Dict[str, str]]:
    data = json.loads(OBJAVERSE_ECOMMERCE_JSON.read_text())
    uids = sorted(data.keys())
    if len(uids) < 30:
        raise RuntimeError("objaverse candidate pool is unexpectedly small")
    step = len(uids) / 30.0
    selected = []
    for i in range(30):
        uid = uids[min(math.floor(i * step), len(uids) - 1)]
        selected.append((uid, data[uid]))

    rows = []
    for uid, rel_path in selected:
        object_id = f"objaverse_{uid}"
        rows.append(
            {
                "id": object_id,
                "object_id": object_id,
                "source_name": "Objaverse-XL_filtered_candidate",
                "pool_name": "pool_A_direct_object_supervision",
                "source_uid": uid,
                "name": uid,
                "label": "objaverse_xl_filtered_candidate",
                "local_path": "",
                "source_model_path": rel_path,
                "texture_root": "",
                "format": "glb",
                "license_bucket": "mixed_per_object_license",
                "download_status": "metadata_only",
                "local_asset_status": "metadata_only",
                "highlight_priority_class": "unknown_pending_remote_metadata",
                "highlight_reason": "selected_from_filtered_ecommerce_subset_without_bulk_download",
                "super_category": "",
                "category": "",
                "style": "",
                "theme": "",
                "material": "",
                "blender_ready_guess": "unknown",
                "needs_conversion": "false",
                "needs_material_bake": "unknown",
                "audit_status": "pilot_blocked_no_local_asset",
                "reject_reason": "source_asset_not_local",
                "notes": "pilot limited to metadata lookup until asset access is approved",
            }
        )
    return rows


def build_pool_a_manifest(three_future_rows: Sequence[Dict[str, str]], objaverse_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    core_rows = read_csv(ABO_CORE_CSV)
    manifest_rows = []
    for row in core_rows:
        manifest_rows.append(
            {
                "pool_name": "pool_A_direct_object_supervision",
                "source_name": "ABO_locked_core",
                "object_id": row["object_id"],
                "source_uid": row["source_uid"],
                "source_model_path": row["source_model_path"],
                "format": row["format"],
                "license_bucket": "cc_by_nc_4_0",
                "local_asset_status": "available",
                "blender_compatible": "true",
                "has_mesh": row["has_mesh"],
                "has_uv": row["has_uv"],
                "has_albedo": row["has_albedo"],
                "has_roughness": row["has_roughness"],
                "has_metallic": row["has_metallic"],
                "needs_conversion": row["needs_conversion"],
                "needs_material_bake": row["needs_material_bake"],
                "structure_status": row["audit_status"],
                "semantic_status": row["semantic_validation_status"],
                "pool_a_status": "core_A_ready_locked",
                "highlight_material_class": "pending_classification_pass",
                "material_class_source": "core_locked_semantic_pool",
                "review_status": row["review_status"],
                "priority_bucket": "core_locked",
                "notes": row["notes"],
            }
        )

    for row in three_future_rows:
        manifest_rows.append(
            {
                "pool_name": row["pool_name"],
                "source_name": row["source_name"],
                "object_id": row["object_id"],
                "source_uid": row["source_uid"],
                "source_model_path": row["source_model_path"],
                "format": row["format"],
                "license_bucket": row["license_bucket"],
                "local_asset_status": row["local_asset_status"],
                "blender_compatible": row["blender_ready_guess"],
                "has_mesh": "unknown",
                "has_uv": "unknown",
                "has_albedo": "unknown",
                "has_roughness": "unknown",
                "has_metallic": "unknown",
                "needs_conversion": row["needs_conversion"],
                "needs_material_bake": row["needs_material_bake"],
                "structure_status": row["audit_status"],
                "semantic_status": "not_started",
                "pool_a_status": "pilot_pending_audit",
                "highlight_material_class": row["highlight_priority_class"],
                "material_class_source": "3dfuture_metadata_heuristic",
                "review_status": "none",
                "priority_bucket": "pilot_extension",
                "notes": row["notes"],
            }
        )

    for row in objaverse_rows:
        manifest_rows.append(
            {
                "pool_name": row["pool_name"],
                "source_name": row["source_name"],
                "object_id": row["object_id"],
                "source_uid": row["source_uid"],
                "source_model_path": row["source_model_path"],
                "format": row["format"],
                "license_bucket": row["license_bucket"],
                "local_asset_status": row["local_asset_status"],
                "blender_compatible": row["blender_ready_guess"],
                "has_mesh": "unknown",
                "has_uv": "unknown",
                "has_albedo": "unknown",
                "has_roughness": "unknown",
                "has_metallic": "unknown",
                "needs_conversion": row["needs_conversion"],
                "needs_material_bake": row["needs_material_bake"],
                "structure_status": row["audit_status"],
                "semantic_status": "blocked_no_local_asset",
                "pool_a_status": "pilot_pending_asset_access",
                "highlight_material_class": row["highlight_priority_class"],
                "material_class_source": "subset_prior_only",
                "review_status": "none",
                "priority_bucket": "pilot_extension",
                "notes": row["notes"],
            }
        )

    return manifest_rows


def build_pool_b_rows() -> List[Dict[str, str]]:
    plan = [
        ("OLATverse", 20, "project_page_only", "preview_signals_look_strong"),
        ("OpenIllumination", 10, "remote_pilot_only", "license_clear_cc_by_4_0"),
        ("ICTPolarReal", 10, "paper_project_only", "download_if_project_access_is_confirmed"),
    ]
    rows = []
    for source_name, count, status, note in plan:
        for index in range(1, count + 1):
            rows.append(
                {
                    "pool_name": "pool_B_controlled_highlight_supervision",
                    "source_name": source_name,
                    "candidate_id": f"{slugify(source_name)}_{index:02d}",
                    "sequence_id": f"{slugify(source_name)}_seq_{index:02d}",
                    "license_bucket": next(row["license_bucket"] for row in SOURCE_REGISTRY_ROWS if row["source_name"] == source_name),
                    "download_status": status,
                    "has_controlled_lighting": "true",
                    "has_mask_signal": "true" if source_name in {"OLATverse", "OpenIllumination"} else "unknown",
                    "has_normals_signal": "true" if source_name == "OLATverse" else ("true" if source_name == "ICTPolarReal" else "unknown"),
                    "has_albedo_signal": "true" if source_name in {"OLATverse", "ICTPolarReal"} else "unknown",
                    "has_specular_signal": "true" if source_name == "ICTPolarReal" else "unknown",
                    "fit_for_highlight_supervision": "true",
                    "pilot_phase": "pilot_v1",
                    "notes": note,
                }
            )
    return rows


def fetch_polyhaven_hdri_rows() -> List[Dict[str, str]]:
    fallback = []
    slot_plan = {
        "hard sunlight": 4,
        "overcast diffuse": 4,
        "window-side indoor": 4,
        "warm sunset": 3,
        "high-contrast indoor": 3,
        "urban night / neon": 2,
    }
    if requests is None:
        for stratum, count in slot_plan.items():
            for index in range(1, count + 1):
                fallback.append(
                    {
                        "pool_name": "pool_D_natural_light_hdri_bank",
                        "source_name": "PolyHaven_HDRI",
                        "candidate_id": f"polyhaven_{slugify(stratum)}_{index:02d}",
                        "asset_id": "",
                        "stratum": stratum,
                        "selection_query": stratum,
                        "license_bucket": "cc0",
                        "download_status": "api_query_failed_fallback_slot",
                        "local_status": "not_downloaded",
                        "notes": "fallback slot; resolve with live Poly Haven API before download",
                    }
                )
        return fallback

    try:
        response = requests.get("https://api.polyhaven.com/assets?t=hdris", timeout=60)
        response.raise_for_status()
        assets = response.json()
    except Exception:
        return fetch_polyhaven_hdri_rows.__wrapped__()  # type: ignore[attr-defined]

    strata_rules = {
        "hard sunlight": ["sun", "sunny", "clear"],
        "overcast diffuse": ["overcast", "cloudy", "cloud"],
        "window-side indoor": ["window", "room", "indoor", "interior"],
        "warm sunset": ["sunset", "dusk", "evening"],
        "high-contrast indoor": ["indoor", "interior", "hall", "studio", "warehouse"],
        "urban night / neon": ["night", "urban", "city", "neon"],
    }

    selected_rows = []
    used_ids = set()
    for stratum, count in slot_plan.items():
        tags = strata_rules[stratum]
        matched = []
        for asset_id, payload in assets.items():
            searchable = " ".join(
                [
                    asset_id,
                    payload.get("name", ""),
                    " ".join(payload.get("tags", []) if isinstance(payload.get("tags"), list) else []),
                ]
            ).lower()
            score = sum(tag in searchable for tag in tags)
            if score > 0:
                matched.append((score, asset_id, payload))
        matched.sort(key=lambda item: (-item[0], item[1]))
        picked = 0
        for _score, asset_id, payload in matched:
            if asset_id in used_ids:
                continue
            used_ids.add(asset_id)
            selected_rows.append(
                {
                    "pool_name": "pool_D_natural_light_hdri_bank",
                    "source_name": "PolyHaven_HDRI",
                    "candidate_id": f"polyhaven_{asset_id}",
                    "asset_id": asset_id,
                    "stratum": stratum,
                    "selection_query": ", ".join(tags),
                    "license_bucket": "cc0",
                    "download_status": "api_curation_ready",
                    "local_status": "not_downloaded",
                    "notes": payload.get("name", asset_id),
                }
            )
            picked += 1
            if picked >= count:
                break
    return selected_rows


def _polyhaven_fallback_wrapper():
    return fetch_polyhaven_hdri_rows()


fetch_polyhaven_hdri_rows.__wrapped__ = lambda: [  # type: ignore[attr-defined]
    {
        "pool_name": "pool_D_natural_light_hdri_bank",
        "source_name": "PolyHaven_HDRI",
        "candidate_id": f"polyhaven_slot_{index:02d}",
        "asset_id": "",
        "stratum": stratum,
        "selection_query": stratum,
        "license_bucket": "cc0",
        "download_status": "api_query_failed_fallback_slot",
        "local_status": "not_downloaded",
        "notes": "fallback slot; resolve with live Poly Haven API before download",
    }
    for stratum, count in {
        "hard sunlight": 4,
        "overcast diffuse": 4,
        "window-side indoor": 4,
        "warm sunset": 3,
        "high-contrast indoor": 3,
        "urban night / neon": 2,
    }.items()
    for index in range(1, count + 1)
]


def build_laval_hdri_rows() -> List[Dict[str, str]]:
    slot_plan = {
        "hard sunlight": 2,
        "overcast diffuse": 2,
        "window-side indoor": 6,
        "warm sunset": 2,
        "high-contrast indoor": 5,
        "urban night / neon": 3,
    }
    rows = []
    for stratum, count in slot_plan.items():
        for index in range(1, count + 1):
            rows.append(
                {
                    "pool_name": "pool_D_natural_light_hdri_bank",
                    "source_name": "Laval_HDR",
                    "candidate_id": f"laval_{slugify(stratum)}_{index:02d}",
                    "asset_id": "",
                    "stratum": stratum,
                    "selection_query": stratum,
                    "license_bucket": "custom_research_project_terms",
                    "download_status": "pilot_slot_pending_manual_selection",
                    "local_status": "not_downloaded",
                    "notes": "manual scene selection required from indoor HDR db before download",
                }
            )
    return rows


def build_pool_c_rows() -> List[Dict[str, str]]:
    return [
        {
            "pool_name": "pool_C_part_material_highlight_semantics",
            "source_name": "3DCoMPaT++",
            "planned_subset": "pilot_not_started",
            "license_bucket": "custom_form_gated_license",
            "download_status": "not_started",
            "annotation_focus": "part-level material semantics",
            "notes": "use for part-material highlight semantics after license form is accepted",
        }
    ]


def build_pool_e_rows() -> List[Dict[str, str]]:
    return [
        {
            "pool_name": "pool_E_material_prior_bank",
            "source_name": "OpenSVBRDF",
            "license_bucket": "custom_portal_terms_unstated",
            "download_status": "not_started",
            "signal_type": "svbrdf_maps",
            "notes": "strong prior source once license and portal flow are confirmed",
        },
        {
            "pool_name": "pool_E_material_prior_bank",
            "source_name": "MatSynth",
            "license_bucket": "mixed_cc0_cc_by",
            "download_status": "remote_pilot_only",
            "signal_type": "material_prior_render_bank",
            "notes": "keep release bundles split by per-material license tags",
        },
    ]


def build_pool_f_rows() -> List[Dict[str, str]]:
    return [
        {
            "pool_name": "pool_F_real_world_eval_holdout",
            "source_name": "Stanford-ORB",
            "license_bucket": "benchmark_project_terms_unstated",
            "download_status": "not_started",
            "planned_use": "real_world_eval_holdout",
            "notes": "benchmark-only until terms are reviewed",
        },
        {
            "pool_name": "pool_F_real_world_eval_holdout",
            "source_name": "OmniObject3D",
            "license_bucket": "portal_terms_plus_dataset_specific",
            "download_status": "portal_login_likely_required",
            "planned_use": "real_world_eval_holdout",
            "notes": "real capture complement; keep separate from train release packages",
        },
    ]


def build_pilot_queue(
    three_future_rows: Sequence[Dict[str, str]],
    objaverse_rows: Sequence[Dict[str, str]],
    pool_b_rows: Sequence[Dict[str, str]],
    hdri_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    queue_rows = []
    for row in three_future_rows:
        queue_rows.append(
            {
                "queue_id": f"queue_{row['object_id']}",
                "pool_name": row["pool_name"],
                "source_name": row["source_name"],
                "candidate_id": row["object_id"],
                "task_type": "local_blender_audit+gt_probe",
                "priority": "P0",
                "gpu_candidate": "gpu1",
                "local_available": "true",
                "status": "ready_to_run",
                "notes": "run first-pass+second-pass locally before any expansion",
            }
        )
    for row in objaverse_rows:
        queue_rows.append(
            {
                "queue_id": f"queue_{row['object_id']}",
                "pool_name": row["pool_name"],
                "source_name": row["source_name"],
                "candidate_id": row["object_id"],
                "task_type": "metadata_lookup_only",
                "priority": "P1",
                "gpu_candidate": "none",
                "local_available": "false",
                "status": "blocked_no_local_asset",
                "notes": "do not bulk download; enrich annotations first",
            }
        )
    for row in pool_b_rows:
        queue_rows.append(
            {
                "queue_id": f"queue_{row['candidate_id']}",
                "pool_name": row["pool_name"],
                "source_name": row["source_name"],
                "candidate_id": row["candidate_id"],
                "task_type": "signal_check_only",
                "priority": "P1" if row["source_name"] != "ICTPolarReal" else "P2",
                "gpu_candidate": "none",
                "local_available": "false",
                "status": row["download_status"],
                "notes": row["notes"],
            }
        )
    for row in hdri_rows:
        queue_rows.append(
            {
                "queue_id": f"queue_{row['candidate_id']}",
                "pool_name": row["pool_name"],
                "source_name": row["source_name"],
                "candidate_id": row["candidate_id"],
                "task_type": "hdri_bank_curation",
                "priority": "P1" if row["source_name"] == "PolyHaven_HDRI" else "P2",
                "gpu_candidate": "none",
                "local_available": "false",
                "status": row["download_status"],
                "notes": row["notes"],
            }
        )
    return queue_rows


def build_source_decision_md(
    three_future_rows: Sequence[Dict[str, str]],
    objaverse_rows: Sequence[Dict[str, str]],
    hdri_rows: Sequence[Dict[str, str]],
) -> str:
    polyhaven_count = sum(1 for row in hdri_rows if row["source_name"] == "PolyHaven_HDRI")
    laval_count = sum(1 for row in hdri_rows if row["source_name"] == "Laval_HDR")
    lines = [
        "# Source Decision v2",
        "",
        "## Working Rule",
        "",
        "- keep existing true source assets intact",
        "- do not bulk download before pilot audit",
        "- split every output and future release by `license_bucket`",
        "- keep a single source-of-truth path for each asset source",
        "",
        "## Pool Decisions",
        "",
        "- `pool_A_direct_object_supervision`: keep `ABO_locked_core` as the only admitted object-level training core right now; add `3D-FUTURE_candidate` and `Objaverse-XL_filtered_candidate` only as pilot extensions pending audit",
        "- `pool_B_controlled_highlight_supervision`: treat `OLATverse`, `OpenIllumination`, and `ICTPolarReal` as signal-check sources first, not bulk downloads",
        "- `pool_C_part_material_highlight_semantics`: keep `3DCoMPaT++` as a gated follow-up source for part/material semantics",
        "- `pool_D_natural_light_hdri_bank`: Poly Haven can be curated immediately from the official API; Laval stays in controlled slot planning until manual selection is confirmed",
        "- `pool_E_material_prior_bank`: keep `OpenSVBRDF` and `MatSynth` separate from Pool-A and use them only as material prior sources",
        "- `pool_F_real_world_eval_holdout`: reserve `Stanford-ORB` and `OmniObject3D` strictly for holdout evaluation",
        "",
        "## Pilot Readiness",
        "",
        f"- ABO locked core ready now: 500 A-ready objects already present in the mini-v1 candidate pool",
        f"- 3D-FUTURE pilot queued now: {len(three_future_rows)} locally available objects",
        f"- Objaverse pilot queued now: {len(objaverse_rows)} metadata-only candidates; asset access still blocked",
        f"- HDRI bank seed count: {polyhaven_count} Poly Haven + {laval_count} Laval slots",
        "",
        "## License Notes",
        "",
        "- ABO is non-commercial on the official AWS registry, so it must remain a separate release bucket",
        "- Objaverse-XL is per-object licensed; do not mix unknown object licenses into a single public bundle",
        "- 3D-FUTURE, 3DCoMPaT++, OmniObject3D, Stanford-ORB, ICTPolarReal, and OLATverse all need custom/gated terms tracked independently",
        "",
        "## Stop Condition",
        "",
        "- after these pilots, stop before bulk expansion and report Pool-A / Pool-B / Pool-D suitability and risk",
        "",
    ]
    return "\n".join(lines)


def build_pool_b_signal_md() -> str:
    lines = [
        "# Pool B Signal Check",
        "",
        "| source | controlled lighting | mask | normals | albedo | specular separation | suitability | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        "| OLATverse | yes | yes | yes | yes | not explicit | high | project page exposes full-bright, OLAT, env relit, masks, normals, diffuse albedo |",
        "| OpenIllumination | yes | yes | unknown | unknown | unknown | medium-high | Hugging Face card clearly states 13 patterns + 142 OLAT and object/composition masks |",
        "| ICTPolarReal | yes | unknown | yes | yes | yes | high if accessible | preprint advertises polarized diffuse/specular decomposition and real-object inverse rendering signals |",
        "",
        "## Readout",
        "",
        "- best low-risk Pool-B pilot source: `OpenIllumination` because the license is already clear (`CC BY 4.0`)",
        "- best high-signal Pool-B source if access is granted: `ICTPolarReal`",
        "- most complete synthetic-style decomposition source from the public project page: `OLATverse`",
        "",
    ]
    return "\n".join(lines)


def build_license_risk_md() -> str:
    grouped = defaultdict(list)
    for row in SOURCE_REGISTRY_ROWS:
        grouped[row["license_bucket"]].append(row["source_name"])

    lines = [
        "# License Risk Report v1",
        "",
        "## Buckets",
        "",
    ]
    for license_bucket, sources in sorted(grouped.items()):
        lines.append(f"- `{license_bucket}`: {', '.join(sorted(sources))}")
    lines.extend(
        [
            "",
            "## Highest Risk",
            "",
            "- `mixed_per_object_license`: Objaverse-XL must stay per-object and cannot be pushed into a single release bundle without filtering by object license",
            "- `cc_by_nc_4_0`: ABO is usable for internal training and research, but must remain isolated from freer redistribution buckets",
            "- custom/gated sources (`3D-FUTURE_candidate`, `3DCoMPaT++`, `OmniObject3D`, `Stanford-ORB`, `ICTPolarReal`, `OLATverse`, `OpenSVBRDF`, `Laval_HDR`) all need project-side term checks before public release",
            "",
            "## Low Risk",
            "",
            "- `cc0`: Poly Haven HDRIs are the cleanest natural light source for Pool-D",
            "- `cc_by_4_0`: OpenIllumination is the cleanest Pool-B pilot source from a release perspective",
            "- `mixed_cc0_cc_by`: MatSynth is manageable if material assets stay tagged and split by license bucket",
            "",
        ]
    )
    return "\n".join(lines)


def build_pool_a_summary_md(three_future_rows: Sequence[Dict[str, str]], objaverse_rows: Sequence[Dict[str, str]]) -> str:
    three_future_classes = Counter(row["highlight_priority_class"] for row in three_future_rows)
    lines = [
        "# Pool A Pilot Summary",
        "",
        "## 3D-FUTURE Pilot",
        "",
        f"- selected_objects: {len(three_future_rows)}",
        "- local_assets: true",
        "- current_state: ready_for_local_blender_audit_and_gpu_gt_probe",
        "",
        "### Heuristic Class Mix",
        "",
    ]
    for bucket, count in sorted(three_future_classes.items()):
        lines.append(f"- {bucket}: {count}")
    lines.extend(
        [
            "",
            "## Objaverse-XL Pilot",
            "",
            f"- selected_objects: {len(objaverse_rows)}",
            "- local_assets: false",
            "- current_state: metadata-only pilot until asset access is approved",
            "- expected_blocker: source asset chain is not local, so second-pass cannot clear beyond metadata stage yet",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    three_future_rows = build_3dfuture_candidates()
    objaverse_rows = build_objaverse_candidates()
    pool_a_rows = build_pool_a_manifest(three_future_rows, objaverse_rows)
    pool_b_rows = build_pool_b_rows()
    pool_c_rows = build_pool_c_rows()
    polyhaven_rows = fetch_polyhaven_hdri_rows()
    laval_rows = build_laval_hdri_rows()
    hdri_rows = polyhaven_rows + laval_rows
    pool_e_rows = build_pool_e_rows()
    pool_f_rows = build_pool_f_rows()
    pilot_queue_rows = build_pilot_queue(three_future_rows, objaverse_rows, pool_b_rows, hdri_rows)

    write_csv(OUTPUT_SOURCE_REGISTRY, SOURCE_REGISTRY_ROWS, SOURCE_REGISTRY_FIELDS)
    write_csv(OUTPUT_POOL_A_MANIFEST, pool_a_rows)
    write_csv(OUTPUT_POOL_B_MANIFEST, pool_b_rows)
    write_csv(OUTPUT_POOL_C_MANIFEST, pool_c_rows)
    write_csv(OUTPUT_POOL_D_MANIFEST, hdri_rows)
    write_csv(OUTPUT_POOL_E_MANIFEST, pool_e_rows)
    write_csv(OUTPUT_POOL_F_MANIFEST, pool_f_rows)
    write_csv(OUTPUT_PILOT_QUEUE, pilot_queue_rows)
    write_csv(OUTPUT_POOL_A_3DFUTURE, three_future_rows)
    OUTPUT_POOL_A_3DFUTURE_JSON.write_text(json.dumps(three_future_rows, indent=2), encoding="utf-8")
    write_csv(OUTPUT_POOL_A_OBJAVERSE, objaverse_rows)
    write_csv(OUTPUT_POOL_B_PILOT, pool_b_rows)
    write_csv(OUTPUT_HDRI_BANK, hdri_rows)

    OUTPUT_SOURCE_DECISION.write_text(
        build_source_decision_md(three_future_rows, objaverse_rows, hdri_rows),
        encoding="utf-8",
    )
    OUTPUT_POOL_B_SIGNAL.write_text(build_pool_b_signal_md(), encoding="utf-8")
    OUTPUT_LICENSE_RISK.write_text(build_license_risk_md(), encoding="utf-8")
    OUTPUT_POOL_A_SUMMARY.write_text(build_pool_a_summary_md(three_future_rows, objaverse_rows), encoding="utf-8")

    print(f"wrote {OUTPUT_SOURCE_REGISTRY}")
    print(f"wrote {OUTPUT_SOURCE_DECISION}")
    print(f"wrote {OUTPUT_POOL_A_MANIFEST}")
    print(f"wrote {OUTPUT_POOL_B_MANIFEST}")
    print(f"wrote {OUTPUT_POOL_C_MANIFEST}")
    print(f"wrote {OUTPUT_POOL_D_MANIFEST}")
    print(f"wrote {OUTPUT_POOL_E_MANIFEST}")
    print(f"wrote {OUTPUT_POOL_F_MANIFEST}")
    print(f"wrote {OUTPUT_LICENSE_RISK}")
    print(f"wrote {OUTPUT_PILOT_QUEUE}")
    print(f"wrote {OUTPUT_POOL_A_3DFUTURE}")
    print(f"wrote {OUTPUT_POOL_A_OBJAVERSE}")
    print(f"wrote {OUTPUT_POOL_B_PILOT}")
    print(f"wrote {OUTPUT_POOL_B_SIGNAL}")
    print(f"wrote {OUTPUT_HDRI_BANK}")


if __name__ == "__main__":
    main()
