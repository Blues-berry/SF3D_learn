#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DOCS_ROOT = PROJECT_ROOT / "docs" / "neural_gaffer_dataset_audit"
INPUT_CSV = DOCS_ROOT / "asset_supervision_audit_second_pass_ecommerce_glb_priority_300.csv"
OUTPUT_CSV = DOCS_ROOT / "asset_supervision_audit_second_pass_ecommerce_glb_priority_300_probed.csv"
SUMMARY_MD = DOCS_ROOT / "asset_supervision_audit_second_pass_ecommerce_glb_priority_300_summary.md"
AUDITOR = "second_pass_glb_json_probe_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    parser.add_argument("--auditor", type=str, default=AUDITOR)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_glb_json(path: Path) -> Dict:
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


def material_probe(doc: Dict) -> Dict[str, object]:
    meshes = doc.get("meshes", [])
    materials = doc.get("materials", [])
    has_mesh = False
    has_uv = False

    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            has_mesh = True
            attrs = prim.get("attributes", {})
            if any(name.startswith("TEXCOORD_") for name in attrs):
                has_uv = True

    has_albedo = False
    has_normal = False
    has_roughness = False
    has_metallic = False
    roughness_valid = False
    metallic_valid = False

    albedo_modes = set()
    rough_modes = set()
    metallic_modes = set()
    warnings = []

    for mat in materials:
        pbr = mat.get("pbrMetallicRoughness", {})

        if "baseColorTexture" in pbr:
            has_albedo = True
            albedo_modes.add("texture")
        if "baseColorFactor" in pbr:
            has_albedo = True
            albedo_modes.add("factor")

        if "normalTexture" in mat:
            has_normal = True

        if "metallicRoughnessTexture" in pbr:
            has_roughness = True
            has_metallic = True
            roughness_valid = True
            metallic_valid = True
            rough_modes.add("texture")
            metallic_modes.add("texture")

        if "roughnessFactor" in pbr:
            has_roughness = True
            rough_modes.add("factor")
            roughness_valid = True
            if float(pbr.get("roughnessFactor", 1.0)) == 1.0 and "metallicRoughnessTexture" not in pbr:
                warnings.append("roughness_factor_default_only")

        if "metallicFactor" in pbr:
            has_metallic = True
            metallic_modes.add("factor")
            metallic_valid = True
            if float(pbr.get("metallicFactor", 1.0)) == 1.0 and "metallicRoughnessTexture" not in pbr:
                warnings.append("metallic_factor_default_only")

    if has_albedo and not albedo_modes:
        warnings.append("albedo_source_unclear")

    return {
        "has_mesh": has_mesh,
        "has_uv": has_uv,
        "has_albedo": has_albedo,
        "has_normal": has_normal,
        "has_roughness": has_roughness,
        "has_metallic": has_metallic,
        "roughness_valid": roughness_valid,
        "metallic_valid": metallic_valid,
        "material_slot_count": len(materials),
        "albedo_modes": sorted(albedo_modes),
        "rough_modes": sorted(rough_modes),
        "metallic_modes": sorted(metallic_modes),
        "warnings": warnings,
    }


def classify(row: Dict[str, str], probe: Dict[str, object]) -> Tuple[str, str, str]:
    if row["needs_conversion"] == "true":
        return "B_fixable", "needs_conversion", "conversion_required"

    if not probe["has_mesh"]:
        return "D_drop", "missing_mesh", "glb_no_mesh"

    if not probe["has_uv"]:
        return "C_render_only", "missing_uv", "mesh_readable_but_no_uv"

    if int(probe["material_slot_count"]) <= 0:
        return "C_render_only", "missing_material", "mesh_readable_but_no_material"

    required = [
        probe["has_albedo"],
        probe["has_roughness"],
        probe["has_metallic"],
        probe["roughness_valid"],
        probe["metallic_valid"],
    ]
    if all(required):
        warning_tags = set(probe["warnings"])
        if "roughness_factor_default_only" in warning_tags or "metallic_factor_default_only" in warning_tags:
            return "B_fixable", "needs_material_probe;default_factor_only", "explicit_pbr_but_default_factor_needs_review"
        return "A_ready", "", "uv_and_pbr_channels_present"

    reasons = []
    if not probe["has_albedo"]:
        reasons.append("missing_albedo")
    if not probe["has_roughness"]:
        reasons.append("missing_roughness")
    if not probe["has_metallic"]:
        reasons.append("missing_metallic")
    if probe["has_roughness"] and not probe["roughness_valid"]:
        reasons.append("roughness_invalid")
    if probe["has_metallic"] and not probe["metallic_valid"]:
        reasons.append("metallic_invalid")

    if reasons:
        # Still keep it fixable if the object has mesh, UV, and material structure.
        return "B_fixable", ";".join(reasons), "pbr_structure_present_but_channels_incomplete"

    return "B_fixable", "needs_material_probe", "requires_manual_review"


def notes_from_probe(probe: Dict[str, object], prior_notes: str) -> str:
    parts = []
    if prior_notes:
        parts.append(prior_notes)
    if probe["albedo_modes"]:
        parts.append("albedo=" + "+".join(probe["albedo_modes"]))
    if probe["rough_modes"]:
        parts.append("roughness=" + "+".join(probe["rough_modes"]))
    if probe["metallic_modes"]:
        parts.append("metallic=" + "+".join(probe["metallic_modes"]))
    if probe["warnings"]:
        parts.append("warnings=" + ",".join(sorted(set(probe["warnings"]))))
    return "; ".join(parts)


def run():
    args = parse_args()
    input_csv = args.input_csv
    output_csv = args.output_csv
    summary_md = args.summary_md
    auditor = args.auditor
    audit_time = utc_now()
    rows: List[Dict[str, str]] = []

    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError("missing_csv_header")
        for row in reader:
            source_path = Path(row["source_model_path"])
            if not source_path.exists():
                row.update(
                    {
                        "has_mesh": "false",
                        "has_uv": "unknown",
                        "has_albedo": "unknown",
                        "has_normal": "unknown",
                        "has_roughness": "unknown",
                        "has_metallic": "unknown",
                        "roughness_valid": "unknown",
                        "metallic_valid": "unknown",
                        "material_slot_count": "unknown",
                        "needs_material_bake": "unknown",
                        "audit_status": "D_drop",
                        "reject_reason": "source_path_missing",
                        "notes": "second_pass_probe_failed",
                        "audit_time": audit_time,
                        "auditor": auditor,
                    }
                )
                rows.append(row)
                continue

            try:
                doc = load_glb_json(source_path)
                probe = material_probe(doc)
            except Exception as exc:
                row.update(
                    {
                        "has_mesh": "false",
                        "has_uv": "unknown",
                        "has_albedo": "unknown",
                        "has_normal": "unknown",
                        "has_roughness": "unknown",
                        "has_metallic": "unknown",
                        "roughness_valid": "unknown",
                        "metallic_valid": "unknown",
                        "material_slot_count": "unknown",
                        "needs_material_bake": "unknown",
                        "audit_status": "D_drop",
                        "reject_reason": "glb_parse_failed",
                        "notes": f"second_pass_probe_failed:{type(exc).__name__}",
                        "audit_time": audit_time,
                        "auditor": auditor,
                    }
                )
                rows.append(row)
                continue

            status, reject_reason, notes = classify(row, probe)
            needs_material_bake = "false" if status == "A_ready" else ("true" if status == "B_fixable" else "unknown")
            row.update(
                {
                    "has_mesh": str(probe["has_mesh"]).lower(),
                    "has_uv": str(probe["has_uv"]).lower(),
                    "has_albedo": str(probe["has_albedo"]).lower(),
                    "has_normal": str(probe["has_normal"]).lower(),
                    "has_roughness": str(probe["has_roughness"]).lower(),
                    "has_metallic": str(probe["has_metallic"]).lower(),
                    "roughness_valid": str(probe["roughness_valid"]).lower(),
                    "metallic_valid": str(probe["metallic_valid"]).lower(),
                    "material_slot_count": str(probe["material_slot_count"]),
                    "needs_material_bake": needs_material_bake,
                    "audit_status": status,
                    "reject_reason": reject_reason,
                    "notes": notes_from_probe(probe, notes),
                    "audit_time": audit_time,
                    "auditor": auditor,
                }
            )
            rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(row["audit_status"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    reason_counts = Counter()
    for row in rows:
        for reason in row["reject_reason"].split(";"):
            reason = reason.strip()
            if reason:
                reason_counts[reason] += 1

    title = input_csv.stem.replace("_", " ")
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- audit_time: {audit_time}")
    lines.append(f"- auditor: {auditor}")
    lines.append(f"- input: {input_csv}")
    lines.append(f"- output: {output_csv}")
    lines.append("")
    lines.append("## Scope Note")
    lines.append("")
    lines.append("- This pass is a GLB JSON/material-structure audit.")
    lines.append("- It confirms mesh, UV, and declared PBR channel presence from local source assets.")
    lines.append("- It does not yet prove that roughness/metallic maps are semantically strong supervision signals or non-constant textures.")
    lines.append("")
    lines.append("## Source Breakdown")
    lines.append("")
    lines.append("| source | count | ratio |")
    lines.append("| --- | ---: | ---: |")
    total = len(rows)
    for source, count in source_counts.most_common():
        ratio = 0.0 if total == 0 else 100.0 * count / total
        lines.append(f"| {source} | {count} | {ratio:.1f}% |")
    lines.append("")
    lines.append("## Status Breakdown")
    lines.append("")
    lines.append("| status | count | ratio |")
    lines.append("| --- | ---: | ---: |")
    for status in ["A_ready", "B_fixable", "C_render_only", "D_drop"]:
        count = status_counts.get(status, 0)
        ratio = 0.0 if total == 0 else 100.0 * count / total
        lines.append(f"| {status} | {count} | {ratio:.1f}% |")
    lines.append("")
    lines.append("## Top Reject Reasons")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("| --- | ---: |")
    for reason, count in reason_counts.most_common(12):
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    lines.append("## Channel Coverage")
    lines.append("")
    for key in ["has_uv", "has_albedo", "has_normal", "has_roughness", "has_metallic", "roughness_valid", "metallic_valid"]:
        count_true = sum(1 for row in rows if row[key] == "true")
        ratio = 0.0 if total == 0 else 100.0 * count_true / total
        lines.append(f"- {key}: {count_true}/{total} ({ratio:.1f}%)")
    lines.append("")
    lines.append("## Decision Hint")
    lines.append("")
    lines.append("- 如果 `A_ready` 已经够形成 mini-v1，就继续扩到 500。")
    lines.append("- 如果 `A_ready` 偏少，但 `B_fixable` 主要集中在材质字段缺失，就优先补 material bake / channel extraction。")
    lines.append("- 如果大面积掉到 `C_render_only`，再考虑引入更干净的新源做增强池。")
    lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {output_csv}")
    print(f"wrote {summary_md}")
    print(dict(status_counts))


if __name__ == "__main__":
    run()
