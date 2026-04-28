#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DOCS_ROOT = PROJECT_ROOT / "docs" / "neural_gaffer_dataset_audit"
INPUT_CSV = DOCS_ROOT / "asset_supervision_audit_second_pass_ecommerce_glb_priority_300_probed.csv"
OUTPUT_CSV = DOCS_ROOT / "asset_supervision_semantic_validation_subset_24.csv"
OUTPUT_JSON = DOCS_ROOT / "asset_supervision_semantic_validation_subset_24.json"
OUTPUT_MD = DOCS_ROOT / "asset_supervision_semantic_validation_subset_24.md"


def read_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def take_evenly(rows: List[dict], count: int) -> List[dict]:
    if count <= 0:
        return []
    if len(rows) <= count:
        return list(rows)
    selected = []
    seen = set()
    for idx in range(count):
        pos = round(idx * (len(rows) - 1) / (count - 1)) if count > 1 else 0
        while pos in seen and pos + 1 < len(rows):
            pos += 1
        seen.add(pos)
        selected.append(rows[pos])
    return selected


def sku_from_path(path_str: str) -> str:
    return Path(path_str).stem


def build_manifest_entry(row: dict, stratum: str, reason: str) -> dict:
    sku = sku_from_path(row["source_model_path"])
    return {
        "id": row["object_id"],
        "label": sku,
        "name": sku,
        "category": "abo_semantic_validation",
        "local_path": row["source_model_path"],
        "semantic_stratum": stratum,
        "selection_reason": reason,
        "material_slot_count": row["material_slot_count"],
        "notes": row["notes"],
    }


def rows_with_mode(rows: Iterable[dict], mode: str) -> List[dict]:
    token = f"albedo={mode}"
    return sorted([row for row in rows if token in row["notes"]], key=lambda row: row["object_id"])


def main():
    rows = [row for row in read_rows(INPUT_CSV) if row["audit_status"] == "A_ready"]
    rows = sorted(rows, key=lambda row: row["object_id"])

    dual_material = [row for row in rows if int(row["material_slot_count"]) > 1]
    texture_only = [
        row
        for row in rows_with_mode(rows, "texture")
        if int(row["material_slot_count"]) == 1
    ]
    factor_texture = [
        row
        for row in rows_with_mode(rows, "factor+texture")
        if int(row["material_slot_count"]) == 1
    ]

    selected = []
    used_ids = set()

    def add_group(candidates: List[dict], count: int, stratum: str, reason: str):
        chosen = take_evenly(candidates, count)
        for row in chosen:
            if row["object_id"] in used_ids:
                continue
            used_ids.add(row["object_id"])
            selected.append((row, stratum, reason))

    add_group(dual_material, min(7, len(dual_material)), "dual_material", "include_all_dual_material_candidates")
    add_group(texture_only, 8, "single_material_texture_only", "cover_texture_only_cases")
    add_group(factor_texture, 9, "single_material_factor_texture", "cover_factor_texture_majority_cases")

    selected_rows = []
    manifest = []
    for row, stratum, reason in selected:
        out_row = dict(row)
        out_row["semantic_stratum"] = stratum
        out_row["selection_reason"] = reason
        out_row["sku"] = sku_from_path(row["source_model_path"])
        selected_rows.append(out_row)
        manifest.append(build_manifest_entry(row, stratum, reason))

    fieldnames = list(selected_rows[0].keys()) if selected_rows else []
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    OUTPUT_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Semantic Validation Subset 24",
        "",
        f"- input: {INPUT_CSV}",
        f"- output_csv: {OUTPUT_CSV}",
        f"- output_json: {OUTPUT_JSON}",
        "",
        "## Selection Rule",
        "",
        "- include all dual-material A_ready objects found in the current 300-object pass",
        "- add 8 single-material texture-only objects",
        "- add 9 single-material factor+texture objects",
        "",
        "## Breakdown",
        "",
        f"- dual_material: {sum(1 for _, s, _ in selected if s == 'dual_material')}",
        f"- single_material_texture_only: {sum(1 for _, s, _ in selected if s == 'single_material_texture_only')}",
        f"- single_material_factor_texture: {sum(1 for _, s, _ in selected if s == 'single_material_factor_texture')}",
        "",
        "## Selected Objects",
        "",
    ]
    for row, stratum, _reason in selected:
        lines.append(f"- {row['object_id']} | {sku_from_path(row['source_model_path'])} | {stratum}")
    lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_MD}")
    print(f"selected={len(selected_rows)}")


if __name__ == "__main__":
    main()
