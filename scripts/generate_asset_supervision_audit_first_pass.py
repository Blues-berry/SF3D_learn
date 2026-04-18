#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
ORIGINAL_ROOT = Path("/4T/CXY/Neural_Gaffer_original")
DOCS_ROOT = PROJECT_ROOT / "docs" / "neural_gaffer_dataset_audit"

ECOMMERCE_MANIFEST = ORIGINAL_ROOT / "external_sources" / "manifests" / "ecommerce_external_2000.json"
LANDSCAPE_MANIFEST = ORIGINAL_ROOT / "external_sources" / "manifests" / "landscape_external_1000.json"
OFFICIAL_MANIFEST = ORIGINAL_ROOT / "subdataset" / "official" / "manifests" / "official_target_2000.json"

FIRST_PASS_CSV = DOCS_ROOT / "asset_supervision_audit_first_pass.csv"
SUMMARY_MD = DOCS_ROOT / "asset_supervision_audit_summary.md"

COLUMNS = [
    "object_id",
    "subset",
    "source",
    "source_uid",
    "source_model_path",
    "texture_root",
    "format",
    "has_mesh",
    "has_uv",
    "has_albedo",
    "has_normal",
    "has_roughness",
    "has_metallic",
    "roughness_valid",
    "metallic_valid",
    "material_slot_count",
    "needs_conversion",
    "needs_material_bake",
    "audit_status",
    "reject_reason",
    "license",
    "notes",
    "audit_time",
    "auditor",
]

UNKNOWN = "unknown"
AUDITOR = "first_pass_auto_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path):
    return json.loads(path.read_text())


def safe_suffix(path_str: str) -> str:
    suffix = Path(path_str).suffix.lower().lstrip(".")
    return suffix or UNKNOWN


def derive_source_from_path(path_str: str, subset: str) -> str:
    path = Path(path_str)
    parts = path.parts
    if "downloads" in parts:
        idx = parts.index("downloads")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if subset == "official_2000":
        return "objaverse"
    return UNKNOWN


def bool_str(value: Optional[bool]) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return UNKNOWN


def count_obj_materials(model_path: Path) -> Tuple[Optional[int], Optional[Path]]:
    usemtl = set()
    mtllibs = []
    try:
        with model_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("usemtl "):
                    usemtl.add(line.split(None, 1)[1].strip())
                elif line.startswith("mtllib "):
                    mtllibs.append(line.split(None, 1)[1].strip())
    except OSError:
        return None, None

    chosen_mtl = None
    newmtl = []
    for mtl_name in mtllibs:
        candidate = model_path.parent / mtl_name
        if candidate.exists():
            chosen_mtl = candidate
            try:
                with candidate.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        line = line.strip()
                        if line.startswith("newmtl "):
                            newmtl.append(line.split(None, 1)[1].strip())
            except OSError:
                pass
            break

    if newmtl:
        return len(set(newmtl)), chosen_mtl
    if usemtl:
        return len(usemtl), chosen_mtl
    return 0, chosen_mtl


def obj_has_mesh(model_path: Path) -> bool:
    has_vertex = False
    has_face = False
    try:
        with model_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("v "):
                    has_vertex = True
                elif line.startswith("f "):
                    has_face = True
                if has_vertex and has_face:
                    return True
    except OSError:
        return False
    return has_vertex and has_face


def fbx_material_count(model_path: Path) -> Optional[int]:
    try:
        data = model_path.read_bytes()
    except OSError:
        return None
    text = data[:2_000_000].decode("utf-8", errors="ignore")
    markers = ["Material::", '"Material"']
    count = 0
    for marker in markers:
        count = max(count, text.count(marker))
    return count if count > 0 else None


def probe_mesh_and_materials(model_path: Path, fmt: str) -> Tuple[Optional[bool], Optional[int], Optional[Path]]:
    if not model_path.exists() or model_path.stat().st_size <= 0:
        return False, None, None

    if fmt == "obj":
        count, mtl_path = count_obj_materials(model_path)
        return obj_has_mesh(model_path), count, mtl_path
    if fmt in {"gltf", "glb"}:
        # first-pass keeps this intentionally cheap: existence + non-empty file
        # is enough to mark the object as probe-worthy, while UV/PBR channels
        # stay unknown until the next round.
        return True, None, model_path.parent
    if fmt == "fbx":
        return True, fbx_material_count(model_path), model_path.parent
    return None, None, None


def texture_root_from_probe(model_path: Path, fmt: str, mtl_or_root: Optional[Path]) -> str:
    if fmt == "obj" and mtl_or_root is not None:
        return str(mtl_or_root.parent)
    if fmt in {"glb", "gltf", "fbx"}:
        return str(model_path.parent)
    return ""


def make_row(
    *,
    object_id: str,
    subset: str,
    source: str,
    source_uid: str,
    source_model_path: str,
    fmt: str,
    has_mesh: Optional[bool],
    material_slot_count: Optional[int],
    needs_conversion: Optional[bool],
    needs_material_bake: str,
    audit_status: str,
    reject_reason: str,
    notes: str,
    texture_root: str,
    audit_time: str,
) -> Dict[str, str]:
    row = {key: "" for key in COLUMNS}
    row.update(
        {
            "object_id": object_id,
            "subset": subset,
            "source": source,
            "source_uid": source_uid,
            "source_model_path": source_model_path,
            "texture_root": texture_root,
            "format": fmt,
            "has_mesh": bool_str(has_mesh),
            "has_uv": UNKNOWN,
            "has_albedo": UNKNOWN,
            "has_normal": UNKNOWN,
            "has_roughness": UNKNOWN,
            "has_metallic": UNKNOWN,
            "roughness_valid": UNKNOWN,
            "metallic_valid": UNKNOWN,
            "material_slot_count": str(material_slot_count) if material_slot_count is not None else UNKNOWN,
            "needs_conversion": bool_str(needs_conversion),
            "needs_material_bake": needs_material_bake,
            "audit_status": audit_status,
            "reject_reason": reject_reason,
            "license": "",
            "notes": notes,
            "audit_time": audit_time,
            "auditor": AUDITOR,
        }
    )
    return row


def rows_from_external_manifest(manifest_path: Path, subset: str, audit_time: str) -> List[Dict[str, str]]:
    items = load_json(manifest_path)
    rows: List[Dict[str, str]] = []
    for item in items:
        object_id = item["uid"]
        source_uid = object_id
        source_model_path = item["path"]
        source = derive_source_from_path(source_model_path, subset)
        fmt = safe_suffix(source_model_path)
        model_path = Path(source_model_path)
        exists = model_path.exists() and model_path.stat().st_size > 0 if model_path.exists() else False
        has_mesh = exists
        material_slot_count = None
        needs_conversion = fmt in {"obj", "fbx"}
        texture_root = str(model_path.parent) if exists else ""

        if not exists:
            status = "D_drop"
            reason = "source_path_missing"
            notes = "drop_in_first_pass"
        else:
            status = "B_fixable"
            if fmt == "obj":
                reason = "needs_obj_to_glb;needs_material_probe"
            elif fmt == "fbx":
                reason = "needs_fbx_to_glb;needs_material_probe"
            else:
                reason = "needs_material_probe"
            notes = "main_pool_candidate"

        rows.append(
            make_row(
                object_id=object_id,
                subset=subset,
                source=source,
                source_uid=source_uid,
                source_model_path=source_model_path,
                fmt=fmt,
                has_mesh=has_mesh,
                material_slot_count=material_slot_count,
                needs_conversion=needs_conversion,
                needs_material_bake=UNKNOWN,
                audit_status=status,
                reject_reason=reason,
                notes=notes,
                texture_root=texture_root,
                audit_time=audit_time,
            )
        )
    return rows


def rows_from_official_manifest(manifest_path: Path, audit_time: str) -> List[Dict[str, str]]:
    items = load_json(manifest_path)
    rows: List[Dict[str, str]] = []
    for object_id, rel_path in items.items():
        fmt = safe_suffix(rel_path)
        rows.append(
            make_row(
                object_id=object_id,
                subset="official_2000",
                source="objaverse",
                source_uid=object_id,
                source_model_path=rel_path,
                fmt=fmt,
                has_mesh=None,
                material_slot_count=None,
                needs_conversion=None,
                needs_material_bake=UNKNOWN,
                audit_status="C_render_only",
                reject_reason="source_asset_not_local",
                notes="legacy_reference_only",
                texture_root="",
                audit_time=audit_time,
            )
        )
    return rows


def write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def top_reasons(rows: Iterable[Dict[str, str]], limit: int = 8) -> List[Tuple[str, int]]:
    counter: Counter[str] = Counter()
    for row in rows:
        for reason in row["reject_reason"].split(";"):
            reason = reason.strip()
            if reason:
                counter[reason] += 1
    return counter.most_common(limit)


def percent(part: int, whole: int) -> str:
    if whole == 0:
        return "0.0%"
    return f"{(part / whole) * 100:.1f}%"


def build_summary(rows: List[Dict[str, str]]) -> str:
    subsets = ["ecommerce", "landscape", "official_2000"]
    status_order = ["A_ready", "B_fixable", "C_render_only", "D_drop"]
    by_subset: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_subset[row["subset"]].append(row)

    lines: List[str] = []
    lines.append("# Asset Supervision Audit Summary")
    lines.append("")
    lines.append(f"- audit_time: {rows[0]['audit_time'] if rows else utc_now()}")
    lines.append(f"- auditor: {AUDITOR}")
    lines.append("- main_supervision_pool: ecommerce + landscape")
    lines.append("- legacy_reference_pool: official_2000")
    lines.append("")
    lines.append("## Per-Subset Totals")
    lines.append("")
    lines.append("| subset | total_objects |")
    lines.append("| --- | ---: |")
    for subset in subsets:
        lines.append(f"| {subset} | {len(by_subset[subset])} |")
    lines.append("")
    lines.append("## Status Breakdown")
    lines.append("")
    lines.append("| subset | A_ready | B_fixable | C_render_only | D_drop |")
    lines.append("| --- | --- | --- | --- | --- |")
    for subset in subsets:
        subset_rows = by_subset[subset]
        total = len(subset_rows)
        counts = Counter(row["audit_status"] for row in subset_rows)
        cells = []
        for status in status_order:
            count = counts.get(status, 0)
            cells.append(f"{count} ({percent(count, total)})")
        lines.append(f"| {subset} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines.append("")

    main_rows = by_subset["ecommerce"] + by_subset["landscape"]
    legacy_rows = by_subset["official_2000"]

    for title, subset_rows in [
        ("Main Supervision Pool", main_rows),
        ("Legacy / Reference Pool", legacy_rows),
    ]:
        total = len(subset_rows)
        counts = Counter(row["audit_status"] for row in subset_rows)
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| status | count | ratio |")
        lines.append("| --- | ---: | ---: |")
        for status in status_order:
            count = counts.get(status, 0)
            lines.append(f"| {status} | {count} | {percent(count, total)} |")
        lines.append("")
        lines.append("### Top Reject Reasons")
        lines.append("")
        lines.append("| reason | count |")
        lines.append("| --- | ---: |")
        for reason, count in top_reasons(subset_rows):
            lines.append(f"| {reason} | {count} |")
        lines.append("")
        next_probe = sum(1 for row in subset_rows if row["audit_status"] == "B_fixable")
        lines.append(f"### Next-Round Probe Count")
        lines.append("")
        lines.append(f"- {next_probe}")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- first-pass 只填最稳定、最便宜、最不容易误判的列。")
    lines.append("- `has_uv` / `has_albedo` / `has_normal` / `has_roughness` / `has_metallic` / `roughness_valid` / `metallic_valid` 在这一轮允许保留 `unknown`。")
    lines.append("- `official_2000` 被单列为 legacy/reference，不参与主监督池 ready rate。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    audit_time = utc_now()
    rows: List[Dict[str, str]] = []
    rows.extend(rows_from_external_manifest(ECOMMERCE_MANIFEST, "ecommerce", audit_time))
    rows.extend(rows_from_external_manifest(LANDSCAPE_MANIFEST, "landscape", audit_time))
    rows.extend(rows_from_official_manifest(OFFICIAL_MANIFEST, audit_time))

    rows.sort(key=lambda row: (row["subset"], row["object_id"]))
    write_csv(rows, FIRST_PASS_CSV)
    SUMMARY_MD.write_text(build_summary(rows), encoding="utf-8")

    counts = Counter(row["subset"] for row in rows)
    print(f"wrote {FIRST_PASS_CSV}")
    print(f"wrote {SUMMARY_MD}")
    for subset, count in sorted(counts.items()):
        print(f"{subset}: {count}")


if __name__ == "__main__":
    main()
