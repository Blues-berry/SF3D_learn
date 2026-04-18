#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DOCS_ROOT = PROJECT_ROOT / "docs" / "neural_gaffer_dataset_audit"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "neural_gaffer_dataset_audit"

INPUT_CSV = DOCS_ROOT / "pool_A_pilot_3dfuture_30.csv"
OUTPUT_CSV = INPUT_CSV
SUMMARY_MD = DOCS_ROOT / "pool_A_pilot_summary.md"
BLENDER_BIN = Path("/4T/CXY/Neural_Gaffer_original/scripts/Objavarse_rendering/blender-3.2.2-linux-x64/blender")
BLENDER_SCRIPT = PROJECT_ROOT / "scripts" / "blender_model_signal_audit.py"
AUDITOR = "blender_obj_signal_audit_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_blender_probe(object_path: Path, scratch_dir: Path) -> dict:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_json = scratch_dir / f"{object_path.parent.name}.json"
    cmd = [
        str(BLENDER_BIN),
        "-b",
        "-P",
        str(BLENDER_SCRIPT),
        "--",
        "--object-path",
        str(object_path),
        "--output-json",
        str(output_json),
    ]
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(output_json.read_text())


def classify(probe: dict) -> tuple[str, str, str]:
    if probe.get("import_status") != "ok":
        return "D_drop", "import_failed", "blender_import_failed"
    if not probe.get("has_mesh"):
        return "D_drop", "missing_mesh", "import_ok_but_no_mesh"
    if not probe.get("has_uv"):
        return "C_render_only", "missing_uv", "mesh_ok_but_no_uv"
    if int(probe.get("material_slot_count", 0)) <= 0:
        return "C_render_only", "missing_material", "mesh_ok_but_no_material_slots"
    if probe.get("has_albedo"):
        if probe.get("has_roughness") or probe.get("has_metallic"):
            return "B_fixable", "needs_conversion;needs_material_bake", "structure_ok_material_signals_partial"
        return "B_fixable", "needs_conversion;missing_roughness;missing_metallic", "albedo_only_obj_material"
    return "C_render_only", "material_signal_missing", "mesh_ok_but_material_signals_weak"


def main():
    rows = list(csv.DictReader(INPUT_CSV.open(newline="", encoding="utf-8")))
    if not rows:
        raise RuntimeError("empty_input_csv")
    scratch_dir = OUTPUT_ROOT / "tmp_pool_a_3dfuture_audit"
    audit_time = utc_now()
    status_counts = Counter()
    reason_counts = Counter()

    output_rows = []
    for row in rows:
        object_path = Path(row["source_model_path"])
        if not object_path.exists():
            row.update(
                {
                    "has_mesh": "false",
                    "has_uv": "false",
                    "has_albedo": "false",
                    "has_normal": "false",
                    "has_roughness": "false",
                    "has_metallic": "false",
                    "material_slot_count": "0",
                    "needs_conversion": "true",
                    "needs_material_bake": "unknown",
                    "audit_status": "D_drop",
                    "reject_reason": "source_path_missing",
                    "notes": f"{row['notes']}; source_path_missing",
                    "audit_time": audit_time,
                    "auditor": AUDITOR,
                }
            )
            status_counts[row["audit_status"]] += 1
            reason_counts[row["reject_reason"]] += 1
            output_rows.append(row)
            continue

        probe = run_blender_probe(object_path, scratch_dir)
        audit_status, reject_reason, note = classify(probe)
        row.update(
            {
                "has_mesh": str(bool(probe.get("has_mesh", False))).lower(),
                "has_uv": str(bool(probe.get("has_uv", False))).lower(),
                "has_albedo": str(bool(probe.get("has_albedo", False))).lower(),
                "has_normal": str(bool(probe.get("has_normal", False))).lower(),
                "has_roughness": str(bool(probe.get("has_roughness", False))).lower(),
                "has_metallic": str(bool(probe.get("has_metallic", False))).lower(),
                "material_slot_count": str(probe.get("material_slot_count", 0)),
                "needs_conversion": "true",
                "needs_material_bake": "true" if not (probe.get("has_roughness") and probe.get("has_metallic")) else "unknown",
                "audit_status": audit_status,
                "reject_reason": reject_reason,
                "notes": f"{row['notes']}; {note}; textures={probe.get('texture_image_count', 0)}; principled_nodes={probe.get('principled_node_count', 0)}",
                "audit_time": audit_time,
                "auditor": AUDITOR,
            }
        )
        status_counts[audit_status] += 1
        if reject_reason:
            reason_counts[reject_reason] += 1
        output_rows.append(row)

    fieldnames = list(output_rows[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "# Pool A Pilot Summary",
        "",
        "## 3D-FUTURE Pilot Audit",
        "",
        f"- audited_objects: {len(output_rows)}",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Top Reasons", ""])
    for reason, count in reason_counts.most_common(10):
        lines.append(f"- {reason}: {count}")
    lines.extend(
        [
            "",
            "## Objaverse-XL Pilot",
            "",
            "- selected_objects: 30",
            "- current_state: metadata_only",
            "- block: source_asset_not_local",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
