#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DOCS_ROOT = PROJECT_ROOT / "docs" / "neural_gaffer_dataset_audit"
FIRST_PASS_CSV = DOCS_ROOT / "asset_supervision_audit_first_pass.csv"

OUT_300 = DOCS_ROOT / "asset_supervision_audit_second_pass_ecommerce_glb_priority_300.csv"
OUT_500 = DOCS_ROOT / "asset_supervision_audit_second_pass_ecommerce_glb_priority_500.csv"
OUT_CONV = DOCS_ROOT / "asset_supervision_audit_second_pass_ecommerce_conversion_pool.csv"
OUT_LAND = DOCS_ROOT / "asset_supervision_audit_second_pass_landscape_ood_pool.csv"
OUT_PLAN = DOCS_ROOT / "asset_supervision_second_pass_plan.md"


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def priority_key(row):
    # Keep the first mini-batch deterministic and easy to diff.
    return (
        row["source"],
        row["object_id"],
    )


def build_plan(rows, hi_rows, conv_rows, land_rows) -> str:
    total = Counter(row["subset"] for row in rows)
    hi_by_source = Counter(row["source"] for row in hi_rows)
    conv_by_format = Counter(row["format"] for row in conv_rows)
    land_by_format = Counter(row["format"] for row in land_rows)

    lines = []
    lines.append("# Asset Supervision Second-Pass Plan")
    lines.append("")
    lines.append("## Route")
    lines.append("")
    lines.append("- 主线: `ecommerce`")
    lines.append("- 后置 OOD: `landscape`")
    lines.append("- reference only: `official_2000`")
    lines.append("")
    lines.append("## Pool Sizes")
    lines.append("")
    lines.append(f"- ecommerce total: {total['ecommerce']}")
    lines.append(f"- landscape total: {total['landscape']}")
    lines.append(f"- official_2000 total: {total['official_2000']}")
    lines.append("")
    lines.append("## Priority Tiers")
    lines.append("")
    lines.append(f"- Tier 1: ecommerce native `glb/gltf`, `needs_conversion = false` -> {len(hi_rows)} objects")
    lines.append(f"- Tier 2: ecommerce conversion pool (`obj/fbx`) -> {len(conv_rows)} objects")
    lines.append(f"- Tier 3: landscape OOD pool -> {len(land_rows)} objects")
    lines.append("")
    lines.append("### Tier 1 Source Breakdown")
    lines.append("")
    lines.append("| source | count |")
    lines.append("| --- | ---: |")
    for source, count in sorted(hi_by_source.items()):
        lines.append(f"| {source} | {count} |")
    lines.append("")
    lines.append("### Tier 2 Format Breakdown")
    lines.append("")
    lines.append("| format | count |")
    lines.append("| --- | ---: |")
    for fmt, count in sorted(conv_by_format.items()):
        lines.append(f"| {fmt} | {count} |")
    lines.append("")
    lines.append("### Tier 3 Format Breakdown")
    lines.append("")
    lines.append("| format | count |")
    lines.append("| --- | ---: |")
    for fmt, count in sorted(land_by_format.items()):
        lines.append(f"| {fmt} | {count} |")
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append(f"- [{OUT_300.name}]({OUT_300})")
    lines.append(f"- [{OUT_500.name}]({OUT_500})")
    lines.append(f"- [{OUT_CONV.name}]({OUT_CONV})")
    lines.append(f"- [{OUT_LAND.name}]({OUT_LAND})")
    lines.append("")
    lines.append("## Second-Pass Scope")
    lines.append("")
    lines.append("第二轮只建议优先补这些列:")
    lines.append("")
    lines.append("- `has_uv`")
    lines.append("- `has_albedo`")
    lines.append("- `has_roughness`")
    lines.append("- `has_metallic`")
    lines.append("- `has_normal`")
    lines.append("- `roughness_valid`")
    lines.append("- `metallic_valid`")
    lines.append("- `needs_material_bake`")
    lines.append("- `reject_reason`")
    lines.append("- `notes`")
    lines.append("")
    lines.append("## Upgrade Rules")
    lines.append("")
    lines.append("### Promote to `A_ready`")
    lines.append("")
    lines.append("- `has_uv = true`")
    lines.append("- `has_albedo = true`")
    lines.append("- `has_roughness = true`")
    lines.append("- `has_metallic = true`")
    lines.append("- `roughness_valid = true`")
    lines.append("- `metallic_valid = true`")
    lines.append("- `needs_conversion = false`")
    lines.append("- `needs_material_bake = false`")
    lines.append("")
    lines.append("### Keep as `B_fixable`")
    lines.append("")
    lines.append("- asset structure is good")
    lines.append("- but still needs conversion / relink / bake / node extraction")
    lines.append("")
    lines.append("### Downgrade to `C_render_only`")
    lines.append("")
    lines.append("- mesh is readable, but RM supervision is not成立")
    lines.append("- e.g. no UV, base color only, or roughness / metallic are constant placeholders")
    lines.append("")
    lines.append("### Mark as `D_drop`")
    lines.append("")
    lines.append("- import failure")
    lines.append("- corrupt file")
    lines.append("- irrecoverable material")
    lines.append("- repair cost is not worth it")
    lines.append("")
    lines.append("## Recommended Next Action")
    lines.append("")
    lines.append("- 先对 `ecommerce_glb_priority_300` 做 second-pass mini-batch probe")
    lines.append("- 如果 `A_ready` 比例足够，再扩到 `500`")
    lines.append("- `ecommerce_conversion_pool` 先只做格式统一，不跟材质 probe 混跑")
    lines.append("- `landscape_ood_pool` 暂时只保留为后置泛化池")
    lines.append("")
    return "\n".join(lines)


def main():
    rows = read_rows(FIRST_PASS_CSV)
    ecommerce = [r for r in rows if r["subset"] == "ecommerce"]
    landscape = [r for r in rows if r["subset"] == "landscape"]

    hi_rows = sorted(
        [
            r
            for r in ecommerce
            if r["format"] in {"glb", "gltf"} and r["needs_conversion"] == "false"
        ],
        key=priority_key,
    )
    conv_rows = sorted(
        [r for r in ecommerce if r["needs_conversion"] == "true"],
        key=priority_key,
    )
    land_rows = sorted(landscape, key=priority_key)

    write_rows(OUT_300, hi_rows[:300])
    write_rows(OUT_500, hi_rows[:500])
    write_rows(OUT_CONV, conv_rows)
    write_rows(OUT_LAND, land_rows)
    OUT_PLAN.write_text(build_plan(rows, hi_rows, conv_rows, land_rows), encoding="utf-8")

    print(f"wrote {OUT_300}")
    print(f"wrote {OUT_500}")
    print(f"wrote {OUT_CONV}")
    print(f"wrote {OUT_LAND}")
    print(f"wrote {OUT_PLAN}")
    print(f"tier1={len(hi_rows)} tier2={len(conv_rows)} tier3={len(land_rows)}")


if __name__ == "__main__":
    main()
