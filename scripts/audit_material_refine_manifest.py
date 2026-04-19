from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REQUIRED_PATH_FIELDS = [
    "canonical_mesh_path",
    "canonical_glb_path",
    "uv_albedo_path",
    "uv_normal_path",
    "uv_prior_roughness_path",
    "uv_prior_metallic_path",
    "uv_target_roughness_path",
    "uv_target_metallic_path",
    "uv_target_confidence_path",
    "canonical_views_json",
    "canonical_buffer_root",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CanonicalAssetRecordV1 material-refine manifests for completeness and target/prior leakage.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=-1, help="-1 means audit all records.")
    parser.add_argument("--identity-warning-threshold", type=float, default=0.95)
    return parser.parse_args()


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    records = payload.get("records") or payload.get("objects") or payload.get("rows")
    if not isinstance(records, list):
        raise TypeError(f"unsupported_manifest_records:{path}")
    return payload, [record for record in records if isinstance(record, dict)]


def resolve_record_path(
    manifest_path: Path,
    payload: dict[str, Any],
    record: dict[str, Any],
    value: str | None,
) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    bundle_root_value = (
        record.get("bundle_root")
        or record.get("canonical_bundle_root")
        or payload.get("canonical_bundle_root")
        or payload.get("bundle_root")
    )
    if bundle_root_value:
        bundle_root = Path(str(bundle_root_value))
        if not bundle_root.is_absolute():
            bundle_root = manifest_path.parent / bundle_root
        candidate = bundle_root / path
        if candidate.exists():
            return candidate
    return manifest_path.parent / path


def file_digest(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gray_stats(path: Path | None) -> dict[str, float] | None:
    if path is None or not path.exists():
        return None
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.quantile(arr, 0.5)),
        "max": float(arr.max()),
    }


def count_view_buffers(buffer_root: Path | None) -> dict[str, int]:
    if buffer_root is None or not buffer_root.exists():
        return {
            "views": 0,
            "rendered_rm_views": 0,
            "mask_views": 0,
            "uv_views": 0,
            "strict_complete_views": 0,
        }
    view_dirs = [path for path in buffer_root.iterdir() if path.is_dir()]
    rendered_rm = 0
    mask_count = 0
    uv_count = 0
    strict_complete = 0
    for view_dir in view_dirs:
        has_image = (view_dir / "rgba.png").exists() or (view_dir / "rgb.png").exists()
        has_mask = (view_dir / "mask.png").exists()
        has_uv = (view_dir / "uv.npy").exists() or (view_dir / "uv.npz").exists()
        has_rm = (view_dir / "roughness.png").exists() and (view_dir / "metallic.png").exists()
        rendered_rm += int(has_image and has_rm)
        mask_count += int(has_mask)
        uv_count += int(has_uv)
        if has_image and has_mask and has_uv and has_rm:
            strict_complete += 1
    return {
        "views": len(view_dirs),
        "rendered_rm_views": rendered_rm,
        "mask_views": mask_count,
        "uv_views": uv_count,
        "strict_complete_views": strict_complete,
    }


def audit_record(
    manifest_path: Path,
    payload: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    resolved = {
        field: resolve_record_path(manifest_path, payload, record, record.get(field))
        for field in REQUIRED_PATH_FIELDS
    }
    missing = [
        field
        for field, path in resolved.items()
        if path is None or not path.exists()
    ]
    prior_roughness_digest = file_digest(resolved["uv_prior_roughness_path"])
    target_roughness_digest = file_digest(resolved["uv_target_roughness_path"])
    prior_metallic_digest = file_digest(resolved["uv_prior_metallic_path"])
    target_metallic_digest = file_digest(resolved["uv_target_metallic_path"])
    same_roughness = (
        prior_roughness_digest is not None
        and prior_roughness_digest == target_roughness_digest
    )
    same_metallic = (
        prior_metallic_digest is not None
        and prior_metallic_digest == target_metallic_digest
    )
    confidence_stats = gray_stats(resolved["uv_target_confidence_path"])
    view_counts = count_view_buffers(resolved["canonical_buffer_root"])
    return {
        "object_id": record.get("object_id"),
        "source_name": record.get("source_name", record.get("generator_id", "unknown")),
        "generator_id": record.get("generator_id", "unknown"),
        "license_bucket": record.get("license_bucket", "unknown"),
        "supervision_tier": record.get("supervision_tier", "unknown"),
        "prior_mode": record.get("prior_mode", "unknown"),
        "has_material_prior": bool(record.get("has_material_prior")),
        "missing_fields": missing,
        "is_complete": not missing,
        "same_roughness": same_roughness,
        "same_metallic": same_metallic,
        "same_rm_pair": same_roughness and same_metallic,
        "confidence": confidence_stats,
        "views": view_counts["views"],
        "rendered_rm_views": view_counts["rendered_rm_views"],
        "mask_views": view_counts["mask_views"],
        "uv_views": view_counts["uv_views"],
        "strict_complete_views": view_counts["strict_complete_views"],
    }


def summarize(rows: list[dict[str, Any]], *, identity_warning_threshold: float) -> dict[str, Any]:
    count = len(rows)
    source_counts = Counter(str(row["source_name"]) for row in rows)
    prior_counts = Counter("with_prior" if row["has_material_prior"] else "without_prior" for row in rows)
    missing_counts = Counter()
    for row in rows:
        missing_counts.update(row["missing_fields"])
    same_rm_pair = sum(1 for row in rows if row["same_rm_pair"])
    complete = sum(1 for row in rows if row["is_complete"])
    rendered_rm_views = sum(int(row["rendered_rm_views"]) for row in rows)
    mask_views = sum(int(row["mask_views"]) for row in rows)
    uv_views = sum(int(row["uv_views"]) for row in rows)
    strict_complete_views = sum(int(row["strict_complete_views"]) for row in rows)
    views = sum(int(row["views"]) for row in rows)
    by_source: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "same_rm_pair": 0.0, "complete": 0.0})
    for row in rows:
        bucket = by_source[str(row["source_name"])]
        bucket["count"] += 1
        bucket["same_rm_pair"] += float(row["same_rm_pair"])
        bucket["complete"] += float(row["is_complete"])
    by_source_final = {}
    for key, bucket in by_source.items():
        denom = max(bucket["count"], 1.0)
        by_source_final[key] = {
            "count": int(bucket["count"]),
            "same_rm_pair_rate": bucket["same_rm_pair"] / denom,
            "complete_rate": bucket["complete"] / denom,
        }
    identity_rate = same_rm_pair / max(count, 1)
    return {
        "records": count,
        "complete_records": complete,
        "complete_rate": complete / max(count, 1),
        "same_rm_pair": same_rm_pair,
        "same_rm_pair_rate": identity_rate,
        "identity_warning_threshold": identity_warning_threshold,
        "identity_warning": identity_rate >= identity_warning_threshold,
        "views": views,
        "rendered_rm_views": rendered_rm_views,
        "rendered_rm_view_rate": rendered_rm_views / max(views, 1),
        "mask_views": mask_views,
        "mask_view_rate": mask_views / max(views, 1),
        "uv_views": uv_views,
        "uv_view_rate": uv_views / max(views, 1),
        "strict_complete_views": strict_complete_views,
        "strict_complete_view_rate": strict_complete_views / max(views, 1),
        "source_counts": dict(source_counts),
        "prior_counts": dict(prior_counts),
        "missing_field_counts": dict(missing_counts),
        "by_source": by_source_final,
    }


def save_plot(summary: dict[str, Any], output_path: Path) -> None:
    labels = list(summary["by_source"].keys()) or ["all"]
    identity = [summary["by_source"][label]["same_rm_pair_rate"] for label in labels]
    complete = [summary["by_source"][label]["complete_rate"] for label in labels]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x - width / 2, complete, width, label="complete")
    axes[0].bar(x + width / 2, identity, width, label="target == prior")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Record Completeness And RM Identity")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    source_counts = summary["source_counts"]
    axes[1].pie(source_counts.values(), labels=source_counts.keys(), autopct="%1.0f%%")
    axes[1].set_title("Source Mix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_html(summary: dict[str, Any], plot_path: Path, output_path: Path) -> None:
    warning = (
        "<strong>Warning:</strong> target RM is identical to prior RM for most records. "
        "Baseline metrics may be trivial."
        if summary["identity_warning"]
        else "No high target/prior identity warning."
    )
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Material Manifest Audit</title>",
        "<style>body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1200px;margin:auto}.card{background:#18202b;border-radius:18px;padding:18px;margin:16px 0}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #334052;padding:8px;text-align:left}img{max-width:100%;border-radius:14px;background:white}.warn{color:#ffd28a}</style>",
        "</head><body><div class='wrap'>",
        "<h1>Material Refine Manifest Audit</h1>",
        f"<div class='card warn'>{warning}</div>",
        f"<div class='card'><img src='{html.escape(plot_path.name)}' alt='audit plot'></div>",
        "<div class='card'><h2>Summary</h2>",
        f"<p>Records: {summary['records']} | Complete: {summary['complete_records']} ({summary['complete_rate']:.2%}) | Target==Prior RM: {summary['same_rm_pair']} ({summary['same_rm_pair_rate']:.2%})</p>",
        f"<p>Rendered RM views: {summary['rendered_rm_views']} / {summary['views']} ({summary['rendered_rm_view_rate']:.2%})</p>",
        f"<p>UV buffer views: {summary['uv_views']} / {summary['views']} ({summary['uv_view_rate']:.2%}) | Strict complete views: {summary['strict_complete_views']} / {summary['views']} ({summary['strict_complete_view_rate']:.2%})</p>",
        "</div><div class='card'><h2>By Source</h2><table><thead><tr><th>Source</th><th>Count</th><th>Complete Rate</th><th>Target==Prior Rate</th></tr></thead><tbody>",
    ]
    for source, bucket in summary["by_source"].items():
        lines.append(
            f"<tr><td>{html.escape(source)}</td><td>{bucket['count']}</td><td>{bucket['complete_rate']:.2%}</td><td>{bucket['same_rm_pair_rate']:.2%}</td></tr>"
        )
    lines.extend(["</tbody></table></div>", "</div></body></html>"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload, records = load_manifest(args.manifest)
    selected = records if args.max_records < 0 else records[: args.max_records]
    rows = [audit_record(args.manifest, payload, record) for record in selected]
    summary = summarize(rows, identity_warning_threshold=args.identity_warning_threshold)
    output_payload = {
        "manifest": str(args.manifest.resolve()),
        "audited_records": len(rows),
        "summary": summary,
        "records": rows,
    }
    summary_path = args.output_dir / "manifest_audit_summary.json"
    plot_path = args.output_dir / "manifest_audit.png"
    html_path = args.output_dir / "manifest_audit.html"
    summary_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    save_plot(summary, plot_path)
    save_html(summary, plot_path, html_path)
    print(json.dumps({"summary": str(summary_path), "plot": str(plot_path), "html": str(html_path), "identity_warning": summary["identity_warning"]}, indent=2))


if __name__ == "__main__":
    main()
