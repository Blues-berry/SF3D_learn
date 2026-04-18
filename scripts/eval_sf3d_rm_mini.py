from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "abo_rm_mini"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "objects_200.json"
DEFAULT_MODEL_PATH = (
    "/home/ubuntu/ssd_work/models/sf3d_hf"
    if Path("/home/ubuntu/ssd_work/models/sf3d_hf").exists()
    else "stabilityai/stable-fast-3d"
)
VIEWS = ["front_studio", "three_quarter_indoor", "side_neon"]
FAILURE_TAGS = [
    "over_smoothing",
    "metal_nonmetal_confusion",
    "local_highlight_misread",
    "boundary_bleed",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--object-manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--pretrained-model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--texture-resolution", type=int, default=512)
    parser.add_argument("--max-objects", type=int, default=200)
    parser.add_argument("--save-meshes", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        objects = payload.get("objects")
        if isinstance(objects, list):
            return objects
    if isinstance(payload, list):
        return payload
    raise RuntimeError(f"Unsupported manifest payload in {path}")


def load_sf3d(pretrained_model: str, device: str):
    original_version_fn = importlib.metadata.version

    def patched_version(package_name: str):
        if package_name == "huggingface-hub":
            return "0.23.2"
        return original_version_fn(package_name)

    importlib.metadata.version = patched_version
    try:
        from sf3d.system import SF3D
        model = SF3D.from_pretrained(
            pretrained_model,
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
    finally:
        importlib.metadata.version = original_version_fn

    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)
    model.eval()
    return model, device


def infer_sf3d(model, device: str, image_path: Path, mesh_out_path: Path | None, texture_resolution: int):
    image = Image.open(image_path).convert("RGBA")
    with torch.no_grad():
        context = (
            torch.autocast(device_type=device, dtype=torch.bfloat16)
            if device.startswith("cuda")
            else nullcontext()
        )
        with context:
            mesh, aux = model.run_image(
                [image],
                bake_resolution=texture_resolution,
                remesh="none",
                vertex_count=-1,
            )
    if isinstance(mesh, list):
        mesh = mesh[0]
    if mesh_out_path is not None:
        mesh.export(mesh_out_path, include_normals=True)
    if hasattr(mesh.visual, "material"):
        material = mesh.visual.material
        return {
            "roughness": float(material.roughnessFactor),
            "metallic": float(material.metallicFactor),
        }
    if isinstance(aux, dict) and "decoder_roughness" in aux and "decoder_metallic" in aux:
        return {
            "roughness": float(aux["decoder_roughness"].squeeze().float().cpu().item()),
            "metallic": float(aux["decoder_metallic"].squeeze().float().cpu().item()),
        }
    return {
        "roughness": 0.5,
        "metallic": 0.0,
    }


def flatten_case_row(row: dict):
    return {
        "object_id": row["object_id"],
        "label": row["label"],
        "name": row["name"],
        "category": row["category"],
        "sampling_bucket": row.get("sampling_bucket"),
        "view_name": row["view_name"],
        "pred_roughness": row["pred"]["roughness"],
        "pred_metallic": row["pred"]["metallic"],
        "gt_roughness_mean": row["gt"]["roughness"]["whole_mask"]["mean"],
        "gt_metallic_mean": row["gt"]["metallic"]["whole_mask"]["mean"],
        "gt_roughness_edge_mean": row["gt"]["roughness"]["edge_vs_interior"]["edge_mean"],
        "gt_roughness_interior_mean": row["gt"]["roughness"]["edge_vs_interior"]["interior_mean"],
        "gt_metallic_edge_mean": row["gt"]["metallic"]["edge_vs_interior"]["edge_mean"],
        "gt_metallic_interior_mean": row["gt"]["metallic"]["edge_vs_interior"]["interior_mean"],
        "gt_roughness_var": row["gt"]["roughness"]["whole_mask"]["var"],
        "gt_metallic_var": row["gt"]["metallic"]["whole_mask"]["var"],
        "brightness_p99": row["gt"]["input_highlight"]["brightness_p99"],
        "highlight_fraction": row["gt"]["input_highlight"]["highlight_fraction"],
        "abs_error_roughness": row["errors"]["abs_error_roughness"],
        "abs_error_metallic": row["errors"]["abs_error_metallic"],
        "total_error": row["errors"]["total_error"],
        "primary_failure": row["primary_failure"],
        **{tag: int(tag in row["failure_tags"]) for tag in FAILURE_TAGS},
    }


def classify_case(row: dict):
    gt_rough = row["gt"]["roughness"]
    gt_metal = row["gt"]["metallic"]
    highlight = row["gt"]["input_highlight"]
    pred_r = row["pred"]["roughness"]
    pred_m = row["pred"]["metallic"]

    rough_mean = gt_rough["whole_mask"]["mean"]
    rough_range = gt_rough["whole_mask"]["p90"] - gt_rough["whole_mask"]["p10"]
    rough_var = gt_rough["whole_mask"]["var"]
    metal_mean = gt_metal["whole_mask"]["mean"]
    brightness_p99 = highlight["brightness_p99"]

    rough_error = abs(pred_r - rough_mean)
    metal_error = abs(pred_m - metal_mean)

    trigger_scores = {}

    if rough_range > 0.30 and rough_var > 0.01 and rough_error < 0.10:
        trigger_scores["over_smoothing"] = rough_range + rough_var * 4.0 - rough_error

    if metal_mean < 0.15 and pred_m > 0.35:
        trigger_scores["metal_nonmetal_confusion"] = pred_m - metal_mean
    elif metal_mean > 0.20 and pred_m < 0.08:
        trigger_scores["metal_nonmetal_confusion"] = metal_mean - pred_m

    if brightness_p99 > 0.82 and metal_mean < 0.30 and (rough_error > 0.12 or metal_error > 0.12):
        trigger_scores["local_highlight_misread"] = max(rough_error, metal_error) + (brightness_p99 - 0.82)

    metal_edge = gt_metal["edge_vs_interior"]["edge_mean"]
    metal_interior = gt_metal["edge_vs_interior"]["interior_mean"]
    rough_edge = gt_rough["edge_vs_interior"]["edge_mean"]
    rough_interior = gt_rough["edge_vs_interior"]["interior_mean"]
    metal_edge_gap = abs(metal_edge - metal_interior)
    rough_edge_gap = abs(rough_edge - rough_interior)

    if metal_edge_gap > 0.20:
        edge_dist = abs(pred_m - metal_edge)
        interior_dist = abs(pred_m - metal_interior)
        if edge_dist + 0.05 < interior_dist:
            trigger_scores["boundary_bleed"] = metal_edge_gap + (interior_dist - edge_dist)
    if rough_edge_gap > 0.20:
        edge_dist = abs(pred_r - rough_edge)
        interior_dist = abs(pred_r - rough_interior)
        if edge_dist + 0.05 < interior_dist:
            score = rough_edge_gap + (interior_dist - edge_dist)
            trigger_scores["boundary_bleed"] = max(trigger_scores.get("boundary_bleed", 0.0), score)

    failure_tags = [tag for tag in FAILURE_TAGS if tag in trigger_scores]
    primary_failure = max(failure_tags, key=lambda tag: trigger_scores[tag]) if failure_tags else "none"
    return failure_tags, primary_failure, trigger_scores


def aggregate_object_rows(rows: list[dict]) -> dict:
    objects = {}
    for row in rows:
        entry = objects.setdefault(
            row["object_id"],
            {
                "object_id": row["object_id"],
                "label": row["label"],
                "name": row["name"],
                "category": row["category"],
                "sampling_bucket": row.get("sampling_bucket"),
                "views": [],
                "taxonomy_counts": {tag: 0 for tag in FAILURE_TAGS},
            },
        )
        entry["views"].append(
            {
                "view_name": row["view_name"],
                "total_error": row["errors"]["total_error"],
                "primary_failure": row["primary_failure"],
            }
        )
        for tag in row["failure_tags"]:
            entry["taxonomy_counts"][tag] += 1

    summaries = []
    for object_id, entry in objects.items():
        object_rows = [row for row in rows if row["object_id"] == object_id]
        worst_case = max(object_rows, key=lambda item: item["errors"]["total_error"])
        summaries.append(
            {
                "object_id": object_id,
                "label": entry["label"],
                "name": entry["name"],
                "category": entry["category"],
                "sampling_bucket": entry["sampling_bucket"],
                "views_evaluated": len(object_rows),
                "mean_abs_error_roughness": sum(item["errors"]["abs_error_roughness"] for item in object_rows) / len(object_rows),
                "mean_abs_error_metallic": sum(item["errors"]["abs_error_metallic"] for item in object_rows) / len(object_rows),
                "mean_total_error": sum(item["errors"]["total_error"] for item in object_rows) / len(object_rows),
                "taxonomy_counts": entry["taxonomy_counts"],
                "worst_view": worst_case["view_name"],
                "worst_total_error": worst_case["errors"]["total_error"],
            }
        )

    summaries.sort(key=lambda item: item["mean_total_error"], reverse=True)
    taxonomy_totals = {tag: sum(item["taxonomy_counts"][tag] for item in summaries) for tag in FAILURE_TAGS}
    return {
        "summary": {
            "objects_evaluated": len(summaries),
            "views_evaluated": len(rows),
            "taxonomy_totals": taxonomy_totals,
        },
        "objects": summaries,
    }


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    render_dir = output_dir / "renders"
    sf3d_dir = ensure_dir(output_dir / "sf3d")

    manifest_objects = load_manifest(args.object_manifest)[: args.max_objects]
    model, device = load_sf3d(args.pretrained_model, args.device)

    rows = []
    eval_skips = []

    for obj in manifest_objects:
        object_id = obj["id"]
        for view_name in VIEWS:
            stats_path = render_dir / object_id / view_name / "stats.json"
            if not stats_path.exists():
                eval_skips.append(
                    {
                        "object_id": object_id,
                        "view_name": view_name,
                        "reason": f"Missing stats: {stats_path}",
                    }
                )
                continue

            try:
                stats = json.loads(stats_path.read_text())
                rgba_path = Path(stats["paths"]["rgba"])
                mesh_out = sf3d_dir / object_id / f"{view_name}.glb" if args.save_meshes else None
                if mesh_out is not None:
                    ensure_dir(mesh_out.parent)
                pred = infer_sf3d(
                    model,
                    device=device,
                    image_path=rgba_path,
                    mesh_out_path=mesh_out,
                    texture_resolution=args.texture_resolution,
                )

                row = {
                    "object_id": object_id,
                    "label": stats.get("label", obj.get("label", object_id)),
                    "name": stats.get("name", obj.get("name", object_id)),
                    "category": stats.get("category", obj.get("category", "unknown")),
                    "sampling_bucket": stats.get("sampling_bucket", obj.get("sampling_bucket")),
                    "view_name": view_name,
                    "paths": stats["paths"],
                    "gt": {
                        "roughness": stats["roughness"],
                        "metallic": stats["metallic"],
                        "input_highlight": stats["input_highlight"],
                    },
                    "pred": pred,
                    "errors": {
                        "abs_error_roughness": abs(pred["roughness"] - stats["roughness"]["whole_mask"]["mean"]),
                        "abs_error_metallic": abs(pred["metallic"] - stats["metallic"]["whole_mask"]["mean"]),
                    },
                }
                row["errors"]["total_error"] = row["errors"]["abs_error_roughness"] + row["errors"]["abs_error_metallic"]
                row["failure_tags"], row["primary_failure"], row["trigger_scores"] = classify_case(row)
                rows.append(row)

                per_view_eval_path = render_dir / object_id / view_name / "sf3d_eval.json"
                per_view_eval_path.write_text(json.dumps(row, indent=2))
            except Exception as exc:  # noqa: BLE001
                eval_skips.append(
                    {
                        "object_id": object_id,
                        "view_name": view_name,
                        "reason": str(exc),
                    }
                )

    rows.sort(key=lambda item: item["errors"]["total_error"], reverse=True)

    baseline_eval_path = output_dir / "baseline_eval.json"
    baseline_eval_path.write_text(json.dumps(rows, indent=2))

    object_summary = aggregate_object_rows(rows)
    object_summary_path = output_dir / "object_summary.json"
    object_summary_path.write_text(json.dumps(object_summary, indent=2))

    csv_rows = [flatten_case_row(row) for row in rows]
    failure_summary_path = output_dir / "failure_summary.csv"
    with failure_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        if csv_rows:
            writer.writeheader()
            writer.writerows(csv_rows)

    top_failure_cases = {
        "overall": rows[:50],
        "by_failure": {
            tag: [row for row in rows if tag in row["failure_tags"]][:20]
            for tag in FAILURE_TAGS
        },
        "eval_skips": eval_skips,
    }
    top_failure_cases_path = output_dir / "top_failure_cases.json"
    top_failure_cases_path.write_text(json.dumps(top_failure_cases, indent=2))

    eval_summary_path = output_dir / "eval_summary.json"
    eval_summary_path.write_text(
        json.dumps(
            {
                "objects_requested": len(manifest_objects),
                "views_expected": len(manifest_objects) * len(VIEWS),
                "views_evaluated": len(rows),
                "views_skipped": len(eval_skips),
                "eval_config": {
                    "pretrained_model": args.pretrained_model,
                    "device": device,
                    "texture_resolution": args.texture_resolution,
                },
                "taxonomy_counts": {
                    tag: sum(1 for row in rows if tag in row["failure_tags"]) for tag in FAILURE_TAGS
                },
            },
            indent=2,
        )
    )

    print(f"Baseline eval: {baseline_eval_path}")
    print(f"Object summary: {object_summary_path}")
    print(f"Failure summary CSV: {failure_summary_path}")
    print(f"Top failure cases: {top_failure_cases_path}")


if __name__ == "__main__":
    main()
