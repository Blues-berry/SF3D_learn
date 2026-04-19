from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "abo_material_probe"
DEFAULT_MODEL_PATH = (
    "/home/ubuntu/ssd_work/models/sf3d_hf"
    if Path("/home/ubuntu/ssd_work/models/sf3d_hf").exists()
    else "stabilityai/stable-fast-3d"
)
DEFAULT_BLENDER_BIN = Path(
    "/4T/CXY/Neural_Gaffer_original/scripts/Objavarse_rendering/blender-3.2.2-linux-x64/blender"
)

OBJECTS = [
    {
        "id": "B073P3562J",
        "label": "lamp_metal_glass",
        "category": "lamp",
        "name": "Rivet Lux Bendable Arm Marble and Brass Table Desk Lamp",
        "path": "J/B073P3562J.glb",
    },
    {
        "id": "B07MF1SRH5",
        "label": "ceramic_stool",
        "category": "stool",
        "name": "Ravenna Home Damask Ceramic Garden Stool",
        "path": "5/B07MF1SRH5.glb",
    },
    {
        "id": "B07L8DQQ4Q",
        "label": "leather_metal_stool",
        "category": "stool",
        "name": "Phoenix Home Lotusville Vintage PU Leather Counter Stool",
        "path": "Q/B07L8DQQ4Q.glb",
    },
    {
        "id": "B07B8NW6GG",
        "label": "wood_iron_mirror",
        "category": "mirror",
        "name": "Stone & Beam Rustic Farmhouse Wood Iron Mirror",
        "path": "G/B07B8NW6GG.glb",
    },
    {
        "id": "B07QX2BYDH",
        "label": "metal_desk",
        "category": "desk",
        "name": "AmazonBasics Gaming Computer Desk",
        "path": "H/B07QX2BYDH.glb",
    },
    {
        "id": "B0764W75LL",
        "label": "faux_linen_headboard",
        "category": "headboard",
        "name": "AmazonBasics Faux Linen Tufted Headboard",
        "path": "L/B0764W75LL.glb",
    },
]

VIEWS = [
    {
        "name": "front_studio",
        "azimuth": 0.0,
        "elevation": 18.0,
        "distance": 2.0,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "studio_small_08_1k.hdr"),
    },
    {
        "name": "three_quarter_indoor",
        "azimuth": 45.0,
        "elevation": 28.0,
        "distance": 2.0,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "peppermint_powerplant_1k.hdr"),
    },
    {
        "name": "side_neon",
        "azimuth": 110.0,
        "elevation": 16.0,
        "distance": 2.05,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "neon_photostudio_1k.hdr"),
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--object-manifest", type=Path, default=None)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--pretrained-model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--cuda-device-index",
        type=str,
        default="0",
        help="Physical CUDA device index to isolate for Blender/SF3D work.",
    )
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--cycles-samples", type=int, default=32)
    parser.add_argument("--texture-resolution", type=int, default=512)
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Maximum objects to process. Defaults to all manifest rows, or all built-in samples when no manifest is provided.",
    )
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--save-meshes", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_objects(manifest_path: Path | None) -> list[dict]:
    if manifest_path is None:
        return OBJECTS
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing object manifest: {manifest_path}")
    if manifest_path.suffix.lower() == ".json":
        data = json.loads(manifest_path.read_text())
        if not isinstance(data, list):
            raise RuntimeError("Object manifest JSON must be a list")
        return data
    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise RuntimeError(f"Unsupported object manifest format: {manifest_path}")


def download_glb(obj: dict, cache_dir: Path) -> Path:
    local_keys = ["local_path", "source_model_path", "object_path"]
    for key in local_keys:
        local_path = obj.get(key)
        if local_path:
            local_path = Path(local_path)
            if local_path.exists():
                return local_path

    out_path = cache_dir / f"{obj['id']}.glb"
    if out_path.exists():
        return out_path
    url = (
        "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/3dmodels/original/"
        + obj["path"]
    )
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return out_path


def run_blender_render(
    blender_bin: Path,
    glb_path: Path,
    object_out_dir: Path,
    resolution: int,
    cycles_samples: int,
    cuda_device_index: str,
):
    views_json = object_out_dir / "views.json"
    views_json.write_text(json.dumps(VIEWS, indent=2))
    blender_script = REPO_ROOT / "scripts" / "abo_material_passes_blender.py"
    cmd = [
        str(blender_bin),
        "-b",
        "-P",
        str(blender_script),
        "--",
        "--object-path",
        str(glb_path),
        "--output-dir",
        str(object_out_dir),
        "--views-json",
        str(views_json),
        "--resolution",
        str(resolution),
        "--cycles-samples",
        str(cycles_samples),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device_index
    # Once a single device is made visible, Blender should use device index 0 internally.
    env["BLENDER_CUDA_DEVICE_INDEX"] = "0"
    subprocess.run(cmd, check=True, env=env)


def load_sf3d(pretrained_model: str, device: str):
    import torch

    from sf3d.system import SF3D

    model = SF3D.from_pretrained(
        pretrained_model,
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)
    model.eval()
    return model, device


def infer_sf3d(
    model,
    device: str,
    image_path: Path,
    mesh_out_path: Path | None,
    texture_resolution: int,
):
    import torch
    image = Image.open(image_path).convert("RGBA")
    with torch.no_grad():
        context = (
            torch.autocast(device_type=device, dtype=torch.bfloat16)
            if device.startswith("cuda")
            else nullcontext()
        )
        with context:
            mesh, _ = model.run_image(
                [image],
                bake_resolution=texture_resolution,
                remesh="none",
                vertex_count=-1,
            )
    if isinstance(mesh, list):
        mesh = mesh[0]
    material = mesh.visual.material
    roughness = float(material.roughnessFactor)
    metallic = float(material.metallicFactor)
    if mesh_out_path is not None:
        mesh.export(mesh_out_path, include_normals=True)
    return {"pred_roughness": roughness, "pred_metallic": metallic}


def image_stats(path: Path):
    arr = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    alpha = arr[..., 3] > 0.01
    values = arr[..., 0]
    if not alpha.any():
        raise RuntimeError(f"No visible pixels in {path}")
    visible = values[alpha]
    interior = np.asarray(
        Image.fromarray((alpha.astype(np.uint8) * 255)).filter(ImageFilter.MinFilter(9))
    )
    interior_mask = interior > 0
    edge_mask = alpha & ~interior_mask
    stats = {
        "mean": float(visible.mean()),
        "std": float(visible.std()),
        "p10": float(np.quantile(visible, 0.10)),
        "p50": float(np.quantile(visible, 0.50)),
        "p90": float(np.quantile(visible, 0.90)),
        "coverage": int(alpha.sum()),
        "edge_mean": float(values[edge_mask].mean()) if edge_mask.any() else float(visible.mean()),
        "interior_mean": float(values[interior_mask].mean())
        if interior_mask.any()
        else float(visible.mean()),
    }
    return stats


def input_highlight_stats(path: Path):
    arr = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    alpha = arr[..., 3] > 0.01
    rgb = arr[..., :3]
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    visible = luminance[alpha]
    return {
        "brightness_p95": float(np.quantile(visible, 0.95)),
        "brightness_p99": float(np.quantile(visible, 0.99)),
        "highlight_fraction": float(((luminance > 0.92) & alpha).sum() / alpha.sum()),
    }


def categorize_case(row: dict):
    tags = []

    rough_range = row["gt_roughness_p90"] - row["gt_roughness_p10"]
    if rough_range > 0.35 and abs(row["pred_roughness"] - row["gt_roughness_mean"]) < 0.10:
        tags.append("over_smoothing")

    if row["gt_metallic_mean"] < 0.15 and row["pred_metallic"] > 0.35:
        tags.append("metal_nonmetal_confusion")
    elif row["gt_metallic_mean"] > 0.20 and row["pred_metallic"] < 0.08:
        tags.append("metal_nonmetal_confusion")

    if row["brightness_p99"] > 0.82 and row["gt_metallic_mean"] < 0.30:
        rough_error = abs(row["pred_roughness"] - row["gt_roughness_mean"])
        metal_error = abs(row["pred_metallic"] - row["gt_metallic_mean"])
        if rough_error > 0.12 or metal_error > 0.12:
            tags.append("highlight_misread")

    metallic_edge_gap = abs(row["gt_metallic_edge_mean"] - row["gt_metallic_interior_mean"])
    roughness_edge_gap = abs(row["gt_roughness_edge_mean"] - row["gt_roughness_interior_mean"])
    if metallic_edge_gap > 0.20:
        edge_dist = abs(row["pred_metallic"] - row["gt_metallic_edge_mean"])
        interior_dist = abs(row["pred_metallic"] - row["gt_metallic_interior_mean"])
        if edge_dist + 0.05 < interior_dist:
            tags.append("boundary_bleed")
    if roughness_edge_gap > 0.20:
        edge_dist = abs(row["pred_roughness"] - row["gt_roughness_edge_mean"])
        interior_dist = abs(row["pred_roughness"] - row["gt_roughness_interior_mean"])
        if edge_dist + 0.05 < interior_dist and "boundary_bleed" not in tags:
            tags.append("boundary_bleed")
    return tags


def build_contact_sheet(rows: list[dict], output_path: Path):
    if not rows:
        return
    tiles = []
    for row in rows:
        rgb = Image.open(row["rgba_path"]).convert("RGBA").resize((192, 192))
        rough = Image.open(row["roughness_path"]).convert("RGBA").resize((192, 192))
        metal = Image.open(row["metallic_path"]).convert("RGBA").resize((192, 192))
        panel = Image.new("RGBA", (640, 236), (18, 18, 18, 255))
        panel.paste(rgb, (12, 12))
        panel.paste(rough, (220, 12))
        panel.paste(metal, (428, 12))
        draw = ImageDraw.Draw(panel)
        pred_r = row["pred_roughness"]
        pred_m = row["pred_metallic"]
        lines = [
            f"{row['object_label']} / {row['view_name']}",
            "pred R/M: "
            + (
                f"{pred_r:.3f} / {pred_m:.3f}"
                if pred_r is not None and pred_m is not None
                else "n/a"
            ),
            f"gt mean R/M: {row['gt_roughness_mean']:.3f} / {row['gt_metallic_mean']:.3f}",
            f"gt std R/M: {row['gt_roughness_std']:.3f} / {row['gt_metallic_std']:.3f}",
            "tags: " + (", ".join(row["tags"]) if row["tags"] else "none"),
        ]
        y = 208
        for line in lines:
            draw.text((12, y), line, fill=(235, 235, 235, 255))
            y += 14
        tiles.append(panel)

    sheet = Image.new("RGBA", (640, 236 * len(tiles)), (0, 0, 0, 255))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, (0, idx * 236))
    sheet.save(output_path)


def write_summary(rows: list[dict], output_path: Path):
    inferred_rows = [row for row in rows if row["pred_roughness"] is not None]
    grouped = {
        "over_smoothing": [],
        "metal_nonmetal_confusion": [],
        "highlight_misread": [],
        "boundary_bleed": [],
    }
    for row in inferred_rows:
        for tag in row["tags"]:
            grouped[tag].append(row)

    lines = [
        "# ABO mini RM baseline probe",
        "",
        f"objects: {len(set(row['object_id'] for row in rows))}",
        f"views: {len(rows)}",
        f"inferred_views: {len(inferred_rows)}",
        "",
    ]
    for tag, cases in grouped.items():
        lines.append(f"## {tag}")
        if not cases:
            lines.append("")
            lines.append("No strong cases triggered by the current heuristic pass.")
            lines.append("")
            continue
        for case in sorted(
            cases,
            key=lambda row: abs(row["pred_roughness"] - row["gt_roughness_mean"])
            + abs(row["pred_metallic"] - row["gt_metallic_mean"]),
            reverse=True,
        )[:6]:
            lines.append(
                "- "
                + f"{case['object_label']} / {case['view_name']}: "
                + f"pred R/M {case['pred_roughness']:.3f}/{case['pred_metallic']:.3f}, "
                + f"gt mean R/M {case['gt_roughness_mean']:.3f}/{case['gt_metallic_mean']:.3f}, "
                + f"gt std R/M {case['gt_roughness_std']:.3f}/{case['gt_metallic_std']:.3f}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines))


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir or (output_dir / "cache"))
    render_dir = ensure_dir(output_dir / "renders")
    sf3d_dir = ensure_dir(output_dir / "sf3d")
    rows = []

    objects = load_objects(args.object_manifest)
    if args.max_objects is None:
        selected_objects = list(objects)
    else:
        selected_objects = list(objects[: args.max_objects])
    for obj in selected_objects:
        object_dir = ensure_dir(render_dir / obj["id"])
        glb_path = download_glb(obj, cache_dir)
        if not args.skip_render:
            run_blender_render(
                args.blender_bin,
                glb_path,
                object_dir,
                resolution=args.render_resolution,
                cycles_samples=args.cycles_samples,
                cuda_device_index=args.cuda_device_index,
            )

    model = None
    device = args.device
    if not args.skip_inference:
        if device.startswith("cuda"):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device_index
        model, device = load_sf3d(args.pretrained_model, args.device)

    for obj in selected_objects:
        for view in VIEWS:
            view_dir = render_dir / obj["id"] / view["name"]
            rgba_path = view_dir / "rgba.png"
            roughness_path = view_dir / "roughness.png"
            metallic_path = view_dir / "metallic.png"

            gt_rough = image_stats(roughness_path)
            gt_metal = image_stats(metallic_path)
            highlight = input_highlight_stats(rgba_path)

            row = {
                "object_id": obj["id"],
                "object_label": obj["label"],
                "object_name": obj["name"],
                "category": obj.get("category", "other"),
                "view_name": view["name"],
                "rgba_path": str(rgba_path.resolve()),
                "roughness_path": str(roughness_path.resolve()),
                "metallic_path": str(metallic_path.resolve()),
                "gt_roughness_mean": gt_rough["mean"],
                "gt_roughness_std": gt_rough["std"],
                "gt_roughness_p10": gt_rough["p10"],
                "gt_roughness_p50": gt_rough["p50"],
                "gt_roughness_p90": gt_rough["p90"],
                "gt_roughness_edge_mean": gt_rough["edge_mean"],
                "gt_roughness_interior_mean": gt_rough["interior_mean"],
                "gt_metallic_mean": gt_metal["mean"],
                "gt_metallic_std": gt_metal["std"],
                "gt_metallic_p10": gt_metal["p10"],
                "gt_metallic_p50": gt_metal["p50"],
                "gt_metallic_p90": gt_metal["p90"],
                "gt_metallic_edge_mean": gt_metal["edge_mean"],
                "gt_metallic_interior_mean": gt_metal["interior_mean"],
                "brightness_p95": highlight["brightness_p95"],
                "brightness_p99": highlight["brightness_p99"],
                "highlight_fraction": highlight["highlight_fraction"],
                "pred_roughness": None,
                "pred_metallic": None,
            }

            if model is not None:
                mesh_out = (
                    sf3d_dir / obj["id"] / f"{view['name']}.glb"
                    if args.save_meshes
                    else None
                )
                if mesh_out is not None:
                    ensure_dir(mesh_out.parent)
                pred = infer_sf3d(
                    model,
                    device=device,
                    image_path=rgba_path,
                    mesh_out_path=mesh_out,
                    texture_resolution=args.texture_resolution,
                )
                row.update(pred)

            row["tags"] = categorize_case(row) if row["pred_roughness"] is not None else []
            rows.append(row)

    rows_path = output_dir / "metrics.json"
    rows_path.write_text(json.dumps(rows, indent=2))
    build_contact_sheet(rows, output_dir / "contact_sheet.png")
    write_summary(rows, output_dir / "failure_summary.md")

    viz_outputs = None
    if not args.skip_visualize:
        from abo_material_visualize import generate_visualizations

        viz_outputs = generate_visualizations(rows_path, output_dir)

    print(f"Wrote {len(rows)} rows to {rows_path}")
    print(f"Summary: {output_dir / 'failure_summary.md'}")
    print(f"Contact sheet: {output_dir / 'contact_sheet.png'}")
    if viz_outputs is not None:
        print(f"Dashboard: {viz_outputs['dashboard']}")
        print(f"HTML report: {viz_outputs['html']}")


if __name__ == "__main__":
    main()
