from __future__ import annotations

import argparse
import html
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


BACKGROUND = (11, 14, 20)
PANEL_BG = (20, 24, 32)
CARD_BG = (28, 33, 44)
TEXT_MAIN = (238, 241, 247)
TEXT_MUTED = (164, 173, 190)
TEXT_FAINT = (120, 129, 147)
TRACK = (66, 75, 91)
GRID = (56, 64, 78)
PRED_COLOR = (255, 144, 64)
MEAN_COLOR = (240, 240, 240)
EDGE_COLOR = (110, 179, 255)
INTERIOR_COLOR = (120, 222, 164)
TAG_COLORS = {
    "over_smoothing": (87, 184, 255),
    "metal_nonmetal_confusion": (255, 116, 116),
    "highlight_misread": (255, 197, 87),
    "boundary_bleed": (183, 128, 255),
}
ROUGHNESS_PALETTE = [
    (22, 40, 84),
    (66, 110, 180),
    (138, 188, 214),
    (252, 217, 120),
    (214, 74, 52),
]
METALLIC_PALETTE = [
    (12, 16, 22),
    (58, 70, 92),
    (126, 149, 176),
    (198, 166, 82),
    (255, 219, 118),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


FONT_L = load_font(28)
FONT_M = load_font(20)
FONT_S = load_font(16)
FONT_XS = load_font(14)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str):
    chars = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def lerp_color(palette, values):
    values = np.clip(values, 0.0, 1.0)
    xs = np.linspace(0.0, 1.0, len(palette))
    channels = []
    for idx in range(3):
        channel = np.interp(values, xs, [color[idx] for color in palette])
        channels.append(channel)
    return np.stack(channels, axis=-1).astype(np.uint8)


def colorize_scalar_image(path: str, palette, tile_size=(256, 256)):
    arr = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    values = arr[..., 0]
    alpha = arr[..., 3] > 0.01
    rgb = lerp_color(palette, values)
    bg = np.zeros_like(rgb)
    bg[..., 0] = BACKGROUND[0]
    bg[..., 1] = BACKGROUND[1]
    bg[..., 2] = BACKGROUND[2]
    rgb = np.where(alpha[..., None], rgb, bg)
    image = Image.fromarray(rgb, mode="RGB")
    return ImageOps.contain(image, tile_size)


def image_tile(image: Image.Image, label: str):
    tile = Image.new("RGB", (280, 322), PANEL_BG)
    tile.paste(image, ((280 - image.width) // 2, 18))
    draw = ImageDraw.Draw(tile)
    draw.text((18, 286), label, font=FONT_S, fill=TEXT_MAIN)
    return tile


def draw_scale(draw: ImageDraw.ImageDraw, x0, y0, x1, y1):
    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=TRACK)
    for tick in range(6):
        x = x0 + (x1 - x0) * tick / 5.0
        draw.line((x, y1 + 5, x, y1 + 12), fill=GRID, width=2)
        label = f"{tick / 5:.1f}"
        draw.text((x - 11, y1 + 14), label, font=FONT_XS, fill=TEXT_FAINT)


def value_to_x(value: float, x0: int, x1: int):
    return int(round(x0 + np.clip(value, 0.0, 1.0) * (x1 - x0)))


def draw_marker(draw, x, y0, y1, color, width=4):
    draw.line((x, y0, x, y1), fill=color, width=width)
    draw.ellipse((x - 5, y0 - 5, x + 5, y0 + 5), fill=color)


def metric_card(
    title: str,
    pred: float,
    gt_mean: float,
    gt_std: float,
    gt_p10: float,
    gt_p90: float,
    edge: float,
    interior: float,
):
    card = Image.new("RGB", (336, 322), CARD_BG)
    draw = ImageDraw.Draw(card)
    draw.text((18, 16), title, font=FONT_M, fill=TEXT_MAIN)
    draw.text(
        (18, 48),
        f"pred {pred:.3f}    gt {gt_mean:.3f}    abs err {abs(pred - gt_mean):.3f}",
        font=FONT_S,
        fill=TEXT_MUTED,
    )

    x0, x1 = 24, 312
    band_y0, band_y1 = 96, 114
    draw_scale(draw, x0, band_y0, x1, band_y1)
    p10_x = value_to_x(gt_p10, x0, x1)
    p90_x = value_to_x(gt_p90, x0, x1)
    draw.rounded_rectangle((p10_x, band_y0, p90_x, band_y1), radius=8, fill=(95, 117, 146))
    draw_marker(draw, value_to_x(gt_mean, x0, x1), band_y0, band_y1, MEAN_COLOR, width=3)
    draw_marker(draw, value_to_x(pred, x0, x1), band_y0 - 8, band_y1 + 8, PRED_COLOR, width=5)
    draw.text((18, 138), "GT p10-p90 band, white=GT mean, orange=prediction", font=FONT_XS, fill=TEXT_FAINT)

    edge_y0, edge_y1 = 196, 214
    draw_scale(draw, x0, edge_y0, x1, edge_y1)
    draw_marker(draw, value_to_x(edge, x0, x1), edge_y0, edge_y1, EDGE_COLOR, width=4)
    draw_marker(draw, value_to_x(interior, x0, x1), edge_y0, edge_y1, INTERIOR_COLOR, width=4)
    draw.text((18, 238), f"edge {edge:.3f}    interior {interior:.3f}    gt std {gt_std:.3f}", font=FONT_S, fill=TEXT_MUTED)
    draw.text((18, 266), "blue=edge visible pixels, green=interior visible pixels", font=FONT_XS, fill=TEXT_FAINT)
    return card


def draw_badges(draw, tags, start_x, y):
    x = start_x
    for tag in tags:
        label = tag.replace("_", " ")
        bbox = draw.textbbox((0, 0), label, font=FONT_XS)
        width = bbox[2] - bbox[0] + 18
        color = TAG_COLORS.get(tag, (120, 120, 140))
        draw.rounded_rectangle((x, y, x + width, y + 24), radius=12, fill=color)
        draw.text((x + 9, y + 4), label, font=FONT_XS, fill=(12, 12, 18))
        x += width + 8


def build_case_panel(row: dict, output_path: Path):
    rgba = Image.open(row["rgba_path"]).convert("RGBA")
    rgba_bg = Image.new("RGBA", rgba.size, BACKGROUND + (255,))
    rgba = Image.alpha_composite(rgba_bg, rgba).convert("RGB")
    rgba = ImageOps.contain(rgba, (256, 256))
    rough = colorize_scalar_image(row["roughness_path"], ROUGHNESS_PALETTE)
    metal = colorize_scalar_image(row["metallic_path"], METALLIC_PALETTE)

    panel = Image.new("RGB", (1616, 398), BACKGROUND)
    draw = ImageDraw.Draw(panel)

    draw.text((24, 18), f"{row['object_label']} / {row['view_name']}", font=FONT_L, fill=TEXT_MAIN)
    draw.text((24, 54), row["object_name"], font=FONT_S, fill=TEXT_MUTED)

    if row["tags"]:
        draw_badges(draw, row["tags"], 24, 82)
    else:
        draw.rounded_rectangle((24, 82, 116, 106), radius=12, fill=(70, 78, 92))
        draw.text((36, 86), "no tags", font=FONT_XS, fill=TEXT_MAIN)

    tiles = [
        image_tile(rgba, "Input RGBA"),
        image_tile(rough, "GT Roughness"),
        image_tile(metal, "GT Metallic"),
        metric_card(
            "Roughness",
            row["pred_roughness"],
            row["gt_roughness_mean"],
            row["gt_roughness_std"],
            row["gt_roughness_p10"],
            row["gt_roughness_p90"],
            row["gt_roughness_edge_mean"],
            row["gt_roughness_interior_mean"],
        ),
        metric_card(
            "Metallic",
            row["pred_metallic"],
            row["gt_metallic_mean"],
            row["gt_metallic_std"],
            row["gt_metallic_p10"],
            row["gt_metallic_p90"],
            row["gt_metallic_edge_mean"],
            row["gt_metallic_interior_mean"],
        ),
    ]
    x = 24
    for tile in tiles:
        panel.paste(tile, (x, 122))
        x += tile.width + 16

    footer = (
        f"brightness p99 {row['brightness_p99']:.3f}   "
        f"highlight frac {row['highlight_fraction']:.4f}   "
        f"total err {row['total_error']:.3f}"
    )
    draw.text((24, 366), footer, font=FONT_S, fill=TEXT_MUTED)
    panel.save(output_path)


def bar(draw, x0, y0, width, value, color, label, text):
    draw.text((x0, y0), label, font=FONT_S, fill=TEXT_MUTED)
    y = y0 + 22
    draw.rounded_rectangle((x0, y, x0 + width, y + 18), radius=9, fill=TRACK)
    draw.rounded_rectangle((x0, y, x0 + int(width * max(0.0, min(1.0, value))), y + 18), radius=9, fill=color)
    draw.text((x0 + width + 10, y - 1), text, font=FONT_S, fill=TEXT_MAIN)


def build_dashboard(rows: list[dict], case_paths: dict[str, Path], output_path: Path):
    by_object = defaultdict(list)
    for row in rows:
        by_object[row["object_label"]].append(row)

    width = 1660
    object_section_h = 90 + 74 * len(by_object)
    top_rows = min(6, len(rows))
    thumb_h = 224
    height = 180 + object_section_h + 56 + top_rows * (thumb_h + 20) + 30
    dashboard = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(dashboard)

    draw.text((28, 20), "ABO RM Visual Dashboard", font=FONT_L, fill=TEXT_MAIN)
    draw.text(
        (28, 58),
        f"{len(set(r['object_id'] for r in rows))} objects    {len(rows)} views    "
        f"mean total err {np.mean([r['total_error'] for r in rows]):.3f}",
        font=FONT_S,
        fill=TEXT_MUTED,
    )

    tag_counts = Counter(tag for row in rows for tag in row["tags"])
    x = 28
    for tag in ["over_smoothing", "metal_nonmetal_confusion", "highlight_misread", "boundary_bleed"]:
        color = TAG_COLORS[tag]
        box = (x, 96, x + 286, 146)
        draw.rounded_rectangle(box, radius=16, fill=color)
        draw.text((x + 16, 108), tag.replace("_", " "), font=FONT_S, fill=(12, 12, 18))
        draw.text((x + 16, 126), f"{tag_counts.get(tag, 0)} cases", font=FONT_S, fill=(20, 20, 30))
        x += 304

    section_y = 176
    draw.text((28, section_y), "Per-object Mean Absolute Error", font=FONT_M, fill=TEXT_MAIN)
    y = section_y + 42
    for object_label, items in sorted(by_object.items()):
        rough_mae = float(np.mean([abs(r["pred_roughness"] - r["gt_roughness_mean"]) for r in items]))
        metal_mae = float(np.mean([abs(r["pred_metallic"] - r["gt_metallic_mean"]) for r in items]))
        draw.text((28, y), object_label, font=FONT_S, fill=TEXT_MAIN)
        bar(draw, 300, y - 4, 240, rough_mae, (102, 180, 255), "roughness", f"{rough_mae:.3f}")
        bar(draw, 760, y - 4, 240, metal_mae, (255, 190, 92), "metallic", f"{metal_mae:.3f}")
        y += 74

    top_y = y + 24
    draw.text((28, top_y), "Top Error Cases", font=FONT_M, fill=TEXT_MAIN)
    y = top_y + 40
    for row in sorted(rows, key=lambda item: item["total_error"], reverse=True)[:top_rows]:
        case_key = row["case_key"]
        panel = Image.open(case_paths[case_key]).convert("RGB")
        panel = ImageOps.contain(panel, (1600, thumb_h))
        dashboard.paste(panel, (28, y))
        y += thumb_h + 20

    dashboard.save(output_path)


def build_tag_strip(rows: list[dict], tag: str, case_paths: dict[str, Path], output_path: Path):
    tagged = [row for row in rows if tag in row["tags"]]
    if not tagged:
        image = Image.new("RGB", (1200, 180), BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.text((28, 28), tag.replace("_", " "), font=FONT_L, fill=TEXT_MAIN)
        draw.text((28, 82), "No matched cases in this run.", font=FONT_M, fill=TEXT_MUTED)
        image.save(output_path)
        return

    tagged = sorted(tagged, key=lambda row: row["total_error"], reverse=True)[:6]
    thumbs = [ImageOps.contain(Image.open(case_paths[row["case_key"]]).convert("RGB"), (1160, 240)) for row in tagged]
    canvas = Image.new("RGB", (1200, 70 + len(thumbs) * 252), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), tag.replace("_", " "), font=FONT_L, fill=TEXT_MAIN)
    y = 70
    for thumb in thumbs:
        canvas.paste(thumb, (20, y))
        y += 252
    canvas.save(output_path)


def relpath(path: Path, start: Path):
    return os.path.relpath(path, start)


def build_html_report(rows: list[dict], output_dir: Path, dashboard_path: Path, case_paths: dict[str, Path], tag_paths: dict[str, Path]):
    html_path = output_dir / "viz" / "report.html"
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>ABO RM Visual Report</title>",
        "<style>",
        "body { background:#0b0e14; color:#eef1f7; font-family: 'DejaVu Sans', Arial, sans-serif; margin:0; }",
        ".wrap { max-width: 1680px; margin: 0 auto; padding: 24px; }",
        ".hero { margin-bottom: 28px; }",
        ".hero img, .tagstrip img, .case img { width: 100%; border-radius: 18px; display:block; }",
        ".stats { display:grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 20px 0 28px; }",
        ".stat { background:#141820; border-radius:18px; padding:16px 18px; }",
        ".stat .k { color:#a4adbe; font-size:14px; }",
        ".stat .v { font-size:28px; margin-top:8px; }",
        ".section { margin-top: 28px; }",
        ".section h2 { margin: 0 0 14px; font-size: 24px; }",
        ".tagstrip { margin: 16px 0 28px; }",
        ".casegrid { display:grid; grid-template-columns: 1fr; gap: 22px; }",
        ".case { background:#141820; border-radius:22px; padding:18px; }",
        ".meta { margin-top: 12px; color:#a4adbe; font-size:15px; }",
        ".badges { margin-top: 10px; }",
        ".badge { display:inline-block; margin-right:8px; margin-top:6px; border-radius:999px; padding:5px 10px; color:#0c0d12; font-size:13px; font-weight:600; }",
        "</style></head><body><div class='wrap'>",
        "<div class='hero'>",
        f"<h1>ABO RM Visual Report</h1><img src='{html.escape(relpath(dashboard_path, html_path.parent))}' alt='dashboard'>",
        "</div>",
    ]

    stat_items = [
        ("Objects", str(len(set(row["object_id"] for row in rows)))),
        ("Views", str(len(rows))),
        ("Mean Total Error", f"{np.mean([row['total_error'] for row in rows]):.3f}"),
        ("Worst Case", f"{max(row['total_error'] for row in rows):.3f}"),
    ]
    lines.append("<div class='stats'>")
    for key, value in stat_items:
        lines.append(f"<div class='stat'><div class='k'>{html.escape(key)}</div><div class='v'>{html.escape(value)}</div></div>")
    lines.append("</div>")

    for tag, path in tag_paths.items():
        lines.append(f"<div class='section'><h2>{html.escape(tag.replace('_', ' '))}</h2>")
        lines.append(
            f"<div class='tagstrip'><img src='{html.escape(relpath(path, html_path.parent))}' alt='{html.escape(tag)}'></div></div>"
        )

    lines.append("<div class='section'><h2>All Cases</h2><div class='casegrid'>")
    for row in sorted(rows, key=lambda item: item["total_error"], reverse=True):
        badges = []
        for tag in row["tags"]:
            color = TAG_COLORS.get(tag, (120, 120, 140))
            badges.append(
                "<span class='badge' style='background:rgb(%d,%d,%d)'>%s</span>"
                % (color[0], color[1], color[2], html.escape(tag.replace("_", " ")))
            )
        case_path = case_paths[row["case_key"]]
        lines.extend(
            [
                "<div class='case'>",
                f"<img src='{html.escape(relpath(case_path, html_path.parent))}' alt='{html.escape(row['case_key'])}'>",
                (
                    "<div class='meta'>"
                    f"{html.escape(row['object_label'])} / {html.escape(row['view_name'])} "
                    f"&nbsp;&nbsp; pred R/M {row['pred_roughness']:.3f}/{row['pred_metallic']:.3f} "
                    f"&nbsp;&nbsp; gt R/M {row['gt_roughness_mean']:.3f}/{row['gt_metallic_mean']:.3f} "
                    f"&nbsp;&nbsp; total err {row['total_error']:.3f}"
                    "</div>"
                ),
                "<div class='badges'>" + ("".join(badges) if badges else "<span class='badge' style='background:#464e5e;color:#eef1f7'>no tags</span>") + "</div>",
                "</div>",
            ]
        )
    lines.append("</div></div></div></body></html>")
    html_path.write_text("\n".join(lines))
    return html_path


def generate_visualizations(metrics_json: Path, output_dir: Path):
    rows = json.loads(metrics_json.read_text())
    for row in rows:
        row["total_error"] = abs(row["pred_roughness"] - row["gt_roughness_mean"]) + abs(
            row["pred_metallic"] - row["gt_metallic_mean"]
        )
        row["case_key"] = f"{row['object_label']}__{row['view_name']}"

    viz_dir = ensure_dir(output_dir / "viz")
    cases_dir = ensure_dir(viz_dir / "cases")
    tags_dir = ensure_dir(viz_dir / "tags")

    case_paths = {}
    for row in rows:
        filename = f"{slugify(row['object_label'])}__{slugify(row['view_name'])}.png"
        case_path = cases_dir / filename
        build_case_panel(row, case_path)
        case_paths[row["case_key"]] = case_path

    dashboard_path = viz_dir / "dashboard.png"
    build_dashboard(rows, case_paths, dashboard_path)

    tag_paths = {}
    for tag in ["over_smoothing", "metal_nonmetal_confusion", "highlight_misread", "boundary_bleed"]:
        path = tags_dir / f"{tag}.png"
        build_tag_strip(rows, tag, case_paths, path)
        tag_paths[tag] = path

    html_path = build_html_report(rows, output_dir, dashboard_path, case_paths, tag_paths)
    return {
        "dashboard": dashboard_path,
        "html": html_path,
        "cases_dir": cases_dir,
        "tags_dir": tags_dir,
    }


def main():
    args = parse_args()
    outputs = generate_visualizations(args.metrics_json, args.output_dir)
    print(f"Dashboard: {outputs['dashboard']}")
    print(f"HTML report: {outputs['html']}")
    print(f"Case panels: {outputs['cases_dir']}")
    print(f"Tag strips: {outputs['tags_dir']}")


if __name__ == "__main__":
    main()
