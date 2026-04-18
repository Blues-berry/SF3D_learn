from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]

BACKGROUND = (11, 14, 20)
PANEL_BG = (20, 24, 32)
CARD_BG = (28, 33, 44)
CARD_ALT = (34, 40, 52)
TEXT_MAIN = (238, 241, 247)
TEXT_MUTED = (164, 173, 190)
TEXT_FAINT = (120, 129, 147)
TRACK = (66, 75, 91)
GRID = (56, 64, 78)
TITLE_BG = (42, 49, 62)
BORDER = (54, 62, 76)
PRED_COLOR = (255, 144, 64)
MEAN_COLOR = (240, 240, 240)
EDGE_COLOR = (110, 179, 255)
INTERIOR_COLOR = (120, 222, 164)
ERROR_COLOR = (255, 111, 111)
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
TAG_COLORS = {
    "over_smoothing": (87, 184, 255),
    "metal_nonmetal_confusion": (255, 116, 116),
    "highlight_misread": (255, 197, 87),
    "boundary_bleed": (183, 128, 255),
}
CATEGORY_BY_LABEL = {
    "lamp_metal_glass": "lamp",
    "ceramic_stool": "stool",
    "leather_metal_stool": "stool",
    "wood_iron_mirror": "mirror",
    "metal_desk": "desk",
    "faux_linen_headboard": "headboard",
}
CATEGORY_DISPLAY = {
    "lamp": "Lamp",
    "stool": "Stool",
    "mirror": "Mirror",
    "desk": "Desk",
    "headboard": "Headboard",
    "other": "Other",
}
TAG_ORDER = [
    "over_smoothing",
    "metal_nonmetal_confusion",
    "highlight_misread",
    "boundary_bleed",
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


def resolve_data_path(path: str | Path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate
    return candidate.resolve()


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


def relpath(path: Path, start: Path):
    return os.path.relpath(path, start)


def lerp_color(palette, values):
    values = np.clip(values, 0.0, 1.0)
    xs = np.linspace(0.0, 1.0, len(palette))
    channels = []
    for idx in range(3):
        channel = np.interp(values, xs, [color[idx] for color in palette])
        channels.append(channel)
    return np.stack(channels, axis=-1).astype(np.uint8)


def colorize_scalar_image(path: str, palette, tile_size=(256, 256)):
    arr = np.asarray(Image.open(resolve_data_path(path)).convert("RGBA")).astype(np.float32) / 255.0
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


def average_visible_rgb(path: str):
    arr = np.asarray(Image.open(resolve_data_path(path)).convert("RGBA")).astype(np.float32) / 255.0
    alpha = arr[..., 3] > 0.01
    rgb = arr[..., :3]
    if not alpha.any():
        return np.array([0.78, 0.78, 0.78], dtype=np.float32)
    visible = rgb[alpha]
    avg = visible.mean(axis=0)
    avg = np.clip(avg * 0.75 + 0.25, 0.0, 1.0)
    return avg


def normalize(v, axis=-1, eps=1e-8):
    denom = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(denom, eps)


def sample_env(reflect):
    top = np.array([0.73, 0.82, 0.96], dtype=np.float32)
    horizon = np.array([0.98, 0.82, 0.67], dtype=np.float32)
    ground = np.array([0.08, 0.10, 0.12], dtype=np.float32)

    t = np.clip(reflect[..., 2] * 0.5 + 0.5, 0.0, 1.0)
    sky_mix = top * t[..., None] + ground * (1.0 - t[..., None])
    warm_gate = np.clip(1.0 - np.abs(reflect[..., 1]) * 1.6, 0.0, 1.0)
    warm_gate *= np.clip(reflect[..., 0] * 0.5 + 0.5, 0.0, 1.0)
    return sky_mix * (1.0 - 0.28 * warm_gate[..., None]) + horizon * (0.28 * warm_gate[..., None])


def render_material_ball(roughness: float, metallic: float, base_rgb, size=220):
    roughness = float(np.clip(roughness, 0.02, 1.0))
    metallic = float(np.clip(metallic, 0.0, 1.0))
    base_rgb = np.clip(np.asarray(base_rgb, dtype=np.float32), 0.08, 1.0)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    x = (xx - cx) / cx
    y = -(yy - cy) / cy
    r2 = x * x + y * y
    mask = r2 <= 1.0
    z = np.zeros_like(x)
    z[mask] = np.sqrt(1.0 - r2[mask])

    n = np.stack([x, y, z], axis=-1)
    n = normalize(n)
    v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v_img = np.broadcast_to(v, n.shape)

    light_dirs = [
        normalize(np.array([-0.45, 0.25, 0.86], dtype=np.float32)),
        normalize(np.array([0.62, 0.18, 0.76], dtype=np.float32)),
        normalize(np.array([-0.18, -0.88, 0.44], dtype=np.float32)),
    ]
    light_cols = [
        np.array([1.20, 1.13, 1.06], dtype=np.float32),
        np.array([0.55, 0.66, 0.92], dtype=np.float32),
        np.array([0.38, 0.30, 0.24], dtype=np.float32),
    ]

    hemi_t = np.clip(n[..., 1] * 0.5 + 0.5, 0.0, 1.0)
    ambient = (
        np.array([0.10, 0.11, 0.12], dtype=np.float32) * (1.0 - hemi_t[..., None])
        + np.array([0.24, 0.27, 0.33], dtype=np.float32) * hemi_t[..., None]
    )

    diffuse_color = base_rgb * (1.0 - metallic)
    f0 = 0.04 * (1.0 - metallic) + base_rgb * metallic
    alpha = max(0.03, roughness * roughness)
    spec_exp = 6.0 + (1.0 - alpha) ** 3 * 240.0

    diffuse = ambient * diffuse_color * 0.65
    specular = np.zeros_like(diffuse)

    for light_dir, light_col in zip(light_dirs, light_cols):
        l = np.broadcast_to(light_dir, n.shape)
        h = normalize(l + v_img)
        ndotl = np.clip(np.sum(n * l, axis=-1, keepdims=True), 0.0, 1.0)
        ndoth = np.clip(np.sum(n * h, axis=-1, keepdims=True), 0.0, 1.0)
        vdh = np.clip(np.sum(v_img * h, axis=-1, keepdims=True), 0.0, 1.0)
        fresnel = f0 + (1.0 - f0) * ((1.0 - vdh) ** 5)
        diffuse += diffuse_color * light_col * ndotl * 0.55
        specular += fresnel * light_col * (ndoth ** spec_exp) * ndotl * (0.18 + 0.82 * (1.0 - roughness))

    view_dot = np.sum(n * v_img, axis=-1, keepdims=True)
    reflect = normalize(2.0 * view_dot * n - v_img)
    env_col = sample_env(reflect)
    env_strength = 0.10 + 0.90 * (1.0 - roughness)
    env_spec = env_col * f0 * env_strength
    rim = ((1.0 - np.clip(n[..., 2], 0.0, 1.0)) ** 2)[..., None] * np.array([0.10, 0.12, 0.16], dtype=np.float32)

    sphere_rgb = diffuse + specular + env_spec + rim
    sphere_rgb = np.clip(sphere_rgb, 0.0, 1.0)
    sphere_rgb = sphere_rgb ** (1.0 / 2.2)

    top_bg = np.array([25, 30, 38], dtype=np.float32) / 255.0
    bottom_bg = np.array([10, 12, 18], dtype=np.float32) / 255.0
    bg_t = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
    background = top_bg[None, None, :] * (1.0 - bg_t) + bottom_bg[None, None, :] * bg_t
    background = np.broadcast_to(background, (size, size, 3)).copy()

    shadow = np.exp(-(((x * 0.92) ** 2) / 0.68 + (((y + 0.82) / 0.20) ** 2)))
    background *= (1.0 - 0.16 * shadow[..., None])

    image = background
    image[mask] = sphere_rgb[mask]
    image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def image_tile(image: Image.Image, label: str, size=(280, 322)):
    tile = Image.new("RGB", size, PANEL_BG)
    draw = ImageDraw.Draw(tile)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=18, fill=PANEL_BG, outline=BORDER, width=2)
    draw.rounded_rectangle((0, 0, size[0] - 1, 42), radius=18, fill=TITLE_BG)
    draw.rectangle((0, 24, size[0] - 1, 42), fill=TITLE_BG)
    draw.text((16, 11), label, font=FONT_S, fill=TEXT_MAIN)
    inner = ImageOps.contain(image, (size[0] - 28, size[1] - 68))
    tile.paste(inner, ((size[0] - inner.width) // 2, 52))
    return tile


def material_render_tile(render: Image.Image, label: str, stats: str, size=(264, 248)):
    tile = Image.new("RGB", size, CARD_BG)
    draw = ImageDraw.Draw(tile)
    header_h = 40
    footer_h = 28
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=18, fill=CARD_BG, outline=BORDER, width=2)
    draw.rounded_rectangle((0, 0, size[0] - 1, header_h), radius=18, fill=TITLE_BG)
    draw.rectangle((0, 22, size[0] - 1, header_h), fill=TITLE_BG)
    draw.text((16, 10), label, font=FONT_S, fill=TEXT_MAIN)
    inner = ImageOps.contain(render, (size[0] - 28, max(48, size[1] - header_h - footer_h - 18)))
    tile.paste(inner, ((size[0] - inner.width) // 2, header_h + 8))
    stats_bbox = draw.textbbox((0, 0), stats, font=FONT_XS)
    stats_x = int((size[0] - (stats_bbox[2] - stats_bbox[0])) / 2)
    draw.text((stats_x, size[1] - footer_h), stats, font=FONT_XS, fill=TEXT_MUTED)
    return tile


def build_material_compare_strip(row: dict, output_path: Path, size=(576, 322)):
    base_rgb = row["_avg_rgb"]
    gt_render = render_material_ball(row["gt_roughness_mean"], row["gt_metallic_mean"], base_rgb, size=184)
    pred_render = render_material_ball(row["pred_roughness"], row["pred_metallic"], base_rgb, size=184)

    strip = Image.new("RGB", size, PANEL_BG)
    draw = ImageDraw.Draw(strip)
    gt_tile = material_render_tile(
        gt_render,
        "GT Material",
        f"R {row['gt_roughness_mean']:.3f} | M {row['gt_metallic_mean']:.3f}",
    )
    pred_tile = material_render_tile(
        pred_render,
        "Pred Material",
        f"R {row['pred_roughness']:.3f} | M {row['pred_metallic']:.3f}",
    )
    draw.text((18, 14), "Material Compare", font=FONT_M, fill=TEXT_MAIN)
    draw.text((424, 18), "R = roughness, M = metallic", font=FONT_XS, fill=TEXT_FAINT)
    strip.paste(gt_tile, (18, 56))
    strip.paste(pred_tile, (size[0] - 18 - pred_tile.width, 56))
    strip.save(output_path)
    return strip


def draw_scale(draw, x0, y0, x1, y1):
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
    draw.text((18, 138), "Band = GT p10-p90, white = GT mean, orange = prediction", font=FONT_XS, fill=TEXT_FAINT)

    edge_y0, edge_y1 = 196, 214
    draw_scale(draw, x0, edge_y0, x1, edge_y1)
    draw_marker(draw, value_to_x(edge, x0, x1), edge_y0, edge_y1, EDGE_COLOR, width=4)
    draw_marker(draw, value_to_x(interior, x0, x1), edge_y0, edge_y1, INTERIOR_COLOR, width=4)
    draw.text((18, 238), f"edge {edge:.3f}    interior {interior:.3f}    gt std {gt_std:.3f}", font=FONT_S, fill=TEXT_MUTED)
    draw.text((18, 266), "Blue = edge visible pixels, green = interior visible pixels", font=FONT_XS, fill=TEXT_FAINT)
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


def merge_tags(row: dict):
    tags = set(row.get("tags") or [])
    rough_error = abs(row["pred_roughness"] - row["gt_roughness_mean"])
    metal_error = abs(row["pred_metallic"] - row["gt_metallic_mean"])
    rough_range = row["gt_roughness_p90"] - row["gt_roughness_p10"]
    metallic_edge_gap = abs(row["gt_metallic_edge_mean"] - row["gt_metallic_interior_mean"])
    roughness_edge_gap = abs(row["gt_roughness_edge_mean"] - row["gt_roughness_interior_mean"])

    if rough_range > 0.35 and rough_error < 0.10:
        tags.add("over_smoothing")

    if row["gt_metallic_mean"] < 0.15 and row["pred_metallic"] > 0.35:
        tags.add("metal_nonmetal_confusion")
    if row["gt_metallic_mean"] > 0.20 and row["pred_metallic"] < 0.08 and metal_error > 0.15:
        tags.add("metal_nonmetal_confusion")

    if row["brightness_p99"] > 0.82 and row["gt_metallic_mean"] < 0.30:
        if rough_error > 0.12 or metal_error > 0.12:
            tags.add("highlight_misread")

    if metallic_edge_gap > 0.20:
        edge_dist = abs(row["pred_metallic"] - row["gt_metallic_edge_mean"])
        interior_dist = abs(row["pred_metallic"] - row["gt_metallic_interior_mean"])
        if edge_dist + 0.05 < interior_dist:
            tags.add("boundary_bleed")
    if roughness_edge_gap > 0.20:
        edge_dist = abs(row["pred_roughness"] - row["gt_roughness_edge_mean"])
        interior_dist = abs(row["pred_roughness"] - row["gt_roughness_interior_mean"])
        if edge_dist + 0.05 < interior_dist:
            tags.add("boundary_bleed")

    row["tags"] = [tag for tag in TAG_ORDER if tag in tags]


def build_case_panel(row: dict, compare_path: Path, output_path: Path):
    rgba = Image.open(resolve_data_path(row["rgba_path"])).convert("RGBA")
    rgba_bg = Image.new("RGBA", rgba.size, BACKGROUND + (255,))
    rgba = Image.alpha_composite(rgba_bg, rgba).convert("RGB")
    rgba = ImageOps.contain(rgba, (256, 256))
    rough = colorize_scalar_image(row["roughness_path"], ROUGHNESS_PALETTE)
    metal = colorize_scalar_image(row["metallic_path"], METALLIC_PALETTE)
    compare = Image.open(compare_path).convert("RGB")

    input_tile = image_tile(rgba, "Input RGBA")
    rough_tile = image_tile(rough, "GT Roughness")
    metal_tile = image_tile(metal, "GT Metallic")
    rough_card = metric_card(
        "Roughness Stats",
        row["pred_roughness"],
        row["gt_roughness_mean"],
        row["gt_roughness_std"],
        row["gt_roughness_p10"],
        row["gt_roughness_p90"],
        row["gt_roughness_edge_mean"],
        row["gt_roughness_interior_mean"],
    )
    metal_card = metric_card(
        "Metallic Stats",
        row["pred_metallic"],
        row["gt_metallic_mean"],
        row["gt_metallic_std"],
        row["gt_metallic_p10"],
        row["gt_metallic_p90"],
        row["gt_metallic_edge_mean"],
        row["gt_metallic_interior_mean"],
    )

    width = 2216
    panel = Image.new("RGB", (width, 530), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    draw.text((24, 18), f"{row['object_label']} / {row['view_name']}", font=FONT_L, fill=TEXT_MAIN)
    draw.text((24, 54), row["object_name"], font=FONT_S, fill=TEXT_MUTED)
    draw.text((24, 76), f"category: {row['category_display']}", font=FONT_XS, fill=TEXT_FAINT)
    draw.text(
        (24, 104),
        "Layout: Input | GT Roughness | GT Metallic | Material Compare | Roughness Stats | Metallic Stats",
        font=FONT_XS,
        fill=TEXT_FAINT,
    )

    if row["tags"]:
        draw_badges(draw, row["tags"], 540, 100)
    else:
        draw.rounded_rectangle((540, 100, 632, 124), radius=12, fill=(70, 78, 92))
        draw.text((552, 104), "no tags", font=FONT_XS, fill=TEXT_MAIN)

    tiles = [input_tile, rough_tile, metal_tile, compare, rough_card, metal_card]
    x = 24
    y = 144
    for tile in tiles:
        panel.paste(tile, (x, y))
        x += tile.width + 16

    footer = (
        f"brightness p99 {row['brightness_p99']:.3f}   "
        f"highlight frac {row['highlight_fraction']:.4f}   "
        f"GT R/M {row['gt_roughness_mean']:.3f}/{row['gt_metallic_mean']:.3f}   "
        f"Pred R/M {row['pred_roughness']:.3f}/{row['pred_metallic']:.3f}   "
        f"total err {row['total_error']:.3f}"
    )
    draw.text((24, 490), footer, font=FONT_S, fill=TEXT_MUTED)
    panel.save(output_path)


def build_report_card(row: dict, compare_path: Path, output_path: Path):
    rgba = Image.open(resolve_data_path(row["rgba_path"])).convert("RGBA")
    rgba_bg = Image.new("RGBA", rgba.size, BACKGROUND + (255,))
    rgba = Image.alpha_composite(rgba_bg, rgba).convert("RGB")
    rough = colorize_scalar_image(row["roughness_path"], ROUGHNESS_PALETTE, tile_size=(148, 132))
    metal = colorize_scalar_image(row["metallic_path"], METALLIC_PALETTE, tile_size=(148, 132))
    rgba = ImageOps.contain(rgba, (148, 132))
    gt_render = render_material_ball(row["gt_roughness_mean"], row["gt_metallic_mean"], row["_avg_rgb"], size=124)
    pred_render = render_material_ball(row["pred_roughness"], row["pred_metallic"], row["_avg_rgb"], size=124)

    card = Image.new("RGB", (560, 476), CARD_ALT)
    draw = ImageDraw.Draw(card)
    draw.text((18, 14), f"{row['object_label']} / {row['view_name']}", font=FONT_S, fill=TEXT_MAIN)
    draw.text((18, 36), row["category_display"], font=FONT_XS, fill=TEXT_FAINT)
    if row["tags"]:
        draw_badges(draw, row["tags"][:2], 116, 32)

    top_y = 70
    top_tiles = [
        image_tile(rgba, "Input", size=(168, 176)),
        image_tile(rough, "GT Roughness", size=(168, 176)),
        image_tile(metal, "GT Metallic", size=(168, 176)),
    ]
    x = 18
    for tile in top_tiles:
        card.paste(tile, (x, top_y))
        x += 178

    bottom_y = 250
    gt_tile = material_render_tile(
        gt_render,
        "GT Material",
        f"R {row['gt_roughness_mean']:.3f} | M {row['gt_metallic_mean']:.3f}",
        size=(252, 176),
    )
    pred_tile = material_render_tile(
        pred_render,
        "Pred Material",
        f"R {row['pred_roughness']:.3f} | M {row['pred_metallic']:.3f}",
        size=(252, 176),
    )
    card.paste(gt_tile, (18, bottom_y))
    card.paste(pred_tile, (290, bottom_y))
    draw.text(
        (18, 448),
        (
            f"GT {row['gt_roughness_mean']:.3f}/{row['gt_metallic_mean']:.3f}  "
            f"Pred {row['pred_roughness']:.3f}/{row['pred_metallic']:.3f}  "
            f"err {row['total_error']:.3f}"
        ),
        font=FONT_XS,
        fill=TEXT_MUTED,
    )
    card.save(output_path)


def bar(draw, x0, y0, width, value, color, label, text):
    draw.text((x0, y0), label, font=FONT_S, fill=TEXT_MUTED)
    y = y0 + 22
    draw.rounded_rectangle((x0, y, x0 + width, y + 18), radius=9, fill=TRACK)
    draw.rounded_rectangle((x0, y, x0 + int(width * max(0.0, min(1.0, value))), y + 18), radius=9, fill=color)
    draw.text((x0 + width + 10, y - 1), text, font=FONT_S, fill=TEXT_MAIN)


def build_dashboard(rows: list[dict], card_paths: dict[str, Path], output_path: Path):
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    width = 1760
    category_h = 92 + 78 * len(by_category)
    top_rows = min(6, len(rows))
    thumb_h = 476
    top_cols = 2
    top_grid_rows = max(1, (top_rows + top_cols - 1) // top_cols)
    height = 184 + category_h + 46 + top_grid_rows * (thumb_h + 18) + 36
    dashboard = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(dashboard)

    draw.text((28, 20), "ABO RM Visual Dashboard", font=FONT_L, fill=TEXT_MAIN)
    draw.text(
        (28, 58),
        (
            f"{len(set(r['object_id'] for r in rows))} objects    {len(rows)} views    "
            f"mean total err {np.mean([r['total_error'] for r in rows]):.3f}"
        ),
        font=FONT_S,
        fill=TEXT_MUTED,
    )

    tag_counts = Counter(tag for row in rows for tag in row["tags"])
    x = 28
    for tag in TAG_ORDER:
        color = TAG_COLORS[tag]
        box = (x, 96, x + 286, 146)
        draw.rounded_rectangle(box, radius=16, fill=color)
        draw.text((x + 16, 108), tag.replace("_", " "), font=FONT_S, fill=(12, 12, 18))
        draw.text((x + 16, 126), f"{tag_counts.get(tag, 0)} cases", font=FONT_S, fill=(20, 20, 30))
        x += 304

    section_y = 176
    draw.text((28, section_y), "Per-category Mean Absolute Error", font=FONT_M, fill=TEXT_MAIN)
    y = section_y + 42
    for category, items in sorted(by_category.items(), key=lambda item: CATEGORY_DISPLAY.get(item[0], item[0])):
        rough_mae = float(np.mean([abs(r["pred_roughness"] - r["gt_roughness_mean"]) for r in items]))
        metal_mae = float(np.mean([abs(r["pred_metallic"] - r["gt_metallic_mean"]) for r in items]))
        draw.text((28, y), CATEGORY_DISPLAY.get(category, category.title()), font=FONT_S, fill=TEXT_MAIN)
        draw.text((180, y), f"{len(items)} views", font=FONT_XS, fill=TEXT_FAINT)
        bar(draw, 300, y - 4, 260, rough_mae, (102, 180, 255), "roughness", f"{rough_mae:.3f}")
        bar(draw, 840, y - 4, 260, metal_mae, (255, 190, 92), "metallic", f"{metal_mae:.3f}")
        y += 78

    top_y = y + 18
    draw.text((28, top_y), "Top Failure Cards", font=FONT_M, fill=TEXT_MAIN)
    draw.text(
        (250, top_y + 4),
        "Card layout: top row = Input, GT Roughness, GT Metallic; bottom row = GT Material, Pred Material",
        font=FONT_XS,
        fill=TEXT_FAINT,
    )
    grid_y = top_y + 56
    col_gap = 24
    card_w = 560
    for idx, row in enumerate(sorted(rows, key=lambda item: item["total_error"], reverse=True)[:top_rows]):
        col = idx % top_cols
        row_idx = idx // top_cols
        x = 28 + col * (card_w + col_gap)
        y = grid_y + row_idx * (thumb_h + 18)
        card = Image.open(card_paths[row["case_key"]]).convert("RGB")
        dashboard.paste(card, (x, y))

    dashboard.save(output_path)


def build_montage(title: str, rows: list[dict], card_paths: dict[str, Path], output_path: Path):
    selected = rows[:6]
    if not selected:
        canvas = Image.new("RGB", (1280, 250), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 26), title, font=FONT_L, fill=TEXT_MAIN)
        draw.text(
            (24, 58),
            "Card layout: top row = Input, GT Roughness, GT Metallic; bottom row = GT Material, Pred Material",
            font=FONT_XS,
            fill=TEXT_FAINT,
        )
        draw.text((24, 92), "No matching cases in this run.", font=FONT_M, fill=TEXT_MUTED)
        canvas.save(output_path)
        return

    cols = 2 if len(selected) <= 4 else 3
    rows_n = 2
    card_w, card_h = 560, 476
    gap = 18
    margin = 22
    title_h = 98
    width = margin * 2 + cols * card_w + (cols - 1) * gap
    height = title_h + margin + rows_n * card_h + (rows_n - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), title, font=FONT_L, fill=TEXT_MAIN)
    draw.text(
        (margin, 56),
        "Card layout: top row = Input, GT Roughness, GT Metallic; bottom row = GT Material, Pred Material",
        font=FONT_XS,
        fill=TEXT_FAINT,
    )

    for idx in range(cols * rows_n):
        col = idx % cols
        row_idx = idx // cols
        x = margin + col * (card_w + gap)
        y = title_h + row_idx * (card_h + gap)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=22, outline=(48, 55, 68), width=2)
        if idx < len(selected):
            card = Image.open(card_paths[selected[idx]["case_key"]]).convert("RGB")
            canvas.paste(card, (x, y))
    canvas.save(output_path)


def build_html_report(
    rows: list[dict],
    output_dir: Path,
    dashboard_path: Path,
    card_paths: dict[str, Path],
    panel_paths: dict[str, Path],
    compare_paths: dict[str, Path],
    montage_paths: dict[str, Path],
):
    html_path = output_dir / "viz" / "report.html"
    categories = sorted({row["category"] for row in rows}, key=lambda cat: CATEGORY_DISPLAY.get(cat, cat))
    category_summary = []
    for category in categories:
        items = [row for row in rows if row["category"] == category]
        category_summary.append(
            {
                "category": category,
                "label": CATEGORY_DISPLAY.get(category, category.title()),
                "count": len(items),
                "rough_mae": float(np.mean([abs(r["pred_roughness"] - r["gt_roughness_mean"]) for r in items])),
                "metal_mae": float(np.mean([abs(r["pred_metallic"] - r["gt_metallic_mean"]) for r in items])),
                "worst": float(max(r["total_error"] for r in items)),
            }
        )

    montage_blocks = []
    overall_path = montage_paths.get("failed_cases")
    if overall_path is not None:
        montage_blocks.append(
            f"<div class='montage'><h3>Top Failed Cases</h3><img src='{html.escape(relpath(overall_path, html_path.parent))}' alt='failed cases montage'></div>"
        )
    for tag in TAG_ORDER:
        path = montage_paths.get(tag)
        if path is not None:
            montage_blocks.append(
                f"<div class='montage'><h3>{html.escape(tag.replace('_', ' '))}</h3><img src='{html.escape(relpath(path, html_path.parent))}' alt='{html.escape(tag)} montage'></div>"
            )

    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>ABO RM Visual Report</title>",
        "<style>",
        "body { background:#0b0e14; color:#eef1f7; font-family:'DejaVu Sans',Arial,sans-serif; margin:0; }",
        ".wrap { max-width: 1820px; margin:0 auto; padding:24px; }",
        ".hero img, .montage img, .case img { width:100%; display:block; border-radius:18px; }",
        ".hero { margin-bottom:28px; }",
        ".stats { display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; margin: 20px 0 28px; }",
        ".stat, .catcard, .case, .montage, .toolbar { background:#141820; border-radius:20px; padding:18px; }",
        ".stat .k, .meta, .small { color:#a4adbe; }",
        ".stat .v { font-size:28px; margin-top:8px; }",
        ".toolbar { margin-bottom: 22px; }",
        ".buttonrow { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }",
        ".filterbtn { border:0; border-radius:999px; padding:8px 14px; background:#2a3140; color:#eef1f7; cursor:pointer; font-size:14px; }",
        ".filterbtn.active { background:#ff9040; color:#11141a; font-weight:700; }",
        ".catgrid { display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:14px; margin-bottom:26px; }",
        ".catcard { cursor:pointer; }",
        ".catcard h3 { margin:0 0 8px; font-size:20px; }",
        ".montages { display:grid; grid-template-columns: 1fr; gap:18px; margin: 28px 0; }",
        ".casegrid { display:grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap:22px; }",
        ".case .meta { margin-top:12px; font-size:14px; }",
        ".badges { margin-top:10px; }",
        ".badge { display:inline-block; margin-right:8px; margin-top:6px; border-radius:999px; padding:5px 10px; color:#0c0d12; font-size:13px; font-weight:600; }",
        ".links { margin-top: 10px; }",
        ".links a { color:#9fd1ff; margin-right:14px; text-decoration:none; }",
        ".links a:hover { text-decoration:underline; }",
        "</style></head><body><div class='wrap'>",
        "<div class='hero'>",
        "<h1>ABO RM Visual Report</h1>",
        f"<img src='{html.escape(relpath(dashboard_path, html_path.parent))}' alt='dashboard'>",
        "</div>",
        "<div class='stats'>",
        f"<div class='stat'><div class='k'>Objects</div><div class='v'>{len(set(row['object_id'] for row in rows))}</div></div>",
        f"<div class='stat'><div class='k'>Views</div><div class='v'>{len(rows)}</div></div>",
        f"<div class='stat'><div class='k'>Mean Total Error</div><div class='v'>{np.mean([row['total_error'] for row in rows]):.3f}</div></div>",
        f"<div class='stat'><div class='k'>Worst Case</div><div class='v'>{max(row['total_error'] for row in rows):.3f}</div></div>",
        "</div>",
        "<div class='toolbar'>",
        "<h2 style='margin:0'>Interactive Filters</h2>",
        "<div class='small'>Filter the gallery by object category and failure type.</div>",
        "<div class='buttonrow' id='categoryButtons'>",
        "<button class='filterbtn active' data-category='all'>All categories</button>",
    ]
    for category in categories:
        lines.append(
            f"<button class='filterbtn' data-category='{html.escape(category)}'>{html.escape(CATEGORY_DISPLAY.get(category, category.title()))}</button>"
        )
    lines.extend(
        [
            "</div>",
            "<div class='buttonrow' id='tagButtons'>",
            "<button class='filterbtn active' data-tag='all'>All tags</button>",
        ]
    )
    for tag in TAG_ORDER:
        lines.append(f"<button class='filterbtn' data-tag='{html.escape(tag)}'>{html.escape(tag.replace('_', ' '))}</button>")
    lines.extend(["</div>", "</div>"])

    lines.append("<div class='catgrid'>")
    for item in category_summary:
        lines.append(
            "<button class='catcard' data-category-card='%s'>"
            "<h3>%s</h3>"
            "<div class='small'>%d views</div>"
            "<div style='margin-top:10px'>roughness MAE %.3f</div>"
            "<div>metallic MAE %.3f</div>"
            "<div style='margin-top:10px' class='small'>worst case %.3f</div>"
            "</button>"
            % (
                html.escape(item["category"]),
                html.escape(item["label"]),
                item["count"],
                item["rough_mae"],
                item["metal_mae"],
                item["worst"],
            )
        )
    lines.append("</div>")

    lines.append("<div class='montages'>" + "".join(montage_blocks) + "</div>")
    lines.append("<h2>Cases</h2><div class='casegrid' id='caseGrid'>")
    for row in sorted(rows, key=lambda item: item["total_error"], reverse=True):
        badges = []
        for tag in row["tags"]:
            color = TAG_COLORS.get(tag, (120, 120, 140))
            badges.append(
                "<span class='badge' style='background:rgb(%d,%d,%d)'>%s</span>"
                % (color[0], color[1], color[2], html.escape(tag.replace("_", " ")))
            )
        if not badges:
            badges = ["<span class='badge' style='background:#464e5e;color:#eef1f7'>no tags</span>"]

        category = html.escape(row["category"])
        tags = " ".join(row["tags"])
        card_path = html.escape(relpath(card_paths[row["case_key"]], html_path.parent))
        panel_path = html.escape(relpath(panel_paths[row["case_key"]], html_path.parent))
        compare_path = html.escape(relpath(compare_paths[row["case_key"]], html_path.parent))
        lines.extend(
            [
                (
                    f"<div class='case' data-category='{category}' data-tags='{html.escape(tags)}'>"
                    f"<img src='{card_path}' alt='{html.escape(row['case_key'])}'>"
                    f"<div class='meta'>{html.escape(row['category_display'])} &nbsp;&nbsp; "
                    f"{html.escape(row['object_label'])} / {html.escape(row['view_name'])} &nbsp;&nbsp; "
                    f"pred {row['pred_roughness']:.3f}/{row['pred_metallic']:.3f} &nbsp;&nbsp; "
                    f"gt {row['gt_roughness_mean']:.3f}/{row['gt_metallic_mean']:.3f} &nbsp;&nbsp; "
                    f"total err {row['total_error']:.3f}</div>"
                    f"<div class='badges'>{''.join(badges)}</div>"
                    f"<div class='links'><a href='{panel_path}'>full panel</a><a href='{compare_path}'>GT vs Pred render</a></div>"
                    "</div>"
                )
            ]
        )
    lines.extend(
        [
            "</div>",
            "<script>",
            "let activeCategory='all';",
            "let activeTag='all';",
            "const cases=[...document.querySelectorAll('.case')];",
            "function syncButtons(selector, attr, value){",
            "  document.querySelectorAll(selector).forEach(btn=>btn.classList.toggle('active', btn.getAttribute(attr)===value));",
            "}",
            "function updateFilters(){",
            "  cases.forEach(card=>{",
            "    const categoryOk = activeCategory==='all' || card.dataset.category===activeCategory;",
            "    const tags = (card.dataset.tags || '').split(/\\s+/).filter(Boolean);",
            "    const tagOk = activeTag==='all' || tags.includes(activeTag);",
            "    card.style.display = (categoryOk && tagOk) ? '' : 'none';",
            "  });",
            "  syncButtons('#categoryButtons .filterbtn', 'data-category', activeCategory);",
            "  syncButtons('#tagButtons .filterbtn', 'data-tag', activeTag);",
            "}",
            "document.querySelectorAll('#categoryButtons .filterbtn').forEach(btn=>btn.addEventListener('click', ()=>{ activeCategory = btn.dataset.category; updateFilters(); }));",
            "document.querySelectorAll('#tagButtons .filterbtn').forEach(btn=>btn.addEventListener('click', ()=>{ activeTag = btn.dataset.tag; updateFilters(); }));",
            "document.querySelectorAll('[data-category-card]').forEach(btn=>btn.addEventListener('click', ()=>{ activeCategory = btn.dataset.categoryCard; updateFilters(); window.scrollTo({top: document.getElementById('caseGrid').offsetTop - 20, behavior:'smooth'}); }));",
            "updateFilters();",
            "</script>",
            "</div></body></html>",
        ]
    )
    html_path.write_text("\n".join(lines))
    return html_path


def enrich_rows(rows: list[dict]):
    avg_rgb_cache = {}
    for row in rows:
        row["total_error"] = abs(row["pred_roughness"] - row["gt_roughness_mean"]) + abs(
            row["pred_metallic"] - row["gt_metallic_mean"]
        )
        row["case_key"] = f"{row['object_label']}__{row['view_name']}"
        category = row.get("category") or CATEGORY_BY_LABEL.get(row["object_label"], "other")
        row["category"] = category
        row["category_display"] = CATEGORY_DISPLAY.get(category, category.title())
        if row["rgba_path"] not in avg_rgb_cache:
            avg_rgb_cache[row["rgba_path"]] = average_visible_rgb(row["rgba_path"])
        row["_avg_rgb"] = avg_rgb_cache[row["rgba_path"]]
        merge_tags(row)


def generate_visualizations(metrics_json: Path, output_dir: Path):
    rows = json.loads(metrics_json.read_text())
    enrich_rows(rows)

    viz_dir = ensure_dir(output_dir / "viz")
    cases_dir = ensure_dir(viz_dir / "cases")
    cards_dir = ensure_dir(viz_dir / "cards")
    materials_dir = ensure_dir(viz_dir / "materials")
    montages_dir = ensure_dir(viz_dir / "montages")

    compare_paths = {}
    panel_paths = {}
    card_paths = {}

    for row in rows:
        slug = f"{slugify(row['object_label'])}__{slugify(row['view_name'])}"
        compare_path = materials_dir / f"{slug}.png"
        panel_path = cases_dir / f"{slug}.png"
        card_path = cards_dir / f"{slug}.png"

        build_material_compare_strip(row, compare_path)
        build_case_panel(row, compare_path, panel_path)
        build_report_card(row, compare_path, card_path)

        compare_paths[row["case_key"]] = compare_path
        panel_paths[row["case_key"]] = panel_path
        card_paths[row["case_key"]] = card_path

    dashboard_path = viz_dir / "dashboard.png"
    build_dashboard(rows, card_paths, dashboard_path)

    montage_paths = {}
    failed_rows = [row for row in rows if row["tags"]]
    if failed_rows:
        failed_rows = sorted(failed_rows, key=lambda row: row["total_error"], reverse=True)
    else:
        failed_rows = sorted(rows, key=lambda row: row["total_error"], reverse=True)
    montage_paths["failed_cases"] = montages_dir / "failed_cases_3x2.png"
    build_montage("Top Failed Cases", failed_rows, card_paths, montage_paths["failed_cases"])

    for tag in TAG_ORDER:
        tagged = sorted([row for row in rows if tag in row["tags"]], key=lambda row: row["total_error"], reverse=True)
        montage_paths[tag] = montages_dir / f"{tag}.png"
        build_montage(tag.replace("_", " ").title(), tagged, card_paths, montage_paths[tag])

    html_path = build_html_report(
        rows,
        output_dir,
        dashboard_path,
        card_paths,
        panel_paths,
        compare_paths,
        montage_paths,
    )
    return {
        "dashboard": dashboard_path,
        "html": html_path,
        "cases_dir": cases_dir,
        "cards_dir": cards_dir,
        "materials_dir": materials_dir,
        "montages_dir": montages_dir,
    }


def main():
    args = parse_args()
    outputs = generate_visualizations(args.metrics_json, args.output_dir)
    print(f"Dashboard: {outputs['dashboard']}")
    print(f"HTML report: {outputs['html']}")
    print(f"Case panels: {outputs['cases_dir']}")
    print(f"Report cards: {outputs['cards_dir']}")
    print(f"Material compares: {outputs['materials_dir']}")
    print(f"Montages: {outputs['montages_dir']}")


if __name__ == "__main__":
    main()
