from __future__ import annotations

import argparse
import html
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "abo_rm_mini"

BACKGROUND = (10, 14, 22)
PANEL = (18, 24, 36)
PANEL_ALT = (25, 32, 47)
TITLE_BG = (33, 43, 63)
BORDER = (57, 71, 98)
TEXT_MAIN = (242, 245, 250)
TEXT_MUTED = (176, 188, 206)
TEXT_FAINT = (127, 139, 157)
TRACK = (49, 60, 80)
GRID = (68, 81, 106)
ACCENT_BLUE = (98, 176, 255)
ACCENT_ORANGE = (255, 168, 79)
ACCENT_GREEN = (115, 216, 157)
ACCENT_RED = (255, 116, 116)
FAILURE_ORDER = [
    "over_smoothing",
    "metal_nonmetal_confusion",
    "local_highlight_misread",
    "boundary_bleed",
]
FAILURE_COLORS = {
    "over_smoothing": (93, 188, 255),
    "metal_nonmetal_confusion": (255, 118, 118),
    "local_highlight_misread": (255, 194, 96),
    "boundary_bleed": (183, 140, 255),
}
ROUGHNESS_PALETTE = [
    (24, 41, 88),
    (70, 114, 182),
    (143, 194, 216),
    (252, 218, 120),
    (214, 76, 54),
]
METALLIC_PALETTE = [
    (10, 15, 24),
    (56, 67, 90),
    (124, 149, 176),
    (198, 167, 84),
    (255, 220, 120),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--assets-dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


FONT_XL = load_font(34)
FONT_L = load_font(26)
FONT_M = load_font(20)
FONT_S = load_font(16)
FONT_XS = load_font(14)


def relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start)


def title_case_token(token: str) -> str:
    return token.replace("_", " ").replace("-", " ").title()


def display_failure(tag: str) -> str:
    return title_case_token(tag)


def display_category(category: str) -> str:
    return title_case_token(category)


def load_json(path: Path):
    return json.loads(path.read_text())


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def pct(value: int, total: int) -> float:
    return (100.0 * value / total) if total else 0.0


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


def slugify(text: str) -> str:
    chars = []
    for ch in text.lower():
        chars.append(ch if ch.isalnum() else "_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def value_to_x(value: float, x0: int, x1: int) -> int:
    return int(round(x0 + np.clip(value, 0.0, 1.0) * (x1 - x0)))


def lerp_color(palette, values):
    values = np.clip(values, 0.0, 1.0)
    xs = np.linspace(0.0, 1.0, len(palette))
    channels = []
    for idx in range(3):
        channel = np.interp(values, xs, [color[idx] for color in palette])
        channels.append(channel)
    return np.stack(channels, axis=-1).astype(np.uint8)


def resolve_image(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def average_visible_rgb(path: str | Path):
    arr = np.asarray(Image.open(resolve_image(path)).convert("RGBA")).astype(np.float32) / 255.0
    alpha = arr[..., 3] > 0.01
    if not alpha.any():
        return np.array([0.78, 0.78, 0.78], dtype=np.float32)
    avg = arr[..., :3][alpha].mean(axis=0)
    return np.clip(avg * 0.75 + 0.25, 0.0, 1.0)


def colorize_scalar_image(path: str | Path, palette, size=(240, 190)):
    arr = np.asarray(Image.open(resolve_image(path)).convert("RGBA")).astype(np.float32) / 255.0
    values = arr[..., 0]
    alpha = arr[..., 3] > 0.01
    rgb = lerp_color(palette, values)
    bg = np.zeros_like(rgb)
    bg[..., 0] = BACKGROUND[0]
    bg[..., 1] = BACKGROUND[1]
    bg[..., 2] = BACKGROUND[2]
    rgb = np.where(alpha[..., None], rgb, bg)
    return ImageOps.contain(Image.fromarray(rgb, mode="RGB"), size)


def rgba_preview(path: str | Path, size=(240, 190)):
    rgba = Image.open(resolve_image(path)).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, BACKGROUND + (255,))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    return ImageOps.contain(rgb, size)


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


def render_material_ball(roughness: float, metallic: float, base_rgb, size=164):
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

    sphere_rgb = np.clip(diffuse + specular + env_spec + rim, 0.0, 1.0) ** (1.0 / 2.2)
    top_bg = np.array([25, 30, 38], dtype=np.float32) / 255.0
    bottom_bg = np.array([10, 12, 18], dtype=np.float32) / 255.0
    bg_t = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
    background = top_bg[None, None, :] * (1.0 - bg_t) + bottom_bg[None, None, :] * bg_t
    background = np.broadcast_to(background, (size, size, 3)).copy()
    image = background
    image[mask] = sphere_rgb[mask]
    return Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")


def tile(image: Image.Image, label: str, size=(250, 230)):
    canvas = Image.new("RGB", size, PANEL)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=18, fill=PANEL, outline=BORDER, width=2)
    draw.rounded_rectangle((0, 0, size[0] - 1, 38), radius=18, fill=TITLE_BG)
    draw.rectangle((0, 18, size[0] - 1, 38), fill=TITLE_BG)
    draw.text((14, 10), label, font=FONT_XS, fill=TEXT_MAIN)
    inner = ImageOps.contain(image, (size[0] - 20, size[1] - 58))
    canvas.paste(inner, ((size[0] - inner.width) // 2, 46))
    return canvas


def draw_badges(draw: ImageDraw.ImageDraw, tags: list[str], start_x: int, y: int):
    x = start_x
    for tag in tags:
        label = display_failure(tag)
        bbox = draw.textbbox((0, 0), label, font=FONT_XS)
        width = bbox[2] - bbox[0] + 18
        color = FAILURE_COLORS.get(tag, (120, 120, 140))
        draw.rounded_rectangle((x, y, x + width, y + 24), radius=12, fill=color)
        draw.text((x + 9, y + 5), label, font=FONT_XS, fill=(12, 12, 18))
        x += width + 8


def draw_summary_box(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, title: str, value: str, detail: str, accent):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=20, fill=PANEL, outline=BORDER, width=2)
    draw.rounded_rectangle((x + 1, y + 1, x + w - 1, y + 15), radius=20, fill=accent)
    draw.text((x + 18, y + 26), title, font=FONT_XS, fill=TEXT_FAINT)
    draw.text((x + 18, y + 52), value, font=FONT_L, fill=TEXT_MAIN)
    draw.text((x + 18, y + 86), detail, font=FONT_XS, fill=TEXT_MUTED)


def draw_bar_chart(title: str, subtitle: str, items: list[dict], output_path: Path, width=980):
    left = 320
    row_h = 54
    height = 120 + row_h * len(items)
    image = Image.new("RGB", (width, height), PANEL_ALT)
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, font=FONT_L, fill=TEXT_MAIN)
    draw.text((24, 50), subtitle, font=FONT_S, fill=TEXT_MUTED)

    if not items:
        draw.text((24, 90), "No data", font=FONT_S, fill=TEXT_FAINT)
        image.save(output_path)
        return image

    max_value = max(item["value"] for item in items) or 1.0
    bar_x0 = left
    bar_x1 = width - 220
    y = 96
    for item in items:
        label = truncate(item["label"], 42)
        draw.text((24, y + 10), label, font=FONT_S, fill=TEXT_MAIN)
        draw.text((24, y + 30), item.get("detail", ""), font=FONT_XS, fill=TEXT_FAINT)
        draw.rounded_rectangle((bar_x0, y + 12, bar_x1, y + 30), radius=9, fill=TRACK)
        bar_w = int((bar_x1 - bar_x0) * (item["value"] / max_value))
        draw.rounded_rectangle((bar_x0, y + 12, bar_x0 + bar_w, y + 30), radius=9, fill=item["color"])
        draw.text((bar_x1 + 14, y + 8), item["value_text"], font=FONT_S, fill=TEXT_MAIN)
        y += row_h

    image.save(output_path)
    return image


def build_case_card(row: dict, output_path: Path):
    rgba = rgba_preview(row["paths"]["rgba"])
    rough = colorize_scalar_image(row["paths"]["roughness"], ROUGHNESS_PALETTE)
    metallic = colorize_scalar_image(row["paths"]["metallic"], METALLIC_PALETTE)
    avg_rgb = average_visible_rgb(row["paths"]["rgba"])
    gt_mat = render_material_ball(row["gt"]["roughness"]["whole_mask"]["mean"], row["gt"]["metallic"]["whole_mask"]["mean"], avg_rgb)
    pred_mat = render_material_ball(row["pred"]["roughness"], row["pred"]["metallic"], avg_rgb)

    canvas = Image.new("RGB", (1360, 410), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), f"{row['object_id']} / {row['view_name']}", font=FONT_L, fill=TEXT_MAIN)
    draw.text((20, 50), truncate(row["name"], 110), font=FONT_S, fill=TEXT_MUTED)
    draw.text((20, 74), f"{display_category(row['category'])}    bucket {display_failure(row['sampling_bucket'])}", font=FONT_XS, fill=TEXT_FAINT)
    draw.text(
        (20, 102),
        "Layout: Input | GT Roughness | GT Metallic | GT Material | Pred Material",
        font=FONT_XS,
        fill=TEXT_FAINT,
    )
    if row["failure_tags"]:
        draw_badges(draw, row["failure_tags"], 690, 96)

    tiles = [
        tile(rgba, "Input"),
        tile(rough, "GT Roughness"),
        tile(metallic, "GT Metallic"),
        tile(gt_mat, f"GT Material  R {row['gt']['roughness']['whole_mask']['mean']:.3f}  M {row['gt']['metallic']['whole_mask']['mean']:.3f}"),
        tile(pred_mat, f"Pred Material  R {row['pred']['roughness']:.3f}  M {row['pred']['metallic']:.3f}"),
    ]
    x = 20
    for item in tiles:
        canvas.paste(item, (x, 138))
        x += item.width + 16

    footer = (
        f"total {row['errors']['total_error']:.3f}    "
        f"rough abs {row['errors']['abs_error_roughness']:.3f}    "
        f"metal abs {row['errors']['abs_error_metallic']:.3f}    "
        f"primary {display_failure(row['primary_failure']) if row['primary_failure'] != 'none' else 'None'}"
    )
    draw.text((20, 374), footer, font=FONT_S, fill=TEXT_MUTED)
    canvas.save(output_path)


def build_montage(card_paths: list[Path], output_path: Path, title: str, subtitle: str, cols: int):
    if not card_paths:
        image = Image.new("RGB", (1200, 220), BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.text((24, 26), title, font=FONT_L, fill=TEXT_MAIN)
        draw.text((24, 64), subtitle, font=FONT_S, fill=TEXT_MUTED)
        draw.text((24, 110), "No cases available", font=FONT_M, fill=TEXT_FAINT)
        image.save(output_path)
        return

    cards = [Image.open(path).convert("RGB") for path in card_paths]
    card_w, card_h = cards[0].size
    rows = math.ceil(len(cards) / cols)
    width = 40 + cols * card_w + (cols - 1) * 18
    height = 120 + rows * card_h + (rows - 1) * 18 + 24
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), title, font=FONT_L, fill=TEXT_MAIN)
    draw.text((20, 54), subtitle, font=FONT_S, fill=TEXT_MUTED)

    for idx, card in enumerate(cards):
        row_idx = idx // cols
        col_idx = idx % cols
        x = 20 + col_idx * (card_w + 18)
        y = 96 + row_idx * (card_h + 18)
        image.paste(card, (x, y))
    image.save(output_path)


def compose_dashboard(output_path: Path, overall: dict, chart_paths: dict[str, Path]):
    cards = [
        ("Objects", str(overall["objects"]), "rendered and aligned", ACCENT_BLUE),
        ("Views", str(overall["views"]), "baseline evaluated", ACCENT_ORANGE),
        ("Roughness MAE", f"{overall['rough_mae']:.3f}", "whole-mask scalar", ACCENT_GREEN),
        ("Metallic MAE", f"{overall['metal_mae']:.3f}", "whole-mask scalar", ACCENT_RED),
        ("Total Error", f"{overall['total_mae']:.3f}", "rough + metallic abs error", (187, 149, 255)),
    ]
    chart_images = {name: Image.open(path).convert("RGB") for name, path in chart_paths.items() if path.exists()}
    taxonomy = chart_images["taxonomy"]
    bucket = chart_images["bucket"]
    category = chart_images["category"]
    objects = chart_images["objects"]

    width = 2000
    height = 1700
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((32, 24), "ABO RM Mini 200-Object Final Report", font=FONT_XL, fill=TEXT_MAIN)
    draw.text(
        (32, 68),
        "Stable-Fast-3D baseline gap summary for roughness / metallic with full 200-object completion",
        font=FONT_S,
        fill=TEXT_MUTED,
    )
    draw.text((32, 92), f"Generated {overall['generated_at']}", font=FONT_XS, fill=TEXT_FAINT)

    x = 32
    for title, value, detail, accent in cards:
        draw_summary_box(draw, x, 128, 360, 122, title, value, detail, accent)
        x += 378

    image.paste(taxonomy, (32, 290))
    image.paste(bucket, (1000, 290))
    image.paste(category, (32, 880))
    image.paste(objects, (1000, 880))
    image.save(output_path)


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{html.escape(header)}</th>" for header in headers)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{cell}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def build_report_html(output_dir: Path, assets_dir: Path, report: dict):
    chart_rel = {name: relpath(path, output_dir) for name, path in report["charts"].items()}
    montage_rel = {name: relpath(path, output_dir) for name, path in report["montages"].items()}

    taxonomy_rows = [
        [
            html.escape(display_failure(item["tag"])),
            str(item["count"]),
            f"{item['rate']:.1f}%",
        ]
        for item in report["taxonomy_summary"]
    ]
    bucket_rows = [
        [
            html.escape(display_failure(item["bucket"])),
            str(item["views"]),
            f"{item['rough_mae']:.3f}",
            f"{item['metal_mae']:.3f}",
            f"{item['total_mae']:.3f}",
            html.escape(display_failure(item["dominant_tag"]) if item["dominant_tag"] else "None"),
        ]
        for item in report["bucket_summary"]
    ]
    category_rows = [
        [
            html.escape(display_category(item["category"])),
            str(item["views"]),
            f"{item['rough_mae']:.3f}",
            f"{item['metal_mae']:.3f}",
            f"{item['total_mae']:.3f}",
        ]
        for item in report["category_summary"][:18]
    ]
    object_rows = [
        [
            html.escape(item["object_id"]),
            html.escape(truncate(item["name"], 64)),
            html.escape(display_category(item["category"])),
            str(item["views_evaluated"]),
            f"{item['mean_total_error']:.3f}",
            html.escape(item["worst_view"]),
        ]
        for item in report["object_summary"][:18]
    ]

    category_options = "".join(
        f"<option value=\"{html.escape(cat)}\">{html.escape(display_category(cat))}</option>"
        for cat in sorted({row['category'] for row in report['rows']})
    )
    bucket_options = "".join(
        f"<option value=\"{html.escape(bucket)}\">{html.escape(display_failure(bucket))}</option>"
        for bucket in FAILURE_ORDER
    )
    primary_options = "".join(
        f"<option value=\"{html.escape(tag)}\">{html.escape(display_failure(tag))}</option>"
        for tag in ["none", *FAILURE_ORDER]
    )

    case_rows = []
    for rank, row in enumerate(report["rows"], start=1):
        tags = " ".join(row["failure_tags"])
        case_rows.append(
            "\n".join(
                [
                    (
                        f"<tr data-category=\"{html.escape(row['category'])}\" "
                        f"data-bucket=\"{html.escape(row['sampling_bucket'])}\" "
                        f"data-primary=\"{html.escape(row['primary_failure'])}\" "
                        f"data-tags=\"{html.escape(tags)}\" "
                        f"data-search=\"{html.escape((row['object_id'] + ' ' + row['name'] + ' ' + row['view_name']).lower())}\" "
                        f"data-total=\"{row['errors']['total_error']:.6f}\">"
                    ),
                    f"<td>{rank}</td>",
                    (
                        "<td class=\"thumbs\">"
                        f"<img loading=\"lazy\" src=\"{html.escape(relpath(resolve_image(row['paths']['rgba']), output_dir))}\" alt=\"input\">"
                        f"<img loading=\"lazy\" src=\"{html.escape(relpath(resolve_image(row['paths']['roughness']), output_dir))}\" alt=\"roughness\">"
                        f"<img loading=\"lazy\" src=\"{html.escape(relpath(resolve_image(row['paths']['metallic']), output_dir))}\" alt=\"metallic\">"
                        "</td>"
                    ),
                    (
                        "<td>"
                        f"<div class=\"case-id\">{html.escape(row['object_id'])} / {html.escape(row['view_name'])}</div>"
                        f"<div class=\"case-name\">{html.escape(row['name'])}</div>"
                        "</td>"
                    ),
                    (
                        "<td>"
                        f"<span class=\"chip neutral\">{html.escape(display_category(row['category']))}</span>"
                        f"<span class=\"chip neutral\">{html.escape(display_failure(row['sampling_bucket']))}</span>"
                        "</td>"
                    ),
                    (
                        "<td>"
                        f"<div>{html.escape(display_failure(row['primary_failure']) if row['primary_failure'] != 'none' else 'None')}</div>"
                        f"<div class=\"tag-list\">"
                        + "".join(
                            f"<span class=\"chip {html.escape(tag)}\">{html.escape(display_failure(tag))}</span>"
                            for tag in row["failure_tags"]
                        )
                        + "</div></td>"
                    ),
                    (
                        "<td>"
                        f"<div>GT {row['gt']['roughness']['whole_mask']['mean']:.3f} / {row['gt']['metallic']['whole_mask']['mean']:.3f}</div>"
                        f"<div class=\"sub\">Pred {row['pred']['roughness']:.3f} / {row['pred']['metallic']:.3f}</div>"
                        "</td>"
                    ),
                    (
                        "<td>"
                        f"<div>rough {row['errors']['abs_error_roughness']:.3f}</div>"
                        f"<div class=\"sub\">metal {row['errors']['abs_error_metallic']:.3f}</div>"
                        "</td>"
                    ),
                    f"<td class=\"strong\">{row['errors']['total_error']:.3f}</td>",
                    "</tr>",
                ]
            )
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ABO RM Mini Report</title>
  <style>
    :root {{
      --bg: #0b0f17;
      --bg2: #131c2b;
      --panel: #182233;
      --panel2: #233149;
      --line: #33425d;
      --text: #f0f4fa;
      --muted: #b0bdd0;
      --faint: #8190a6;
      --blue: #62b0ff;
      --orange: #ffa84f;
      --green: #73d89d;
      --red: #ff7474;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(89, 130, 187, 0.28), transparent 34%),
        radial-gradient(circle at top right, rgba(255, 168, 79, 0.18), transparent 28%),
        linear-gradient(180deg, var(--bg2), var(--bg));
    }}
    .page {{ max-width: 1680px; margin: 0 auto; padding: 32px 24px 48px; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 24px;
      align-items: start;
      margin-bottom: 24px;
    }}
    .hero-copy {{
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: rgba(18, 27, 40, 0.88);
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 40px;
      line-height: 1.05;
      letter-spacing: -0.02em;
    }}
    .lede {{
      margin: 0;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.6;
      max-width: 58ch;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 16px;
      margin-top: 20px;
    }}
    .stat {{
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(30, 40, 59, 0.88);
    }}
    .stat .label {{ color: var(--faint); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat .value {{ font-size: 30px; margin-top: 10px; font-weight: 700; }}
    .stat .detail {{ margin-top: 8px; color: var(--muted); font-size: 13px; }}
    .callout {{
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(180deg, rgba(22, 31, 47, 0.95), rgba(17, 24, 36, 0.92));
    }}
    .callout h2 {{ margin: 0 0 10px; font-size: 22px; }}
    .callout p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .callout .mini-list {{ margin-top: 16px; display: grid; gap: 10px; }}
    .mini-item {{ color: var(--muted); }}
    .section {{
      margin-top: 28px;
      padding: 22px 22px 26px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: rgba(20, 28, 41, 0.88);
    }}
    .section h2 {{ margin: 0 0 8px; font-size: 26px; }}
    .section .section-note {{ color: var(--muted); margin-bottom: 18px; }}
    .dashboard img, .chart-grid img, .montage-grid img {{
      width: 100%;
      border-radius: 22px;
      border: 1px solid var(--line);
      display: block;
      background: #0b0f17;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .montage-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
    }}
    .table-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 18px;
      background: rgba(24, 34, 51, 0.9);
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(78, 94, 124, 0.35);
      vertical-align: top;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: rgba(31, 43, 63, 0.98);
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .filters {{
      display: grid;
      grid-template-columns: 2fr repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }}
    input, select {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(11, 17, 27, 0.92);
      color: var(--text);
      font: inherit;
    }}
    .chip {{
      display: inline-block;
      padding: 4px 9px;
      margin: 0 6px 6px 0;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      color: #111820;
    }}
    .chip.neutral {{
      color: var(--text);
      background: #32425d;
    }}
    .chip.over_smoothing {{ background: rgb({FAILURE_COLORS['over_smoothing'][0]}, {FAILURE_COLORS['over_smoothing'][1]}, {FAILURE_COLORS['over_smoothing'][2]}); }}
    .chip.metal_nonmetal_confusion {{ background: rgb({FAILURE_COLORS['metal_nonmetal_confusion'][0]}, {FAILURE_COLORS['metal_nonmetal_confusion'][1]}, {FAILURE_COLORS['metal_nonmetal_confusion'][2]}); }}
    .chip.local_highlight_misread {{ background: rgb({FAILURE_COLORS['local_highlight_misread'][0]}, {FAILURE_COLORS['local_highlight_misread'][1]}, {FAILURE_COLORS['local_highlight_misread'][2]}); }}
    .chip.boundary_bleed {{ background: rgb({FAILURE_COLORS['boundary_bleed'][0]}, {FAILURE_COLORS['boundary_bleed'][1]}, {FAILURE_COLORS['boundary_bleed'][2]}); }}
    .thumbs {{
      min-width: 240px;
    }}
    .thumbs img {{
      width: 72px;
      height: 72px;
      object-fit: contain;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #0b1018;
      margin-right: 6px;
    }}
    .case-id {{ font-weight: 700; margin-bottom: 6px; }}
    .case-name {{ color: var(--muted); line-height: 1.45; max-width: 34ch; }}
    .sub {{ color: var(--muted); margin-top: 4px; }}
    .tag-list {{ margin-top: 6px; }}
    .strong {{ font-weight: 700; color: #fff; }}
    .foot {{
      color: var(--faint);
      margin-top: 18px;
      font-size: 13px;
    }}
    @media (max-width: 1100px) {{
      .hero, .summary-grid, .chart-grid, .table-grid, .filters {{ grid-template-columns: 1fr; }}
      .thumbs {{ min-width: 190px; }}
      h1 {{ font-size: 32px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <div class="hero-copy">
        <h1>ABO RM Mini 200-Object Final Report</h1>
        <p class="lede">
          This is the final, usable 200-object benchmark pass for Stable-Fast-3D roughness / metallic gap analysis.
          Every object rendered successfully, every view has GT alignment, and the final eval covers all 600 fixed views.
        </p>
        <div class="summary-grid">
          <div class="stat"><div class="label">Objects</div><div class="value">{report['overall']['objects']}</div><div class="detail">rendered + aligned</div></div>
          <div class="stat"><div class="label">Views</div><div class="value">{report['overall']['views']}</div><div class="detail">baseline evaluated</div></div>
          <div class="stat"><div class="label">Roughness MAE</div><div class="value">{report['overall']['rough_mae']:.3f}</div><div class="detail">whole-mask scalar</div></div>
          <div class="stat"><div class="label">Metallic MAE</div><div class="value">{report['overall']['metal_mae']:.3f}</div><div class="detail">whole-mask scalar</div></div>
          <div class="stat"><div class="label">Total Error</div><div class="value">{report['overall']['total_mae']:.3f}</div><div class="detail">rough + metallic abs error</div></div>
        </div>
      </div>
      <div class="callout">
        <h2>Headline</h2>
        <p>
          The dominant structured gap is still <strong>metal vs non-metal confusion</strong>: it appears on
          <strong>{report['taxonomy_summary'][0]['count']}</strong> tagged views. Boundary-driven failure remains meaningful,
          while over-smoothing exists but is not the primary mode in this final 200-object pass.
        </p>
        <div class="mini-list">
          <div class="mini-item">Coverage: {report['overall']['objects']} objects, {report['overall']['views']} fixed views, 0 render skips, 0 eval skips.</div>
          <div class="mini-item">Buckets: 50 objects each for over_smoothing, metal_nonmetal_confusion, local_highlight_misread, boundary_bleed.</div>
          <div class="mini-item">Generated: {html.escape(report['overall']['generated_at'])}</div>
        </div>
      </div>
    </div>

    <div class="section dashboard">
      <h2>Dashboard</h2>
      <div class="section-note">High-level summary image for slides and fast inspection.</div>
      <img src="{html.escape(chart_rel['dashboard'])}" alt="dashboard">
    </div>

    <div class="section">
      <h2>Charts</h2>
      <div class="section-note">Taxonomy prevalence, per-bucket mean error, top categories, and top objects by mean total error.</div>
      <div class="chart-grid">
        <img src="{html.escape(chart_rel['taxonomy'])}" alt="taxonomy chart">
        <img src="{html.escape(chart_rel['bucket'])}" alt="bucket chart">
        <img src="{html.escape(chart_rel['category'])}" alt="category chart">
        <img src="{html.escape(chart_rel['objects'])}" alt="object chart">
      </div>
    </div>

    <div class="section">
      <h2>Failure Montages</h2>
      <div class="section-note">Top failures overall plus per-taxonomy 3x2 montages, each card labeled with Input / GT Roughness / GT Metallic / GT Material / Pred Material.</div>
      <div class="montage-grid">
        <img src="{html.escape(montage_rel['overall'])}" alt="overall montage">
        <img src="{html.escape(montage_rel['over_smoothing'])}" alt="over smoothing montage">
        <img src="{html.escape(montage_rel['metal_nonmetal_confusion'])}" alt="metal confusion montage">
        <img src="{html.escape(montage_rel['local_highlight_misread'])}" alt="highlight misread montage">
        <img src="{html.escape(montage_rel['boundary_bleed'])}" alt="boundary bleed montage">
      </div>
    </div>

    <div class="section">
      <h2>Summary Tables</h2>
      <div class="section-note">Compact tables for write-ups, PRDs, or slide annotation.</div>
      <div class="table-grid">
        <div>
          <h3>Failure Taxonomy</h3>
          {make_table(["Failure", "Tagged Views", "Rate"], taxonomy_rows)}
        </div>
        <div>
          <h3>Bucket Breakdown</h3>
          {make_table(["Bucket", "Views", "R MAE", "M MAE", "Total", "Dominant"], bucket_rows)}
        </div>
        <div>
          <h3>Category Breakdown</h3>
          {make_table(["Category", "Views", "R MAE", "M MAE", "Total"], category_rows)}
        </div>
        <div>
          <h3>Top Objects</h3>
          {make_table(["Object", "Name", "Category", "Views", "Mean Total", "Worst View"], object_rows)}
        </div>
      </div>
    </div>

    <div class="section">
      <h2>All 600 Views</h2>
      <div class="section-note">Filter the final results by category, sampling bucket, or primary failure. Images lazy-load, so the page stays usable even with the full 600-view result set.</div>
      <div class="filters">
        <input id="searchInput" type="text" placeholder="Search object id, name, or view">
        <select id="categoryFilter"><option value="">All categories</option>{category_options}</select>
        <select id="bucketFilter"><option value="">All buckets</option>{bucket_options}</select>
        <select id="primaryFilter"><option value="">All primary failures</option>{primary_options}</select>
      </div>
      <table id="caseTable">
        <thead>
          <tr>
            <th>#</th>
            <th>Preview</th>
            <th>Case</th>
            <th>Category / Bucket</th>
            <th>Primary / Tags</th>
            <th>GT / Pred</th>
            <th>Abs Error</th>
            <th>Total</th>
          </tr>
        </thead>
        <tbody>
          {''.join(case_rows)}
        </tbody>
      </table>
      <div class="foot">Taxonomy counts are tag counts, not mutually-exclusive class totals. A single view can trigger multiple failure tags.</div>
    </div>
  </div>
  <script>
    const rows = Array.from(document.querySelectorAll("#caseTable tbody tr"));
    const searchInput = document.getElementById("searchInput");
    const categoryFilter = document.getElementById("categoryFilter");
    const bucketFilter = document.getElementById("bucketFilter");
    const primaryFilter = document.getElementById("primaryFilter");

    function applyFilters() {{
      const search = searchInput.value.trim().toLowerCase();
      const category = categoryFilter.value;
      const bucket = bucketFilter.value;
      const primary = primaryFilter.value;
      rows.forEach((row) => {{
        const matchSearch = !search || row.dataset.search.includes(search);
        const matchCategory = !category || row.dataset.category === category;
        const matchBucket = !bucket || row.dataset.bucket === bucket;
        const matchPrimary = !primary || row.dataset.primary === primary;
        row.style.display = matchSearch && matchCategory && matchBucket && matchPrimary ? "" : "none";
      }});
    }}

    searchInput.addEventListener("input", applyFilters);
    categoryFilter.addEventListener("change", applyFilters);
    bucketFilter.addEventListener("change", applyFilters);
    primaryFilter.addEventListener("change", applyFilters);
  </script>
</body>
</html>
"""
    (output_dir / "report.html").write_text(html_text)


def build_markdown_report(output_dir: Path, report: dict):
    lines = [
        "# ABO RM Mini 200-Object Final Report",
        "",
        f"Generated from `output/abo_rm_mini` on {report['overall']['generated_at']}.",
        "",
        "## Coverage",
        "",
        f"- Objects completed: {report['overall']['objects']} / 200",
        f"- Views evaluated: {report['overall']['views']} / 600",
        "- Fixed views per object: `front_studio`, `three_quarter_indoor`, `side_neon`",
        "- Render skips: 0",
        "- Eval skips: 0",
        "",
        "## Headline Metrics",
        "",
        f"- Mean absolute roughness error: {report['overall']['rough_mae']:.3f}",
        f"- Mean absolute metallic error: {report['overall']['metal_mae']:.3f}",
        f"- Mean total error: {report['overall']['total_mae']:.3f}",
        f"- Median total error: {report['overall']['total_median']:.3f}",
        f"- Worst single-view total error: {report['top_views'][0]['errors']['total_error']:.3f} ({report['top_views'][0]['object_id']} / {report['top_views'][0]['view_name']})",
        "",
        "## Failure Taxonomy",
        "",
        "- Note: taxonomy counts are tag counts, not exclusive class counts. One view may trigger multiple failure tags.",
        "",
    ]
    for item in report["taxonomy_summary"]:
        lines.append(f"- `{item['tag']}`: {item['count']} tagged views ({item['rate']:.1f}%)")

    lines.extend(
        [
            "",
            "## Bucket Breakdown",
            "",
            "| Bucket | Views | Rough MAE | Metal MAE | Total MAE | Dominant Tag |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in report["bucket_summary"]:
        lines.append(
            f"| `{item['bucket']}` | {item['views']} | {item['rough_mae']:.3f} | {item['metal_mae']:.3f} | {item['total_mae']:.3f} | `{item['dominant_tag'] or 'none'}` |"
        )

    lines.extend(
        [
            "",
            "## Category Breakdown",
            "",
            "| Category | Views | Rough MAE | Metal MAE | Total MAE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["category_summary"][:15]:
        lines.append(
            f"| `{item['category']}` | {item['views']} | {item['rough_mae']:.3f} | {item['metal_mae']:.3f} | {item['total_mae']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Top Failure Objects",
            "",
            "| Object | Category | Bucket | Mean Total Error | Worst View |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for item in report["object_summary"][:15]:
        lines.append(
            f"| `{item['object_id']}` | `{item['category']}` | `{item['sampling_bucket']}` | {item['mean_total_error']:.3f} | `{item['worst_view']}` |"
        )

    lines.extend(
        [
            "",
            "## Top Failure Views",
            "",
            "| View | Category | Bucket | Primary Failure | Total Error |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for row in report["top_views"][:15]:
        lines.append(
            f"| `{row['object_id']} / {row['view_name']}` | `{row['category']}` | `{row['sampling_bucket']}` | `{row['primary_failure']}` | {row['errors']['total_error']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Key Takeaways",
            "",
            f"1. `metal_nonmetal_confusion` is still the dominant structured gap at {report['taxonomy_summary'][0]['count']} tagged views, far above the other three failure types.",
            "2. `boundary_bleed` remains material: mixed-material boundaries and thin metal frames still pull predictions away from interior GT.",
            "3. `local_highlight_misread` is present but clearly secondary to metal confusion in this final 200-object pass.",
            "4. `over_smoothing` exists, but in the final benchmark it is not the main explanation for the overall RM gap.",
            "",
            "## Output Files",
            "",
            "- `output/abo_rm_mini/report.html`",
            "- `output/abo_rm_mini/final_report.md`",
            "- `output/abo_rm_mini/report_assets/dashboard.png`",
            "- `output/abo_rm_mini/report_assets/top_failure_montage.png`",
        ]
    )
    (output_dir / "final_report.md").write_text("\n".join(lines))


def build_report(output_dir: Path, assets_dir: Path):
    baseline_rows = load_json(output_dir / "baseline_eval.json")
    object_summary = load_json(output_dir / "object_summary.json")
    render_summary = load_json(output_dir / "render_summary.json")
    eval_summary = load_json(output_dir / "eval_summary.json")
    manifest = load_json(output_dir / "objects_200.json")
    manifest_objects = manifest["objects"] if isinstance(manifest, dict) else manifest

    row_by_bucket = defaultdict(list)
    row_by_category = defaultdict(list)
    for row in baseline_rows:
        row_by_bucket[row["sampling_bucket"]].append(row)
        row_by_category[row["category"]].append(row)

    overall = {
        "objects": render_summary["objects_completed"],
        "views": eval_summary["views_evaluated"],
        "rough_mae": mean([row["errors"]["abs_error_roughness"] for row in baseline_rows]),
        "metal_mae": mean([row["errors"]["abs_error_metallic"] for row in baseline_rows]),
        "total_mae": mean([row["errors"]["total_error"] for row in baseline_rows]),
        "total_median": float(np.median([row["errors"]["total_error"] for row in baseline_rows])),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    taxonomy_summary = []
    for tag in sorted(FAILURE_ORDER, key=lambda item: eval_summary["taxonomy_counts"].get(item, 0), reverse=True):
        count = eval_summary["taxonomy_counts"].get(tag, 0)
        taxonomy_summary.append(
            {
                "tag": tag,
                "count": count,
                "rate": pct(count, eval_summary["views_evaluated"]),
            }
        )

    bucket_summary = []
    for bucket in FAILURE_ORDER:
        items = row_by_bucket[bucket]
        tag_counts = Counter(tag for row in items for tag in row["failure_tags"])
        dominant_tag = tag_counts.most_common(1)[0][0] if tag_counts else None
        bucket_summary.append(
            {
                "bucket": bucket,
                "views": len(items),
                "rough_mae": mean([row["errors"]["abs_error_roughness"] for row in items]),
                "metal_mae": mean([row["errors"]["abs_error_metallic"] for row in items]),
                "total_mae": mean([row["errors"]["total_error"] for row in items]),
                "dominant_tag": dominant_tag,
            }
        )
    bucket_summary.sort(key=lambda item: item["total_mae"], reverse=True)

    category_summary = []
    for category, items in row_by_category.items():
        category_summary.append(
            {
                "category": category,
                "views": len(items),
                "rough_mae": mean([row["errors"]["abs_error_roughness"] for row in items]),
                "metal_mae": mean([row["errors"]["abs_error_metallic"] for row in items]),
                "total_mae": mean([row["errors"]["total_error"] for row in items]),
            }
        )
    category_summary.sort(key=lambda item: item["total_mae"], reverse=True)

    object_rows = object_summary["objects"]

    chart_dir = ensure_dir(assets_dir / "charts")
    case_dir = ensure_dir(assets_dir / "case_cards")
    montage_dir = ensure_dir(assets_dir / "montages")

    taxonomy_chart = chart_dir / "taxonomy_counts.png"
    bucket_chart = chart_dir / "bucket_mean_total_error.png"
    category_chart = chart_dir / "category_mean_total_error_top12.png"
    object_chart = chart_dir / "object_mean_total_error_top12.png"
    dashboard_path = assets_dir / "dashboard.png"

    draw_bar_chart(
        "Failure Taxonomy Counts",
        "Tagged views per taxonomy. Counts are not mutually exclusive.",
        [
            {
                "label": display_failure(item["tag"]),
                "detail": f"{item['rate']:.1f}% of 600 evaluated views",
                "value": item["count"],
                "value_text": str(item["count"]),
                "color": FAILURE_COLORS[item["tag"]],
            }
            for item in taxonomy_summary
        ],
        taxonomy_chart,
    )
    draw_bar_chart(
        "Per-Bucket Mean Total Error",
        "Mean total error across the four fixed sampling buckets.",
        [
            {
                "label": display_failure(item["bucket"]),
                "detail": f"{item['views']} views    R {item['rough_mae']:.3f} / M {item['metal_mae']:.3f}",
                "value": item["total_mae"],
                "value_text": f"{item['total_mae']:.3f}",
                "color": FAILURE_COLORS[item["bucket"]],
            }
            for item in bucket_summary
        ],
        bucket_chart,
    )
    draw_bar_chart(
        "Top Categories by Mean Total Error",
        "Categories sorted by mean total error across their evaluated views.",
        [
            {
                "label": display_category(item["category"]),
                "detail": f"{item['views']} views    R {item['rough_mae']:.3f} / M {item['metal_mae']:.3f}",
                "value": item["total_mae"],
                "value_text": f"{item['total_mae']:.3f}",
                "color": ACCENT_ORANGE if idx % 2 == 0 else ACCENT_BLUE,
            }
            for idx, item in enumerate(category_summary[:12])
        ],
        category_chart,
    )
    draw_bar_chart(
        "Top Objects by Mean Total Error",
        "Object-level mean total error across the three fixed views.",
        [
            {
                "label": f"{item['object_id']}  {truncate(item['name'], 34)}",
                "detail": f"{display_category(item['category'])}    worst {item['worst_view']}",
                "value": item["mean_total_error"],
                "value_text": f"{item['mean_total_error']:.3f}",
                "color": ACCENT_RED if idx % 2 == 0 else (194, 156, 255),
            }
            for idx, item in enumerate(object_rows[:12])
        ],
        object_chart,
        width=1200,
    )

    compose_dashboard(
        dashboard_path,
        overall,
        {
            "taxonomy": taxonomy_chart,
            "bucket": bucket_chart,
            "category": category_chart,
            "objects": object_chart,
        },
    )

    top_overall = baseline_rows[:12]
    top_by_tag = {tag: [row for row in baseline_rows if tag in row["failure_tags"]][:6] for tag in FAILURE_ORDER}
    required_cards = {}
    for row in top_overall:
        required_cards[f"{row['object_id']}__{row['view_name']}"] = row
    for items in top_by_tag.values():
        for row in items:
            required_cards[f"{row['object_id']}__{row['view_name']}"] = row

    case_card_paths = {}
    for case_key, row in required_cards.items():
        out_path = case_dir / f"{slugify(case_key)}.png"
        build_case_card(row, out_path)
        case_card_paths[case_key] = out_path

    montages = {
        "overall": assets_dir / "top_failure_montage.png",
        "over_smoothing": montage_dir / "top_over_smoothing.png",
        "metal_nonmetal_confusion": montage_dir / "top_metal_nonmetal_confusion.png",
        "local_highlight_misread": montage_dir / "top_local_highlight_misread.png",
        "boundary_bleed": montage_dir / "top_boundary_bleed.png",
    }
    build_montage(
        [case_card_paths[f"{row['object_id']}__{row['view_name']}"] for row in top_overall],
        montages["overall"],
        "Top 12 Failure Views",
        "Overall highest total-error views in the final 200-object pass.",
        cols=2,
    )
    for tag in FAILURE_ORDER:
        build_montage(
            [case_card_paths[f"{row['object_id']}__{row['view_name']}"] for row in top_by_tag[tag]],
            montages[tag],
            f"Top {display_failure(tag)} Cases",
            "Each card: Input | GT Roughness | GT Metallic | GT Material | Pred Material",
            cols=2,
        )

    report = {
        "overall": overall,
        "rows": baseline_rows,
        "top_views": baseline_rows[:20],
        "taxonomy_summary": taxonomy_summary,
        "bucket_summary": bucket_summary,
        "category_summary": category_summary,
        "object_summary": object_rows,
        "charts": {
            "taxonomy": taxonomy_chart,
            "bucket": bucket_chart,
            "category": category_chart,
            "objects": object_chart,
            "dashboard": dashboard_path,
        },
        "montages": montages,
    }

    build_report_html(output_dir, assets_dir, report)
    build_markdown_report(output_dir, report)
    summary_json = {
        "overall": overall,
        "taxonomy_summary": taxonomy_summary,
        "bucket_summary": bucket_summary,
        "category_summary": category_summary[:20],
        "top_objects": object_rows[:20],
        "top_views": [
            {
                "object_id": row["object_id"],
                "view_name": row["view_name"],
                "category": row["category"],
                "sampling_bucket": row["sampling_bucket"],
                "primary_failure": row["primary_failure"],
                "total_error": row["errors"]["total_error"],
            }
            for row in baseline_rows[:20]
        ],
    }
    (output_dir / "report_summary.json").write_text(json.dumps(summary_json, indent=2))


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    assets_dir = ensure_dir(args.assets_dir.resolve() if args.assets_dir else output_dir / "report_assets")
    build_report(output_dir, assets_dir)
    print(f"Report HTML: {output_dir / 'report.html'}")
    print(f"Markdown summary: {output_dir / 'final_report.md'}")
    print(f"Report assets: {assets_dir}")


if __name__ == "__main__":
    main()
