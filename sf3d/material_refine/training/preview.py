from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sf3d.material_refine.io import tensor_to_pil
from .common import format_metric

def grayscale_rgb_image(tensor: torch.Tensor) -> Image.Image:
    return tensor_to_pil(tensor.detach().cpu(), grayscale=True).convert("RGB")


def enhanced_grayscale_rgb_image(
    tensor: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    gamma: float = 0.65,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> Image.Image:
    value = tensor.detach().cpu().float().clamp(0.0, 1.0).squeeze(0).numpy()
    valid = np.ones_like(value, dtype=bool)
    if mask is not None:
        valid = mask.detach().cpu().float().squeeze(0).numpy() > 0.05
    visible_values = value[valid]
    if visible_values.size >= 8:
        low = float(np.percentile(visible_values, lower_percentile))
        high = float(np.percentile(visible_values, upper_percentile))
        if high > low + 1e-5:
            value = (value - low) / (high - low)
    value = np.clip(value, 0.0, 1.0)
    value = np.power(value, max(float(gamma), 1e-4))
    if mask is not None:
        value = value * valid.astype(np.float32)
    data = (value * 255.0).round().astype(np.uint8)
    return Image.fromarray(np.stack([data, data, data], axis=-1), mode="RGB")


def enhanced_rgb_tensor_image(
    tensor: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    gamma: float = 0.65,
) -> Image.Image:
    image = tensor.detach().cpu().float()[:3].clamp(0.0, 1.0).numpy()
    valid = np.ones(image.shape[1:], dtype=bool)
    if mask is not None:
        valid = mask.detach().cpu().float().squeeze(0).numpy() > 0.05
    for channel in range(image.shape[0]):
        channel_values = image[channel]
        visible_values = channel_values[valid]
        if visible_values.size >= 8:
            low = float(np.percentile(visible_values, 2.0))
            high = float(np.percentile(visible_values, 98.0))
            if high > low + 1e-5:
                image[channel] = (channel_values - low) / (high - low)
    image = np.power(np.clip(image, 0.0, 1.0), max(float(gamma), 1e-4))
    if mask is not None:
        image = image * valid[None, :, :].astype(np.float32)
    data = (np.moveaxis(image, 0, -1) * 255.0).round().astype(np.uint8)
    return Image.fromarray(data, mode="RGB")


def rgb_tensor_image(
    tensor: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> Image.Image:
    image = tensor_to_pil(tensor.detach().cpu()[:3]).convert("RGB")
    if mask is None:
        return image
    mask_np = mask.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()[..., None]
    image_np = np.asarray(image).astype(np.float32)
    background = np.array([20.0, 24.0, 30.0], dtype=np.float32)
    blended = image_np * mask_np + background * (1.0 - mask_np)
    return Image.fromarray(blended.clip(0.0, 255.0).round().astype(np.uint8), mode="RGB")


def delta_heatmap_image(
    reference: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> Image.Image:
    ref_np = reference.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()
    tgt_np = target.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()
    delta = np.abs(ref_np - tgt_np)
    if mask is not None:
        delta = delta * mask.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()
    delta = np.clip(delta * 3.5, 0.0, 1.0)
    heat = np.zeros((*delta.shape, 3), dtype=np.uint8)
    heat[..., 0] = (delta * 255.0).round().astype(np.uint8)
    heat[..., 1] = (np.sqrt(delta) * 170.0).round().astype(np.uint8)
    heat[..., 2] = ((1.0 - delta) * 45.0).round().astype(np.uint8)
    return Image.fromarray(heat, mode="RGB")


def preview_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    font_names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (14, 18, 24),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    draw.text(
        (x0 + (x1 - x0 - text_width) / 2, y0 + (y1 - y0 - text_height) / 2),
        text,
        font=font,
        fill=fill,
    )


def compact_labeled_image(
    image: Image.Image,
    label: str,
    *,
    size: int = 176,
    header_height: int = 34,
) -> Image.Image:
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + header_height), (247, 249, 252))
    tile.paste(image, (0, header_height))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size - 1, header_height - 1), fill=(239, 243, 248))
    draw_text_centered(
        draw,
        (0, 0, size, header_height),
        label,
        font=preview_font(18, bold=True),
    )
    return tile


def labeled_tile(
    image: Image.Image,
    label: str,
    detail: str | None = None,
    *,
    size: int = 224,
) -> Image.Image:
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    header_height = 36 if detail else 24
    tile = Image.new("RGB", (image.width, image.height + header_height), (245, 247, 250))
    tile.paste(image, (0, header_height))
    draw = ImageDraw.Draw(tile)
    draw.text((6, 4), label, fill=(16, 18, 22))
    if detail:
        draw.text((6, 18), detail, fill=(82, 90, 102))
    return tile


def select_preview_view_indices(view_names: list[str], *, max_views: int = 4) -> list[int]:
    preferred_tokens = [
        "front_mid",
        "three_quarter_mid",
        "grazing_left",
        "thin_boundary_close",
        "front_high",
        "grazing_right",
        "top_oblique",
        "side_low",
    ]
    selected: list[int] = []
    for token in preferred_tokens:
        for index, name in enumerate(view_names):
            if index in selected:
                continue
            if str(name).startswith(token):
                selected.append(index)
                break
        if len(selected) >= max_views:
            return selected[:max_views]
    for index, _name in enumerate(view_names):
        if index not in selected:
            selected.append(index)
        if len(selected) >= max_views:
            break
    return selected[:max_views]


def preview_view_label(view_name: str) -> str:
    return str(view_name).split("__", 1)[0].replace("_", " ")


def sanitize_preview_filename_component(value: Any, *, default: str = "unknown", max_length: int = 96) -> str:
    text = default if value is None else str(value).strip()
    if not text:
        text = default
    safe_chars = []
    last_was_separator = False
    for char in text:
        if char.isascii() and (char.isalnum() or char in {"-", "_", "."}):
            safe_chars.append(char)
            last_was_separator = False
        elif not last_was_separator:
            safe_chars.append("_")
            last_was_separator = True
    safe = "".join(safe_chars).strip("._")
    if not safe:
        safe = default
    return safe[:max_length].strip("._") or default


def first_preview_identity(*values: Any) -> str:
    for value in values:
        text = "" if value is None else str(value).strip()
        if text and text.lower() not in {"none", "null", "unknown"}:
            return text
    return "unknown"


PRIOR_VARIANT_DISPLAY_ORDER = [
    "large_gap_prior",
    "medium_gap_prior",
    "mild_gap_prior",
    "near_gt_prior",
    "no_prior_bootstrap",
]


def prior_variant_distribution(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("prior_variant_type") or "unknown") for item in items)
    ordered: dict[str, int] = {}
    for variant in PRIOR_VARIANT_DISPLAY_ORDER:
        if counts.get(variant):
            ordered[variant] = int(counts[variant])
    for variant in sorted(counts):
        if variant not in ordered:
            ordered[variant] = int(counts[variant])
    return ordered


def format_prior_distribution(distribution: dict[str, int] | None) -> str:
    if not distribution:
        return "unknown"
    return ", ".join(f"{variant}={count}" for variant, count in distribution.items())


def _fit_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    candidate = text
    while candidate and draw.textbbox((0, 0), candidate + suffix, font=font)[2] > max_width:
        candidate = candidate[:-1]
    return (candidate.rstrip() + suffix) if candidate else suffix


def append_prior_distribution_footer(image_path: Path, distribution_text: str) -> None:
    if not distribution_text:
        return
    image = Image.open(image_path).convert("RGB")
    footer_height = 28
    canvas = Image.new("RGB", (image.width, image.height + footer_height), (255, 255, 255))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, image.height, image.width, image.height + footer_height), fill=(239, 243, 248))
    footer_font = preview_font(13)
    text = _fit_text_to_width(
        draw,
        f"selected prior distribution: {distribution_text}",
        font=footer_font,
        max_width=image.width - 24,
    )
    draw.text((12, image.height + 7), text, font=footer_font, fill=(54, 63, 78))
    canvas.save(image_path)


def preview_effect_bucket(gain_total: float) -> str:
    if gain_total > 0.02:
        return "positive_gain"
    if gain_total < -0.02:
        return "regression"
    return "near_zero"


def should_select_validation_preview(
    *,
    mode: str,
    selected_count: int,
    max_count: int,
    prior_variant_type: str,
    effect_bucket: str,
    seen_variants: set[str],
    seen_effects: set[str],
) -> bool:
    if selected_count >= max_count:
        return False
    if mode == "first":
        return True
    if mode == "balanced_by_variant":
        return True
    if selected_count < max(1, min(max_count, 2)):
        return True
    if mode == "balanced":
        return prior_variant_type not in seen_variants
    return effect_bucket not in seen_effects or prior_variant_type not in seen_variants


def _preview_item_unique_key(item: dict[str, Any]) -> str:
    for key in ("case_id", "path", "pair_id", "prior_variant_id", "object_id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return f"item_{id(item)}"


def _preview_item_gain(item: dict[str, Any]) -> float:
    value = item.get("gain_total", item.get("improvement_total", 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _per_variant_slot_plan(slot_count: int) -> tuple[int, int, int]:
    if slot_count <= 0:
        return (0, 0, 0)
    if slot_count == 1:
        return (1, 0, 0)
    if slot_count == 2:
        return (1, 1, 0)
    if slot_count == 3:
        return (1, 1, 1)
    if slot_count == 4:
        return (2, 1, 1)
    if slot_count == 5:
        return (2, 1, 2)
    improved = 2
    near_zero = 2
    regressed = 2
    remaining = slot_count - 6
    for index in range(remaining):
        bucket = index % 3
        if bucket == 0:
            near_zero += 1
        elif bucket == 1:
            improved += 1
        else:
            regressed += 1
    return (improved, near_zero, regressed)


def select_variant_balanced_records(
    records: list[dict[str, Any]],
    *,
    max_count: int,
    mode: str = "balanced_by_variant",
) -> list[dict[str, Any]]:
    unique_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = _preview_item_unique_key(record)
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(record)
    if max_count <= 0 or len(unique_records) <= max_count:
        return unique_records
    if mode == "first":
        return unique_records[:max_count]
    if mode == "effect_showcase":
        mode = "balanced_by_variant"
    if mode == "random_balanced_by_variant":
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in unique_records:
            grouped[str(record.get("prior_variant_type", "unknown") or "unknown")].append(record)
        ordered_variants = sorted(grouped)
        variant_count = max(len(ordered_variants), 1)
        base_quota = max_count // variant_count
        remainder = max_count % variant_count
        rng = np.random.default_rng(42)
        per_variant_selected: dict[str, list[dict[str, Any]]] = {}
        leftovers: dict[str, list[dict[str, Any]]] = {}
        for index, variant in enumerate(ordered_variants):
            quota = base_quota + (1 if index < remainder else 0)
            rows = list(grouped[variant])
            rng.shuffle(rows)
            per_variant_selected[variant] = rows[:quota]
            leftovers[variant] = rows[quota:]
        ordered_selection: list[dict[str, Any]] = []
        max_depth = max((len(items) for items in per_variant_selected.values()), default=0)
        for depth in range(max_depth):
            for variant in ordered_variants:
                items = per_variant_selected[variant]
                if depth < len(items):
                    ordered_selection.append(items[depth])
                    if len(ordered_selection) >= max_count:
                        return ordered_selection
        for variant in ordered_variants:
            for item in leftovers[variant]:
                ordered_selection.append(item)
                if len(ordered_selection) >= max_count:
                    return ordered_selection
        return ordered_selection

    if mode != "balanced_by_variant":
        if mode == "balanced":
            by_variant_first: list[dict[str, Any]] = []
            seen_variants: set[str] = set()
            leftovers: list[dict[str, Any]] = []
            for record in unique_records:
                variant = str(record.get("prior_variant_type", "unknown") or "unknown")
                if variant not in seen_variants:
                    seen_variants.add(variant)
                    by_variant_first.append(record)
                else:
                    leftovers.append(record)
            return (by_variant_first + leftovers)[:max_count]
        return unique_records[:max_count]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in unique_records:
        grouped[str(record.get("prior_variant_type", "unknown") or "unknown")].append(record)
    ordered_variants = sorted(grouped)
    variant_count = max(len(ordered_variants), 1)
    base_quota = max_count // variant_count
    remainder = max_count % variant_count

    per_variant_selected: dict[str, list[dict[str, Any]]] = {}
    leftovers: dict[str, list[dict[str, Any]]] = {}
    for index, variant in enumerate(ordered_variants):
        quota = base_quota + (1 if index < remainder else 0)
        rows = grouped[variant]
        if quota <= 0:
            per_variant_selected[variant] = []
            leftovers[variant] = rows
            continue
        improved_n, near_n, regressed_n = _per_variant_slot_plan(quota)
        improved = sorted(rows, key=_preview_item_gain, reverse=True)
        near_zero = sorted(rows, key=lambda item: abs(_preview_item_gain(item)))
        regressed = sorted(rows, key=_preview_item_gain)
        selected: list[dict[str, Any]] = []
        selected_keys: set[str] = set()

        for candidates, count in (
            (improved, improved_n),
            (near_zero, near_n),
            (regressed, regressed_n),
        ):
            current_len = len(selected)
            for item in candidates:
                if len(selected) - current_len >= count or len(selected) >= quota:
                    break
                key = _preview_item_unique_key(item)
                if key in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(key)
        if len(selected) < quota:
            for item in improved:
                if len(selected) >= quota:
                    break
                key = _preview_item_unique_key(item)
                if key in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(key)
        per_variant_selected[variant] = selected
        leftovers[variant] = [item for item in rows if _preview_item_unique_key(item) not in selected_keys]

    ordered_selection: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    max_depth = max((len(items) for items in per_variant_selected.values()), default=0)
    for depth in range(max_depth):
        for variant in ordered_variants:
            items = per_variant_selected[variant]
            if depth >= len(items):
                continue
            item = items[depth]
            key = _preview_item_unique_key(item)
            if key in seen_keys:
                continue
            ordered_selection.append(item)
            seen_keys.add(key)
            if len(ordered_selection) >= max_count:
                return ordered_selection
    if len(ordered_selection) < max_count:
        for variant in ordered_variants:
            for item in sorted(leftovers[variant], key=_preview_item_gain, reverse=True):
                key = _preview_item_unique_key(item)
                if key in seen_keys:
                    continue
                ordered_selection.append(item)
                seen_keys.add(key)
                if len(ordered_selection) >= max_count:
                    return ordered_selection
    return ordered_selection


def _preview_output_path(
    preview_dir: Path,
    *,
    object_id: str,
    prior_variant_type: str,
    pair_id: str,
    prior_variant_id: str,
    preview_slot: int,
) -> Path:
    pair_or_prior_variant_id = first_preview_identity(pair_id, prior_variant_id)
    safe_preview_slot = max(int(preview_slot), 0)
    safe_object_id = sanitize_preview_filename_component(object_id)
    safe_prior_variant_type = sanitize_preview_filename_component(prior_variant_type)
    safe_pair_or_prior_variant_id = sanitize_preview_filename_component(
        pair_or_prior_variant_id,
        max_length=128,
    )
    return preview_dir / (
        f"{safe_preview_slot:03d}__{safe_object_id}__{safe_prior_variant_type}__"
        f"{safe_pair_or_prior_variant_id}.png"
    )


def finalize_selected_preview_items(
    output_dir: Path,
    *,
    validation_label: str,
    preview_items: list[dict[str, Any]],
    mode: str,
    max_count: int,
) -> list[dict[str, Any]]:
    selected_items = select_variant_balanced_records(
        preview_items,
        max_count=max_count,
        mode=mode,
    )
    distribution = prior_variant_distribution(selected_items)
    distribution_text = format_prior_distribution(distribution)
    preview_dir = output_dir / "validation_previews" / validation_label
    candidate_dir = preview_dir / "_candidates"
    for slot, item in enumerate(selected_items):
        item["selected_prior_distribution"] = dict(distribution)
        item["selected_prior_distribution_text"] = distribution_text
        source_path = Path(str(item.get("path") or ""))
        if not source_path.exists():
            continue
        target_path = _preview_output_path(
            preview_dir,
            object_id=str(item.get("object_id") or "unknown"),
            prior_variant_type=str(item.get("prior_variant_type") or "unknown"),
            pair_id=str(item.get("pair_id") or ""),
            prior_variant_id=str(item.get("prior_variant_id") or ""),
            preview_slot=slot,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()
        source_path.replace(target_path)
        append_prior_distribution_footer(target_path, distribution_text)
        item["path"] = str(target_path.resolve())
        item["preview_slot"] = slot
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir, ignore_errors=True)
    return selected_items


def save_preview_contact_sheet(
    output_dir: Path,
    *,
    validation_label: str,
    preview_paths: list[str],
) -> Path | None:
    if not preview_paths:
        return None
    images = [Image.open(path).convert("RGB") for path in preview_paths]
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    columns = 2 if len(images) > 1 else 1
    rows = math.ceil(len(images) / columns)
    header_height = 42
    gutter = 14
    canvas = Image.new(
        "RGB",
        (
            columns * tile_width + gutter * (columns + 1),
            header_height + rows * tile_height + gutter * (rows + 1),
        ),
        (10, 15, 22),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), f"{validation_label} validation previews", fill=(236, 242, 248))
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        x0 = gutter + col * (tile_width + gutter)
        y0 = header_height + gutter + row * (tile_height + gutter)
        if image.size != (tile_width, tile_height):
            image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        canvas.paste(image, (x0, y0))
    preview_dir = output_dir / "validation_previews" / validation_label
    preview_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet_path = preview_dir / "_contact_sheet.png"
    canvas.save(contact_sheet_path)
    return contact_sheet_path


def save_validation_preview(
    output_dir: Path,
    *,
    validation_label: str,
    object_id: str,
    baseline: torch.Tensor,
    refined: torch.Tensor,
    uv_albedo: torch.Tensor,
    uv_normal: torch.Tensor,
    target_roughness: torch.Tensor,
    target_metallic: torch.Tensor,
    confidence: torch.Tensor,
    view_features: torch.Tensor,
    view_masks: torch.Tensor,
    view_names: list[str],
    baseline_roughness_mae: float,
    baseline_metallic_mae: float,
    refined_roughness_mae: float,
    refined_metallic_mae: float,
    pair_id: str = "",
    target_bundle_id: str = "",
    prior_variant_id: str = "",
    prior_variant_type: str = "unknown",
    prior_quality_bin: str = "unknown",
    training_role: str = "unknown",
    view_rm_mae_delta: float | None = None,
    preview_slot: int | None = 0,
    source_name: str = "unknown_source",
    material_family: str = "unknown_material",
    prior_label: str = "unknown_prior",
) -> Path:
    preview_root = output_dir / "validation_previews" / validation_label
    preview_dir = preview_root if preview_slot is not None else preview_root / "_candidates"
    preview_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = select_preview_view_indices(view_names, max_views=1)
    preview_view_index = selected_indices[0] if selected_indices else 0
    preview_view = rgb_tensor_image(
        view_features[preview_view_index, 0:3],
        mask=view_masks[preview_view_index],
    )
    preview_view_enhanced = enhanced_rgb_tensor_image(
        view_features[preview_view_index, 0:3],
        mask=view_masks[preview_view_index],
    )

    tile_size = 128
    row_label_width = 90
    gutter = 8
    title_height = 94
    columns = 7
    tile_width = tile_size
    tile_height = tile_size + 34
    canvas_width = row_label_width + columns * tile_width + (columns + 1) * gutter
    rows = [
        (
            "Input",
            [
                compact_labeled_image(preview_view, "View", size=tile_size),
                compact_labeled_image(preview_view_enhanced, "View γ", size=tile_size),
                compact_labeled_image(rgb_tensor_image(uv_albedo), "Albedo", size=tile_size),
                compact_labeled_image(enhanced_rgb_tensor_image(uv_albedo), "Albedo γ", size=tile_size),
                compact_labeled_image(rgb_tensor_image(uv_normal), "Normal", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(confidence), "Conf", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(confidence), "Conf γ", size=tile_size),
            ],
        ),
        (
            "Rough",
            [
                compact_labeled_image(grayscale_rgb_image(baseline[0:1]), "Input Prior", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(baseline[0:1], mask=confidence), "Prior γ", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(target_roughness), "GT", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(target_roughness, mask=confidence), "GT γ", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(refined[0:1]), "Pred", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(refined[0:1], mask=confidence), "Pred γ", size=tile_size),
                compact_labeled_image(delta_heatmap_image(refined[0:1], target_roughness), "Err", size=tile_size),
            ],
        ),
        (
            "Metal",
            [
                compact_labeled_image(grayscale_rgb_image(baseline[1:2]), "Input Prior", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(baseline[1:2], mask=confidence), "Prior γ", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(target_metallic), "GT", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(target_metallic, mask=confidence), "GT γ", size=tile_size),
                compact_labeled_image(grayscale_rgb_image(refined[1:2]), "Pred", size=tile_size),
                compact_labeled_image(enhanced_grayscale_rgb_image(refined[1:2], mask=confidence), "Pred γ", size=tile_size),
                compact_labeled_image(delta_heatmap_image(refined[1:2], target_metallic), "Err", size=tile_size),
            ],
        ),
    ]
    canvas_height = title_height + len(rows) * tile_height + (len(rows) + 1) * gutter
    canvas = Image.new("RGB", size=(canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    baseline_total = baseline_roughness_mae + baseline_metallic_mae
    refined_total = refined_roughness_mae + refined_metallic_mae
    improvement = baseline_total - refined_total
    title_font = preview_font(21, bold=True)
    detail_font = preview_font(15)
    id_font = preview_font(12)
    draw.text((12, 10), str(object_id), font=title_font, fill=(8, 12, 18))
    draw.text(
        (12, 38),
        (
            f"Input Prior {baseline_total:.4f} | Pred {refined_total:.4f} | "
            f"UV gain {improvement:+.4f} | view RM delta {format_metric(view_rm_mae_delta, 4)} | {prior_label} | "
            f"variant={prior_variant_type} | quality={prior_quality_bin} | role={training_role}"
        ),
        font=detail_font,
        fill=(68, 76, 88),
    )
    identity_line = (
        f"pair={pair_id or 'unknown'} | target={target_bundle_id or 'unknown'} | "
        f"prior_variant={prior_variant_id or 'unknown'}"
    )
    identity_detail_line = f"{identity_line} | {material_family} | {source_name}"
    title_line_width = canvas_width - 24
    detail_bbox = draw.textbbox((12, 62), identity_detail_line, font=id_font)
    id_bbox = draw.textbbox((12, 62), identity_line, font=id_font)
    if detail_bbox[2] - detail_bbox[0] <= title_line_width:
        draw.text((12, 62), identity_detail_line, font=id_font, fill=(94, 104, 118))
    elif id_bbox[2] - id_bbox[0] <= title_line_width:
        draw.text((12, 62), identity_line, font=id_font, fill=(94, 104, 118))
    row_font = preview_font(20, bold=True)
    for row_index, (row_label, row_tiles) in enumerate(rows):
        y0 = title_height + gutter + row_index * (tile_height + gutter)
        draw_text_centered(
            draw,
            (0, y0, row_label_width, y0 + tile_height),
            row_label,
            font=row_font,
            fill=(24, 32, 44),
        )
        for col_index, tile in enumerate(row_tiles):
            x0 = row_label_width + gutter + col_index * (tile_width + gutter)
            canvas.paste(tile, (x0, y0))
    if preview_slot is None:
        case_id = sanitize_preview_filename_component(
            first_preview_identity(pair_id, prior_variant_id, object_id),
            max_length=128,
        )
        safe_object_id = sanitize_preview_filename_component(object_id)
        safe_prior_variant_type = sanitize_preview_filename_component(prior_variant_type)
        output_path = preview_dir / f"{safe_object_id}__{safe_prior_variant_type}__{case_id}.png"
    else:
        output_path = _preview_output_path(
            preview_root,
            object_id=object_id,
            prior_variant_type=prior_variant_type,
            pair_id=pair_id,
            prior_variant_id=prior_variant_id,
            preview_slot=preview_slot,
        )
    canvas.save(output_path)
    return output_path


def build_preview_integrity_report(preview_items: list[dict[str, Any]]) -> dict[str, Any]:
    required_metadata = [
        "case_id",
        "object_id",
        "pair_id",
        "target_bundle_id",
        "prior_variant_id",
        "prior_variant_type",
        "prior_quality_bin",
        "training_role",
    ]
    paths = [str(item.get("path") or "") for item in preview_items]
    path_counts = Counter(path for path in paths if path)
    duplicate_paths = {
        path: count
        for path, count in sorted(path_counts.items())
        if count > 1
    }
    missing_files = [
        path
        for path in paths
        if path and not Path(path).exists()
    ]
    missing_metadata_items = []
    for index, item in enumerate(preview_items):
        missing_fields = [
            field
            for field in required_metadata
            if field not in item or str(item.get(field) or "").strip() == ""
        ]
        if missing_fields:
            missing_metadata_items.append(
                {
                    "index": index,
                    "path": str(item.get("path") or ""),
                    "object_id": str(item.get("object_id") or ""),
                    "missing_fields": missing_fields,
                }
            )
    return {
        "enabled": True,
        "ok": not duplicate_paths and not missing_files and not missing_metadata_items,
        "items": len(preview_items),
        "unique_paths": len(path_counts),
        "duplicate_paths": duplicate_paths,
        "missing_files": missing_files,
        "required_metadata_fields": required_metadata,
        "missing_metadata_items": missing_metadata_items,
    }
