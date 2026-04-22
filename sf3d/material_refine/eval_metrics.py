from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter

from sf3d.material_refine.experiment import make_json_serializable
from sf3d.material_refine.io import tensor_to_pil

METAL_THRESHOLD = 0.20
GROUP_DIAGNOSTIC_MIN_COUNT = 16
MAIN_METRIC_NAMES = [
    "uv_rm_mae",
    "view_rm_mae",
    "proxy_render_psnr",
    "proxy_render_ssim",
    "proxy_render_lpips",
]
MATERIAL_SPECIFIC_METRIC_NAMES = [
    "boundary_bleed_score",
    "metal_nonmetal_confusion",
    "highlight_localization_error",
    "rm_gradient_preservation",
    "prior_residual_safety",
    "confidence_calibrated_error",
    "material_family_breakdown",
]


def confidence_weighted_mean(
    value: torch.Tensor,
    confidence: torch.Tensor,
) -> float:
    weight = float(confidence.sum().item())
    if weight <= 0.0:
        return float(value.mean().item())
    return float((value * confidence).sum().item() / weight)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def optional_mean(values: list[Any]) -> float | None:
    finite_values = [finite_or_none(value) for value in values]
    finite_values = [value for value in finite_values if value is not None]
    if not finite_values:
        return None
    return float(np.mean(finite_values))


def optional_delta(
    baseline: float | None,
    refined: float | None,
    *,
    higher_is_better: bool,
) -> float | None:
    if baseline is None or refined is None:
        return None
    return float(refined - baseline if higher_is_better else baseline - refined)


def metric_pair(
    *,
    baseline: float | None,
    refined: float | None,
    higher_is_better: bool,
    count: int,
    mode: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "baseline": baseline,
        "refined": refined,
        "delta": optional_delta(
            baseline,
            refined,
            higher_is_better=higher_is_better,
        ),
        "higher_is_better": higher_is_better,
        "available_count": int(count),
    }
    if mode:
        payload["mode"] = mode
    return payload


def masked_mean_np(values: np.ndarray, mask: np.ndarray) -> float | None:
    visible = values[mask]
    if visible.size == 0:
        return None
    return float(visible.mean())


def compute_masked_psnr(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    visible = mask.astype(bool)
    if prediction.shape != target.shape or visible.sum() == 0:
        return None
    diff = prediction - target
    if diff.ndim == 3:
        visible = np.broadcast_to(visible[..., None], diff.shape)
    mse = float(np.mean(np.square(diff[visible])))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def compute_masked_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    visible = mask.astype(bool)
    if prediction.shape != target.shape or visible.sum() == 0:
        return None
    try:
        from skimage.metrics import structural_similarity

        masked_prediction = prediction.copy()
        masked_target = target.copy()
        if prediction.ndim == 3:
            masked_prediction[~visible] = 0.0
            masked_target[~visible] = 0.0
            return float(
                structural_similarity(
                    masked_target,
                    masked_prediction,
                    channel_axis=-1,
                    data_range=1.0,
                )
            )
        masked_prediction[~visible] = 0.0
        masked_target[~visible] = 0.0
        return float(
            structural_similarity(
                masked_target,
                masked_prediction,
                data_range=1.0,
            )
        )
    except Exception:
        pass

    if prediction.ndim == 3:
        pred_values = prediction[visible].reshape(-1, prediction.shape[-1])
        target_values = target[visible].reshape(-1, target.shape[-1])
    else:
        pred_values = prediction[visible].reshape(-1, 1)
        target_values = target[visible].reshape(-1, 1)
    if pred_values.size == 0:
        return None
    scores = []
    c1 = 0.01**2
    c2 = 0.03**2
    for channel_idx in range(pred_values.shape[1]):
        x = pred_values[:, channel_idx].astype(np.float64)
        y = target_values[:, channel_idx].astype(np.float64)
        mux = float(x.mean())
        muy = float(y.mean())
        vx = float(x.var())
        vy = float(y.var())
        covariance = float(((x - mux) * (y - muy)).mean())
        numerator = (2.0 * mux * muy + c1) * (2.0 * covariance + c2)
        denominator = (mux * mux + muy * muy + c1) * (vx + vy + c2)
        scores.append(numerator / denominator if denominator > 0.0 else 0.0)
    return float(np.clip(np.mean(scores), -1.0, 1.0))


def initialize_lpips_metric(enabled: bool, device: str) -> tuple[Any | None, dict[str, Any]]:
    if not enabled:
        return None, {
            "available": False,
            "reason": "disabled_by_config",
        }
    try:
        import lpips  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional environment.
        return None, {
            "available": False,
            "reason": f"lpips_import_failed:{type(exc).__name__}:{exc}",
        }
    try:
        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model, {
            "available": True,
            "reason": "ok",
            "net": "alex",
        }
    except Exception as exc:  # pragma: no cover - depends on optional environment.
        return None, {
            "available": False,
            "reason": f"lpips_init_failed:{type(exc).__name__}:{exc}",
        }


def compute_lpips_distance(
    model: Any | None,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    device: str,
) -> float | None:
    if model is None or prediction.shape != target.shape or mask.sum() == 0:
        return None
    pred = prediction.copy()
    ref = target.copy()
    visible = mask.astype(bool)
    if pred.ndim != 3 or pred.shape[-1] != 3:
        return None
    pred[~visible] = 0.0
    ref[~visible] = 0.0
    pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
    ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).float()
    pred_tensor = pred_tensor.to(device) * 2.0 - 1.0
    ref_tensor = ref_tensor.to(device) * 2.0 - 1.0
    with torch.no_grad():
        score = model(pred_tensor, ref_tensor)
    return float(score.detach().cpu().reshape(-1)[0].item())


def compute_lpips_distance_batch(
    model: Any | None,
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    masks: list[np.ndarray],
    device: str,
) -> list[float | None]:
    if model is None:
        return [None for _ in predictions]
    valid_indices = []
    pred_tensors = []
    ref_tensors = []
    for idx, (prediction, target, mask) in enumerate(zip(predictions, targets, masks, strict=True)):
        if prediction.shape != target.shape or mask.sum() == 0:
            continue
        if prediction.ndim != 3 or prediction.shape[-1] != 3:
            continue
        visible = mask.astype(bool)
        pred = prediction.copy()
        ref = target.copy()
        pred[~visible] = 0.0
        ref[~visible] = 0.0
        pred_tensors.append(torch.from_numpy(pred).permute(2, 0, 1).float())
        ref_tensors.append(torch.from_numpy(ref).permute(2, 0, 1).float())
        valid_indices.append(idx)
    scores: list[float | None] = [None for _ in predictions]
    if not pred_tensors:
        return scores
    pred_batch = torch.stack(pred_tensors, dim=0).to(device) * 2.0 - 1.0
    ref_batch = torch.stack(ref_tensors, dim=0).to(device) * 2.0 - 1.0
    with torch.no_grad():
        batch_scores = model(pred_batch, ref_batch).detach().cpu().reshape(-1).tolist()
    for idx, score in zip(valid_indices, batch_scores, strict=True):
        scores[idx] = float(score)
    return scores


def normalize_view_normal(normal: np.ndarray) -> np.ndarray:
    decoded = normal * 2.0 - 1.0
    norm = np.linalg.norm(decoded, axis=0, keepdims=True)
    return decoded / np.maximum(norm, 1e-6)


def proxy_render_from_uv_material(
    *,
    albedo: np.ndarray,
    normal: np.ndarray,
    rm_map: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    roughness = np.clip(rm_map[0], 0.02, 1.0)
    metallic = np.clip(rm_map[1], 0.0, 1.0)
    n = normalize_view_normal(np.clip(normal, 0.0, 1.0))
    light = np.asarray([0.35, -0.45, 0.82], dtype=np.float32)
    light = light / np.linalg.norm(light)
    view = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    half_vec = light + view
    half_vec = half_vec / np.linalg.norm(half_vec)
    ndotl = np.clip(np.sum(n * light[:, None, None], axis=0), 0.0, 1.0)
    ndoth = np.clip(np.sum(n * half_vec[:, None, None], axis=0), 0.0, 1.0)
    spec_power = 2.0 + (1.0 - roughness) * 64.0
    specular = np.power(ndoth, spec_power) * (0.04 * (1.0 - metallic) + metallic)
    diffuse = albedo * (0.18 + 0.82 * ndotl[None, :, :]) * (1.0 - 0.55 * metallic[None, :, :])
    color = diffuse + specular[None, :, :] * (0.35 + 0.65 * albedo)
    color = np.clip(color, 0.0, 1.0)
    color *= mask[None, :, :].astype(np.float32)
    return np.moveaxis(color, 0, -1)


def save_rgb_tensor_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(data, mode="RGB").save(path)


def save_error_heatmap(path: Path, error: torch.Tensor) -> None:
    value = error.detach().cpu()
    if value.ndim == 3:
        value = value.sum(dim=0, keepdim=True)
    value = value / max(float(value.max().item()), 1e-6)
    tensor_to_pil(value.clamp(0.0, 1.0), grayscale=True).save(path)


def gradient_magnitude_np(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        values = values[None, ...]
    grad_x = np.diff(values, axis=-1, append=values[..., -1:])
    grad_y = np.diff(values, axis=-2, append=values[..., -1:, :])
    return np.sqrt(np.square(grad_x) + np.square(grad_y)).mean(axis=0)


def make_edge_band(
    roughness: np.ndarray,
    metallic: np.ndarray,
    mask: np.ndarray,
    *,
    dilation: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    visible = mask.astype(bool)
    if not visible.any():
        return visible, visible
    grad = gradient_magnitude_np(np.stack([roughness, metallic], axis=0))
    visible_grad = grad[visible]
    threshold = max(float(np.quantile(visible_grad, 0.75)), 0.025)
    edge_seed = (grad >= threshold) & visible
    if not edge_seed.any():
        interior_mask = np.asarray(
            Image.fromarray((visible.astype(np.uint8) * 255)).filter(ImageFilter.MinFilter(9))
        ) > 0
        edge_seed = visible & ~interior_mask
    edge = np.asarray(
        Image.fromarray((edge_seed.astype(np.uint8) * 255)).filter(
            ImageFilter.MaxFilter(max(int(dilation), 1) | 1)
        )
    ) > 0
    edge &= visible
    interior = visible & ~edge
    if not interior.any():
        interior = visible
    return edge, interior


def compute_boundary_bleed_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    edge, interior = make_edge_band(target[0], target[1], mask)
    error = np.abs(prediction - target).sum(axis=0)
    edge_error = masked_mean_np(error, edge)
    interior_error = masked_mean_np(error, interior)
    score = None
    if edge_error is not None and interior_error is not None:
        score = float(edge_error - interior_error)
    return {
        "score": score,
        "edge_error": edge_error,
        "interior_error": interior_error,
        "edge_pixel_rate": float(edge.sum() / max(mask.sum(), 1)),
    }


def compute_gradient_preservation(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    pred_grad = gradient_magnitude_np(prediction)
    target_grad = gradient_magnitude_np(target)
    visible = mask.astype(bool)
    if visible.sum() < 2:
        return None
    x = pred_grad[visible].astype(np.float64)
    y = target_grad[visible].astype(np.float64)
    if float(x.std()) <= 1e-8 or float(y.std()) <= 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_highlight_localization(
    prediction: np.ndarray,
    target: np.ndarray,
    reference_rgb: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    visible = mask.astype(bool)
    if visible.sum() < 8:
        return {
            "error": None,
            "iou": None,
            "center_distance": None,
            "highlight_pixel_rate": 0.0,
            "rm_error": None,
        }
    luminance = (
        0.2126 * reference_rgb[..., 0]
        + 0.7152 * reference_rgb[..., 1]
        + 0.0722 * reference_rgb[..., 2]
    )
    visible_luminance = luminance[visible]
    threshold = max(float(np.quantile(visible_luminance, 0.90)), 0.72)
    highlight = (luminance >= threshold) & visible
    if highlight.sum() < 4:
        return {
            "error": None,
            "iou": None,
            "center_distance": None,
            "highlight_pixel_rate": float(highlight.sum() / max(visible.sum(), 1)),
            "rm_error": None,
        }
    response = (1.0 - prediction[0]) * (0.55 + 0.45 * prediction[1])
    response_values = response[visible]
    response_threshold = float(
        np.quantile(response_values, 1.0 - min(float(highlight.sum() / visible.sum()), 0.50))
    )
    predicted_highlight = (response >= response_threshold) & visible
    intersection = float((highlight & predicted_highlight).sum())
    union = float((highlight | predicted_highlight).sum())
    iou = intersection / max(union, 1.0)
    yy, xx = np.indices(mask.shape)
    highlight_weight = highlight.astype(np.float64)
    predicted_weight = predicted_highlight.astype(np.float64)
    h_sum = max(float(highlight_weight.sum()), 1.0)
    p_sum = max(float(predicted_weight.sum()), 1.0)
    h_center = np.asarray(
        [
            float((xx * highlight_weight).sum() / h_sum),
            float((yy * highlight_weight).sum() / h_sum),
        ]
    )
    p_center = np.asarray(
        [
            float((xx * predicted_weight).sum() / p_sum),
            float((yy * predicted_weight).sum() / p_sum),
        ]
    )
    diag = math.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2)
    center_distance = float(np.linalg.norm(h_center - p_center) / max(diag, 1.0))
    rm_error = float(np.abs(prediction[:, highlight] - target[:, highlight]).mean())
    return {
        "error": float((1.0 - iou) + center_distance),
        "iou": float(iou),
        "center_distance": center_distance,
        "highlight_pixel_rate": float(highlight.sum() / max(visible.sum(), 1)),
        "rm_error": rm_error,
    }


def compute_residual_safety(
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, float]:
    baseline_np = baseline.detach().cpu().numpy()
    refined_np = refined.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    confidence_np = confidence.detach().cpu().numpy()[0]
    visible = confidence_np > 0.01
    if not visible.any():
        visible = np.ones_like(confidence_np, dtype=bool)
    baseline_error = np.abs(baseline_np - target_np).sum(axis=0)
    refined_error = np.abs(refined_np - target_np).sum(axis=0)
    residual = np.abs(refined_np - baseline_np).sum(axis=0)
    target_gap = np.abs(target_np - baseline_np).sum(axis=0)
    changed = (residual > 0.03) & visible
    improvement = (refined_error + 1e-6 < baseline_error - 0.01) & visible
    regression = (refined_error > baseline_error + 0.01) & visible
    safe_region = ((target_gap > 0.05) | (confidence_np < 0.60)) & visible
    safe_improvement = changed & improvement & safe_region
    unnecessary_change = changed & (baseline_error < 0.03) & ~improvement
    changed_count = max(int(changed.sum()), 1)
    visible_count = max(int(visible.sum()), 1)
    safe_rate = float(safe_improvement.sum() / changed_count)
    unnecessary_rate = float(unnecessary_change.sum() / changed_count)
    regression_rate = float(regression.sum() / visible_count)
    return {
        "changed_pixel_rate": float(changed.sum() / visible_count),
        "safe_improvement_rate": safe_rate,
        "unnecessary_change_rate": unnecessary_rate,
        "regression_rate": regression_rate,
        "safety_score": float(safe_rate - unnecessary_rate - regression_rate),
    }


def compute_confidence_bins(
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, dict[str, float | None]]:
    baseline_error = (baseline.detach().cpu() - target.detach().cpu()).abs().sum(dim=0).numpy()
    refined_error = (refined.detach().cpu() - target.detach().cpu()).abs().sum(dim=0).numpy()
    conf = confidence.detach().cpu().numpy()[0]
    bins = {
        "low": conf < 0.34,
        "mid": (conf >= 0.34) & (conf < 0.67),
        "high": conf >= 0.67,
    }
    result = {}
    for name, mask in bins.items():
        if not mask.any():
            result[name] = {
                "pixel_count": 0.0,
                "baseline_total_mae": None,
                "refined_total_mae": None,
                "improvement_total": None,
            }
            continue
        baseline_mean = float(baseline_error[mask].mean())
        refined_mean = float(refined_error[mask].mean())
        result[name] = {
            "pixel_count": float(mask.sum()),
            "baseline_total_mae": baseline_mean,
            "refined_total_mae": refined_mean,
            "improvement_total": baseline_mean - refined_mean,
        }
    return result


def compute_binary_f1(labels: list[int], scores: list[float], threshold: float) -> float:
    if not labels:
        return 0.0
    tp = fp = fn = 0
    for label, score in zip(labels, scores, strict=True):
        pred = int(score >= threshold)
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_binary_auroc(labels: list[int], scores: list[float]) -> float:
    if not labels:
        return 0.0
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(np.asarray(scores, dtype=np.float64))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    positive_rank_sum = float(sum(ranks[idx] for idx, label in enumerate(labels) if label == 1))
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def binary_confusion_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float,
) -> dict[str, float]:
    if not labels:
        return {
            "f1": 0.0,
            "auroc": 0.0,
            "balanced_accuracy": 0.0,
            "confusion_rate": 0.0,
            "positive_count": 0.0,
            "negative_count": 0.0,
        }
    tp = tn = fp = fn = 0
    for label, score in zip(labels, scores, strict=True):
        pred = int(score >= threshold)
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
        else:
            fn += 1
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "f1": compute_binary_f1(labels, scores, threshold),
        "auroc": compute_binary_auroc(labels, scores),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "confusion_rate": float((fp + fn) / max(len(labels), 1)),
        "positive_count": float(sum(labels)),
        "negative_count": float(len(labels) - sum(labels)),
    }


def collect_optional_pair(
    rows: list[dict[str, Any]],
    baseline_key: str,
    refined_key: str,
    *,
    higher_is_better: bool,
    mode: str | None = None,
) -> dict[str, Any]:
    baseline_values = [row.get(baseline_key) for row in rows]
    refined_values = [row.get(refined_key) for row in rows]
    available_count = min(
        len([value for value in baseline_values if finite_or_none(value) is not None]),
        len([value for value in refined_values if finite_or_none(value) is not None]),
    )
    return metric_pair(
        baseline=optional_mean(baseline_values),
        refined=optional_mean(refined_values),
        higher_is_better=higher_is_better,
        count=available_count,
        mode=mode,
    )


def summarize_confidence_bins(
    bin_rows: list[dict[str, dict[str, float | None]]],
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for bin_name in ("low", "mid", "high"):
        baseline_values = []
        refined_values = []
        improvement_values = []
        pixel_counts = []
        for row in bin_rows:
            item = row.get(bin_name, {})
            baseline_values.append(item.get("baseline_total_mae"))
            refined_values.append(item.get("refined_total_mae"))
            improvement_values.append(item.get("improvement_total"))
            pixel_count = finite_or_none(item.get("pixel_count"))
            if pixel_count is not None:
                pixel_counts.append(pixel_count)
        summary[bin_name] = {
            "sample_count": len([value for value in baseline_values if finite_or_none(value) is not None]),
            "pixel_count": float(np.sum(pixel_counts)) if pixel_counts else 0.0,
            "baseline_total_mae": optional_mean(baseline_values),
            "refined_total_mae": optional_mean(refined_values),
            "improvement_total": optional_mean(improvement_values),
        }
    return summary


def build_metric_availability(
    *,
    rows: list[dict[str, Any]],
    dataset_size: int,
    render_metric_mode: str,
    lpips_status: dict[str, Any],
    effective_view_supervision_rate: float,
) -> dict[str, dict[str, Any]]:
    def count_pair(baseline_key: str, refined_key: str) -> int:
        return min(
            len([row for row in rows if finite_or_none(row.get(baseline_key)) is not None]),
            len([row for row in rows if finite_or_none(row.get(refined_key)) is not None]),
        )

    return {
        "uv_rm_mae": {
            "available": dataset_size > 0,
            "available_count": dataset_size,
            "reason": "uv_target_rm_and_confidence",
        },
        "view_rm_mae": {
            "available": count_pair("baseline_total_mae", "refined_total_mae") > 0,
            "available_count": count_pair("baseline_total_mae", "refined_total_mae"),
            "reason": "view_uv_and_view_rm_targets" if effective_view_supervision_rate > 0.0 else "missing_view_uv_or_view_rm_targets",
        },
        "proxy_render_psnr": {
            "available": count_pair("baseline_psnr", "refined_psnr") > 0,
            "available_count": count_pair("baseline_psnr", "refined_psnr"),
            "reason": render_metric_mode,
        },
        "proxy_render_ssim": {
            "available": count_pair("baseline_ssim", "refined_ssim") > 0,
            "available_count": count_pair("baseline_ssim", "refined_ssim"),
            "reason": render_metric_mode,
        },
        "proxy_render_lpips": {
            "available": count_pair("baseline_lpips", "refined_lpips") > 0,
            "available_count": count_pair("baseline_lpips", "refined_lpips"),
            "reason": lpips_status.get("reason", "unknown"),
            "lpips_available": bool(lpips_status.get("available", False)),
        },
    }


def build_metric_warnings(
    *,
    summary_payload: dict[str, Any],
    diagnostic_min_group_count: int,
) -> list[str]:
    warnings = []
    availability = summary_payload.get("metric_availability", {})
    for name in MAIN_METRIC_NAMES:
        item = availability.get(name, {})
        if not item.get("available", False):
            warnings.append(f"metric_unavailable:{name}:{item.get('reason', 'unknown')}")
    metal = summary_payload.get("metrics_material_specific", {}).get("metal_nonmetal_confusion", {})
    for prefix in ("uv_level", "view_level", "object_level"):
        level_item = metal.get(prefix, {})
        for variant in ("baseline", "refined"):
            item = level_item.get(variant, {}) if isinstance(level_item, dict) else {}
            if item.get("f1") == 1.0 and item.get("auroc") == 0.0:
                warnings.append(f"metal_metric_degenerate:{prefix}:{variant}:f1=1.0_auroc=0.0")
            if item.get("positive_count", 0.0) == 0.0 or item.get("negative_count", 0.0) == 0.0:
                warnings.append(f"metal_metric_single_class:{prefix}:{variant}")
    by_group = summary_payload.get("metrics_by_group", {})
    for group_name, group_values in by_group.items():
        if not isinstance(group_values, dict):
            continue
        for group_id, group_item in group_values.items():
            if isinstance(group_item, dict) and int(group_item.get("count", 0)) < diagnostic_min_group_count:
                warnings.append(f"group_diagnostic_only:{group_name}:{group_id}:count={group_item.get('count', 0)}")
    disagreement = summary_payload.get("metrics_diagnostics", {}).get("metric_disagreement", {})
    if disagreement.get("has_disagreement"):
        warnings.extend(disagreement.get("warnings", []))
    return warnings


def build_metric_disagreement(
    *,
    uv_improvement: float,
    view_improvement: float | None,
    object_improvement: float | None,
) -> dict[str, Any]:
    values = {
        "uv_level_improvement_total": uv_improvement,
        "view_level_improvement_total": view_improvement,
        "object_level_improvement_total": object_improvement,
    }
    signs = {
        key: (None if value is None else (1 if value > 1e-6 else -1 if value < -1e-6 else 0))
        for key, value in values.items()
    }
    finite_signs = {key: value for key, value in signs.items() if value is not None}
    non_zero_signs = {key: value for key, value in finite_signs.items() if value != 0}
    has_disagreement = len(set(non_zero_signs.values())) > 1
    warnings = []
    if has_disagreement:
        warnings.append(
            "metric_disagreement:uv/view/object improvements have conflicting signs; do not use as paper-stage conclusion until audited"
        )
    return {
        "has_disagreement": bool(has_disagreement),
        "values": values,
        "signs": signs,
        "warnings": warnings,
    }


def write_metric_disagreement_report(
    output_dir: Path,
    disagreement: dict[str, Any],
) -> tuple[Path, Path]:
    json_path = output_dir / "metric_disagreement_report.json"
    html_path = output_dir / "metric_disagreement_report.html"
    json_path.write_text(
        json.dumps(make_json_serializable(disagreement), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = []
    for key, value in disagreement.get("values", {}).items():
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (
                key,
                "n/a" if value is None else f"{float(value):.6f}",
                disagreement.get("signs", {}).get(key),
            )
        )
    warning_html = "".join(
        f"<li>{str(warning)}</li>" for warning in disagreement.get("warnings", [])
    ) or "<li>none</li>"
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'>",
                "<title>Metric Disagreement Report</title>",
                "<style>body{font-family:Arial,sans-serif;background:#111827;color:#f8fafc;margin:24px;}table{border-collapse:collapse;width:100%;}td,th{border-bottom:1px solid #334155;padding:8px;text-align:left}.bad{color:#fca5a5}.good{color:#86efac}</style>",
                "</head><body>",
                "<h1>Metric Disagreement Report</h1>",
                f"<p>Status: <strong class='{'bad' if disagreement.get('has_disagreement') else 'good'}'>{'DISAGREEMENT' if disagreement.get('has_disagreement') else 'OK'}</strong></p>",
                "<table><thead><tr><th>Metric Level</th><th>Improvement</th><th>Sign</th></tr></thead><tbody>",
                *rows,
                "</tbody></table>",
                "<h2>Warnings</h2><ul>",
                warning_html,
                "</ul></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, html_path


def write_diagnostic_cases(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    top_k: int = 24,
    failure_tags: list[str] | None = None,
) -> Path:
    failure_tags = failure_tags or []

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "object_id",
            "view_name",
            "generator_id",
            "source_name",
            "material_family",
            "paper_split",
            "baseline_total_mae",
            "refined_total_mae",
            "improvement_total",
            "baseline_primary_failure",
            "refined_primary_failure",
            "baseline_boundary_bleed_score",
            "refined_boundary_bleed_score",
            "baseline_highlight_localization_error",
            "refined_highlight_localization_error",
            "baseline_rm_gradient_preservation",
            "refined_rm_gradient_preservation",
            "prior_residual_safety_score",
        ]
        return {key: make_json_serializable(row.get(key)) for key in keys}

    improved = sorted(rows, key=lambda row: finite_or_none(row.get("improvement_total")) or -999.0, reverse=True)
    regressed = sorted(rows, key=lambda row: finite_or_none(row.get("improvement_total")) or 999.0)
    uncertain = sorted(
        rows,
        key=lambda row: (
            abs(finite_or_none(row.get("improvement_total")) or 0.0),
            -(finite_or_none(row.get("baseline_total_mae")) or 0.0),
        ),
    )
    by_failure = {}
    for tag in failure_tags:
        tagged = [
            row for row in rows if tag in row.get("baseline_tags", []) or tag in row.get("refined_tags", [])
        ]
        by_failure[tag] = [compact(row) for row in sorted(tagged, key=lambda row: finite_or_none(row.get("refined_total_mae")) or 0.0, reverse=True)[:top_k]]
    payload = {
        "top_improved": [compact(row) for row in improved[:top_k]],
        "top_regressed": [compact(row) for row in regressed[:top_k]],
        "top_uncertain": [compact(row) for row in uncertain[:top_k]],
        "top_failure_cases": by_failure,
    }
    path = output_dir / "diagnostic_cases.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def count_failure_tags(rows: list[dict[str, Any]]) -> tuple[Counter, Counter]:
    baseline_tag_counts = Counter()
    refined_tag_counts = Counter()
    for row in rows:
        for tag in row.get("baseline_tags", []):
            baseline_tag_counts[tag] += 1
        for tag in row.get("refined_tags", []):
            refined_tag_counts[tag] += 1
    return baseline_tag_counts, refined_tag_counts
