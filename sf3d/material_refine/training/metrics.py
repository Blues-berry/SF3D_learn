from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch
import torch.nn.functional as F

from sf3d.material_refine.experiment import wandb

_LPIPS_MODEL: Any | None = None
_LPIPS_DEVICE: str | None = None
_LPIPS_FAILURE: str | None = None
METALLIC_THRESHOLD = 0.5
MATERIAL_CONTEXT_GLOSSY_THRESHOLD = 0.45

def total_variation_loss(tensor: torch.Tensor) -> torch.Tensor:
    loss_x = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs().mean()
    loss_y = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs().mean()
    return loss_x + loss_y


def rm_gradient_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    grad_x = torch.zeros_like(tensor)
    grad_y = torch.zeros_like(tensor)
    grad_x[:, :, :, 1:] = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs()
    grad_y[:, :, 1:, :] = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs()
    return (grad_x + grad_y).amax(dim=1, keepdim=True)


def edge_aware_l1_loss(
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    edge_weight = rm_gradient_magnitude(target).detach()
    edge_weight = edge_weight / edge_weight.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(epsilon)
    edge_weight = edge_weight.clamp(0.0, 1.0) * confidence
    return ((refined - target).abs() * edge_weight).sum() / edge_weight.sum().clamp_min(1.0)


def boundary_bleed_loss(
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    kernel_size: int,
    epsilon: float,
) -> torch.Tensor:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    edge_weight = rm_gradient_magnitude(target).detach()
    edge_weight = edge_weight / edge_weight.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(epsilon)
    edge_band = F.max_pool2d(
        edge_weight.clamp(0.0, 1.0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).clamp(0.0, 1.0)
    boundary_weight = edge_band * confidence
    interior_weight = (1.0 - edge_band) * confidence
    error = (refined - target).abs()
    boundary_error = (error * boundary_weight).sum() / boundary_weight.sum().clamp_min(1.0)
    interior_error = (error * interior_weight).sum() / interior_weight.sum().clamp_min(1.0)
    return boundary_error + F.relu(boundary_error - interior_error)


def gradient_preservation_loss(
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    refined_dx = refined[:, :, :, 1:] - refined[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    weight_dx = torch.minimum(confidence[:, :, :, 1:], confidence[:, :, :, :-1]).expand_as(refined_dx)
    refined_dy = refined[:, :, 1:, :] - refined[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    weight_dy = torch.minimum(confidence[:, :, 1:, :], confidence[:, :, :-1, :]).expand_as(refined_dy)
    if channel_weights is not None:
        weight_dx = weight_dx * channel_weights
        weight_dy = weight_dy * channel_weights
    if sample_weights is None:
        loss_dx = ((refined_dx - target_dx).abs() * weight_dx).sum()
        loss_dy = ((refined_dy - target_dy).abs() * weight_dy).sum()
        weight = weight_dx.sum() + weight_dy.sum()
        return (loss_dx + loss_dy) / weight.clamp_min(1.0)
    per_sample_dx = ((refined_dx - target_dx).abs() * weight_dx).flatten(1).sum(dim=1) / weight_dx.flatten(1).sum(dim=1).clamp_min(1.0)
    per_sample_dy = ((refined_dy - target_dy).abs() * weight_dy).flatten(1).sum(dim=1) / weight_dy.flatten(1).sum(dim=1).clamp_min(1.0)
    per_sample = 0.5 * (per_sample_dx + per_sample_dy)
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)


def metallic_classification_loss(
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    metallic_pred = refined[:, 1:2].float().clamp(1e-4, 1.0 - 1e-4)
    metallic_target = (target[:, 1:2].float() >= METALLIC_THRESHOLD).to(metallic_pred.dtype)
    per_texel = -(
        metallic_target * metallic_pred.log()
        + (1.0 - metallic_target) * (1.0 - metallic_pred).log()
    )
    weight = confidence.float()
    return (per_texel * weight).sum() / weight.sum().clamp_min(1.0)


def material_context_loss(
    material_logits: torch.Tensor | None,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    if material_logits is None:
        return target.new_zeros(())
    class_count = int(material_logits.shape[1])
    if class_count < 2:
        return target.new_zeros(())
    roughness = target[:, 0:1].float()
    metallic = target[:, 1:2].float()
    labels = torch.zeros_like(roughness, dtype=torch.long)
    glossy_label = 1 if class_count > 1 else 0
    metal_label = min(2, class_count - 1)
    labels = torch.where(
        (metallic >= METALLIC_THRESHOLD),
        torch.full_like(labels, metal_label),
        labels,
    )
    labels = torch.where(
        (metallic < METALLIC_THRESHOLD) & (roughness < MATERIAL_CONTEXT_GLOSSY_THRESHOLD),
        torch.full_like(labels, glossy_label),
        labels,
    )
    per_texel = F.cross_entropy(material_logits.float(), labels[:, 0], reduction="none")
    weight = confidence[:, 0].float()
    return (per_texel * weight).sum() / weight.sum().clamp_min(1.0)


def residual_safety_loss(
    refined: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    target_delta = (target - baseline).abs().detach()
    unnecessary_region = (target_delta < margin).to(refined.dtype) * confidence
    residual = (refined - baseline).abs()
    return (residual * unnecessary_region).sum() / unnecessary_region.sum().clamp_min(1.0)


def sample_uv_maps_to_view(uv_maps: torch.Tensor, view_uvs: torch.Tensor) -> torch.Tensor:
    grid = torch.where(torch.isfinite(view_uvs), view_uvs, torch.zeros_like(view_uvs)).clone()
    grid[..., 0] = grid[..., 0] * 2.0 - 1.0
    grid[..., 1] = (1.0 - grid[..., 1]) * 2.0 - 1.0
    batch, views, height, width, _ = grid.shape
    repeated_maps = (
        uv_maps.unsqueeze(1)
        .expand(-1, views, -1, -1, -1)
        .reshape(batch * views, uv_maps.shape[1], uv_maps.shape[2], uv_maps.shape[3])
    )
    sampled = F.grid_sample(
        repeated_maps,
        grid.reshape(batch * views, height, width, 2),
        mode="bilinear",
        align_corners=True,
    )
    return sampled.view(batch, views, sampled.shape[1], height, width)


def view_uv_valid_mask(view_uvs: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(view_uvs).all(dim=-1)
    in_range = (
        (view_uvs[..., 0] >= 0.0)
        & (view_uvs[..., 0] <= 1.0)
        & (view_uvs[..., 1] >= 0.0)
        & (view_uvs[..., 1] <= 1.0)
    )
    return (finite & in_range).to(view_uvs.dtype).unsqueeze(2)


def extract_metadata_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def rm_channel_weight_tensor(
    args: argparse.Namespace,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    return like.new_tensor(
        [
            float(getattr(args, "roughness_channel_weight", 1.0)),
            float(getattr(args, "metallic_channel_weight", 1.0)),
        ]
    ).view(1, 2, 1, 1)


def variant_loss_weight_tensor(
    batch: dict[str, Any],
    args: argparse.Namespace,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    weights = getattr(args, "train_variant_loss_weights", {}) or {}
    variants = list(batch.get("prior_variant_type") or [])
    if not weights or not variants:
        size = int(batch["uv_albedo"].shape[0])
        return torch.ones(size, device=device, dtype=torch.float32)
    values = [float(weights.get(str(variant or "unknown"), 1.0)) for variant in variants]
    return torch.tensor(values, device=device, dtype=torch.float32).clamp_min(1.0e-6)


def weighted_total_variation_loss(
    tensor: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    loss_x = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    loss_y = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    per_sample = loss_x + loss_y
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)


def weighted_masked_channel_mean(
    error: torch.Tensor,
    weight: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    per_sample = (error * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(dim=1).clamp_min(1.0)
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    target = torch.cat(
        [batch["uv_target_roughness"], batch["uv_target_metallic"]],
        dim=1,
    )
    confidence = batch["uv_target_confidence"].clamp(0.0, 1.0)
    refined = outputs["refined"]
    coarse = outputs["coarse"]
    baseline = outputs.get("input_prior", outputs["baseline"])
    prior_confidence = batch.get("input_prior_confidence", batch["uv_prior_confidence"]).clamp(0.0, 1.0)
    sample_weights = variant_loss_weight_tensor(batch, args, device=refined.device).to(refined.dtype)
    channel_weights = rm_channel_weight_tensor(args, like=refined)
    channel_confidence = confidence.expand_as(target) * channel_weights

    refine_l1 = weighted_masked_channel_mean((refined - target).abs(), channel_confidence, sample_weights)
    coarse_l1 = weighted_masked_channel_mean((coarse - target).abs(), channel_confidence, sample_weights)
    target_prior_delta = (target - baseline).abs().mean(dim=1, keepdim=True)
    prior_safe_mask = (
        confidence
        * prior_confidence
        * (target_prior_delta <= float(args.residual_safety_margin)).to(refined.dtype)
    )
    prior_consistency = weighted_masked_channel_mean(
        (refined - baseline).abs(),
        prior_safe_mask.expand_as(target),
        sample_weights,
    )
    smoothness = weighted_total_variation_loss(refined, sample_weights)
    edge_aware = (
        weighted_masked_channel_mean(
            (refined - target).abs(),
            (
                (
                    rm_gradient_magnitude(target).detach()
                    / rm_gradient_magnitude(target).detach().flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(args.edge_aware_epsilon)
                ).clamp(0.0, 1.0)
                * confidence
            ).expand_as(target)
            * channel_weights,
            sample_weights,
        )
        if args.edge_aware_weight > 0.0
        else refined.new_zeros(())
    )
    boundary_bleed = (
        boundary_bleed_loss(
            refined,
            target,
            confidence,
            kernel_size=args.boundary_band_kernel,
            epsilon=args.edge_aware_epsilon,
        )
        if args.boundary_bleed_weight > 0.0
        else refined.new_zeros(())
    )
    gradient_preservation = (
        gradient_preservation_loss(
            refined,
            target,
            confidence,
            sample_weights=sample_weights,
            channel_weights=channel_weights,
        )
        if args.gradient_preservation_weight > 0.0
        else refined.new_zeros(())
    )
    metallic_classification = (
        metallic_classification_loss(refined, target, confidence)
        if args.metallic_classification_weight > 0.0
        else refined.new_zeros(())
    )
    material_context = (
        material_context_loss(outputs.get("material_logits"), target, confidence)
        if args.material_context_weight > 0.0
        else refined.new_zeros(())
    )
    residual_safety = (
        weighted_masked_channel_mean(
            (refined - baseline).abs(),
            (
                (target - baseline).abs().detach().mean(dim=1, keepdim=True) < float(args.residual_safety_margin)
            ).to(refined.dtype).expand_as(target)
            * confidence.expand_as(target),
            sample_weights,
        )
        if args.residual_safety_weight > 0.0
        else refined.new_zeros(())
    )
    residual_gate = outputs.get("change_gate", outputs.get("residual_gate"))
    residual_delta = outputs.get("delta_rm", outputs.get("residual_delta"))
    residual_gate_mean = (
        residual_gate.mean() if residual_gate is not None else refined.new_zeros(())
    )
    roughness_gate_mean = (
        residual_gate[:, 0:1].mean()
        if isinstance(residual_gate, torch.Tensor) and residual_gate.ndim == 4 and residual_gate.shape[1] >= 1
        else residual_gate_mean
    )
    metallic_gate_mean = (
        residual_gate[:, 1:2].mean()
        if isinstance(residual_gate, torch.Tensor) and residual_gate.ndim == 4 and residual_gate.shape[1] >= 2
        else residual_gate_mean
    )
    residual_delta_abs = (
        residual_delta.abs().mean() if residual_delta is not None else refined.new_zeros(())
    )
    view_uncertainty_gate = outputs.get("view_uncertainty_residual_gate_uv")
    bleed_risk_gate = outputs.get("bleed_risk_residual_gate_uv")
    topology_residual_gate = outputs.get("topology_residual_gate_uv")
    render_support_gate = outputs.get("render_support_gate")
    inverse_material_gate = outputs.get("inverse_material_gate_uv")
    residual_channel_gate = outputs.get("residual_channel_gate_uv")
    roughness_safety_gate = outputs.get("roughness_safety_gate_uv")
    metallic_safety_gate = outputs.get("metallic_safety_gate_uv")
    metallic_evidence = outputs.get("metallic_evidence_uv")
    metallic_cap_strength = outputs.get("metallic_cap_strength_uv")
    evidence_update_budget = outputs.get("evidence_update_budget_uv")
    evidence_update_support = outputs.get("evidence_update_support_uv")
    view_uncertainty_gate_mean = (
        view_uncertainty_gate.mean() if view_uncertainty_gate is not None else refined.new_ones(())
    )
    bleed_risk_gate_mean = (
        bleed_risk_gate.mean() if bleed_risk_gate is not None else refined.new_ones(())
    )
    topology_residual_gate_mean = (
        topology_residual_gate.mean() if topology_residual_gate is not None else refined.new_ones(())
    )
    render_support_gate_mean = (
        render_support_gate.mean() if render_support_gate is not None else refined.new_ones(())
    )
    inverse_material_gate_mean = (
        inverse_material_gate.mean() if inverse_material_gate is not None else refined.new_ones(())
    )
    residual_channel_gate_mean = (
        residual_channel_gate.mean() if residual_channel_gate is not None else refined.new_ones(())
    )
    roughness_safety_gate_mean = (
        roughness_safety_gate.mean() if roughness_safety_gate is not None else refined.new_ones(())
    )
    metallic_safety_gate_mean = (
        metallic_safety_gate.mean() if metallic_safety_gate is not None else refined.new_ones(())
    )
    metallic_evidence_mean = (
        metallic_evidence.mean() if metallic_evidence is not None else refined.new_zeros(())
    )
    metallic_cap_strength_mean = (
        metallic_cap_strength.mean() if metallic_cap_strength is not None else refined.new_zeros(())
    )
    evidence_update_budget_mean = (
        evidence_update_budget.mean()
        if evidence_update_budget is not None
        else refined.new_ones(())
    )
    evidence_update_support_mean = (
        evidence_update_support.mean()
        if evidence_update_support is not None
        else refined.new_ones(())
    )
    diagnostics = outputs.get("diagnostics") or {}

    def diagnostic_mean(name: str, fallback: torch.Tensor) -> torch.Tensor:
        value = diagnostics.get(name)
        if isinstance(value, torch.Tensor):
            return value.mean()
        return fallback

    change_gate_mean = diagnostic_mean("change_gate_mean", residual_gate_mean)
    roughness_gate_mean = diagnostic_mean("roughness_gate_mean", roughness_gate_mean)
    metallic_gate_mean = diagnostic_mean("metallic_gate_mean", metallic_gate_mean)
    mean_abs_delta = diagnostic_mean("mean_abs_delta", residual_delta_abs)
    prior_reliability_tensor = outputs.get("prior_reliability")
    prior_reliability_mean = diagnostic_mean(
        "prior_reliability_mean",
        prior_reliability_tensor.mean()
        if isinstance(prior_reliability_tensor, torch.Tensor)
        else prior_confidence.mean(),
    )
    boundary_delta_mean = diagnostic_mean(
        "boundary_delta_mean",
        refined.new_zeros(()),
    )

    sampled_view_rm_loss_enabled = bool(args.enable_sampled_view_rm_loss) or args.view_consistency_mode != "disabled"
    if sampled_view_rm_loss_enabled and batch.get("view_uvs") is not None:
        sampled_refined = sample_uv_maps_to_view(refined, batch["view_uvs"])
        view_mask = batch["view_masks"].clamp(0.0, 1.0)
        view_mask = view_mask * view_uv_valid_mask(batch["view_uvs"]).to(view_mask.device)
        supervision_mask = batch["has_effective_view_supervision"].to(view_mask.device)
        view_mask = view_mask * supervision_mask.view(-1, 1, 1, 1, 1)
        # Some generated canonical bundles store per-view RM PNGs as RGBA masks
        # with 0/1 byte RGB values, which become nearly black after normalization.
        # The reliable supervision source is the UV target plus UV correspondence.
        view_targets = sample_uv_maps_to_view(target, batch["view_uvs"])
        view_channel_weights = channel_weights.view(1, 1, 2, 1, 1)
        view_consistency = weighted_masked_channel_mean(
            (sampled_refined - view_targets).abs().reshape(refined.shape[0], -1, sampled_refined.shape[-2], sampled_refined.shape[-1]),
            (
                view_mask.expand_as(sampled_refined) * view_channel_weights
            ).reshape(refined.shape[0], -1, sampled_refined.shape[-2], sampled_refined.shape[-1]),
            sample_weights,
        )
    else:
        view_consistency = refined.new_zeros(())

    total = (
        args.refine_weight * refine_l1
        + args.coarse_weight * coarse_l1
        + args.prior_consistency_weight * prior_consistency
        + args.smoothness_weight * smoothness
        + max(float(args.view_consistency_weight), float(args.sampled_view_rm_loss_weight)) * view_consistency
        + args.edge_aware_weight * edge_aware
        + args.boundary_bleed_weight * boundary_bleed
        + args.gradient_preservation_weight * gradient_preservation
        + args.metallic_classification_weight * metallic_classification
        + args.material_context_weight * material_context
        + args.residual_safety_weight * residual_safety
    )
    return {
        "total": total,
        "loss_uv": refine_l1.detach(),
        "loss_prior_safe": (prior_consistency + residual_safety).detach(),
        "loss_boundary": boundary_bleed.detach(),
        "loss_gradient": (gradient_preservation + edge_aware).detach(),
        "refine_l1": refine_l1.detach(),
        "coarse_l1": coarse_l1.detach(),
        "prior_consistency": prior_consistency.detach(),
        "smoothness": smoothness.detach(),
        "view_consistency": view_consistency.detach(),
        "edge_aware": edge_aware.detach(),
        "boundary_bleed": boundary_bleed.detach(),
        "gradient_preservation": gradient_preservation.detach(),
        "metallic_classification": metallic_classification.detach(),
        "material_context": material_context.detach(),
        "residual_safety": residual_safety.detach(),
        "residual_gate_mean": residual_gate_mean.detach(),
        "roughness_gate_mean": roughness_gate_mean.detach(),
        "metallic_gate_mean": metallic_gate_mean.detach(),
        "residual_delta_abs": residual_delta_abs.detach(),
        "change_gate_mean": change_gate_mean.detach(),
        "mean_abs_delta": mean_abs_delta.detach(),
        "prior_reliability_mean": prior_reliability_mean.detach(),
        "boundary_delta_mean": boundary_delta_mean.detach(),
        "view_uncertainty_gate_mean": view_uncertainty_gate_mean.detach(),
        "bleed_risk_gate_mean": bleed_risk_gate_mean.detach(),
        "topology_residual_gate_mean": topology_residual_gate_mean.detach(),
        "render_support_gate_mean": render_support_gate_mean.detach(),
        "inverse_material_gate_mean": inverse_material_gate_mean.detach(),
        "residual_channel_gate_mean": residual_channel_gate_mean.detach(),
        "roughness_safety_gate_mean": roughness_safety_gate_mean.detach(),
        "metallic_safety_gate_mean": metallic_safety_gate_mean.detach(),
        "metallic_evidence_mean": metallic_evidence_mean.detach(),
        "metallic_cap_strength_mean": metallic_cap_strength_mean.detach(),
        "evidence_update_budget_mean": evidence_update_budget_mean.detach(),
        "evidence_update_support_mean": evidence_update_support_mean.detach(),
    }

def filter_train_wandb_logs(logs: dict[str, Any]) -> dict[str, Any]:
    allowed_exact = {
        "optim/lr",
        "throughput/samples_per_second",
        "throughput/seconds_per_batch",
        "train/total",
        "train/refine_l1",
        "train/coarse_l1",
        "train/prior_consistency",
        "train/view_consistency",
        "train/grad_norm",
        "train/effective_view_supervision_rate",
    }
    filtered: dict[str, Any] = {}
    for key, value in logs.items():
        if key in allowed_exact:
            filtered[key] = value
    return filtered


def filter_validation_wandb_logs(logs: dict[str, Any]) -> dict[str, Any]:
    allowed_exact = {
        "best/selection_metric",
        "best/epoch",
        "val/gain_total",
        "val/prior_aware/score",
        "val/object_level/regression_rate",
        "val/case_level/regression_rate",
    }
    return {
        key: value
        for key, value in logs.items()
        if key in allowed_exact
        or key in {
            "val/rm_proxy/view_mae/delta",
            "val/rm_proxy/view_mse/delta",
            "val/rm_proxy/view_psnr/delta",
        }
        or (key.startswith("val/by_variant/") and key.endswith("/gain_total"))
        or (key.startswith("val/by_variant/") and key.endswith("/gain_relative"))
        or (key.startswith("val/by_variant/") and key.endswith("/gain_potential_normalized"))
    }


def add_step_context(
    logs: dict[str, Any],
    *,
    epoch: int,
    optimizer_step: int,
    global_batch_step: int | None = None,
    learning_rate: float | None = None,
    progress_fraction: float | None = None,
) -> dict[str, Any]:
    enriched = dict(logs)
    if learning_rate is not None:
        enriched["optim/lr"] = float(learning_rate)
    return enriched


def configure_wandb_step_metrics(run: Any | None) -> None:
    if run is None or wandb is None:
        return
    try:
        for metric_key in (
            "optim/lr",
            "throughput/samples_per_second",
            "throughput/seconds_per_batch",
            "train/total",
            "train/refine_l1",
            "train/coarse_l1",
            "train/prior_consistency",
            "train/view_consistency",
            "train/grad_norm",
            "train/effective_view_supervision_rate",
            "val/input_prior_total_mae",
            "val/refined_total_mae",
            "val/gain_total",
            "val/prior_aware/score",
            "val/rm_proxy/view_mae/baseline",
            "val/rm_proxy/view_mae/refined",
            "val/rm_proxy/view_mae/delta",
            "val/rm_proxy/view_mse/baseline",
            "val/rm_proxy/view_mse/refined",
            "val/rm_proxy/view_mse/delta",
            "val/rm_proxy/view_psnr/baseline",
            "val/rm_proxy/view_psnr/refined",
            "val/rm_proxy/view_psnr/delta",
            "val/object_level/avg_improvement_total",
            "val/object_level/regression_rate",
            "val/case_level/avg_improvement_total",
            "val/case_level/regression_rate",
            "best/selection_metric",
            "best/epoch",
        ):
            wandb.define_metric(metric_key)
    except Exception as exc:  # noqa: BLE001 - W&B metric setup should not block training.
        print(f"[wandb:warning] define_metric_failed={type(exc).__name__}: {exc}")

def should_compute_render_proxy_validation(
    args: argparse.Namespace,
    validation_label: str,
) -> bool:
    interval = int(getattr(args, "render_proxy_validation_milestone_interval", 0) or 0)
    if interval <= 0:
        return False
    milestone = validation_milestone_index(validation_label)
    if milestone is None:
        return False
    return milestone > 0 and milestone % interval == 0


def psnr_from_mse(mse: float) -> float | None:
    if mse < 0.0 or math.isnan(mse):
        return None
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def masked_global_ssim_torch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
    if prediction.numel() == 0 or target.shape != prediction.shape:
        return None
    weight = mask.expand_as(prediction).to(prediction.dtype)
    denom = weight.sum().clamp_min(1.0)
    mu_x = (prediction * weight).sum() / denom
    mu_y = (target * weight).sum() / denom
    var_x = ((prediction - mu_x).square() * weight).sum() / denom
    var_y = ((target - mu_y).square() * weight).sum() / denom
    cov_xy = ((prediction - mu_x) * (target - mu_y) * weight).sum() / denom
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return float(ssim.detach().clamp(-1.0, 1.0).cpu().item())


def get_lpips_model(enabled: bool, device: str) -> Any | None:
    global _LPIPS_DEVICE, _LPIPS_FAILURE, _LPIPS_MODEL
    if not enabled:
        return None
    if _LPIPS_MODEL is not None and _LPIPS_DEVICE == device:
        return _LPIPS_MODEL
    if _LPIPS_FAILURE is not None:
        return None
    try:
        import lpips  # type: ignore

        model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:  # noqa: BLE001 - optional metric should not block training.
        _LPIPS_FAILURE = f"{type(exc).__name__}: {exc}"
        print(f"[metric:warning] lpips_unavailable={_LPIPS_FAILURE}")
        return None
    _LPIPS_MODEL = model
    _LPIPS_DEVICE = device
    return _LPIPS_MODEL


def rm_to_lpips_rgb(rm_view: torch.Tensor) -> torch.Tensor:
    if rm_view.shape[1] == 1:
        rgb = rm_view.expand(-1, 3, -1, -1)
    else:
        rough = rm_view[:, 0:1]
        metal = rm_view[:, 1:2]
        rgb = torch.cat([rough, metal, rough], dim=1)
    return rgb.clamp(0.0, 1.0) * 2.0 - 1.0


def compute_lpips_rm_proxy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: str,
    enabled: bool,
    max_images: int,
) -> float | None:
    model = get_lpips_model(enabled, device)
    if model is None or max_images <= 0:
        return None
    pred = prediction.reshape(-1, prediction.shape[2], prediction.shape[3], prediction.shape[4])
    tgt = target.reshape(-1, target.shape[2], target.shape[3], target.shape[4])
    m = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
    valid = m.flatten(1).sum(dim=1) > 8
    if not bool(valid.any()):
        return None
    pred = pred[valid][:max_images] * m[valid][:max_images]
    tgt = tgt[valid][:max_images] * m[valid][:max_images]
    with torch.no_grad():
        distances = model(rm_to_lpips_rgb(pred), rm_to_lpips_rgb(tgt))
    return float(distances.mean().detach().cpu().item())


def safe_wandb_key(value: Any) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value or "unknown")
    )
    return safe.strip("_") or "unknown"


def finalize_render_proxy_metrics(store: dict[str, float]) -> dict[str, Any]:
    denom = max(float(store.get("denom", 0.0)), 1.0)
    baseline_mae = float(store.get("baseline_abs", 0.0) / denom)
    refined_mae = float(store.get("refined_abs", 0.0) / denom)
    baseline_mse = float(store.get("baseline_sq", 0.0) / denom)
    refined_mse = float(store.get("refined_sq", 0.0) / denom)
    ssim_count = max(float(store.get("ssim_count", 0.0)), 1.0)
    lpips_count = max(float(store.get("lpips_count", 0.0)), 1.0)
    return {
        "enabled": True,
        "available": bool(store.get("available_batches", 0.0) > 0.0),
        "batches": int(store.get("batches", 0.0)),
        "available_batches": int(store.get("available_batches", 0.0)),
        "samples": int(store.get("samples", 0.0)),
        "baseline_view_rm_mae": baseline_mae,
        "refined_view_rm_mae": refined_mae,
        "view_rm_mae_delta": baseline_mae - refined_mae,
        "baseline_proxy_rm_mse": baseline_mse,
        "refined_proxy_rm_mse": refined_mse,
        "proxy_rm_mse_delta": baseline_mse - refined_mse,
        "baseline_proxy_rm_psnr": psnr_from_mse(baseline_mse),
        "refined_proxy_rm_psnr": psnr_from_mse(refined_mse),
        "baseline_proxy_rm_ssim": float(store.get("baseline_ssim", 0.0) / ssim_count)
        if store.get("ssim_count", 0.0) > 0.0
        else None,
        "refined_proxy_rm_ssim": float(store.get("refined_ssim", 0.0) / ssim_count)
        if store.get("ssim_count", 0.0) > 0.0
        else None,
        "baseline_proxy_rm_lpips": float(store.get("baseline_lpips", 0.0) / lpips_count)
        if store.get("lpips_count", 0.0) > 0.0
        else None,
        "refined_proxy_rm_lpips": float(store.get("refined_lpips", 0.0) / lpips_count)
        if store.get("lpips_count", 0.0) > 0.0
        else None,
        "proxy_rm_psnr_delta": (
            None
            if psnr_from_mse(baseline_mse) is None or psnr_from_mse(refined_mse) is None
            else psnr_from_mse(refined_mse) - psnr_from_mse(baseline_mse)
        ),
        "lpips_status": "available" if store.get("lpips_count", 0.0) > 0.0 else (_LPIPS_FAILURE or "not_computed"),
        "mode": "view_projected_rm_proxy",
    }


def update_render_proxy_metrics(
    store: dict[str, float],
    *,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    device: str,
) -> None:
    if batch.get("view_uvs") is None:
        return
    view_mask = batch["view_masks"].clamp(0.0, 1.0)
    view_mask = view_mask * view_uv_valid_mask(batch["view_uvs"]).to(view_mask.device)
    supervision_mask = batch["has_effective_view_supervision"].to(view_mask.device)
    view_mask = view_mask * supervision_mask.view(-1, 1, 1, 1, 1)
    denom = float(view_mask.sum().detach().cpu().item()) * float(refined.shape[1])
    if denom <= 0.0:
        return
    target_views = sample_uv_maps_to_view(target, batch["view_uvs"])
    baseline_views = sample_uv_maps_to_view(baseline, batch["view_uvs"])
    refined_views = sample_uv_maps_to_view(refined, batch["view_uvs"])
    baseline_ssim = masked_global_ssim_torch(baseline_views, target_views, view_mask)
    refined_ssim = masked_global_ssim_torch(refined_views, target_views, view_mask)
    if baseline_ssim is not None and refined_ssim is not None:
        store["baseline_ssim"] += baseline_ssim
        store["refined_ssim"] += refined_ssim
        store["ssim_count"] += 1.0
    if bool(getattr(args, "val_enable_lpips", True)):
        baseline_lpips = compute_lpips_rm_proxy(
            baseline_views,
            target_views,
            view_mask,
            device=device,
            enabled=True,
            max_images=int(getattr(args, "val_lpips_max_images", 12)),
        )
        refined_lpips = compute_lpips_rm_proxy(
            refined_views,
            target_views,
            view_mask,
            device=device,
            enabled=True,
            max_images=int(getattr(args, "val_lpips_max_images", 12)),
        )
        if baseline_lpips is not None and refined_lpips is not None:
            store["baseline_lpips"] += baseline_lpips
            store["refined_lpips"] += refined_lpips
            store["lpips_count"] += 1.0
    baseline_error = baseline_views - target_views
    refined_error = refined_views - target_views
    store["baseline_abs"] += float((baseline_error.abs() * view_mask).sum().detach().cpu().item())
    store["refined_abs"] += float((refined_error.abs() * view_mask).sum().detach().cpu().item())
    store["baseline_sq"] += float(((baseline_error ** 2) * view_mask).sum().detach().cpu().item())
    store["refined_sq"] += float(((refined_error ** 2) * view_mask).sum().detach().cpu().item())
    store["denom"] += denom
    store["available_batches"] += 1.0
    store["samples"] += float(refined.shape[0])


def per_sample_view_rm_mae_delta(
    *,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor | None:
    if batch.get("view_uvs") is None:
        return None
    view_mask = batch["view_masks"].clamp(0.0, 1.0)
    view_mask = view_mask * view_uv_valid_mask(batch["view_uvs"]).to(view_mask.device)
    supervision_mask = batch["has_effective_view_supervision"].to(view_mask.device)
    view_mask = view_mask * supervision_mask.view(-1, 1, 1, 1, 1)
    denom = view_mask.flatten(1).sum(dim=1).clamp_min(1.0) * float(refined.shape[1])
    target_views = sample_uv_maps_to_view(target, batch["view_uvs"])
    baseline_views = sample_uv_maps_to_view(baseline, batch["view_uvs"])
    refined_views = sample_uv_maps_to_view(refined, batch["view_uvs"])
    baseline_mae = ((baseline_views - target_views).abs() * view_mask).flatten(1).sum(dim=1) / denom
    refined_mae = ((refined_views - target_views).abs() * view_mask).flatten(1).sum(dim=1) / denom
    has_support = view_mask.flatten(1).sum(dim=1) > 0.0
    return torch.where(has_support, baseline_mae - refined_mae, torch.full_like(baseline_mae, float("nan")))


def finalize_residual_gate_diagnostics(store: dict[str, float]) -> dict[str, Any]:
    denom = max(float(store.get("denom", 0.0)), 1.0)
    case_count = max(float(store.get("cases", 0.0)), 1.0)
    return {
        "changed_pixel_rate": float(store.get("changed", 0.0) / denom),
        "unnecessary_change_rate": float(store.get("unnecessary", 0.0) / denom),
        "regression_rate": float(store.get("regression", 0.0) / denom),
        "safe_improvement_rate": float(store.get("safe_improvement", 0.0) / denom),
        "mean_residual_abs": float(store.get("residual_abs", 0.0) / denom),
        "case_count": int(store.get("cases", 0.0)),
        "mean_case_changed_pixel_rate": float(store.get("case_changed_rate", 0.0) / case_count),
        "mean_case_regression_rate": float(store.get("case_regression_rate", 0.0) / case_count),
    }


def update_gate_mean_diagnostics(
    store: dict[str, dict[str, float]],
    *,
    prior_variant_type: str,
    outputs: dict[str, torch.Tensor],
    item_index: int,
    confidence: torch.Tensor,
) -> None:
    gate = outputs.get("change_gate", outputs.get("residual_gate"))
    if not isinstance(gate, torch.Tensor):
        return
    if gate.ndim != 4 or item_index >= gate.shape[0]:
        return
    sample_confidence = confidence[item_index : item_index + 1].float().clamp(0.0, 1.0)
    denom = sample_confidence.flatten(1).sum().clamp_min(1.0)
    sample_gate = gate[item_index : item_index + 1].float()
    if sample_gate.shape[1] == 1:
        roughness_gate = sample_gate
        metallic_gate = sample_gate
    else:
        roughness_gate = sample_gate[:, 0:1]
        metallic_gate = sample_gate[:, 1:2]
    total_gate = sample_gate.mean(dim=1, keepdim=True)
    bucket = store.setdefault(
        str(prior_variant_type or "unknown"),
        {
            "count": 0.0,
            "gate_mean": 0.0,
            "roughness_gate_mean": 0.0,
            "metallic_gate_mean": 0.0,
        },
    )
    bucket["count"] += 1.0
    bucket["gate_mean"] += float((total_gate * sample_confidence).flatten(1).sum().detach().cpu().item() / denom.item())
    bucket["roughness_gate_mean"] += float(
        (roughness_gate * sample_confidence).flatten(1).sum().detach().cpu().item() / denom.item()
    )
    bucket["metallic_gate_mean"] += float(
        (metallic_gate * sample_confidence).flatten(1).sum().detach().cpu().item() / denom.item()
    )


def finalize_gate_mean_diagnostics(store: dict[str, dict[str, float]]) -> dict[str, Any]:
    rows = {}
    for variant, bucket in sorted(store.items()):
        count = max(float(bucket.get("count", 0.0)), 1.0)
        rows[variant] = {
            "count": int(bucket.get("count", 0.0)),
            "gate_mean": float(bucket.get("gate_mean", 0.0) / count),
            "roughness_gate_mean": float(bucket.get("roughness_gate_mean", 0.0) / count),
            "metallic_gate_mean": float(bucket.get("metallic_gate_mean", 0.0) / count),
        }
    return rows


def update_residual_gate_diagnostics(
    store: dict[str, float],
    cases: list[dict[str, Any]],
    *,
    object_id: str,
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    margin: float,
) -> None:
    residual = (refined - baseline).abs()
    target_delta = (target - baseline).abs()
    baseline_error = (baseline - target).abs()
    refined_error = (refined - target).abs()
    confidence = confidence.clamp(0.0, 1.0)
    denom = float(confidence.sum().detach().cpu().item()) * float(refined.shape[0])
    if denom <= 0.0:
        return
    changed = (residual > margin).to(refined.dtype) * confidence
    unnecessary = ((target_delta < margin).to(refined.dtype) * changed)
    regression = ((refined_error > (baseline_error + 1e-6)).to(refined.dtype) * confidence)
    safe_improvement = ((refined_error < (baseline_error - 1e-6)).to(refined.dtype) * confidence)
    changed_sum = float(changed.sum().detach().cpu().item())
    regression_sum = float(regression.sum().detach().cpu().item())
    store["denom"] += denom
    store["changed"] += changed_sum
    store["unnecessary"] += float(unnecessary.sum().detach().cpu().item())
    store["regression"] += regression_sum
    store["safe_improvement"] += float(safe_improvement.sum().detach().cpu().item())
    store["residual_abs"] += float((residual * confidence).sum().detach().cpu().item())
    store["cases"] += 1.0
    store["case_changed_rate"] += changed_sum / denom
    store["case_regression_rate"] += regression_sum / denom
    cases.append(
        {
            "object_id": object_id,
            "changed_pixel_rate": changed_sum / denom,
            "unnecessary_change_rate": float(unnecessary.sum().detach().cpu().item()) / denom,
            "regression_rate": regression_sum / denom,
            "safe_improvement_rate": float(safe_improvement.sum().detach().cpu().item()) / denom,
            "mean_residual_abs": float((residual * confidence).sum().detach().cpu().item()) / denom,
        }
    )


def _metric_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def update_validation_special_metrics(
    store: dict[str, float],
    *,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    _ = batch, baseline
    confidence = confidence.float().clamp(0.0, 1.0)
    error = (refined.float() - target.float()).abs()
    channel_weight = confidence.expand_as(error)

    target_edge = rm_gradient_magnitude(target.float()).detach()
    edge_max = target_edge.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    edge_norm = (target_edge / edge_max).clamp(0.0, 1.0)
    edge_band = F.max_pool2d(edge_norm, kernel_size=7, stride=1, padding=3).clamp(0.0, 1.0)
    boundary_weight = edge_band.expand_as(error) * channel_weight
    interior_weight = (1.0 - edge_band).expand_as(error) * channel_weight
    boundary_denom = boundary_weight.sum()
    interior_denom = interior_weight.sum()
    if _metric_float(boundary_denom) > 0.0 and _metric_float(interior_denom) > 0.0:
        boundary_error = (error * boundary_weight).sum() / boundary_denom.clamp_min(1.0)
        interior_error = (error * interior_weight).sum() / interior_denom.clamp_min(1.0)
        store["boundary_bleed_score"] += _metric_float(boundary_error - interior_error)
        store["boundary_bleed_count"] += 1.0

    metallic_target = (target[:, 1:2].float() >= METALLIC_THRESHOLD).to(refined.dtype)
    metallic_pred = (refined[:, 1:2].float() >= METALLIC_THRESHOLD).to(refined.dtype)
    metal_denom = confidence.sum()
    if _metric_float(metal_denom) > 0.0:
        confusion = ((metallic_pred != metallic_target).to(refined.dtype) * confidence).sum()
        store["metal_confusion_rate"] += _metric_float(confusion / metal_denom.clamp_min(1.0))
        store["metal_confusion_count"] += 1.0

    highlight_mask = (target[:, 0:1].float() <= 0.35).to(refined.dtype) * confidence
    highlight_denom = highlight_mask.sum()
    if _metric_float(highlight_denom) > 0.0:
        highlight_error = (error.mean(dim=1, keepdim=True) * highlight_mask).sum()
        store["highlight_localization_error"] += _metric_float(highlight_error / highlight_denom.clamp_min(1.0))
        store["highlight_count"] += 1.0

    refined_grad = rm_gradient_magnitude(refined.float())
    target_grad = rm_gradient_magnitude(target.float())
    grad_weight = confidence
    grad_denom = grad_weight.sum()
    if _metric_float(grad_denom) > 0.0:
        refined_mean = (refined_grad * grad_weight).sum() / grad_denom.clamp_min(1.0)
        target_mean = (target_grad * grad_weight).sum() / grad_denom.clamp_min(1.0)
        refined_centered = refined_grad - refined_mean
        target_centered = target_grad - target_mean
        var_refined = (refined_centered.square() * grad_weight).sum() / grad_denom.clamp_min(1.0)
        var_target = (target_centered.square() * grad_weight).sum() / grad_denom.clamp_min(1.0)
        if _metric_float(var_refined) > 1e-8 and _metric_float(var_target) > 1e-8:
            corr = (
                (refined_centered * target_centered * grad_weight).sum()
                / grad_denom.clamp_min(1.0)
                / (var_refined.sqrt() * var_target.sqrt()).clamp_min(1e-6)
            )
            store["rm_gradient_preservation"] += _metric_float(corr.clamp(-1.0, 1.0))
            store["rm_gradient_count"] += 1.0


def finalize_validation_special_metrics(store: dict[str, float]) -> dict[str, Any]:
    def mean(name: str, count_name: str) -> float | None:
        count = float(store.get(count_name, 0.0))
        if count <= 0.0:
            return None
        return float(store.get(name, 0.0) / count)

    return {
        "boundary_bleed_score": mean("boundary_bleed_score", "boundary_bleed_count"),
        "metal_confusion_rate": mean("metal_confusion_rate", "metal_confusion_count"),
        "highlight_localization_error": mean("highlight_localization_error", "highlight_count"),
        "rm_gradient_preservation": mean("rm_gradient_preservation", "rm_gradient_count"),
        "metric_availability": {
            "boundary_bleed_score_batches": int(store.get("boundary_bleed_count", 0.0)),
            "metal_confusion_rate_batches": int(store.get("metal_confusion_count", 0.0)),
            "highlight_localization_error_batches": int(store.get("highlight_count", 0.0)),
            "rm_gradient_preservation_batches": int(store.get("rm_gradient_count", 0.0)),
        },
    }

def compute_validation_selection_metric(
    val_payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    variant_order = [
        "near_gt_prior",
        "mild_gap_prior",
        "medium_gap_prior",
        "large_gap_prior",
        "no_prior_bootstrap",
    ]
    fixed_percentiles = {
        "near_gt_prior": 0.95,
        "mild_gap_prior": 0.75,
        "medium_gap_prior": 0.50,
        "large_gap_prior": 0.25,
        "no_prior_bootstrap": 0.05,
    }
    uv_total = float((val_payload.get("uv_mae") or {}).get("total", 0.0) or 0.0)
    input_prior_total = float(
        (
            val_payload.get("input_prior_uv_mae")
            or val_payload.get("baseline_uv_mae")
            or {}
        ).get("total", 0.0)
        or 0.0
    )
    uv_gain = float((val_payload.get("improvement_uv_mae") or {}).get("total", input_prior_total - uv_total) or 0.0)
    if args.validation_selection_metric == "uv_total":
        return uv_total, {
            "uv_total": uv_total,
            "input_prior_total": input_prior_total,
            "uv_gain": uv_gain,
            "selection_metric": uv_total,
            "lower_is_better": True,
        }

    render_proxy = val_payload.get("render_proxy_validation") or {}
    residual_diag = val_payload.get("residual_gate_diagnostics") or {}
    view_penalty = 0.0
    mse_penalty = 0.0
    psnr_penalty = 0.0
    render_guard_available = bool(render_proxy.get("available"))
    if render_guard_available:
        view_penalty = max(0.0, -float(render_proxy.get("view_rm_mae_delta", 0.0) or 0.0))
        mse_penalty = max(0.0, -float(render_proxy.get("proxy_rm_mse_delta", 0.0) or 0.0))
        psnr_penalty = max(0.0, -float(render_proxy.get("proxy_rm_psnr_delta", 0.0) or 0.0))
    regression_penalty = max(
        0.0,
        float(residual_diag.get("regression_rate", 0.0) or 0.0),
    )
    variant_gain_rows = {}
    for group_key, metrics in (val_payload.get("improvement_group_metrics") or {}).items():
        if not str(group_key).startswith("prior_variant_type/"):
            continue
        variant_name = str(group_key).split("/", 1)[1]
        variant_gain_rows[variant_name] = float((metrics or {}).get("total_mae", 0.0) or 0.0)
    available_variant_gains = [variant_gain_rows[name] for name in variant_order if name in variant_gain_rows]
    variant_balanced_uv_gain = (
        float(sum(available_variant_gains) / max(len(available_variant_gains), 1))
        if available_variant_gains
        else uv_gain
    )
    case_entries = list(((val_payload.get("case_level") or {}).get("entries") or []))
    near_gt_entries = []
    withprior_entries = []
    potential_rows_by_variant: dict[str, list[float]] = {name: [] for name in variant_order}
    baseline_values = []
    for entry in case_entries:
        baseline_value = float(entry.get("baseline_total_mae", 0.0) or 0.0)
        baseline_values.append(baseline_value)
    sorted_baselines = sorted(baseline_values)
    case_prior_aware_rows = []
    for entry in case_entries:
        case_id = str(entry.get("case_id") or "")
        variant_name = case_id.rsplit("|", 1)[-1] if "|" in case_id else "unknown"
        avg_gain = float(entry.get("avg_improvement_total", 0.0) or 0.0)
        baseline_value = float(entry.get("baseline_total_mae", 0.0) or 0.0)
        relative_error_reduction = avg_gain / max(baseline_value, 1.0e-6)
        if sorted_baselines:
            empirical_percentile = (
                sum(1 for value in sorted_baselines if value >= baseline_value) / max(len(sorted_baselines), 1)
            )
        else:
            empirical_percentile = 0.5
        fixed_weight = float(getattr(args, "hybrid_prior_percentile_fixed_weight", 0.7))
        empirical_weight = float(getattr(args, "hybrid_prior_percentile_empirical_weight", 0.3))
        hybrid_prior_quality_percentile = (
            fixed_weight * fixed_percentiles.get(variant_name, 0.5)
            + empirical_weight * empirical_percentile
        )
        potential_weight = 1.0 / max(
            1.0 - hybrid_prior_quality_percentile,
            float(getattr(args, "hybrid_prior_min_potential", 0.10)),
        )
        potential_normalized_gain = relative_error_reduction * potential_weight
        case_prior_aware_rows.append(
            {
                "case_id": case_id,
                "prior_variant_type": variant_name,
                "absolute_gain": avg_gain,
                "baseline_total_mae": baseline_value,
                "relative_error_reduction": relative_error_reduction,
                "fixed_prior_quality_percentile": fixed_percentiles.get(variant_name, 0.5),
                "empirical_prior_quality_percentile": empirical_percentile,
                "hybrid_prior_quality_percentile": hybrid_prior_quality_percentile,
                "potential_weight": potential_weight,
                "potential_normalized_gain": potential_normalized_gain,
            }
        )
        if variant_name == "near_gt_prior":
            near_gt_entries.append(avg_gain)
        if variant_name != "no_prior_bootstrap":
            withprior_entries.append(avg_gain)
        if variant_name in potential_rows_by_variant:
            potential_rows_by_variant[variant_name].append(potential_normalized_gain)
    near_gt_regression_rate = (
        sum(1 for value in near_gt_entries if value < -1.0e-6) / max(len(near_gt_entries), 1)
        if near_gt_entries
        else 0.0
    )
    withprior_regression_rate = (
        sum(1 for value in withprior_entries if value < -1.0e-6) / max(len(withprior_entries), 1)
        if withprior_entries
        else 0.0
    )
    near_gt_penalty = (
        float(getattr(args, "selection_metric_near_gt_regression_multiplier", 2.0))
        * near_gt_regression_rate
    )
    withprior_penalty = (
        float(getattr(args, "selection_metric_withprior_regression_multiplier", 1.5))
        * withprior_regression_rate
    )
    variant_potential_rows = {
        name: (
            float(sum(values) / max(len(values), 1))
            if values
            else None
        )
        for name, values in potential_rows_by_variant.items()
    }
    available_variant_potential = [value for value in variant_potential_rows.values() if value is not None]
    variant_balanced_potential_gain = (
        float(sum(available_variant_potential) / max(len(available_variant_potential), 1))
        if available_variant_potential
        else 0.0
    )
    penalty_total = (
        float(args.selection_view_rm_penalty) * view_penalty
        + float(getattr(args, "selection_mse_penalty", 0.5)) * mse_penalty
        + float(args.selection_psnr_penalty) * psnr_penalty
        + float(args.selection_residual_regression_penalty) * regression_penalty
    )
    if args.validation_selection_metric == "hybrid_potential_gain_render_guarded":
        selection_metric = -variant_balanced_potential_gain + penalty_total + near_gt_penalty + withprior_penalty
    elif args.validation_selection_metric == "variant_balanced_gain_render_guarded":
        selection_metric = -variant_balanced_uv_gain + penalty_total + near_gt_penalty + withprior_penalty
    elif args.validation_selection_metric == "gain_render_guarded":
        selection_metric = -uv_gain + penalty_total
    else:
        selection_metric = uv_total + penalty_total
    return selection_metric, {
        "uv_total": uv_total,
        "input_prior_total": input_prior_total,
        "uv_gain": uv_gain,
        "variant_balanced_uv_gain": variant_balanced_uv_gain,
        "by_variant_gain": {name: variant_gain_rows.get(name) for name in variant_order},
        "by_variant_potential_normalized_gain": variant_potential_rows,
        "case_prior_aware": case_prior_aware_rows,
        "variant_balanced_potential_gain": variant_balanced_potential_gain,
        "view_penalty": view_penalty,
        "mse_penalty": mse_penalty,
        "psnr_penalty": psnr_penalty,
        "regression_penalty": regression_penalty,
        "near_gt_regression_rate": near_gt_regression_rate,
        "withprior_regression_rate": withprior_regression_rate,
        "near_gt_penalty": near_gt_penalty,
        "withprior_penalty": withprior_penalty,
        "penalty_total": penalty_total,
        "render_guard_available": render_guard_available,
        "selection_metric": selection_metric,
        "lower_is_better": True,
    }
