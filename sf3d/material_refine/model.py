from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sf3d.models.utils import BaseModule


def _group_count(channels: int, max_groups: int = 8) -> int:
    channels = max(1, int(channels))
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ChannelGate2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(4, channels // max(1, reduction))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (0.5 + self.net(x))


class FeatureAdapter(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GroupNorm(num_groups=_group_count(hidden), num_channels=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[3].bias)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 0:
            return x
        return x + torch.tanh(self.scale) * self.net(x)


class MultiScaleBridge2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 2)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels),
                    nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels),
                    nn.SiLU(),
                    nn.Conv2d(channels, hidden, 1),
                    nn.GroupNorm(num_groups=_group_count(hidden), num_channels=hidden),
                    nn.SiLU(),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.merge = nn.Sequential(
            nn.Conv2d(hidden * len(self.branches), channels, 1),
            nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.merge[-1].weight)
        nn.init.zeros_(self.merge[-1].bias)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = torch.cat([branch(x) for branch in self.branches], dim=1)
        return x + torch.tanh(self.scale) * self.merge(context)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels),
            nn.SiLU(),
        )
        self.channel_gate = ChannelGate2d(out_channels)
        self.residual_proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.channel_gate(self.layers(x))
        residual = self.residual_proj(x)
        return hidden + torch.tanh(self.residual_scale) * residual


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.conv(x)
        return hidden, self.pool(hidden)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int) -> None:
        super().__init__()
        self.in_conv = ConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.mid = ConvBlock(base_channels * 4, base_channels * 4)
        self.mid_gate = ChannelGate2d(base_channels * 4)
        self.mid_context = MultiScaleBridge2d(base_channels * 4)
        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels)
        self.out_adapter = FeatureAdapter(base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        skip1, x1 = self.down1(x0)
        skip2, x2 = self.down2(x1)
        mid = self.mid_context(self.mid_gate(self.mid(x2)))
        up1 = self.up1(mid, skip2)
        up2 = self.out_adapter(self.up2(up1, skip1))
        return self.out_conv(up2)


class MultiViewEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = view_features.shape
        flat = view_features.view(batch * views, channels, height, width)
        encoded = self.encoder(flat)
        return encoded.view(batch, views, encoded.shape[1], height, width)


def _stable_bucket(value: Any, bucket_count: int) -> int:
    if bucket_count <= 1:
        return 0
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value) % bucket_count
    text = str(value).strip()
    if not text:
        return 0
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return 1 + (int(digest[:8], 16) % (bucket_count - 1))


def _metadata_indices(value: Any, bucket_count: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long).remainder(max(int(bucket_count), 1))
    if isinstance(value, (list, tuple)):
        indices = [_stable_bucket(item, bucket_count) for item in value]
    else:
        indices = [_stable_bucket(value, bucket_count)]
    return torch.tensor(indices, device=device, dtype=torch.long)


def _as_uv_mask(batch: dict[str, Any], reference: torch.Tensor) -> torch.Tensor:
    uv_mask = batch.get("uv_mask")
    if isinstance(uv_mask, torch.Tensor):
        return uv_mask.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)
    return torch.ones(
        reference.shape[0],
        1,
        reference.shape[-2],
        reference.shape[-1],
        device=reference.device,
        dtype=reference.dtype,
    )


class DualPathPriorInitialization(nn.Module):
    """A. Generator/source-aware with-prior adaptation plus no-prior bootstrap."""

    def __init__(
        self,
        *,
        domain_channels: int,
        prior_feature_channels: int,
        base_channels: int,
    ) -> None:
        super().__init__()
        hidden_channels = max(8, base_channels // 2)
        prior_in_channels = 3 + 3 + 1 + 2 + 1 + domain_channels
        bootstrap_in_channels = 3 + 3 + 1 + domain_channels
        self.prior_adapter = nn.Sequential(
            ConvBlock(prior_in_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1 + prior_feature_channels, 1),
        )
        self.bootstrapper = SmallUNet(
            bootstrap_in_channels,
            out_channels=3 + prior_feature_channels,
            base_channels=hidden_channels,
        )
        adapter_out = self.prior_adapter[-1]
        nn.init.zeros_(adapter_out.weight)
        nn.init.zeros_(adapter_out.bias)
        adapter_out.bias.data[0] = 8.0
        nn.init.zeros_(self.bootstrapper.out_conv.weight)
        nn.init.zeros_(self.bootstrapper.out_conv.bias)
        self.bootstrapper.out_conv.bias.data[2] = -1.0

    def forward(
        self,
        *,
        uv_albedo: torch.Tensor,
        uv_normal: torch.Tensor,
        uv_mask: torch.Tensor,
        prior: torch.Tensor,
        prior_confidence: torch.Tensor,
        domain_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prior_confidence = prior_confidence.clamp(0.0, 1.0) * uv_mask
        adapter_raw = self.prior_adapter(
            torch.cat(
                [uv_albedo, uv_normal, uv_mask, prior, prior_confidence, domain_feat],
                dim=1,
            )
        )
        prior_reliability = torch.sigmoid(adapter_raw[:, :1]) * prior_confidence
        prior_feat = adapter_raw[:, 1:]

        bootstrap_raw = self.bootstrapper(torch.cat([uv_albedo, uv_normal, uv_mask, domain_feat], dim=1))
        bootstrap_rm = torch.sigmoid(bootstrap_raw[:, :2])
        bootstrap_confidence = torch.sigmoid(bootstrap_raw[:, 2:3]) * uv_mask
        bootstrap_feat = bootstrap_raw[:, 3:]

        rm_init = prior * prior_reliability + bootstrap_rm * (1.0 - prior_reliability)
        init_confidence = torch.where(
            prior_confidence > 0.0,
            prior_reliability,
            bootstrap_confidence,
        ).clamp(0.0, 1.0)
        prior_feat_uv = prior_feat * prior_confidence + bootstrap_feat * (1.0 - prior_confidence)
        return {
            "rm_init_uv": rm_init.clamp(0.0, 1.0),
            "init_confidence_uv": init_confidence,
            "prior_feat_uv": prior_feat_uv,
            "prior_reliability_uv": prior_reliability,
            "bootstrap_confidence_uv": bootstrap_confidence,
            "bootstrap_rm_uv": bootstrap_rm,
        }


class MaterialSensitiveMultiViewEncoder(nn.Module):
    """B. Multi-view encoder with material/highlight/view-quality side heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        material_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )
        self.context_conv = ConvBlock(out_channels, out_channels)
        self.context_merge = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels),
            nn.SiLU(),
        )
        self.feature_gate = ChannelGate2d(out_channels)
        self.material_head = nn.Conv2d(out_channels, max(2, int(material_classes)), 1)
        self.highlight_head = nn.Sequential(
            nn.Conv2d(out_channels + 2, max(8, out_channels // 2), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(8, out_channels // 2), 1, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.Conv2d(out_channels + 2, max(8, out_channels // 2), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(8, out_channels // 2), 1, 1),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(out_channels + 4, max(8, out_channels // 2)),
            nn.SiLU(),
            nn.Linear(max(8, out_channels // 2), 1),
        )
        nn.init.zeros_(self.boundary_head[-1].weight)
        nn.init.zeros_(self.boundary_head[-1].bias)

    def forward(self, view_features: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, views, channels, height, width = view_features.shape
        flat = view_features.view(batch * views, channels, height, width)
        base_encoded = self.backbone(flat)
        context = self.context_conv(F.avg_pool2d(base_encoded, kernel_size=2, stride=2, ceil_mode=True))
        context = F.interpolate(context, size=(height, width), mode="bilinear", align_corners=False)
        encoded = self.feature_gate(self.context_merge(torch.cat([base_encoded, context], dim=1)))
        rgb = flat[:, :3].clamp(0.0, 1.0)
        mask = flat[:, 3:4].clamp(0.0, 1.0) if channels > 3 else torch.ones_like(rgb[:, :1])
        normal = flat[:, 5:8] if channels >= 8 else torch.zeros_like(rgb)
        local_highlight = (rgb.amax(dim=1, keepdim=True) - rgb.mean(dim=1, keepdim=True)).clamp_min(0.0)
        silhouette = _gradient_edge(mask)
        grazing = (1.0 - normal[:, 2:3].abs().clamp(0.0, 1.0)) if channels >= 8 else torch.zeros_like(mask)
        material_logits = self.material_head(encoded)
        highlight_response = torch.sigmoid(
            self.highlight_head(torch.cat([encoded, local_highlight, silhouette], dim=1))
        )
        boundary_response = torch.sigmoid(
            self.boundary_head(torch.cat([encoded, silhouette, grazing], dim=1))
        )

        pooled_feat = encoded.mean(dim=(-2, -1))
        material_entropy = _entropy_from_logits(material_logits, dim=1)
        stats = torch.cat(
            [
                pooled_feat,
                mask.mean(dim=(-2, -1)),
                local_highlight.mean(dim=(-2, -1)),
                grazing.mean(dim=(-2, -1)),
                material_entropy.mean(dim=(-2, -1)),
            ],
            dim=1,
        )
        view_quality = torch.sigmoid(self.quality_head(stats)).view(batch, views)
        return {
            "encoded": encoded.view(batch, views, encoded.shape[1], height, width),
            "material_logits": material_logits.view(batch, views, material_logits.shape[1], height, width),
            "highlight_response": highlight_response.view(batch, views, 1, height, width),
            "view_quality": view_quality,
            "highlight_cue": local_highlight.view(batch, views, 1, height, width),
            "silhouette_boundary_cue": silhouette.view(batch, views, 1, height, width),
            "grazing_cue": grazing.view(batch, views, 1, height, width),
            "boundary_response": boundary_response.view(batch, views, 1, height, width),
            "material_entropy": material_entropy.view(batch, views, 1, height, width),
        }


class HardViewRouter(nn.Module):
    """C. Region-specific view importance estimator for global/boundary/highlight fusion."""

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        hidden_channels = max(8, feature_channels // 2)
        self.router = nn.Sequential(
            nn.Linear(feature_channels + 8, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3),
        )

    def forward(
        self,
        *,
        encoded_views: torch.Tensor,
        view_masks: torch.Tensor,
        view_importance: torch.Tensor | None,
        view_quality: torch.Tensor | None,
        highlight_response: torch.Tensor | None,
        silhouette_boundary_cue: torch.Tensor | None,
        grazing_cue: torch.Tensor | None,
        boundary_response: torch.Tensor | None,
        material_logits: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch, views, channels, _height, _width = encoded_views.shape
        pooled = encoded_views.mean(dim=(-2, -1))
        coverage = view_masks.clamp(0.0, 1.0).mean(dim=(-3, -2, -1))
        if view_quality is None:
            view_quality = torch.ones_like(coverage)
        if view_importance is None:
            view_importance = torch.ones_like(coverage)
        if highlight_response is None:
            highlight_score = torch.zeros_like(coverage)
        else:
            highlight_score = highlight_response.mean(dim=(-3, -2, -1))
        if silhouette_boundary_cue is None:
            boundary_score = torch.zeros_like(coverage)
        else:
            boundary_score = silhouette_boundary_cue.mean(dim=(-3, -2, -1))
        if grazing_cue is None:
            grazing_score = torch.zeros_like(coverage)
        else:
            grazing_score = grazing_cue.mean(dim=(-3, -2, -1))
        if boundary_response is None:
            learned_boundary_score = torch.zeros_like(coverage)
        else:
            learned_boundary_score = boundary_response.mean(dim=(-3, -2, -1))
        if material_logits is None:
            material_entropy = torch.zeros_like(coverage)
        else:
            material_entropy = _entropy_from_logits(material_logits, dim=2).mean(dim=(-3, -2, -1))
        stats = torch.cat(
            [
                pooled,
                coverage.unsqueeze(-1),
                view_quality.unsqueeze(-1),
                view_importance.to(encoded_views.device, encoded_views.dtype).unsqueeze(-1),
                highlight_score.unsqueeze(-1),
                boundary_score.unsqueeze(-1),
                grazing_score.unsqueeze(-1),
                learned_boundary_score.unsqueeze(-1),
                material_entropy.unsqueeze(-1),
            ],
            dim=-1,
        )
        logits = self.router(stats.view(batch * views, channels + 8)).view(batch, views, 3)
        base_mask = (coverage > 0.0).to(encoded_views.dtype)
        weights = torch.softmax(logits.masked_fill(base_mask.unsqueeze(-1) <= 0, -1e4), dim=1)
        weights = weights * base_mask.unsqueeze(-1)
        normalizer = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / normalizer
        return {
            "global_weight_k": weights[:, :, 0],
            "boundary_weight_k": weights[:, :, 1],
            "highlight_weight_k": weights[:, :, 2],
        }


def _normalize_per_sample(tensor: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    scale_shape = [tensor.shape[0], *([1] * (tensor.ndim - 1))]
    scale = tensor.flatten(1).amax(dim=1).view(*scale_shape).clamp_min(epsilon)
    return (tensor / scale).clamp(0.0, 1.0)


def _entropy_from_logits(logits: torch.Tensor, dim: int) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=dim)
    entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=dim, keepdim=True)
    return _normalize_per_sample(entropy)


def _gradient_edge(tensor: torch.Tensor) -> torch.Tensor:
    grad_x = torch.zeros_like(tensor)
    grad_y = torch.zeros_like(tensor)
    grad_x[:, :, :, 1:] = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs()
    grad_y[:, :, 1:, :] = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs()
    edge = (grad_x + grad_y).amax(dim=1, keepdim=True)
    return _normalize_per_sample(edge)


class UVFeatureFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def _scatter_view_features(
        self,
        atlas_size: int,
        view_features: torch.Tensor,
        view_uvs: torch.Tensor,
        view_masks: torch.Tensor,
        view_importance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views, channels, height, width = view_features.shape
        atlas_pixels = atlas_size * atlas_size
        atlas = view_features.new_zeros(batch, channels, atlas_pixels)
        counts = view_features.new_zeros(batch, 1, atlas_pixels)

        if view_importance is None:
            weights = view_features.new_ones(batch, views)
        else:
            weights = view_importance.to(device=view_features.device, dtype=view_features.dtype).clamp_min(0.0)

        finite_uv = torch.isfinite(view_uvs).all(dim=-1)
        valid = (view_masks[:, :, 0] > 0.5) & finite_uv & (weights[:, :, None, None] > 0.0)
        safe_uv = torch.where(
            finite_uv.unsqueeze(-1),
            view_uvs,
            torch.zeros_like(view_uvs),
        )
        x = (safe_uv[..., 0].clamp(0, 1) * (atlas_size - 1)).round().long()
        y = ((1.0 - safe_uv[..., 1].clamp(0, 1)) * (atlas_size - 1)).round().long()
        index = (y * atlas_size + x).reshape(batch, views * height * width)

        weighted_valid = valid.to(dtype=view_features.dtype) * weights[:, :, None, None]
        feature = view_features * weighted_valid[:, :, None]
        feature = feature.permute(0, 2, 1, 3, 4).reshape(batch, channels, views * height * width)
        count_src = weighted_valid.reshape(batch, 1, views * height * width)

        atlas.scatter_add_(2, index[:, None].expand(batch, channels, -1), feature)
        counts.scatter_add_(2, index[:, None], count_src)

        atlas = atlas.view(batch, channels, atlas_size, atlas_size)
        counts = counts.view(batch, 1, atlas_size, atlas_size)
        fused = atlas / counts.clamp_min(1.0)
        return fused, (counts > 0).float()

    def forward(
        self,
        encoded_views: torch.Tensor,
        *,
        atlas_size: int,
        view_uvs: torch.Tensor | None,
        view_masks: torch.Tensor,
        view_importance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if view_uvs is not None:
            return self._scatter_view_features(
                atlas_size,
                encoded_views,
                view_uvs,
                view_masks,
                view_importance=view_importance,
            )

        if view_importance is not None:
            weights = view_importance.to(device=encoded_views.device, dtype=encoded_views.dtype)
            weights = weights.clamp_min(0.0).view(encoded_views.shape[0], encoded_views.shape[1], 1, 1, 1)
            pooled = (encoded_views * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            pooled = encoded_views.mean(dim=1)
        pooled = F.interpolate(
            pooled,
            size=(atlas_size, atlas_size),
            mode="bilinear",
            align_corners=False,
        )
        validity = torch.ones(
            pooled.shape[0],
            1,
            atlas_size,
            atlas_size,
            device=pooled.device,
            dtype=pooled.dtype,
        )
        return pooled, validity


class TriBranchUVFusion(nn.Module):
    """D. Fuse view evidence into global, boundary, and highlight UV branches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuser = UVFeatureFusion(channels)
        self.branch_refine = ConvBlock(channels * 3 + 2, channels * 3)
        self.branch_confidence = nn.Conv2d(channels * 3 + 2, 3, 1)
        self.branch_mix_scale = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.branch_confidence.weight)
        nn.init.zeros_(self.branch_confidence.bias)

    def forward(
        self,
        encoded_views: torch.Tensor,
        *,
        atlas_size: int,
        view_uvs: torch.Tensor | None,
        view_masks: torch.Tensor,
        routing: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        global_uv_feat, global_valid = self.fuser(
            encoded_views,
            atlas_size=atlas_size,
            view_uvs=view_uvs,
            view_masks=view_masks,
            view_importance=routing.get("global_weight_k"),
        )
        boundary_uv_feat, boundary_valid = self.fuser(
            encoded_views,
            atlas_size=atlas_size,
            view_uvs=view_uvs,
            view_masks=view_masks,
            view_importance=routing.get("boundary_weight_k"),
        )
        highlight_uv_feat, highlight_valid = self.fuser(
            encoded_views,
            atlas_size=atlas_size,
            view_uvs=view_uvs,
            view_masks=view_masks,
            view_importance=routing.get("highlight_weight_k"),
        )
        view_coverage = ((global_valid + boundary_valid + highlight_valid) / 3.0).clamp(0.0, 1.0)
        branch_stack = torch.stack([global_uv_feat, boundary_uv_feat, highlight_uv_feat], dim=1)
        view_uncertainty = branch_stack.var(dim=1, unbiased=False).mean(dim=1, keepdim=True).sqrt()
        view_uncertainty = _normalize_per_sample(view_uncertainty) * view_coverage
        branch_context = torch.cat(
            [global_uv_feat, boundary_uv_feat, highlight_uv_feat, view_coverage, view_uncertainty],
            dim=1,
        )
        branch_refine = self.branch_refine(branch_context)
        refine_global, refine_boundary, refine_highlight = torch.chunk(branch_refine, 3, dim=1)
        branch_confidence = torch.sigmoid(self.branch_confidence(branch_context))
        mix_scale = torch.tanh(self.branch_mix_scale)
        global_uv_feat = global_uv_feat + mix_scale * refine_global * branch_confidence[:, 0:1]
        boundary_uv_feat = boundary_uv_feat + mix_scale * refine_boundary * branch_confidence[:, 1:2]
        highlight_uv_feat = highlight_uv_feat + mix_scale * refine_highlight * branch_confidence[:, 2:3]
        return {
            "global_uv_feat": global_uv_feat,
            "boundary_uv_feat": boundary_uv_feat,
            "highlight_uv_feat": highlight_uv_feat,
            "view_coverage_uv": view_coverage,
            "view_uncertainty_uv": view_uncertainty,
            "branch_confidence_uv": branch_confidence,
        }


class BoundarySafetyModule(nn.Module):
    """E. Predict conservative update gates near UV/material boundaries."""

    def __init__(self, *, boundary_feature_channels: int, base_channels: int) -> None:
        super().__init__()
        hidden_channels = max(8, base_channels // 2)
        in_channels = 1 + 1 + 1 + boundary_feature_channels + 3 + 3 + 2
        self.head = nn.Sequential(
            ConvBlock(in_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 3, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.head[-1].bias.data[0] = -3.0
        self.head[-1].bias.data[1] = 3.0
        self.head[-1].bias.data[2] = 2.0

    def forward(
        self,
        *,
        uv_mask: torch.Tensor,
        boundary_cues: torch.Tensor,
        boundary_uv_feat: torch.Tensor,
        uv_normal: torch.Tensor,
        uv_albedo: torch.Tensor,
        rm_init: torch.Tensor,
        strength: float,
    ) -> dict[str, torch.Tensor]:
        boundary_band = F.max_pool2d(boundary_cues.clamp(0.0, 1.0), kernel_size=7, stride=1, padding=3)
        distance_to_boundary = (1.0 - boundary_band).clamp(0.0, 1.0) * uv_mask
        raw = self.head(
            torch.cat(
                [
                    uv_mask,
                    boundary_band,
                    distance_to_boundary,
                    boundary_uv_feat,
                    uv_normal,
                    uv_albedo,
                    rm_init,
                ],
                dim=1,
            )
        )
        bleed_risk = torch.sigmoid(raw[:, :1]) * boundary_band
        safe_update_mask = torch.sigmoid(raw[:, 1:2]) * uv_mask
        boundary_stability = torch.sigmoid(raw[:, 2:3])
        strength = float(max(0.0, min(1.0, strength)))
        boundary_gate = (1.0 - strength * bleed_risk * (1.0 - safe_update_mask)).clamp(1.0 - strength, 1.0)
        return {
            "boundary_gate_uv": boundary_gate,
            "safe_update_mask_uv": safe_update_mask,
            "bleed_risk_uv": bleed_risk,
            "boundary_stability_hint_uv": boundary_stability,
            "distance_to_boundary_uv": distance_to_boundary,
        }


class MaterialTopologyReasoning(nn.Module):
    """F. Lightweight patch-token reasoning for region-level material consistency."""

    def __init__(
        self,
        *,
        in_channels: int,
        feature_channels: int,
        material_classes: int,
        patch_size: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        feature_channels = max(8, int(feature_channels))
        patch_size = max(4, int(patch_size))
        self.patch_size = patch_size
        self.pre_local = ConvBlock(in_channels, feature_channels)
        self.patch_embed = nn.Conv2d(in_channels, feature_channels, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_channels,
            nhead=max(1, int(heads)),
            dim_feedforward=max(feature_channels * 2, 32),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.reasoner = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(layers)))
        self.out = nn.Conv2d(feature_channels, feature_channels + max(2, int(material_classes)) + 1, 1)
        self.post_local = ConvBlock(feature_channels * 2, feature_channels)
        self.detail_scale = nn.Parameter(torch.zeros(1))
        self.feature_channels = feature_channels
        self.material_classes = max(2, int(material_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = x.shape[-2:]
        local_feat = self.pre_local(x)
        tokens_2d = self.patch_embed(x)
        token_height, token_width = tokens_2d.shape[-2:]
        tokens = tokens_2d.flatten(2).transpose(1, 2)
        reasoned = self.reasoner(tokens).transpose(1, 2).view(
            x.shape[0],
            self.feature_channels,
            token_height,
            token_width,
        )
        raw = self.out(reasoned)
        raw = F.interpolate(raw, size=(height, width), mode="bilinear", align_corners=False)
        topology_feat = raw[:, : self.feature_channels]
        detail_feat = self.post_local(torch.cat([topology_feat, local_feat], dim=1))
        topology_feat = topology_feat + torch.tanh(self.detail_scale) * detail_feat
        material_logits = raw[:, self.feature_channels : self.feature_channels + self.material_classes]
        region_consistency = torch.sigmoid(raw[:, self.feature_channels + self.material_classes :])
        return {
            "topology_feat_uv": topology_feat,
            "material_region_logits": material_logits,
            "region_consistency_score": region_consistency,
        }


class RenderConsistencyInverseHead(nn.Module):
    """H. Lightweight render-response proxy and inverse material consistency gate."""

    def __init__(
        self,
        *,
        feature_channels: int,
        highlight_channels: int,
        base_channels: int,
    ) -> None:
        super().__init__()
        hidden_channels = max(8, base_channels // 2)
        in_channels = feature_channels + highlight_channels + 1 + 2 + 1
        self.head = nn.Sequential(
            ConvBlock(in_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 4, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.head[-1].bias.data[0] = 2.0
        self.head[-1].bias.data[3] = -3.0

    def forward(
        self,
        *,
        global_uv_feat: torch.Tensor,
        highlight_uv_feat: torch.Tensor,
        view_coverage: torch.Tensor,
        rm_refined_proxy: torch.Tensor,
        boundary_cues: torch.Tensor,
        strength: float,
    ) -> dict[str, torch.Tensor]:
        raw = self.head(
            torch.cat(
                [global_uv_feat, highlight_uv_feat, view_coverage, rm_refined_proxy, boundary_cues],
                dim=1,
            )
        )
        support = torch.sigmoid(raw[:, :1]) * view_coverage.clamp(0.0, 1.0)
        highlight_alignment = torch.sigmoid(raw[:, 1:2])
        material_response = torch.sigmoid(raw[:, 2:3])
        inconsistency = torch.sigmoid(raw[:, 3:4]) * (1.0 - support)
        strength = float(max(0.0, min(1.0, strength)))
        inverse_gate = (1.0 - strength * inconsistency).clamp(1.0 - strength, 1.0)
        return {
            "render_support_gate": support,
            "inverse_material_gate_uv": inverse_gate,
            "inverse_highlight_alignment_uv": highlight_alignment,
            "inverse_material_response_uv": material_response,
            "inverse_inconsistency_uv": inconsistency,
        }


class MaterialEvidenceCalibrationModule(nn.Module):
    """I. Conservative material evidence calibration for no-prior metallic safety.

    Round14 showed that a single residual gate can reduce UV error while pushing
    no-prior metallic values too high.  This module estimates whether the UV
    evidence really supports a metallic update and turns that into channel-wise
    residual gates.  The roughness path stays permissive; the metallic path is
    conservative unless prior/view/topology evidence supports the change.
    """

    def __init__(
        self,
        *,
        domain_channels: int,
        material_classes: int,
        base_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.material_classes = max(2, int(material_classes))
        hidden_channels = max(8, int(hidden_channels))
        in_channels = (
            3  # albedo
            + 3  # normal
            + 2  # calibrated/init RM
            + 2  # original prior RM
            + 1  # prior confidence
            + max(0, int(domain_channels))
            + 1  # boundary cue
            + 1  # view highlight response
            + 1  # view material entropy
            + 1  # material transition
            + 1  # view uncertainty
            + 1  # safe update mask
            + 1  # bleed risk
            + 1  # topology region consistency
            + self.material_classes
            + 3  # tri-branch confidence
        )
        self.encoder = ConvBlock(in_channels, hidden_channels)
        self.context = MultiScaleBridge2d(hidden_channels)
        self.head = nn.Conv2d(hidden_channels, 6, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Safe warm-start defaults: roughness can move, no-prior metallic needs evidence.
        self.head.bias.data[0] = -2.2  # metallic evidence
        self.head.bias.data[1] = -2.0  # no-prior metallic residual gate
        self.head.bias.data[2] = 3.0  # roughness residual gate
        self.head.bias.data[3] = 1.5  # material confidence
        self.head.bias.data[4] = 2.0  # non-metal probability
        self.head.bias.data[5] = -2.4  # metallic ceiling detail

    def forward(
        self,
        *,
        uv_albedo: torch.Tensor,
        uv_normal: torch.Tensor,
        initial_rm: torch.Tensor,
        prior_rm: torch.Tensor,
        prior_confidence: torch.Tensor,
        domain_feat: torch.Tensor,
        boundary_cues: torch.Tensor,
        view_highlight_response: torch.Tensor,
        view_material_entropy: torch.Tensor,
        material_transition: torch.Tensor,
        view_uncertainty: torch.Tensor,
        safe_update_mask: torch.Tensor,
        bleed_risk: torch.Tensor,
        region_consistency: torch.Tensor,
        topology_material_logits: torch.Tensor | None,
        branch_confidence: torch.Tensor,
        uv_mask: torch.Tensor,
        strength: float,
    ) -> dict[str, torch.Tensor]:
        if topology_material_logits is None:
            topology_probs = uv_albedo.new_zeros(
                uv_albedo.shape[0],
                self.material_classes,
                uv_albedo.shape[-2],
                uv_albedo.shape[-1],
            )
        else:
            topology_probs = torch.softmax(topology_material_logits[:, : self.material_classes], dim=1)
            if topology_probs.shape[1] < self.material_classes:
                pad = uv_albedo.new_zeros(
                    uv_albedo.shape[0],
                    self.material_classes - topology_probs.shape[1],
                    uv_albedo.shape[-2],
                    uv_albedo.shape[-1],
                )
                topology_probs = torch.cat([topology_probs, pad], dim=1)

        if domain_feat.shape[1] == 0:
            domain_part = uv_albedo.new_zeros(uv_albedo.shape[0], 0, uv_albedo.shape[-2], uv_albedo.shape[-1])
        else:
            domain_part = domain_feat
        if branch_confidence.shape[1] != 3:
            branch_confidence = uv_albedo.new_zeros(
                uv_albedo.shape[0],
                3,
                uv_albedo.shape[-2],
                uv_albedo.shape[-1],
            )

        features = torch.cat(
            [
                uv_albedo,
                uv_normal,
                initial_rm,
                prior_rm,
                prior_confidence,
                domain_part,
                boundary_cues,
                view_highlight_response,
                view_material_entropy,
                material_transition,
                view_uncertainty,
                safe_update_mask,
                bleed_risk,
                region_consistency,
                topology_probs,
                branch_confidence,
            ],
            dim=1,
        )
        raw = self.head(self.context(self.encoder(features)))
        metallic_evidence = torch.sigmoid(raw[:, 0:1]) * uv_mask
        learned_metal_gate = torch.sigmoid(raw[:, 1:2])
        roughness_gate = torch.sigmoid(raw[:, 2:3]) * uv_mask
        material_confidence = torch.sigmoid(raw[:, 3:4]) * uv_mask
        nonmetal_probability = torch.sigmoid(raw[:, 4:5]) * uv_mask
        ceiling_detail = torch.sigmoid(raw[:, 5:6])

        prior_confidence = prior_confidence.clamp(0.0, 1.0) * uv_mask
        no_prior = (1.0 - prior_confidence).clamp(0.0, 1.0)
        clean_view_support = (
            view_highlight_response.clamp(0.0, 1.0)
            * (1.0 - view_uncertainty.clamp(0.0, 1.0))
            * region_consistency.clamp(0.0, 1.0)
        )
        view_evidence = (
            0.50 * metallic_evidence
            + 0.20 * clean_view_support
            + 0.18 * ceiling_detail
            + 0.12 * (1.0 - nonmetal_probability)
        ).clamp(0.0, 1.0)
        no_prior_ceiling = (0.02 + 0.80 * view_evidence).clamp(0.02, 1.0)
        prior_ceiling = (prior_rm[:, 1:2] + 0.35 * (1.0 - prior_confidence)).clamp(0.0, 1.0)
        metallic_ceiling = (prior_confidence * prior_ceiling + no_prior * no_prior_ceiling).clamp(0.0, 1.0)

        strength = float(max(0.0, min(1.0, strength)))
        nonmetal_safety = (0.50 + 0.50 * nonmetal_probability).clamp(0.0, 1.0)
        cap_strength = (strength * no_prior * nonmetal_safety).clamp(0.0, 1.0)
        capped_metallic = torch.minimum(initial_rm[:, 1:2], metallic_ceiling)
        calibrated_metallic = initial_rm[:, 1:2] * (1.0 - cap_strength) + capped_metallic * cap_strength
        calibrated_initial = torch.cat([initial_rm[:, :1], calibrated_metallic], dim=1)

        metallic_gate = prior_confidence + no_prior * learned_metal_gate
        metallic_gate = metallic_gate * (1.0 - strength * nonmetal_probability * no_prior).clamp(1.0 - strength, 1.0)
        metallic_gate = metallic_gate * (1.0 - 0.5 * strength * bleed_risk.clamp(0.0, 1.0))
        metallic_gate = metallic_gate.clamp(0.0, 1.0) * uv_mask
        residual_channel_gate = torch.cat([roughness_gate, metallic_gate], dim=1)
        return {
            "calibrated_initial_uv": calibrated_initial.clamp(0.0, 1.0),
            "residual_channel_gate_uv": residual_channel_gate.clamp(0.0, 1.0),
            "roughness_safety_gate_uv": roughness_gate.clamp(0.0, 1.0),
            "metallic_safety_gate_uv": metallic_gate.clamp(0.0, 1.0),
            "metallic_evidence_uv": metallic_evidence.clamp(0.0, 1.0),
            "metallic_ceiling_uv": metallic_ceiling.clamp(0.0, 1.0),
            "material_confidence_uv": material_confidence.clamp(0.0, 1.0),
            "nonmetal_probability_uv": nonmetal_probability.clamp(0.0, 1.0),
            "metallic_cap_strength_uv": cap_strength.clamp(0.0, 1.0),
        }


class MaterialRefiner(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        uv_input_channels: int = 9
        view_input_channels: int = 11
        view_feature_channels: int = 32
        base_channels: int = 32
        delta_scale: float = 0.35
        disable_view_fusion: bool = False
        disable_prior_inputs: bool = False
        disable_residual_head: bool = False
        enable_residual_gate: bool = False
        residual_gate_bias: float = -0.5
        min_residual_gate: float = 0.05
        max_residual_gate: float = 1.0
        prior_confidence_gate_strength: float = 0.0
        enable_boundary_context: bool = False
        boundary_context_strength: float = 0.25
        enable_material_context: bool = False
        material_context_classes: int = 3
        material_delta_scale: float = 0.06
        enable_render_consistency_gate: bool = False
        render_gate_strength: float = 0.25
        enable_dual_path_prior_init: bool = False
        enable_domain_prior_calibration: bool = False
        domain_feature_channels: int = 8
        prior_feature_channels: int = 16
        max_generator_embeddings: int = 64
        max_source_embeddings: int = 128
        enable_material_sensitive_view_encoder: bool = False
        enable_hard_view_routing: bool = False
        enable_tri_branch_fusion: bool = False
        enable_boundary_safety_module: bool = False
        boundary_safety_strength: float = 0.35
        boundary_residual_suppression_strength: float = 0.0
        view_uncertainty_residual_suppression_strength: float = 0.0
        bleed_risk_residual_suppression_strength: float = 0.0
        enable_material_topology_reasoning: bool = False
        topology_feature_channels: int = 16
        topology_patch_size: int = 16
        topology_layers: int = 2
        topology_heads: int = 2
        topology_residual_suppression_strength: float = 0.0
        enable_confidence_gated_trunk: bool = False
        uncertainty_gate_strength: float = 0.25
        enable_inverse_material_check: bool = False
        inverse_check_strength: float = 0.25
        enable_material_evidence_calibration: bool = False
        material_evidence_channels: int = 16
        material_evidence_strength: float = 0.75
        enable_evidence_update_budget: bool = False
        evidence_update_budget_strength: float = 0.70
        evidence_update_budget_floor: float = 0.04

    cfg: Config

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg or {})

    def configure(self) -> None:
        material_classes = max(2, int(self.cfg.material_context_classes))
        if self.cfg.enable_material_sensitive_view_encoder:
            self.view_encoder = MaterialSensitiveMultiViewEncoder(
                self.cfg.view_input_channels,
                self.cfg.view_feature_channels,
                material_classes=material_classes,
            )
        else:
            self.view_encoder = MultiViewEncoder(
                self.cfg.view_input_channels, self.cfg.view_feature_channels
            )
        self.view_fuser = UVFeatureFusion(self.cfg.view_feature_channels)
        if self.cfg.enable_tri_branch_fusion:
            self.tri_branch_fuser = TriBranchUVFusion(self.cfg.view_feature_channels)
        if self.cfg.enable_hard_view_routing:
            self.hard_view_router = HardViewRouter(self.cfg.view_feature_channels)

        self.domain_channels = (
            max(1, int(self.cfg.domain_feature_channels))
            if (self.cfg.enable_domain_prior_calibration or self.cfg.enable_dual_path_prior_init)
            else 0
        )
        if self.domain_channels > 0:
            self.generator_embedding = nn.Embedding(
                max(2, int(self.cfg.max_generator_embeddings)),
                self.domain_channels,
            )
            self.source_embedding = nn.Embedding(
                max(2, int(self.cfg.max_source_embeddings)),
                self.domain_channels,
            )
            self.domain_merge = nn.Sequential(
                nn.Conv2d(self.domain_channels * 2, self.domain_channels, 1),
                nn.SiLU(),
            )
        if self.cfg.enable_dual_path_prior_init:
            self.prior_initializer = DualPathPriorInitialization(
                domain_channels=self.domain_channels,
                prior_feature_channels=max(1, int(self.cfg.prior_feature_channels)),
                base_channels=self.cfg.base_channels,
            )

        init_channels = self.cfg.uv_input_channels + self.cfg.view_feature_channels + 1
        self.init_feature_adapter = FeatureAdapter(init_channels)
        self.init_head = SmallUNet(
            init_channels,
            out_channels=2,
            base_channels=self.cfg.base_channels,
        )

        fused_view_channels = self.cfg.view_feature_channels
        if self.cfg.enable_tri_branch_fusion:
            fused_view_channels = self.cfg.view_feature_channels * 3
        refine_channels = self.cfg.uv_input_channels + fused_view_channels + 1 + 2 + 1
        if self.cfg.enable_dual_path_prior_init:
            refine_channels += max(1, int(self.cfg.prior_feature_channels)) + self.domain_channels + 1
        if self.cfg.enable_tri_branch_fusion:
            refine_channels += 4
        if self.cfg.enable_material_sensitive_view_encoder:
            refine_channels += 3
        if self.cfg.enable_boundary_safety_module:
            refine_channels += 3
        if self.cfg.enable_material_topology_reasoning:
            refine_channels += max(8, int(self.cfg.topology_feature_channels)) + 1
        if self.cfg.enable_inverse_material_check:
            refine_channels += 1

        if self.cfg.enable_confidence_gated_trunk:
            refine_out_channels = 5 + material_classes
        else:
            refine_out_channels = 3 if self.cfg.enable_residual_gate else 2
        self.refine_feature_adapter = FeatureAdapter(refine_channels)
        self.refine_head = SmallUNet(
            refine_channels,
            out_channels=refine_out_channels,
            base_channels=self.cfg.base_channels,
        )
        if self.cfg.enable_confidence_gated_trunk:
            nn.init.zeros_(self.refine_head.out_conv.weight)
            nn.init.zeros_(self.refine_head.out_conv.bias)
        context_channels = max(8, min(32, self.cfg.base_channels))
        if self.cfg.enable_boundary_context:
            self.boundary_gate_head = nn.Sequential(
                ConvBlock(3, context_channels),
                nn.Conv2d(context_channels, 1, 1),
            )
        if self.cfg.enable_material_context:
            material_classes = max(2, int(self.cfg.material_context_classes))
            material_in_channels = self.cfg.uv_input_channels + 2 + 2
            self.material_context_head = SmallUNet(
                material_in_channels,
                out_channels=material_classes + 2,
                base_channels=max(8, self.cfg.base_channels // 2),
            )
        if self.cfg.enable_render_consistency_gate:
            render_gate_channels = self.cfg.view_feature_channels + 1 + 2 + 1
            self.render_gate_head = nn.Sequential(
                ConvBlock(render_gate_channels, context_channels),
                nn.Conv2d(context_channels, 1, 1),
            )
        if self.cfg.enable_boundary_safety_module:
            self.boundary_safety_module = BoundarySafetyModule(
                boundary_feature_channels=self.cfg.view_feature_channels,
                base_channels=self.cfg.base_channels,
            )
        if self.cfg.enable_material_topology_reasoning:
            self.topology_module = MaterialTopologyReasoning(
                in_channels=self.cfg.uv_input_channels + 2 + 1 + 1,
                feature_channels=max(8, int(self.cfg.topology_feature_channels)),
                material_classes=material_classes,
                patch_size=self.cfg.topology_patch_size,
                layers=self.cfg.topology_layers,
                heads=self.cfg.topology_heads,
            )
        if self.cfg.enable_inverse_material_check:
            self.inverse_check_head = RenderConsistencyInverseHead(
                feature_channels=self.cfg.view_feature_channels,
                highlight_channels=self.cfg.view_feature_channels,
                base_channels=self.cfg.base_channels,
            )
        if self.cfg.enable_material_evidence_calibration:
            self.material_evidence_calibrator = MaterialEvidenceCalibrationModule(
                domain_channels=self.domain_channels,
                material_classes=material_classes,
                base_channels=self.cfg.base_channels,
                hidden_channels=max(8, int(self.cfg.material_evidence_channels)),
            )

    def build_uv_inputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prior_roughness = batch["uv_prior_roughness"]
        prior_metallic = batch["uv_prior_metallic"]
        prior_confidence = batch["uv_prior_confidence"]
        if self.cfg.disable_prior_inputs:
            prior_roughness = torch.full_like(prior_roughness, 0.5)
            prior_metallic = torch.zeros_like(prior_metallic)
            prior_confidence = torch.zeros_like(prior_confidence)
        return torch.cat(
            [
                batch["uv_albedo"],
                batch["uv_normal"],
                prior_roughness,
                prior_metallic,
                prior_confidence,
            ],
            dim=1,
        )

    def build_domain_features(
        self,
        batch: dict[str, Any],
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if self.domain_channels <= 0:
            return reference.new_zeros(reference.shape[0], 0, reference.shape[-2], reference.shape[-1])
        generator_indices = _metadata_indices(
            batch.get("generator_id"),
            int(self.cfg.max_generator_embeddings),
            reference.device,
        )
        source_indices = _metadata_indices(
            batch.get("source_name"),
            int(self.cfg.max_source_embeddings),
            reference.device,
        )
        if generator_indices.numel() == 1 and reference.shape[0] > 1:
            generator_indices = generator_indices.expand(reference.shape[0])
        if source_indices.numel() == 1 and reference.shape[0] > 1:
            source_indices = source_indices.expand(reference.shape[0])
        generator_indices = generator_indices[: reference.shape[0]]
        source_indices = source_indices[: reference.shape[0]]
        generator_feat = self.generator_embedding(generator_indices).to(dtype=reference.dtype)
        source_feat = self.source_embedding(source_indices).to(dtype=reference.dtype)
        domain = torch.cat([generator_feat, source_feat], dim=1).view(
            reference.shape[0],
            self.domain_channels * 2,
            1,
            1,
        )
        domain = domain.expand(-1, -1, reference.shape[-2], reference.shape[-1])
        return self.domain_merge(domain)

    def build_boundary_cues(
        self,
        batch: dict[str, torch.Tensor],
        *,
        prior: torch.Tensor,
        fused_valid: torch.Tensor | None,
        view_boundary_response: torch.Tensor | None = None,
        material_transition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        albedo_edge = _gradient_edge(batch["uv_albedo"])
        normal_edge = _gradient_edge(batch["uv_normal"])
        prior_edge = _gradient_edge(prior)
        boundary = 0.40 * albedo_edge + 0.28 * normal_edge + 0.17 * prior_edge
        if view_boundary_response is not None and view_boundary_response.numel() > 0:
            boundary = boundary + 0.10 * view_boundary_response.clamp(0.0, 1.0)
        if material_transition is not None and material_transition.numel() > 0:
            boundary = boundary + 0.05 * material_transition.clamp(0.0, 1.0)
        if fused_valid is not None:
            boundary = boundary + 0.10 * _gradient_edge(fused_valid)
        return _normalize_per_sample(boundary)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        atlas_size = batch["uv_albedo"].shape[-1]
        uv_inputs = self.build_uv_inputs(batch)
        uv_mask = _as_uv_mask(batch, batch["uv_albedo"])
        domain_feat = self.build_domain_features(batch, reference=batch["uv_albedo"])

        encoded_payload = self.view_encoder(batch["view_features"])
        view_side_outputs: dict[str, torch.Tensor] = {}
        if isinstance(encoded_payload, dict):
            encoded_views = encoded_payload["encoded"]
            view_side_outputs = encoded_payload
        else:
            encoded_views = encoded_payload

        routing: dict[str, torch.Tensor] | None = None
        tri_fusion_outputs: dict[str, torch.Tensor] | None = None
        branch_confidence_uv = batch["uv_albedo"].new_zeros(
            batch["uv_albedo"].shape[0],
            3,
            atlas_size,
            atlas_size,
        )
        view_boundary_response_uv = batch["uv_albedo"].new_zeros(
            batch["uv_albedo"].shape[0],
            1,
            atlas_size,
            atlas_size,
        )
        view_highlight_response_uv = torch.zeros_like(view_boundary_response_uv)
        view_material_entropy_uv = torch.zeros_like(view_boundary_response_uv)
        material_transition_uv = torch.zeros_like(view_boundary_response_uv)
        if self.cfg.disable_view_fusion:
            fused_views = torch.zeros(
                batch["uv_albedo"].shape[0],
                self.cfg.view_feature_channels,
                atlas_size,
                atlas_size,
                device=batch["uv_albedo"].device,
                dtype=batch["uv_albedo"].dtype,
            )
            fused_valid = torch.zeros(
                batch["uv_albedo"].shape[0],
                1,
                atlas_size,
                atlas_size,
                device=batch["uv_albedo"].device,
                dtype=batch["uv_albedo"].dtype,
            )
            global_uv_feat = fused_views
            boundary_uv_feat = fused_views
            highlight_uv_feat = fused_views
            view_uncertainty_uv = fused_valid
        else:
            if self.cfg.enable_hard_view_routing:
                routing = self.hard_view_router(
                    encoded_views=encoded_views,
                    view_masks=batch["view_masks"],
                    view_importance=batch.get("view_importance"),
                    view_quality=view_side_outputs.get("view_quality"),
                    highlight_response=view_side_outputs.get("highlight_response"),
                    silhouette_boundary_cue=view_side_outputs.get("silhouette_boundary_cue"),
                    grazing_cue=view_side_outputs.get("grazing_cue"),
                    boundary_response=view_side_outputs.get("boundary_response"),
                    material_logits=view_side_outputs.get("material_logits"),
                )
            if self.cfg.enable_tri_branch_fusion:
                if routing is None:
                    base_weight = batch.get("view_importance")
                    if base_weight is None:
                        base_weight = torch.ones(
                            encoded_views.shape[0],
                            encoded_views.shape[1],
                            device=encoded_views.device,
                            dtype=encoded_views.dtype,
                        )
                    base_weight = base_weight.to(device=encoded_views.device, dtype=encoded_views.dtype)
                    base_weight = base_weight / base_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    routing = {
                        "global_weight_k": base_weight,
                        "boundary_weight_k": base_weight,
                        "highlight_weight_k": base_weight,
                    }
                tri_fusion_outputs = self.tri_branch_fuser(
                    encoded_views,
                    atlas_size=atlas_size,
                    view_uvs=batch.get("view_uvs"),
                    view_masks=batch["view_masks"],
                    routing=routing,
                )
                global_uv_feat = tri_fusion_outputs["global_uv_feat"]
                boundary_uv_feat = tri_fusion_outputs["boundary_uv_feat"]
                highlight_uv_feat = tri_fusion_outputs["highlight_uv_feat"]
                fused_views = global_uv_feat
                fused_valid = tri_fusion_outputs["view_coverage_uv"]
                view_uncertainty_uv = tri_fusion_outputs["view_uncertainty_uv"]
                branch_confidence_uv = tri_fusion_outputs["branch_confidence_uv"]
            else:
                fused_views, fused_valid = self.view_fuser(
                    encoded_views,
                    atlas_size=atlas_size,
                    view_uvs=batch.get("view_uvs"),
                    view_masks=batch["view_masks"],
                    view_importance=batch.get("view_importance"),
                )
                global_uv_feat = fused_views
                boundary_uv_feat = fused_views
                highlight_uv_feat = fused_views
                view_uncertainty_uv = torch.zeros_like(fused_valid)

            if self.cfg.enable_material_sensitive_view_encoder and view_side_outputs:
                global_weight = routing.get("global_weight_k") if routing is not None else batch.get("view_importance")
                if global_weight is None:
                    global_weight = view_side_outputs.get("view_quality")

                boundary_response_views = view_side_outputs.get("boundary_response")
                highlight_response_views = view_side_outputs.get("highlight_response")
                material_entropy_views = view_side_outputs.get("material_entropy")
                if material_entropy_views is None and view_side_outputs.get("material_logits") is not None:
                    material_entropy_views = _entropy_from_logits(
                        view_side_outputs["material_logits"],
                        dim=2,
                    )
                side_views = [
                    boundary_response_views if boundary_response_views is not None else torch.zeros_like(view_boundary_response_uv).unsqueeze(1).expand(-1, encoded_views.shape[1], -1, encoded_views.shape[-2], encoded_views.shape[-1]),
                    highlight_response_views if highlight_response_views is not None else torch.zeros_like(view_highlight_response_uv).unsqueeze(1).expand(-1, encoded_views.shape[1], -1, encoded_views.shape[-2], encoded_views.shape[-1]),
                    material_entropy_views if material_entropy_views is not None else torch.zeros_like(view_material_entropy_uv).unsqueeze(1).expand(-1, encoded_views.shape[1], -1, encoded_views.shape[-2], encoded_views.shape[-1]),
                ]
                side_uv, _ = self.view_fuser(
                    torch.cat(side_views, dim=2),
                    atlas_size=atlas_size,
                    view_uvs=batch.get("view_uvs"),
                    view_masks=batch["view_masks"],
                    view_importance=global_weight,
                )
                view_boundary_response_uv = side_uv[:, 0:1]
                view_highlight_response_uv = side_uv[:, 1:2]
                view_material_entropy_uv = side_uv[:, 2:3]
                material_transition_uv = _gradient_edge(view_material_entropy_uv)

        prior = torch.cat([batch["uv_prior_roughness"], batch["uv_prior_metallic"]], dim=1)
        prior_confidence = batch["uv_prior_confidence"]
        if self.cfg.disable_prior_inputs:
            prior = torch.cat(
                [
                    torch.full_like(batch["uv_prior_roughness"], 0.5),
                    torch.zeros_like(batch["uv_prior_metallic"]),
                ],
                dim=1,
            )
            prior_confidence = torch.zeros_like(prior_confidence)

        prior_init_outputs: dict[str, torch.Tensor] | None = None
        if self.cfg.enable_dual_path_prior_init:
            prior_init_outputs = self.prior_initializer(
                uv_albedo=batch["uv_albedo"],
                uv_normal=batch["uv_normal"],
                uv_mask=uv_mask,
                prior=prior,
                prior_confidence=prior_confidence,
                domain_feat=domain_feat,
            )
            coarse = prior_init_outputs["bootstrap_rm_uv"]
            initial = prior_init_outputs["rm_init_uv"]
            init_confidence_uv = prior_init_outputs["init_confidence_uv"]
            prior_feat_uv = prior_init_outputs["prior_feat_uv"]
        else:
            init_features = torch.cat([uv_inputs, fused_views, fused_valid], dim=1)
            init_features = self.init_feature_adapter(init_features)
            coarse = torch.sigmoid(self.init_head(init_features))
            initial = coarse * (1.0 - prior_confidence) + prior * prior_confidence
            init_confidence_uv = prior_confidence
            prior_feat_uv = batch["uv_albedo"].new_zeros(
                batch["uv_albedo"].shape[0],
                0,
                atlas_size,
                atlas_size,
            )
        boundary_cues = self.build_boundary_cues(
            batch,
            prior=prior,
            fused_valid=fused_valid,
            view_boundary_response=view_boundary_response_uv,
            material_transition=material_transition_uv,
        )

        boundary_safety_outputs: dict[str, torch.Tensor] | None = None
        safe_update_mask_uv = torch.ones_like(prior_confidence)
        bleed_risk_uv = torch.zeros_like(prior_confidence)
        if self.cfg.enable_boundary_safety_module:
            boundary_safety_outputs = self.boundary_safety_module(
                uv_mask=uv_mask,
                boundary_cues=boundary_cues,
                boundary_uv_feat=boundary_uv_feat,
                uv_normal=batch["uv_normal"],
                uv_albedo=batch["uv_albedo"],
                rm_init=initial,
                strength=self.cfg.boundary_safety_strength,
            )
            safe_update_mask_uv = boundary_safety_outputs["safe_update_mask_uv"]
            bleed_risk_uv = boundary_safety_outputs["bleed_risk_uv"]

        topology_outputs: dict[str, torch.Tensor] | None = None
        topology_feat_uv = batch["uv_albedo"].new_zeros(batch["uv_albedo"].shape[0], 0, atlas_size, atlas_size)
        region_consistency_score = torch.ones_like(prior_confidence)
        topology_material_logits = None
        if self.cfg.enable_material_topology_reasoning:
            topology_outputs = self.topology_module(
                torch.cat([uv_inputs, initial, boundary_cues, prior_confidence], dim=1)
            )
            topology_feat_uv = topology_outputs["topology_feat_uv"]
            region_consistency_score = topology_outputs["region_consistency_score"]
            topology_material_logits = topology_outputs["material_region_logits"]

        material_evidence_outputs: dict[str, torch.Tensor] | None = None
        residual_channel_gate_uv = initial.new_ones(initial.shape)
        roughness_safety_gate_uv = torch.ones_like(prior_confidence)
        metallic_safety_gate_uv = torch.ones_like(prior_confidence)
        metallic_evidence_uv = torch.zeros_like(prior_confidence)
        metallic_ceiling_uv = torch.ones_like(prior_confidence)
        material_confidence_uv = torch.ones_like(prior_confidence)
        nonmetal_probability_uv = torch.zeros_like(prior_confidence)
        metallic_cap_strength_uv = torch.zeros_like(prior_confidence)
        if self.cfg.enable_material_evidence_calibration:
            material_evidence_outputs = self.material_evidence_calibrator(
                uv_albedo=batch["uv_albedo"],
                uv_normal=batch["uv_normal"],
                initial_rm=initial,
                prior_rm=prior,
                prior_confidence=prior_confidence,
                domain_feat=domain_feat,
                boundary_cues=boundary_cues,
                view_highlight_response=view_highlight_response_uv,
                view_material_entropy=view_material_entropy_uv,
                material_transition=material_transition_uv,
                view_uncertainty=view_uncertainty_uv,
                safe_update_mask=safe_update_mask_uv,
                bleed_risk=bleed_risk_uv,
                region_consistency=region_consistency_score,
                topology_material_logits=topology_material_logits,
                branch_confidence=branch_confidence_uv,
                uv_mask=uv_mask,
                strength=self.cfg.material_evidence_strength,
            )
            initial = material_evidence_outputs["calibrated_initial_uv"]
            residual_channel_gate_uv = material_evidence_outputs["residual_channel_gate_uv"]
            roughness_safety_gate_uv = material_evidence_outputs["roughness_safety_gate_uv"]
            metallic_safety_gate_uv = material_evidence_outputs["metallic_safety_gate_uv"]
            metallic_evidence_uv = material_evidence_outputs["metallic_evidence_uv"]
            metallic_ceiling_uv = material_evidence_outputs["metallic_ceiling_uv"]
            material_confidence_uv = material_evidence_outputs["material_confidence_uv"]
            nonmetal_probability_uv = material_evidence_outputs["nonmetal_probability_uv"]
            metallic_cap_strength_uv = material_evidence_outputs["metallic_cap_strength_uv"]

        inverse_outputs: dict[str, torch.Tensor] | None = None
        inverse_inconsistency_uv = torch.zeros_like(prior_confidence)

        view_feature_for_refine = (
            torch.cat([global_uv_feat, boundary_uv_feat, highlight_uv_feat], dim=1)
            if self.cfg.enable_tri_branch_fusion
            else fused_views
        )
        refine_parts = [uv_inputs, view_feature_for_refine, fused_valid, initial, prior_confidence]
        if self.cfg.enable_dual_path_prior_init:
            refine_parts.extend([prior_feat_uv, domain_feat, init_confidence_uv])
        if self.cfg.enable_tri_branch_fusion:
            refine_parts.extend([view_uncertainty_uv, branch_confidence_uv])
        if self.cfg.enable_material_sensitive_view_encoder:
            refine_parts.extend(
                [
                    view_boundary_response_uv,
                    view_highlight_response_uv,
                    material_transition_uv,
                ]
            )
        if self.cfg.enable_boundary_safety_module:
            refine_parts.extend(
                [
                    boundary_safety_outputs["boundary_gate_uv"],
                    safe_update_mask_uv,
                    bleed_risk_uv,
                ]
            )
        if self.cfg.enable_material_topology_reasoning:
            refine_parts.extend([topology_feat_uv, region_consistency_score])
        if self.cfg.enable_inverse_material_check:
            refine_parts.append(inverse_inconsistency_uv)
        refine_features = self.refine_feature_adapter(torch.cat(refine_parts, dim=1))

        uncertainty_uv = torch.zeros_like(prior_confidence)
        boundary_stability_uv = torch.ones_like(prior_confidence)
        trunk_material_logits = None
        if self.cfg.disable_residual_head:
            delta = torch.zeros_like(initial)
            residual_gate = torch.zeros_like(prior_confidence)
        else:
            refine_raw = self.refine_head(refine_features)
            delta = torch.tanh(refine_raw[:, :2]) * self.cfg.delta_scale
            residual_gate = torch.ones_like(prior_confidence)
            if self.cfg.enable_confidence_gated_trunk:
                residual_gate = torch.sigmoid(refine_raw[:, 2:3] + self.cfg.residual_gate_bias)
                min_gate = float(max(0.0, min(1.0, self.cfg.min_residual_gate)))
                residual_gate = min_gate + (1.0 - min_gate) * residual_gate
                uncertainty_uv = torch.sigmoid(refine_raw[:, 3:4])
                boundary_stability_uv = torch.sigmoid(refine_raw[:, 4:5])
                trunk_material_logits = refine_raw[:, 5:]
                if self.cfg.uncertainty_gate_strength > 0.0:
                    strength = float(max(0.0, min(1.0, self.cfg.uncertainty_gate_strength)))
                    residual_gate = residual_gate * (1.0 - strength * uncertainty_uv)
                residual_gate = residual_gate * (0.5 + 0.5 * boundary_stability_uv)
            elif self.cfg.enable_residual_gate:
                residual_gate = torch.sigmoid(refine_raw[:, 2:3] + self.cfg.residual_gate_bias)
                min_gate = float(max(0.0, min(1.0, self.cfg.min_residual_gate)))
                residual_gate = min_gate + (1.0 - min_gate) * residual_gate
            if self.cfg.prior_confidence_gate_strength > 0.0:
                strength = float(max(0.0, min(1.0, self.cfg.prior_confidence_gate_strength)))
                confidence_gate = 1.0 - prior_confidence.clamp(0.0, 1.0) * strength
                residual_gate = residual_gate * confidence_gate.clamp_min(float(self.cfg.min_residual_gate))
        residual_gate = residual_gate * residual_channel_gate_uv
        max_gate = float(max(0.0, min(1.0, self.cfg.max_residual_gate)))
        if max_gate < 1.0:
            residual_gate = residual_gate.clamp_max(max_gate)
        boundary_gate = torch.ones_like(prior_confidence)
        if self.cfg.enable_boundary_context:
            strength = float(max(0.0, min(1.0, self.cfg.boundary_context_strength)))
            boundary_logits = self.boundary_gate_head(
                torch.cat([boundary_cues, prior_confidence, fused_valid], dim=1)
            )
            boundary_gain = (2.0 * torch.sigmoid(boundary_logits) - 1.0) * (
                0.25 + 0.75 * boundary_cues
            )
            boundary_gate = (1.0 + strength * boundary_gain).clamp(1.0 - strength, 1.0 + strength)
            residual_gate = residual_gate * boundary_gate
        if self.cfg.enable_boundary_safety_module:
            boundary_gate = boundary_gate * boundary_safety_outputs["boundary_gate_uv"]
            residual_gate = residual_gate * boundary_safety_outputs["boundary_gate_uv"]
            residual_gate = residual_gate * (0.25 + 0.75 * safe_update_mask_uv)
            boundary_suppression = float(
                max(0.0, min(1.0, self.cfg.boundary_residual_suppression_strength))
            )
            if boundary_suppression > 0.0:
                residual_gate = residual_gate * (1.0 - boundary_suppression * boundary_cues).clamp(
                    1.0 - boundary_suppression,
                    1.0,
                )
        view_uncertainty_residual_gate_uv = torch.ones_like(prior_confidence)
        view_uncertainty_suppression = float(
            max(0.0, min(1.0, self.cfg.view_uncertainty_residual_suppression_strength))
        )
        if view_uncertainty_suppression > 0.0:
            view_uncertainty_residual_gate_uv = (
                1.0 - view_uncertainty_suppression * view_uncertainty_uv.clamp(0.0, 1.0)
            ).clamp(1.0 - view_uncertainty_suppression, 1.0)
            residual_gate = residual_gate * view_uncertainty_residual_gate_uv
        bleed_risk_residual_gate_uv = torch.ones_like(prior_confidence)
        bleed_risk_suppression = float(
            max(0.0, min(1.0, self.cfg.bleed_risk_residual_suppression_strength))
        )
        if bleed_risk_suppression > 0.0:
            bleed_risk_residual_gate_uv = (
                1.0 - bleed_risk_suppression * bleed_risk_uv.clamp(0.0, 1.0)
            ).clamp(1.0 - bleed_risk_suppression, 1.0)
            residual_gate = residual_gate * bleed_risk_residual_gate_uv
        topology_residual_gate_uv = torch.ones_like(prior_confidence)
        topology_suppression = float(
            max(0.0, min(1.0, self.cfg.topology_residual_suppression_strength))
        )
        if topology_suppression > 0.0 and self.cfg.enable_material_topology_reasoning:
            topology_residual_gate_uv = (
                1.0 - topology_suppression * (1.0 - region_consistency_score.clamp(0.0, 1.0))
            ).clamp(1.0 - topology_suppression, 1.0)
            residual_gate = residual_gate * topology_residual_gate_uv

        evidence_update_budget_uv = torch.ones_like(initial)
        evidence_update_support_uv = torch.ones_like(prior_confidence)
        if self.cfg.enable_evidence_update_budget:
            budget_strength = float(
                max(0.0, min(1.0, self.cfg.evidence_update_budget_strength))
            )
            budget_floor = float(max(0.0, min(0.5, self.cfg.evidence_update_budget_floor)))
            branch_support = branch_confidence_uv.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
            view_cleanliness = (1.0 - view_uncertainty_uv.clamp(0.0, 1.0)).clamp(0.0, 1.0)
            boundary_safety = (
                safe_update_mask_uv.clamp(0.0, 1.0)
                * (1.0 - 0.75 * bleed_risk_uv.clamp(0.0, 1.0))
            ).clamp(0.0, 1.0)
            topology_support = region_consistency_score.clamp(0.0, 1.0)
            confidence_support = material_confidence_uv.clamp(0.0, 1.0)
            prior_support = prior_confidence.clamp(0.0, 1.0)
            no_prior = (1.0 - prior_support).clamp(0.0, 1.0)
            generic_support = (
                0.30 * branch_support
                + 0.25 * view_cleanliness
                + 0.20 * boundary_safety
                + 0.15 * topology_support
                + 0.10 * confidence_support
            ).clamp(0.0, 1.0)
            roughness_support = (
                prior_support * (0.70 + 0.30 * generic_support)
                + no_prior * generic_support
            ).clamp(0.0, 1.0)
            metallic_support = (
                0.40 * metallic_evidence_uv.clamp(0.0, 1.0)
                + 0.20 * (1.0 - nonmetal_probability_uv.clamp(0.0, 1.0))
                + 0.15 * view_highlight_response_uv.clamp(0.0, 1.0)
                + 0.15 * topology_support
                + 0.10 * branch_support
            ).clamp(0.0, 1.0)
            metallic_support = (metallic_support * boundary_safety * view_cleanliness).clamp(
                0.0,
                1.0,
            )
            metallic_budget = (
                prior_support * (0.55 + 0.45 * generic_support)
                + no_prior * metallic_support
            ).clamp(0.0, 1.0)
            roughness_budget = budget_floor + (1.0 - budget_floor) * roughness_support
            metallic_budget = budget_floor + (1.0 - budget_floor) * metallic_budget
            evidence_update_budget_uv = torch.cat([roughness_budget, metallic_budget], dim=1)
            evidence_update_budget_uv = evidence_update_budget_uv.clamp(budget_floor, 1.0) * uv_mask
            evidence_update_support_uv = generic_support.clamp(0.0, 1.0) * uv_mask
            evidence_budget_gate = (
                1.0 - budget_strength * (1.0 - evidence_update_budget_uv)
            ).clamp(budget_floor, 1.0)
            residual_gate = residual_gate * evidence_budget_gate

        material_logits = trunk_material_logits if trunk_material_logits is not None else topology_material_logits
        material_delta = torch.zeros_like(initial)
        if self.cfg.enable_material_context:
            material_raw = self.material_context_head(
                torch.cat([uv_inputs, initial, boundary_cues, prior_confidence], dim=1)
            )
            material_classes = max(2, int(self.cfg.material_context_classes))
            material_logits = material_raw[:, :material_classes]
            material_delta = torch.tanh(material_raw[:, material_classes : material_classes + 2])
            material_delta = material_delta * float(max(0.0, self.cfg.material_delta_scale))
            delta = delta + material_delta

        provisional_refined = (initial + delta * residual_gate).clamp(0.0, 1.0)

        render_support_gate = torch.ones_like(prior_confidence)
        if self.cfg.enable_render_consistency_gate:
            strength = float(max(0.0, min(1.0, self.cfg.render_gate_strength)))
            render_feature = (
                fused_views
                + 0.35 * highlight_uv_feat * (0.5 + 0.5 * view_uncertainty_uv)
                + 0.20 * global_uv_feat * view_highlight_response_uv
                + 0.15 * boundary_uv_feat * material_transition_uv
            )
            render_logits = self.render_gate_head(
                torch.cat([render_feature, fused_valid, provisional_refined, boundary_cues], dim=1)
            )
            render_support = torch.sigmoid(render_logits) * fused_valid.clamp(0.0, 1.0)
            render_support_gate = (1.0 - strength * (1.0 - render_support)).clamp(
                1.0 - strength,
                1.0,
            )
            residual_gate = residual_gate * render_support_gate
            provisional_refined = (initial + delta * residual_gate).clamp(0.0, 1.0)
        inverse_material_gate_uv = torch.ones_like(prior_confidence)
        if self.cfg.enable_inverse_material_check:
            inverse_outputs = self.inverse_check_head(
                global_uv_feat=global_uv_feat,
                highlight_uv_feat=highlight_uv_feat,
                view_coverage=fused_valid,
                rm_refined_proxy=provisional_refined,
                boundary_cues=boundary_cues,
                strength=self.cfg.inverse_check_strength,
            )
            inverse_inconsistency_uv = inverse_outputs["inverse_inconsistency_uv"]
            inverse_material_gate_uv = inverse_outputs["inverse_material_gate_uv"]
            residual_gate = residual_gate * inverse_material_gate_uv
            provisional_refined = (initial + delta * residual_gate).clamp(0.0, 1.0)
        refined = provisional_refined

        return {
            "coarse": coarse,
            "baseline": prior,
            "initial": initial,
            "rm_init_uv": initial,
            "init_confidence_uv": init_confidence_uv,
            "refined": refined,
            "residual_delta": delta,
            "residual_gate": residual_gate,
            "uncertainty_uv": uncertainty_uv,
            "boundary_cues": boundary_cues,
            "boundary_gate": boundary_gate,
            "safe_update_mask_uv": safe_update_mask_uv,
            "bleed_risk_uv": bleed_risk_uv,
            "boundary_stability_uv": boundary_stability_uv,
            "material_logits": material_logits,
            "material_region_logits": topology_material_logits,
            "material_delta": material_delta,
            "material_evidence_outputs": material_evidence_outputs,
            "residual_channel_gate_uv": residual_channel_gate_uv,
            "roughness_safety_gate_uv": roughness_safety_gate_uv,
            "metallic_safety_gate_uv": metallic_safety_gate_uv,
            "metallic_evidence_uv": metallic_evidence_uv,
            "metallic_ceiling_uv": metallic_ceiling_uv,
            "material_confidence_uv": material_confidence_uv,
            "nonmetal_probability_uv": nonmetal_probability_uv,
            "metallic_cap_strength_uv": metallic_cap_strength_uv,
            "render_support_gate": render_support_gate,
            "inverse_material_gate_uv": inverse_material_gate_uv,
            "view_uncertainty_residual_gate_uv": view_uncertainty_residual_gate_uv,
            "bleed_risk_residual_gate_uv": bleed_risk_residual_gate_uv,
            "topology_residual_gate_uv": topology_residual_gate_uv,
            "evidence_update_budget_uv": evidence_update_budget_uv,
            "evidence_update_support_uv": evidence_update_support_uv,
            "provisional_refined_uv": provisional_refined,
            "fused_views": fused_views,
            "fused_validity": fused_valid,
            "global_uv_feat": global_uv_feat,
            "boundary_uv_feat": boundary_uv_feat,
            "highlight_uv_feat": highlight_uv_feat,
            "view_coverage_uv": fused_valid,
            "view_uncertainty_uv": view_uncertainty_uv,
            "branch_confidence_uv": branch_confidence_uv,
            "view_boundary_response_uv": view_boundary_response_uv,
            "view_highlight_response_uv": view_highlight_response_uv,
            "view_material_entropy_uv": view_material_entropy_uv,
            "material_transition_uv": material_transition_uv,
            "prior_feat_uv": prior_feat_uv,
            "domain_feat_uv": domain_feat,
            "topology_feat_uv": topology_feat_uv,
            "region_consistency_score": region_consistency_score,
            "view_material_logits": view_side_outputs.get("material_logits"),
            "view_highlight_response": view_side_outputs.get("highlight_response"),
            "view_quality": view_side_outputs.get("view_quality"),
            "view_routing": routing,
            "tri_fusion_outputs": tri_fusion_outputs,
            "prior_init_outputs": prior_init_outputs,
            "boundary_safety_outputs": boundary_safety_outputs,
            "topology_outputs": topology_outputs,
            "inverse_outputs": inverse_outputs,
        }
