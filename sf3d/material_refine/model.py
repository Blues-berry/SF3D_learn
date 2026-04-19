from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sf3d.models.utils import BaseModule


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, out_channels)), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, out_channels)), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


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
        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        skip1, x1 = self.down1(x0)
        skip2, x2 = self.down2(x1)
        mid = self.mid(x2)
        up1 = self.up1(mid, skip2)
        up2 = self.up2(up1, skip1)
        return self.out_conv(up2)


class MultiViewEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, out_channels)), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, out_channels)), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = view_features.shape
        flat = view_features.view(batch * views, channels, height, width)
        encoded = self.encoder(flat)
        return encoded.view(batch, views, encoded.shape[1], height, width)


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views, channels, height, width = view_features.shape
        atlas = torch.zeros(
            batch,
            channels,
            atlas_size * atlas_size,
            device=view_features.device,
            dtype=view_features.dtype,
        )
        counts = torch.zeros(
            batch,
            1,
            atlas_size * atlas_size,
            device=view_features.device,
            dtype=view_features.dtype,
        )

        for batch_idx in range(batch):
            for view_idx in range(views):
                uv = view_uvs[batch_idx, view_idx]
                mask = view_masks[batch_idx, view_idx, 0] > 0.5
                if not mask.any():
                    continue
                feature = view_features[batch_idx, view_idx].permute(1, 2, 0).reshape(-1, channels)
                uv_flat = uv.reshape(-1, 2)
                valid = mask.reshape(-1)
                valid = valid & torch.isfinite(uv_flat).all(dim=-1)
                if not valid.any():
                    continue
                uv_flat = uv_flat[valid]
                feature = feature[valid]
                x = (uv_flat[:, 0].clamp(0, 1) * (atlas_size - 1)).round().long()
                y = ((1.0 - uv_flat[:, 1].clamp(0, 1)) * (atlas_size - 1)).round().long()
                index = y * atlas_size + x
                atlas[batch_idx].scatter_add_(
                    1,
                    index.unsqueeze(0).expand(channels, -1),
                    feature.transpose(0, 1),
                )
                counts[batch_idx].scatter_add_(
                    1,
                    index.unsqueeze(0),
                    torch.ones(
                        1,
                        index.shape[0],
                        device=view_features.device,
                        dtype=view_features.dtype,
                    ),
                )

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if view_uvs is not None:
            return self._scatter_view_features(
                atlas_size,
                encoded_views,
                view_uvs,
                view_masks,
            )

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


class MaterialRefiner(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        uv_input_channels: int = 9
        view_input_channels: int = 11
        view_feature_channels: int = 32
        base_channels: int = 32
        delta_scale: float = 0.35

    cfg: Config

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg or {})

    def configure(self) -> None:
        self.view_encoder = MultiViewEncoder(
            self.cfg.view_input_channels, self.cfg.view_feature_channels
        )
        self.view_fuser = UVFeatureFusion(self.cfg.view_feature_channels)
        init_channels = self.cfg.uv_input_channels + self.cfg.view_feature_channels + 1
        refine_channels = init_channels + 3
        self.init_head = SmallUNet(
            init_channels,
            out_channels=2,
            base_channels=self.cfg.base_channels,
        )
        self.refine_head = SmallUNet(
            refine_channels,
            out_channels=2,
            base_channels=self.cfg.base_channels,
        )

    def build_uv_inputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                batch["uv_albedo"],
                batch["uv_normal"],
                batch["uv_prior_roughness"],
                batch["uv_prior_metallic"],
                batch["uv_prior_confidence"],
            ],
            dim=1,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        atlas_size = batch["uv_albedo"].shape[-1]
        uv_inputs = self.build_uv_inputs(batch)
        encoded_views = self.view_encoder(batch["view_features"])
        fused_views, fused_valid = self.view_fuser(
            encoded_views,
            atlas_size=atlas_size,
            view_uvs=batch.get("view_uvs"),
            view_masks=batch["view_masks"],
        )

        init_features = torch.cat([uv_inputs, fused_views, fused_valid], dim=1)
        coarse = torch.sigmoid(self.init_head(init_features))
        prior = torch.cat(
            [batch["uv_prior_roughness"], batch["uv_prior_metallic"]],
            dim=1,
        )
        prior_confidence = batch["uv_prior_confidence"]
        initial = coarse * (1.0 - prior_confidence) + prior * prior_confidence

        refine_features = torch.cat(
            [uv_inputs, fused_views, fused_valid, initial, prior_confidence],
            dim=1,
        )
        delta = torch.tanh(self.refine_head(refine_features)) * self.cfg.delta_scale
        refined = (initial + delta).clamp(0.0, 1.0)

        return {
            "coarse": coarse,
            "baseline": prior,
            "initial": initial,
            "refined": refined,
            "fused_views": fused_views,
            "fused_validity": fused_valid,
        }
