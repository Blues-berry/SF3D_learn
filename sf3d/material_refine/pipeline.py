from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sf3d.material_refine.io import (
    apply_refined_maps_to_mesh,
    extract_mesh_material_inputs,
    save_atlas_bundle,
)
from sf3d.material_refine.model import MaterialRefiner


def _infer_device(cuda_device_index: int | None = None, fallback: str | None = None) -> str:
    if fallback is not None:
        return fallback
    if torch.cuda.is_available():
        if cuda_device_index is None:
            return "cuda"
        return f"cuda:{cuda_device_index}"
    return "cpu"


class MaterialRefinementPipeline:
    def __init__(
        self,
        model: MaterialRefiner,
        *,
        device: str = "cpu",
        atlas_size: int = 512,
        buffer_resolution: int = 256,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.atlas_size = atlas_size
        self.buffer_resolution = buffer_resolution

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        cuda_device_index: int | None = None,
        model_cfg_overrides: dict[str, Any] | None = None,
    ) -> "MaterialRefinementPipeline":
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        cfg = dict(checkpoint.get("model_cfg", {}))
        if model_cfg_overrides:
            cfg.update(model_cfg_overrides)
        model = MaterialRefiner(cfg)
        model.load_state_dict(checkpoint["model"], strict=True)
        runtime_device = _infer_device(cuda_device_index=cuda_device_index, fallback=device)
        atlas_size = int(checkpoint.get("atlas_size", 512))
        buffer_resolution = int(checkpoint.get("buffer_resolution", 256))
        return cls(
            model,
            device=runtime_device,
            atlas_size=atlas_size,
            buffer_resolution=buffer_resolution,
        )

    def _prepare_batch_from_mesh(
        self,
        mesh,
        *,
        atlas_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        target_atlas_size = atlas_size or self.atlas_size
        inputs = extract_mesh_material_inputs(mesh, target_atlas_size)
        batch = {
            "uv_albedo": inputs["uv_albedo"].unsqueeze(0).to(self.device),
            "uv_normal": inputs["uv_normal"].unsqueeze(0).to(self.device),
            "uv_prior_roughness": inputs["uv_prior_roughness"].unsqueeze(0).to(self.device),
            "uv_prior_metallic": inputs["uv_prior_metallic"].unsqueeze(0).to(self.device),
            "uv_prior_confidence": inputs["uv_prior_confidence"].unsqueeze(0).to(self.device),
            "input_prior_roughness": inputs["uv_prior_roughness"].unsqueeze(0).to(self.device),
            "input_prior_metallic": inputs["uv_prior_metallic"].unsqueeze(0).to(self.device),
            "input_prior_confidence": inputs["uv_prior_confidence"].unsqueeze(0).to(self.device),
            "view_features": torch.zeros(
                1,
                1,
                self.model.cfg.view_input_channels,
                self.buffer_resolution,
                self.buffer_resolution,
                dtype=torch.float32,
                device=self.device,
            ),
            "view_uvs": None,
            "view_masks": torch.zeros(
                1,
                1,
                1,
                self.buffer_resolution,
                self.buffer_resolution,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        return batch

    @torch.no_grad()
    def refine_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device_batch: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        outputs = self.model(device_batch)
        return {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in outputs.items()
        }

    @torch.no_grad()
    def refine_mesh(
        self,
        mesh,
        *,
        atlas_size: int | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        batch = self._prepare_batch_from_mesh(mesh, atlas_size=atlas_size)
        outputs = self.refine_batch(batch)
        refined_mesh = apply_refined_maps_to_mesh(
            mesh,
            outputs["refined"][0, 0],
            outputs["refined"][0, 1],
            basecolor=batch["uv_albedo"][0].cpu(),
            normal_map=batch["uv_normal"][0].cpu(),
        )
        atlas_paths = None
        if output_dir is not None:
            atlas_paths = save_atlas_bundle(
                Path(output_dir),
                baseline_roughness=outputs["baseline"][0, 0],
                baseline_metallic=outputs["baseline"][0, 1],
                refined_roughness=outputs["refined"][0, 0],
                refined_metallic=outputs["refined"][0, 1],
                confidence=batch["uv_prior_confidence"][0].cpu(),
            )
        return {
            "mesh": refined_mesh,
            "baseline_maps": outputs["baseline"][0],
            "refined_maps": outputs["refined"][0],
            "atlas_paths": atlas_paths,
        }
