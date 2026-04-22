from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bake_material_refine_uv_targets import bake_uv_targets
from sf3d.material_refine.manifest_quality import audit_manifest


def save_gray(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(value, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L").save(path)


def save_rgba(path: Path, alpha: np.ndarray) -> None:
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = 180
    rgba[..., 3] = (np.clip(alpha, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def test_bake_uv_targets_synthesizes_nontrivial_prior(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    uv_dir = bundle_dir / "uv"
    uv_dir.mkdir(parents=True, exist_ok=True)
    albedo_path = uv_dir / "uv_albedo.png"
    target_roughness_src = tmp_path / "gt_roughness.png"
    target_metallic_src = tmp_path / "gt_metallic.png"

    alpha = np.ones((64, 64), dtype=np.float32)
    alpha[:6, :] = 0.0
    save_rgba(albedo_path, alpha)
    roughness = np.tile(np.linspace(0.1, 0.9, 64, dtype=np.float32), (64, 1))
    metallic = np.tile(np.linspace(0.9, 0.1, 64, dtype=np.float32), (64, 1))
    save_gray(target_roughness_src, roughness)
    save_gray(target_metallic_src, metallic)

    payload = bake_uv_targets(
        bundle_dir=bundle_dir,
        uv_albedo_path=albedo_path,
        uv_prior_roughness_path=None,
        uv_prior_metallic_path=None,
        uv_target_roughness_source_path=target_roughness_src,
        uv_target_metallic_source_path=target_metallic_src,
        canonical_buffer_root=None,
        atlas_resolution=64,
        default_roughness=0.5,
        default_metallic=0.0,
        synthesize_nontrivial_prior=True,
        allow_prior_copy_fallback=True,
    )

    assert payload["target_source_type"] == "gt_render_baked"
    assert payload["target_is_prior_copy"] is False
    assert float(payload["target_prior_identity"]) < 0.95
    assert float(payload["target_confidence_mean"]) > 0.60
    assert float(payload["target_coverage"]) > 0.80


def test_audit_manifest_blocks_auxiliary_placeholder_records(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    uv_dir = bundle_dir / "uv"
    buffer_root = bundle_dir / "buffers"
    view_dir = buffer_root / "front_studio"
    view_dir.mkdir(parents=True, exist_ok=True)

    save_rgba(view_dir / "rgba.png", np.ones((32, 32), dtype=np.float32))
    save_gray(view_dir / "roughness.png", np.full((32, 32), 0.4, dtype=np.float32))
    save_gray(view_dir / "metallic.png", np.full((32, 32), 0.2, dtype=np.float32))
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB").save(view_dir / "uv.png")
    (buffer_root / "_field_sources.json").write_text(
        json.dumps(
            {
                "views": {
                    "front_studio": {
                        "fields": {
                            "rgba": "rendered",
                            "uv": "synthetic_placeholder",
                            "roughness": "rendered",
                            "metallic": "rendered",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "views.json").write_text(json.dumps(["front_studio"]), encoding="utf-8")
    uv_dir.mkdir(parents=True, exist_ok=True)
    save_gray(uv_dir / "uv_albedo.png", np.full((32, 32), 0.6, dtype=np.float32))
    save_gray(uv_dir / "uv_normal.png", np.full((32, 32), 0.5, dtype=np.float32))
    save_gray(uv_dir / "uv_prior_roughness.png", np.full((32, 32), 0.2, dtype=np.float32))
    save_gray(uv_dir / "uv_prior_metallic.png", np.full((32, 32), 0.1, dtype=np.float32))
    save_gray(uv_dir / "uv_target_roughness.png", np.full((32, 32), 0.9, dtype=np.float32))
    save_gray(uv_dir / "uv_target_metallic.png", np.full((32, 32), 0.6, dtype=np.float32))
    save_gray(uv_dir / "uv_target_confidence.png", np.full((32, 32), 0.9, dtype=np.float32))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": "canonical_asset_record_v1",
                "records": [
                    {
                        "object_id": "aux_001",
                        "source_name": "3D-FUTURE_candidate",
                        "generator_id": "3d_future_candidate",
                        "license_bucket": "custom_tianchi_terms",
                        "supervision_role": "auxiliary_upgrade_queue",
                        "supervision_tier": "strong",
                        "canonical_mesh_path": str((bundle_dir / "mesh.glb").resolve()),
                        "canonical_glb_path": str((bundle_dir / "mesh.glb").resolve()),
                        "uv_albedo_path": str((uv_dir / "uv_albedo.png").resolve()),
                        "uv_normal_path": str((uv_dir / "uv_normal.png").resolve()),
                        "uv_prior_roughness_path": str((uv_dir / "uv_prior_roughness.png").resolve()),
                        "uv_prior_metallic_path": str((uv_dir / "uv_prior_metallic.png").resolve()),
                        "uv_target_roughness_path": str((uv_dir / "uv_target_roughness.png").resolve()),
                        "uv_target_metallic_path": str((uv_dir / "uv_target_metallic.png").resolve()),
                        "uv_target_confidence_path": str((uv_dir / "uv_target_confidence.png").resolve()),
                        "canonical_views_json": str((bundle_dir / "views.json").resolve()),
                        "canonical_buffer_root": str(buffer_root.resolve()),
                        "view_supervision_ready": False,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (bundle_dir / "mesh.glb").write_bytes(b"glTF")

    payload = audit_manifest(manifest_path)
    summary = payload["summary"]
    assert summary["paper_stage_eligible_records"] == 0
    assert summary["effective_view_supervision_record_rate"] == 0.0
    assert summary["supervision_role_counts"]["auxiliary_upgrade_queue"] == 1


def test_audit_manifest_flags_high_identity_targets(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    uv_dir = bundle_dir / "uv"
    buffer_root = bundle_dir / "buffers"
    view_dir = buffer_root / "front_studio"
    view_dir.mkdir(parents=True, exist_ok=True)

    save_rgba(view_dir / "rgba.png", np.ones((32, 32), dtype=np.float32))
    save_gray(view_dir / "mask.png", np.ones((32, 32), dtype=np.float32))
    save_gray(view_dir / "depth.png", np.full((32, 32), 0.5, dtype=np.float32))
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB").save(view_dir / "normal.png")
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB").save(view_dir / "position.png")
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB").save(view_dir / "uv.png")
    save_gray(view_dir / "visibility.png", np.ones((32, 32), dtype=np.float32))
    save_gray(view_dir / "roughness.png", np.full((32, 32), 0.7, dtype=np.float32))
    save_gray(view_dir / "metallic.png", np.full((32, 32), 0.7, dtype=np.float32))
    (buffer_root / "_field_sources.json").write_text(
        json.dumps(
            {
                "views": {
                    "front_studio": {
                        "fields": {
                            "rgba": "rendered",
                            "mask": "rendered",
                            "depth": "rendered",
                            "normal": "rendered",
                            "position": "rendered",
                            "uv": "rendered",
                            "visibility": "rendered",
                            "roughness": "rendered",
                            "metallic": "rendered",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "views.json").write_text(json.dumps(["front_studio"]), encoding="utf-8")
    uv_dir.mkdir(parents=True, exist_ok=True)
    save_gray(uv_dir / "uv_albedo.png", np.full((32, 32), 0.6, dtype=np.float32))
    save_gray(uv_dir / "uv_normal.png", np.full((32, 32), 0.5, dtype=np.float32))
    prior_roughness = np.full((32, 32), 0.65, dtype=np.float32)
    prior_metallic = np.full((32, 32), 0.68, dtype=np.float32)
    target_roughness = np.full((32, 32), 0.70, dtype=np.float32)
    target_metallic = np.full((32, 32), 0.72, dtype=np.float32)
    save_gray(uv_dir / "uv_prior_roughness.png", prior_roughness)
    save_gray(uv_dir / "uv_prior_metallic.png", prior_metallic)
    save_gray(uv_dir / "uv_target_roughness.png", target_roughness)
    save_gray(uv_dir / "uv_target_metallic.png", target_metallic)
    save_gray(uv_dir / "uv_target_confidence.png", np.full((32, 32), 0.95, dtype=np.float32))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": "canonical_asset_record_v1",
                "records": [
                    {
                        "object_id": "abo_high_identity_001",
                        "source_name": "ABO_locked_core",
                        "generator_id": "abo_locked_core",
                        "license_bucket": "cc_by_nc_4_0",
                        "supervision_role": "paper_main",
                        "supervision_tier": "strong",
                        "canonical_mesh_path": str((bundle_dir / "mesh.glb").resolve()),
                        "canonical_glb_path": str((bundle_dir / "mesh.glb").resolve()),
                        "uv_albedo_path": str((uv_dir / "uv_albedo.png").resolve()),
                        "uv_normal_path": str((uv_dir / "uv_normal.png").resolve()),
                        "uv_prior_roughness_path": str((uv_dir / "uv_prior_roughness.png").resolve()),
                        "uv_prior_metallic_path": str((uv_dir / "uv_prior_metallic.png").resolve()),
                        "uv_target_roughness_path": str((uv_dir / "uv_target_roughness.png").resolve()),
                        "uv_target_metallic_path": str((uv_dir / "uv_target_metallic.png").resolve()),
                        "uv_target_confidence_path": str((uv_dir / "uv_target_confidence.png").resolve()),
                        "canonical_views_json": str((bundle_dir / "views.json").resolve()),
                        "canonical_buffer_root": str(buffer_root.resolve()),
                        "view_supervision_ready": True,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (bundle_dir / "mesh.glb").write_bytes(b"glTF")

    payload = audit_manifest(manifest_path)
    row = payload["records"][0]
    assert row["target_prior_identity"] >= 0.95
    assert row["target_is_prior_copy"] is True
    assert row["paper_stage_eligible"] is False
    assert payload["summary"]["target_prior_identity_rate"] == 1.0
