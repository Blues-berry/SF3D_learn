from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "abo_rm_mini"
DEFAULT_BLENDER_BIN = Path(
    "/4T/CXY/Neural_Gaffer_original/scripts/Objavarse_rendering/blender-3.2.2-linux-x64/blender"
)
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "objects_200.json"
BUCKET_HTTP_ROOT = "https://amazon-berkeley-objects.s3.amazonaws.com"
DEFAULT_DOWNLOAD_CONNECT_TIMEOUT = 30
DEFAULT_DOWNLOAD_READ_TIMEOUT = 180
DEFAULT_DOWNLOAD_RETRIES = 6
DEFAULT_DOWNLOAD_BACKOFF_SECONDS = 2.0
PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]
VIEWS = [
    {
        "name": "front_studio",
        "azimuth": 0.0,
        "elevation": 18.0,
        "distance": 2.0,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "studio_small_08_1k.hdr"),
    },
    {
        "name": "three_quarter_indoor",
        "azimuth": 45.0,
        "elevation": 28.0,
        "distance": 2.0,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "peppermint_powerplant_1k.hdr"),
    },
    {
        "name": "side_neon",
        "azimuth": 110.0,
        "elevation": 16.0,
        "distance": 2.05,
        "hdri": str(REPO_ROOT / "demo_files" / "hdri" / "neon_photostudio_1k.hdr"),
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--object-manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER_BIN)
    parser.add_argument(
        "--cuda-device-index",
        type=str,
        default="0",
        help="Physical CUDA device index to isolate for Blender rendering.",
    )
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--cycles-samples", type=int, default=32)
    parser.add_argument("--max-objects", type=int, default=200)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument("--download-connect-timeout", type=int, default=DEFAULT_DOWNLOAD_CONNECT_TIMEOUT)
    parser.add_argument("--download-read-timeout", type=int, default=DEFAULT_DOWNLOAD_READ_TIMEOUT)
    parser.add_argument("--download-retries", type=int, default=DEFAULT_DOWNLOAD_RETRIES)
    parser.add_argument("--download-backoff-seconds", type=float, default=DEFAULT_DOWNLOAD_BACKOFF_SECONDS)
    parser.add_argument("--allow-env-proxy", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing object manifest: {path}")
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        objects = payload.get("objects")
        if isinstance(objects, list):
            return objects
    if isinstance(payload, list):
        return payload
    raise RuntimeError(f"Unsupported manifest payload in {path}")


def make_http_session(allow_env_proxy: bool) -> requests.Session:
    session = requests.Session()
    session.trust_env = allow_env_proxy
    return session


def scrub_proxy_env(env: dict[str, str], allow_env_proxy: bool) -> dict[str, str]:
    if allow_env_proxy:
        return env
    cleaned = dict(env)
    for key in PROXY_ENV_KEYS:
        cleaned.pop(key, None)
    return cleaned


def describe_proxy_mode(allow_env_proxy: bool) -> dict[str, object]:
    configured = {key: os.environ[key] for key in PROXY_ENV_KEYS if os.environ.get(key)}
    return {
        "allow_env_proxy": allow_env_proxy,
        "active_env_proxy_keys": sorted(configured.keys()),
        "active_env_proxy_values": configured,
    }


def http_get_with_retry(
    session: requests.Session,
    url: str,
    *,
    timeout: tuple[int, int],
    retries: int,
    backoff_seconds: float,
    stream: bool = False,
) -> requests.Response:
    errors = []
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout, stream=stream)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            message = f"attempt {attempt}/{retries}: {exc}"
            errors.append(message)
            if attempt == retries:
                break
            sleep_seconds = min(backoff_seconds * (2 ** (attempt - 1)), 30.0)
            print(f"[download retry] {url} :: {message}; sleeping {sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Download failed for {url}: {' | '.join(errors)}")


def download_glb(
    obj: dict,
    cache_dir: Path,
    *,
    session: requests.Session,
    timeout: tuple[int, int],
    retries: int,
    backoff_seconds: float,
) -> Path:
    local_keys = ["local_path", "source_model_path", "object_path"]
    for key in local_keys:
        value = obj.get(key)
        if value:
            local_path = Path(value)
            if local_path.exists():
                return local_path

    path = obj.get("path")
    if not path:
        raise RuntimeError(f"Object {obj.get('id')} has no model path")

    out_path = cache_dir / f"{obj['id']}.glb"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    tmp_path = out_path.with_suffix(".glb.part")
    if tmp_path.exists():
        tmp_path.unlink()
    url = f"{BUCKET_HTTP_ROOT}/3dmodels/original/{path}"
    response = http_get_with_retry(
        session,
        url,
        timeout=timeout,
        retries=retries,
        backoff_seconds=backoff_seconds,
        stream=True,
    )
    try:
        with response, tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(out_path)
    except Exception:  # noqa: BLE001
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return out_path


def run_blender_render(
    blender_bin: Path,
    glb_path: Path,
    object_out_dir: Path,
    resolution: int,
    cycles_samples: int,
    *,
    cuda_device_index: str,
    allow_env_proxy: bool,
):
    views_json = object_out_dir / "views.json"
    views_json.write_text(json.dumps(VIEWS, indent=2))
    blender_script = REPO_ROOT / "scripts" / "abo_material_passes_blender.py"
    cmd = [
        str(blender_bin),
        "-b",
        "-P",
        str(blender_script),
        "--",
        "--object-path",
        str(glb_path),
        "--output-dir",
        str(object_out_dir),
        "--views-json",
        str(views_json),
        "--resolution",
        str(resolution),
        "--cycles-samples",
        str(cycles_samples),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device_index
    env["BLENDER_CUDA_DEVICE_INDEX"] = "0"
    env = scrub_proxy_env(env, allow_env_proxy=allow_env_proxy)
    subprocess.run(cmd, check=True, env=env)


def save_mask_from_rgba(rgba_path: Path, mask_path: Path):
    rgba = np.asarray(Image.open(rgba_path).convert("RGBA")).astype(np.float32) / 255.0
    alpha = rgba[..., 3] > 0.01
    mask = (alpha.astype(np.uint8) * 255)
    Image.fromarray(mask, mode="L").save(mask_path)
    return alpha


def compute_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_partitions(mask: np.ndarray):
    interior = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").filter(ImageFilter.MinFilter(9))
    ) > 0
    edge = mask & ~interior
    return edge, interior


def scalar_stats(path: Path, mask: np.ndarray):
    arr = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    values = arr[..., 0]
    visible = values[mask]
    if visible.size == 0:
        raise RuntimeError(f"No visible pixels in {path}")
    edge_mask, interior_mask = mask_partitions(mask)
    return {
        "whole_mask": {
            "mean": float(visible.mean()),
            "std": float(visible.std()),
            "var": float(visible.var()),
            "p10": float(np.quantile(visible, 0.10)),
            "p50": float(np.quantile(visible, 0.50)),
            "p90": float(np.quantile(visible, 0.90)),
        },
        "edge_vs_interior": {
            "edge_mean": float(values[edge_mask].mean()) if edge_mask.any() else float(visible.mean()),
            "interior_mean": float(values[interior_mask].mean()) if interior_mask.any() else float(visible.mean()),
            "edge_coverage_px": int(edge_mask.sum()),
            "interior_coverage_px": int(interior_mask.sum()),
        },
    }


def highlight_stats(rgba_path: Path, mask: np.ndarray):
    arr = np.asarray(Image.open(rgba_path).convert("RGBA")).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    visible = luminance[mask]
    return {
        "brightness_mean": float(visible.mean()),
        "brightness_std": float(visible.std()),
        "brightness_var": float(visible.var()),
        "brightness_p95": float(np.quantile(visible, 0.95)),
        "brightness_p99": float(np.quantile(visible, 0.99)),
        "highlight_fraction": float(((luminance > 0.92) & mask).sum() / mask.sum()),
    }


def required_view_files(view_dir: Path):
    return [
        view_dir / "rgba.png",
        view_dir / "roughness.png",
        view_dir / "metallic.png",
    ]


def completed_stats_files(object_dir: Path) -> list[Path]:
    return [object_dir / view["name"] / "stats.json" for view in VIEWS]


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir or (output_dir / "cache" / "models"))
    render_dir = ensure_dir(output_dir / "renders")
    manifest_objects = load_manifest(args.object_manifest)[: args.max_objects]
    download_timeout = (args.download_connect_timeout, args.download_read_timeout)
    download_session = make_http_session(allow_env_proxy=args.allow_env_proxy)

    render_skips = []
    object_summaries = []

    for obj in manifest_objects:
        object_id = obj["id"]
        object_dir = ensure_dir(render_dir / object_id)
        object_summary = {
            "object_id": object_id,
            "name": obj.get("name", object_id),
            "sampling_bucket": obj.get("sampling_bucket"),
            "views": {},
            "status": "ok",
        }
        try:
            existing_stats = completed_stats_files(object_dir)
            if not args.force_render and all(path.exists() for path in existing_stats):
                for view, stats_path in zip(VIEWS, existing_stats):
                    object_summary["views"][view["name"]] = {
                        "status": "ok",
                        "stats_path": str(stats_path.resolve()),
                    }
                object_summaries.append(object_summary)
                continue
            glb_path = download_glb(
                obj,
                cache_dir,
                session=download_session,
                timeout=download_timeout,
                retries=args.download_retries,
                backoff_seconds=args.download_backoff_seconds,
            )
            needs_render = args.force_render
            for view in VIEWS:
                if not all(path.exists() for path in required_view_files(object_dir / view["name"])):
                    needs_render = True
                    break
            if not args.skip_render and needs_render:
                run_blender_render(
                    args.blender_bin,
                    glb_path,
                    object_dir,
                    resolution=args.render_resolution,
                    cycles_samples=args.cycles_samples,
                    cuda_device_index=args.cuda_device_index,
                    allow_env_proxy=args.allow_env_proxy,
                )

            for view in VIEWS:
                view_dir = ensure_dir(object_dir / view["name"])
                rgba_path = view_dir / "rgba.png"
                roughness_path = view_dir / "roughness.png"
                metallic_path = view_dir / "metallic.png"
                mask_path = view_dir / "mask.png"

                for path in [rgba_path, roughness_path, metallic_path]:
                    if not path.exists():
                        raise FileNotFoundError(f"Missing rendered file: {path}")

                mask = save_mask_from_rgba(rgba_path, mask_path)
                stats = {
                    "object_id": object_id,
                    "name": obj.get("name", object_id),
                    "label": obj.get("label", object_id),
                    "category": obj.get("category", "unknown"),
                    "sampling_bucket": obj.get("sampling_bucket"),
                    "view_name": view["name"],
                    "paths": {
                        "rgba": str(rgba_path.resolve()),
                        "mask": str(mask_path.resolve()),
                        "roughness": str(roughness_path.resolve()),
                        "metallic": str(metallic_path.resolve()),
                    },
                    "mask": {
                        "coverage_px": int(mask.sum()),
                        "coverage_ratio": float(mask.sum() / mask.size),
                        "bbox_xyxy": compute_bbox(mask),
                    },
                    "roughness": scalar_stats(roughness_path, mask),
                    "metallic": scalar_stats(metallic_path, mask),
                    "input_highlight": highlight_stats(rgba_path, mask),
                }
                stats_path = view_dir / "stats.json"
                stats_path.write_text(json.dumps(stats, indent=2))
                object_summary["views"][view["name"]] = {
                    "status": "ok",
                    "stats_path": str(stats_path.resolve()),
                }
        except Exception as exc:  # noqa: BLE001
            object_summary["status"] = "skipped"
            object_summary["skip_reason"] = str(exc)
            render_skips.append(
                {
                    "object_id": object_id,
                    "name": obj.get("name", object_id),
                    "reason": str(exc),
                }
            )
        object_summaries.append(object_summary)

    render_summary = {
        "manifest_path": str(args.object_manifest.resolve()),
        "objects_requested": len(manifest_objects),
        "objects_completed": sum(1 for item in object_summaries if item["status"] == "ok"),
        "objects_skipped": len(render_skips),
        "render_config": {
            "resolution": args.render_resolution,
            "cycles_samples": args.cycles_samples,
            "views": [view["name"] for view in VIEWS],
        },
        "download_config": {
            "connect_timeout": args.download_connect_timeout,
            "read_timeout": args.download_read_timeout,
            "retries": args.download_retries,
            "backoff_seconds": args.download_backoff_seconds,
            **describe_proxy_mode(args.allow_env_proxy),
        },
        "views_per_object": [view["name"] for view in VIEWS],
        "skipped_objects": render_skips,
    }
    (output_dir / "render_summary.json").write_text(json.dumps(render_summary, indent=2))
    (output_dir / "render_object_summary.json").write_text(json.dumps(object_summaries, indent=2))

    print(f"Render root: {render_dir}")
    print(f"Render summary: {output_dir / 'render_summary.json'}")
    print(f"Object render summary: {output_dir / 'render_object_summary.json'}")


if __name__ == "__main__":
    main()
