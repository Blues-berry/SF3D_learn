from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


def wandb_is_available() -> bool:
    return wandb is not None


def resolve_wandb_mode(mode: str) -> str:
    normalized = str(mode or "auto").strip().lower()
    if normalized != "auto":
        return normalized
    return "online" if os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").exists() else "offline"


def parse_tag_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = [str(part).strip() for part in value]
    return [part for part in parts if part]


def flatten_for_logging(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_for_logging(value, prefix=full_key))
        else:
            flattened[full_key] = value
    return flattened


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sanitize_log_dict(logs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sanitized: dict[str, Any] = {}
    skipped: dict[str, Any] = {}
    for key, value in logs.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                skipped[key] = "non_scalar_tensor"
                continue
            value = value.detach().item()
        elif isinstance(value, np.ndarray):
            if value.size != 1:
                skipped[key] = "non_scalar_array"
                continue
            value = value.item()
        elif isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float) and not math.isfinite(value):
            skipped[key] = value
            continue

        sanitized[key] = value
    return sanitized, skipped


def maybe_init_wandb(
    *,
    enabled: bool,
    project: str,
    job_type: str,
    config: dict[str, Any],
    mode: str = "auto",
    name: str | None = None,
    group: str | None = None,
    tags: str | list[str] | None = None,
    run_id: str | None = None,
    resume: str | None = None,
    dir_path: str | Path | None = None,
) -> Any | None:
    if not enabled:
        return None
    if wandb is None:
        raise ImportError(
            "W&B logging was requested but wandb is not installed. Install it with `pip install wandb`."
        )

    resolved_mode = resolve_wandb_mode(mode)
    init_kwargs: dict[str, Any] = {
        "project": project,
        "job_type": job_type,
        "config": config,
        "mode": resolved_mode,
        "tags": parse_tag_list(tags),
    }
    if name:
        init_kwargs["name"] = name
    if group:
        init_kwargs["group"] = group
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = resume or "allow"
    if dir_path:
        init_kwargs["dir"] = str(Path(dir_path))
    online_init_timeout = float(os.environ.get("WANDB_INIT_TIMEOUT", "180"))
    offline_init_timeout = float(os.environ.get("WANDB_OFFLINE_INIT_TIMEOUT", "90"))
    if hasattr(wandb, "Settings"):
        init_kwargs["settings"] = wandb.Settings(
            start_method="thread",
            init_timeout=online_init_timeout if resolved_mode == "online" else offline_init_timeout,
            x_service_wait=60.0 if resolved_mode == "online" else 30.0,
        )
    attempts = 2 if resolved_mode == "online" else 1
    for attempt in range(1, attempts + 1):
        try:
            return wandb.init(**init_kwargs)
        except Exception as exc:
            if resolved_mode != "online" or attempt >= attempts:
                last_exc = exc
                break
            print(
                json.dumps(
                    {
                        "wandb_init_retry": attempt,
                        "wandb_init_failed": f"{type(exc).__name__}: {exc}",
                        "wandb_init_timeout": online_init_timeout,
                    },
                    ensure_ascii=False,
                )
            )
            time.sleep(5.0)
    exc = last_exc
    try:
        if resolved_mode == "online":
            print(
                json.dumps(
                    {
                        "wandb_init_failed": f"{type(exc).__name__}: {exc}",
                        "wandb_fallback_mode": "offline",
                    },
                    ensure_ascii=False,
                )
            )
            init_kwargs["mode"] = "offline"
            if hasattr(wandb, "Settings"):
                init_kwargs["settings"] = wandb.Settings(
                    start_method="thread",
                    init_timeout=offline_init_timeout,
                    x_service_wait=30.0,
                )
            return wandb.init(**init_kwargs)
        raise exc
    except Exception as fallback_exc:
        print(
            json.dumps(
                {
                    "wandb_disabled_after_init_failure": f"{type(fallback_exc).__name__}: {fallback_exc}",
                    "wandb_requested_mode": resolved_mode,
                },
                ensure_ascii=False,
            )
        )
        return None


def log_path_artifact(
    run: Any | None,
    *,
    name: str,
    artifact_type: str,
    paths: list[str | Path],
) -> None:
    if run is None or wandb is None:
        return
    artifact = wandb.Artifact(name=name, type=artifact_type)
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        if path.is_dir():
            artifact.add_dir(str(path), name=path.name)
        else:
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
