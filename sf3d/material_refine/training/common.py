from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from sf3d.material_refine.experiment import make_json_serializable

REPO_ROOT = Path(__file__).resolve().parents[3]

def format_gb(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}GB"


def format_metric(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.{digits}f}"


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return "n/a"
    total_seconds = int(round(seconds))
    days, remainder = divmod(total_seconds, 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return "n/a"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    return format_duration(seconds)


def short_path(path: str | Path | None) -> str:
    if path is None:
        return "n/a"
    try:
        path = Path(path)
        if path.is_absolute():
            return str(path.relative_to(REPO_ROOT))
    except Exception:
        pass
    return str(path)

def maybe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def maybe_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number

def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_serializable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
