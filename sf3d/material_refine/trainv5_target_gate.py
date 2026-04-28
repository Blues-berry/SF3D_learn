from __future__ import annotations

from pathlib import Path
from typing import Any

TARGET_GATE_VERSION = "TrainV5_target_truth_gate_v2"
TARGET_TRUTH_REQUIRED_PATHS = (
    "canonical_buffer_root",
    "uv_target_roughness_path",
    "uv_target_metallic_path",
    "uv_target_confidence_path",
    "canonical_views_json",
)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def path_exists(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).exists()


def trainv5_target_truth_gate(
    record: dict[str, Any],
    *,
    mean_threshold: float = 0.08,
    p95_threshold: float = 0.20,
    required_path_fields: tuple[str, ...] = TARGET_TRUTH_REQUIRED_PATHS,
) -> tuple[bool, list[str]]:
    """Single TrainV5 gate: target supervision truth/alignment only.

    Prior/target similarity, target_is_prior_copy, material labels, source
    balance, and paper eligibility are diagnostics, not TrainV5 gates.
    """

    blockers: list[str] = []
    mean = finite_float(record.get("target_view_alignment_mean"))
    p95 = finite_float(record.get("target_view_alignment_p95"))
    if not bool_value(record.get("target_as_pred_pass")):
        blockers.append("target_as_pred_pass_false")
    if mean is None:
        blockers.append("missing_target_view_alignment_mean")
    elif mean >= float(mean_threshold):
        blockers.append("target_view_alignment_mean_fail")
    if p95 is None:
        blockers.append("missing_target_view_alignment_p95")
    elif p95 >= float(p95_threshold):
        blockers.append("target_view_alignment_p95_fail")
    if not bool_value(record.get("view_supervision_ready", True)):
        blockers.append("view_supervision_not_ready")
    for field in required_path_fields:
        if not path_exists(record.get(field)):
            blockers.append(f"missing_{field}")
    return not blockers, blockers


def target_prior_relation_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_is_prior_copy": bool_value(record.get("target_is_prior_copy"))
        or bool_value(record.get("copied_from_prior")),
        "target_prior_identity": finite_float(record.get("target_prior_identity")),
        "target_source_type": str(record.get("target_source_type") or ""),
    }


def with_target_truth_gate_fields(record: dict[str, Any]) -> dict[str, Any]:
    item = dict(record)
    gate_ok, blockers = trainv5_target_truth_gate(item)
    item["target_gate_version"] = TARGET_GATE_VERSION
    item["target_truth_gate_pass"] = gate_ok
    item["target_truth_gate_blockers"] = blockers
    item["target_prior_relation_diagnostic"] = target_prior_relation_diagnostics(item)
    return item
