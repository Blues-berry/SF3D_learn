from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decide whether Round8 is safe to promote as the Round9 initialization checkpoint.",
    )
    parser.add_argument(
        "--round6-summary",
        type=Path,
        default=Path("output/material_refine_paper/stage1_round6_wandb_clean_eval/summary.json"),
    )
    parser.add_argument(
        "--round7-summary",
        type=Path,
        default=Path("output/material_refine_paper/stage1_round7_gradient_guard_eval/summary.json"),
    )
    parser.add_argument(
        "--round8-summary",
        type=Path,
        default=Path("output/material_refine_paper/stage1_round8_boundary_band_eval/summary.json"),
    )
    parser.add_argument(
        "--round8-checkpoint",
        type=Path,
        default=Path("output/material_refine_paper/stage1_round8_boundary_band/best.pt"),
    )
    parser.add_argument(
        "--fallback-checkpoint",
        type=Path,
        default=Path("output/material_refine_paper/stage1_round7_gradient_guard/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/material_refine_paper/round8_promotion_decision"),
    )
    parser.add_argument("--uv-close-ratio", type=float, default=1.05)
    parser.add_argument("--boundary-improvement-ratio", type=float, default=0.90)
    parser.add_argument("--min-psnr-delta", type=float, default=-0.05)
    parser.add_argument("--min-ssim-delta", type=float, default=-0.002)
    parser.add_argument("--min-lpips-delta", type=float, default=0.0)
    parser.add_argument("--safety-tolerance", type=float, default=0.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(payload: dict[str, Any] | None, path: list[str], default: Any = None) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def extract_metrics(payload: dict[str, Any] | None) -> dict[str, float | None]:
    return {
        "uv_refined": get_nested(payload, ["metrics_main", "uv_rm_mae", "total", "refined"]),
        "uv_delta": get_nested(payload, ["metrics_main", "uv_rm_mae", "total", "delta"]),
        "view_delta": get_nested(payload, ["metrics_main", "view_rm_mae", "total", "delta"]),
        "psnr_delta": get_nested(payload, ["metrics_main", "proxy_render_psnr", "delta"]),
        "ssim_delta": get_nested(payload, ["metrics_main", "proxy_render_ssim", "delta"]),
        "lpips_delta": get_nested(payload, ["metrics_main", "proxy_render_lpips", "delta"]),
        "boundary_refined": get_nested(
            payload, ["metrics_material_specific", "boundary_bleed_score", "refined"]
        ),
        "boundary_delta": get_nested(payload, ["metrics_material_specific", "boundary_bleed_score", "delta"]),
        "safety_score": get_nested(
            payload, ["metrics_material_specific", "prior_residual_safety", "safety_score"]
        ),
    }


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def pass_condition(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round8 Promotion Decision",
        "",
        f"- status: `{payload['status']}`",
        f"- promote_to_round9_init: `{payload['promote_to_round9_init']}`",
        f"- recommended_checkpoint: `{payload['recommended_checkpoint']}`",
        f"- reason: `{payload['reason']}`",
        "",
        "## Conditions",
        "",
    ]
    for item in payload.get("conditions", []):
        lines.append(f"- `{item['name']}`: `{item['passed']}`")
    lines.extend(["", "## Metrics", ""])
    for round_name, metrics in payload.get("metrics", {}).items():
        lines.append(f"### {round_name}")
        for key, value in metrics.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    round6 = load_json(args.round6_summary)
    round7 = load_json(args.round7_summary)
    round8 = load_json(args.round8_summary)
    metrics = {
        "round6": extract_metrics(round6),
        "round7": extract_metrics(round7),
        "round8": extract_metrics(round8),
    }
    if round8 is None:
        payload = {
            "status": "pending",
            "promote_to_round9_init": False,
            "recommended_checkpoint": str(args.fallback_checkpoint),
            "reason": "round8_summary_missing_wait_for_full_eval",
            "round8_summary": str(args.round8_summary),
            "metrics": metrics,
            "conditions": [],
        }
    else:
        r6 = {key: finite_number(value) for key, value in metrics["round6"].items()}
        r7 = {key: finite_number(value) for key, value in metrics["round7"].items()}
        r8 = {key: finite_number(value) for key, value in metrics["round8"].items()}
        conditions = [
            pass_condition(
                "uv_close_to_round7",
                r8["uv_refined"] is not None
                and r7["uv_refined"] is not None
                and r8["uv_refined"] <= r7["uv_refined"] * float(args.uv_close_ratio),
                {"round8_uv": r8["uv_refined"], "round7_uv": r7["uv_refined"]},
            ),
            pass_condition(
                "boundary_better_than_round7",
                r8["boundary_refined"] is not None
                and r7["boundary_refined"] is not None
                and r8["boundary_refined"]
                <= r7["boundary_refined"] * float(args.boundary_improvement_ratio),
                {
                    "round8_boundary": r8["boundary_refined"],
                    "round7_boundary": r7["boundary_refined"],
                },
            ),
            pass_condition(
                "psnr_not_markedly_degraded",
                r8["psnr_delta"] is not None and r8["psnr_delta"] >= float(args.min_psnr_delta),
                {"round8_psnr_delta": r8["psnr_delta"], "threshold": args.min_psnr_delta},
            ),
            pass_condition(
                "ssim_not_markedly_degraded",
                r8["ssim_delta"] is not None and r8["ssim_delta"] >= float(args.min_ssim_delta),
                {"round8_ssim_delta": r8["ssim_delta"], "threshold": args.min_ssim_delta},
            ),
            pass_condition(
                "lpips_not_degraded",
                r8["lpips_delta"] is not None and r8["lpips_delta"] >= float(args.min_lpips_delta),
                {"round8_lpips_delta": r8["lpips_delta"], "threshold": args.min_lpips_delta},
            ),
            pass_condition(
                "safety_not_below_round6",
                r8["safety_score"] is not None
                and r6["safety_score"] is not None
                and r8["safety_score"] >= r6["safety_score"] - float(args.safety_tolerance),
                {"round8_safety": r8["safety_score"], "round6_safety": r6["safety_score"]},
            ),
        ]
        promote = all(item["passed"] for item in conditions)
        payload = {
            "status": "complete",
            "promote_to_round9_init": promote,
            "recommended_checkpoint": str(args.round8_checkpoint if promote else args.fallback_checkpoint),
            "reason": "round8_passed_all_gates" if promote else "round8_failed_one_or_more_gates",
            "round8_summary": str(args.round8_summary),
            "metrics": metrics,
            "conditions": conditions,
        }
    json_path = args.output_dir / "round8_promotion_decision.json"
    md_path = args.output_dir / "round8_promotion_decision.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(md_path, payload)
    print(json.dumps({"decision": str(json_path), **payload}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
