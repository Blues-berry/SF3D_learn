#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from refresh_material_refine_partial_manifest import find_missing_paths  # noqa: E402
from sf3d.material_refine.trainv5_target_gate import TARGET_GATE_VERSION, trainv5_target_truth_gate  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def records(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def skipped(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [row for row in payload.get("skipped_records", []) if isinstance(row, dict)]
    return []


def parse_workers(text: str) -> list[int]:
    out = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value > 0:
            out.append(value)
    return out or [1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=REPO_ROOT / "output/material_refine_trainV5_abc/B_track/full_1155_rebake/full_1155_rebake_input_manifest.json",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=REPO_ROOT / "output/material_refine_trainV5_benchmarks/full_1155_parallel",
    )
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--parallel-workers-list", type=str, default="1,2")
    parser.add_argument("--atlas-resolution", type=int, default=1024)
    parser.add_argument("--render-resolution", type=int, default=320)
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument("--view-light-protocol", type=str, default="production_32")
    parser.add_argument(
        "--hdri-bank-json",
        type=Path,
        default=REPO_ROOT / "output/highlight_pool_a_8k/aux_sources/polyhaven_hdri_bank.json",
    )
    parser.add_argument("--min-hdri-count", type=int, default=900)
    parser.add_argument("--max-hdri-lights", type=int, default=4)
    parser.add_argument("--cuda-device-index", type=str, default="0")
    parser.add_argument("--refresh-partial-every", type=int, default=1)
    parser.add_argument("--alignment-mean-threshold", type=float, default=0.08)
    parser.add_argument("--alignment-p95-threshold", type=float, default=0.20)
    parser.add_argument("--speedup-threshold", type=float, default=1.6)
    parser.add_argument("--max-pass-rate-drop", type=float, default=0.005)
    parser.add_argument("--gpu-sample-interval", type=float, default=2.0)
    return parser.parse_args()


def build_subset_manifest(input_manifest: Path, output_manifest: Path, sample_count: int) -> dict[str, Any]:
    payload = read_json(input_manifest, {})
    rows = records(payload)
    selected = rows[:sample_count]
    subset = dict(payload) if isinstance(payload, dict) else {}
    subset["generated_at_utc"] = utc_now()
    subset["source_manifest"] = str(input_manifest.resolve())
    subset["records"] = selected
    subset["sample_count"] = len(selected)
    write_json(output_manifest, subset)
    return subset


def gpu_sampler(stop_event: threading.Event, gpu_index: int, interval: float, sink: list[dict[str, float]]) -> None:
    while not stop_event.is_set():
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if gpu_index < len(lines):
                parts = [part.strip() for part in lines[gpu_index].split(",")]
                if len(parts) >= 3:
                    try:
                        sink.append(
                            {
                                "gpu_util": float(parts[0]),
                                "mem_util": float(parts[1]),
                                "mem_used_mib": float(parts[2]),
                            }
                        )
                    except ValueError:
                        pass
        stop_event.wait(interval)


def numeric_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def evaluate_run(manifest_path: Path) -> dict[str, Any]:
    payload = read_json(manifest_path, {})
    prepared = records(payload)
    skipped_rows = skipped(payload)
    gate_results = [trainv5_target_truth_gate(row) for row in prepared]
    pass_count = sum(1 for ok, _ in gate_results if ok)
    blockers = Counter(reason for ok, blocker_list in gate_results if not ok for reason in blocker_list)
    missing_bundle_paths = 0
    corrupted_records = 0
    for row in prepared:
        bundle_root = Path(str(row.get("canonical_bundle_root") or ""))
        if bundle_root.exists():
            missing = find_missing_paths(bundle_root)
            missing_bundle_paths += len(missing)
            if missing:
                corrupted_records += 1
    return {
        "prepared_records": len(prepared),
        "skipped_records": len(skipped_rows),
        "target_gate_version": TARGET_GATE_VERSION,
        "target_truth_gate_pass": pass_count,
        "target_truth_gate_fail": max(len(prepared) - pass_count, 0),
        "pass_rate": (pass_count / len(prepared)) if prepared else 0.0,
        "gate_blockers": dict(blockers),
        "corrupted_records": corrupted_records,
        "missing_bundle_paths": missing_bundle_paths,
    }


def run_benchmark_case(args: argparse.Namespace, subset_manifest: Path, worker_count: int, case_root: Path) -> dict[str, Any]:
    if case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True, exist_ok=True)
    prepared_root = case_root / "prepared"
    output_manifest = case_root / "benchmark_manifest.json"
    partial_manifest = case_root / "benchmark_partial_manifest.json"
    summary_json = case_root / "benchmark_prepare_summary.json"
    summary_md = case_root / "benchmark_prepare_summary.md"
    stdout_log = case_root / "benchmark_prepare_stdout.log"
    command = [
        sys.executable,
        "scripts/prepare_material_refine_dataset.py",
        "--input-manifest",
        str(subset_manifest),
        "--output-root",
        str(prepared_root),
        "--output-manifest",
        str(output_manifest),
        "--split",
        "full",
        "--atlas-resolution",
        str(args.atlas_resolution),
        "--render-resolution",
        str(args.render_resolution),
        "--cycles-samples",
        str(args.cycles_samples),
        "--view-light-protocol",
        str(args.view_light_protocol),
        "--hdri-bank-json",
        str(args.hdri_bank_json),
        "--min-hdri-count",
        str(args.min_hdri_count),
        "--max-hdri-lights",
        str(args.max_hdri_lights),
        "--cuda-device-index",
        str(args.cuda_device_index),
        "--parallel-workers",
        str(worker_count),
        "--rebake-version",
        "rebake_v2",
        "--disable-render-cache",
        "--disallow-prior-copy-fallback",
        "--target-view-alignment-mean-threshold",
        str(args.alignment_mean_threshold),
        "--target-view-alignment-p95-threshold",
        str(args.alignment_p95_threshold),
        "--partial-manifest",
        str(partial_manifest),
        "--refresh-partial-every",
        str(args.refresh_partial_every),
        "--summary-json",
        str(summary_json),
        "--summary-md",
        str(summary_md),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device_index)
    pythonpath_parts = [str(REPO_ROOT), str(REPO_ROOT / "scripts")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    stop_event = threading.Event()
    gpu_samples: list[dict[str, float]] = []
    sampler = threading.Thread(
        target=gpu_sampler,
        args=(stop_event, int(args.cuda_device_index), float(args.gpu_sample_interval), gpu_samples),
        daemon=True,
    )
    start_ts = time.time()
    sampler.start()
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False)
    stop_event.set()
    sampler.join(timeout=5)
    end_ts = time.time()
    write_text(stdout_log, "\n".join(result.stdout.splitlines()))
    duration_seconds = max(end_ts - start_ts, 1e-6)
    evaluation = evaluate_run(output_manifest)
    processed = max(evaluation["prepared_records"] + evaluation["skipped_records"], 0)
    partial_payload = read_json(partial_manifest, {})
    partial_ok = partial_manifest.exists() and len(records(partial_payload)) > 0
    gpu_util_values = [sample["gpu_util"] for sample in gpu_samples if "gpu_util" in sample]
    mem_used_values = [sample["mem_used_mib"] for sample in gpu_samples if "mem_used_mib" in sample]
    return {
        "worker_count": worker_count,
        "command": " ".join(shlex.quote(part) for part in command),
        "returncode": result.returncode,
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_seconds / 60.0,
        "records_per_hour": (processed / duration_seconds) * 3600.0 if duration_seconds > 0 else None,
        "mean_wall_time_per_object_seconds": duration_seconds / processed if processed > 0 else None,
        "gpu_util_mean": numeric_mean(gpu_util_values),
        "gpu_memory_peak_mib": max(mem_used_values) if mem_used_values else None,
        "gpu_samples": gpu_samples,
        "partial_manifest_refresh_correct": partial_ok,
        "partial_manifest_records": len(records(partial_payload)),
        "stdout_log": str(stdout_log.resolve()),
        "stdout_tail": result.stdout.splitlines()[-120:],
        **evaluation,
    }


def recommend_case(results: list[dict[str, Any]], speedup_threshold: float, max_pass_rate_drop: float) -> dict[str, Any]:
    by_worker = {int(item["worker_count"]): item for item in results}
    baseline = by_worker.get(1)
    if baseline is None or not baseline.get("records_per_hour"):
        return {"recommended_worker_count": None, "reason": "missing_baseline_worker_1"}
    baseline_throughput = float(baseline["records_per_hour"])
    baseline_pass_rate = float(baseline.get("pass_rate") or 0.0)
    candidates: list[dict[str, Any]] = []
    for worker, item in sorted(by_worker.items()):
        if worker == 1:
            continue
        throughput = float(item.get("records_per_hour") or 0.0)
        pass_rate = float(item.get("pass_rate") or 0.0)
        speedup = throughput / baseline_throughput if baseline_throughput > 0 else 0.0
        pass_rate_drop = baseline_pass_rate - pass_rate
        qualifies = (
            item.get("returncode") == 0
            and bool(item.get("partial_manifest_refresh_correct"))
            and int(item.get("corrupted_records") or 0) == 0
            and speedup >= speedup_threshold
            and pass_rate_drop <= max_pass_rate_drop
        )
        entry = dict(item)
        entry["speedup_vs_baseline"] = speedup
        entry["pass_rate_drop_vs_baseline"] = pass_rate_drop
        entry["qualifies"] = qualifies
        candidates.append(entry)
    winners = [item for item in candidates if item["qualifies"]]
    if not winners:
        return {
            "recommended_worker_count": None,
            "reason": "no_candidate_met_thresholds",
            "baseline_worker_count": 1,
            "baseline_records_per_hour": baseline_throughput,
            "candidates": candidates,
        }
    winners.sort(key=lambda item: (float(item.get("records_per_hour") or 0.0), -int(item["worker_count"])), reverse=True)
    best = winners[0]
    return {
        "recommended_worker_count": int(best["worker_count"]),
        "reason": "best_qualifying_speedup",
        "baseline_worker_count": 1,
        "baseline_records_per_hour": baseline_throughput,
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    args.benchmark_root.mkdir(parents=True, exist_ok=True)
    subset_manifest = args.benchmark_root / "benchmark_input_manifest.json"
    subset_payload = build_subset_manifest(args.input_manifest, subset_manifest, args.sample_count)
    worker_list = parse_workers(args.parallel_workers_list)
    results: list[dict[str, Any]] = []
    for worker in worker_list:
        case_root = args.benchmark_root / f"parallel_workers_{worker}"
        results.append(run_benchmark_case(args, subset_manifest, worker, case_root))
        write_json(case_root / "benchmark_result.json", results[-1])
    recommendation = recommend_case(results, float(args.speedup_threshold), float(args.max_pass_rate_drop))
    summary = {
        "generated_at_utc": utc_now(),
        "input_manifest": str(args.input_manifest.resolve()),
        "benchmark_root": str(args.benchmark_root.resolve()),
        "sample_count": len(records(subset_payload)),
        "worker_list": worker_list,
        "render_resolution": args.render_resolution,
        "cycles_samples": args.cycles_samples,
        "view_light_protocol": args.view_light_protocol,
        "results": results,
        "recommendation": recommendation,
    }
    write_json(args.benchmark_root / "benchmark_summary.json", summary)
    lines = [
        "# TrainV5 B1 Parallel Benchmark",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- sample_count: `{summary['sample_count']}`",
        f"- worker_list: `{json.dumps(worker_list)}`",
        f"- render_resolution: `{args.render_resolution}`",
        f"- cycles_samples: `{args.cycles_samples}`",
        f"- view_light_protocol: `{args.view_light_protocol}`",
        f"- recommendation: `{json.dumps(recommendation, ensure_ascii=False)}`",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## parallel_workers={result['worker_count']}",
                "",
                f"- returncode: `{result['returncode']}`",
                f"- records_per_hour: `{result['records_per_hour']}`",
                f"- mean_wall_time_per_object_seconds: `{result['mean_wall_time_per_object_seconds']}`",
                f"- pass_rate: `{result['pass_rate']}`",
                f"- target_truth_gate_fail: `{result['target_truth_gate_fail']}`",
                f"- corrupted_records: `{result['corrupted_records']}`",
                f"- partial_manifest_refresh_correct: `{str(result['partial_manifest_refresh_correct']).lower()}`",
                f"- gpu_util_mean: `{result['gpu_util_mean']}`",
                f"- gpu_memory_peak_mib: `{result['gpu_memory_peak_mib']}`",
                "",
            ]
        )
    write_text(args.benchmark_root / "benchmark_summary.md", "\n".join(lines))
    print(json.dumps(summary["recommendation"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
