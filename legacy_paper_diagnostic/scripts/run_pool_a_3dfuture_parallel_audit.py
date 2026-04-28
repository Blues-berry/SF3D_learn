#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DOCS_ROOT = PROJECT_ROOT / "docs" / "neural_gaffer_dataset_audit"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "neural_gaffer_dataset_audit"

DEFAULT_INPUT_CSV = DOCS_ROOT / "pool_A_pilot_3dfuture_30.csv"
DEFAULT_OUTPUT_CSV = DOCS_ROOT / "pool_A_pilot_3dfuture_30_gpu1_audit.csv"
DEFAULT_SUMMARY_MD = DOCS_ROOT / "pool_A_pilot_3dfuture_gpu1_audit_summary.md"
DEFAULT_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "audit_pool_a_3dfuture_pilot.py"
DEFAULT_BLENDER_BIN = Path(
    "/4T/CXY/Neural_Gaffer_original/scripts/Objavarse_rendering/blender-3.2.2-linux-x64/blender"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Pool-A 3D-FUTURE Blender signal audit in disjoint parallel shards."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--audit-script", type=Path, default=DEFAULT_AUDIT_SCRIPT)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--cuda-device-index", type=str, default="1")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shard-root", type=Path, default=None)
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames is None:
        raise RuntimeError(f"missing_csv_header:{path}")
    return list(reader.fieldnames), rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shard_rows(rows: list[dict[str, str]], workers: int) -> list[list[tuple[int, dict[str, str]]]]:
    active_workers = max(1, min(workers, len(rows)))
    shards: list[list[tuple[int, dict[str, str]]]] = [[] for _ in range(active_workers)]
    for index, row in enumerate(rows):
        shards[index % active_workers].append((index, row))
    return shards


def run_shard(
    *,
    shard_id: int,
    input_csv: Path,
    output_csv: Path,
    summary_md: Path,
    log_path: Path,
    args: argparse.Namespace,
) -> int:
    cmd = [
        sys.executable,
        str(args.audit_script),
        "--input-csv",
        str(input_csv),
        "--output-csv",
        str(output_csv),
        "--summary-md",
        str(summary_md),
        "--blender-bin",
        str(args.blender_bin),
        "--cuda-device-index",
        str(args.cuda_device_index),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device_index)
    env["BLENDER_CUDA_DEVICE_INDEX"] = "0"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"shard_id={shard_id}\n")
        log.write("cmd=" + " ".join(cmd) + "\n")
        log.flush()
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        return process.wait()


def build_summary(
    *,
    summary_md: Path,
    input_csv: Path,
    output_csv: Path,
    shard_root: Path,
    worker_count: int,
    cuda_device_index: str,
    rows: list[dict[str, str]],
    return_codes: dict[int, int],
) -> None:
    status_counts = Counter(row.get("audit_status", "") or "missing_status" for row in rows)
    reason_counts = Counter(row.get("reject_reason", "") or "none" for row in rows)
    source_counts = Counter(row.get("source_name", "") or "unknown" for row in rows)
    material_counts = Counter(row.get("highlight_priority_class", "") or "unknown" for row in rows)

    lines = [
        "# Pool A 3D-FUTURE GPU1 Parallel Audit",
        "",
        f"- input_csv: `{input_csv}`",
        f"- output_csv: `{output_csv}`",
        f"- shard_root: `{shard_root}`",
        f"- cuda_device_index: `{cuda_device_index}`",
        f"- worker_count: {worker_count}",
        f"- audited_objects: {len(rows)}",
        "",
        "## Shard Return Codes",
        "",
    ]
    for shard_id, code in sorted(return_codes.items()):
        lines.append(f"- shard_{shard_id:02d}: {code}")

    lines.extend(["", "## Audit Status", ""])
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")

    lines.extend(["", "## Top Reasons", ""])
    for reason, count in reason_counts.most_common(10):
        lines.append(f"- {reason}: {count}")

    lines.extend(["", "## Source Mix", ""])
    for source, count in sorted(source_counts.items()):
        lines.append(f"- {source}: {count}")

    lines.extend(["", "## Highlight Class Mix", ""])
    for material_class, count in sorted(material_counts.items()):
        lines.append(f"- {material_class}: {count}")

    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise RuntimeError("workers_must_be_positive")
    fieldnames, rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError("empty_input_csv")
    if not args.blender_bin.exists():
        raise RuntimeError(f"missing_blender_bin:{args.blender_bin}")
    if not args.audit_script.exists():
        raise RuntimeError(f"missing_audit_script:{args.audit_script}")

    shard_root = args.shard_root or OUTPUT_ROOT / f"pool_a_3dfuture_parallel_audit_{utc_stamp()}"
    shard_root.mkdir(parents=True, exist_ok=True)
    shards = shard_rows(rows, args.workers)
    worker_count = len(shards)

    manifest = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "summary_md": str(args.summary_md),
        "cuda_device_index": str(args.cuda_device_index),
        "worker_count": worker_count,
        "shards": [],
    }
    shard_processes: list[tuple[int, subprocess.Popen, Path, Path, list[int]]] = []

    for shard_id, shard in enumerate(shards):
        shard_fieldnames = ["_input_order", *fieldnames]
        shard_input = shard_root / f"shard_{shard_id:02d}_input.csv"
        shard_output = shard_root / f"shard_{shard_id:02d}_output.csv"
        shard_summary = shard_root / f"shard_{shard_id:02d}_summary.md"
        shard_log = shard_root / f"shard_{shard_id:02d}.log"
        shard_rows_with_order = [
            {"_input_order": str(index), **row}
            for index, row in shard
        ]
        write_csv(shard_input, shard_fieldnames, shard_rows_with_order)

        cmd = [
            sys.executable,
            str(args.audit_script),
            "--input-csv",
            str(shard_input),
            "--output-csv",
            str(shard_output),
            "--summary-md",
            str(shard_summary),
            "--blender-bin",
            str(args.blender_bin),
            "--cuda-device-index",
            str(args.cuda_device_index),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device_index)
        env["BLENDER_CUDA_DEVICE_INDEX"] = "0"
        log_handle = shard_log.open("w", encoding="utf-8")
        log_handle.write(f"shard_id={shard_id}\n")
        log_handle.write("cmd=" + " ".join(cmd) + "\n")
        log_handle.flush()
        process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        shard_processes.append((shard_id, process, shard_output, shard_log, [index for index, _row in shard]))
        manifest["shards"].append(
            {
                "shard_id": shard_id,
                "input_csv": str(shard_input),
                "output_csv": str(shard_output),
                "summary_md": str(shard_summary),
                "log": str(shard_log),
                "row_count": len(shard),
                "input_orders": [index for index, _row in shard],
                "pid": process.pid,
            }
        )

    (shard_root / "parallel_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"shard_root": str(shard_root), "worker_count": worker_count}, indent=2), flush=True)

    return_codes: dict[int, int] = {}
    output_rows_by_index: dict[int, dict[str, str]] = {}
    for shard_id, process, shard_output, _shard_log, _input_orders in shard_processes:
        return_codes[shard_id] = process.wait()
        if return_codes[shard_id] != 0:
            continue
        shard_fieldnames, shard_output_rows = read_csv(shard_output)
        if "_input_order" not in shard_fieldnames:
            raise RuntimeError(f"missing_input_order_in_shard_output:{shard_output}")
        for row in shard_output_rows:
            order = int(row.pop("_input_order"))
            output_rows_by_index[order] = row

    failed = {shard_id: code for shard_id, code in return_codes.items() if code != 0}
    if failed:
        raise RuntimeError(f"parallel_audit_failed:{failed}")
    if len(output_rows_by_index) != len(rows):
        raise RuntimeError(
            f"parallel_audit_row_mismatch:expected={len(rows)} actual={len(output_rows_by_index)}"
        )

    ordered_rows = [output_rows_by_index[index] for index in range(len(rows))]
    write_csv(args.output_csv, fieldnames, ordered_rows)
    build_summary(
        summary_md=args.summary_md,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        shard_root=shard_root,
        worker_count=worker_count,
        cuda_device_index=str(args.cuda_device_index),
        rows=ordered_rows,
        return_codes=return_codes,
    )
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.summary_md}")


if __name__ == "__main__":
    main()
