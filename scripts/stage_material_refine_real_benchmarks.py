#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/ubuntu/ssd_work/projects/stable-fast-3d")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "material_refine_real_benchmarks"
OPENILLUMINATION_API = "https://huggingface.co/api/datasets/OpenIllumination/OpenIllumination"
OPENILLUMINATION_RESOLVE = "https://huggingface.co/datasets/OpenIllumination/OpenIllumination/resolve/main/{path}?download=true"
STANFORD_ORB_FILES = {
    "ground_truth": {
        "url": "https://downloads.cs.stanford.edu/viscam/StanfordORB/ground_truth.tar.gz",
        "expected_bytes": 5057562274,
        "pool_name": "pool_F_real_world_eval_holdout",
        "license_bucket": "benchmark_project_terms_unstated",
        "planned_use": "real_world_eval_holdout",
    },
    "blender_LDR": {
        "url": "https://downloads.cs.stanford.edu/viscam/StanfordORB/blender_LDR.tar.gz",
        "expected_bytes": 11611255093,
        "pool_name": "pool_F_real_world_eval_holdout",
        "license_bucket": "benchmark_project_terms_unstated",
        "planned_use": "real_world_eval_holdout",
    },
    "blender_HDR": {
        "url": "https://downloads.cs.stanford.edu/viscam/StanfordORB/blender_HDR.tar.gz",
        "expected_bytes": 0,
        "pool_name": "pool_F_real_world_eval_holdout",
        "license_bucket": "benchmark_project_terms_unstated",
        "planned_use": "real_world_hdr_eval_holdout",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Stage real-lighting Pool-B/Pool-F benchmark datasets without mixing them into the paper training pool.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--download-stanford-orb", action="store_true")
    parser.add_argument("--download-openillumination", action="store_true")
    parser.add_argument("--include-orb-hdr", action="store_true")
    parser.add_argument("--openillumination-objects", type=str, default="obj_01_egg,obj_02_cup")
    parser.add_argument("--openillumination-max-files-per-object", type=int, default=300)
    parser.add_argument("--wget-tries", type=int, default=0, help="0 means wget retry forever.")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_json(url: str, timeout: int = 90) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "stable-fast-3d-real-benchmark-stager/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_wget(url: str, output_path: Path, *, tries: int) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "huggingface.co" in url:
        retry_count = 1000000 if int(tries) == 0 else int(tries)
        cmd = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            str(retry_count),
            "--retry-delay",
            "2",
            "-C",
            "-",
            "-o",
            str(output_path),
            url,
        ]
        started = time.time()
        process = subprocess.run(cmd, check=False)
        return {
            "command": cmd,
            "exit_code": int(process.returncode),
            "elapsed_seconds": round(time.time() - started, 3),
            "local_path": str(output_path),
            "local_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        }
    cmd = [
        "wget",
        "-4",
        "-c",
        "--timeout=30",
        "--read-timeout=30",
        f"--tries={int(tries)}",
        "-O",
        str(output_path),
        url,
    ]
    started = time.time()
    process = subprocess.run(cmd, check=False)
    return {
        "command": cmd,
        "exit_code": int(process.returncode),
        "elapsed_seconds": round(time.time() - started, 3),
        "local_path": str(output_path),
        "local_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
    }


def stage_stanford_orb(output_root: Path, *, download: bool, include_hdr: bool, wget_tries: int) -> list[dict[str, Any]]:
    orb_root = output_root / "stanford_orb"
    selected = ["ground_truth", "blender_LDR"]
    if include_hdr:
        selected.append("blender_HDR")
    rows = []
    for key in selected:
        spec = STANFORD_ORB_FILES[key]
        local_path = orb_root / f"{key}.tar.gz"
        row = {
            "pool_name": spec["pool_name"],
            "source_name": "Stanford-ORB",
            "benchmark_role": spec["planned_use"],
            "asset_id": key,
            "license_bucket": spec["license_bucket"],
            "download_url": spec["url"],
            "expected_size_bytes": spec["expected_bytes"],
            "local_path": str(local_path),
            "download_status": "not_requested",
            "notes": "benchmark_only_do_not_train",
        }
        if local_path.exists() and local_path.stat().st_size > 0:
            row["download_status"] = "present"
            row["local_size_bytes"] = local_path.stat().st_size
        if download and row["download_status"] != "present":
            result = run_wget(spec["url"], local_path, tries=wget_tries)
            row.update(result)
            row["download_status"] = "downloaded" if result["exit_code"] == 0 else "partial_or_failed"
        rows.append(row)
    return rows


def select_openillumination_files(api_payload: dict[str, Any], object_ids: list[str], max_files_per_object: int) -> list[str]:
    siblings = [item.get("rfilename", "") for item in api_payload.get("siblings", []) if isinstance(item, dict)]
    selected = [
        name
        for name in siblings
        if name in {"README.md", "open_illumination.py", "open_illumination_v2.py", "light_pos.npy", ".gitattributes"}
    ]
    wanted_suffixes = (".json", ".txt", ".npy", ".png")
    wanted_tokens = ("/com_masked_thumbnail/", "/obj_mask/", "/com_mask/", "/transforms")
    for object_id in object_ids:
        object_matches = []
        for name in siblings:
            if f"/{object_id}/" not in name:
                continue
            if not name.endswith(wanted_suffixes):
                continue
            if name.endswith(".png") and not any(token in name for token in wanted_tokens):
                continue
            object_matches.append(name)
        selected.extend(sorted(object_matches)[: max(int(max_files_per_object), 0)])
    deduped = []
    seen = set()
    for item in selected:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def stage_openillumination(
    output_root: Path,
    *,
    download: bool,
    object_ids: list[str],
    max_files_per_object: int,
    wget_tries: int,
) -> list[dict[str, Any]]:
    oi_root = output_root / "openillumination"
    api_payload = fetch_json(OPENILLUMINATION_API)
    files = select_openillumination_files(api_payload, object_ids, max_files_per_object)
    rows = []
    for relpath in files:
        local_path = oi_root / relpath
        url = OPENILLUMINATION_RESOLVE.format(path=relpath)
        row = {
            "pool_name": "pool_B_controlled_highlight_supervision",
            "source_name": "OpenIllumination",
            "benchmark_role": "controlled_real_lighting_aux_eval",
            "asset_id": relpath,
            "license_bucket": "cc_by_4_0",
            "download_url": url,
            "local_path": str(local_path),
            "download_status": "not_requested",
            "notes": "auxiliary_or_eval_only_do_not_mix_into_pool_a",
        }
        if local_path.exists() and local_path.stat().st_size > 0:
            row["download_status"] = "present"
            row["local_size_bytes"] = local_path.stat().st_size
        if download and row["download_status"] != "present":
            result = run_wget(url, local_path, tries=wget_tries)
            row.update(result)
            row["download_status"] = "downloaded" if result["exit_code"] == 0 else "partial_or_failed"
        rows.append(row)
    return rows


def access_probe_rows() -> list[dict[str, Any]]:
    return [
        {
            "pool_name": "pool_B_controlled_highlight_supervision",
            "source_name": "OLATverse",
            "benchmark_role": "controlled_real_lighting_aux_eval",
            "license_bucket": "license_not_posted_project_page",
            "download_status": "access_probe_only",
            "notes": "paper reports 765 objects and about 9M images; do not auto-mix until download/license endpoint is confirmed",
        },
        {
            "pool_name": "pool_B_controlled_highlight_supervision",
            "source_name": "ICTPolarReal",
            "benchmark_role": "polarized_specular_aux_eval",
            "license_bucket": "license_not_posted_preprint_project",
            "download_status": "access_probe_only",
            "notes": "paper reports polarized HDR material signals; do not auto-mix until project access and license are confirmed",
        },
    ]


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    object_ids = [item.strip() for item in args.openillumination_objects.split(",") if item.strip()]
    rows = []
    rows.extend(
        stage_stanford_orb(
            args.output_root,
            download=args.download_stanford_orb,
            include_hdr=args.include_orb_hdr,
            wget_tries=args.wget_tries,
        )
    )
    rows.extend(
        stage_openillumination(
            args.output_root,
            download=args.download_openillumination,
            object_ids=object_ids,
            max_files_per_object=args.openillumination_max_files_per_object,
            wget_tries=args.wget_tries,
        )
    )
    rows.extend(access_probe_rows())
    summary = {
        "manifest_version": "sf3d_real_benchmark_stage_v1",
        "created_unix": time.time(),
        "records": len(rows),
        "downloaded_or_present": sum(row.get("download_status") in {"downloaded", "present"} for row in rows),
        "partial_or_failed": sum(row.get("download_status") == "partial_or_failed" for row in rows),
        "records_by_source": {
            source: sum(row.get("source_name") == source for row in rows)
            for source in sorted({str(row.get("source_name")) for row in rows})
        },
    }
    write_json(args.output_root / "real_benchmark_stage_manifest.json", {"summary": summary, "records": rows})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
