#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPANSION_ROOT = REPO_ROOT / "output/material_refine_expansion_candidates"
DEFAULT_MERGED_MANIFEST = EXPANSION_ROOT / "merged_expansion_candidate_manifest.json"
DEFAULT_OBJAVERSE_ROOT = REPO_ROOT / "output/highlight_pool_a_8k/aux_sources/objaverse_xl"
DEFAULT_OBJAVERSE_STAGE_ROOT = REPO_ROOT / "output/highlight_pool_a_8k/objaverse_cached_increment_material_priority"
DEFAULT_OBJAVERSE_SOURCE_DIR = EXPANSION_ROOT / "material_priority_objaverse_increment"
DEFAULT_STAGE_ROOT = EXPANSION_ROOT / "material_priority_stage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Stage material-priority source downloads and connect them to the active TrainV5 queue flow.",
    )
    parser.add_argument("--expansion-root", type=Path, default=EXPANSION_ROOT)
    parser.add_argument("--merged-manifest", type=Path, default=DEFAULT_MERGED_MANIFEST)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--objaverse-root", type=Path, default=DEFAULT_OBJAVERSE_ROOT)
    parser.add_argument("--objaverse-stage-root", type=Path, default=DEFAULT_OBJAVERSE_STAGE_ROOT)
    parser.add_argument("--objaverse-source-dir", type=Path, default=DEFAULT_OBJAVERSE_SOURCE_DIR)
    parser.add_argument("--objaverse-target", type=int, default=1200)
    parser.add_argument("--objaverse-sources", type=str, default="smithsonian,thingiverse,sketchfab")
    parser.add_argument(
        "--priority-material-families",
        type=str,
        default="mixed_thin_boundary,glass_metal,ceramic_glazed_lacquer,metal_dominant,glossy_non_metal",
    )
    parser.add_argument(
        "--target-material-family-ratios",
        type=str,
        default="mixed_thin_boundary=0.28,glass_metal=0.24,ceramic_glazed_lacquer=0.22,metal_dominant=0.18,glossy_non_metal=0.08",
    )
    parser.add_argument("--source-priority", type=str, default="smithsonian,thingiverse,sketchfab")
    parser.add_argument("--download-objaverse", action="store_true")
    parser.add_argument("--download-mode", choices=("direct", "proxy-probe", "mirror-probe", "off"), default="direct")
    parser.add_argument("--download-probe-size", type=int, default=100)
    parser.add_argument("--min-download-success-rate", type=float, default=0.20)
    parser.add_argument("--max-retry-rounds", type=int, default=2)
    parser.add_argument("--max-actions-per-run", type=int, default=4)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-second-pass", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def run_cmd(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def file_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def manifest_rows(path: Path | None) -> list[dict[str, Any]]:
    payload = read_json(path, {})
    rows = payload.get("records", []) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def summarize_increment_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "source": dict(Counter(str(row.get("source") or "unknown") for row in rows)),
        "material": dict(Counter(str(row.get("highlight_material_class") or "unknown") for row in rows)),
        "download_status": dict(Counter(str(row.get("download_status") or "unknown") for row in rows)),
    }


def pool_state(pool: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(pool["manifest_path"]))
    payload = read_json(manifest_path, {})
    rows = manifest_rows(manifest_path)
    downloaded = sum(bool(row.get("local_path")) and Path(str(row.get("local_path"))).exists() for row in rows)
    selected = len(rows)
    retry_round = int(payload.get("retry_round") or pool.get("retry_round") or 0) if isinstance(payload, dict) else 0
    state = dict(pool)
    state.update(
        {
            "selected_count": selected,
            "downloaded_count": downloaded,
            "missing_count": max(selected - downloaded, 0),
            "retry_round": retry_round,
            "selected_parquet": str(payload.get("selected_parquet") or pool.get("selected_parquet") or ""),
            "manifest_path": str(manifest_path),
        }
    )
    return state


def aggregate_rows(pools: list[dict[str, Any]], target_total: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_by_uid: dict[str, dict[str, Any]] = {}
    downloaded_rows: list[dict[str, Any]] = []
    for pool in pools:
        for row in manifest_rows(Path(str(pool["manifest_path"]))):
            source_uid = str(row.get("source_uid") or "")
            if not source_uid:
                continue
            local_path = str(row.get("local_path") or "")
            current_is_downloaded = bool(local_path) and Path(local_path).exists()
            existing = selected_by_uid.get(source_uid)
            if existing is None:
                selected_by_uid[source_uid] = dict(row)
            else:
                existing_path = str(existing.get("local_path") or "")
                existing_is_downloaded = bool(existing_path) and Path(existing_path).exists()
                if current_is_downloaded and not existing_is_downloaded:
                    selected_by_uid[source_uid] = dict(row)
            selected_by_uid[source_uid]["selection_pool_id"] = pool.get("pool_id")
            if current_is_downloaded:
                item = dict(row)
                item["selection_pool_id"] = pool.get("pool_id")
                downloaded_rows.append(item)
    selected_rows = list(selected_by_uid.values())
    selected_rows.sort(key=lambda row: (str(row.get("selection_pool_id") or ""), str(row.get("source_uid") or "")))
    downloaded_rows.sort(key=lambda row: (str(row.get("selection_pool_id") or ""), str(row.get("source_uid") or "")))
    return selected_rows, downloaded_rows[:target_total]


def stage_objaverse_pool(
    *,
    pool_dir: Path,
    pool_id: str,
    target: int,
    args: argparse.Namespace,
    selection_mode: str,
    selected_parquet: Path | None = None,
    existing_manifest: Path | None = None,
    exclude_selection_manifest: Path | None = None,
    should_download: bool = False,
) -> tuple[Path, dict[str, Any]]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/stage_objaverse_cached_increment.py",
        "--objaverse-root",
        str(args.objaverse_root),
        "--output-root",
        str(pool_dir),
        "--sources",
        args.objaverse_sources,
        "--target",
        str(target),
        "--priority-material-families",
        args.priority_material_families,
        "--target-material-family-ratios",
        args.target_material_family_ratios,
        "--source-priority",
        args.source_priority,
        "--exclude-manifest",
        str(args.merged_manifest),
        "--selection-mode",
        selection_mode,
        "--selection-pool-id",
        pool_id,
    ]
    if selected_parquet is not None:
        cmd.extend(["--selected-parquet", str(selected_parquet)])
    if existing_manifest is not None:
        cmd.extend(["--existing-manifest", str(existing_manifest)])
    if exclude_selection_manifest is not None:
        cmd.extend(["--exclude-selection-manifest", str(exclude_selection_manifest)])
    if should_download:
        cmd.append("--download")
    result = run_cmd(cmd, cwd=REPO_ROOT)
    manifest_path = pool_dir / "objaverse_cached_increment_manifest.json"
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"] or f"stage_objaverse_pool_failed:{pool_id}")
    return manifest_path, result


def build_aggregate_source_manifest(source_dir: Path, aggregate_downloaded_manifest: Path) -> dict[str, str]:
    output_json = source_dir / "objaverse_1200_source_candidate_manifest.json"
    output_csv = source_dir / "objaverse_1200_source_candidate_manifest.csv"
    output_md = source_dir / "objaverse_1200_source_candidate_manifest_summary.md"
    compat_json = source_dir / "source_candidate_manifest.json"
    compat_csv = source_dir / "source_candidate_manifest.csv"
    compat_md = source_dir / "source_candidate_manifest_summary.md"
    result = run_cmd(
        [
            sys.executable,
            "scripts/build_objaverse_increment_manifest.py",
            "--input-json",
            str(aggregate_downloaded_manifest),
            "--output-root",
            str(source_dir),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ],
        cwd=REPO_ROOT,
    )
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"] or "build_aggregate_source_manifest_failed")
    shutil.copy2(output_json, compat_json)
    shutil.copy2(output_csv, compat_csv)
    shutil.copy2(output_md, compat_md)
    return {
        "json": str(output_json),
        "csv": str(output_csv),
        "md": str(output_md),
        "compat_json": str(compat_json),
        "compat_csv": str(compat_csv),
        "compat_md": str(compat_md),
    }


def write_objaverse_status(
    *,
    args: argparse.Namespace,
    objaverse_plan_root: Path,
    pools: list[dict[str, Any]],
    aggregate_selected: list[dict[str, Any]],
    aggregate_downloaded: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    source_manifest_files: dict[str, str],
    effective_download_mode: str,
) -> dict[str, Any]:
    downloaded_total = len(aggregate_downloaded)
    selected_total = len(aggregate_selected)
    missing_total = max(int(args.objaverse_target) - downloaded_total, 0)
    last_action = actions[-1] if actions else {}
    last_manifest_path = Path(str(last_action.get("manifest_path") or "")) if last_action else None
    last_payload = read_json(last_manifest_path, {}) if last_manifest_path and last_manifest_path.exists() else {}
    latest_attempted = int(last_payload.get("download_attempted_count") or 0) if isinstance(last_payload, dict) else 0
    latest_downloaded = int(last_payload.get("downloaded_count") or 0) if isinstance(last_payload, dict) else 0
    latest_error = str(last_payload.get("download_error") or "") if isinstance(last_payload, dict) else ""
    latest_success_rate = float(latest_downloaded) / float(latest_attempted) if latest_attempted > 0 else 0.0
    retryable_pools = [
        str(pool["pool_id"])
        for pool in pools
        if int(pool.get("missing_count") or 0) > 0 and int(pool.get("retry_round") or 0) < int(args.max_retry_rounds)
    ]
    topup_needed = downloaded_total < int(args.objaverse_target) and not retryable_pools
    payload = {
        "generated_at_utc": utc_now(),
        "target_total": int(args.objaverse_target),
        "selected_total": selected_total,
        "downloaded_total": downloaded_total,
        "missing_total": missing_total,
        "retry_round": max((int(pool.get("retry_round") or 0) for pool in pools), default=0),
        "topup_needed": bool(topup_needed),
        "ready_exact_1200": downloaded_total >= int(args.objaverse_target),
        "download_mode": effective_download_mode,
        "download_attempted_latest": latest_attempted,
        "downloaded_latest": latest_downloaded,
        "download_success_rate_latest": latest_success_rate,
        "download_failure_reason_latest": latest_error,
        "download_success_rate_aggregate": (float(downloaded_total) / float(selected_total)) if selected_total > 0 else 0.0,
        "source_manifest_files": source_manifest_files,
        "pool_count": len(pools),
        "pools": pools,
        "actions": actions,
        "aggregate_selected_summary": summarize_increment_rows(aggregate_selected),
        "aggregate_downloaded_summary": summarize_increment_rows(aggregate_downloaded),
    }
    write_json(objaverse_plan_root / "objaverse_1200_download_status.json", payload)
    lines = [
        "# Objaverse 1200 Download Status",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- target_total: `{payload['target_total']}`",
        f"- selected_total: `{payload['selected_total']}`",
        f"- downloaded_total: `{payload['downloaded_total']}`",
        f"- missing_total: `{payload['missing_total']}`",
        f"- retry_round: `{payload['retry_round']}`",
        f"- topup_needed: `{str(payload['topup_needed']).lower()}`",
        f"- ready_exact_1200: `{str(payload['ready_exact_1200']).lower()}`",
        f"- download_mode: `{payload['download_mode']}`",
        f"- download_attempted_latest: `{payload['download_attempted_latest']}`",
        f"- downloaded_latest: `{payload['downloaded_latest']}`",
        f"- download_success_rate_latest: `{payload['download_success_rate_latest']}`",
        f"- download_failure_reason_latest: `{payload['download_failure_reason_latest']}`",
        f"- source_manifest_json: `{source_manifest_files.get('json', '')}`",
        "",
        "## Pools",
        "",
    ]
    for pool in pools:
        lines.append(
            f"- {pool['pool_id']}: selected=`{pool.get('selected_count')}`, downloaded=`{pool.get('downloaded_count')}`, missing=`{pool.get('missing_count')}`, retry_round=`{pool.get('retry_round')}`, mode=`{pool.get('selection_mode')}`"
        )
    if actions:
        lines.extend(["", "## Actions", ""])
        for action in actions:
            lines.append(f"- `{action.get('action')}` -> rc=`{action.get('returncode')}` pool=`{action.get('pool_id')}`")
    write_text(objaverse_plan_root / "objaverse_1200_download_status.md", "\n".join(lines))
    return payload


def stage_objaverse_increment(args: argparse.Namespace, *, should_download: bool, effective_download_mode: str) -> dict[str, Any]:
    objaverse_plan_root = args.stage_root / "objaverse_1200_serial"
    objaverse_plan_root.mkdir(parents=True, exist_ok=True)
    args.objaverse_source_dir.mkdir(parents=True, exist_ok=True)
    state_path = objaverse_plan_root / "objaverse_1200_plan_state.json"
    aggregate_raw_manifest = objaverse_plan_root / "objaverse_1200_aggregate_increment_manifest.json"
    aggregate_downloaded_manifest = objaverse_plan_root / "objaverse_1200_downloaded_increment_manifest.json"
    state = read_json(state_path, {"pools": [], "history": []})
    if not isinstance(state, dict):
        state = {"pools": [], "history": []}
    pools = [pool_state(pool) for pool in state.get("pools", []) if isinstance(pool, dict)]
    actions: list[dict[str, Any]] = []

    for _step in range(max(int(args.max_actions_per_run), 1)):
        downloaded_total = len(aggregate_rows(pools, int(args.objaverse_target))[1]) if pools else 0
        if downloaded_total >= int(args.objaverse_target):
            break

        retry_pool = next(
            (
                pool
                for pool in pools
                if int(pool.get("missing_count") or 0) > 0 and int(pool.get("retry_round") or 0) < int(args.max_retry_rounds)
            ),
            None,
        )
        if retry_pool is not None:
            manifest_path, result = stage_objaverse_pool(
                pool_dir=Path(str(retry_pool["pool_dir"])),
                pool_id=str(retry_pool["pool_id"]),
                target=int(retry_pool.get("target") or 0),
                args=args,
                selection_mode="retry_missing",
                selected_parquet=Path(str(retry_pool["selected_parquet"])),
                existing_manifest=Path(str(retry_pool["manifest_path"])),
                should_download=should_download,
            )
            actions.append(
                {
                    "timestamp_utc": utc_now(),
                    "action": "retry_missing",
                    "pool_id": retry_pool["pool_id"],
                    "manifest_path": str(manifest_path),
                    **result,
                }
            )
            for index, pool in enumerate(state.get("pools", [])):
                if pool.get("pool_id") == retry_pool["pool_id"]:
                    state["pools"][index]["manifest_path"] = str(manifest_path)
                    state["pools"][index]["selected_parquet"] = str(Path(str(retry_pool["selected_parquet"])))
            pools = [pool_state(pool) for pool in state.get("pools", []) if isinstance(pool, dict)]
            continue

        if not pools:
            pool_dir = objaverse_plan_root / "base_selection"
            manifest_path, result = stage_objaverse_pool(
                pool_dir=pool_dir,
                pool_id="base_selection",
                target=int(args.objaverse_target),
                args=args,
                selection_mode="initial_selection",
                should_download=should_download,
            )
            state.setdefault("pools", []).append(
                {
                    "pool_id": "base_selection",
                    "selection_mode": "initial_selection",
                    "target": int(args.objaverse_target),
                    "pool_dir": str(pool_dir),
                    "manifest_path": str(manifest_path),
                    "selected_parquet": str(pool_dir / "objaverse_cached_selected.parquet"),
                }
            )
            actions.append(
                {
                    "timestamp_utc": utc_now(),
                    "action": "initial_selection",
                    "pool_id": "base_selection",
                    "manifest_path": str(manifest_path),
                    **result,
                }
            )
            pools = [pool_state(pool) for pool in state.get("pools", []) if isinstance(pool, dict)]
            continue

        gap = max(int(args.objaverse_target) - len(aggregate_rows(pools, int(args.objaverse_target))[1]), 0)
        if gap <= 0:
            break
        topup_index = 1 + sum(1 for pool in state.get("pools", []) if str(pool.get("selection_mode")) == "topup_selection")
        pool_id = f"topup_round_{topup_index}"
        pool_dir = objaverse_plan_root / pool_id
        manifest_path, result = stage_objaverse_pool(
            pool_dir=pool_dir,
            pool_id=pool_id,
            target=gap,
            args=args,
            selection_mode="topup_selection",
            exclude_selection_manifest=aggregate_raw_manifest if aggregate_raw_manifest.exists() else None,
            should_download=should_download,
        )
        state.setdefault("pools", []).append(
            {
                "pool_id": pool_id,
                "selection_mode": "topup_selection",
                "target": gap,
                "pool_dir": str(pool_dir),
                "manifest_path": str(manifest_path),
                "selected_parquet": str(pool_dir / "objaverse_cached_selected.parquet"),
            }
        )
        actions.append(
            {
                "timestamp_utc": utc_now(),
                "action": "topup_selection",
                "pool_id": pool_id,
                "manifest_path": str(manifest_path),
                **result,
            }
        )
        pools = [pool_state(pool) for pool in state.get("pools", []) if isinstance(pool, dict)]

    aggregate_selected, aggregate_downloaded = aggregate_rows(pools, int(args.objaverse_target))
    write_json(
        aggregate_raw_manifest,
        {
            "generated_at_utc": utc_now(),
            "target_total": int(args.objaverse_target),
            "pool_count": len(pools),
            "selected_count": len(aggregate_selected),
            "downloaded_count": len(aggregate_downloaded),
            "missing_count": max(int(args.objaverse_target) - len(aggregate_downloaded), 0),
            "records": aggregate_selected,
        },
    )
    write_json(
        aggregate_downloaded_manifest,
        {
            "generated_at_utc": utc_now(),
            "target_total": int(args.objaverse_target),
            "downloaded_count": len(aggregate_downloaded),
            "records": aggregate_downloaded,
        },
    )
    source_manifest_files = build_aggregate_source_manifest(args.objaverse_source_dir, aggregate_downloaded_manifest)
    state["generated_at_utc"] = utc_now()
    state["target_total"] = int(args.objaverse_target)
    state["pools"] = [
        {
            "pool_id": pool["pool_id"],
            "selection_mode": pool["selection_mode"],
            "target": int(pool.get("target") or 0),
            "pool_dir": str(pool["pool_dir"]),
            "manifest_path": str(pool["manifest_path"]),
            "selected_parquet": str(pool["selected_parquet"]),
        }
        for pool in pools
    ]
    state.setdefault("history", []).extend(actions)
    write_json(state_path, state)
    status = write_objaverse_status(
        args=args,
        objaverse_plan_root=objaverse_plan_root,
        pools=pools,
        aggregate_selected=aggregate_selected,
        aggregate_downloaded=aggregate_downloaded,
        actions=actions,
        source_manifest_files=source_manifest_files,
        effective_download_mode=effective_download_mode,
    )
    return {
        "status": status,
        "source_manifest_path": Path(source_manifest_files.get("compat_json") or args.objaverse_source_dir / "source_candidate_manifest.json"),
        "aggregate_manifest": aggregate_raw_manifest,
    }


def summarize_source_candidate_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    records = [row for row in rows if isinstance(row, dict)]
    return {
        "records": len(records),
        "source_name": dict(Counter(str(row.get("source_name") or "unknown") for row in records)),
        "license_bucket": dict(Counter(str(row.get("license_bucket") or "unknown") for row in records)),
        "material_family": dict(
            Counter(
                str(row.get("material_family") or row.get("highlight_material_class") or "unknown")
                for row in records
            )
        ),
    }


def write_source_sidecars(source_dir: Path, summary: dict[str, Any]) -> None:
    write_text(
        source_dir / "source_progress.md",
        "\n".join(
            [
                "# Source Progress",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- records: `{summary['records']}`",
                f"- source_name: `{json.dumps(summary['source_name'], ensure_ascii=False)}`",
            ]
        ),
    )
    write_text(
        source_dir / "source_license_summary.md",
        "\n".join(
            [
                "# Source License Summary",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- license_bucket: `{json.dumps(summary['license_bucket'], ensure_ascii=False)}`",
            ]
        ),
    )
    write_text(
        source_dir / "source_material_family_guess.md",
        "\n".join(
            [
                "# Source Material Family Guess",
                "",
                f"- generated_at_utc: `{utc_now()}`",
                f"- material_family: `{json.dumps(summary['material_family'], ensure_ascii=False)}`",
            ]
        ),
    )


def main() -> None:
    args = parse_args()
    args.stage_root.mkdir(parents=True, exist_ok=True)
    args.objaverse_stage_root.mkdir(parents=True, exist_ok=True)
    args.objaverse_source_dir.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = []
    effective_download_mode = str(args.download_mode)
    proxy_env_present = any(os.environ.get(key) for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))
    effective_target = int(args.objaverse_target)
    should_download = bool(args.download_objaverse) and effective_download_mode != "off"
    objaverse_plan_root = args.stage_root / "objaverse_1200_serial"
    objaverse_download_status = objaverse_plan_root / "objaverse_1200_download_status.json"
    download_attempted = 0
    download_succeeded = 0
    download_failure_reason = ""
    download_success_rate = 0.0
    source_summary = {"records": 0, "source_name": {}, "license_bucket": {}, "material_family": {}}
    source_manifest_path = args.objaverse_source_dir / "source_candidate_manifest.json"
    objaverse_plan_status: dict[str, Any] = {}

    if effective_target > 0:
        try:
            objaverse_stage = stage_objaverse_increment(
                args,
                should_download=should_download,
                effective_download_mode=effective_download_mode,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        objaverse_plan_status = objaverse_stage["status"]
        command_log.extend(objaverse_plan_status.get("actions") or [])
        download_attempted = int(objaverse_plan_status.get("download_attempted_latest") or 0)
        download_succeeded = int(objaverse_plan_status.get("downloaded_latest") or 0)
        download_failure_reason = str(objaverse_plan_status.get("download_failure_reason_latest") or "")
        download_success_rate = float(objaverse_plan_status.get("download_success_rate_latest") or 0.0)
        source_manifest_path = Path(str(objaverse_stage.get("source_manifest_path") or source_manifest_path))

        if file_exists(source_manifest_path):
            source_summary = summarize_source_candidate_manifest(source_manifest_path)
            write_source_sidecars(args.objaverse_source_dir, source_summary)

    merge_result = None
    if not args.skip_merge:
        merge_result = run_cmd(
            [
                sys.executable,
                "scripts/merge_material_refine_expansion_candidates.py",
                "--expansion-root",
                str(args.expansion_root),
                "--output-manifest",
                str(args.merged_manifest),
            ],
            cwd=REPO_ROOT,
        )
        command_log.append(merge_result)
        if merge_result["returncode"] != 0:
            raise SystemExit(merge_result["stderr"] or merge_result["stdout"] or "merge_expansion_candidates_failed")

    second_pass_result = None
    if not args.skip_second_pass:
        second_pass_result = run_cmd(
            [
                sys.executable,
                "scripts/build_material_refine_trainV5_expansion_second_pass.py",
                "--input-manifest",
                str(args.merged_manifest),
            ],
            cwd=REPO_ROOT,
        )
        command_log.append(second_pass_result)
        if second_pass_result["returncode"] != 0:
            raise SystemExit(second_pass_result["stderr"] or second_pass_result["stdout"] or "second_pass_failed")

    repair_result = None
    if not args.skip_repair:
        repair_result = run_cmd(
            [
                sys.executable,
                "scripts/build_material_refine_trainV5_repair_and_expansion_plan.py",
                "--second-pass-dir",
                str(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass"),
            ],
            cwd=REPO_ROOT,
        )
        command_log.append(repair_result)
        if repair_result["returncode"] != 0:
            raise SystemExit(repair_result["stderr"] or repair_result["stdout"] or "repair_and_queue_build_failed")

    objaverse_artifacts_result = run_cmd(
        [
            sys.executable,
            "scripts/build_trainv5_objaverse_1200_serial_artifacts.py",
            "--source-manifest",
            str(source_manifest_path),
            "--second-pass-dir",
            str(REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass"),
            "--target-total",
            str(args.objaverse_target),
        ],
        cwd=REPO_ROOT,
    )
    command_log.append(objaverse_artifacts_result)
    if objaverse_artifacts_result["returncode"] != 0:
        raise SystemExit(objaverse_artifacts_result["stderr"] or objaverse_artifacts_result["stdout"] or "objaverse_1200_artifact_build_failed")

    summary = {
        "generated_at_utc": utc_now(),
        "objaverse_target": args.objaverse_target,
        "effective_objaverse_target": effective_target,
        "objaverse_sources": [item.strip() for item in args.objaverse_sources.split(",") if item.strip()],
        "priority_material_families": [item.strip() for item in args.priority_material_families.split(",") if item.strip()],
        "target_material_family_rats": args.target_material_family_ratios,
        "download_objaverse": should_download,
        "download_mode": effective_download_mode,
        "download_transport": effective_download_mode,
        "download_probe_size": int(args.download_probe_size),
        "min_download_success_rate": float(args.min_download_success_rate),
        "proxy_env_present": proxy_env_present,
        "download_attempted": int(download_attempted),
        "download_succeeded": int(download_succeeded),
        "download_success_rate": float(download_success_rate),
        "download_failure_reason": download_failure_reason,
        "download_below_threshold": bool(
            should_download and download_attempted > 0 and download_success_rate < float(args.min_download_success_rate)
        ),
        "source_candidate_manifest": str(source_manifest_path.resolve()) if source_manifest_path.exists() else str(source_manifest_path),
        "objaverse_1200_download_status": str(objaverse_download_status.resolve()) if objaverse_download_status.exists() else str(objaverse_download_status),
        "objaverse_1200_ready_exact_1200": bool(objaverse_plan_status.get("ready_exact_1200")),
        "objaverse_1200_missing_total": int(objaverse_plan_status.get("missing_total") or 0),
        "objaverse_1200_retry_round": int(objaverse_plan_status.get("retry_round") or 0),
        "objaverse_1200_topup_needed": bool(objaverse_plan_status.get("topup_needed")),
        "merged_manifest": str(args.merged_manifest.resolve()),
        "queue_latest": str((REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/trainV5_plus_rebake_queue_latest.json").resolve()),
        "objaverse_1200_second_pass_summary": str((REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/objaverse_1200_serial/objaverse_1200_second_pass_summary.json").resolve()),
        "objaverse_1200_repair_summary": str((REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/objaverse_1200_serial/objaverse_1200_repair_summary.json").resolve()),
        "objaverse_1200_rebake_input_manifest": str((REPO_ROOT / "output/material_refine_trainV5/expansion_second_pass/objaverse_1200_serial/objaverse_1200_rebake_input_manifest.json").resolve()),
        "command_log_count": len(command_log),
        "source_summary": source_summary,
    }
    write_json(args.stage_root / "material_priority_stage_summary.json", {"summary": summary, "commands": command_log})
    write_text(
        args.stage_root / "material_priority_stage_summary.md",
        "\n".join(
            [
                "# Material Priority Stage Summary",
                "",
                f"- generated_at_utc: `{summary['generated_at_utc']}`",
                f"- objaverse_target: `{summary['objaverse_target']}`",
                f"- effective_objaverse_target: `{summary['effective_objaverse_target']}`",
                f"- objaverse_sources: `{json.dumps(summary['objaverse_sources'], ensure_ascii=False)}`",
                f"- priority_material_families: `{json.dumps(summary['priority_material_families'], ensure_ascii=False)}`",
                f"- download_objaverse: `{str(summary['download_objaverse']).lower()}`",
                f"- download_mode: `{summary['download_mode']}`",
                f"- download_transport: `{summary['download_transport']}`",
                f"- proxy_env_present: `{str(summary['proxy_env_present']).lower()}`",
                f"- download_attempted: `{summary['download_attempted']}`",
                f"- download_succeeded: `{summary['download_succeeded']}`",
                f"- download_success_rate: `{summary['download_success_rate']}`",
                f"- download_failure_reason: `{summary['download_failure_reason']}`",
                f"- download_below_threshold: `{str(summary['download_below_threshold']).lower()}`",
                f"- objaverse_1200_download_status: `{summary['objaverse_1200_download_status']}`",
                f"- objaverse_1200_ready_exact_1200: `{str(summary['objaverse_1200_ready_exact_1200']).lower()}`",
                f"- objaverse_1200_missing_total: `{summary['objaverse_1200_missing_total']}`",
                f"- objaverse_1200_retry_round: `{summary['objaverse_1200_retry_round']}`",
                f"- objaverse_1200_topup_needed: `{str(summary['objaverse_1200_topup_needed']).lower()}`",
                f"- source_candidate_manifest: `{summary['source_candidate_manifest']}`",
                f"- merged_manifest: `{summary['merged_manifest']}`",
                f"- queue_latest: `{summary['queue_latest']}`",
                f"- objaverse_1200_second_pass_summary: `{summary['objaverse_1200_second_pass_summary']}`",
                f"- objaverse_1200_repair_summary: `{summary['objaverse_1200_repair_summary']}`",
                f"- objaverse_1200_rebake_input_manifest: `{summary['objaverse_1200_rebake_input_manifest']}`",
                f"- source_summary: `{json.dumps(summary['source_summary'], ensure_ascii=False)}`",
                "",
                "This stage does not launch GPU rebake or training. It only stages sources, merges candidates, and refreshes CPU-side queue artifacts.",
            ]
        ),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
