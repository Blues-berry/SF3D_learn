#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPANSION_ROOT = REPO_ROOT / "output/material_refine_expansion_candidates"
DEFAULT_PLAN_ROOT = DEFAULT_EXPANSION_ROOT / "material_priority_stage/objaverse_1200_serial"
DEFAULT_OBJAVERSE_ROOT = REPO_ROOT / "output/highlight_pool_a_8k/aux_sources/objaverse_xl"
DEFAULT_MERGED_MANIFEST = DEFAULT_EXPANSION_ROOT / "merged_expansion_candidate_manifest.json"
DEFAULT_SOURCE_DIR = DEFAULT_EXPANSION_ROOT / "material_priority_objaverse_increment"
DEFAULT_TARGET = 1200
DEFAULT_MAX_RETRY_ROUNDS = 2
DEFAULT_MAX_ACTIONS_PER_RUN = 4


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--expansion-root", type=Path, default=DEFAULT_EXPANSION_ROOT)
    parser.add_argument("--merged-manifest", type=Path, default=DEFAULT_MERGED_MANIFEST)
    parser.add_argument("--objaverse-root", type=Path, default=DEFAULT_OBJAVERSE_ROOT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--sources", type=str, default="smithsonian,thingiverse,sketchfab")
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
    parser.add_argument("--download-mode", choices=("direct", "proxy-probe", "mirror-probe", "off"), default="direct")
    parser.add_argument("--download-objaverse", action="store_true")
    parser.add_argument("--max-retry-rounds", type=int, default=DEFAULT_MAX_RETRY_ROUNDS)
    parser.add_argument("--max-actions-per-run", type=int, default=DEFAULT_MAX_ACTIONS_PER_RUN)
    return parser.parse_args()


def manifest_rows(path: Path | None) -> list[dict[str, Any]]:
    payload = read_json(path, {})
    rows = payload.get("records", []) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "source": dict(Counter(str(row.get("source") or "unknown") for row in rows)),
        "material": dict(Counter(str(row.get("highlight_material_class") or "unknown") for row in rows)),
        "download_status": dict(Counter(str(row.get("download_status") or "unknown") for row in rows)),
    }


def pool_state(pool: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(pool["manifest_path"])
    payload = read_json(manifest_path, {})
    rows = manifest_rows(manifest_path)
    downloaded = sum(bool(row.get("local_path")) and Path(str(row.get("local_path"))).exists() for row in rows)
    selected = len(rows)
    missing = max(selected - downloaded, 0)
    retry_round = int(payload.get("retry_round") or 0) if isinstance(payload, dict) else int(pool.get("retry_round") or 0)
    state = dict(pool)
    state.update(
        {
            "selected_count": selected,
            "downloaded_count": downloaded,
            "missing_count": missing,
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
        for row in manifest_rows(Path(pool["manifest_path"])):
            source_uid = str(row.get("source_uid") or "")
            if not source_uid:
                continue
            existing = selected_by_uid.get(source_uid)
            local_path = str(row.get("local_path") or "")
            current_is_downloaded = bool(local_path) and Path(local_path).exists()
            if existing is None:
                selected_by_uid[source_uid] = dict(row)
            else:
                existing_is_downloaded = bool(existing.get("local_path")) and Path(str(existing.get("local_path"))).exists()
                if current_is_downloaded and not existing_is_downloaded:
                    selected_by_uid[source_uid] = dict(row)
            selected_by_uid[source_uid]["selection_pool_id"] = pool.get("pool_id")
    for pool in pools:
        for row in manifest_rows(Path(pool["manifest_path"])):
            local_path = str(row.get("local_path") or "")
            if local_path and Path(local_path).exists():
                item = dict(row)
                item["selection_pool_id"] = pool.get("pool_id")
                downloaded_rows.append(item)
    downloaded_rows.sort(key=lambda row: (str(row.get("selection_pool_id") or ""), str(row.get("source_uid") or "")))
    frozen_downloaded = downloaded_rows[:target_total]
    selected_rows = list(selected_by_uid.values())
    selected_rows.sort(key=lambda row: (str(row.get("selection_pool_id") or ""), str(row.get("source_uid") or "")))
    return selected_rows, frozen_downloaded


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def stage_pool(
    *,
    pool_dir: Path,
    pool_id: str,
    target: int,
    args: argparse.Namespace,
    selection_mode: str,
    selected_parquet: Path | None = None,
    existing_manifest: Path | None = None,
    exclude_selection_manifest: Path | None = None,
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
        args.sources,
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
    if args.download_objaverse and args.download_mode != "off":
        cmd.append("--download")
    result = run_cmd(cmd)
    manifest_path = pool_dir / "objaverse_cached_increment_manifest.json"
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"] or f"stage_pool_failed:{pool_id}")
    return manifest_path, result


def build_aggregate_source_manifest(args: argparse.Namespace, aggregate_downloaded_manifest: Path) -> dict[str, Any]:
    output_json = args.source_dir / "objaverse_1200_source_candidate_manifest.json"
    output_csv = args.source_dir / "objaverse_1200_source_candidate_manifest.csv"
    output_md = args.source_dir / "objaverse_1200_source_candidate_manifest_summary.md"
    compat_json = args.source_dir / "source_candidate_manifest.json"
    compat_csv = args.source_dir / "source_candidate_manifest.csv"
    compat_md = args.source_dir / "source_candidate_manifest_summary.md"
    result = run_cmd(
        [
            sys.executable,
            "scripts/build_objaverse_increment_manifest.py",
            "--input-json",
            str(aggregate_downloaded_manifest),
            "--output-root",
            str(args.source_dir),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
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


def write_status(
    *,
    args: argparse.Namespace,
    pools: list[dict[str, Any]],
    aggregate_selected: list[dict[str, Any]],
    aggregate_downloaded: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    source_manifest_files: dict[str, Any],
) -> None:
    status_json = args.plan_root / "objaverse_1200_download_status.json"
    status_md = args.plan_root / "objaverse_1200_download_status.md"
    downloaded_total = len(aggregate_downloaded)
    selected_total = len(aggregate_selected)
    missing_total = max(args.target_total - downloaded_total, 0)
    last_action = actions[-1] if actions else {}
    last_manifest_path = Path(str(last_action.get("manifest_path") or "")) if last_action else None
    last_manifest_payload = read_json(last_manifest_path, {}) if last_manifest_path and last_manifest_path.exists() else {}
    latest_attempted = int(last_manifest_payload.get("download_attempted_count") or 0) if isinstance(last_manifest_payload, dict) else 0
    latest_downloaded = int(last_manifest_payload.get("downloaded_count") or 0) if isinstance(last_manifest_payload, dict) else 0
    latest_error = str(last_manifest_payload.get("download_error") or "") if isinstance(last_manifest_payload, dict) else ""
    latest_success_rate = float(latest_downloaded) / float(latest_attempted) if latest_attempted > 0 else 0.0
    retryable_pools = [
        pool["pool_id"]
        for pool in pools
        if int(pool.get("missing_count") or 0) > 0 and int(pool.get("retry_round") or 0) < args.max_retry_rounds
    ]
    topup_needed = downloaded_total < args.target_total and not retryable_pools
    payload = {
        "generated_at_utc": utc_now(),
        "target_total": int(args.target_total),
        "selected_total": selected_total,
        "downloaded_total": downloaded_total,
        "missing_total": missing_total,
        "retry_round": max((int(pool.get("retry_round") or 0) for pool in pools), default=0),
        "topup_needed": bool(topup_needed),
        "ready_exact_1200": downloaded_total >= args.target_total,
        "download_mode": args.download_mode,
        "download_attempted_latest": latest_attempted,
        "downloaded_latest": latest_downloaded,
        "download_success_rate_latest": latest_success_rate,
        "download_failure_reason_latest": latest_error,
        "download_success_rate_aggregate": (float(downloaded_total) / float(selected_total)) if selected_total > 0 else 0.0,
        "source_manifest_files": source_manifest_files,
        "pool_count": len(pools),
        "pools": pools,
        "actions": actions,
        "aggregate_selected_summary": summarize_rows(aggregate_selected),
        "aggregate_downloaded_summary": summarize_rows(aggregate_downloaded),
    }
    write_json(status_json, payload)
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
    write_text(status_md, "\n".join(lines))


def main() -> None:
    print(
        "DEPRECATED: use `python datasetscrip/trainv5_dataset.py ingest` for material-priority source staging.",
        file=sys.stderr,
    )
    args = parse_args()
    args.plan_root.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.plan_root / "objaverse_1200_plan_state.json"
    aggregate_raw_manifest = args.plan_root / "objaverse_1200_aggregate_increment_manifest.json"
    aggregate_downloaded_manifest = args.plan_root / "objaverse_1200_downloaded_increment_manifest.json"
    state = read_json(state_path, {"pools": [], "history": []})
    pools = [pool_state(pool) for pool in state.get("pools", [])]
    actions: list[dict[str, Any]] = []

    for _step in range(max(int(args.max_actions_per_run), 1)):
        downloaded_total = len(aggregate_rows(pools, args.target_total)[1]) if pools else 0
        if downloaded_total >= args.target_total:
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
            manifest_path, result = stage_pool(
                pool_dir=Path(retry_pool["pool_dir"]),
                pool_id=str(retry_pool["pool_id"]),
                target=int(retry_pool.get("target") or 0),
                args=args,
                selection_mode="retry_missing",
                selected_parquet=Path(str(retry_pool["selected_parquet"])),
                existing_manifest=Path(str(retry_pool["manifest_path"])),
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
            pools = [pool_state(pool) for pool in state.get("pools", [])]
            for index, pool in enumerate(state.get("pools", [])):
                if pool.get("pool_id") == retry_pool["pool_id"]:
                    state["pools"][index]["manifest_path"] = str(manifest_path)
                    state["pools"][index]["selected_parquet"] = str(Path(retry_pool["selected_parquet"]))
            pools = [pool_state(pool) for pool in state.get("pools", [])]
            continue

        if not pools:
            base_dir = args.plan_root / "base_selection"
            manifest_path, result = stage_pool(
                pool_dir=base_dir,
                pool_id="base_selection",
                target=int(args.target_total),
                args=args,
                selection_mode="initial_selection",
            )
            pool_entry = {
                "pool_id": "base_selection",
                "selection_mode": "initial_selection",
                "target": int(args.target_total),
                "pool_dir": str(base_dir),
                "manifest_path": str(manifest_path),
                "selected_parquet": str(base_dir / "objaverse_cached_selected.parquet"),
            }
            state.setdefault("pools", []).append(pool_entry)
            actions.append(
                {
                    "timestamp_utc": utc_now(),
                    "action": "initial_selection",
                    "pool_id": "base_selection",
                    "manifest_path": str(manifest_path),
                    **result,
                }
            )
            pools = [pool_state(pool) for pool in state.get("pools", [])]
            continue

        downloaded_total = len(aggregate_rows(pools, args.target_total)[1])
        gap = max(int(args.target_total) - int(downloaded_total), 0)
        if gap <= 0:
            break
        topup_index = 1 + sum(1 for pool in state.get("pools", []) if str(pool.get("selection_mode")) == "topup_selection")
        topup_pool_id = f"topup_round_{topup_index}"
        topup_dir = args.plan_root / topup_pool_id
        manifest_path, result = stage_pool(
            pool_dir=topup_dir,
            pool_id=topup_pool_id,
            target=gap,
            args=args,
            selection_mode="topup_selection",
            exclude_selection_manifest=aggregate_raw_manifest if aggregate_raw_manifest.exists() else None,
        )
        pool_entry = {
            "pool_id": topup_pool_id,
            "selection_mode": "topup_selection",
            "target": gap,
            "pool_dir": str(topup_dir),
            "manifest_path": str(manifest_path),
            "selected_parquet": str(topup_dir / "objaverse_cached_selected.parquet"),
        }
        state.setdefault("pools", []).append(pool_entry)
        actions.append(
            {
                "timestamp_utc": utc_now(),
                "action": "topup_selection",
                "pool_id": topup_pool_id,
                "manifest_path": str(manifest_path),
                **result,
            }
        )
        pools = [pool_state(pool) for pool in state.get("pools", [])]

    aggregate_selected, aggregate_downloaded = aggregate_rows(pools, args.target_total)
    aggregate_payload = {
        "generated_at_utc": utc_now(),
        "target_total": int(args.target_total),
        "pool_count": len(pools),
        "selected_count": len(aggregate_selected),
        "downloaded_count": len(aggregate_downloaded),
        "missing_count": max(int(args.target_total) - len(aggregate_downloaded), 0),
        "records": aggregate_selected,
    }
    write_json(aggregate_raw_manifest, aggregate_payload)
    write_json(
        aggregate_downloaded_manifest,
        {
            "generated_at_utc": utc_now(),
            "target_total": int(args.target_total),
            "downloaded_count": len(aggregate_downloaded),
            "records": aggregate_downloaded,
        },
    )
    source_manifest_files = build_aggregate_source_manifest(args, aggregate_downloaded_manifest)

    state["generated_at_utc"] = utc_now()
    state["target_total"] = int(args.target_total)
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
    write_status(
        args=args,
        pools=pools,
        aggregate_selected=aggregate_selected,
        aggregate_downloaded=aggregate_downloaded,
        actions=actions,
        source_manifest_files=source_manifest_files,
    )


if __name__ == "__main__":
    main()
