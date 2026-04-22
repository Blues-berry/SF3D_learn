#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "material_refine_dataset_factory_gpu0.json"
DEFAULT_STATE = REPO_ROOT / "output" / "material_refine_dataset_factory" / "factory_state.json"
PYTHON_BIN = Path("/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Continuously stage SF3D material-refine sources and schedule GPU0-only dataset preprocessing.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--once", action="store_true", help="Run one scheduling pass and exit.")
    parser.add_argument("--loop", action="store_true", help="Run forever, sleeping according to config.gpu_policy.poll_seconds.")
    parser.add_argument("--start-downloads", action="store_true", help="Start enabled download/source-refresh retry sessions.")
    parser.add_argument("--start-render", action="store_true", help="Start a GPU0 longrun render if no active longrun shards are found.")
    parser.add_argument("--audit", action="store_true", help="Audit the newest merged/full manifest if available.")
    parser.add_argument("--force-render", action="store_true", help="Start another longrun even if GPU0 appears busy.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_capture(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def tmux_ls() -> list[str]:
    result = run_capture(["tmux", "ls"])
    if result.returncode != 0:
        return []
    return [line.split(":", 1)[0] for line in result.stdout.splitlines() if line.strip()]


def session_exists(session: str) -> bool:
    return session in set(tmux_ls())


def gpu_rows() -> list[dict[str, int]]:
    result = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, int]] = []
    if result.returncode != 0:
        return rows
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
                "utilization_gpu_percent": int(parts[3]),
            }
        )
    return rows


def active_longrun_sessions(config: dict[str, Any]) -> list[str]:
    prefix = str(config["longrun"].get("session_prefix", "sf3d_longrun_material_refine"))
    return sorted(
        session
        for session in tmux_ls()
        if session.startswith(f"{prefix}_shard") or session == f"{prefix}_merged_monitor"
    )


def active_longrun_shard_sessions(config: dict[str, Any]) -> list[str]:
    prefix = str(config["longrun"].get("session_prefix", "sf3d_longrun_material_refine"))
    return sorted(session for session in tmux_ls() if session.startswith(f"{prefix}_shard"))


def shell_join(items: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in items)


def proxy_probe_shell(probe_url: str | None, candidates: list[str], timeout_seconds: int) -> str:
    if not probe_url:
        return ""
    candidate_text = "|".join(candidates or ["env", "direct"])
    return f"""
  PROBE_URL={shlex.quote(str(probe_url))}
  PROXY_CANDIDATES={shlex.quote(candidate_text)}
  PROXY_PROBE_TIMEOUT={int(timeout_seconds)}
  best_proxy=""
  best_ms=999999999
  IFS='|' read -r -a proxy_candidates <<< "${{PROXY_CANDIDATES}}"
  for candidate in "${{proxy_candidates[@]}}"; do
    if [ "${{candidate}}" = "env" ]; then
      candidate="${{HTTPS_PROXY:-${{HTTP_PROXY:-}}}}"
    fi
    if [ -z "${{candidate}}" ]; then
      continue
    fi
    start_ms=$(date +%s%3N)
    if [ "${{candidate}}" = "direct" ]; then
      env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \\
        curl -fsSIL --max-time "${{PROXY_PROBE_TIMEOUT}}" "${{PROBE_URL}}" >/dev/null 2>&1
    else
      HTTP_PROXY="${{candidate}}" HTTPS_PROXY="${{candidate}}" http_proxy="${{candidate}}" https_proxy="${{candidate}}" \\
        curl -fsSIL --max-time "${{PROXY_PROBE_TIMEOUT}}" "${{PROBE_URL}}" >/dev/null 2>&1
    fi
    rc=$?
    end_ms=$(date +%s%3N)
    elapsed=$((end_ms - start_ms))
    if [ $rc -eq 0 ] && [ $elapsed -lt $best_ms ]; then
      best_ms=$elapsed
      best_proxy="${{candidate}}"
    fi
  done
  if [ -n "${{best_proxy}}" ]; then
    if [ "${{best_proxy}}" = "direct" ]; then
      unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
    else
      export HTTP_PROXY="${{best_proxy}}" HTTPS_PROXY="${{best_proxy}}" http_proxy="${{best_proxy}}" https_proxy="${{best_proxy}}"
    fi
    echo "==== selected_proxy ${{best_proxy}} ${{best_ms}}ms for ${{PROBE_URL}} ===="
  else
    echo "==== selected_proxy keep_existing none_reachable for ${{PROBE_URL}} ===="
  fi
"""


def start_retry_session(
    *,
    session: str,
    command: list[str],
    output_root: Path,
    retry_seconds: int,
    proxy_probe_url: str | None,
    proxy_candidates: list[str],
    proxy_probe_timeout_seconds: int,
    dry_run: bool,
) -> dict[str, Any]:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_script = log_dir / f"{session}.run.sh"
    log_path = log_dir / f"{session}.log"
    script = f"""#!/usr/bin/env bash
set -uo pipefail
cd {shlex.quote(str(REPO_ROOT))}
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "==== ${{attempt}} $(date -Iseconds) ===="
{proxy_probe_shell(proxy_probe_url, proxy_candidates, proxy_probe_timeout_seconds)}
  {shell_join(command)}
  rc=$?
  echo "==== exit ${{rc}} $(date -Iseconds) ===="
  if [ $rc -eq 0 ]; then
    break
  fi
  sleep {int(retry_seconds)}
done
"""
    if dry_run:
        return {"session": session, "action": "dry_run_start", "run_script": str(run_script)}
    run_script.write_text(script, encoding="utf-8")
    run_script.chmod(0o755)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, f"{shlex.quote(str(run_script))} > {shlex.quote(str(log_path))} 2>&1"],
        cwd=REPO_ROOT,
        check=True,
    )
    return {"session": session, "action": "started", "run_script": str(run_script), "log": str(log_path)}


def start_downloads(config: dict[str, Any], *, dry_run: bool) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for source in config.get("downloads", []):
        if not bool(source.get("enabled", True)):
            actions.append({"name": source.get("name"), "action": "disabled"})
            continue
        session = str(source["session"])
        if session_exists(session):
            actions.append({"name": source.get("name"), "session": session, "action": "already_running"})
            continue
        actions.append(
            {
                "name": source.get("name"),
                **start_retry_session(
                    session=session,
                    command=[str(item) for item in source["command"]],
                    output_root=REPO_ROOT / str(source["output_root"]),
                    retry_seconds=int(source.get("retry_seconds", 900)),
                    proxy_probe_url=source.get("proxy_probe_url") or config.get("download_network", {}).get("default_probe_url"),
                    proxy_candidates=[
                        str(item)
                        for item in source.get(
                            "proxy_candidates",
                            config.get("download_network", {}).get("proxy_candidates", ["env", "direct"]),
                        )
                    ],
                    proxy_probe_timeout_seconds=int(
                        source.get(
                            "proxy_probe_timeout_seconds",
                            config.get("download_network", {}).get("proxy_probe_timeout_seconds", 8),
                        )
                    ),
                    dry_run=dry_run,
                ),
            }
        )
    return actions


def launch_longrun(config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    longrun = config["longrun"]
    cleanup_orphan_monitor(config, dry_run=dry_run)
    output_root = REPO_ROOT / "output" / "material_refine_dataset_factory" / (
        "longrun_gpu0_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    env = os.environ.copy()
    input_manifests = resolve_input_manifests(config)
    if not input_manifests:
        return {"action": "not_launched", "reason": "no_input_manifests"}
    env.update(
        {
            "SESSION_PREFIX": str(longrun.get("session_prefix", "sf3d_longrun_material_refine")),
            "LONGRUN_INPUT_MANIFESTS": ",".join(str(path) for path in input_manifests),
            "PAPER_MAIN_SOURCES": str(longrun["paper_main_sources"]),
            "AUXILIARY_SOURCES": str(longrun.get("auxiliary_sources", "")),
            "PRIORITY_MATERIAL_FAMILIES": str(longrun["priority_material_families"]),
            "TARGET_MATERIAL_FAMILY_RATIOS": ",".join(
                f"{key}={value}" for key, value in config.get("material_quota", {}).items()
            ),
            "PAPER_FRONTLOAD_RECORDS": str(longrun.get("paper_frontload_records", 0)),
            "PREFER_PAPER_MAIN_FIRST": "1" if bool(longrun.get("prefer_paper_main_first", True)) else "0",
            "MAX_RECORDS": str(longrun["max_records"]),
            "SHARDS": str(longrun["shards"]),
            "GPU_LIST": str(config["gpu_policy"].get("render_gpu_list", "0")),
            "WORKERS_PER_GPU": str(longrun["workers_per_gpu"]),
            "VIEW_LIGHT_PROTOCOL": str(longrun["view_light_protocol"]),
            "MAX_HDRI_LIGHTS": str(longrun["max_hdri_lights"]),
            "MIN_HDRI_COUNT": str(longrun["min_hdri_count"]),
            "ATLAS_RESOLUTION": str(longrun["atlas_resolution"]),
            "RENDER_RESOLUTION": str(longrun["render_resolution"]),
            "CYCLES_SAMPLES": str(longrun["cycles_samples"]),
            "REFRESH_PARTIAL_EVERY": str(longrun["refresh_partial_every"]),
            "START_MERGED_MONITOR": "1",
            "MERGED_MONITOR_POLL_SECONDS": str(longrun.get("merged_monitor_poll_seconds", 120)),
            "OUTPUT_ROOT": str(output_root.relative_to(REPO_ROOT)),
            "HDRI_SELECTION_OFFSET": str(next_hdri_offset(config)),
        }
    )
    command = ["bash", "scripts/launch_material_refine_longrun_dataset_tmux.sh"]
    if dry_run:
        return {"action": "dry_run_launch", "output_root": str(output_root), "env": {k: env[k] for k in env if k in {
            "MAX_RECORDS", "SHARDS", "GPU_LIST", "WORKERS_PER_GPU", "VIEW_LIGHT_PROTOCOL", "HDRI_SELECTION_OFFSET", "OUTPUT_ROOT", "PAPER_FRONTLOAD_RECORDS"
        }}}
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        "action": "launched" if result.returncode == 0 else "launch_failed",
        "returncode": result.returncode,
        "output_root": str(output_root),
        "stdout_tail": result.stdout.splitlines()[-40:],
    }


def cleanup_orphan_monitor(config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    prefix = str(config["longrun"].get("session_prefix", "sf3d_longrun_material_refine"))
    monitor_session = f"{prefix}_merged_monitor"
    if active_longrun_shard_sessions(config) or not session_exists(monitor_session):
        return {"action": "none"}
    if dry_run:
        return {"action": "dry_run_kill_orphan_monitor", "session": monitor_session}
    subprocess.run(["tmux", "kill-session", "-t", monitor_session], cwd=REPO_ROOT, check=False)
    return {"action": "killed_orphan_monitor", "session": monitor_session}


def next_hdri_offset(config: dict[str, Any]) -> int:
    longrun = config["longrun"]
    start = int(longrun.get("hdri_selection_offset_start", 0))
    step = int(longrun.get("hdri_selection_offset_step", 200))
    existing = sorted((REPO_ROOT / "output" / "material_refine_dataset_factory").glob("longrun_gpu0_*"))
    return start + step * len(existing)


def resolve_input_manifests(config: dict[str, Any]) -> list[str]:
    longrun = config["longrun"]
    paths: list[Path] = []
    for value in longrun.get("input_manifests", []):
        path = REPO_ROOT / str(value)
        if path.exists():
            paths.append(path)
    for pattern in longrun.get("input_manifest_globs", []):
        paths.extend(sorted(REPO_ROOT.glob(str(pattern))))
    seen: set[str] = set()
    resolved: list[str] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            resolved.append(str(path.relative_to(REPO_ROOT)))
    return resolved


def canonicalize_download_manifests(config: dict[str, Any], *, dry_run: bool) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in config.get("canonicalize_downloads", []):
        if not bool(item.get("enabled", True)):
            actions.append({"name": item.get("name"), "action": "disabled"})
            continue
        input_json = REPO_ROOT / str(item["input_json"])
        output_root = REPO_ROOT / str(item["output_root"])
        output_json = output_root / "material_refine_manifest_objaverse_increment.json"
        if not input_json.exists():
            actions.append({"name": item.get("name"), "action": "missing_input", "input_json": str(input_json)})
            continue
        if output_json.exists() and output_json.stat().st_mtime >= input_json.stat().st_mtime:
            actions.append({"name": item.get("name"), "action": "up_to_date", "output_json": str(output_json)})
            continue
        command = [
            str(PYTHON_BIN),
            "scripts/build_objaverse_increment_manifest.py",
            "--input-json",
            str(input_json),
            "--output-root",
            str(output_root),
        ]
        if dry_run:
            actions.append({"name": item.get("name"), "action": "dry_run_canonicalize", "command": command})
            continue
        result = run_capture(command)
        actions.append(
            {
                "name": item.get("name"),
                "action": "canonicalized" if result.returncode == 0 else "canonicalize_failed",
                "input_json": str(input_json),
                "output_json": str(output_json),
                "returncode": result.returncode,
                "stdout_tail": result.stdout.splitlines()[-30:],
            }
        )
    return actions


def manifest_record_count(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    records = payload.get("records")
    return len(records) if isinstance(records, list) else 0


def newest_manifest_candidate(config: dict[str, Any]) -> Path | None:
    patterns = [
        "output/material_refine_dataset_factory/paper_unlock_gpu0_*/canonical_manifest_monitor_merged.json",
        "output/material_refine_dataset_factory/paper_unlock_gpu0_*/prepared_shard_*/full/canonical_manifest_full.json",
        "output/material_refine_dataset_factory/longrun_gpu0_*/canonical_manifest_monitor_merged.json",
        "output/material_refine_dataset_factory/longrun_gpu0_*/prepared_shard_*/full/canonical_manifest_full.json",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(REPO_ROOT.glob(pattern))
    min_records = int(config.get("quality", {}).get("min_manifest_records_for_audit", 1))
    paths = [
        path
        for path in paths
        if path.exists() and manifest_record_count(path) >= min_records
    ]
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def audit_newest(config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    manifest = newest_manifest_candidate(config)
    if manifest is None:
        return {
            "action": "no_manifest_found",
            "reason": "no_factory_manifest_with_min_records",
            "min_records": int(config.get("quality", {}).get("min_manifest_records_for_audit", 1)),
        }
    out_root = REPO_ROOT / str(config["quality"].get("quality_output_root", "output/material_refine_dataset_factory/quality"))
    out_dir = out_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    command = [
        str(PYTHON_BIN),
        "scripts/evaluate_material_refine_dataset_quality.py",
        "--manifest",
        str(manifest),
        "--output-dir",
        str(out_dir),
        "--min-paper-eligible",
        str(config["quality"].get("min_paper_eligible", 128)),
        "--min-view-ready-rate",
        str(config["quality"].get("min_view_ready_rate", 0.8)),
        "--min-strict-complete-rate",
        str(config["quality"].get("min_strict_complete_rate", 0.8)),
    ]
    if dry_run:
        return {"action": "dry_run_audit", "manifest": str(manifest), "command": command}
    result = run_capture(command)
    return {
        "action": "audited" if result.returncode == 0 else "audit_failed",
        "manifest": str(manifest),
        "output_dir": str(out_dir),
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-40:],
    }


def should_launch_render(config: dict[str, Any], *, force: bool) -> tuple[bool, str]:
    active = active_longrun_shard_sessions(config)
    if active:
        return False, f"active_longrun_shard_sessions={len(active)}"
    if force:
        return True, "force_render"
    rows = gpu_rows()
    gpu0 = next((row for row in rows if row["index"] == 0), None)
    if gpu0 is None:
        return False, "gpu0_not_found"
    max_used = int(config["gpu_policy"].get("max_gpu0_used_mb_for_new_launch", 4096))
    if gpu0["memory_used_mb"] > max_used:
        return False, f"gpu0_busy={gpu0['memory_used_mb']}MB>{max_used}MB"
    return True, "gpu0_idle"


def run_once(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "updated_at_utc": utc_now(),
        "config": str(args.config.resolve()),
        "gpu": gpu_rows(),
        "tmux_sessions": tmux_ls(),
        "actions": [],
    }
    if args.start_downloads:
        state["actions"].append({"downloads": start_downloads(config, dry_run=args.dry_run)})
        state["actions"].append({"canonicalize_downloads": canonicalize_download_manifests(config, dry_run=args.dry_run)})
    if args.start_render:
        launch, reason = should_launch_render(config, force=args.force_render)
        if launch:
            state["actions"].append({"render": launch_longrun(config, dry_run=args.dry_run)})
        else:
            state["actions"].append({"render": {"action": "not_launched", "reason": reason}})
    if args.audit:
        state["actions"].append({"audit": audit_newest(config, dry_run=args.dry_run)})
    state["active_longrun_sessions"] = active_longrun_sessions(config)
    return state


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    if not args.once and not args.loop:
        args.once = True
    while True:
        state = run_once(args, config)
        write_json(args.state_json, state)
        print(json.dumps(state, indent=2, ensure_ascii=False))
        if args.once:
            break
        time.sleep(int(config.get("gpu_policy", {}).get("poll_seconds", 300)))


if __name__ == "__main__":
    main()
