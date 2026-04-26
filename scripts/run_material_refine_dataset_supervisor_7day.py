#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob as globlib
import json
import os
import re
import signal
import shlex
import subprocess
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "material_refine_dataset_supervisor_7day_gpu0.json"


def repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def glob_paths(pattern: str) -> list[Path]:
    if not pattern:
        return []
    if Path(pattern).is_absolute():
        return sorted(Path(match) for match in globlib.glob(pattern, recursive=True))
    return sorted(REPO_ROOT.glob(pattern))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Seven-day data-only supervisor for SF3D material-refine dataset production. "
            "It keeps downloads, GPU0 preprocessing, partial manifest merges, audits, "
            "and buffer validation moving without launching training."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--once", action="store_true", help="Run one supervision pass and exit.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now_dt() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_now() -> str:
    return utc_now_dt().isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_capture(
    cmd: list[str],
    *,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def shell_join(values: list[str | Path]) -> str:
    return " ".join(shell_quote(value) for value in values)


def tmux_sessions() -> list[str]:
    result = run_capture(["tmux", "ls"])
    if result.returncode != 0:
        return []
    return [line.split(":", 1)[0] for line in result.stdout.splitlines() if line.strip()]


def session_exists(session: str, sessions: set[str] | None = None) -> bool:
    if sessions is None:
        sessions = set(tmux_sessions())
    return session in sessions


def start_tmux_session(session: str, command: str, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"session": session, "action": "dry_run_start", "command": command}
    result = run_capture(["tmux", "new-session", "-d", "-s", session, command])
    return {
        "session": session,
        "action": "started" if result.returncode == 0 else "start_failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-20:],
    }


def kill_tmux_session(session: str, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"session": session, "action": "dry_run_kill"}
    result = run_capture(["tmux", "kill-session", "-t", session])
    return {
        "session": session,
        "action": "killed" if result.returncode == 0 else "kill_failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-20:],
    }


def gpu_rows() -> list[dict[str, Any]]:
    result = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
                "utilization_gpu_percent": int(parts[4]),
            }
        )
    return rows


def gpu_process_rows() -> list[dict[str, Any]]:
    result = run_capture(["nvidia-smi", "pmon", "-c", "1"])
    if result.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10 or parts[1] == "-":
            continue
        rows.append(
            {
                "gpu": int(parts[0]),
                "pid": int(parts[1]),
                "type": parts[2],
                "sm": parts[3],
                "mem": parts[4],
                "command": parts[-1],
            }
        )
    return rows


def process_command(pid: int) -> str:
    result = run_capture(["ps", "-p", str(pid), "-o", "command="])
    return result.stdout.strip() if result.returncode == 0 else ""


def data_processes_on_gpu(gpu_index: int) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for row in gpu_process_rows():
        if row.get("gpu") != gpu_index:
            continue
        cmd = process_command(int(row["pid"]))
        if any(token in cmd for token in ("stable-fast-3d", "prepare_material_refine_dataset", "abo_material_passes_blender")):
            processes.append(dict(row, full_command=cmd))
    return processes


def data_processes_on_gpu1() -> list[dict[str, Any]]:
    return data_processes_on_gpu(1)


def gpu1_data_policy_offenders(config: dict[str, Any], gpu_snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policy = config.get("gpu_policy", {})
    gpu1_processes = data_processes_on_gpu1()
    if not gpu1_processes:
        return []
    allow_limited = bool(policy.get("allow_gpu1_data_processes", False))
    if not allow_limited:
        return [
            {
                "reason": "gpu1_data_processes_not_allowed",
                "processes": gpu1_processes,
            }
        ]
    limit_mb = int(policy.get("max_gpu1_data_memory_mb", 0) or 0)
    gpu1_rows = [row for row in gpu_snapshot if int(row.get("index", -1)) == 1]
    used_mb = int(gpu1_rows[0].get("memory_used_mb", 0)) if gpu1_rows else 0
    if limit_mb > 0 and used_mb > limit_mb:
        return [
            {
                "reason": "gpu1_memory_above_limit",
                "memory_used_mb": used_mb,
                "limit_mb": limit_mb,
                "processes": gpu1_processes,
            }
        ]
    return []


def discover_latest_roots(patterns: list[str], *, latest_only: bool) -> list[Path]:
    roots: list[Path] = []
    for pattern in patterns:
        roots.extend(path for path in glob_paths(pattern) if path.is_dir())
    roots = sorted(set(roots), key=lambda path: path.stat().st_mtime)
    if latest_only and roots:
        return [roots[-1]]
    return roots


def discover_latest_root(pattern: str) -> Path | None:
    roots = sorted((path for path in glob_paths(pattern) if path.is_dir()), key=lambda path: path.stat().st_mtime)
    return roots[-1] if roots else None


def production_root_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    production = dict(config.get("production_roots") or {})
    nested_groups = production.pop("groups", []) or []
    groups: list[dict[str, Any]] = []
    if production.get("patterns"):
        production.setdefault("name", "production_longrun")
        groups.append(production)

    defaults = {
        "latest_only": bool(production.get("latest_only", True)),
        "merge_output_name": str(production.get("merge_output_name", "canonical_manifest_supervisor_merged.json")),
        "required_prefixes": [str(item) for item in production.get("required_prefixes", [])],
    }
    for index, item in enumerate(nested_groups):
        if not isinstance(item, dict) or not item.get("patterns"):
            continue
        if not bool(item.get("enabled", True)):
            continue
        group = dict(defaults)
        group.update(item)
        group.setdefault("name", f"production_group_{index}")
        groups.append(group)
    return groups


def cleanup_disabled_production_groups(
    config: dict[str, Any],
    *,
    sessions: set[str],
    dry_run: bool,
) -> list[dict[str, Any]]:
    production = dict(config.get("production_roots") or {})
    nested_groups = production.get("groups", []) or []
    default_prefixes = [str(item) for item in production.get("required_prefixes", [])]
    actions: list[dict[str, Any]] = []
    for index, item in enumerate(nested_groups):
        if not isinstance(item, dict) or bool(item.get("enabled", True)):
            continue
        prefixes = [str(prefix) for prefix in item.get("required_prefixes", default_prefixes)]
        matching_sessions = sorted(
            session
            for session in sessions
            if any(session.startswith(prefix) for prefix in prefixes)
        )
        group_actions = [
            kill_tmux_session(session, dry_run=dry_run)
            for session in matching_sessions
        ]
        actions.append(
            {
                "name": str(item.get("name") or f"disabled_group_{index}"),
                "action": "cleaned_disabled_group_sessions" if matching_sessions else "no_sessions",
                "sessions": matching_sessions,
                "actions": group_actions,
            }
        )
    return actions


def run_script_session_name(run_script: Path) -> str:
    name = run_script.name
    if name.endswith(".run.sh"):
        return name[: -len(".run.sh")]
    return run_script.stem


def shard_index_from_session(session: str) -> int | None:
    match = re.search(r"_shard(\d+)_gpu\d+$", session)
    if not match:
        return None
    return int(match.group(1))


def shard_has_full_manifest(root: Path, session: str) -> bool:
    shard_idx = shard_index_from_session(session)
    if shard_idx is None:
        return False
    return (root / f"prepared_shard_{shard_idx}" / "full" / "canonical_manifest_full.json").exists()


def restart_missing_root_sessions(
    root: Path,
    *,
    prefixes: list[str],
    sessions: set[str],
    dry_run: bool,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for run_script in sorted((root / "logs").glob("*.run.sh")):
        session = run_script_session_name(run_script)
        if not any(session.startswith(prefix) for prefix in prefixes):
            continue
        if session_exists(session, sessions):
            actions.append({"session": session, "action": "already_running"})
            continue
        if "_shard" in session and shard_has_full_manifest(root, session):
            actions.append({"session": session, "action": "complete_not_restarted"})
            continue
        command = shell_quote(run_script)
        actions.append(start_tmux_session(session, command, dry_run=dry_run))
    return actions


def partial_manifests(root: Path) -> list[Path]:
    paths: list[Path] = []
    for name in ("canonical_manifest_monitor_partial.json", "canonical_manifest_partial.json", "canonical_manifest_full.json"):
        paths.extend(sorted(root.glob(f"prepared_shard_*/full/{name}")))
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def manifest_record_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    records = payload.get("records")
    return len(records) if isinstance(records, list) else 0


def merge_root_manifest(
    root: Path,
    *,
    output_name: str,
    python_bin: Path,
    dry_run: bool,
) -> dict[str, Any]:
    manifests = [path for path in partial_manifests(root) if manifest_record_count(path) > 0]
    output_manifest = root / output_name
    if not manifests:
        return {"root": str(root), "action": "no_partial_manifests", "output_manifest": str(output_manifest)}
    command: list[str | Path] = [
        python_bin,
        "scripts/merge_material_refine_partial_manifests.py",
        "--output-manifest",
        output_manifest,
    ]
    for manifest in manifests:
        command.extend(["--manifest", manifest])
    if dry_run:
        return {
            "root": str(root),
            "action": "dry_run_merge",
            "partials": len(manifests),
            "output_manifest": str(output_manifest),
        }
    result = run_capture([str(item) for item in command], timeout=300)
    return {
        "root": str(root),
        "action": "merged" if result.returncode == 0 else "merge_failed",
        "partials": len(manifests),
        "output_manifest": str(output_manifest),
        "records": manifest_record_count(output_manifest),
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-20:],
    }


def summarize_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"manifest": str(path), "exists": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"manifest": str(path), "exists": True, "error": str(exc)}
    records = payload.get("records") or []
    summary = dict(payload.get("summary") or {})
    for key in (
        "source_name",
        "supervision_role",
        "target_quality_tier",
        "target_source_type",
        "material_family",
        "paper_split",
        "lighting_bank_id",
    ):
        if key not in summary:
            summary[key] = dict(Counter(str(record.get(key) or "unknown") for record in records))
    eligible = 0
    prior_copy = 0
    for record in records:
        is_eligible = (
            str(record.get("supervision_role") or "") == "paper_main"
            and str(record.get("target_quality_tier") or "") in {"paper_strong", "paper_pseudo"}
            and str(record.get("target_is_prior_copy")).lower() not in {"true", "1", "yes"}
        )
        eligible += int(is_eligible)
        prior_copy += int(str(record.get("target_is_prior_copy")).lower() in {"true", "1", "yes"})
    summary.setdefault("paper_stage_eligible_records_estimate", eligible)
    summary.setdefault("target_prior_copy_records_estimate", prior_copy)
    return {
        "manifest": str(path),
        "exists": True,
        "records": len(records),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "summary": summary,
    }


def quality_marker(output_root: Path, manifest: Path, kind: str) -> Path:
    safe_name = manifest.parent.name
    return output_root / "markers" / f"{kind}_{safe_name}.json"


def marker_is_recent(marker: Path, every_seconds: int) -> bool:
    if not marker.exists():
        return False
    age = time.time() - marker.stat().st_mtime
    return age < every_seconds


def run_quality_audit(
    manifest: Path,
    *,
    config: dict[str, Any],
    output_root: Path,
    python_bin: Path,
    dry_run: bool,
) -> dict[str, Any]:
    quality = config.get("quality", {})
    every_seconds = int(quality.get("audit_every_seconds", 900))
    marker = quality_marker(output_root, manifest, "audit")
    if marker_is_recent(marker, every_seconds):
        return {"manifest": str(manifest), "action": "recently_audited", "marker": str(marker)}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_root / "quality" / f"{timestamp}_{manifest.parent.name}"
    command = [
        str(python_bin),
        "scripts/evaluate_material_refine_dataset_quality.py",
        "--manifest",
        str(manifest),
        "--output-dir",
        str(out_dir),
        "--min-paper-eligible",
        str(quality.get("min_paper_eligible", 128)),
        "--min-view-ready-rate",
        str(quality.get("min_view_ready_rate", 0.8)),
        "--min-strict-complete-rate",
        str(quality.get("min_strict_complete_rate", 0.8)),
    ]
    if dry_run:
        return {"manifest": str(manifest), "action": "dry_run_audit", "command": command}
    result = run_capture(command, timeout=900)
    write_json(
        marker,
        {
            "updated_at_utc": utc_now(),
            "manifest": str(manifest),
            "output_dir": str(out_dir),
            "returncode": result.returncode,
        },
    )
    return {
        "manifest": str(manifest),
        "action": "audited" if result.returncode == 0 else "audit_failed",
        "output_dir": str(out_dir),
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-30:],
    }


def run_buffer_validation(
    manifest: Path,
    *,
    config: dict[str, Any],
    output_root: Path,
    python_bin: Path,
    dry_run: bool,
) -> dict[str, Any]:
    quality = config.get("quality", {})
    every_seconds = int(quality.get("buffer_validation_every_seconds", 1800))
    marker = quality_marker(output_root, manifest, "buffers")
    if marker_is_recent(marker, every_seconds):
        return {"manifest": str(manifest), "action": "recently_validated", "marker": str(marker)}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_root / "buffer_validation" / f"{timestamp}_{manifest.parent.name}"
    command = [
        str(python_bin),
        "scripts/validate_material_refine_buffers.py",
        "--manifest",
        str(manifest),
        "--output-dir",
        str(out_dir),
    ]
    if dry_run:
        return {"manifest": str(manifest), "action": "dry_run_validate_buffers", "command": command}
    result = run_capture(command, timeout=900)
    write_json(
        marker,
        {
            "updated_at_utc": utc_now(),
            "manifest": str(manifest),
            "output_dir": str(out_dir),
            "returncode": result.returncode,
        },
    )
    return {
        "manifest": str(manifest),
        "action": "validated" if result.returncode == 0 else "validation_failed",
        "output_dir": str(out_dir),
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-30:],
    }


def ensure_factory(config: dict[str, Any], *, sessions: set[str], python_bin: Path, dry_run: bool) -> dict[str, Any]:
    factory = config.get("factory", {})
    if not bool(factory.get("enabled", True)):
        return {"action": "disabled"}
    session = str(factory["session"])
    if session_exists(session, sessions):
        return {"session": session, "action": "already_running"}
    run_script = repo_path(factory["run_script"])
    log_path = repo_path(factory["log"])
    state_json = repo_path(factory["state_json"])
    command_items: list[str | Path] = [
        python_bin,
        "scripts/run_material_refine_dataset_factory.py",
        "--config",
        repo_path(factory["config"]),
        *[str(flag) for flag in factory.get("loop_flags", [])],
        "--state-json",
        state_json,
    ]
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shell_quote(REPO_ROOT)}",
            f"{shell_join(command_items)} > {shell_quote(log_path)} 2>&1",
            "",
        ]
    )
    if not dry_run:
        run_script.parent.mkdir(parents=True, exist_ok=True)
        run_script.write_text(script, encoding="utf-8")
        run_script.chmod(0o755)
    return start_tmux_session(session, shell_quote(run_script), dry_run=dry_run)


def read_factory_config(config: dict[str, Any]) -> dict[str, Any]:
    factory = config.get("factory", {})
    path = repo_path(factory.get("config", ""))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def ps_rows() -> list[dict[str, Any]]:
    result = run_capture(["ps", "-eo", "pid,ppid,stat,etimes,command"])
    if result.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "stat": parts[2],
                "etimes": int(parts[3]),
                "command": parts[4],
            }
        )
    return rows


def source_process_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    output_root = str(source.get("output_root") or "")
    command_items = [str(item) for item in source.get("command", [])]
    script_token = next((item for item in command_items if item.startswith("scripts/")), "")
    if not output_root or not script_token:
        return []
    return [
        row
        for row in ps_rows()
        if output_root in str(row.get("command") or "") and script_token in str(row.get("command") or "")
    ]


def terminate_source_processes(source: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    rows = source_process_rows(source)
    pids = sorted({int(row["pid"]) for row in rows})
    if not pids:
        return {"action": "no_processes"}
    if dry_run:
        return {"action": "dry_run_terminate_processes", "pids": pids}
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)
    killed: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            killed.append(pid)
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            killed.append(pid)
    return {"action": "terminated_processes", "pids": pids, "gone_after_term": killed}


def newest_progress_mtime(paths: list[Path]) -> float:
    newest = 0.0
    for path in paths:
        if not path.exists():
            continue
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
        if path.is_dir():
            try:
                for child in path.rglob("*"):
                    if child.is_file():
                        newest = max(newest, child.stat().st_mtime)
            except OSError:
                continue
    return newest


def restart_download_session(source: dict[str, Any], *, reason: str, sessions: set[str], dry_run: bool) -> dict[str, Any]:
    session = str(source.get("session") or "")
    output_root = repo_path(source.get("output_root") or "")
    run_script = output_root / "logs" / f"{session}.run.sh"
    log_path = output_root / "logs" / f"{session}.log"
    actions: list[dict[str, Any]] = []
    if session and session in sessions:
        actions.append(kill_tmux_session(session, dry_run=dry_run))
    actions.append(terminate_source_processes(source, dry_run=dry_run))
    if not run_script.exists():
        return {
            "session": session,
            "action": "restart_deferred_missing_run_script",
            "reason": reason,
            "run_script": str(run_script),
            "actions": actions,
        }
    command = f"bash {shell_quote(run_script)} > {shell_quote(log_path)} 2>&1"
    actions.append(start_tmux_session(session, command, dry_run=dry_run))
    return {"session": session, "action": "restarted", "reason": reason, "actions": actions}


def ensure_download_health(config: dict[str, Any], *, sessions: set[str], dry_run: bool) -> list[dict[str, Any]]:
    factory_config = read_factory_config(config)
    if not factory_config:
        return [{"action": "factory_config_unavailable"}]
    actions: list[dict[str, Any]] = []
    now = time.time()
    for source in factory_config.get("downloads", []):
        if not bool(source.get("enabled", True)):
            continue
        session = str(source.get("session") or "")
        output_root = repo_path(source.get("output_root") or "")
        log_path = output_root / "logs" / f"{session}.log"
        process_rows = source_process_rows(source)
        stopped = [row for row in process_rows if "T" in str(row.get("stat") or "")]
        if stopped:
            actions.append(
                restart_download_session(
                    source,
                    reason=f"stopped_processes:{[row['pid'] for row in stopped]}",
                    sessions=sessions,
                    dry_run=dry_run,
                )
            )
            continue
        if session and session not in sessions:
            output_manifest = output_root / "objaverse_cached_increment_manifest.json"
            polyhaven_manifest = output_root / "polyhaven_material_bank_manifest.json"
            if output_manifest.exists() or polyhaven_manifest.exists() or log_path.exists():
                actions.append(
                    restart_download_session(
                        source,
                        reason="missing_tmux_session",
                        sessions=sessions,
                        dry_run=dry_run,
                    )
                )
            else:
                actions.append({"session": session, "action": "waiting_for_factory_run_script"})
            continue
        stall_seconds = int(source.get("stall_seconds", 0) or 0)
        if stall_seconds <= 0 or not session or session not in sessions:
            actions.append({"session": session, "action": "healthy_or_unwatched"})
            continue
        progress_paths = [repo_path(path) for path in source.get("progress_paths", [])]
        newest = max(
            newest_progress_mtime(progress_paths),
            log_path.stat().st_mtime if log_path.exists() else 0.0,
        )
        if newest > 0 and now - newest > stall_seconds:
            actions.append(
                restart_download_session(
                    source,
                    reason=f"stalled_no_progress_seconds={int(now - newest)}",
                    sessions=sessions,
                    dry_run=dry_run,
                )
            )
        else:
            actions.append(
                {
                    "session": session,
                    "action": "healthy",
                    "seconds_since_progress": max(0, int(now - newest)) if newest else None,
                }
            )
    return actions


def active_sessions_with_prefix(prefix: str, sessions: set[str]) -> list[str]:
    return sorted(session for session in sessions if session.startswith(prefix))


def current_paper_eligible(root: Path, merge_output_name: str) -> int:
    candidates = [
        root / merge_output_name,
        root / "canonical_manifest_monitor_merged.json",
    ]
    best = 0
    for candidate in candidates:
        if not candidate.exists():
            continue
        summary = summarize_manifest(candidate)
        if not summary.get("exists"):
            continue
        details = summary.get("summary") or {}
        best = max(
            best,
            int(
                details.get("paper_stage_eligible_records")
                or details.get("paper_stage_eligible_records_estimate")
                or 0
            ),
        )
    return best


def launch_paper_unlock(
    config: dict[str, Any],
    *,
    sessions: set[str],
    python_bin: Path,
    dry_run: bool,
) -> dict[str, Any]:
    paper = config.get("paper_unlock", {})
    env = os.environ.copy()
    launch_env = {str(key): str(value) for key, value in paper.get("launch_env", {}).items()}
    output_root = REPO_ROOT / "output" / "material_refine_dataset_factory" / (
        "paper_unlock_gpu0_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    launch_env["OUTPUT_ROOT"] = str(output_root.relative_to(REPO_ROOT))
    launch_env["PYTHON_BIN"] = str(python_bin)
    env.update(launch_env)
    prefix = str(launch_env.get("SESSION_PREFIX", "sf3d_paper_unlock_material_refine"))
    monitor_session = f"{prefix}_merged_monitor"
    actions: list[dict[str, Any]] = []
    if monitor_session in sessions and not active_sessions_with_prefix(f"{prefix}_shard", sessions):
        actions.append(kill_tmux_session(monitor_session, dry_run=dry_run))
    command = ["bash", "scripts/launch_material_refine_longrun_dataset_tmux.sh"]
    if dry_run:
        return {"action": "dry_run_launch", "output_root": str(output_root), "env": launch_env, "pre_actions": actions}
    result = run_capture(command, env=env, timeout=300)
    return {
        "action": "launched" if result.returncode == 0 else "launch_failed",
        "output_root": str(output_root),
        "returncode": result.returncode,
        "pre_actions": actions,
        "stdout_tail": result.stdout.splitlines()[-40:],
    }


def ensure_paper_unlock(
    config: dict[str, Any],
    *,
    sessions: set[str],
    python_bin: Path,
    dry_run: bool,
) -> dict[str, Any]:
    paper = config.get("paper_unlock", {})
    if not bool(paper.get("enabled", True)):
        return {"action": "disabled"}
    latest_root = discover_latest_root(str(paper["root_pattern"]))
    prefixes = [str(item) for item in paper.get("required_prefixes", [])]
    actions: list[dict[str, Any]] = []
    if latest_root is not None:
        actions.extend(
            restart_missing_root_sessions(
                latest_root,
                prefixes=prefixes,
                sessions=sessions,
                dry_run=dry_run,
            )
        )
        merge = merge_root_manifest(
            latest_root,
            output_name=str(paper.get("merge_output_name", "canonical_manifest_supervisor_merged.json")),
            python_bin=python_bin,
            dry_run=dry_run,
        )
        actions.append({"merge": merge})
        eligible = current_paper_eligible(latest_root, str(paper.get("merge_output_name")))
        active_shards = active_sessions_with_prefix("sf3d_paper_unlock_material_refine_shard", set(tmux_sessions()))
        if eligible >= int(paper.get("target_eligible_records", 128)):
            return {
                "action": "paper_unlock_ready",
                "root": str(latest_root),
                "eligible_records": eligible,
                "actions": actions,
            }
        if active_shards:
            return {
                "action": "paper_unlock_running",
                "root": str(latest_root),
                "eligible_records": eligible,
                "active_shards": active_shards,
                "actions": actions,
            }
    actions.append({"launch": launch_paper_unlock(config, sessions=set(tmux_sessions()), python_bin=python_bin, dry_run=dry_run)})
    return {"action": "paper_unlock_launched_or_relaunched", "latest_root": str(latest_root) if latest_root else None, "actions": actions}


def scan_recent_errors(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    scan = config.get("log_scan", {})
    if not bool(scan.get("enabled", True)):
        return {"action": "disabled"}
    patterns = [str(item) for item in scan.get("patterns", [])]
    if not patterns:
        return {"action": "no_patterns"}
    regex = re.compile("|".join(re.escape(pattern) for pattern in patterns))
    tail_bytes = int(scan.get("tail_bytes_per_log", 2 * 1024 * 1024))
    max_lines = int(scan.get("max_lines", 120))
    scan_from_end_on_first_seen = bool(scan.get("scan_from_end_on_first_seen", True))
    offsets_path = output_root / "log_scan_offsets.json"
    try:
        offsets = json.loads(offsets_path.read_text(encoding="utf-8")) if offsets_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        offsets = {}
    if not isinstance(offsets, dict):
        offsets = {}
    ignore_names = {
        "sf3d_material_refine_dataset_7day_supervisor.log",
        *[str(item) for item in scan.get("ignore_names", [])],
    }
    ignore_path_contains = [str(item) for item in scan.get("ignore_path_contains", [])]
    log_roots: list[Path] = []
    for production in production_root_groups(config):
        log_roots.extend(
            root / "logs"
            for root in discover_latest_roots(
                [str(item) for item in production.get("patterns", [])],
                latest_only=bool(production.get("latest_only", True)),
            )
        )
    paper = config.get("paper_unlock", {})
    paper_root = discover_latest_root(str(paper.get("root_pattern", ""))) if paper else None
    if paper_root is not None:
        log_roots.append(paper_root / "logs")
    log_roots.append(output_root)
    factory = config.get("factory", {})
    if factory.get("log"):
        log_roots.append(repo_path(factory["log"]))
    if factory.get("config"):
        factory_config_path = repo_path(factory["config"])
        if factory_config_path.exists():
            try:
                factory_config = json.loads(factory_config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                factory_config = {}
            for item in factory_config.get("downloads", []):
                output_dir = repo_path(item.get("output_root", ""))
                session = str(item.get("session") or "")
                if session:
                    log_roots.append(output_dir / "logs" / f"{session}.log")

    log_paths: list[Path] = []
    seen: set[str] = set()
    for root in log_roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".log":
            candidates = [root]
        else:
            candidates = sorted(root.rglob("*.log"))
        for path in candidates:
            if path.name in ignore_names:
                continue
            if any(token and token in str(path) for token in ignore_path_contains):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            log_paths.append(path)

    matches: list[str] = []
    new_offsets: dict[str, int] = {}
    for path in log_paths:
        key = str(path.resolve())
        try:
            file_size = path.stat().st_size
        except OSError:
            continue
        previous_offset = offsets.get(key)
        if previous_offset is None and scan_from_end_on_first_seen:
            new_offsets[key] = file_size
            continue
        if isinstance(previous_offset, int):
            start_offset = max(0, min(previous_offset, file_size))
        elif tail_bytes > 0:
            start_offset = max(0, file_size - tail_bytes)
        else:
            start_offset = 0
        try:
            with path.open("rb") as handle:
                handle.seek(start_offset, os.SEEK_SET)
                raw = handle.read()
        except OSError:
            continue
        new_offsets[key] = file_size
        text = raw.decode("utf-8", errors="replace")
        for offset, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{path}:{start_offset}+{offset}:{line}")
    lines = matches[-max_lines:]
    payload = {"action": "scanned", "count": len(lines), "lines": lines}
    write_json(offsets_path, new_offsets)
    write_json(output_root / "recent_errors.json", payload)
    return payload


def status_markdown(status: dict[str, Any]) -> str:
    gpu_lines = []
    for row in status.get("gpu", []):
        gpu_lines.append(
            f"- GPU{row['index']}: {row['memory_used_mb']}/{row['memory_total_mb']} MB, util {row['utilization_gpu_percent']}%"
        )
    root_lines = []
    for item in status.get("roots", []):
        summary = item.get("manifest_summary") or {}
        details = summary.get("summary") or {}
        root_lines.append(
            "- "
            + f"`{item.get('root')}` records={summary.get('records', 0)} "
            + f"eligible={details.get('paper_stage_eligible_records') or details.get('paper_stage_eligible_records_estimate') or 0} "
            + f"tiers={details.get('target_quality_tier', {})}"
        )
    actions = status.get("actions", {})
    lines = [
        "# SF3D Material Refine 7-Day Dataset Supervisor",
        "",
        f"- updated_at_utc: `{status.get('updated_at_utc')}`",
        f"- deadline_utc: `{status.get('deadline_utc')}`",
        f"- dry_run: `{status.get('dry_run')}`",
        f"- tmux_sessions: `{len(status.get('tmux_sessions', []))}`",
        "",
        "## GPU",
        "",
        *(gpu_lines or ["- unavailable"]),
        "",
        "## Data Roots",
        "",
        *(root_lines or ["- no roots discovered yet"]),
        "",
        "## Key Actions",
        "",
        f"- factory: `{actions.get('factory', {}).get('action')}`",
        f"- paper_unlock: `{actions.get('paper_unlock', {}).get('action')}`",
        f"- recent_errors: `{status.get('recent_errors', {}).get('count', 0)}`",
        "",
    ]
    return "\n".join(lines)


def supervise_once(
    *,
    config: dict[str, Any],
    start_time: datetime,
    deadline: datetime,
    dry_run: bool,
) -> dict[str, Any]:
    output_root = repo_path(config.get("output_root", "output/material_refine_dataset_factory/supervisor_7day"))
    python_bin = Path(str(config.get("python_bin", "/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python")))
    sessions = set(tmux_sessions())
    actions: dict[str, Any] = {}
    actions["factory"] = ensure_factory(config, sessions=sessions, python_bin=python_bin, dry_run=dry_run)
    sessions = set(tmux_sessions())
    actions["downloads"] = ensure_download_health(config, sessions=sessions, dry_run=dry_run)
    sessions = set(tmux_sessions())
    actions["disabled_groups"] = cleanup_disabled_production_groups(config, sessions=sessions, dry_run=dry_run)
    sessions = set(tmux_sessions())

    roots: list[dict[str, Any]] = []
    for production in production_root_groups(config):
        group_name = str(production.get("name", "production_longrun"))
        for root in discover_latest_roots(
            [str(item) for item in production.get("patterns", [])],
            latest_only=bool(production.get("latest_only", True)),
        ):
            root_actions = restart_missing_root_sessions(
                root,
                prefixes=[str(item) for item in production.get("required_prefixes", [])],
                sessions=sessions,
                dry_run=dry_run,
            )
            merge = merge_root_manifest(
                root,
                output_name=str(production.get("merge_output_name", "canonical_manifest_supervisor_merged.json")),
                python_bin=python_bin,
                dry_run=dry_run,
            )
            manifest = root / str(production.get("merge_output_name", "canonical_manifest_supervisor_merged.json"))
            roots.append(
                {
                    "root": str(root),
                    "type": group_name,
                    "actions": root_actions,
                    "merge": merge,
                    "manifest_summary": summarize_manifest(manifest),
                }
            )

    actions["paper_unlock"] = ensure_paper_unlock(config, sessions=set(tmux_sessions()), python_bin=python_bin, dry_run=dry_run)
    paper = config.get("paper_unlock", {})
    latest_paper = discover_latest_root(str(paper.get("root_pattern", ""))) if paper else None
    if latest_paper is not None:
        paper_manifest = latest_paper / str(paper.get("merge_output_name", "canonical_manifest_supervisor_merged.json"))
        roots.append(
            {
                "root": str(latest_paper),
                "type": "paper_unlock",
                "manifest_summary": summarize_manifest(paper_manifest),
            }
        )

    quality_actions: list[dict[str, Any]] = []
    if bool(config.get("quality", {}).get("enabled", True)):
        min_records = int(config.get("quality", {}).get("min_manifest_records_for_audit", 8))
        for root_item in roots:
            summary = root_item.get("manifest_summary") or {}
            manifest_value = summary.get("manifest")
            if not manifest_value or int(summary.get("records") or 0) < min_records:
                continue
            manifest = Path(str(manifest_value))
            quality_actions.append(
                {
                    "audit": run_quality_audit(
                        manifest,
                        config=config,
                        output_root=output_root,
                        python_bin=python_bin,
                        dry_run=dry_run,
                    )
                }
            )
            quality_actions.append(
                {
                    "buffer_validation": run_buffer_validation(
                        manifest,
                        config=config,
                        output_root=output_root,
                        python_bin=python_bin,
                        dry_run=dry_run,
                    )
                }
            )
    actions["quality"] = quality_actions
    recent_errors = scan_recent_errors(config, output_root)
    gpu_snapshot = gpu_rows()

    status = {
        "updated_at_utc": utc_now(),
        "started_at_utc": start_time.isoformat().replace("+00:00", "Z"),
        "deadline_utc": deadline.isoformat().replace("+00:00", "Z"),
        "seconds_remaining": max(0, int((deadline - utc_now_dt()).total_seconds())),
        "dry_run": dry_run,
        "config": str((REPO_ROOT / str(DEFAULT_CONFIG)).resolve()),
        "gpu": gpu_snapshot,
        "gpu1_data_processes": data_processes_on_gpu1(),
        "gpu1_data_process_offenders": gpu1_data_policy_offenders(config, gpu_snapshot)
        if bool(config.get("gpu_policy", {}).get("warn_if_data_process_on_gpu1", True))
        else [],
        "tmux_sessions": tmux_sessions(),
        "download_sessions": {
            session: session in set(tmux_sessions())
            for session in config.get("downloads", {}).get("required_sessions", [])
        },
        "roots": roots,
        "actions": actions,
        "recent_errors": recent_errors,
    }
    write_json(output_root / "status.json", status)
    append_jsonl(output_root / "status_history.jsonl", status)
    (output_root / "status.md").write_text(status_markdown(status) + "\n", encoding="utf-8")
    return status


def main() -> None:
    args = parse_args()
    start_time = utc_now_dt()
    while True:
        config = read_json(args.config)
        deadline = start_time + timedelta(days=float(config.get("duration_days", 7)))
        poll_seconds = int(config.get("poll_seconds", 300))
        status = supervise_once(config=config, start_time=start_time, deadline=deadline, dry_run=args.dry_run)
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
        if args.once or utc_now_dt() >= deadline:
            break
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
