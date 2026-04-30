from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Wait for a current material-refine run to finish and then launch the next training on a target GPU.",
    )
    parser.add_argument("--current-run-dir", type=Path, required=True)
    parser.add_argument("--next-config", type=Path, default=None)
    parser.add_argument("--next-launcher", type=Path, default=None)
    parser.add_argument("--next-output-dir", type=Path, default=None)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--launch-mode", choices=["tmux", "subprocess"], default="tmux")
    parser.add_argument("--tmux-session", type=str, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--launch-log", type=Path, default=None)
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def current_training_complete(run_dir: Path) -> tuple[bool, dict[str, Any]]:
    train_state = load_json(run_dir / "train_state.json") or {}
    train_args = load_json(run_dir / "train_args.json") or {}
    epochs = int(train_args.get("epochs", 0) or 0)
    last_epoch = int(train_state.get("last_epoch", 0) or 0)
    best_exists = (run_dir / "best.pt").exists()
    latest_exists = (run_dir / "latest.pt").exists()
    final_log_marker = False
    train_log = run_dir / "train.log"
    if train_log.exists():
        try:
            final_log_marker = "[final] output_dir=" in train_log.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            final_log_marker = False
    completed = False
    if epochs > 0:
        completed = last_epoch >= epochs and (best_exists or latest_exists)
    if not completed and final_log_marker and (best_exists or latest_exists):
        completed = True
    detail = {
        "epochs": epochs,
        "last_epoch": last_epoch,
        "best_exists": best_exists,
        "latest_exists": latest_exists,
        "final_log_marker": final_log_marker,
        "completed": completed,
    }
    return completed, detail


def gpu_index_to_uuid() -> dict[int, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
        ],
        text=True,
    )
    mapping: dict[int, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        mapping[int(parts[0])] = parts[1]
    return mapping


def gpu_compute_processes(gpu_index: int) -> list[dict[str, Any]]:
    uuid_map = gpu_index_to_uuid()
    gpu_uuid = uuid_map.get(gpu_index)
    if gpu_uuid is None:
        raise RuntimeError(f"gpu_index_not_found:{gpu_index}")
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    processes: list[dict[str, Any]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        if parts[0] != gpu_uuid:
            continue
        try:
            used_memory_mib = int(parts[3].split()[0])
        except Exception:
            used_memory_mib = -1
        processes.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_memory_mib": used_memory_mib,
            }
        )
    return processes


def default_next_output_dir(repo_root: Path, next_config: Path) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return repo_root / "output" / f"{next_config.stem}_{timestamp}"


def resolve_launch_spec(args: argparse.Namespace) -> tuple[str, Path | None]:
    repo_root = args.repo_root.resolve()
    if args.next_launcher is not None:
        launcher = args.next_launcher.resolve()
        command = f"cd {shlex.quote(str(repo_root))} && bash {shlex.quote(str(launcher))}"
        return command, None
    if args.next_config is None:
        raise ValueError("either --next-config or --next-launcher is required")
    next_config = args.next_config.resolve()
    next_output_dir = (
        args.next_output_dir.resolve()
        if args.next_output_dir is not None
        else default_next_output_dir(repo_root, next_config)
    )
    command = (
        f"cd {shlex.quote(str(repo_root))} && "
        f"python scripts/train_material_refiner.py "
        f"--config {shlex.quote(str(next_config))} "
        f"--output-dir {shlex.quote(str(next_output_dir))}"
    )
    return command, next_output_dir


def launch_next_training(
    *,
    command: str,
    launch_mode: str,
    tmux_session: str | None,
    launch_log: Path,
) -> dict[str, Any]:
    launch_log.parent.mkdir(parents=True, exist_ok=True)
    wrapped_command = f"{command} 2>&1 | tee {shlex.quote(str(launch_log))}"
    if launch_mode == "tmux":
        session_name = tmux_session or f"mr_chain_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, wrapped_command],
            check=True,
        )
        return {"launch_mode": "tmux", "tmux_session": session_name, "launch_log": str(launch_log.resolve())}
    process = subprocess.Popen(["bash", "-lc", wrapped_command], cwd=str(REPO_ROOT))
    return {"launch_mode": "subprocess", "pid": int(process.pid), "launch_log": str(launch_log.resolve())}


def main() -> None:
    args = parse_args()
    state_file = args.state_file or (args.current_run_dir / "chain_next_training_state.json")
    launch_log = args.launch_log or (args.current_run_dir / "chain_next_training_launch.log")
    command, resolved_output_dir = resolve_launch_spec(args)

    while True:
        completed, training_detail = current_training_complete(args.current_run_dir)
        state_payload = {
            "stage": "waiting_for_current_training",
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "current_run_dir": str(args.current_run_dir.resolve()),
            "gpu_index": int(args.gpu_index),
            "next_config": str(args.next_config.resolve()) if args.next_config is not None else None,
            "next_launcher": str(args.next_launcher.resolve()) if args.next_launcher is not None else None,
            "resolved_next_output_dir": str(resolved_output_dir.resolve()) if resolved_output_dir is not None else None,
            "training_detail": training_detail,
        }
        save_json(state_file, state_payload)
        if completed:
            break
        time.sleep(max(int(args.poll_seconds), 10))

    while True:
        gpu_processes = gpu_compute_processes(args.gpu_index)
        state_payload = {
            "stage": "waiting_for_gpu_free",
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "current_run_dir": str(args.current_run_dir.resolve()),
            "gpu_index": int(args.gpu_index),
            "gpu_processes": gpu_processes,
            "resolved_next_output_dir": str(resolved_output_dir.resolve()) if resolved_output_dir is not None else None,
        }
        save_json(state_file, state_payload)
        if not gpu_processes:
            break
        time.sleep(max(int(args.poll_seconds), 10))

    launch_info = launch_next_training(
        command=command,
        launch_mode=str(args.launch_mode),
        tmux_session=args.tmux_session,
        launch_log=launch_log,
    )
    save_json(
        state_file,
        {
            "stage": "launched",
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "current_run_dir": str(args.current_run_dir.resolve()),
            "gpu_index": int(args.gpu_index),
            "command": command,
            "resolved_next_output_dir": str(resolved_output_dir.resolve()) if resolved_output_dir is not None else None,
            "launch_info": launch_info,
        },
    )
    print(json.dumps(load_json(state_file), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
