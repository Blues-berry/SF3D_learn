#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping config: {path}")
    return payload


def repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def require_file(path: Path, label: str, *, allow_missing: bool = False) -> str | None:
    if path.exists():
        return None
    message = f"missing {label}: {path}"
    if allow_missing:
        return f"warning:{message}"
    raise FileNotFoundError(message)


def load_flat_config(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        require_file(path, "config")
        for key, value in load_yaml(path).items():
            if not isinstance(value, dict):
                merged[key] = value
    return merged


def check_imports(module_names: list[str]) -> list[str]:
    missing = []
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - preflight should report import cause.
            missing.append(f"{name}: {type(exc).__name__}: {exc}")
    return missing


def gpu_summary(python_bin: Path) -> dict[str, Any]:
    code = """
import json
import torch
payload = {"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(), "gpus": []}
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        payload["gpus"].append({
            "index": idx,
            "name": props.name,
            "total_gb": round(props.total_memory / 1024 ** 3, 2),
            "capability": f"{props.major}.{props.minor}",
        })
print(json.dumps(payload, ensure_ascii=False))
"""
    try:
        out = subprocess.check_output([str(python_bin), "-c", code], text=True)
        return json.loads(out)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def preflight(spec_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    python_bin = repo_path(spec.get("python_bin")) or Path(sys.executable)
    run_root = repo_path(spec.get("run_root") or "output/material_refine_experiment")
    train_cfgs = [repo_path(path) for path in spec.get("train", {}).get("configs", [])]
    eval_cfgs = [repo_path(path) for path in spec.get("eval", {}).get("configs", [])]
    train_cfgs = [path for path in train_cfgs if path is not None]
    eval_cfgs = [path for path in eval_cfgs if path is not None]

    warnings: list[str] = []
    require_file(python_bin, "python_bin")
    require_file(spec_path, "experiment spec")
    for path in train_cfgs + eval_cfgs:
        require_file(path, "config")

    train_flat = load_flat_config(train_cfgs) if train_cfgs else {}
    eval_flats = [load_flat_config([path]) for path in eval_cfgs]
    if spec.get("preflight", {}).get("require_train_manifest", True):
        manifest = repo_path(train_flat.get("train_manifest") or train_flat.get("manifest"))
        if manifest:
            require_file(manifest, "train manifest")
    if spec.get("preflight", {}).get("require_val_manifest", True):
        manifest = repo_path(train_flat.get("val_manifest"))
        if manifest:
            require_file(manifest, "val manifest")
    if spec.get("preflight", {}).get("require_eval_manifests", True):
        for cfg_path, flat in zip(eval_cfgs, eval_flats):
            manifest = repo_path(flat.get("manifest"))
            if manifest:
                require_file(manifest, f"eval manifest for {cfg_path.name}")
            checkpoint = repo_path(flat.get("checkpoint"))
            if checkpoint and not checkpoint.exists():
                if run_root and checkpoint == run_root / "best.pt":
                    warnings.append(f"checkpoint will be produced by training: {checkpoint}")
                else:
                    require_file(checkpoint, f"eval checkpoint for {cfg_path.name}")

    missing_imports = check_imports(spec.get("preflight", {}).get("required_imports", []))
    if missing_imports:
        raise RuntimeError("missing required imports: " + "; ".join(missing_imports))
    gpus = gpu_summary(python_bin)
    if spec.get("preflight", {}).get("require_cuda", True) and not gpus.get("cuda_available", False):
        raise RuntimeError(f"CUDA unavailable according to {python_bin}")

    return {
        "spec": str(spec_path),
        "python_bin": str(python_bin),
        "run_root": str(run_root),
        "train_configs": [str(path) for path in train_cfgs],
        "eval_configs": [str(path) for path in eval_cfgs],
        "gpu_summary": gpus,
        "warnings": warnings,
    }


def shell_join(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def run_with_tee(cmd: list[str | Path], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[runner] command={shell_join(cmd)}")
    print(f"[runner] log={log_path}")
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n[runner] start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        log_f.write(f"[runner] command={shell_join(cmd)}\n")
        proc = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        rc = proc.wait()
        log_f.write(f"[runner] exit_code={rc}\n")
        return rc


def run_foreground(spec_path: Path, spec: dict[str, Any], dry_run: bool) -> int:
    python_bin = repo_path(spec.get("python_bin")) or Path(sys.executable)
    run_root = repo_path(spec.get("run_root") or "output/material_refine_experiment")
    assert run_root is not None
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report = preflight(spec_path, spec)
    (run_root / "experiment_preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if dry_run:
        return 0

    env = os.environ.copy()
    for key, value in (spec.get("env") or {}).items():
        env[str(key)] = str(value)

    if spec.get("train", {}).get("enabled", True):
        cmd: list[str | Path] = [python_bin, "scripts/train_material_refiner.py"]
        for cfg in spec.get("train", {}).get("configs", []):
            cmd.extend(["--config", cfg])
        rc = run_with_tee(cmd, log_dir / spec.get("train", {}).get("log_name", "train.log"), env)
        if rc != 0:
            return rc

    if spec.get("eval", {}).get("enabled", True) and spec.get("eval", {}).get("run_after_train", True):
        for cfg in spec.get("eval", {}).get("configs", []):
            cfg_path = Path(cfg)
            cmd = [python_bin, "scripts/eval_material_refiner.py", "--config", cfg]
            rc = run_with_tee(cmd, log_dir / f"{cfg_path.stem}.log", env)
            if rc != 0:
                return rc
    return 0


def launch_tmux(spec_path: Path, spec: dict[str, Any], dry_run: bool) -> int:
    session = str(spec.get("tmux_session") or spec.get("experiment_name") or "material_refine_experiment")
    cmd = [
        sys.executable,
        str(Path(__file__).relative_to(REPO_ROOT)),
        "--spec",
        str(spec_path.relative_to(REPO_ROOT) if spec_path.is_relative_to(REPO_ROOT) else spec_path),
        "--foreground",
    ]
    if dry_run:
        cmd.append("--dry-run")
    tmux_cmd = ["tmux", "new-session", "-d", "-s", session, f"cd {shlex.quote(str(REPO_ROOT))} && {shell_join(cmd)}"]
    print(f"[runner] launch_tmux={shell_join(tmux_cmd)}")
    return subprocess.call(tmux_cmd, cwd=REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project-level launcher for material refine experiments.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--foreground", action="store_true", help="Run the experiment in the current shell.")
    parser.add_argument("--launch-tmux", action="store_true", help="Launch the foreground runner in a tmux session.")
    parser.add_argument("--dry-run", action="store_true", help="Only run preflight and print commands.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = repo_path(args.spec)
    if spec_path is None:
        raise ValueError("missing spec")
    spec = load_yaml(spec_path)
    if args.launch_tmux:
        return launch_tmux(spec_path, spec, args.dry_run)
    return run_foreground(spec_path, spec, args.dry_run or not args.foreground)


if __name__ == "__main__":
    raise SystemExit(main())
