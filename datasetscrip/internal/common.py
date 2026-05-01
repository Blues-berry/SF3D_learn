from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def records(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def repo_root(config: dict[str, Any]) -> Path:
    return Path(str(config["repo_root"])).resolve()


def repo_path(config: dict[str, Any], key: str) -> Path:
    value = Path(str(config[key]))
    return value if value.is_absolute() else repo_root(config) / value


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    dry_run: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str] | None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return None
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result

