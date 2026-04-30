#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--b-root", type=Path, required=True)
    parser.add_argument("--batch-name", type=str, required=True)
    parser.add_argument("--session-prefix", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_slug = args.batch_name
    rebake_root = args.b_root / "B_track" / batch_slug
    preflight = read_json(rebake_root / "B_track_preflight.json", {})
    if not preflight:
        raise SystemExit(f"missing_preflight:{rebake_root / 'B_track_preflight.json'}")
    records = int(preflight.get("records") or 0)
    if records <= 0:
        raise SystemExit("empty_preflight_records")

    command_draft = Path(str((read_json(rebake_root / f"{batch_slug}_decision.json", {}) or {}).get("command_draft") or rebake_root / f"run_{batch_slug}_gpu0.sh"))
    input_manifest = rebake_root / f"{batch_slug}_input_manifest.json"
    partial_manifest = rebake_root / f"{batch_slug}_partial_manifest.json"
    output_manifest = rebake_root / f"{batch_slug}_manifest.json"
    prepared_root = rebake_root / "prepared"
    finalize_log = rebake_root / f"{batch_slug}_finalize.log"
    finalize_status_json = rebake_root / f"{batch_slug}_finalize_status.json"
    finalize_status_md = rebake_root / f"{batch_slug}_finalize_status.md"
    finalize_script = rebake_root / f"run_finalize_after_{batch_slug}.sh"

    if not command_draft.exists():
        raise SystemExit(f"missing_command_draft:{command_draft}")
    if not input_manifest.exists():
        raise SystemExit(f"missing_input_manifest:{input_manifest}")

    prefix = args.session_prefix or f"trainv5_{batch_slug}"
    full_session = f"{prefix}_full_rebake"
    truth_session = f"{prefix}_truth_monitor"
    finalize_session = f"{prefix}_finalize_watcher"

    finalize_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {REPO_ROOT}",
                f'REBAKE_SESSION="{full_session}"',
                f'FINAL_MANIFEST="{output_manifest}"',
                f'STATUS_JSON="{finalize_status_json}"',
                f'STATUS_MD="{finalize_status_md}"',
                f'LOG_PATH="{finalize_log}"',
                'SLEEP_SECONDS="${TRAINV5_B_FINALIZE_WATCH_SECONDS:-300}"',
                "",
                "write_status() {",
                '  local status="$1"',
                '  local reason="$2"',
                '  python - "$STATUS_JSON" "$STATUS_MD" "$status" "$reason" <<'"'"'PY'"'"'',
                "from __future__ import annotations",
                "import json, sys",
                "from datetime import datetime, timezone",
                "from pathlib import Path",
                "json_path = Path(sys.argv[1])",
                "md_path = Path(sys.argv[2])",
                "status = sys.argv[3]",
                "reason = sys.argv[4]",
                'now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")',
                "payload = {",
                '  "generated_at_utc": now,',
                '  "status": status,',
                '  "reason": reason,',
                f'  "final_manifest": "{output_manifest}",',
                "}",
                'json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")',
                'md_path.write_text("\\n".join(["# TrainV5 B Batch Finalize Watcher","","- generated_at_utc: `"+now+"`","- status: `"+status+"`","- reason: `"+reason+"`","- starts_training: `false`"]) + "\\n", encoding="utf-8")',
                "PY",
                "}",
                "",
                "manifest_complete() {",
                '  python - "$FINAL_MANIFEST" <<'"'"'PY'"'"'',
                "from __future__ import annotations",
                "import json, sys",
                "from pathlib import Path",
                "path = Path(sys.argv[1])",
                "try:",
                '    payload = json.loads(path.read_text(encoding="utf-8"))',
                "except Exception:",
                "    raise SystemExit(1)",
                'counts = payload.get("counts") if isinstance(payload, dict) else None',
                "if not isinstance(counts, dict):",
                "    raise SystemExit(1)",
                'records = payload.get("records", [])',
                'skipped = payload.get("skipped_records", [])',
                f"if len(records) + len(skipped) >= {records}:",
                "    raise SystemExit(0)",
                "raise SystemExit(1)",
                "PY",
                "}",
                "",
                'write_status "waiting" "B batch rebake is still running"',
                "while true; do",
                "  if manifest_complete; then",
                '    write_status "finalizing" "Final manifest complete; running finalize_material_refine_trainV5_b_track.py"',
                '    if { echo "[finalize watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting finalize_material_refine_trainV5_b_track.py"; python scripts/finalize_material_refine_trainV5_b_track.py --b-root "' + str(rebake_root) + '" --full-manifest "' + str(output_manifest) + '"; echo "[finalize watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) finalize complete"; } 2>&1 | tee -a "$LOG_PATH"; then',
                '      write_status "complete" "B-track finalize completed"',
                "      exit 0",
                "    else",
                '      write_status "blocked" "Finalize script failed; inspect batch finalize log"',
                "      exit 3",
                "    fi",
                "  fi",
                '  if ! tmux has-session -t "$REBAKE_SESSION" 2>/dev/null; then',
                '    write_status "blocked" "Rebake tmux session ended before a complete final manifest was written"',
                "    exit 2",
                "  fi",
                '  sleep "$SLEEP_SECONDS"',
                "done",
                "",
            ]
        ),
        encoding="utf-8",
    )
    finalize_script.chmod(0o755)

    for session in (full_session, truth_session, finalize_session):
        check = subprocess.run(["tmux", "has-session", "-t", session], cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if check.returncode == 0:
            raise SystemExit(f"session_exists:{session}")

    subprocess.run(["tmux", "new-session", "-d", "-s", full_session, f"cd {REPO_ROOT} && bash {command_draft}"], cwd=REPO_ROOT, check=True)
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            truth_session,
            (
                f"cd {REPO_ROOT} && python scripts/monitor_material_refine_trainV5_b_rebake.py "
                f"--input-manifest {input_manifest} "
                f"--partial-manifest {partial_manifest} "
                f"--final-manifest {output_manifest} "
                f"--output-dir {rebake_root} "
                f"--output-root {prepared_root} "
                f"--dataoutput-root dataoutput "
                f"--total {records} "
                f"--interval-seconds 30"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", finalize_session, f"cd {REPO_ROOT} && bash {finalize_script}"], cwd=REPO_ROOT, check=True)
    decision = {
        "generated_at_utc": utc_now(),
        "batch_name": args.batch_name,
        "rebake_root": str(rebake_root),
        "full_session": full_session,
        "truth_session": truth_session,
        "finalize_session": finalize_session,
        "command_draft": str(command_draft),
        "input_manifest": str(input_manifest),
        "full_rebake_launched": True,
    }
    write_json(rebake_root / f"{batch_slug}_launch_status.json", decision)
    write_text(
        rebake_root / f"{batch_slug}_launch_status.md",
        "\n".join(
            [
                f"# {args.batch_name} Launch Status",
                "",
                f"- generated_at_utc: `{decision['generated_at_utc']}`",
                f"- full_rebake_launched: `true`",
                f"- full_session: `{full_session}`",
                f"- truth_session: `{truth_session}`",
                f"- finalize_session: `{finalize_session}`",
                f"- command_draft: `{command_draft}`",
            ]
        ),
    )


if __name__ == "__main__":
    main()
