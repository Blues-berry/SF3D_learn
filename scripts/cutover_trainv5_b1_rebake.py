#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_B_ROOT = REPO_ROOT / "output/material_refine_trainV5_abc/B_track/full_1155_rebake"
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "output/material_refine_trainV5_abc/B_track/legacy_slow_run_snapshots"
DEFAULT_INPUT_MANIFEST = DEFAULT_B_ROOT / "full_1155_rebake_input_manifest.json"
DEFAULT_HDRI_BANK = REPO_ROOT / "output/highlight_pool_a_8k/aux_sources/polyhaven_hdri_bank.json"
SESSION_NAMES = (
    "trainv5_b1_full_rebake",
    "trainv5_b1_truth_monitor",
    "trainv5_b1_finalize_watcher",
)


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
    parser.add_argument("--b-root", type=Path, default=DEFAULT_B_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--parallel-workers", type=int, required=True)
    parser.add_argument("--render-resolution", type=int, default=320)
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument("--view-light-protocol", type=str, default="production_32")
    parser.add_argument("--atlas-resolution", type=int, default=1024)
    parser.add_argument("--hdri-bank-json", type=Path, default=DEFAULT_HDRI_BANK)
    parser.add_argument("--min-hdri-count", type=int, default=900)
    parser.add_argument("--max-hdri-lights", type=int, default=4)
    parser.add_argument("--cuda-device-index", type=str, default="0")
    parser.add_argument("--target-view-alignment-mean-threshold", type=float, default=0.08)
    parser.add_argument("--target-view-alignment-p95-threshold", type=float, default=0.20)
    return parser.parse_args()


def kill_tmux_session(name: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", name], cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def main() -> None:
    print(
        "DEPRECATED: use `python datasetscrip/trainv5_dataset.py launch` with a frozen B-track queue instead of B1-specific cutover.",
        file=sys.stderr,
    )
    args = parse_args()
    progress_snapshot = read_json(args.b_root / "progress_live.json", {})
    input_payload = read_json(args.input_manifest, {})
    total_records = len(input_payload.get("records", [])) if isinstance(input_payload, dict) else 0
    if total_records <= 0:
        raise SystemExit(f"empty_input_manifest:{args.input_manifest}")

    archive_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = args.archive_root / f"full_1155_rebake_{archive_stamp}"
    archive_dir.parent.mkdir(parents=True, exist_ok=True)

    copied_input_manifest = read_json(args.input_manifest, None)
    if copied_input_manifest is None:
        raise SystemExit(f"missing_input_manifest:{args.input_manifest}")

    for session in SESSION_NAMES:
        kill_tmux_session(session)

    if args.b_root.exists():
        shutil.move(str(args.b_root), str(archive_dir))

    args.b_root.mkdir(parents=True, exist_ok=True)
    input_manifest = args.b_root / "full_1155_rebake_input_manifest.json"
    write_json(input_manifest, copied_input_manifest)

    prepared_root = args.b_root / "prepared"
    output_manifest = args.b_root / "full_1155_rebake_manifest.json"
    partial_manifest = args.b_root / "full_1155_partial_manifest.json"
    summary_json = args.b_root / "full_1155_prepare_summary.json"
    summary_md = args.b_root / "full_1155_prepare_summary.md"
    run_script = args.b_root / "run_full_1155_rebake_gpu0.sh"
    monitor_log = args.b_root / "full_1155_monitor.log"
    finalize_script = args.b_root / "run_finalize_after_full_1155_rebake.sh"
    finalize_log = args.b_root / "full_1155_finalize.log"

    run_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {REPO_ROOT}",
                " ".join(
                    [
                        "CUDA_VISIBLE_DEVICES=0",
                        "python",
                        "scripts/prepare_material_refine_dataset.py",
                        "--input-manifest",
                        str(input_manifest),
                        "--output-root",
                        str(prepared_root),
                        "--output-manifest",
                        str(output_manifest),
                        "--split",
                        "full",
                        "--atlas-resolution",
                        str(args.atlas_resolution),
                        "--render-resolution",
                        str(args.render_resolution),
                        "--cycles-samples",
                        str(args.cycles_samples),
                        "--view-light-protocol",
                        str(args.view_light_protocol),
                        "--hdri-bank-json",
                        str(args.hdri_bank_json),
                        "--min-hdri-count",
                        str(args.min_hdri_count),
                        "--max-hdri-lights",
                        str(args.max_hdri_lights),
                        "--cuda-device-index",
                        str(args.cuda_device_index),
                        "--parallel-workers",
                        str(args.parallel_workers),
                        "--rebake-version",
                        "rebake_v2",
                        "--disable-render-cache",
                        "--disallow-prior-copy-fallback",
                        "--target-view-alignment-mean-threshold",
                        str(args.target_view_alignment_mean_threshold),
                        "--target-view-alignment-p95-threshold",
                        str(args.target_view_alignment_p95_threshold),
                        "--partial-manifest",
                        str(partial_manifest),
                        "--refresh-partial-every",
                        "1",
                        "--summary-json",
                        str(summary_json),
                        "--summary-md",
                        str(summary_md),
                    ]
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    finalize_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {REPO_ROOT}",
                f'REBAKE_SESSION="trainv5_b1_full_rebake"',
                f'REBAKE_DIR="{args.b_root}"',
                f'FINAL_MANIFEST="{output_manifest}"',
                f'STATUS_JSON="{args.b_root / "full_1155_finalize_status.json"}"',
                f'STATUS_MD="{args.b_root / "full_1155_finalize_status.md"}"',
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
                '  "rebake_session": "trainv5_b1_full_rebake",',
                f'  "final_manifest": "{output_manifest}",',
                "}",
                'json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")',
                'md_path.write_text("\\n".join(["# Full 1155 Finalize Watcher","","- generated_at_utc: `"+now+"`","- status: `"+status+"`","- reason: `"+reason+"`","- starts_training: `false`","- starts_upstream_generation: `false`"]) + "\\n", encoding="utf-8")',
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
                f"if len(records) + len(skipped) >= {total_records}:",
                "    raise SystemExit(0)",
                "raise SystemExit(1)",
                "PY",
                "}",
                "",
                'write_status "waiting" "B1 full rebake is still running"',
                "while true; do",
                "  if manifest_complete; then",
                '    write_status "finalizing" "B1 final manifest complete; running audit and manifest generation"',
                '    if { echo "[finalize watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting finalize_material_refine_trainV5_b_track.py"; python scripts/finalize_material_refine_trainV5_b_track.py; echo "[finalize watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) finalize complete"; } 2>&1 | tee -a "$LOG_PATH"; then',
                '      write_status "complete" "B1 audit, B2 TrainV5_plus_full, and B3 merged AB generation completed"',
                "      exit 0",
                "    else",
                '      write_status "blocked" "Finalize script failed after B1 final manifest completed; inspect full_1155_finalize.log"',
                "      exit 3",
                "    fi",
                "  fi",
                '  if ! tmux has-session -t "$REBAKE_SESSION" 2>/dev/null; then',
                '    write_status "blocked" "B1 tmux session ended before a complete final manifest was written"',
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

    decision = {
        "generated_at_utc": utc_now(),
        "archive_dir": str(archive_dir),
        "new_b_root": str(args.b_root),
        "parallel_workers": args.parallel_workers,
        "render_resolution": args.render_resolution,
        "cycles_samples": args.cycles_samples,
        "view_light_protocol": args.view_light_protocol,
        "archived_progress_snapshot": progress_snapshot,
    }
    write_json(args.b_root / "cutover_decision.json", decision)
    write_text(
        args.b_root / "cutover_decision.md",
        "\n".join(
            [
                "# TrainV5 B1 Cutover",
                "",
                f"- generated_at_utc: `{decision['generated_at_utc']}`",
                f"- archive_dir: `{decision['archive_dir']}`",
                f"- new_b_root: `{decision['new_b_root']}`",
                f"- parallel_workers: `{decision['parallel_workers']}`",
                f"- render_resolution: `{decision['render_resolution']}`",
                f"- cycles_samples: `{decision['cycles_samples']}`",
                f"- view_light_protocol: `{decision['view_light_protocol']}`",
            ]
        ),
    )

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", "trainv5_b1_full_rebake", f"cd {REPO_ROOT} && bash {run_script}"],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            "trainv5_b1_truth_monitor",
            (
                f"cd {REPO_ROOT} && python scripts/monitor_material_refine_trainV5_b_rebake.py "
                f"--input-manifest {input_manifest} "
                f"--partial-manifest {partial_manifest} "
                f"--final-manifest {output_manifest} "
                f"--output-dir {args.b_root} "
                f"--output-root {prepared_root} "
                f"--dataoutput-root dataoutput "
                f"--total {total_records} "
                f"--interval-seconds 30"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", "trainv5_b1_finalize_watcher", f"cd {REPO_ROOT} && bash {finalize_script}"],
        cwd=REPO_ROOT,
        check=True,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
