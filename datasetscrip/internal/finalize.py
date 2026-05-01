from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .common import read_json, records, repo_path, repo_root, run_cmd
from .status import refresh_status


def batch_root(config: dict[str, Any], batch: str) -> Path:
    return repo_path(config, "abc_root") / "B_track" / batch


def run_finalize(args: Any, config: dict[str, Any]) -> None:
    if not args.batch:
        raise SystemExit("--batch is required for finalize")
    b_root = batch_root(config, args.batch)
    full_manifest = Path(args.full_manifest) if args.full_manifest else b_root / f"{args.batch}_manifest.json"
    if args.dry_run:
        manifest_used = full_manifest
        payload = read_json(full_manifest, {})
        if not records(payload):
            partial = b_root / f"{args.batch}_partial_manifest.json"
            partial_payload = read_json(partial, {})
            if records(partial_payload):
                manifest_used = partial
                payload = partial_payload
        prepared = len(records(payload))
        skipped = len(payload.get("skipped_records", [])) if isinstance(payload, dict) else 0
        print(
            json.dumps(
                {
                    "batch": args.batch,
                    "b_root": str(b_root),
                    "full_manifest": str(full_manifest),
                    "manifest_used_for_estimate": str(manifest_used),
                    "prepared_records": prepared,
                    "skipped_records": skipped,
                    "estimated_max_training_pairs": prepared * 5,
                    "dry_run": True,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    run_cmd(
        [
            sys.executable,
            "scripts/finalize_material_refine_trainV5_b_track.py",
            "--b-root",
            str(b_root),
            "--full-manifest",
            str(full_manifest),
        ],
        cwd=repo_root(config),
    )
    refresh_status(config)

