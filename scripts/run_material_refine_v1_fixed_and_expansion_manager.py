#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/home/ubuntu/ssd_work/conda_envs/sf3d/bin/python")
V1_ROOT = REPO_ROOT / "output/material_refine_v1_fixed"
V1_RELEASE_ROOT = V1_ROOT / "releases"
EXPANSION_ROOT = REPO_ROOT / "output/material_refine_expansion_candidates"
FACTORY_CONFIG = REPO_ROOT / "configs/material_refine_dataset_factory_gpu0.json"
BASE_CANDIDATES = [
    ("locked346", REPO_ROOT / "output/material_refine_paper/stage1_locked346/stage1_locked346_manifest.json"),
    (
        "current_main_346",
        REPO_ROOT / "output/material_refine_paper/latest_dataset_check_20260421/stage1_subset_merged490/paper_stage1_subset_manifest.json",
    ),
    (
        "paper_stage_rehearsal_210",
        REPO_ROOT / "output/material_refine_r_v2_dayrun/paper_stage_rehearsal_210/eval_all/manifest_snapshot.json",
    ),
    (
        "stage1_v3_latest_balanced",
        REPO_ROOT / "output/material_refine_paper/stage1_v3_dataset_latest/stage1_v3_balanced_paper_manifest.json",
    ),
    (
        "stage1_v4_no3dfuture_latest",
        REPO_ROOT / "output/material_refine_paper/stage1_v4_no3dfuture_latest/stage1_v4_no3dfuture_balanced_paper_manifest.json",
    ),
]
OBJECT_SOURCE_MANIFESTS = {
    "legacy_no3dfuture_v4_candidate_only": [
        "output/material_refine_rebake_v2/no3dfuture_v4_candidate_only_manifest.json",
    ],
    "objaverse_cached_factory": [
        "output/material_refine_aux_downloads/objaverse_cached_factory_canonical/material_refine_manifest_objaverse_increment.json",
    ],
    "objaverse_cached_scarce_factory": [
        "output/material_refine_aux_downloads/objaverse_cached_scarce_factory_canonical/material_refine_manifest_objaverse_increment.json",
    ],
    "objaverse_smithsonian_factory": [
        "output/material_refine_aux_downloads/objaverse_smithsonian_factory_canonical/material_refine_manifest_objaverse_increment.json",
    ],
    "objaverse_thingiverse_factory": [
        "output/material_refine_aux_downloads/objaverse_thingiverse_factory_canonical/material_refine_manifest_objaverse_increment.json",
    ],
    "objaverse_sketchfab_scarce_factory": [
        "output/material_refine_aux_downloads/objaverse_sketchfab_scarce_factory_canonical/material_refine_manifest_objaverse_increment.json",
    ],
    "objaverse_permissive_mixed_factory": [
        "output/material_refine_aux_downloads/objaverse_permissive_mixed_factory_canonical/material_refine_manifest_objaverse_increment.json",
    ],
    "objaverse_strict_factory": [
        "output/material_refine_aux_downloads/objaverse_strict_factory_canonical/material_refine_manifest_objaverse_increment.json",
    ],
    "local_object_sources": [
        "output/material_refine_aux_downloads/local_object_sources_canonical/material_refine_manifest_local_object_increment.json",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Data-only manager for A: v1_fixed rebake and B: expansion candidate queue.",
    )
    parser.add_argument("--select-base", action="store_true")
    parser.add_argument("--start-a-rebake", action="store_true")
    parser.add_argument("--build-a-release", action="store_true")
    parser.add_argument("--start-expansion-downloads", action="store_true")
    parser.add_argument("--build-expansion-candidates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--factory-config", type=Path, default=FACTORY_CONFIG)
    parser.add_argument("--max-download-sessions", type=int, default=8)
    parser.add_argument(
        "--force-expansion-download-names",
        type=str,
        default="",
        help="Comma-separated factory download names to start even if their primary progress path exists.",
    )
    parser.add_argument("--a-max-records", type=int, default=0, help="0 means all base records.")
    parser.add_argument("--a-render-resolution", type=int, default=320)
    parser.add_argument("--a-cycles-samples", type=int, default=8)
    parser.add_argument("--a-atlas-resolution", type=int, default=1024)
    parser.add_argument("--a-view-light-protocol", type=str, default="stress_24")
    parser.add_argument(
        "--a-parallel-workers",
        type=int,
        default=6,
        help="GPU0-only Blender workers for the v1_fixed rebake. GPU1 remains hidden by CUDA_VISIBLE_DEVICES=0.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def tmux_sessions() -> set[str]:
    result = subprocess.run(["tmux", "ls"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return set()
    return {line.split(":", 1)[0] for line in result.stdout.splitlines() if line.strip()}


def shell_join(values: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(value)) for value in values)


def stable_key(record: dict[str, Any]) -> str:
    return str(record.get("canonical_object_id") or record.get("object_id") or record.get("source_uid") or "")


def existing_path(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).exists()


def split_name(record: dict[str, Any]) -> str:
    explicit = str(record.get("default_split") or "").lower()
    if explicit in {"train", "val", "test"}:
        return explicit
    paper = str(record.get("paper_split") or "").lower()
    if "val" in paper:
        return "val"
    if "test" in paper or "holdout" in paper or "ood" in paper:
        return "test"
    return "train"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    raw_or_canonical = sum(
        any(existing_path(record.get(key)) for key in ("source_model_path", "canonical_glb_path", "canonical_mesh_path"))
        for record in records
    )
    return {
        "records": len(records),
        "split": dict(Counter(split_name(record) for record in records)),
        "paper_split": dict(Counter(str(record.get("paper_split") or "unknown") for record in records)),
        "raw_or_canonical_asset_available": raw_or_canonical,
        "with_prior": sum(bool(record.get("has_material_prior")) for record in records),
        "without_prior": sum(not bool(record.get("has_material_prior")) for record in records),
        "source_name": dict(Counter(str(record.get("source_name") or "unknown") for record in records)),
        "material_family": dict(Counter(str(record.get("material_family") or "unknown") for record in records)),
        "license_bucket": dict(Counter(str(record.get("license_bucket") or "unknown") for record in records)),
    }


def choose_base_manifest() -> tuple[str, Path, list[dict[str, Any]], dict[str, Any]]:
    for name, path in BASE_CANDIDATES:
        if not path.exists():
            continue
        payload = read_json(path)
        records = [record for record in payload.get("records", []) if isinstance(record, dict)]
        if records:
            return name, path, records, payload
    raise SystemExit("no_v1_base_manifest_found")


def invalidate_base_record(record: dict[str, Any]) -> dict[str, Any]:
    item = dict(record)
    item["legacy_canonical_buffer_root"] = item.get("canonical_buffer_root") or ""
    item["legacy_uv_target_roughness_path"] = item.get("uv_target_roughness_path") or ""
    item["legacy_uv_target_metallic_path"] = item.get("uv_target_metallic_path") or ""
    item["legacy_uv_target_confidence_path"] = item.get("uv_target_confidence_path") or ""
    item["legacy_target_source_type"] = item.get("target_source_type") or ""
    item["legacy_target_quality_tier"] = item.get("target_quality_tier") or ""
    item["target_view_contract_version"] = "legacy_invalidated_for_v1_fixed"
    item["stored_view_target_valid_for_paper"] = False
    item["paper_stage_eligible"] = False
    item["paper_stage_eligible_v1_fixed"] = False
    item["dataset_role"] = "stage1_v1_fixed_base_pending_rebake"
    item["v1_fixed_rebake_status"] = "pending"
    for key in (
        "canonical_buffer_root",
        "uv_target_roughness_path",
        "uv_target_metallic_path",
        "uv_target_confidence_path",
        "target_confidence_summary",
        "target_source_type",
        "target_quality_tier",
        "target_prior_identity",
        "view_supervision_ready",
        "effective_view_supervision_rate",
    ):
        item[key] = "" if key.endswith("_path") or key in {"canonical_buffer_root", "target_source_type", "target_quality_tier"} else None
    item["default_split"] = split_name(record)
    item["include_in_full"] = True
    item["include_in_smoke"] = False
    return item


def select_base(args: argparse.Namespace) -> dict[str, Any]:
    name, path, records, _payload = choose_base_manifest()
    if args.a_max_records > 0:
        records = records[: int(args.a_max_records)]
    fixed_records = [invalidate_base_record(record) for record in records]
    payload = {
        "manifest_version": "canonical_asset_record_v1_stage1_v1_fixed_base",
        "generated_at_utc": utc_now(),
        "base_source_name": name,
        "base_source_manifest": str(path.resolve()),
        "policy": "legacy_view_and_target_invalidated_rebuild_derived_fields_only",
        "summary": summarize(fixed_records),
        "records": fixed_records,
    }
    base_manifest = V1_ROOT / "base_manifest_v1.json"
    write_json(base_manifest, payload)
    summary = payload["summary"]
    lines = [
        "# stage1_v1_fixed Base Selection",
        "",
        f"- selected_manifest: `{path}`",
        f"- base_name: `{name}`",
        f"- records: `{summary['records']}`",
        f"- train/val/test: `{json.dumps(summary['split'], ensure_ascii=False)}`",
        f"- raw/canonical asset usable: `{summary['raw_or_canonical_asset_available']}`",
        f"- with_prior / without_prior: `{summary['with_prior']} / {summary['without_prior']}`",
        f"- source distribution: `{json.dumps(summary['source_name'], ensure_ascii=False)}`",
        f"- material_family distribution: `{json.dumps(summary['material_family'], ensure_ascii=False)}`",
        "- old view/target status: `invalidated`; legacy paths are stored in `legacy_*` fields only.",
    ]
    (V1_ROOT / "base_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"action": "base_selected", "base_manifest": str(base_manifest), "records": len(fixed_records)}


def start_a_rebake(args: argparse.Namespace) -> dict[str, Any]:
    base_manifest = V1_ROOT / "base_manifest_v1.json"
    if not base_manifest.exists():
        select_base(args)
    session = "sf3d_material_refine_v1_fixed_rebake_gpu0"
    if session in tmux_sessions():
        return {"action": "already_running", "session": session}
    log_root = V1_ROOT / "logs"
    rebake_root = V1_ROOT / "rebaked_bundles"
    log_root.mkdir(parents=True, exist_ok=True)
    V1_RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    output_manifest = V1_RELEASE_ROOT / "stage1_v1_fixed_rebaked_full_manifest.json"
    partial_manifest = V1_RELEASE_ROOT / "stage1_v1_fixed_rebaked_partial_manifest.json"
    run_script = log_root / f"{session}.run.sh"
    log_path = log_root / f"{session}.log"
    prepare_cmd = [
        PYTHON_BIN,
        "scripts/prepare_material_refine_dataset.py",
        "--input-manifest",
        base_manifest,
        "--output-root",
        rebake_root,
        "--output-manifest",
        output_manifest,
        "--split",
        "full",
        "--atlas-resolution",
        str(args.a_atlas_resolution),
        "--render-resolution",
        str(args.a_render_resolution),
        "--cycles-samples",
        str(args.a_cycles_samples),
        "--view-light-protocol",
        args.a_view_light_protocol,
        "--hdri-bank-json",
        "output/highlight_pool_a_8k/aux_sources/polyhaven_hdri_bank.json",
        "--min-hdri-count",
        "900",
        "--max-hdri-lights",
        "3",
        "--cuda-device-index",
        "0",
        "--parallel-workers",
        str(args.a_parallel_workers),
        "--rebake-version",
        "v1_fixed_rebake",
        "--disable-render-cache",
        "--disallow-prior-copy-fallback",
        "--target-view-alignment-mean-threshold",
        "0.08",
        "--target-view-alignment-p95-threshold",
        "0.20",
        "--refresh-partial-every",
        "1",
        "--partial-manifest",
        partial_manifest,
    ]
    release_cmd = [
        PYTHON_BIN,
        "scripts/build_material_refine_v1_fixed_release.py",
        "--input-manifest",
        output_manifest,
        "--output-root",
        V1_RELEASE_ROOT,
    ]
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            "export CUDA_VISIBLE_DEVICES=0",
            "export PYTHONUNBUFFERED=1",
            f"{shell_join(prepare_cmd)}",
            f"{shell_join(release_cmd)}",
            "",
        ]
    )
    if args.dry_run:
        return {"action": "dry_run_start_a_rebake", "session": session, "command": script}
    run_script.write_text(script, encoding="utf-8")
    run_script.chmod(0o755)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, f"{shlex.quote(str(run_script))} > {shlex.quote(str(log_path))} 2>&1"],
        cwd=REPO_ROOT,
        check=True,
    )
    return {
        "action": "started",
        "session": session,
        "log": str(log_path),
        "output_manifest": str(output_manifest),
    }


def build_a_release() -> dict[str, Any]:
    manifest = V1_RELEASE_ROOT / "stage1_v1_fixed_rebaked_full_manifest.json"
    if not manifest.exists():
        partials = [
            path
            for path in (
                V1_RELEASE_ROOT / "stage1_v1_fixed_rebaked_partial_manifest.json",
                V1_ROOT / "rebaked_bundles/full/canonical_manifest_partial.json",
            )
            if path.exists()
        ]
        if partials:
            manifest = partials[-1]
        else:
            return {"action": "missing_rebaked_manifest", "expected": str(manifest)}
    cmd = [
        PYTHON_BIN,
        "scripts/build_material_refine_v1_fixed_release.py",
        "--input-manifest",
        manifest,
        "--output-root",
        V1_RELEASE_ROOT,
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        "action": "release_built" if result.returncode == 0 else "release_failed",
        "returncode": result.returncode,
        "input_manifest": str(manifest),
        "stdout_tail": result.stdout.splitlines()[-60:],
    }


def proxy_probe_shell(source: dict[str, Any], config: dict[str, Any]) -> str:
    network = config.get("download_network", {})
    probe_url = source.get("proxy_probe_url") or network.get("default_probe_url")
    candidates = source.get("proxy_candidates") or network.get("proxy_candidates") or ["env", "direct"]
    timeout = int(source.get("proxy_probe_timeout_seconds", network.get("proxy_probe_timeout_seconds", 8)))
    if not probe_url:
        return ""
    text = "|".join(str(item) for item in candidates)
    return f"""
PROBE_URL={shlex.quote(str(probe_url))}
PROXY_CANDIDATES={shlex.quote(text)}
best_proxy=""
best_ms=999999999
IFS='|' read -r -a proxy_candidates <<< "${{PROXY_CANDIDATES}}"
for candidate in "${{proxy_candidates[@]}}"; do
  if [ "${{candidate}}" = "env" ]; then candidate="${{HTTPS_PROXY:-${{HTTP_PROXY:-}}}}"; fi
  if [ -z "${{candidate}}" ]; then continue; fi
  start_ms=$(date +%s%3N)
  if [ "${{candidate}}" = "direct" ]; then
    env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy curl -fsSIL --max-time {timeout} "${{PROBE_URL}}" >/dev/null 2>&1
  else
    HTTP_PROXY="${{candidate}}" HTTPS_PROXY="${{candidate}}" http_proxy="${{candidate}}" https_proxy="${{candidate}}" curl -fsSIL --max-time {timeout} "${{PROBE_URL}}" >/dev/null 2>&1
  fi
  rc=$?
  end_ms=$(date +%s%3N)
  elapsed=$((end_ms - start_ms))
  if [ $rc -eq 0 ] && [ $elapsed -lt $best_ms ]; then best_ms=$elapsed; best_proxy="${{candidate}}"; fi
done
if [ -n "${{best_proxy}}" ] && [ "${{best_proxy}}" != "direct" ]; then
  export HTTP_PROXY="${{best_proxy}}" HTTPS_PROXY="${{best_proxy}}" http_proxy="${{best_proxy}}" https_proxy="${{best_proxy}}"
fi
echo "selected_proxy=${{best_proxy:-keep_existing}}"
"""


def start_expansion_downloads(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = read_json(args.factory_config)
    sessions = tmux_sessions()
    force_names = {item.strip() for item in str(args.force_expansion_download_names).split(",") if item.strip()}
    actions: list[dict[str, Any]] = []
    started = 0
    for source in config.get("downloads", []):
        if not bool(source.get("enabled", True)):
            actions.append({"name": source.get("name"), "action": "disabled"})
            continue
        if started >= int(args.max_download_sessions):
            actions.append({"name": source.get("name"), "action": "deferred_max_download_sessions"})
            continue
        session = str(source["session"]).replace("sf3d_factory_", "sf3d_expansion_")
        if session in sessions:
            actions.append({"name": source.get("name"), "session": session, "action": "already_running"})
            continue
        progress_paths = [REPO_ROOT / path for path in source.get("progress_paths", []) if isinstance(path, str)]
        skip_default = bool(config.get("download_network", {}).get("skip_if_primary_progress_exists_default", True))
        skip_if_exists = bool(source.get("skip_if_primary_progress_exists", skip_default))
        if skip_if_exists and progress_paths and progress_paths[0].exists() and str(source.get("name")) not in force_names:
            actions.append({"name": source.get("name"), "action": "skipped_primary_progress_exists", "path": str(progress_paths[0])})
            continue
        log_dir = EXPANSION_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        run_script = log_dir / f"{session}.run.sh"
        log_path = log_dir / f"{session}.log"
        command = [str(item) for item in source["command"]]
        timeout_seconds = int(source.get("timeout_seconds", 0) or 0)
        command_line = shell_join(command)
        if timeout_seconds > 0:
            command_line = f"timeout -k 60s {timeout_seconds} {command_line}"
        script = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                f"cd {shlex.quote(str(REPO_ROOT))}",
                "export CUDA_VISIBLE_DEVICES=",
                "attempt=0",
                "while true; do",
                "  attempt=$((attempt + 1))",
                "  echo \"==== attempt ${attempt} $(date -Iseconds) ====\"",
                proxy_probe_shell(source, config),
                f"  {command_line}",
                "  rc=$?",
                "  echo \"==== exit ${rc} $(date -Iseconds) ====\"",
                "  if [ $rc -eq 0 ]; then break; fi",
                f"  sleep {int(source.get('retry_seconds', 900))}",
                "done",
                "",
            ]
        )
        if args.dry_run:
            actions.append({"name": source.get("name"), "session": session, "action": "dry_run_start", "run_script": str(run_script)})
            continue
        run_script.write_text(script, encoding="utf-8")
        run_script.chmod(0o755)
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, f"{shlex.quote(str(run_script))} > {shlex.quote(str(log_path))} 2>&1"],
            cwd=REPO_ROOT,
            check=True,
        )
        started += 1
        actions.append({"name": source.get("name"), "session": session, "action": "started", "log": str(log_path)})
    return actions


def canonicalize_existing_downloads(config_path: Path) -> list[dict[str, Any]]:
    config = read_json(config_path)
    actions: list[dict[str, Any]] = []
    local_cfg = config.get("local_object_sources", {})
    if bool(local_cfg.get("enabled", False)):
        cmd = [
            PYTHON_BIN,
            "scripts/build_material_refine_local_object_manifest.py",
            "--output-root",
            REPO_ROOT / str(local_cfg.get("output_root", "output/material_refine_aux_downloads/local_object_sources_canonical")),
            "--max-total-records",
            str(local_cfg.get("max_total_records", 3600)),
        ]
        result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        actions.append({"name": "local_object_sources", "returncode": result.returncode, "stdout_tail": result.stdout.splitlines()[-20:]})
    for item in config.get("canonicalize_downloads", []):
        if not bool(item.get("enabled", True)):
            continue
        input_json = REPO_ROOT / str(item["input_json"])
        if not input_json.exists():
            actions.append({"name": item.get("name"), "action": "missing_input", "input_json": str(input_json)})
            continue
        cmd = [
            PYTHON_BIN,
            "scripts/build_objaverse_increment_manifest.py",
            "--input-json",
            input_json,
            "--output-root",
            REPO_ROOT / str(item["output_root"]),
        ]
        result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        actions.append(
            {
                "name": item.get("name"),
                "action": "canonicalized" if result.returncode == 0 else "canonicalize_failed",
                "returncode": result.returncode,
                "stdout_tail": result.stdout.splitlines()[-20:],
            }
        )
    return actions


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    return [record for record in payload.get("records", []) if isinstance(record, dict)]


def expansion_record(record: dict[str, Any], *, source_group: str, source_manifest: Path) -> dict[str, Any] | None:
    text = " ".join(
        str(record.get(key) or "")
        for key in ("source_name", "generator_id", "source_dataset", "source_model_path")
    ).lower()
    if any(token in text for token in ("3d-future", "3d_future", "3dfuture")):
        return None
    item = dict(record)
    item["dataset_role"] = "expansion_candidate"
    item["expansion_source_group"] = source_group
    item["expansion_source_manifest"] = str(source_manifest)
    if not item.get("material_family"):
        item["material_family"] = item.get("highlight_material_class") or item.get("category_bucket") or "unknown_pending_second_pass"
    item["target_view_contract_version"] = "not_rebaked_yet"
    item["stored_view_target_valid_for_paper"] = False
    item["paper_stage_eligible"] = False
    item["paper_stage_eligible_v1_fixed"] = False
    item["candidate_pool_only"] = True
    item["supervision_role"] = "auxiliary_upgrade_queue"
    return item


def write_source_reports(source_dir: Path, source_group: str, records: list[dict[str, Any]], source_paths: list[str]) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": "canonical_asset_record_v1_expansion_candidate_source",
        "generated_at_utc": utc_now(),
        "source_group": source_group,
        "source_paths": source_paths,
        "summary": summarize(records),
        "records": records,
    }
    write_json(source_dir / "source_candidate_manifest.json", payload)
    summary = payload["summary"]
    (source_dir / "source_progress.md").write_text(
        "\n".join(
            [
                f"# {source_group} Progress",
                "",
                f"- candidate_objects: `{len(records)}`",
                f"- source_paths: `{json.dumps(source_paths, ensure_ascii=False)}`",
                f"- source_distribution: `{json.dumps(summary['source_name'], ensure_ascii=False)}`",
                f"- material_family: `{json.dumps(summary['material_family'], ensure_ascii=False)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source_dir / "source_license_summary.md").write_text(
        "# License Summary\n\n"
        + f"- license_bucket: `{json.dumps(summary['license_bucket'], ensure_ascii=False)}`\n"
        + "- status: `candidate_only_until_license_and_rebake_gate`\n",
        encoding="utf-8",
    )
    (source_dir / "source_material_family_guess.md").write_text(
        "# Material Family Guess\n\n"
        + f"- material_family: `{json.dumps(summary['material_family'], ensure_ascii=False)}`\n"
        + "- status: `metadata_or_path_guess; requires second-pass validation before trainable use`\n",
        encoding="utf-8",
    )


def write_polyhaven_reports() -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for name, path, role in (
        (
            "polyhaven_hdri_bank",
            REPO_ROOT / "output/highlight_pool_a_8k/aux_sources/polyhaven_hdri_bank.json",
            "lighting_bank_only",
        ),
        (
            "polyhaven_material_bank",
            REPO_ROOT / "output/highlight_pool_a_8k/aux_sources/polyhaven_materials_factory/polyhaven_material_bank_manifest.json",
            "material_auxiliary_only",
        ),
    ):
        source_dir = EXPANSION_ROOT / name
        source_dir.mkdir(parents=True, exist_ok=True)
        records = load_manifest_records(path)
        write_json(
            source_dir / "source_candidate_manifest.json",
            {
                "manifest_version": "sf3d_expansion_non_object_auxiliary",
                "generated_at_utc": utc_now(),
                "source_group": name,
                "source_path": str(path),
                "dataset_role": role,
                "records": [],
                "auxiliary_record_count": len(records),
            },
        )
        (source_dir / "source_progress.md").write_text(
            f"# {name} Progress\n\n- role: `{role}`\n- auxiliary_records: `{len(records)}`\n- source_path: `{path}`\n",
            encoding="utf-8",
        )
        (source_dir / "source_license_summary.md").write_text(
            f"# {name} License Summary\n\n- role: `{role}`\n- object_main_pool: `false`\n",
            encoding="utf-8",
        )
        (source_dir / "source_material_family_guess.md").write_text(
            f"# {name} Material/Lighting Role\n\n- role: `{role}`\n- object_quota_counted: `false`\n",
            encoding="utf-8",
        )
        actions.append({"source_group": name, "role": role, "auxiliary_records": len(records)})
    return actions


def build_expansion_candidates(args: argparse.Namespace) -> dict[str, Any]:
    canonicalize_actions = canonicalize_existing_downloads(args.factory_config)
    all_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_actions: list[dict[str, Any]] = []
    for source_group, patterns in OBJECT_SOURCE_MANIFESTS.items():
        source_records: list[dict[str, Any]] = []
        source_paths: list[str] = []
        for pattern in patterns:
            for match in sorted(glob.glob(str(REPO_ROOT / pattern))):
                path = Path(match)
                source_paths.append(str(path))
                for record in load_manifest_records(path):
                    item = expansion_record(record, source_group=source_group, source_manifest=path)
                    if item is None:
                        continue
                    key = stable_key(item)
                    if not key:
                        continue
                    source_records.append(item)
                    if key not in seen:
                        seen.add(key)
                        all_records.append(item)
        write_source_reports(EXPANSION_ROOT / source_group, source_group, source_records, source_paths)
        source_actions.append({"source_group": source_group, "records": len(source_records), "source_paths": source_paths})
    aux_actions = write_polyhaven_reports()
    merged = {
        "manifest_version": "canonical_asset_record_v1_merged_expansion_candidate",
        "generated_at_utc": utc_now(),
        "policy": "candidate_only_not_rebaked_not_trainable",
        "summary": summarize(all_records),
        "records": all_records,
    }
    write_json(EXPANSION_ROOT / "merged_expansion_candidate_manifest.json", merged)
    status_lines = [
        "# Expansion Candidate Status",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- merged_candidate_objects: `{len(all_records)}`",
        f"- material_family: `{json.dumps(merged['summary']['material_family'], ensure_ascii=False)}`",
        f"- source_name: `{json.dumps(merged['summary']['source_name'], ensure_ascii=False)}`",
        f"- with_prior / without_prior: `{merged['summary']['with_prior']} / {merged['summary']['without_prior']}`",
        "",
        "## Source Actions",
        "",
    ]
    for action in source_actions:
        status_lines.append(f"- {action['source_group']}: `{action['records']}` candidates")
    status_lines.extend(
        [
            "",
            "## Decision",
            "",
            "- These records remain `expansion_candidate` only.",
            "- They must not enter `stage1_v1_fixed_trainable` or paper candidate manifests before rebake/audit promotion.",
            "- PolyHaven HDRI/material banks are auxiliary resources, not object main-pool records.",
        ]
    )
    (EXPANSION_ROOT / "expansion_status.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    return {
        "action": "expansion_candidates_built",
        "records": len(all_records),
        "canonicalize_actions": canonicalize_actions,
        "source_actions": source_actions,
        "aux_actions": aux_actions,
    }


def main() -> None:
    args = parse_args()
    if not any(
        (
            args.select_base,
            args.start_a_rebake,
            args.build_a_release,
            args.start_expansion_downloads,
            args.build_expansion_candidates,
        )
    ):
        args.select_base = True
        args.start_a_rebake = True
        args.build_expansion_candidates = True

    V1_ROOT.mkdir(parents=True, exist_ok=True)
    V1_RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    EXPANSION_ROOT.mkdir(parents=True, exist_ok=True)
    actions: list[dict[str, Any]] = []
    if args.select_base:
        actions.append({"select_base": select_base(args)})
    if args.start_a_rebake:
        actions.append({"a_rebake": start_a_rebake(args)})
    if args.start_expansion_downloads:
        actions.append({"expansion_downloads": start_expansion_downloads(args)})
    if args.build_expansion_candidates:
        actions.append({"expansion_candidates": build_expansion_candidates(args)})
    if args.build_a_release:
        actions.append({"a_release": build_a_release()})
    state = {
        "updated_at_utc": utc_now(),
        "mode": "v1_fixed_plus_expansion_data_only",
        "gpu_policy": "A-track GPU0 priority; GPU1 idle; no R training launch",
        "actions": actions,
        "tmux_sessions": sorted(tmux_sessions()),
    }
    write_json(V1_ROOT / "v1_fixed_and_expansion_manager_state.json", state)
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
