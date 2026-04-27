#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path(sys.executable)
TARGETS = {
    "metal_dominant": 12,
    "ceramic_glazed_lacquer": 10,
    "glass_metal": 10,
    "mixed_thin_boundary": 10,
    "glossy_non_metal": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build a no-3D-FUTURE rebake_v2 pilot_64 manifest and run strict contract audit.",
    )
    parser.add_argument("--candidate-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output/material_refine_rebake_v2/pilot_64_no3dfuture")
    parser.add_argument("--target-records", type=int, default=64)
    parser.add_argument("--run-prepare", action="store_true", help="Render/rebake the selected pilot before auditing.")
    parser.add_argument(
        "--prepare-output-root",
        type=Path,
        default=Path("/4T/CXY/material_refine_rebake_v2_no3dfuture_longrun/pilot_64_no3dfuture"),
    )
    parser.add_argument("--prepared-manifest", type=Path, default=None)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--atlas-resolution", type=int, default=1024)
    parser.add_argument("--render-resolution", type=int, default=320)
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument("--view-light-protocol", type=str, default="production_32")
    parser.add_argument(
        "--hdri-bank-json",
        type=Path,
        default=REPO_ROOT / "output/highlight_pool_a_8k/aux_sources/polyhaven_hdri_bank.json",
    )
    parser.add_argument("--min-hdri-count", type=int, default=900)
    parser.add_argument("--max-hdri-lights", type=int, default=4)
    parser.add_argument("--parallel-workers", type=int, default=1)
    return parser.parse_args()


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []) or []:
            if not isinstance(record, dict):
                continue
            source = f"{record.get('source_name', '')} {record.get('generator_id', '')} {record.get('canonical_object_id', '')}".lower()
            if any(token in source for token in ("3d-future", "3d_future", "3dfuture")):
                continue
            key = str(record.get("canonical_object_id") or record.get("object_id") or record.get("source_uid") or record.get("source_model_path") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            item = dict(record)
            item["rebake_version"] = item.get("rebake_version") or "rebake_v2_pending"
            item["target_view_contract_version"] = item.get("target_view_contract_version") or "rebake_v2_pending"
            item["stored_view_target_valid_for_paper"] = False
            item["paper_stage_eligible_rebake_v2"] = False
            item["candidate_pool_only"] = True
            item["pilot_64_candidate"] = True
            records.append(item)
    return records


def priority(record: dict[str, Any]) -> tuple[int, int, float, str]:
    family = str(record.get("material_family") or "unknown")
    scarce_score = 1 if family in TARGETS and family != "glossy_non_metal" else 0
    no_prior = 1 if not bool(record.get("has_material_prior")) else 0
    confidence = float(record.get("target_confidence_mean") or 0.0)
    key = str(record.get("canonical_object_id") or record.get("object_id") or "")
    return scarce_score, no_prior, confidence, key


def select_pilot(records: list[dict[str, Any]], target_records: int) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("material_family") or "unknown")].append(record)
    for group in grouped.values():
        group.sort(key=priority, reverse=True)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    blockers: list[str] = []
    for family, target in TARGETS.items():
        cap = target if family != "glossy_non_metal" else min(target, 16)
        candidates = grouped.get(family, [])
        take = min(cap, len(candidates))
        if family != "glossy_non_metal" and take < target:
            blockers.append(f"pilot_material[{family}]={take} below {target}")
        for record in candidates[:take]:
            selected.append(record)
            selected_ids.add(str(record.get("canonical_object_id") or record.get("object_id")))

    remaining = [record for record in records if str(record.get("canonical_object_id") or record.get("object_id")) not in selected_ids]
    remaining.sort(key=priority, reverse=True)
    for record in remaining:
        if len(selected) >= target_records:
            break
        if str(record.get("material_family") or "unknown") == "glossy_non_metal" and sum(str(r.get("material_family")) == "glossy_non_metal" for r in selected) >= 16:
            continue
        selected.append(record)
    no_prior = sum(not bool(record.get("has_material_prior")) for record in selected)
    if no_prior < 12:
        blockers.append(f"pilot_without_prior={no_prior} below 12")
    if len(selected) < target_records:
        blockers.append(f"pilot_records={len(selected)} below {target_records}")
    families = {str(record.get("material_family") or "unknown") for record in selected}
    if len(families & set(TARGETS)) < 4:
        blockers.append(f"pilot_material_family_coverage={len(families & set(TARGETS))} below 4")
    return selected, blockers


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = load_records(args.candidate_manifest)
    selected, material_blockers = select_pilot(records, int(args.target_records))
    manifest_path = args.output_root / "pilot_64_manifest_rebake_v2.json"
    payload = {
        "manifest_version": "canonical_asset_record_v1_rebake_v2_pilot",
        "subset_name": "pilot_64_no3dfuture_rebake_v2",
        "summary": {
            "records": len(selected),
            "material_family": dict(Counter(str(record.get("material_family") or "unknown") for record in selected)),
            "source_name": dict(Counter(str(record.get("source_name") or "unknown") for record in selected)),
            "has_material_prior": {
                "true": sum(bool(record.get("has_material_prior")) for record in selected),
                "false": sum(not bool(record.get("has_material_prior")) for record in selected),
            },
            "pilot_material_blockers": material_blockers,
        },
        "records": selected,
    }
    write_json(manifest_path, payload)
    (args.output_root / "pilot_64_debug_panels").mkdir(parents=True, exist_ok=True)

    prepare_result: dict[str, Any] = {"action": "skipped"}
    audit_manifest = manifest_path
    prepared_manifest = args.prepared_manifest or (args.output_root / "pilot_64_manifest_rebake_v2_prepared.json")
    if args.run_prepare and material_blockers:
        prepare_result = {
            "action": "skipped_material_blockers",
            "material_blockers": material_blockers,
        }
    elif args.run_prepare:
        prepare_cmd = [
            str(PYTHON_BIN),
            str(REPO_ROOT / "scripts/prepare_material_refine_dataset.py"),
            "--input-manifest",
            str(manifest_path),
            "--output-root",
            str(args.prepare_output_root),
            "--output-manifest",
            str(prepared_manifest),
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
            str(args.gpu_id),
            "--parallel-workers",
            str(args.parallel_workers),
            "--rebake-version",
            "rebake_v2",
            "--disable-render-cache",
            "--disallow-prior-copy-fallback",
            "--refresh-partial-every",
            "1",
        ]
        prepare = subprocess.run(prepare_cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        prepare_result = {
            "action": "prepared" if prepare.returncode == 0 else "prepare_failed",
            "returncode": prepare.returncode,
            "prepared_manifest": str(prepared_manifest),
            "prepare_output_root": str(args.prepare_output_root),
            "stdout_tail": prepare.stdout.splitlines()[-80:],
        }
        if prepare.returncode == 0 and prepared_manifest.exists():
            audit_manifest = prepared_manifest

    audit_cmd = [
        str(PYTHON_BIN),
        str(REPO_ROOT / "scripts/audit_material_refine_rebake_v2_contract.py"),
        "--manifest",
        str(audit_manifest),
        "--output-root",
        str(args.output_root),
        "--min-balanced-records",
        "64",
    ]
    result = subprocess.run(audit_cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    audit_json = args.output_root / "stage1_v4_no3dfuture_rebake_v2_dataset_audit.json"
    audit = json.loads(audit_json.read_text(encoding="utf-8")) if audit_json.exists() else {}
    pass_gate = (
        result.returncode == 0
        and not material_blockers
        and (not args.run_prepare or prepare_result.get("action") == "prepared")
        and audit.get("decision") == "READY_FOR_R_TRAINING_CANDIDATE"
        and int(audit.get("promoted_records") or 0) >= 64
    )

    rejects_src = args.output_root / "rejects_manifest_rebake_v2.json"
    rejects_dst = args.output_root / "pilot_64_rejects.json"
    if rejects_src.exists():
        rejects_dst.write_text(rejects_src.read_text(encoding="utf-8"), encoding="utf-8")
    material_md = [
        "# pilot_64 Material Family Audit",
        "",
        f"- pilot_64_rebake_v2_pass: `{pass_gate}`",
        f"- material_family: `{json.dumps(payload['summary']['material_family'], ensure_ascii=False)}`",
        f"- source_name: `{json.dumps(payload['summary']['source_name'], ensure_ascii=False)}`",
        f"- has_material_prior: `{json.dumps(payload['summary']['has_material_prior'], ensure_ascii=False)}`",
        f"- material_blockers: `{json.dumps(material_blockers, ensure_ascii=False)}`",
    ]
    (args.output_root / "pilot_64_material_family_audit.md").write_text("\n".join(material_md) + "\n", encoding="utf-8")
    target_md = [
        "# pilot_64 Target/View Alignment Audit",
        "",
        f"- pilot_64_rebake_v2_pass: `{pass_gate}`",
        f"- audit_returncode: `{result.returncode}`",
        f"- decision: `{audit.get('decision', 'missing')}`",
        f"- target_view_blockers: `{json.dumps(audit.get('target_view_blockers', []), ensure_ascii=False)}`",
        f"- gate_blocker_counts: `{json.dumps(audit.get('gate_blocker_counts', {}), ensure_ascii=False)}`",
    ]
    (args.output_root / "pilot_64_target_view_alignment_audit.md").write_text("\n".join(target_md) + "\n", encoding="utf-8")
    if not pass_gate:
        failure = [
            "# pilot_64 rebake_v2 Failure Note",
            "",
            "- pilot_64_rebake_v2_pass = `false`",
            "- 7-day longrun must not be started.",
            f"- material_blockers = `{json.dumps(material_blockers, ensure_ascii=False)}`",
            f"- audit_decision = `{audit.get('decision', 'missing')}`",
            f"- target_view_blockers = `{json.dumps(audit.get('target_view_blockers', []), ensure_ascii=False)}`",
            f"- gate_blocker_counts = `{json.dumps(audit.get('gate_blocker_counts', {}), ensure_ascii=False)}`",
        ]
        (args.output_root / "pilot_64_failure_note.md").write_text("\n".join(failure) + "\n", encoding="utf-8")
    decision = {
        "pilot_64_rebake_v2_pass": pass_gate,
        "manifest": str(manifest_path),
        "audit_manifest": str(audit_manifest),
        "prepare": prepare_result,
        "audit_json": str(audit_json),
        "material_blockers": material_blockers,
        "audit_decision": audit.get("decision"),
        "audit_returncode": result.returncode,
        "audit_stdout_tail": result.stdout.splitlines()[-60:],
    }
    write_json(args.output_root / "pilot_64_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
