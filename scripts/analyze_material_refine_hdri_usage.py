#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Analyze Poly Haven HDRI bank coverage and actual material-refine manifest usage.",
    )
    parser.add_argument("--hdri-bank", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--manifest-glob", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--min-bank-records", type=int, default=900)
    parser.add_argument("--min-used-hdri-assets", type=int, default=256)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key) or "unknown") for record in records))


def manifest_records(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else []
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def resolve_manifests(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for path in args.manifest:
        if path.exists():
            paths.append(path)
    for pattern in args.manifest_glob:
        paths.extend(sorted(REPO_ROOT.glob(str(pattern))))
    seen: set[str] = set()
    resolved: list[Path] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved


def collect_hdri_ids(record: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for value in record.get("hdri_asset_ids") or []:
        if value not in {None, ""}:
            ids.append(str(value))
    for value in record.get("view_rgba_paths") or []:
        # View paths do not encode HDRIs reliably; keep this hook for future schema expansion.
        _ = value
    views = record.get("views")
    if isinstance(views, list):
        for view in views:
            if isinstance(view, dict) and view.get("lighting_asset_id") not in {None, ""}:
                ids.append(str(view["lighting_asset_id"]))
    return ids


def summarize_usage(bank_records: list[dict[str, Any]], manifests: list[Path]) -> dict[str, Any]:
    bank_by_id = {str(record.get("asset_id")): record for record in bank_records if record.get("asset_id")}
    bank_local_exists = {
        asset_id: (REPO_ROOT / str(record.get("local_path"))).exists()
        for asset_id, record in bank_by_id.items()
    }
    used_ids = Counter()
    used_by_manifest: dict[str, Counter[str]] = {}
    used_by_material: dict[str, Counter[str]] = defaultdict(Counter)
    used_by_protocol: dict[str, Counter[str]] = defaultdict(Counter)
    manifest_summaries: dict[str, Any] = {}

    for manifest in manifests:
        records = manifest_records(manifest)
        manifest_counter: Counter[str] = Counter()
        missing_hdri_records = 0
        for record in records:
            ids = collect_hdri_ids(record)
            if not ids:
                missing_hdri_records += 1
            material_family = str(record.get("material_family") or "unknown")
            protocol = str(record.get("view_light_protocol") or "unknown")
            for asset_id in ids:
                used_ids[asset_id] += 1
                manifest_counter[asset_id] += 1
                used_by_material[material_family][asset_id] += 1
                used_by_protocol[protocol][asset_id] += 1
        used_by_manifest[str(manifest.resolve())] = manifest_counter
        manifest_summaries[str(manifest.resolve())] = {
            "records": len(records),
            "records_without_hdri_asset_ids": missing_hdri_records,
            "unique_hdri_assets": len(manifest_counter),
            "top_hdri_assets": dict(manifest_counter.most_common(20)),
            "material_family": distribution(records, "material_family"),
            "view_light_protocol": distribution(records, "view_light_protocol"),
            "lighting_bank_id": distribution(records, "lighting_bank_id"),
        }

    used_strata = Counter()
    missing_from_bank = []
    for asset_id in used_ids:
        record = bank_by_id.get(asset_id)
        if record is None:
            missing_from_bank.append(asset_id)
            continue
        used_strata[str(record.get("stratum") or "unknown")] += used_ids[asset_id]

    return {
        "hdri_bank": {
            "records": len(bank_records),
            "downloaded_records": sum(str(record.get("download_status")) == "downloaded" for record in bank_records),
            "local_existing_records": sum(bool(exists) for exists in bank_local_exists.values()),
            "stratum": distribution(bank_records, "stratum"),
            "license_bucket": distribution(bank_records, "license_bucket"),
            "resolution": distribution(bank_records, "resolution"),
        },
        "usage": {
            "manifests": [str(path.resolve()) for path in manifests],
            "used_hdri_assets": len(used_ids),
            "used_asset_references": sum(used_ids.values()),
            "unused_bank_assets": max(0, len(bank_records) - len(set(used_ids) & set(bank_by_id))),
            "used_stratum_references": dict(used_strata),
            "top_hdri_assets": dict(used_ids.most_common(30)),
            "missing_used_assets_from_bank": sorted(missing_from_bank)[:100],
            "by_manifest": manifest_summaries,
            "unique_by_material_family": {
                material: len(counter)
                for material, counter in sorted(used_by_material.items())
            },
            "unique_by_view_light_protocol": {
                protocol: len(counter)
                for protocol, counter in sorted(used_by_protocol.items())
            },
        },
    }


def add_readiness(report: dict[str, Any], *, min_bank_records: int, min_used_hdri_assets: int) -> None:
    blockers = []
    bank_records = int(report["hdri_bank"]["records"])
    used_assets = int(report["usage"]["used_hdri_assets"])
    if bank_records < int(min_bank_records):
        blockers.append(f"hdri_bank_records={bank_records} below {int(min_bank_records)}")
    if used_assets < int(min_used_hdri_assets):
        blockers.append(f"used_hdri_assets={used_assets} below {int(min_used_hdri_assets)}")
    if report["usage"]["missing_used_assets_from_bank"]:
        blockers.append(f"missing_used_assets_from_bank={len(report['usage']['missing_used_assets_from_bank'])}")
    report["readiness"] = {
        "ready": not blockers,
        "blockers": blockers,
        "min_bank_records": int(min_bank_records),
        "min_used_hdri_assets": int(min_used_hdri_assets),
    }


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Material Refine HDRI Usage Report",
        "",
        f"- ready: `{report['readiness']['ready']}`",
        f"- blockers: `{json.dumps(report['readiness']['blockers'], ensure_ascii=False)}`",
        f"- bank_records: `{report['hdri_bank']['records']}`",
        f"- downloaded_records: `{report['hdri_bank']['downloaded_records']}`",
        f"- local_existing_records: `{report['hdri_bank']['local_existing_records']}`",
        f"- used_hdri_assets: `{report['usage']['used_hdri_assets']}`",
        f"- used_asset_references: `{report['usage']['used_asset_references']}`",
        f"- unused_bank_assets: `{report['usage']['unused_bank_assets']}`",
        "",
        "## Bank Strata",
        "",
        f"- stratum: `{json.dumps(report['hdri_bank']['stratum'], ensure_ascii=False)}`",
        "",
        "## Usage",
        "",
        f"- used_stratum_references: `{json.dumps(report['usage']['used_stratum_references'], ensure_ascii=False)}`",
        f"- unique_by_material_family: `{json.dumps(report['usage']['unique_by_material_family'], ensure_ascii=False)}`",
        f"- unique_by_view_light_protocol: `{json.dumps(report['usage']['unique_by_view_light_protocol'], ensure_ascii=False)}`",
        "",
        "## Manifests",
        "",
    ]
    for manifest, summary in report["usage"]["by_manifest"].items():
        lines.append(f"### {manifest}")
        lines.append(f"- records: `{summary['records']}`")
        lines.append(f"- records_without_hdri_asset_ids: `{summary['records_without_hdri_asset_ids']}`")
        lines.append(f"- unique_hdri_assets: `{summary['unique_hdri_assets']}`")
        lines.append(f"- view_light_protocol: `{json.dumps(summary['view_light_protocol'], ensure_ascii=False)}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report_html(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for manifest, summary in report["usage"]["by_manifest"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(manifest)}</td>"
            f"<td>{summary['records']}</td>"
            f"<td>{summary['records_without_hdri_asset_ids']}</td>"
            f"<td>{summary['unique_hdri_assets']}</td>"
            f"<td><code>{html.escape(json.dumps(summary['view_light_protocol'], ensure_ascii=False))}</code></td>"
            "</tr>"
        )
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>HDRI Usage Report</title>",
        "<style>body{font-family:Arial,sans-serif;background:#111820;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1320px;margin:auto}.card{background:#1a2430;border-radius:16px;padding:18px;margin:16px 0}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #344256;padding:8px;text-align:left;vertical-align:top}code{color:#b9e6ff;white-space:pre-wrap}</style>",
        "</head><body><div class='wrap'>",
        "<h1>Material Refine HDRI Usage Report</h1>",
        "<div class='card'>",
        f"<p>ready: <code>{report['readiness']['ready']}</code></p>",
        f"<p>blockers: <code>{html.escape(json.dumps(report['readiness']['blockers'], ensure_ascii=False))}</code></p>",
        f"<p>bank: <code>{report['hdri_bank']['records']}</code>, used assets: <code>{report['usage']['used_hdri_assets']}</code></p>",
        "</div><div class='card'><table><thead><tr><th>Manifest</th><th>Records</th><th>No HDRI IDs</th><th>Unique HDRIs</th><th>Protocol</th></tr></thead><tbody>",
        *rows,
        "</tbody></table></div>",
        "</div></body></html>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bank_payload = load_json(args.hdri_bank)
    bank_records = bank_payload.get("records") if isinstance(bank_payload, dict) else bank_payload
    if not isinstance(bank_records, list):
        raise TypeError(f"hdri_bank_missing_records:{args.hdri_bank}")
    manifests = resolve_manifests(args)
    report = summarize_usage([record for record in bank_records if isinstance(record, dict)], manifests)
    add_readiness(
        report,
        min_bank_records=int(args.min_bank_records),
        min_used_hdri_assets=int(args.min_used_hdri_assets),
    )
    write_json(args.output_json, report)
    if args.output_md:
        write_report_md(args.output_md, report)
    if args.output_html:
        write_report_html(args.output_html, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
