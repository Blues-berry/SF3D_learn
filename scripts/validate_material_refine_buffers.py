from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sf3d.material_refine.manifest_quality import audit_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly validate canonical view buffers for material refinement supervision readiness.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = audit_manifest(args.manifest)
    summary = audit_payload["summary"]
    payload = {
        "manifest": audit_payload["manifest"],
        "records": summary["records"],
        "buffer_field_rates": summary["buffer_field_rates"],
        "effective_view_supervision_record_rate": summary["effective_view_supervision_record_rate"],
        "effective_view_supervision_view_rate": summary["effective_view_supervision_view_rate"],
        "strict_complete_record_rate": summary["strict_complete_record_rate"],
        "strict_complete_view_rate": summary["strict_complete_view_rate"],
    }
    json_path = args.output_dir / "buffer_validation_summary.json"
    html_path = args.output_dir / "buffer_validation_summary.html"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for field, rate in sorted(payload["buffer_field_rates"].items()):
        rows.append(f"<tr><td>{html.escape(field)}</td><td>{float(rate):.2%}</td></tr>")
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Buffer Validation Summary</title>",
                "<style>body{font-family:Arial,sans-serif;background:#10151d;color:#edf2f7;margin:0;padding:24px}.wrap{max-width:1200px;margin:auto}.card{background:#18202b;border-radius:18px;padding:18px;margin:16px 0}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #334052;padding:8px;text-align:left}</style>",
                "</head><body><div class='wrap'>",
                "<h1>Material Refine Strict Buffer Validation</h1>",
                f"<div class='card'><p>effective_view_supervision_record_rate: {payload['effective_view_supervision_record_rate']:.2%}</p><p>strict_complete_view_rate: {payload['strict_complete_view_rate']:.2%}</p></div>",
                "<div class='card'><table><thead><tr><th>Field</th><th>Rate</th></tr></thead><tbody>",
                *rows,
                "</tbody></table></div>",
                "</div></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "html": str(html_path)}, indent=2))


if __name__ == "__main__":
    main()
