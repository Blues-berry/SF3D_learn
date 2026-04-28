from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_material_refiner import save_training_visualizations  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export training curve PNG/HTML from material-refine history.json.",
    )
    parser.add_argument("--history-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.history_json.parent
    history = json.loads(args.history_json.read_text())
    paths = save_training_visualizations(history, output_dir)
    print(json.dumps(paths, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
