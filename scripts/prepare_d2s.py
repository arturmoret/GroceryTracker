"""CLI entry point: prepare MVTec D2S after manual archive download."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grocery_detection.data.download_d2s import prepare  # noqa: E402
from grocery_detection.utils.config import load_yaml, repo_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MVTec D2S dataset")
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to data config YAML (default: configs/data.yaml)",
    )
    args = parser.parse_args()

    root = repo_root()
    cfg = load_yaml(root / args.config)
    raw_dir = root / cfg["paths"]["raw_dir"]
    d2s_dir = root / cfg["paths"]["d2s_dir"]

    return prepare(raw_dir, d2s_dir)


if __name__ == "__main__":
    sys.exit(main())
