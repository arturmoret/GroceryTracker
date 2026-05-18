"""YAML config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    """Return absolute path to repository root (parent of `src/`)."""
    return Path(__file__).resolve().parents[3]
