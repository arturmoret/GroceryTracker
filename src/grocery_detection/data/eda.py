"""Exploratory Data Analysis helpers for MVTec D2S."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_coco(annotation_path: str | Path) -> dict[str, Any]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def all_class_names(coco: dict[str, Any]) -> list[str]:
    return [c["name"] for c in coco["categories"]]


def count_instances_per_class(coco: dict[str, Any]) -> dict[str, int]:
    """Return {class_name: nº bbox annotations}."""
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    counter: Counter[str] = Counter()
    for ann in coco["annotations"]:
        counter[id_to_name[ann["category_id"]]] += 1
    return dict(counter)


def count_images_per_class(coco: dict[str, Any]) -> dict[str, int]:
    """Return {class_name: nº images that contain at least one instance}."""
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    image_classes: dict[int, set[str]] = {}
    for ann in coco["annotations"]:
        image_classes.setdefault(ann["image_id"], set()).add(id_to_name[ann["category_id"]])
    counter: Counter[str] = Counter()
    for classes in image_classes.values():
        for name in classes:
            counter[name] += 1
    return dict(counter)


def flatten_target_classes(classes_yaml: dict[str, Any]) -> list[str]:
    """Read configs/classes.yaml structure and return a flat list of names."""
    flat: list[str] = []
    for group in classes_yaml["target_classes"]:
        flat.extend(group["items"])
    return flat


def verify_minimums(
    counts: dict[str, int],
    targets: list[str],
    minimum: int,
) -> dict[str, dict[str, Any]]:
    """For each target class, return its count + whether it meets the minimum."""
    result: dict[str, dict[str, Any]] = {}
    for name in targets:
        count = counts.get(name, 0)
        result[name] = {
            "count": count,
            "passes": count >= minimum,
            "missing_in_dataset": name not in counts,
        }
    return result


def fuzzy_suggest(target: str, available: list[str], top_k: int = 3) -> list[str]:
    """Cheap suggestion for class names that don't match exactly.

    Useful when D2S uses slightly different naming than configs/classes.yaml.
    """
    target_l = target.lower()
    scored = [(name, _similarity(target_l, name.lower())) for name in available]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored[:top_k]]


def _similarity(a: str, b: str) -> float:
    """Tiny similarity score based on shared 3-grams."""
    if not a or not b:
        return 0.0
    grams_a = {a[i : i + 3] for i in range(len(a) - 2)}
    grams_b = {b[i : i + 3] for i in range(len(b) - 2)}
    if not grams_a or not grams_b:
        return 0.0
    inter = len(grams_a & grams_b)
    union = len(grams_a | grams_b)
    return inter / union if union else 0.0
