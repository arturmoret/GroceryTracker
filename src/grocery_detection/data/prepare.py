"""Generate filtered train/val/test splits from D2S annotations."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def stratified_split_images(
    coco: dict[str, Any],
    test_frac: float = 0.70,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split images into (val, test) stratified by dominant class.

    "Dominant class" = the class with the most instances in that image.
    Returns two COCO dicts; the val side gets (1 - test_frac) of each
    stratum and the test side gets the rest.
    """
    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    by_class: dict[int, list[int]] = {}
    for img_id, anns in anns_by_img.items():
        dominant = Counter(a["category_id"] for a in anns).most_common(1)[0][0]
        by_class.setdefault(dominant, []).append(img_id)

    rng = random.Random(seed)
    val_ids: set[int] = set()
    test_ids: set[int] = set()
    for _cls, ids in by_class.items():
        ids_sorted = sorted(ids)
        rng.shuffle(ids_sorted)
        n_test = int(round(len(ids_sorted) * test_frac))
        test_ids.update(ids_sorted[:n_test])
        val_ids.update(ids_sorted[n_test:])

    return _subset_by_image_ids(coco, val_ids), _subset_by_image_ids(coco, test_ids)


def _subset_by_image_ids(coco: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "images": [im for im in coco["images"] if im["id"] in image_ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in image_ids],
        "categories": [dict(c) for c in coco["categories"]],
    }
    for key in ("info", "licenses"):
        if key in coco:
            out[key] = coco[key]
    return out


def write_coco(coco: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
