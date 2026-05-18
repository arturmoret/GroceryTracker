"""Filter a COCO-format dataset to a subset of target class names."""

from __future__ import annotations

from typing import Any


def filter_coco_to_classes(
    coco: dict[str, Any],
    target_classes: list[str],
    contiguous_ids: bool = True,
) -> dict[str, Any]:
    """Return a new COCO dict containing only the target classes.

    - Annotations referring to non-target categories are dropped.
    - Images that end up with zero annotations are dropped.
    - If `contiguous_ids` is True, category ids are renumbered to 1..N
      following the order of `target_classes` (stable, deterministic).
    """
    target_set = set(target_classes)
    matched_cats = [c for c in coco["categories"] if c["name"] in target_set]

    missing = target_set - {c["name"] for c in matched_cats}
    if missing:
        raise ValueError(
            f"Target classes not present in COCO categories: {sorted(missing)}"
        )

    if contiguous_ids:
        order = {name: i for i, name in enumerate(target_classes, start=1)}
        matched_cats.sort(key=lambda c: order[c["name"]])
        old_to_new = {c["id"]: order[c["name"]] for c in matched_cats}
        new_categories = [{**c, "id": old_to_new[c["id"]]} for c in matched_cats]
    else:
        old_to_new = {c["id"]: c["id"] for c in matched_cats}
        new_categories = [dict(c) for c in matched_cats]

    new_annotations: list[dict[str, Any]] = []
    kept_image_ids: set[int] = set()
    for ann in coco["annotations"]:
        if ann["category_id"] in old_to_new:
            new_ann = dict(ann)
            new_ann["category_id"] = old_to_new[ann["category_id"]]
            new_annotations.append(new_ann)
            kept_image_ids.add(ann["image_id"])

    new_images = [im for im in coco["images"] if im["id"] in kept_image_ids]

    out: dict[str, Any] = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories,
    }
    for key in ("info", "licenses"):
        if key in coco:
            out[key] = coco[key]
    return out
