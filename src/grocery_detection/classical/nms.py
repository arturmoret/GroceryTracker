"""Non-maximum suppression for detections."""

from __future__ import annotations

import numpy as np

from .iou import iou_matrix


def nms_single_class(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    """NMS over a single class. Returns indices of kept boxes (descending score)."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[i : i + 1], boxes[rest])[0]
        order = rest[ious <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_thresh: float = 0.5,
    top_k: int | None = None,
) -> np.ndarray:
    """NMS independently per class. Returns sorted indices of kept boxes (best first)."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    kept_global: list[int] = []
    for cls in np.unique(labels):
        mask = labels == cls
        cls_idx = np.where(mask)[0]
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        kept_local = nms_single_class(cls_boxes, cls_scores, iou_thresh)
        kept_global.extend(cls_idx[kept_local].tolist())
    kept_global = np.array(kept_global, dtype=np.int64)
    if kept_global.size == 0:
        return kept_global
    order = np.argsort(-scores[kept_global])
    kept_global = kept_global[order]
    if top_k is not None and kept_global.size > top_k:
        kept_global = kept_global[:top_k]
    return kept_global
