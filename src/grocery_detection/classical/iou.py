"""IoU utilities for COCO-format xywh bboxes."""

from __future__ import annotations

import numpy as np


def iou_xywh(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two (x, y, w, h) bboxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between Nx4 and Mx4 arrays of (x, y, w, h). Returns NxM matrix."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    a = boxes_a.astype(np.float32)
    b = boxes_b.astype(np.float32)
    # Convert xywh -> xyxy
    a_x1, a_y1 = a[:, 0], a[:, 1]
    a_x2, a_y2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    b_x1, b_y1 = b[:, 0], b[:, 1]
    b_x2, b_y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]

    ix1 = np.maximum(a_x1[:, None], b_x1[None, :])
    iy1 = np.maximum(a_y1[:, None], b_y1[None, :])
    ix2 = np.minimum(a_x2[:, None], b_x2[None, :])
    iy2 = np.minimum(a_y2[:, None], b_y2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih

    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0).astype(np.float32)
