"""Label Selective Search proposals against ground-truth bboxes."""

from __future__ import annotations

import numpy as np

from .iou import iou_matrix

BACKGROUND = 0


def label_proposals(
    proposals: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pos_iou: float = 0.5,
    neg_iou: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Label each proposal as positive (class id), negative (BACKGROUND=0) or ignore (-1).

    Args:
        proposals: (N, 4) xywh.
        gt_boxes:  (M, 4) xywh.
        gt_labels: (M,) ints. Caller convention: class ids start at 1 (1..K),
                   background is reserved as 0, ignore as -1.
        pos_iou: IoU >= pos_iou with some GT -> positive (label = that GT class).
        neg_iou: IoU < neg_iou with ALL GTs -> negative (BACKGROUND).
        Anything between -> ignore (-1, not used for training).

    Returns (labels, ious_max) — both length N. `ious_max` is the IoU each
    proposal achieves against its best-matching GT (0 if no GT).
    """
    n = proposals.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    labels = np.full(n, -1, dtype=np.int64)

    if gt_boxes.shape[0] == 0:
        # No GT — everything is background.
        labels[:] = BACKGROUND
        return labels, np.zeros((n,), dtype=np.float32)

    ious = iou_matrix(proposals, gt_boxes)  # (N, M)
    best_gt = ious.argmax(axis=1)
    best_iou = ious[np.arange(n), best_gt]

    pos_mask = best_iou >= pos_iou
    labels[pos_mask] = gt_labels[best_gt[pos_mask]]

    neg_mask = best_iou < neg_iou
    labels[neg_mask] = BACKGROUND

    return labels, best_iou.astype(np.float32)
