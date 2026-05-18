"""Selective Search region proposals using OpenCV contrib (ximgproc)."""

from __future__ import annotations

import cv2
import numpy as np

Bbox = tuple[int, int, int, int]


def selective_search(
    image: np.ndarray,
    mode: str = "fast",
    max_proposals: int | None = 1000,
) -> np.ndarray:
    """Run Selective Search on a BGR image.

    Returns an Nx4 array of (x, y, w, h) integer proposals, ordered by the
    score Selective Search assigns. `mode` is "fast" (default, ~1000 proposals)
    or "quality" (~2000, slower).
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    elif mode == "quality":
        ss.switchToSelectiveSearchQuality()
    else:
        raise ValueError(f"mode must be 'fast' or 'quality', got {mode}")
    rects = ss.process()
    if max_proposals is not None:
        rects = rects[:max_proposals]
    return rects.astype(np.int32)


def resize_for_proposals(
    image: np.ndarray,
    max_side: int = 640,
) -> tuple[np.ndarray, float]:
    """Resize so that max(h, w) ≤ max_side. Return (resized_image, scale).

    `scale` is the factor applied (≤ 1.0). To undo: bbox_orig = bbox_resized / scale.
    """
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_side:
        return image, 1.0
    scale = max_side / float(largest)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_proposals(proposals: np.ndarray, scale: float) -> np.ndarray:
    """Rescale (x, y, w, h) proposals from a resized image back to the original."""
    if scale == 1.0 or proposals.size == 0:
        return proposals.astype(np.int32)
    inv = 1.0 / scale
    out = proposals.astype(np.float32) * inv
    return np.round(out).astype(np.int32)
