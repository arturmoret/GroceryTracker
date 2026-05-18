"""SIFT keypoint + descriptor extraction."""

from __future__ import annotations

import cv2
import numpy as np

_SIFT: cv2.SIFT | None = None


def _sift_instance() -> cv2.SIFT:
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create()
    return _SIFT


def compute_sift(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
) -> tuple[list, np.ndarray]:
    """Detect SIFT keypoints + descriptors on a crop (or whole image).

    Returns (keypoints, descriptors). `descriptors` has shape (N, 128), or
    an empty (0, 128) array when no keypoints are found.
    """
    crop = _crop_or_full(image, bbox)
    if crop.size == 0:
        return [], np.zeros((0, 128), dtype=np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    keypoints, descriptors = _sift_instance().detectAndCompute(gray, None)
    if descriptors is None:
        return list(keypoints) if keypoints is not None else [], np.zeros((0, 128), dtype=np.float32)
    return list(keypoints), descriptors.astype(np.float32)


def _crop_or_full(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return image
    x, y, w, h = bbox
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = max(x0, int(x + w)), max(y0, int(y + h))
    return image[y0:y1, x0:x1]
