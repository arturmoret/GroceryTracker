"""HOG descriptor extraction over fixed-size crops."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog

HOG_TARGET_SIZE = (64, 64)  # (width, height) — square crops for grocery items
HOG_PARAMS: dict = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def hog_dim(target_size: tuple[int, int] = HOG_TARGET_SIZE) -> int:
    """Compute the expected feature vector length for given HOG settings."""
    w, h = target_size
    pcw, pch = HOG_PARAMS["pixels_per_cell"]
    cbw, cbh = HOG_PARAMS["cells_per_block"]
    n_cells_w = w // pcw
    n_cells_h = h // pch
    n_blocks_w = n_cells_w - cbw + 1
    n_blocks_h = n_cells_h - cbh + 1
    return n_blocks_w * n_blocks_h * cbw * cbh * HOG_PARAMS["orientations"]


def compute_hog(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
    target_size: tuple[int, int] = HOG_TARGET_SIZE,
    visualize: bool = False,
):
    """Compute HOG features for a crop (or the whole image if `bbox` is None).

    Returns a 1D feature vector. If `visualize=True`, returns a tuple
    (feature_vector, viz_image).
    """
    crop = _crop_or_full(image, bbox)
    if crop.size == 0:
        feats = np.zeros(hog_dim(target_size), dtype=np.float32)
        if visualize:
            return feats, np.zeros((target_size[1], target_size[0]), dtype=np.float32)
        return feats

    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    if visualize:
        feats, viz = hog(crop, visualize=True, **HOG_PARAMS)
        return feats.astype(np.float32), viz
    feats = hog(crop, visualize=False, **HOG_PARAMS)
    return feats.astype(np.float32)


def _crop_or_full(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return image
    x, y, w, h = bbox
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = max(x0, int(x + w)), max(y0, int(y + h))
    return image[y0:y1, x0:x1]
