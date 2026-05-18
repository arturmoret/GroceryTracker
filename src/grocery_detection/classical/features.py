"""Feature extraction per proposal: HOG + BoVW concatenated, L2-normalized."""

from __future__ import annotations

import numpy as np

from .descriptors.bovw import encode_bovw
from .descriptors.hog import HOG_TARGET_SIZE, compute_hog, hog_dim
from .descriptors.sift import compute_sift


def feature_dim(codebook, target_size: tuple[int, int] = HOG_TARGET_SIZE) -> int:
    """Total feature dim = HOG_dim + K."""
    return hog_dim(target_size) + codebook.n_clusters


def extract_features_batch(
    image: np.ndarray,
    proposals: np.ndarray,
    codebook,
    target_size: tuple[int, int] = HOG_TARGET_SIZE,
) -> np.ndarray:
    """Extract feature vector for each proposal.

    Optimisation: SIFT is computed ONCE on the full image; keypoints are then
    filtered per proposal by position. ~10x faster than running SIFT per crop.

    Returns array of shape (N, HOG_dim + K), each row L2-normalized and >= 0
    (clipped to keep AdditiveChi2Sampler-compatible).
    """
    n = proposals.shape[0]
    d = feature_dim(codebook, target_size)
    out = np.zeros((n, d), dtype=np.float32)
    if n == 0:
        return out

    # Full-image SIFT
    kp_all, desc_all = compute_sift(image)
    if kp_all:
        kp_xy = np.array([[kp.pt[0], kp.pt[1]] for kp in kp_all], dtype=np.float32)
    else:
        kp_xy = np.zeros((0, 2), dtype=np.float32)

    H, W = image.shape[:2]
    hd = hog_dim(target_size)
    for i, (x, y, w, h) in enumerate(proposals):
        x = max(0, int(x)); y = max(0, int(y))
        w = max(1, int(w)); h = max(1, int(h))
        # Clip to image
        if x + w > W:
            w = W - x
        if y + h > H:
            h = H - y
        if w <= 0 or h <= 0:
            continue
        bbox = (x, y, w, h)

        # HOG part (per-bbox crop)
        hog_feat = compute_hog(image, bbox=bbox, target_size=target_size)

        # BoVW part: filter full-image SIFTs by location
        if kp_xy.shape[0]:
            inside = (
                (kp_xy[:, 0] >= x) & (kp_xy[:, 0] < x + w) &
                (kp_xy[:, 1] >= y) & (kp_xy[:, 1] < y + h)
            )
            desc_in = desc_all[inside]
        else:
            desc_in = np.zeros((0, 128), dtype=np.float32)
        bovw_feat = encode_bovw(desc_in, codebook)

        feat = np.concatenate([hog_feat, bovw_feat])
        # Clip negatives (numerical safety for chi2)
        feat = np.clip(feat, 0.0, None)
        norm = float(np.linalg.norm(feat))
        if norm > 0.0:
            feat = feat / norm
        out[i] = feat

    return out
