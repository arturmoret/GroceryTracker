"""Hard negative mining: add false positives back to the negative pool and retrain."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classifier import ClassicalSVM
from .features import extract_features_batch
from .iou import iou_matrix
from .labeling import BACKGROUND
from .proposals import resize_for_proposals, scale_proposals, selective_search


def mine_hard_negatives(
    classifier: ClassicalSVM,
    coco: dict[str, Any],
    img_dir: Path,
    codebook,
    proposals_mode: str = "fast",
    proposals_max_per_image: int = 300,
    proposals_max_side: int = 640,
    fp_score_thresh: float = 0.0,
    neg_iou: float = 0.3,
    max_new_per_image: int = 20,
    seed: int = 42,
    progress_every: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Find proposals that the current SVMs label as positive but are actually
    background (IoU < neg_iou with all GTs). Extract their features.

    Returns (X_hardneg, y_hardneg) where y_hardneg is all BACKGROUND.
    The caller concatenates these with the original training set and re-fits.
    """
    rng = random.Random(seed)
    images = coco["images"]
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    chunks_X: list[np.ndarray] = []
    n_new = 0
    t0 = time.time()

    for idx, im in enumerate(images, 1):
        path = img_dir / im["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            continue
        anns = anns_by_img.get(im["id"], [])
        gt_boxes = (
            np.array([a["bbox"] for a in anns], dtype=np.float32)
            if anns else np.zeros((0, 4), dtype=np.float32)
        )

        resized, scale = resize_for_proposals(img, max_side=proposals_max_side)
        ss = selective_search(resized, mode=proposals_mode, max_proposals=proposals_max_per_image)
        proposals = scale_proposals(ss, scale).astype(np.float32)
        if proposals.shape[0] == 0:
            continue

        feats = extract_features_batch(img, proposals, codebook)
        scores = classifier.decision_function(feats)  # (N, C)
        best_score = scores.max(axis=1)
        is_predicted_positive = best_score > fp_score_thresh

        # Compute IoU vs GT to identify "should have been background".
        if gt_boxes.shape[0]:
            ious = iou_matrix(proposals, gt_boxes).max(axis=1)
        else:
            ious = np.zeros(proposals.shape[0], dtype=np.float32)
        is_truly_background = ious < neg_iou

        hard_neg_mask = is_predicted_positive & is_truly_background
        hard_idx = np.where(hard_neg_mask)[0]
        if hard_idx.size == 0:
            continue

        # Keep highest-scoring FPs (most informative)
        order = np.argsort(-best_score[hard_idx])
        hard_idx = hard_idx[order]
        if hard_idx.size > max_new_per_image:
            hard_idx = hard_idx[:max_new_per_image]

        chunks_X.append(feats[hard_idx])
        n_new += hard_idx.size

        if idx % progress_every == 0 or idx == len(images):
            print(
                f"[hard-neg] [{idx:4d}/{len(images)}] elapsed {time.time()-t0:.0f}s "
                f"hard_neg_total={n_new}",
                flush=True,
            )

    if not chunks_X:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.vstack(chunks_X)
    y = np.full(X.shape[0], BACKGROUND, dtype=np.int64)
    # Deterministic shuffle so order doesn't leak image identity into the SVM fit.
    perm = np.arange(X.shape[0])
    np.random.RandomState(seed).shuffle(perm)
    return X[perm], y[perm]
