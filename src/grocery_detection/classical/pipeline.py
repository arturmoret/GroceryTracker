"""End-to-end classical detection pipeline: image -> bbox + class + score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classifier import ClassicalSVM
from .features import extract_features_batch
from .nms import nms_per_class
from .preprocessing import preprocess
from .proposals import resize_for_proposals, scale_proposals, selective_search


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # xywh
    class_id: int
    score: float


def detect(
    image: np.ndarray,
    classifier: ClassicalSVM,
    codebook,
    proposals_mode: str = "fast",
    proposals_max_per_image: int = 300,
    proposals_max_side: int = 640,
    score_thresh: float = 0.0,
    nms_iou: float = 0.5,
    top_k: int | None = 100,
    preprocessing_cfg: dict | None = None,
) -> list[Detection]:
    """Run the full classical detector on one image.

    Steps: preprocess (optional) -> Selective Search -> feature extraction
    (batched SIFT) -> chi^2 SVM scoring -> per-proposal max-class + threshold
    -> NMS per class.
    """
    image = preprocess(image, preprocessing_cfg)
    resized, scale = resize_for_proposals(image, max_side=proposals_max_side)
    ss = selective_search(resized, mode=proposals_mode, max_proposals=proposals_max_per_image)
    proposals = scale_proposals(ss, scale).astype(np.float32)
    if proposals.shape[0] == 0:
        return []

    X = extract_features_batch(image, proposals, codebook)
    scores_per_class = classifier.decision_function(X)  # (N, n_target_classes)

    # For each proposal, take its highest-scoring target class.
    best_cls_idx = scores_per_class.argmax(axis=1)
    best_score = scores_per_class[np.arange(proposals.shape[0]), best_cls_idx]
    best_class_id = np.array(classifier.target_class_ids)[best_cls_idx]

    keep = best_score >= score_thresh
    if not keep.any():
        return []
    boxes = proposals[keep]
    scores = best_score[keep]
    labels = best_class_id[keep]

    kept = nms_per_class(boxes, scores, labels, iou_thresh=nms_iou, top_k=top_k)

    return [
        Detection(
            bbox=(int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])),
            class_id=int(labels[i]),
            score=float(scores[i]),
        )
        for i in kept
    ]
