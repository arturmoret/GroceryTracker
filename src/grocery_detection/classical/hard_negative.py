"""Hard negative mining: add false positives back to the negative pool and retrain.

Soporta checkpoint a nivel de ronda (state JSON con `completed_rounds`) y dentro
de la mining loop (cada N imágenes). Reanudable tras corte de Colab.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.atomic import atomic_savez_compressed, atomic_write_json
from .classifier import ClassicalSVM
from .features import extract_features_batch
from .iou import iou_matrix
from .labeling import BACKGROUND
from .preprocessing import preprocess
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
    preprocessing_cfg: dict | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 100,
    resume: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Find proposals que las SVMs etiquetan como positivo pero IoU<neg_iou con GT.

    Reanudable: si `checkpoint_path` apunta a un .npz con X y processed_ids,
    salta imágenes ya procesadas.

    Returns (X_hardneg, y_hardneg) con y todo BACKGROUND.
    """
    images = coco["images"]
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    chunks_X: list[np.ndarray] = []
    processed_ids: set[int] = set()

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        try:
            cp = np.load(checkpoint_path, allow_pickle=False)
            if "processed_ids" in cp.files and cp["X"].shape[0] >= 0:
                X0 = cp["X"]
                if X0.shape[0] > 0:
                    chunks_X.append(X0)
                processed_ids = {int(i) for i in cp["processed_ids"].tolist()}
                print(
                    f"[hard-neg] [resume] checkpoint: {X0.shape[0]} hard-neg samples, "
                    f"{len(processed_ids)} imgs ya procesadas",
                    flush=True,
                )
        except Exception as e:
            print(f"[hard-neg] [resume] error checkpoint: {e!r}. Empiezo de cero.", flush=True)

    n_new = sum(int(c.shape[0]) for c in chunks_X)
    n_new_since_ckpt = 0
    t0 = time.time()

    for idx, im in enumerate(images, 1):
        if im["id"] in processed_ids:
            continue
        path = img_dir / im["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            processed_ids.add(im["id"])
            n_new_since_ckpt += 1
            continue
        img = preprocess(img, preprocessing_cfg)
        anns = anns_by_img.get(im["id"], [])
        gt_boxes = (
            np.array([a["bbox"] for a in anns], dtype=np.float32)
            if anns else np.zeros((0, 4), dtype=np.float32)
        )

        resized, scale = resize_for_proposals(img, max_side=proposals_max_side)
        ss = selective_search(resized, mode=proposals_mode, max_proposals=proposals_max_per_image)
        proposals = scale_proposals(ss, scale).astype(np.float32)
        if proposals.shape[0] == 0:
            processed_ids.add(im["id"])
            n_new_since_ckpt += 1
            continue

        feats = extract_features_batch(img, proposals, codebook)
        scores = classifier.decision_function(feats)  # (N, C)
        best_score = scores.max(axis=1)
        is_predicted_positive = best_score > fp_score_thresh

        if gt_boxes.shape[0]:
            ious = iou_matrix(proposals, gt_boxes).max(axis=1)
        else:
            ious = np.zeros(proposals.shape[0], dtype=np.float32)
        is_truly_background = ious < neg_iou

        hard_neg_mask = is_predicted_positive & is_truly_background
        hard_idx = np.where(hard_neg_mask)[0]
        if hard_idx.size > 0:
            order = np.argsort(-best_score[hard_idx])
            hard_idx = hard_idx[order]
            if hard_idx.size > max_new_per_image:
                hard_idx = hard_idx[:max_new_per_image]
            chunks_X.append(feats[hard_idx])
            n_new += hard_idx.size

        processed_ids.add(im["id"])
        n_new_since_ckpt += 1

        if idx % progress_every == 0 or idx == len(images):
            print(
                f"[hard-neg] [{idx:4d}/{len(images)}] elapsed {time.time()-t0:.0f}s "
                f"hard_neg_total={n_new} done={len(processed_ids)}",
                flush=True,
            )

        if checkpoint_path is not None and n_new_since_ckpt >= checkpoint_every:
            _save_partial(checkpoint_path, chunks_X, processed_ids)
            n_new_since_ckpt = 0
            print(
                f"[hard-neg] [checkpoint] saved → {checkpoint_path} "
                f"({len(processed_ids)} imgs, {n_new} hard-negs)",
                flush=True,
            )

    if not chunks_X:
        X = np.zeros((0, 1), dtype=np.float32)
    else:
        X = np.vstack(chunks_X)
    y = np.full(X.shape[0], BACKGROUND, dtype=np.int64)

    # Deterministic shuffle so order doesn't leak image identity into the SVM fit.
    if X.shape[0] > 0:
        perm = np.arange(X.shape[0])
        np.random.RandomState(seed).shuffle(perm)
        X = X[perm]
        y = y[perm]

    if checkpoint_path is not None:
        _save_partial(checkpoint_path, [X] if X.shape[0] else [], processed_ids)

    return X, y


def _save_partial(
    path: Path,
    chunks_X: list[np.ndarray],
    processed_ids: set[int],
) -> None:
    if chunks_X:
        X = np.vstack(chunks_X)
    else:
        X = np.zeros((0,), dtype=np.float32)
    pids = np.array(sorted(processed_ids), dtype=np.int64)
    atomic_savez_compressed(path, X=X, processed_ids=pids)


# ---------------------------------------------------------------------------
# Round-level state (qué ronda hemos terminado completamente)
# ---------------------------------------------------------------------------

def load_round_state(path: Path) -> int:
    """Return número de la última ronda completada. 0 si no hay state."""
    if not path.exists():
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            return int(json.load(f).get("completed_rounds", 0))
    except Exception:
        return 0


def save_round_state(path: Path, completed_rounds: int) -> None:
    atomic_write_json(path, {"completed_rounds": completed_rounds})
