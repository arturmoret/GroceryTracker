"""Build feature matrices for training the chi^2 SVM.

Soporta checkpoint incremental — la función guarda progreso periódicamente y
puede reanudar tras una interrupción (Colab desconecta, kernel cae). Ver
`checkpoint_path` y `resume` en `build_training_features`.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.atomic import atomic_savez_compressed
from .descriptors.bovw import load_codebook
from .features import extract_features_batch, feature_dim
from .labeling import BACKGROUND, label_proposals
from .preprocessing import preprocess
from .proposals import resize_for_proposals, scale_proposals, selective_search


def build_training_features(
    coco: dict[str, Any],
    img_dir: Path,
    codebook_path: Path,
    proposals_mode: str = "fast",
    proposals_max_per_image: int = 300,
    proposals_max_side: int = 640,
    pos_iou: float = 0.5,
    neg_iou: float = 0.3,
    max_neg_per_image: int = 10,
    seed: int = 42,
    progress_every: int = 25,
    preprocessing_cfg: dict | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 100,
    resume: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate over train images, run SS, label proposals, extract features.

    Args:
        checkpoint_path: si se da, guarda (X, y, processed_ids) periódicamente.
        checkpoint_every: cada cuántas imágenes nuevas procesadas se vuelca el checkpoint.
        resume: si True y el checkpoint existe, lo carga y salta imágenes ya procesadas.

    Returns:
        X: (N, D) float32 feature matrix.
        y: (N,)   int64   labels. BACKGROUND=0, target classes start at 1.
    """
    codebook = load_codebook(codebook_path)
    d = feature_dim(codebook)
    rng = random.Random(seed)
    images = coco["images"]
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    chunks_X: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    processed_ids: set[int] = set()

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        try:
            cp = np.load(checkpoint_path, allow_pickle=False)
            if "processed_ids" not in cp.files:
                print(
                    f"[train-set] [resume] checkpoint {checkpoint_path} sin processed_ids, "
                    "formato viejo — ignorando y empezando de cero.",
                    flush=True,
                )
            else:
                X0, y0 = cp["X"], cp["y"]
                if X0.shape[0] > 0:
                    chunks_X.append(X0)
                    chunks_y.append(y0)
                processed_ids = {int(i) for i in cp["processed_ids"].tolist()}
                print(
                    f"[train-set] [resume] checkpoint cargado: {X0.shape[0]} samples, "
                    f"{len(processed_ids)} imágenes ya procesadas",
                    flush=True,
                )
        except Exception as e:
            print(f"[train-set] [resume] error leyendo checkpoint: {e!r}. Empezando de cero.", flush=True)

    n_pos = sum(int((c > 0).sum()) for c in chunks_y)
    n_neg = sum(int((c == BACKGROUND).sum()) for c in chunks_y)
    n_skipped = 0
    n_new_since_ckpt = 0
    t0 = time.time()

    for idx, im in enumerate(images, 1):
        if im["id"] in processed_ids:
            continue
        path = img_dir / im["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            n_skipped += 1
            processed_ids.add(im["id"])
            n_new_since_ckpt += 1
            continue
        img = preprocess(img, preprocessing_cfg)
        anns = anns_by_img.get(im["id"], [])
        gt_boxes = (
            np.array([a["bbox"] for a in anns], dtype=np.float32)
            if anns else np.zeros((0, 4), dtype=np.float32)
        )
        gt_labels = (
            np.array([a["category_id"] for a in anns], dtype=np.int64)
            if anns else np.zeros((0,), dtype=np.int64)
        )

        # GT boxes are themselves positive proposals (max IoU = 1).
        # SS proposals provide additional positives and the negatives.
        resized, scale = resize_for_proposals(img, max_side=proposals_max_side)
        ss = selective_search(resized, mode=proposals_mode, max_proposals=proposals_max_per_image)
        ss = scale_proposals(ss, scale)
        proposals = (
            np.vstack([gt_boxes, ss]).astype(np.float32)
            if gt_boxes.shape[0] else ss.astype(np.float32)
        )

        labels, _ = label_proposals(
            proposals, gt_boxes, gt_labels,
            pos_iou=pos_iou, neg_iou=neg_iou,
        )

        # Select all positives + subsample negatives
        pos_idx = np.where(labels > 0)[0]
        neg_idx = np.where(labels == BACKGROUND)[0]
        if neg_idx.size > max_neg_per_image:
            sel = rng.sample(range(neg_idx.size), max_neg_per_image)
            neg_idx = neg_idx[sel]
        sel_idx = np.concatenate([pos_idx, neg_idx]).astype(np.int64)
        if sel_idx.size == 0:
            processed_ids.add(im["id"])
            n_new_since_ckpt += 1
            continue
        sel_props = proposals[sel_idx]
        sel_labels = labels[sel_idx]

        feats = extract_features_batch(img, sel_props, codebook)
        chunks_X.append(feats)
        chunks_y.append(sel_labels)
        n_pos += int((sel_labels > 0).sum())
        n_neg += int((sel_labels == BACKGROUND).sum())
        processed_ids.add(im["id"])
        n_new_since_ckpt += 1

        if idx % progress_every == 0 or idx == len(images):
            print(
                f"[train-set] [{idx:4d}/{len(images)}] elapsed {time.time()-t0:.0f}s "
                f"pos={n_pos} neg={n_neg} skipped={n_skipped} done={len(processed_ids)}",
                flush=True,
            )

        if checkpoint_path is not None and n_new_since_ckpt >= checkpoint_every:
            _save_partial(checkpoint_path, chunks_X, chunks_y, processed_ids)
            n_new_since_ckpt = 0
            print(
                f"[train-set] [checkpoint] saved → {checkpoint_path} "
                f"({len(processed_ids)} imgs)",
                flush=True,
            )

    if not chunks_X:
        return np.zeros((0, d), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.vstack(chunks_X)
    y = np.concatenate(chunks_y)

    if checkpoint_path is not None:
        _save_partial(checkpoint_path, [X], [y], processed_ids)
        print(
            f"[train-set] [checkpoint] final → {checkpoint_path} "
            f"({len(processed_ids)} imgs)",
            flush=True,
        )

    return X, y


def _save_partial(
    path: Path,
    chunks_X: list[np.ndarray],
    chunks_y: list[np.ndarray],
    processed_ids: set[int],
) -> None:
    if chunks_X:
        X = np.vstack(chunks_X)
        y = np.concatenate(chunks_y)
    else:
        X = np.zeros((0,), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
    pids = np.array(sorted(processed_ids), dtype=np.int64)
    atomic_savez_compressed(path, X=X, y=y, processed_ids=pids)


def load_coco(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_features(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    """Guarda X, y (sin processed_ids). Atómico."""
    atomic_savez_compressed(path, X=X, y=y)


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(path)
    return npz["X"], npz["y"]
