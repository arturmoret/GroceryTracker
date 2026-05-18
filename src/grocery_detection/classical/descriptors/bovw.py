"""Bag-of-Visual-Words: k-means codebook + histogram encoding."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def train_codebook(
    descriptors: np.ndarray,
    n_clusters: int = 300,
    seed: int = 42,
    verbose: int = 0,
    n_init: int = 1,
    init: str = "random",
    max_iter: int = 30,
) -> KMeans:
    """Fit a k-means codebook on an (N, 128) SIFT-descriptor matrix.

    Uses full sklearn KMeans (not MiniBatchKMeans) because MiniBatch can hang
    silently in some Windows sklearn builds. Default settings are tuned for
    fast convergence with adequate quality for BoVW on the 20-class subset:
    `init='random'` (avoids the O(N·K²) k-means++ init bottleneck),
    `n_clusters=300`, `max_iter=30`. KMeans logs one line per iteration with
    verbose=1.
    """
    if descriptors.ndim != 2:
        raise ValueError(f"descriptors must be 2-D, got shape {descriptors.shape}")
    if descriptors.shape[0] < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} descriptors to train K={n_clusters} clusters, "
            f"got {descriptors.shape[0]}."
        )
    km = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=n_init,
        init=init,
        max_iter=max_iter,
        verbose=verbose,
    )
    km.fit(descriptors.astype(np.float32))
    return km


def encode_bovw(
    descriptors: np.ndarray,
    codebook: MiniBatchKMeans,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a set of descriptors into a BoVW histogram of length K.

    `descriptors` shape (N, 128). Returns a 1-D float32 vector of length
    K = codebook.n_clusters. If `normalize=True`, applies L2 normalisation.
    Empty input yields a zero vector.
    """
    K = codebook.n_clusters
    if descriptors.shape[0] == 0:
        return np.zeros(K, dtype=np.float32)
    assignments = codebook.predict(descriptors.astype(np.float32))
    hist = np.bincount(assignments, minlength=K).astype(np.float32)
    if normalize:
        norm = float(np.linalg.norm(hist))
        if norm > 0.0:
            hist = hist / norm
    return hist


def save_codebook(codebook: MiniBatchKMeans, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(codebook, f)


def load_codebook(path: Path) -> MiniBatchKMeans:
    with open(path, "rb") as f:
        return pickle.load(f)
