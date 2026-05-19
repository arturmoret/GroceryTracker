"""Chi-squared SVM classifier for the classical pipeline.

Uses Vedaldi & Zisserman's *additive chi-squared* feature map (sklearn's
`AdditiveChi2Sampler`) followed by a one-vs-rest linear SVM. This approximates
the exact chi-squared kernel SVM at a fraction of the cost.

Memory-aware design: the chi-squared expansion is applied ONCE (outside the
OvR loop), then 20 LinearSVC binary classifiers are fit serially (`n_jobs=1`
by default). Parallel OvR with the AdditiveChi2 transform inside a Pipeline
crashes 8 GB Macs because each joblib worker copies the expanded feature
matrix; serial fitting keeps peak RAM well below 1 GB even for ~20k samples.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from .labeling import BACKGROUND


@dataclass
class ClassicalSVM:
    """Trained AdditiveChi2 sampler + OvR LinearSVC, with target class id mapping."""

    sampler: AdditiveChi2Sampler
    svm: OneVsRestClassifier
    target_class_ids: list[int]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_target_classes) margin matrix.

        Target classes that were absent from training labels get column -inf
        so they are never the argmax and are always filtered by score_thresh.
        """
        X = np.clip(X.astype(np.float32, copy=False), 0.0, None)
        X_t = self.sampler.transform(X)
        seen = self.svm.classes_
        raw = self.svm.decision_function(X_t)
        # Binary edge case: when only 2 classes were seen, OvR collapses to a
        # single binary classifier whose decision_function is 1-D.
        if raw.ndim == 1:
            raw = np.stack([-raw, raw], axis=1)
        n = X.shape[0]
        out = np.full((n, len(self.target_class_ids)), -np.inf, dtype=np.float32)
        for j, c in enumerate(self.target_class_ids):
            matches = np.where(seen == c)[0]
            if matches.size:
                out[:, j] = raw[:, int(matches[0])]
        return out


def train_chi2_svm(
    X: np.ndarray,
    y: np.ndarray,
    target_class_ids: list[int],
    C: float = 1.0,
    sample_steps: int = 1,
    max_iter: int = 5000,
    seed: int = 42,
    verbose: int = 0,
    n_jobs: int = 1,
) -> ClassicalSVM:
    """Fit AdditiveChi2Sampler + OvR LinearSVC.

    Chi-squared expansion is done once (single allocation). LinearSVCs are
    then fit serially by default (`n_jobs=1`) — this is the memory-safe path
    on macOS, which copies data per joblib worker.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    # AdditiveChi2Sampler requires non-negative input.
    X = np.clip(X.astype(np.float32, copy=False), 0.0, None)

    missing_in_y = sorted(set(target_class_ids) - set(y.tolist()))
    if missing_in_y:
        print(f"  [warn] target classes ausentes en y: {missing_in_y}")

    # Step 1: fit + transform chi2 expansion ONCE.
    print(f"  [classifier] AdditiveChi2 expansion (sample_steps={sample_steps})...", flush=True)
    sampler = AdditiveChi2Sampler(sample_steps=sample_steps)
    X_t = sampler.fit_transform(X)
    print(f"  [classifier] Expanded X: {X.shape} -> {X_t.shape}  "
          f"(~{X_t.nbytes / 1024 / 1024:.0f} MB)", flush=True)

    # Step 2: OvR LinearSVC on the transformed matrix.
    print(f"  [classifier] Fitting OvR LinearSVC (n_jobs={n_jobs})...", flush=True)
    svm = OneVsRestClassifier(
        LinearSVC(C=C, dual="auto", max_iter=max_iter, random_state=seed, verbose=verbose),
        n_jobs=n_jobs,
    )
    svm.fit(X_t, y)

    return ClassicalSVM(sampler=sampler, svm=svm, target_class_ids=list(target_class_ids))


def save(clf: ClassicalSVM, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)


def load(path: Path) -> ClassicalSVM:
    with open(path, "rb") as f:
        return pickle.load(f)
