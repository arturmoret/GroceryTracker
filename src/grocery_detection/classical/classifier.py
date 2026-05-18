"""Chi-squared SVM classifier for the classical pipeline.

Uses Vedaldi & Zisserman's *additive chi-squared* feature map (sklearn's
`AdditiveChi2Sampler`) followed by a linear SVM. This approximates the
exact chi-squared kernel SVM at a fraction of the training/inference cost,
without requiring a precomputed Gram matrix kept in memory.

Output shape from `decision_function`: (n_samples, n_classes_target),
where each column is the signed margin for that target class (1..K). The
BACKGROUND class is treated as the "negative" in the one-vs-rest training
but is *not* a predicted class at inference — proposals are emitted only
for target classes that score above a threshold.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .labeling import BACKGROUND


@dataclass
class ClassicalSVM:
    """Trained classifier + the target-class id list it was trained on."""

    model: Pipeline
    target_class_ids: list[int]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_target_classes) margin matrix.

        Target classes that were absent from the training labels (no binary
        SVM was fit for them) get column `-inf` so they are never the argmax
        and are always filtered by a positive `score_thresh`.
        """
        seen = self.model.named_steps["svm"].classes_
        raw = self.model.decision_function(X)
        # Binary edge case: decision_function returns shape (n_samples,) — sklearn
        # collapses OvR when only 2 classes exist. We expand to 2 columns where
        # column 0 corresponds to seen[0] and column 1 to seen[1].
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
    sample_steps: int = 2,
    max_iter: int = 5000,
    seed: int = 42,
    verbose: int = 0,
) -> ClassicalSVM:
    """Fit AdditiveChi2Sampler + LinearSVC OvR.

    Negative class (BACKGROUND) is included in `y` so that each binary SVM
    learns class-vs-rest with background as part of "rest".
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    # AdditiveChi2Sampler requires non-negative input.
    X = np.clip(X.astype(np.float32, copy=False), 0.0, None)
    seen = sorted(set(target_class_ids) | {BACKGROUND})
    missing_in_y = sorted(set(target_class_ids) - set(y.tolist()))
    if missing_in_y:
        print(f"  [warn] target classes ausentes en y: {missing_in_y}")

    pipeline = Pipeline([
        ("chi2", AdditiveChi2Sampler(sample_steps=sample_steps)),
        ("svm", OneVsRestClassifier(
            LinearSVC(C=C, dual="auto", max_iter=max_iter, random_state=seed, verbose=verbose),
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X, y)
    return ClassicalSVM(model=pipeline, target_class_ids=list(target_class_ids))


def save(clf: ClassicalSVM, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)


def load(path: Path) -> ClassicalSVM:
    with open(path, "rb") as f:
        return pickle.load(f)
