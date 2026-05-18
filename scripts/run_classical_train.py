"""CLI: build training features + fit the chi^2 SVM. Saves the classifier to disk."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

print("[boot] entrypoint reached", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

print("[boot] importing project modules (sklearn)...", flush=True)
import numpy as np  # noqa: E402

from grocery_detection.classical.classifier import save, train_chi2_svm  # noqa: E402
from grocery_detection.classical.descriptors.bovw import load_codebook  # noqa: E402
from grocery_detection.classical.training_set import (  # noqa: E402
    build_training_features,
    load_coco,
    load_features,
    save_features,
)
from grocery_detection.utils.config import load_yaml, repo_root  # noqa: E402
from grocery_detection.utils.seed import set_seed  # noqa: E402

print("[boot] imports done", flush=True)


def flatten_target_classes(classes_cfg: dict) -> list[str]:
    return [item for group in classes_cfg["target_classes"] for item in group["items"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train classical pipeline SVM.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--classes-config", default="configs/classes.yaml")
    parser.add_argument("--classical-config", default="configs/classical.yaml")
    parser.add_argument("--split", default="train", help="Which COCO split to train on (default: train).")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="Force recomputation of (X, y); otherwise reuse the cache.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    root = repo_root()
    data_cfg = load_yaml(root / args.data_config)
    classes_cfg = load_yaml(root / args.classes_config)
    cls_cfg = load_yaml(root / args.classical_config)

    img_dir = root / data_cfg["paths"]["d2s_images"]
    split_path = root / data_cfg["filtered_splits"][args.split]
    codebook_path = root / cls_cfg["paths"]["codebook"]
    features_path = root / cls_cfg["paths"]["features_cache"]
    model_path = root / cls_cfg["paths"]["model"]

    # Target classes -> ids = 1..K (orden de classes.yaml, igual que prepare_splits.py)
    targets = flatten_target_classes(classes_cfg)
    target_ids = list(range(1, len(targets) + 1))
    print(f"[setup] Target classes: {len(targets)} (ids 1..{len(targets)})", flush=True)
    print(f"[setup] Split          : {args.split} -> {split_path}", flush=True)
    print(f"[setup] Codebook       : {codebook_path}", flush=True)
    print(f"[setup] Features cache : {features_path}", flush=True)
    print(f"[setup] Output model   : {model_path}", flush=True)

    # --- Step 1: build/load training features ---
    if features_path.exists() and not args.rebuild_features:
        print(f"[features] Reusing cache {features_path}", flush=True)
        X, y = load_features(features_path)
    else:
        print(f"[features] Building from scratch (--rebuild-features={args.rebuild_features})", flush=True)
        coco = load_coco(split_path)
        prop_cfg = cls_cfg["proposals"]
        lab_cfg = cls_cfg["labeling"]
        ts_cfg = cls_cfg["training_set"]
        t0 = time.time()
        X, y = build_training_features(
            coco=coco,
            img_dir=img_dir,
            codebook_path=codebook_path,
            proposals_mode=prop_cfg["mode"],
            proposals_max_per_image=prop_cfg["max_per_image"],
            proposals_max_side=prop_cfg["max_side"],
            pos_iou=lab_cfg["pos_iou"],
            neg_iou=lab_cfg["neg_iou"],
            max_neg_per_image=ts_cfg["max_neg_per_image"],
            seed=ts_cfg["seed"],
        )
        print(f"[features] Done in {time.time()-t0:.0f}s  X={X.shape}, y={y.shape}", flush=True)
        save_features(features_path, X, y)
        print(f"[features] Cached at {features_path}", flush=True)

    # --- Step 2: fit classifier ---
    clf_cfg = cls_cfg["classifier"]
    unique, counts = np.unique(y, return_counts=True)
    print(f"[svm] Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}", flush=True)
    print(f"[svm] Training AdditiveChi2 + LinearSVC OvR ...", flush=True)
    t1 = time.time()
    clf = train_chi2_svm(
        X, y,
        target_class_ids=target_ids,
        C=clf_cfg["C"],
        sample_steps=clf_cfg["sample_steps"],
        max_iter=clf_cfg["max_iter"],
        seed=args.seed,
    )
    print(f"[svm] Trained in {time.time()-t1:.0f}s", flush=True)

    save(clf, model_path)
    print(f"[done] Classifier saved: {model_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
