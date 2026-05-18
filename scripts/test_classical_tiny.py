"""Tiny smoke test: train on 5 imgs + infer on 3 imgs. ~1-2 min total.

Verifica que toda la cadena del pipeline clásico ejecuta end-to-end SIN crashes
antes de lanzar el entrenamiento completo. NO produce un detector útil — solo
testea que la maquinaria está cableada bien.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

print("[boot] entrypoint reached", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

print("[boot] importing project modules (sklearn)...", flush=True)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from grocery_detection.classical.classifier import train_chi2_svm  # noqa: E402
from grocery_detection.classical.descriptors.bovw import load_codebook  # noqa: E402
from grocery_detection.classical.pipeline import detect  # noqa: E402
from grocery_detection.classical.training_set import build_training_features, load_coco  # noqa: E402
from grocery_detection.utils.config import load_yaml, repo_root  # noqa: E402
from grocery_detection.utils.seed import set_seed  # noqa: E402

print("[boot] imports done", flush=True)


def flatten_target_classes(classes_cfg: dict) -> list[str]:
    return [item for group in classes_cfg["target_classes"] for item in group["items"]]


def main() -> int:
    set_seed(42)
    root = repo_root()
    data_cfg = load_yaml(root / "configs" / "data.yaml")
    classes_cfg = load_yaml(root / "configs" / "classes.yaml")
    cls_cfg = load_yaml(root / "configs" / "classical.yaml")

    img_dir = root / data_cfg["paths"]["d2s_images"]
    train_path = root / data_cfg["filtered_splits"]["train"]
    test_path = root / data_cfg["filtered_splits"]["test"]
    codebook_path = root / cls_cfg["paths"]["codebook"]

    targets = flatten_target_classes(classes_cfg)
    target_ids = list(range(1, len(targets) + 1))

    # ---- Tiny train ----
    print("[tiny-train] Loading 5 train images...", flush=True)
    coco = load_coco(train_path)
    coco_tiny = {
        "images": coco["images"][:5],
        "annotations": [a for a in coco["annotations"]
                        if a["image_id"] in {im["id"] for im in coco["images"][:5]}],
        "categories": coco["categories"],
    }
    t0 = time.time()
    X, y = build_training_features(
        coco_tiny,
        img_dir=img_dir,
        codebook_path=codebook_path,
        proposals_mode="fast",
        proposals_max_per_image=100,    # reducido para velocidad
        proposals_max_side=480,
        pos_iou=cls_cfg["labeling"]["pos_iou"],
        neg_iou=cls_cfg["labeling"]["neg_iou"],
        max_neg_per_image=5,
        seed=42,
        progress_every=1,
    )
    print(f"[tiny-train] Features: X={X.shape}, y={y.shape}  (took {time.time()-t0:.0f}s)", flush=True)
    print(f"[tiny-train] Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}", flush=True)

    print("[tiny-train] Training SVM (small)...", flush=True)
    t0 = time.time()
    clf = train_chi2_svm(
        X, y,
        target_class_ids=target_ids,
        C=cls_cfg["classifier"]["C"],
        sample_steps=cls_cfg["classifier"]["sample_steps"],
        max_iter=1000,
        seed=42,
    )
    print(f"[tiny-train] Trained in {time.time()-t0:.1f}s", flush=True)

    # ---- Tiny infer ----
    print("[tiny-infer] Detecting on 3 test images...", flush=True)
    codebook = load_codebook(codebook_path)
    test_coco = load_coco(test_path)
    total_det = 0
    t0 = time.time()
    for im in test_coco["images"][:3]:
        img = cv2.imread(str(img_dir / im["file_name"]))
        dets = detect(
            img, classifier=clf, codebook=codebook,
            proposals_mode="fast",
            proposals_max_per_image=100,
            proposals_max_side=480,
            score_thresh=cls_cfg["inference"]["score_thresh"],
            nms_iou=cls_cfg["inference"]["nms_iou"],
            top_k=20,
        )
        total_det += len(dets)
        cls_summary = {}
        for d in dets:
            cls_summary[d.class_id] = cls_summary.get(d.class_id, 0) + 1
        print(f"  {im['file_name']}: {len(dets)} detections  classes={cls_summary}", flush=True)
    print(f"[tiny-infer] {total_det} detections total ({time.time()-t0:.1f}s)", flush=True)

    print("\n[done] Pipeline clásico funciona end-to-end sin crashes.", flush=True)
    print("       Siguiente paso: scripts/run_classical_train.py (entreno completo)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
