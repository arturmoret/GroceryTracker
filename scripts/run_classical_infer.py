"""CLI: run the classical detector over a COCO split and write predictions JSON."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

print("[boot] entrypoint reached", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

print("[boot] importing project modules (sklearn)...", flush=True)
import cv2  # noqa: E402

from grocery_detection.classical.classifier import load  # noqa: E402
from grocery_detection.classical.descriptors.bovw import load_codebook  # noqa: E402
from grocery_detection.classical.pipeline import detect  # noqa: E402
from grocery_detection.classical.training_set import load_coco  # noqa: E402
from grocery_detection.utils.config import load_yaml, repo_root  # noqa: E402
from grocery_detection.utils.seed import set_seed  # noqa: E402

print("[boot] imports done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run classical detector and write predictions.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--classical-config", default="configs/classical.yaml")
    parser.add_argument("--split", default="test", help="Which split to infer on (default: test).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run on the first N images (debug).")
    parser.add_argument("--out", default=None,
                        help="Override output JSON path (default from classical.yaml).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    root = repo_root()
    data_cfg = load_yaml(root / args.data_config)
    cls_cfg = load_yaml(root / args.classical_config)

    img_dir = root / data_cfg["paths"]["d2s_images"]
    split_path = root / data_cfg["filtered_splits"][args.split]
    codebook_path = root / cls_cfg["paths"]["codebook"]
    model_path = root / cls_cfg["paths"]["model"]
    out_path = root / (args.out if args.out else cls_cfg["paths"]["predictions"])

    print(f"[setup] Split          : {args.split} -> {split_path}", flush=True)
    print(f"[setup] Codebook       : {codebook_path}", flush=True)
    print(f"[setup] Classifier     : {model_path}", flush=True)
    print(f"[setup] Output JSON    : {out_path}", flush=True)

    coco = load_coco(split_path)
    codebook = load_codebook(codebook_path)
    clf = load(model_path)

    images = coco["images"]
    if args.limit is not None:
        images = images[: args.limit]
        print(f"[setup] LIMIT activo: {len(images)} imágenes", flush=True)

    prop_cfg = cls_cfg["proposals"]
    inf_cfg = cls_cfg["inference"]

    predictions: list[dict] = []
    skipped = 0
    t0 = time.time()
    log_every = max(1, len(images) // 30)
    n_det_total = 0

    for i, im in enumerate(images, 1):
        path = img_dir / im["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            skipped += 1
            continue
        detections = detect(
            img,
            classifier=clf,
            codebook=codebook,
            proposals_mode=prop_cfg["mode"],
            proposals_max_per_image=prop_cfg["max_per_image"],
            proposals_max_side=prop_cfg["max_side"],
            score_thresh=inf_cfg["score_thresh"],
            nms_iou=inf_cfg["nms_iou"],
            top_k=inf_cfg["top_k_per_image"],
            preprocessing_cfg=cls_cfg.get("preprocessing"),
        )
        for d in detections:
            x, y, w, h = d.bbox
            predictions.append({
                "image_id": im["id"],
                "category_id": d.class_id,
                "bbox": [int(x), int(y), int(w), int(h)],
                "score": d.score,
            })
        n_det_total += len(detections)

        if i % log_every == 0 or i == len(images):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(images) - i)
            print(
                f"[infer] [{i:4d}/{len(images)}] elapsed {elapsed:.0f}s "
                f"eta {eta:.0f}s  det_total={n_det_total} skipped={skipped}",
                flush=True,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"[done] {len(predictions)} predictions written: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
