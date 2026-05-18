"""CLI: train the BoVW codebook from a sample of train images."""

from __future__ import annotations

import sys

# Boot ping BEFORE heavy imports (sklearn/skimage/cv2 can take seconds on first load
# and the user needs visible confirmation that the script started).
print("[boot] entrypoint reached", flush=True)

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

print("[boot] importing cv2 + numpy...", flush=True)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

print("[boot] importing project modules (will import sklearn)...", flush=True)
from grocery_detection.classical.descriptors.bovw import save_codebook, train_codebook  # noqa: E402
from grocery_detection.classical.descriptors.sift import compute_sift  # noqa: E402
from grocery_detection.classical.proposals import resize_for_proposals  # noqa: E402
from grocery_detection.utils.config import load_yaml, repo_root  # noqa: E402
from grocery_detection.utils.seed import set_seed  # noqa: E402

print("[boot] imports done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the BoVW codebook (k-means over sampled SIFT descriptors)."
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--n-images", type=int, default=300,
                        help="How many train images to sample for SIFT extraction.")
    parser.add_argument("--max-side", type=int, default=640,
                        help="Resize so the longest side <= this many px before SIFT.")
    parser.add_argument("--n-clusters", type=int, default=300,
                        help="K (codebook size). 300 funciona bien para 20 clases.")
    parser.add_argument("--max-desc-per-image", type=int, default=400,
                        help="Cap descriptors per image to balance contribution.")
    parser.add_argument("--n-init", type=int, default=1,
                        help="MiniBatchKMeans n_init (default 1; raise for quality).")
    parser.add_argument("--out", default="data/processed/codebook.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    root = repo_root()
    data_cfg = load_yaml(root / args.data_config)
    train_json = root / data_cfg["filtered_splits"]["train"]
    img_dir = root / data_cfg["paths"]["d2s_images"]

    with open(train_json, encoding="utf-8") as f:
        coco = json.load(f)
    images = list(coco["images"])
    rng = random.Random(args.seed)
    rng.shuffle(images)
    sample = images[: args.n_images]

    print(f"[sift] Sampling SIFTs from {len(sample)} train images "
          f"(cap {args.max_desc_per_image}/img, resize <= {args.max_side}px).", flush=True)

    all_desc: list[np.ndarray] = []
    skipped = 0
    t0 = time.time()
    log_every = max(1, len(sample) // 30)  # ~30 progress lines total
    for i, im in enumerate(sample, 1):
        path = img_dir / im["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            skipped += 1
            continue
        img, _ = resize_for_proposals(img, max_side=args.max_side)
        _, desc = compute_sift(img)
        if desc.shape[0] == 0:
            continue
        if desc.shape[0] > args.max_desc_per_image:
            idx = rng.sample(range(desc.shape[0]), args.max_desc_per_image)
            desc = desc[idx]
        all_desc.append(desc)
        if i % log_every == 0 or i == len(sample):
            print(f"[sift]   [{i:4d}/{len(sample)}] elapsed {time.time()-t0:.1f}s", flush=True)

    if not all_desc:
        print("ERROR: no SIFT descriptors collected.", flush=True)
        return 1

    descriptors = np.vstack(all_desc)
    print(f"[sift] Total descriptors: {descriptors.shape[0]:,} "
          f"(skipped {skipped} unreadable images, SIFT total {time.time()-t0:.1f}s)", flush=True)
    if descriptors.shape[0] < args.n_clusters:
        print(f"ERROR: only {descriptors.shape[0]} descriptors; "
              f"need >= {args.n_clusters}. Increase --n-images.", flush=True)
        return 1

    print(f"[kmeans] Training KMeans K={args.n_clusters} "
          f"on {descriptors.shape[0]:,} descriptors (n_init={args.n_init}, init=random)...", flush=True)
    print("[kmeans]   verbose=1 imprime una linea POR iteracion (max 30 lineas)", flush=True)
    t1 = time.time()
    codebook = train_codebook(
        descriptors,
        n_clusters=args.n_clusters,
        seed=args.seed,
        verbose=1,
        n_init=args.n_init,
    )
    print(f"[kmeans] Done in {time.time()-t1:.1f}s, inertia={codebook.inertia_:.0f}", flush=True)

    out_path = root / args.out
    save_codebook(codebook, out_path)
    print(f"[done] Codebook saved: {out_path}  (K={codebook.n_clusters})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
