"""CLI: inferencia YOLOv8 sobre split test + export a COCO results JSON.

Output: `reports/predictions/yolo_test.json` (formato COCO results) — entrada
del framework de evaluación común (H7), comparable directamente con
`reports/predictions/classical_test.json`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

print("[boot] entrypoint reached", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

print("[boot] importing project modules (ultralytics carga lazy)...", flush=True)
from grocery_detection.deep.export_coco import (  # noqa: E402
    load_image_id_map,
    yolo_to_coco_predictions,
)
from grocery_detection.deep.infer import predict_paths  # noqa: E402
from grocery_detection.utils.config import load_yaml, repo_root  # noqa: E402
from grocery_detection.utils.seed import set_seed  # noqa: E402

print("[boot] imports done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8 inferencia + COCO export.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--deep-config", default="configs/deep_yolo.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--weights", default=None,
        help="Override del best.pt (default: paths.best_weights de deep_yolo.yaml).",
    )
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    root = repo_root()
    data_cfg = load_yaml(root / args.data_config)
    deep_cfg = load_yaml(root / args.deep_config)

    weights = Path(args.weights) if args.weights else (root / deep_cfg["paths"]["best_weights"])
    if not weights.exists():
        print(f"[error] weights no encontradas: {weights}", flush=True)
        print("        Corre primero: uv run python scripts/run_deep_train.py", flush=True)
        return 1

    split_path = root / data_cfg["filtered_splits"][args.split]
    image_id_by_filename = load_image_id_map(split_path)

    images_root = root / "data/processed/yolo/images"
    if not images_root.is_dir():
        print(f"[error] no existe {images_root}. Corre prepare_yolo_dataset.py.", flush=True)
        return 1

    image_paths = [
        images_root / fn for fn in image_id_by_filename.keys() if (images_root / fn).exists()
    ]
    if not image_paths:
        print("[error] ninguna imagen del split test encontrada en yolo/images/", flush=True)
        return 1

    conf = args.conf if args.conf is not None else deep_cfg["inference"]["conf"]
    iou = args.iou if args.iou is not None else deep_cfg["inference"]["iou"]
    imgsz = deep_cfg["training"]["imgsz"]
    batch = deep_cfg["training"]["batch"]

    print(f"[setup] weights : {weights}", flush=True)
    print(f"[setup] images  : {len(image_paths)} (split={args.split})", flush=True)
    print(f"[setup] conf={conf} iou={iou} imgsz={imgsz} batch={batch}", flush=True)

    results = predict_paths(
        weights=weights,
        image_paths=image_paths,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        batch=batch,
    )
    print(f"[infer] {len(results)} resultados", flush=True)

    out_path = root / deep_cfg["paths"]["predictions"]
    n = yolo_to_coco_predictions(results, image_id_by_filename, out_path, category_offset=1)
    print(f"[done] {n} predicciones COCO → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
