"""CLI: fine-tune de YOLOv8s sobre el subset 20-class D2S.

Requiere:
  - `data/processed/yolo/dataset.yaml` (output de `scripts/prepare_yolo_dataset.py`)
  - GPU recomendada (Colab T4 free funciona). En CPU es inviable.

Reanudable: ultralytics guarda checkpoint `last.pt` al final de cada época.
Si entrenas con `resume=True` y `model = YOLO('runs/.../last.pt')` continúa.
Este script aún no expone `resume` automático — añadir si se vuelve necesario.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

print("[boot] entrypoint reached", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

print("[boot] importing project modules (ultralytics carga lazy)...", flush=True)
from grocery_detection.deep.train import train_yolo  # noqa: E402
from grocery_detection.utils.config import load_yaml, repo_root  # noqa: E402
from grocery_detection.utils.seed import set_seed  # noqa: E402

print("[boot] imports done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8s sobre 20-class D2S.")
    parser.add_argument("--deep-config", default="configs/deep_yolo.yaml")
    parser.add_argument("--run-name", default="yolov8s_d2s20")
    parser.add_argument(
        "--weights", default=None,
        help="Override del checkpoint inicial (default: pesos de configs/deep_yolo.yaml).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    root = repo_root()
    cfg = load_yaml(root / args.deep_config)

    dataset_yaml = root / cfg["paths"]["dataset_yaml"]
    output_dir = root / cfg["paths"]["output_dir"]
    best_target = root / cfg["paths"]["best_weights"]

    if not dataset_yaml.exists():
        print(f"[error] Dataset YAML no encontrado: {dataset_yaml}", flush=True)
        print("        Corre primero: uv run python scripts/prepare_yolo_dataset.py", flush=True)
        return 1

    print(f"[setup] deep config : {args.deep_config}", flush=True)
    print(f"[setup] dataset yaml: {dataset_yaml}", flush=True)
    print(f"[setup] output dir  : {output_dir}", flush=True)
    print(f"[setup] run name    : {args.run_name}", flush=True)

    best = train_yolo(
        cfg=cfg,
        dataset_yaml=dataset_yaml,
        output_dir=output_dir,
        weights=args.weights,
        run_name=args.run_name,
    )
    print(f"[done] best.pt en {best}", flush=True)

    # Copia el best.pt a la ubicación canónica (data/processed/yolo_best.pt) para
    # que run_deep_infer.py lo encuentre sin tener que apuntar al runs/ profundo.
    best_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(best), str(best_target))
    print(f"[done] copiado → {best_target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
