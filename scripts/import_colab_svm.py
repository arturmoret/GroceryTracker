"""Import the SVM artifact trained on Colab into the project format.

Usage:
    1. Coloca `classical_svm_artifact.pkl` (descargado de Colab) en `data/processed/`.
    2. Corre: uv run python scripts/import_colab_svm.py
    3. Genera data/processed/classical_svm.pkl listo para el pipeline.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grocery_detection.classical.classifier import ClassicalSVM, save  # noqa: E402
from grocery_detection.utils.config import load_yaml, repo_root  # noqa: E402


def main() -> int:
    root = repo_root()
    cls_cfg = load_yaml(root / "configs" / "classical.yaml")
    artifact_path = root / "data" / "processed" / "classical_svm_artifact.pkl"
    out_path = root / cls_cfg["paths"]["model"]

    if not artifact_path.exists():
        print(f"ERROR: no encontrado {artifact_path}")
        print("Coloca primero el .pkl descargado de Colab en esa ruta.")
        return 1

    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)

    needed = {"sampler", "svm", "target_class_ids"}
    missing = needed - set(artifact.keys())
    if missing:
        print(f"ERROR: artifact incompleto, faltan claves {missing}")
        return 1

    clf = ClassicalSVM(
        sampler=artifact["sampler"],
        svm=artifact["svm"],
        target_class_ids=list(artifact["target_class_ids"]),
    )
    save(clf, out_path)
    print(f"OK: ClassicalSVM guardado en {out_path}")
    print(f"  sklearn (Colab): {artifact.get('sklearn_version', '?')}")
    print(f"  target_class_ids: {clf.target_class_ids}")
    print(f"  classes seen por la SVM: {clf.svm.classes_.tolist()}")
    print()
    print("Siguiente paso: abrir notebooks/04_classical_results.ipynb")
    print("o correr scripts/run_classical_infer.py si quieres el JSON completo de predicciones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
