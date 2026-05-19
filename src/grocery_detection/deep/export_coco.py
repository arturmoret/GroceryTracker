"""YOLO predictions → COCO JSON.

El framework de evaluación común (H7) consume predicciones en formato COCO
results.json. Este módulo produce ese formato a partir de los `Results` de
ultralytics, alineado con los `image_id` y `category_id` del split COCO
filtrado del proyecto.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.atomic import atomic_write_json


def load_image_id_map(coco_path: Path | str) -> dict[str, int]:
    """{filename: image_id} desde un split COCO."""
    with open(Path(coco_path), encoding="utf-8") as f:
        coco = json.load(f)
    return {im["file_name"]: int(im["id"]) for im in coco["images"]}


def yolo_to_coco_predictions(
    results: list[Any],
    image_id_by_filename: dict[str, int],
    out_path: Path | str,
    category_offset: int = 1,
) -> int:
    """Convierte `Results` ultralytics → COCO results JSON.

    Args:
        results: list[ultralytics.engine.results.Results] (uno por imagen).
        image_id_by_filename: map del split COCO test → image_id (usar
            `load_image_id_map(test.json)`).
        out_path: destino del JSON (se escribe atómicamente).
        category_offset: lo que se suma a la clase YOLO (0-idx) para obtener
            el `category_id` COCO (1-idx en este proyecto).

    Returns:
        nº de predicciones escritas.
    """
    out_path = Path(out_path)
    predictions: list[dict] = []
    for r in results:
        fname = Path(r.path).name
        image_id = image_id_by_filename.get(fname)
        if image_id is None:
            continue  # imagen fuera del split test esperado
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            w = float(x2 - x1)
            h = float(y2 - y1)
            predictions.append({
                "image_id": int(image_id),
                "category_id": int(cls[i]) + category_offset,
                "bbox": [float(x1), float(y1), w, h],
                "score": float(conf[i]),
            })

    atomic_write_json(out_path, predictions)
    return len(predictions)
