"""YOLOv8 inferencia sobre split test."""

from __future__ import annotations

from pathlib import Path


def predict_paths(
    weights: Path,
    image_paths: list[Path | str],
    conf: float = 0.001,
    iou: float = 0.7,
    imgsz: int = 640,
    batch: int = 16,
):
    """Corre inferencia sobre lista de paths absolutos. Retorna list[Results]."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return model.predict(
        source=[str(p) for p in image_paths],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        batch=batch,
        save=False,
        save_txt=False,
        save_conf=False,
        stream=False,
        verbose=False,
    )
