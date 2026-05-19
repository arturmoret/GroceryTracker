"""YOLOv8 fine-tune wrapper sobre ultralytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_yolo(
    cfg: dict[str, Any],
    dataset_yaml: Path,
    output_dir: Path,
    weights: str | None = None,
    run_name: str = "yolov8s_d2s20",
) -> Path:
    """Fine-tune YOLO con hyperparams de `configs/deep_yolo.yaml`.

    Args:
        cfg: dict cargado de deep_yolo.yaml.
        dataset_yaml: path al dataset YAML (output de `prepare_yolo_dataset.py`).
        output_dir: directorio donde ultralytics crea `runs/<run_name>/`.
        weights: opcionalmente override del checkpoint de partida.
        run_name: nombre de la subcarpeta dentro de output_dir.

    Returns:
        Path absoluto al `best.pt` resultante.
    """
    from ultralytics import YOLO  # import perezoso: pesado, solo cargar si entrena

    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentations"]
    init_weights = weights or cfg["model"]["weights"]

    model = YOLO(init_weights)

    results = model.train(
        data=str(dataset_yaml),
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch"],
        imgsz=train_cfg["imgsz"],
        optimizer=train_cfg["optimizer"],
        lr0=train_cfg["lr0"],
        lrf=train_cfg["lrf"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
        warmup_epochs=train_cfg["warmup_epochs"],
        patience=train_cfg["patience"],
        seed=train_cfg["seed"],
        workers=train_cfg["workers"],
        freeze=train_cfg.get("freeze_backbone_epochs"),
        hsv_h=aug_cfg["hsv_h"],
        hsv_s=aug_cfg["hsv_s"],
        hsv_v=aug_cfg["hsv_v"],
        degrees=aug_cfg["degrees"],
        translate=aug_cfg["translate"],
        scale=aug_cfg["scale"],
        shear=aug_cfg["shear"],
        perspective=aug_cfg["perspective"],
        flipud=aug_cfg["flipud"],
        fliplr=aug_cfg["fliplr"],
        mosaic=aug_cfg["mosaic"],
        mixup=aug_cfg["mixup"],
        copy_paste=aug_cfg["copy_paste"],
        erasing=aug_cfg["erasing"],
        save=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    return best
