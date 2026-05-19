"""COCO annotations → YOLO format.

YOLO espera por imagen un .txt con líneas `<class_idx> <cx> <cy> <w> <h>`
con coordenadas normalizadas 0-1. Nuestro COCO usa `category_id` 1..20;
YOLO usa 0..19. Se aplica un offset configurable (`category_offset=1` por
defecto) al convertir.

Imágenes sin anotaciones reciben un .txt VACÍO (necesario para que YOLO
las considere "negative samples", no las ignore).
"""

from __future__ import annotations

import json
from pathlib import Path


def write_yolo_labels(
    coco_path: Path | str,
    labels_dir: Path | str,
    category_offset: int = 1,
) -> int:
    """Genera un .txt por imagen en `labels_dir`.

    Returns nº de archivos escritos.
    """
    coco_path = Path(coco_path)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    written = 0
    for im in coco["images"]:
        w = float(im["width"])
        h = float(im["height"])
        anns = anns_by_img.get(im["id"], [])
        lines: list[str] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            cls_idx = int(ann["category_id"]) - category_offset
            if cls_idx < 0:
                continue
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        stem = Path(im["file_name"]).stem
        label_path = labels_dir / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    return written


def write_split_list(
    coco_path: Path | str,
    list_path: Path | str,
    path_template: str,
) -> int:
    """Genera un .txt con un path por línea para un split YOLO.

    Args:
        path_template: format string con `{filename}`, p.ej. `images/{filename}`.

    Returns nº de líneas escritas.
    """
    coco_path = Path(coco_path)
    list_path = Path(list_path)
    list_path.parent.mkdir(parents=True, exist_ok=True)

    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    lines = [path_template.format(filename=im["file_name"]) for im in coco["images"]]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def collect_filenames(coco_paths: list[Path | str]) -> set[str]:
    """Lista de filenames únicos en uno o varios COCO JSONs."""
    out: set[str] = set()
    for cp in coco_paths:
        with open(Path(cp), encoding="utf-8") as f:
            coco = json.load(f)
        for im in coco["images"]:
            out.add(im["file_name"])
    return out
