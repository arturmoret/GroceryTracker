"""CLI: prepara dataset YOLO desde los splits COCO filtrados.

Layout resultante:

    data/d2s/
        images/                    ← D2S originales (no se tocan)
        labels/                    ← .txt YOLO por imagen (este script los escribe)
    data/processed/yolo/
        train.txt                  ← paths ABSOLUTOS a las imgs train
        val.txt
        test.txt
        dataset.yaml               ← config para ultralytics

YOLO encuentra cada .txt de label reemplazando `/images/` → `/labels/` en el
path de la imagen, así que `data/d2s/{images,labels}/<x>.{jpg,txt}` queda
alineado.

Idempotente: se puede re-correr cada sesión Colab (las imgs+labels en
`/content/d2s/` son volátiles entre sesiones, así que conviene regenerar
labels en cada arranque).

Rápido (~5000 archivos pequeños, segundos). No necesita symlink ni copia de
imágenes — apunta directamente a las que ya extrajo `prepare_d2s.py` o
`colab_helper.setup_dataset`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402
from rich.console import Console  # noqa: E402

from grocery_detection.data.coco_to_yolo import write_split_list, write_yolo_labels  # noqa: E402
from grocery_detection.utils.config import load_yaml, repo_root  # noqa: E402

console = Console()


def flatten_target_classes(classes_cfg: dict) -> list[str]:
    return [item for group in classes_cfg["target_classes"] for item in group["items"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera dataset YOLO desde splits COCO.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--classes-config", default="configs/classes.yaml")
    args = parser.parse_args()

    root = repo_root()
    data_cfg = load_yaml(root / args.data_config)
    classes_cfg = load_yaml(root / args.classes_config)
    target_classes = flatten_target_classes(classes_cfg)

    d2s_dir = (root / data_cfg["paths"]["d2s_dir"]).resolve()
    images_dir = d2s_dir / "images"
    labels_dir = d2s_dir / "labels"

    if not images_dir.is_dir():
        console.print(f"[red]No existe {images_dir}. Corre prepare_d2s.py primero.[/red]")
        return 1

    yolo_dir = (root / "data/processed/yolo").resolve()
    yolo_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    coco_paths = {s: root / data_cfg["filtered_splits"][s] for s in splits}
    for s, p in coco_paths.items():
        if not p.exists():
            console.print(f"[red]Falta split {s}: {p}. Corre prepare_splits.py primero.[/red]")
            return 1

    console.print(f"[bold]YOLO dataset[/bold]")
    console.print(f"  imgs (D2S): {images_dir}")
    console.print(f"  labels    : {labels_dir}")
    console.print(f"  splits/yaml: {yolo_dir}")
    console.print(f"  classes   : {len(target_classes)}")

    # 1. Labels: un .txt por imagen (la union de todos los splits).
    for s in splits:
        n = write_yolo_labels(coco_paths[s], labels_dir, category_offset=1)
        console.print(f"  [cyan]{s}[/cyan]: {n} labels escritos en {labels_dir}")

    # 2. split.txt con paths ABSOLUTOS a las imágenes.
    image_template = str(images_dir / "{filename}")
    for s in splits:
        out = yolo_dir / f"{s}.txt"
        n = write_split_list(coco_paths[s], out, image_template)
        console.print(f"  {out.name}: {n} líneas")

    # 3. dataset.yaml — train/val/test apuntan a los .txt absolutos.
    dataset_yaml = {
        "train": str((yolo_dir / "train.txt").resolve()),
        "val": str((yolo_dir / "val.txt").resolve()),
        "test": str((yolo_dir / "test.txt").resolve()),
        "nc": len(target_classes),
        "names": {i: name for i, name in enumerate(target_classes)},
    }
    yaml_path = yolo_dir / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    console.print(f"  [green]{yaml_path}[/green] escrito")

    console.print("\n[bold green]Listo.[/bold green] Siguiente: scripts/run_deep_train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
