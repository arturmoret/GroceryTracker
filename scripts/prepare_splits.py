"""CLI entry point: generate filtered train/val/test splits from D2S."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from grocery_detection.data.filter_classes import filter_coco_to_classes  # noqa: E402
from grocery_detection.data.prepare import (  # noqa: E402
    stratified_split_images,
    write_coco,
)
from grocery_detection.utils.config import load_yaml, repo_root  # noqa: E402
from grocery_detection.utils.seed import set_seed  # noqa: E402

console = Console()


def flatten_target_classes(classes_cfg: dict) -> list[str]:
    return [item for group in classes_cfg["target_classes"] for item in group["items"]]


def per_class_counts(coco: dict) -> dict[str, int]:
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    counter: Counter[str] = Counter()
    for ann in coco["annotations"]:
        counter[id_to_name[ann["category_id"]]] += 1
    return dict(counter)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate filtered train/val/test splits for the target classes."
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--classes-config", default="configs/classes.yaml")
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.70,
        help="Fraction of D2S validation that goes to the test split (default 0.70).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    root = repo_root()
    data_cfg = load_yaml(root / args.data_config)
    classes_cfg = load_yaml(root / args.classes_config)
    targets = flatten_target_classes(classes_cfg)

    ann_dir = root / data_cfg["paths"]["d2s_annotations"]
    src_train = ann_dir / data_cfg["d2s_splits"]["train"]
    src_val = ann_dir / data_cfg["d2s_splits"]["val"]
    out_train = root / data_cfg["filtered_splits"]["train"]
    out_val = root / data_cfg["filtered_splits"]["val"]
    out_test = root / data_cfg["filtered_splits"]["test"]

    console.print(f"[bold]Targets[/bold]: {len(targets)} clases")
    console.print(f"[bold]Source train[/bold]: {src_train}")
    console.print(f"[bold]Source val[/bold]  : {src_val}")
    console.print(f"[bold]Test fraction[/bold]: {args.test_frac:.0%}  | seed={args.seed}")

    with open(src_train, encoding="utf-8") as f:
        d2s_train = json.load(f)
    with open(src_val, encoding="utf-8") as f:
        d2s_val = json.load(f)

    train_filtered = filter_coco_to_classes(d2s_train, targets, contiguous_ids=True)
    val_full_filtered = filter_coco_to_classes(d2s_val, targets, contiguous_ids=True)
    val_split, test_split = stratified_split_images(
        val_full_filtered, test_frac=args.test_frac, seed=args.seed
    )

    write_coco(train_filtered, out_train)
    write_coco(val_split, out_val)
    write_coco(test_split, out_test)

    train_c = per_class_counts(train_filtered)
    val_c = per_class_counts(val_split)
    test_c = per_class_counts(test_split)

    table = Table(title="Conteos por clase tras filtrado + split", show_lines=False)
    table.add_column("class", style="cyan", no_wrap=True)
    table.add_column("train", justify="right")
    table.add_column("val", justify="right")
    table.add_column("test", justify="right")
    for name in targets:
        table.add_row(name, str(train_c.get(name, 0)), str(val_c.get(name, 0)), str(test_c.get(name, 0)))
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{sum(train_c.values())}[/bold]",
        f"[bold]{sum(val_c.values())}[/bold]",
        f"[bold]{sum(test_c.values())}[/bold]",
    )
    console.print(table)

    console.print("\n[green]Splits guardados:[/green]")
    console.print(f"  train: {out_train}  ({len(train_filtered['images'])} imgs)")
    console.print(f"  val  : {out_val}  ({len(val_split['images'])} imgs)")
    console.print(f"  test : {out_test}  ({len(test_split['images'])} imgs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
