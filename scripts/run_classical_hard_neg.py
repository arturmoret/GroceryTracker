"""CLI: hard negative mining loop. Re-fits SVM with mined FPs.

Checkpointing
-------------
Resume a dos niveles:
- **Por ronda**: `paths.hardneg_state` guarda `completed_rounds`. Al relanzar
  se salta a la ronda siguiente.
- **Dentro de ronda**: la mining loop volca un `.npz` parcial en
  `paths.hardneg_checkpoint` cada N imgs (igual que build_training_features).
  Si se corta a mitad de una ronda, al relanzar se reanuda esa ronda sin
  re-extraer features de imágenes ya escaneadas.

Al completar una ronda se actualiza features_cache + model + state, y se
borra el checkpoint parcial (la próxima ronda empezará en blanco contra
una SVM nueva).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

print("[boot] entrypoint reached", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

print("[boot] importing project modules (sklearn)...", flush=True)
import numpy as np  # noqa: E402

from grocery_detection.classical.classifier import load, save, train_chi2_svm  # noqa: E402
from grocery_detection.classical.descriptors.bovw import load_codebook  # noqa: E402
from grocery_detection.classical.hard_negative import (  # noqa: E402
    load_round_state,
    mine_hard_negatives,
    save_round_state,
)
from grocery_detection.classical.training_set import (  # noqa: E402
    load_coco,
    load_features,
    save_features,
)
from grocery_detection.utils.config import load_yaml, repo_root  # noqa: E402
from grocery_detection.utils.seed import set_seed  # noqa: E402

print("[boot] imports done", flush=True)


def flatten_target_classes(classes_cfg: dict) -> list[str]:
    return [item for group in classes_cfg["target_classes"] for item in group["items"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Hard negative mining + retrain.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--classes-config", default="configs/classes.yaml")
    parser.add_argument("--classical-config", default="configs/classical.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--reset", action="store_true",
        help="Borra el round state + checkpoint parcial; empieza desde la ronda 1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    root = repo_root()
    data_cfg = load_yaml(root / args.data_config)
    classes_cfg = load_yaml(root / args.classes_config)
    cls_cfg = load_yaml(root / args.classical_config)

    img_dir = root / data_cfg["paths"]["d2s_images"]
    split_path = root / data_cfg["filtered_splits"][args.split]
    codebook_path = root / cls_cfg["paths"]["codebook"]
    features_path = root / cls_cfg["paths"]["features_cache"]
    model_path = root / cls_cfg["paths"]["model"]
    hn_partial = root / cls_cfg["paths"]["hardneg_checkpoint"]
    hn_state = root / cls_cfg["paths"]["hardneg_state"]
    checkpoint_every = int(cls_cfg.get("checkpoint_every", 100))

    if not features_path.exists():
        print(f"[error] Features cache missing: {features_path}", flush=True)
        print("        Corre primero: uv run python scripts/run_classical_train.py", flush=True)
        return 1
    if not model_path.exists():
        print(f"[error] Trained classifier missing: {model_path}", flush=True)
        return 1

    if args.reset:
        for p in (hn_partial, hn_state):
            if p.exists():
                p.unlink()
                print(f"[reset] borrado {p}", flush=True)

    targets = flatten_target_classes(classes_cfg)
    target_ids = list(range(1, len(targets) + 1))

    print(f"[setup] Loading base features {features_path}", flush=True)
    X, y = load_features(features_path)
    print(f"[setup] Loading classifier   {model_path}", flush=True)
    clf = load(model_path)
    codebook = load_codebook(codebook_path)

    hn_cfg = cls_cfg["hard_negative"]
    prop_cfg = cls_cfg["proposals"]
    clf_cfg = cls_cfg["classifier"]
    coco = load_coco(split_path)

    completed = load_round_state(hn_state)
    print(f"[setup] Rondas completadas previamente: {completed}/{hn_cfg['rounds']}", flush=True)
    if completed >= hn_cfg["rounds"]:
        print("[done] Todas las rondas ya hechas. Nada que hacer (usa --reset para forzar).", flush=True)
        return 0

    for r in range(completed + 1, hn_cfg["rounds"] + 1):
        print(f"\n========== HARD NEG ROUND {r}/{hn_cfg['rounds']} ==========", flush=True)
        t0 = time.time()
        X_new, y_new = mine_hard_negatives(
            classifier=clf,
            coco=coco,
            img_dir=img_dir,
            codebook=codebook,
            proposals_mode=prop_cfg["mode"],
            proposals_max_per_image=prop_cfg["max_per_image"],
            proposals_max_side=prop_cfg["max_side"],
            fp_score_thresh=hn_cfg["fp_score_thresh"],
            neg_iou=cls_cfg["labeling"]["neg_iou"],
            max_new_per_image=hn_cfg["max_new_per_image"],
            seed=args.seed + r,
            preprocessing_cfg=cls_cfg.get("preprocessing"),
            checkpoint_path=hn_partial,
            checkpoint_every=checkpoint_every,
            resume=True,
        )
        print(f"[round {r}] Mined {X_new.shape[0]} hard negatives in {time.time()-t0:.0f}s", flush=True)
        if X_new.shape[0] == 0:
            print(f"[round {r}] No new hard negatives — convergencia, stop.", flush=True)
            save_round_state(hn_state, r)
            if hn_partial.exists():
                hn_partial.unlink()
            break

        X = np.vstack([X, X_new])
        y = np.concatenate([y, y_new])
        print(f"[round {r}] Train set ahora: X={X.shape}, y={y.shape}", flush=True)

        print(f"[round {r}] Refitting classifier...", flush=True)
        t1 = time.time()
        clf = train_chi2_svm(
            X, y,
            target_class_ids=target_ids,
            C=clf_cfg["C"],
            sample_steps=clf_cfg["sample_steps"],
            max_iter=clf_cfg["max_iter"],
            n_jobs=clf_cfg.get("n_jobs", 1),
            seed=args.seed,
        )
        print(f"[round {r}] Refit done in {time.time()-t1:.0f}s", flush=True)

        save_features(features_path, X, y)
        save(clf, model_path)
        save_round_state(hn_state, r)
        if hn_partial.exists():
            hn_partial.unlink()
        print(
            f"[round {r}] Persistidos features+model+state. Partial limpiado.",
            flush=True,
        )

    print(f"[done] Final cached: features={features_path}, model={model_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
