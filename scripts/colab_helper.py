"""Helpers para los notebooks de Colab del pipeline clásico.

Reduce el boilerplate de:
  - montar Google Drive
  - clonar/pull del repo
  - instalar deps necesarias (sin torch/ultralytics)
  - extraer MVTec D2S al disco local de Colab (más rápido que leer de Drive)
  - sincronizar artifacts entre Drive y el repo local
  - ejecutar scripts con streaming de stdout

Layout esperado en Drive (debe crearlo manualmente el usuario una sola vez):

    /content/drive/MyDrive/grocery-detection/
        raw/
            d2s_images_v*.tar.xz        (descargado de MVTec, sube tú a Drive)
            d2s_annotations_v*.tar.xz
        processed/
            train.json                  (sube tú: prepare_splits.py local)
            val.json
            test.json
            codebook.pkl                (sube tú: train_codebook.py local)
            classical_features.npz      (generado por colab_build_features.ipynb)
            classical_svm.pkl           (generado por colab_train_svm.ipynb)
            .hardneg_state.json         (generado por colab_hard_neg.ipynb)
        predictions/
            classical_test.json         (generado por colab_infer.ipynb)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

DRIVE_ROOT_DEFAULT = "/content/drive/MyDrive/grocery-detection"
REPO_DIR_DEFAULT = "/content/repo"
D2S_LOCAL_DEFAULT = "/content/d2s"


def mount_drive() -> None:
    """Monta Google Drive en /content/drive. Idempotente."""
    if os.path.ismount("/content/drive"):
        print("[drive] ya montado en /content/drive", flush=True)
        return
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    print("[drive] montado", flush=True)


def install_deps() -> None:
    """Instala las deps mínimas para el pipeline clásico (sin torch/ultralytics)."""
    print("[deps] instalando opencv-contrib + scikit-image + rich + pyyaml...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "opencv-contrib-python", "scikit-image", "rich", "pyyaml"],
        check=True,
    )
    print("[deps] OK", flush=True)


def setup_dataset(
    drive_root: str = DRIVE_ROOT_DEFAULT,
    d2s_local: str = D2S_LOCAL_DEFAULT,
    repo_dir: str = REPO_DIR_DEFAULT,
) -> None:
    """Extrae D2S de Drive a disco local de Colab y symlinka data/d2s en el repo.

    Si /content/d2s ya tiene la estructura esperada (re-run en misma sesión),
    salta la extracción.
    """
    d2s_local_p = Path(d2s_local)
    images_dir = d2s_local_p / "images"
    annotations_dir = d2s_local_p / "annotations"

    already_extracted = (
        images_dir.exists()
        and annotations_dir.exists()
        and any(annotations_dir.glob("*.json"))
    )
    if already_extracted:
        print(f"[dataset] D2S ya extraído en {d2s_local}", flush=True)
    else:
        raw_dir = Path(drive_root) / "raw"
        if not raw_dir.is_dir():
            raise FileNotFoundError(
                f"No existe {raw_dir}. Sube los d2s_*.tar.xz a Drive primero."
            )
        archives = sorted(raw_dir.glob("d2s_*_v*.tar.xz"))
        if len(archives) < 2:
            raise FileNotFoundError(
                f"Faltan archivos D2S en {raw_dir}. Encontrados: {[a.name for a in archives]}.\n"
                "Necesarios: d2s_images_v*.tar.xz Y d2s_annotations_v*.tar.xz."
            )
        d2s_local_p.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            local_archive = Path("/content") / archive.name
            print(f"[dataset] copiando {archive.name} ({archive.stat().st_size/1024/1024:.0f} MB) → /content/...", flush=True)
            shutil.copy(str(archive), str(local_archive))
            print(f"[dataset] extrayendo {archive.name}...", flush=True)
            with tarfile.open(local_archive, "r:xz") as tar:
                tar.extractall(str(d2s_local_p))
            local_archive.unlink()
        print(f"[dataset] D2S listo en {d2s_local}", flush=True)

    repo_data_d2s = Path(repo_dir) / "data" / "d2s"
    repo_data_d2s.parent.mkdir(parents=True, exist_ok=True)
    if repo_data_d2s.is_symlink() or repo_data_d2s.exists():
        if repo_data_d2s.is_symlink():
            repo_data_d2s.unlink()
        else:
            shutil.rmtree(repo_data_d2s)
    repo_data_d2s.symlink_to(d2s_local)
    print(f"[dataset] symlink {repo_data_d2s} → {d2s_local}", flush=True)


def sync_from_drive(
    filenames: list[str],
    drive_subdir: str = "processed",
    repo_subdir: str = "data/processed",
    drive_root: str = DRIVE_ROOT_DEFAULT,
    repo_dir: str = REPO_DIR_DEFAULT,
) -> None:
    """Copia archivos desde Drive al repo local. Salta los que no existen."""
    drive_dir = Path(drive_root) / drive_subdir
    local_dir = Path(repo_dir) / repo_subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src = drive_dir / name
        dst = local_dir / name
        if src.exists():
            print(f"[sync↓] {src.name} → {dst}", flush=True)
            shutil.copy(str(src), str(dst))
        else:
            print(f"[sync↓] (skip) {src} no existe en Drive", flush=True)


def sync_to_drive(
    filenames: list[str],
    drive_subdir: str = "processed",
    repo_subdir: str = "data/processed",
    drive_root: str = DRIVE_ROOT_DEFAULT,
    repo_dir: str = REPO_DIR_DEFAULT,
) -> None:
    """Copia artifacts del repo local a Drive (sube)."""
    drive_dir = Path(drive_root) / drive_subdir
    local_dir = Path(repo_dir) / repo_subdir
    drive_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src = local_dir / name
        dst = drive_dir / name
        if src.exists():
            print(f"[sync↑] {src.name} → {dst}", flush=True)
            shutil.copy(str(src), str(dst))
        else:
            print(f"[sync↑] (skip) {src} no existe en local", flush=True)


def run_script(
    script_rel_path: str,
    *args: str,
    repo_dir: str = REPO_DIR_DEFAULT,
) -> int:
    """Lanza un script Python con cwd = repo. Streamea stdout en vivo."""
    cmd = [sys.executable, "-u", str(Path(repo_dir) / script_rel_path), *args]
    print(f"[run] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    print(f"[run] exit code {rc}", flush=True)
    return rc
