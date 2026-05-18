"""MVTec D2S dataset preparation.

D2S requires accepting a license. This module does NOT auto-download.
It checks for the archives in `data/raw/` and extracts them into `data/d2s/`.
If the archives are missing, it prints clear instructions.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()

EXPECTED_ARCHIVES = [
    "d2s_images_v1.tar.xz",
    "d2s_annotations_v1.tar.xz",
]


def is_extracted(d2s_dir: Path) -> bool:
    if not d2s_dir.is_dir():
        return False
    images_dir = d2s_dir / "images"
    annotations_dir = d2s_dir / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        return False
    if not any(images_dir.iterdir()):
        return False
    if not any(annotations_dir.glob("*.json")):
        return False
    return True


def archives_present(raw_dir: Path) -> list[Path]:
    return [raw_dir / name for name in EXPECTED_ARCHIVES if (raw_dir / name).is_file()]


def print_download_instructions(raw_dir: Path) -> None:
    msg = (
        "[bold]MVTec D2S no está descargado.[/bold]\n\n"
        "D2S requiere aceptar la licencia. Pasos:\n\n"
        "1. Abrir: https://www.mvtec.com/company/research/datasets/mvtec-d2s\n"
        "2. Rellenar formulario, aceptar términos.\n"
        "3. Descargar:\n"
        "   - d2s_images_v1.tar.xz\n"
        "   - d2s_annotations_v1.tar.xz\n"
        f"4. Mover ambos archivos a:\n   [cyan]{raw_dir}[/cyan]\n"
        "5. Volver a ejecutar este script."
    )
    console.print(Panel(msg, title="Acción requerida", border_style="yellow"))


def extract_archive(archive: Path, dest: Path) -> None:
    console.print(f"Extrayendo [cyan]{archive.name}[/cyan] → [cyan]{dest}[/cyan]")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(dest)


def prepare(raw_dir: Path, d2s_dir: Path) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)

    if is_extracted(d2s_dir):
        console.print(f"[green]D2S ya extraído en {d2s_dir}.[/green]")
        return 0

    present = archives_present(raw_dir)
    if len(present) < len(EXPECTED_ARCHIVES):
        print_download_instructions(raw_dir)
        return 1

    d2s_dir.mkdir(parents=True, exist_ok=True)
    for archive in present:
        extract_archive(archive, d2s_dir)

    if is_extracted(d2s_dir):
        console.print(f"[green]D2S listo en {d2s_dir}.[/green]")
        return 0

    console.print(
        "[red]Extracción terminó pero la estructura esperada no aparece. "
        "Inspeccionar el contenido de data/d2s manualmente.[/red]"
    )
    return 2
