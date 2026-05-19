"""Atomic file write helpers.

Pensado para destinos lentos o poco fiables (Google Drive FUSE en Colab,
NFS, etc.): escribir a `<path>.tmp` y renombrar al destino final hace que
el archivo "definitivo" aparezca de golpe. Si el proceso muere a mitad de
la escritura, el `.tmp` queda huérfano (basura) pero el archivo real
sigue intacto.

Si el destino no existía y la escritura se interrumpe, el archivo
simplemente no aparece. Mejor que un .npz/.json corrupto.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def atomic_savez_compressed(path: Path | str, **arrays: np.ndarray) -> None:
    """`np.savez_compressed` con escritura atómica (tmp + rename).

    Pasa los arrays como kwargs igual que `np.savez_compressed`:

        atomic_savez_compressed("foo.npz", X=X, y=y)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # `np.savez_compressed(path_str, ...)` añadiría .npz al nombre — usamos
    # file-object para evitar ese rename implícito.
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def atomic_write_json(path: Path | str, data: Any, *, ensure_ascii: bool = False) -> None:
    """`json.dump` con escritura atómica."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii)
    tmp.replace(path)


def atomic_write_pickle(path: Path | str, obj: Any, protocol: int | None = None) -> None:
    """`pickle.dump` con escritura atómica."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
    tmp.replace(path)
