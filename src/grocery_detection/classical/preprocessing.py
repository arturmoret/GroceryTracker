"""Handcrafted image preprocessing for the classical pipeline.

Pipeline en este orden:

    1. Gray World white balance    -> corrige tinte de iluminacion
    2. CLAHE sobre canal L de Lab  -> iguala contraste local
    3. Bilateral denoising         -> reduce ruido conservando bordes

Justificacion del orden:
- White balance primero porque opera sobre la imagen tal cual viene de la
  camara. Si va despues de CLAHE, el contraste artificial confunde la
  estimacion de medias.
- CLAHE en segundo lugar para igualar contraste sobre colores ya neutros.
- Denoising al final porque CLAHE puede amplificar ruido en zonas oscuras y
  queremos limpiarlo antes de Selective Search / SIFT / HOG.

Todas las funciones operan en formato **BGR uint8** (compatible directo con
`cv2.imread`). No requieren deps mas alla de opencv + numpy.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    """Gray World white balance. Asume input BGR uint8."""
    if image.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibi {image.dtype}")
    img_f = image.astype(np.float32)
    mean_per_channel = img_f.reshape(-1, 3).mean(axis=0)
    mean_global = mean_per_channel.mean()
    scale = mean_global / np.maximum(mean_per_channel, 1e-6)
    return np.clip(img_f * scale, 0, 255).astype(np.uint8)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """CLAHE en el canal L de Lab. Asume input BGR uint8."""
    if image.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibi {image.dtype}")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def bilateral_denoise(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """Filtro bilateral (suaviza ruido preservando bordes). Asume BGR uint8."""
    if image.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibi {image.dtype}")
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


# ---------------------------------------------------------------------------
# Orquestador: lee la config y aplica los pasos activos
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "white_balance": True,
    "clahe": True,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": [8, 8],
    "denoising": True,
    "denoise_diameter": 7,
    "denoise_sigma_color": 50.0,
    "denoise_sigma_space": 50.0,
}


def preprocess(image: np.ndarray, cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Pipeline completo. `cfg` es el bloque `preprocessing` de classical.yaml.

    Si `cfg.enabled` es False, devuelve la imagen sin modificar.
    Cada paso individual puede desactivarse con su flag (white_balance,
    clahe, denoising) o ajustarse con sus parametros.
    """
    if cfg is None:
        cfg = {}
    merged = {**DEFAULTS, **cfg}
    if not merged["enabled"]:
        return image

    out = image
    if merged["white_balance"]:
        out = gray_world_white_balance(out)
    if merged["clahe"]:
        out = apply_clahe(
            out,
            clip_limit=float(merged["clahe_clip_limit"]),
            tile_grid_size=tuple(merged["clahe_tile_grid_size"]),
        )
    if merged["denoising"]:
        out = bilateral_denoise(
            out,
            diameter=int(merged["denoise_diameter"]),
            sigma_color=float(merged["denoise_sigma_color"]),
            sigma_space=float(merged["denoise_sigma_space"]),
        )
    return out
