# Grocery Detection — Visión Clásica vs Deep Learning

Proyecto ABP UAB. Comparativa de dos pipelines de detección (clásico vs deep learning) sobre un subset de **MVTec D2S** filtrado a productos disponibles en Mercadona.

> Briefing técnico completo en [PROYECTO.md](PROYECTO.md). **Léelo primero** si vas a tocar algo.

## Quickstart (PC nuevo, recién clonado)

### 1. Setup automático

**Windows (PowerShell)**:

```powershell
.\scripts\setup.ps1
```

**Linux / macOS**:

```bash
bash scripts/setup.sh
```

El script:

- Instala `uv` si no está presente.
- Sincroniza dependencias (`uv sync`).
- Crea estructura `data/raw/`, `data/d2s/`, `data/processed/`.

### 2. Descargar MVTec D2S manualmente

D2S requiere aceptar la licencia → **no se puede descargar automáticamente**.

1. Abrir https://www.mvtec.com/company/research/datasets/mvtec-d2s
2. Rellenar formulario, aceptar términos.
3. Descargar los dos archivos:
   - `d2s_images_v1.tar.xz`
   - `d2s_annotations_v1.tar.xz`
4. Mover ambos a `data/raw/`.
5. Extraer ejecutando:

```powershell
uv run python scripts/prepare_d2s.py
```

### 3. EDA — verificar las 14 clases

Abrir el notebook EDA (Jupyter o VS Code) y correr todas las celdas:

```powershell
uv run jupyter notebook notebooks/00_dataset_eda.ipynb
```

El notebook reporta:

- Listado de todas las clases reales en D2S (para verificar los nombres exactos).
- Conteos por clase (train, val, test).
- Validación de las 14 clases objetivo declaradas en `configs/classes.yaml` contra el mínimo (≥150 train / ≥30 test).
- Visualización de ejemplos por clase.

Si alguna clase queda corta de muestras, ajustar `configs/classes.yaml` y volver a correr el notebook.

## Estructura del repo

Ver sección 8 de [PROYECTO.md](PROYECTO.md). En H1 sólo se han materializado los módulos necesarios para EDA; el resto se crea en hitos siguientes.

## Stack

- Python 3.11
- `uv` (gestión de venv + deps)
- OpenCV contrib (Selective Search, SIFT)
- scikit-learn (SVM, k-means)
- PyTorch + ultralytics (YOLOv8s)
- pycocotools (evaluación)

## Estado actual

- ✅ H1 (en progreso): esqueleto repo + descarga/extracción D2S + EDA notebook.
- ⏳ H2-H10: pendiente. Ver PROYECTO.md sección 9 para hoja de ruta.
