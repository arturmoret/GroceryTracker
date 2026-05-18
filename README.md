# Grocery Detection — Visión Clásica vs Deep Learning

Proyecto ABP UAB. Comparativa de dos pipelines de detección (clásico vs deep learning) sobre un subset de **MVTec D2S** filtrado a 20 productos compatibles con Mercadona.

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

El script instala `uv` (si falta), sincroniza dependencias y crea `data/raw|d2s|processed`.

### 2. Descargar MVTec D2S manualmente

D2S requiere aceptar la licencia → **no se puede descargar automáticamente**.

1. Abrir https://www.mvtec.com/company/research/datasets/mvtec-d2s
2. Rellenar formulario, aceptar términos.
3. Descargar:
   - `d2s_images_v*.tar.xz` (~6 GB)
   - `d2s_annotations_v*.tar.xz` (~40 MB)
4. Mover ambos a `data/raw/`.
5. Extraer:

```powershell
uv run python scripts/prepare_d2s.py
```

### 3. (H1) EDA — verificar las 20 clases

```powershell
uv run jupyter notebook notebooks/00_dataset_eda.ipynb
```

Reporta categorías reales de D2S, conteos por clase y validación contra `configs/classes.yaml` (mínimo ≥60 train / ≥60 test).

### 4. (H2) Generar splits filtrados

```powershell
uv run python scripts/prepare_splits.py
```

Filtra `D2S_training` y `D2S_validation` a las 20 clases y divide val en val (30%) + test (70%) estratificado por clase dominante. Output: `data/processed/train.json`, `val.json`, `test.json` con IDs de categoría renumerados 1..20. Reproducible (seed=42).

Documentación de criterios + visualizaciones en `notebooks/01_class_selection.ipynb`.

### 5. (H3) Entrenar el codebook BoVW

```powershell
uv run python scripts/train_codebook.py
```

Muestrea SIFTs de 300 imágenes train (cap 400 desc/img), entrena MiniBatchKMeans K=1500 y guarda `data/processed/codebook.pkl`. ~1-3 min en CPU. Reproducible (seed=42).

Demo end-to-end de cada componente del pipeline clásico (Selective Search → HOG → SIFT → BoVW → feature concatenado) en `notebooks/02_classical_dev.ipynb`.

## Estructura del repo

Ver sección 8 de [PROYECTO.md](PROYECTO.md). Solo se materializan los módulos del hito actual; el resto se crea conforme avanza la hoja de ruta.

## Stack

- Python 3.11
- `uv` (gestión de venv + deps)
- OpenCV contrib (Selective Search, SIFT)
- scikit-learn (SVM, k-means)
- PyTorch + ultralytics (YOLOv8s)
- pycocotools (evaluación)

## Estado actual

- ✅ **H1**: esqueleto repo + `prepare_d2s.py` + `00_dataset_eda.ipynb`. 20 clases verificadas en D2S con ≥60 train / ≥60 val por clase.
- ✅ **H2**: `prepare_splits.py` + `01_class_selection.ipynb`. Splits filtrados en `data/processed/` listos para H3+.
- ✅ **H3**: componentes pipeline clásico (`classical/proposals.py`, `descriptors/{hog,sift,bovw}.py`) + `train_codebook.py` + `02_classical_dev.ipynb`. Codebook entrenado en `data/processed/codebook.pkl`.
- ⏳ **H4-H10**: ver [PROYECTO.md sección 9](PROYECTO.md).
