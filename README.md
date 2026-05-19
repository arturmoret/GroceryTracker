# Grocery Detection — Visión Clásica vs Deep Learning

Proyecto ABP UAB. Comparativa de dos pipelines de detección (clásico vs deep learning) sobre un subset de **MVTec D2S** filtrado a 20 productos compatibles con Mercadona.

> Briefing técnico completo en [PROYECTO.md](PROYECTO.md). **Léelo primero** si vas a tocar algo.

---

## Workflow Colab (recomendado — todo el cómputo fuera del PC local)

Pensado para máquinas lentas. Los notebooks de Colab cubren extracción de features, entrenamiento de SVM, hard negative mining e inferencia. **Tu PC solo sube/baja archivos a Google Drive.**

### Setup único en Drive

Crea `/MyDrive/grocery-detection/` con esta estructura:

```
/MyDrive/grocery-detection/
    raw/
        d2s_images_v*.tar.xz           ← descarga MVTec, acepta licencia
        d2s_annotations_v*.tar.xz
    processed/
        train.json                     ← generar local (ver "Pasos baratos")
        val.json
        test.json
        codebook.pkl
```

### Pasos baratos en local (1-5 min, opcionales en Colab)

1. **Splits filtrados** (`prepare_splits.py`):

    ```powershell
    uv run python scripts/prepare_splits.py
    ```

    Sube `data/processed/{train,val,test}.json` a `Drive/grocery-detection/processed/`.

2. **Codebook BoVW** (`train_codebook.py`):

    ```powershell
    uv run python scripts/train_codebook.py
    ```

    Sube `data/processed/codebook.pkl` a `Drive/grocery-detection/processed/`.

> Si tu PC no puede ni con esto, los notebooks de Colab incluyen celdas opcionales para correrlos allí.

### Pasos caros — Colab

Abre los notebooks **en orden** desde [colab.research.google.com](https://colab.research.google.com) → *File → Upload notebook* (o GitHub).

| Orden | Notebook | Tarda | Producto |
|---|---|---|---|
| 1 | [`notebooks/colab_build_features.ipynb`](notebooks/colab_build_features.ipynb) | ~2-5 h | `classical_features.npz` |
| 2 | [`notebooks/colab_train_svm.ipynb`](notebooks/colab_train_svm.ipynb) | ~5-15 min | `classical_svm.pkl` |
| 3 | [`notebooks/colab_hard_neg.ipynb`](notebooks/colab_hard_neg.ipynb) | ~2-5 h × rondas | features + svm actualizados |
| 4 | [`notebooks/colab_infer.ipynb`](notebooks/colab_infer.ipynb) | ~1-3 h | `predictions/classical_test.json` |

Cada notebook:

- Clona el repo, instala deps, monta Drive.
- Extrae D2S de Drive al disco local de Colab (rápido).
- Baja artifacts previos de Drive, ejecuta el script con **checkpoint cada 100 imgs**, sube outputs de vuelta a Drive.
- Si Colab desconecta: reabres el notebook, **Run All**, y reanuda donde quedó.

### Sincronizar al PC local

```powershell
# Bajar el JSON final de predicciones desde Drive a data local (manual o vía Drive desktop).
# Luego desde el repo:
git pull
```

---

## Workflow local (fallback)

Si prefieres correr todo en el PC sin Colab:

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

### 3. EDA + splits + codebook

```powershell
uv run jupyter notebook notebooks/00_dataset_eda.ipynb
uv run python scripts/prepare_splits.py
uv run python scripts/train_codebook.py
```

### 4. Pipeline clásico end-to-end

```powershell
uv run python scripts/run_classical_train.py     # build features + fit SVM (lento)
uv run python scripts/run_classical_hard_neg.py  # opcional, mining + refit
uv run python scripts/run_classical_infer.py     # predicciones sobre test
```

Todos soportan **checkpoint**: si interrumpes y relanzas, reanuda desde donde quedó. Para forzar reset usa `--rebuild-features`, `--reset` o `--rebuild` (ver `--help` de cada script).

Para wrap completo overnight: `.\scripts\run_overnight.ps1` (o `.sh`).

---

## Estructura del repo

Ver sección 8 de [PROYECTO.md](PROYECTO.md). Solo se materializan los módulos del hito actual; el resto se crea conforme avanza la hoja de ruta.

## Stack

- Python 3.11
- `uv` (gestión de venv + deps)
- OpenCV contrib (Selective Search, SIFT)
- scikit-learn (SVM χ² aproximada vía AdditiveChi2Sampler + LinearSVC, KMeans)
- PyTorch + ultralytics (YOLOv8s — H6)
- pycocotools (evaluación común — H7)

## Estado actual

- ✅ **H1**: esqueleto repo + `prepare_d2s.py` + `00_dataset_eda.ipynb`. 20 clases verificadas en D2S con ≥60 train / ≥60 val por clase.
- ✅ **H2**: `prepare_splits.py` + `01_class_selection.ipynb`. Splits filtrados en `data/processed/` listos para H3+.
- ✅ **H3**: componentes pipeline clásico (`classical/proposals.py`, `descriptors/{hog,sift,bovw}.py`) + `train_codebook.py` (K=300) + `02_classical_dev.ipynb`. Codebook entrenado en `data/processed/codebook.pkl`.
- ✅ **H4**: `run_classical_train.py` — extracción de features con checkpoint + AdditiveChi2 + LinearSVC OvR.
- ✅ **H5**: `run_classical_hard_neg.py` — mining loop con resume a nivel ronda + intra-ronda.
- ✅ **Colab workflow clásico**: `notebooks/colab_{build_features,train_svm,hard_neg,infer}.ipynb` + `scripts/colab_helper.py`. Todo H1-H5 ejecutable sin tocar el PC local.
- 🛠️ **H6 (DL, en marcha)**: scaffolding completo de YOLOv8s listo (`src/grocery_detection/deep/*`, `configs/deep_yolo.yaml`, `scripts/{prepare_yolo_dataset,run_deep_train,run_deep_infer}.py`, `notebooks/colab_yolo_{train,infer}.ipynb`). Llevado por compañero. Ver [`ONBOARDING_DL.md`](ONBOARDING_DL.md).
- ⏳ **H7-H10**: ver [PROYECTO.md sección 9](PROYECTO.md).
