# Correr el pipeline clásico (H1-H5) en Fedora Linux

Guía de copia-pega para arrancar el pipeline clásico en un PC Fedora sin pasar por Colab. Cubre instalación, descarga del dataset y ejecución end-to-end con resume automático.

> Asume Fedora 38+ con Python 3.11 disponible. Si tu Python es más antiguo, no pasa nada — `uv` se encarga.

---

## 0. Paquetes del sistema (una sola vez)

```bash
sudo dnf install -y git curl tar xz mesa-libGL
```

- `mesa-libGL` lo necesita `opencv-contrib-python` para no petar al `import cv2`.
- `tar xz` para extraer los `.tar.xz` de D2S (`xz` viene casi siempre, no estorba).

## 1. Clonar el repo

```bash
cd ~              # o donde quieras
git clone https://github.com/arturmoret/GroceryTracker.git
cd GroceryTracker
```

## 2. Setup (instala `uv` + dependencias)

```bash
bash scripts/setup.sh
```

El script:

- Instala `uv` (gestor de venv + paquetes) si no está en PATH.
- Corre `uv sync --all-extras` → crea `.venv/` con todas las deps (numpy, opencv-contrib, scikit-learn, scikit-image, ultralytics, pycocotools, jupyter…).
- Crea `data/raw/`, `data/d2s/`, `data/processed/`.

Salida esperada al final:

```
==> Setup completado.
Siguientes pasos: ...
```

Si falla la instalación de `uv` y no aparece en `PATH`, abre una shell nueva y vuelve a correr el comando — el instalador suele añadir `~/.local/bin` al PATH al inicio de la sesión siguiente.

## 3. Descargar MVTec D2S manualmente

D2S requiere aceptar licencia → **no auto-download posible**.

1. Abrir: https://www.mvtec.com/company/research/datasets/mvtec-d2s
2. Aceptar términos, descargar los dos archivos (~6 GB + ~40 MB):
   - `d2s_images_v*.tar.xz`
   - `d2s_annotations_v*.tar.xz`
3. Mover ambos a `data/raw/`:

    ```bash
    mv ~/Downloads/d2s_images_v*.tar.xz data/raw/
    mv ~/Downloads/d2s_annotations_v*.tar.xz data/raw/
    ```

4. Extraer:

    ```bash
    uv run python scripts/prepare_d2s.py
    ```

    Deja `data/d2s/images/` (~21k jpgs) y `data/d2s/annotations/` (3 JSONs).

## 4. Verificar EDA (H1)

```bash
uv run jupyter notebook notebooks/00_dataset_eda.ipynb
```

Ejecutar todas las celdas. Reporta conteos por clase y valida las 20 clases definidas en `configs/classes.yaml`.

> Si prefieres VS Code: ábrelo en el repo, instala la extensión Python + Jupyter, selecciona el kernel `.venv` (el que crea `uv sync`) y abre el notebook desde el explorador.

## 5. Generar splits filtrados (H2)

```bash
uv run python scripts/prepare_splits.py
```

Produce:

- `data/processed/train.json`  (train filtrado a 20 clases, IDs renumerados 1..20)
- `data/processed/val.json`
- `data/processed/test.json`

Reproducible con `seed=42`. ~30s.

## 6. Entrenar codebook BoVW (H3)

```bash
uv run python scripts/train_codebook.py
```

Sampling SIFT sobre 300 imágenes train + KMeans K=300 → `data/processed/codebook.pkl`. ~3-5 min en CPU decente.

## 7. Pipeline clásico end-to-end (H4 + H5)

Tres scripts. Todos con checkpoint reanudable.

### H4 — Build features + SVM fit

```bash
uv run python scripts/run_classical_train.py
```

Qué hace:

1. **Feature extraction** sobre cada imagen train: Selective Search → HOG + SIFT + BoVW. Vuelca checkpoint a `data/processed/classical_features.npz` cada 100 imgs.
2. **SVM χ²** vía `AdditiveChi2Sampler + LinearSVC OvR`. Guarda en `data/processed/classical_svm.pkl`.

Tiempo estimado: **2-4 h en CPU decente**, según procesador y disco.

Si se corta (terminal cerrada, OOM, lo que sea), **vuelve a lanzar el mismo comando** y reanuda donde quedó.

Flags útiles:

- `--rebuild-features` → borra el checkpoint y vuelve a empezar.
- `--skip-svm-fit` → solo construye `classical_features.npz` y termina (útil si vas a entrenar la SVM aparte).

### H5 — Hard negative mining (opcional)

```bash
uv run python scripts/run_classical_hard_neg.py
```

Mining loop con `hard_negative.rounds` rondas (default 2). Por defecto en `configs/classical.yaml`:

```yaml
hard_negative:
  rounds: 2
  ...
```

Resume a 2 niveles:

- Por ronda: `.hardneg_state.json` registra rondas completadas.
- Dentro de ronda: checkpoint cada 100 imgs en `classical_features.hardneg_partial.npz`.

Tiempo: similar a build features × nº rondas.

Flags:

- `--reset` → borra state + partial, arranca desde la ronda 1.

### Inferencia sobre test

```bash
uv run python scripts/run_classical_infer.py
```

Recorre `test.json`, aplica el detector, escribe predicciones COCO JSON a `reports/predictions/classical_test.json`. Vuelca cada 100 imgs.

Resume: si el JSON ya existe, salta los `image_id` ya predichos.

Flags:

- `--rebuild` → borra el JSON y empieza de cero.
- `--limit N` → solo las primeras N imgs (debug).

### Atajo nocturno

Encadena los tres con logs timestamped:

```bash
bash scripts/run_overnight.sh
```

Output va a `logs/<timestamp>_<stage>.log`. Si una etapa falla, las siguientes no se ejecutan.

## 8. Dónde quedan los artifacts

```
data/processed/
    train.json
    val.json
    test.json
    codebook.pkl
    classical_features.npz        ← X, y, processed_ids (checkpoint H4)
    classical_svm.pkl             ← clasificador entrenado
    classical_features.hardneg_partial.npz  ← intra-ronda H5 (efímero)
    .hardneg_state.json           ← rondas completadas

reports/predictions/
    classical_test.json           ← predicciones del detector clásico
```

## 9. Comprobaciones rápidas

Antes de tirarte a H4, verifica que la maquinaria está cableada:

```bash
uv run python scripts/test_classical_tiny.py
```

Entrena con 5 imgs + infiere sobre 3. ~1-2 min. Si pasa, todo el pipeline está OK end-to-end.

## 10. Notebooks de visualización (no obligatorios)

Para inspeccionar resultados intermedios:

```bash
uv run jupyter notebook notebooks/
```

Notebooks ya generados al hacer el work en casa:

- `00_dataset_eda.ipynb` — EDA de D2S y validación de clases (H1).
- `01_class_selection.ipynb` — criterios + visualización de las 20 clases (H2).
- `02_classical_dev.ipynb` — demo de cada componente del pipeline clásico.
- `03_training_visualization.ipynb` — proposals, labels, features ya extraídas.
- `04_classical_results.ipynb` — métricas + matriz confusión sobre test.
- `05_preprocessing_viz.ipynb` — efecto del preprocesado (WB+CLAHE+denoise).

## 11. Gotchas conocidas

- **`ImportError: libGL.so.1`** al `import cv2` → `sudo dnf install -y mesa-libGL`.
- **OOM** durante SVM fit con `n_jobs=-1` → en `configs/classical.yaml`, `classifier.n_jobs` ya es 1 por defecto (modo memoria-mínima). Si tienes ≥16 GB libres, súbelo a `-1` para que vaya más rápido.
- **`opencv-python` conflictúa con `opencv-contrib-python`** → pyproject ya pide solo contrib. Si por error tienes ambos: `uv pip uninstall opencv-python opencv-python-headless`.
- **Selective Search lentísimo** → ya viene en `mode: fast` y `max_per_image: 300`. Bajar `proposals.max_side` de 640 a 480 acelera ~30% sacrificando recall un poco.

## 12. Si quieres pasar artifacts a/desde Colab

Los `.pkl` y `.npz` generados aquí son compatibles con los notebooks Colab (`colab_train_svm.ipynb`, `colab_infer.ipynb`). Subes los archivos a `Drive/grocery-detection/processed/` y los notebooks los recogen.

Inversamente, si entrenaste en Colab y quieres seguir local: baja los `.pkl`/`.npz` a `data/processed/` y los scripts los reusan.

---

**TL;DR** comandos de un solo bloque (asumiendo D2S ya descargado en `data/raw/`):

```bash
bash scripts/setup.sh
uv run python scripts/prepare_d2s.py
uv run python scripts/prepare_splits.py
uv run python scripts/train_codebook.py
uv run python scripts/test_classical_tiny.py     # smoke test
bash scripts/run_overnight.sh                    # H4 + H5 + inferencia con logs
```
