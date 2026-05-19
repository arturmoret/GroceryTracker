# Onboarding — compañero del pipeline DL (H6)

Bienvenido. Este doc te pone al día en 10 minutos sobre el estado del proyecto y lo que te toca a ti.

## TL;DR

- Yo (Artur) llevo **toda la visión clásica** (H1-H5). Está terminado: extracción de features, SVM χ², hard negative mining, inferencia, todo reanudable y corriendo en Colab + Drive.
- Tú llevas **toda la parte de Deep Learning** (H6): **fine-tune de YOLOv8s** sobre el mismo conjunto de 20 clases, mismo split test, generar predicciones en el mismo formato (COCO results JSON).
- Después (H7) hay un framework de evaluación común que come las dos predicciones y genera la comparativa. Eso lo haré yo o lo haremos juntos según veamos.

**Lo crítico**: tu output tiene que ser **plug-and-play** con el framework de evaluación común. Por eso ya te dejé el scaffolding montado (configs, scripts, notebooks, módulos). Si lo respetas, no tienes que pelearte con compatibilidad.

---

## 1. Estado del proyecto

Estado completo en [`PROYECTO.md`](PROYECTO.md). Resumen para ti:

- **Dataset**: MVTec D2S, subset de 20 clases (ver `configs/classes.yaml`).
- **Splits**: `data/processed/train.json` / `val.json` / `test.json` en formato COCO. Categorías re-numeradas a 1..20 contiguos.
- **Tarea**: detección (bbox + clase).
- **Pipeline clásico**: terminado. Predicciones en `reports/predictions/classical_test.json`.
- **Pipeline DL (tu parte)**: scaffolding listo. **Modelo decidido: YOLOv8s** (pretrained COCO, fine-tune). Razón en PROYECTO.md §5.

### Las 20 clases

Subset diseñado para que haya **triadas/pares confusables** (4 manzanas, 3 pasta Reggia, 2 Coca-Cola, etc.) — el experimento académico es ver dónde el clásico se rompe y dónde DL gana. Detalle: PROYECTO.md §3.

IDs COCO van 1..20 en el orden de `configs/classes.yaml`. YOLO necesita IDs 0..19 → la conversión la hace `coco_to_yolo.py` automáticamente.

---

## 2. Lo que te he dejado scaffoldado

### Código

```
configs/
    deep_yolo.yaml                    # Hyperparams + augmentations
src/grocery_detection/
    data/coco_to_yolo.py              # COCO → YOLO label format (write txt files + split lists)
    deep/
        __init__.py
        train.py                      # wrapper alrededor de ultralytics.YOLO.train
        infer.py                      # wrapper de YOLO.predict
        export_coco.py                # YOLO Results → COCO results JSON
scripts/
    prepare_yolo_dataset.py           # convierte splits COCO → estructura YOLO
    run_deep_train.py                 # CLI: lanza el fine-tune
    run_deep_infer.py                 # CLI: predict + export COCO JSON
notebooks/
    colab_yolo_train.ipynb            # Colab: entrenamiento end-to-end
    colab_yolo_infer.ipynb            # Colab: inferencia + export
```

### Configs

`configs/deep_yolo.yaml` ya tiene defaults razonables: 100 épocas, batch 16, imgsz 640, SGD + cosine LR, augmentations específicas de dominio retail (HSV jitter, mosaic, mixup, fliplr=0.5, **flipud=0** porque los productos no se ven invertidos en uso real). Tunéalo si quieres.

---

## 3. Tu workflow (Colab)

Si no tienes GPU local, todo se hace desde Colab + Google Drive. **No necesitas tocar Windows/Linux**.

### Prerequisitos en Drive

Ya están todos los splits + raw del dataset en Drive (yo los subí cuando preparé H1-H5):

```
MyDrive/grocery-detection/
    raw/
        d2s_images_v*.tar.xz          ← yo lo subí
        d2s_annotations_v*.tar.xz
    processed/
        train.json                    ← yo lo generé
        val.json
        test.json
```

> Si no tienes acceso al Drive donde está esto, te lo comparto. Alternativamente puedes regenerar todo desde 0 corriendo las celdas opcionales de `colab_build_features.ipynb` (genera train/val/test/codebook).

### Paso 1 — `colab_yolo_train.ipynb`

1. Abre el notebook en Colab (vía GitHub: File → Open notebook → GitHub → `arturmoret/GroceryTracker`).
2. **Runtime → Change runtime type → GPU** (importante, sin GPU es inviable).
3. Run all.

Hace:

1. Clona el repo + instala deps (opencv-contrib, sklearn, ultralytics).
2. Monta tu Drive.
3. Extrae D2S a `/content/d2s` (rápido, ~5-10 min, una vez por sesión).
4. Symlinka `data/processed/` ↔ Drive (todas las escrituras caen en Drive).
5. Corre `prepare_yolo_dataset.py` (idempotente, ~10 s) → labels YOLO en `data/d2s/labels/` + `dataset.yaml`.
6. Corre `run_deep_train.py` → entrena 100 épocas (o early stopping antes), guarda `yolo_runs/yolov8s_d2s20/weights/best.pt` y copia a `data/processed/yolo_best.pt`.

Tiempo en T4 free: **~2-3 horas** para 100 épocas. Suele cortar antes con early stopping (patience=20).

**Si Colab desconecta a mitad**: reabres el notebook, Run All, y descomenta la celda 9b para reanudar desde `last.pt` (que está en Drive).

### Paso 2 — `colab_yolo_infer.ipynb`

Mismo patrón: Run All. Aplica `yolo_best.pt` sobre el split test, exporta `MyDrive/grocery-detection/predictions/yolo_test.json` en formato COCO results.

Tiempo: **~10-20 min** en T4.

---

## 4. Tu workflow (local con GPU)

Si tienes GPU local NVIDIA con CUDA, vas más rápido y sin desconexiones. Equivalente al workflow de [`RUN_LOCAL_FEDORA.md`](RUN_LOCAL_FEDORA.md) pero solo lo del pipeline DL:

```bash
git clone https://github.com/arturmoret/GroceryTracker.git
cd GroceryTracker

# Linux:
bash scripts/setup.sh
# Windows:
.\scripts\setup.ps1

# Descarga manual de D2S a data/raw/ (acepta licencia MVTec).
uv run python scripts/prepare_d2s.py     # extrae D2S
uv run python scripts/prepare_splits.py  # genera train/val/test JSONs (rápido)
uv run python scripts/prepare_yolo_dataset.py    # YOLO labels + dataset.yaml

# Entrenar (necesita GPU NVIDIA + CUDA):
uv run python scripts/run_deep_train.py

# Inferencia:
uv run python scripts/run_deep_infer.py
```

Outputs locales en `data/processed/yolo_best.pt` y `reports/predictions/yolo_test.json`.

---

## 5. Contrato del output (importante)

Tu pipeline tiene que producir exactamente este formato para que el eval común (H7) lo consuma sin fricción:

**Archivo**: `reports/predictions/yolo_test.json` (en Colab → `MyDrive/grocery-detection/predictions/yolo_test.json`).

**Formato**: COCO results JSON (lista de detecciones, sin envoltorio adicional):

```json
[
  {
    "image_id": 123,
    "category_id": 5,
    "bbox": [x, y, w, h],
    "score": 0.87
  },
  ...
]
```

- `image_id`: igual al `id` del `image` en `test.json`.
- `category_id`: 1..20 (orden de `configs/classes.yaml`). **No 0..19** — el `export_coco.py` ya hace el offset por ti.
- `bbox`: `[x, y, w, h]` en píxeles absolutos (NO normalizados), `xywh` no `xyxy`. Lo gestiona `export_coco.py`.
- `score`: confianza 0-1.

El módulo `src/grocery_detection/deep/export_coco.py` ya hace esta conversión correctamente. Si lo respetas, output garantizado-compatible.

---

## 6. Reglas del juego (no romper lo del clásico)

Para que no haya fricción entre nuestros pipelines:

### NO toques

- `src/grocery_detection/classical/`  (todo)
- `scripts/run_classical_*`
- `scripts/colab_helper.py` (helpers compartidos; si necesitas algo nuevo, **añade** funciones, no edites las existentes)
- `notebooks/colab_build_features.ipynb`, `colab_train_svm.ipynb`, `colab_hard_neg.ipynb`, `colab_infer.ipynb`
- `configs/classes.yaml`  (las 20 clases están cerradas — si propones cambio, hablamos primero)
- `configs/classical.yaml`
- `configs/data.yaml`  (paths del dataset; añade campos si los necesitas, no quites)
- `data/processed/{train,val,test}.json`  (splits cerrados, reproducible con seed=42)
- `data/processed/classical_*`  (artifacts del pipeline clásico)
- `reports/predictions/classical_test.json`  (predicciones mías)

### SÍ es tuyo

- `configs/deep_yolo.yaml`
- `src/grocery_detection/deep/*`
- `src/grocery_detection/data/coco_to_yolo.py`  (puedes mejorar si encuentras bugs)
- `scripts/prepare_yolo_dataset.py`, `run_deep_train.py`, `run_deep_infer.py`
- `notebooks/colab_yolo_train.ipynb`, `colab_yolo_infer.ipynb`
- `data/d2s/labels/*` (lo genera prepare_yolo_dataset.py)
- `data/processed/yolo_runs/`, `data/processed/yolo_best.pt`, `data/processed/yolo/`
- `reports/predictions/yolo_test.json`

### Git workflow

- Trabaja en `main` o en una rama `dl/<feature>` — como prefieras.
- Antes de push, `git pull --rebase` para no chocar con mis commits.
- Commits descriptivos en español o inglés, lo que veas.

---

## 7. Lecturas recomendadas (en orden)

1. **[`PROYECTO.md`](PROYECTO.md)** — briefing completo del proyecto. Sección 5 es la tuya.
2. **[`RUN_COLAB.md`](RUN_COLAB.md)** — workflow Colab + Drive (referencia general, aunque hablo del clásico).
3. **[`configs/classes.yaml`](configs/classes.yaml)** — las 20 clases con razonamiento de triadas confusables.
4. **[`configs/deep_yolo.yaml`](configs/deep_yolo.yaml)** — hyperparams de YOLOv8s. Sin tunear con datos reales aún — vas a tener que iterar.
5. **`src/grocery_detection/deep/train.py`** — wrapper. Léelo entero, son ~70 líneas.
6. **`src/grocery_detection/deep/export_coco.py`** — esta es la pieza crítica para garantizar compatibilidad de output.
7. **[`reports/H1-H5_overview.pdf`](reports/H1-H5_overview.pdf)** y **[`reports/onboarding_colaboradores.pdf`](reports/onboarding_colaboradores.pdf)** — PDFs generados con visualizaciones de lo que ya hay hecho.

---

## 8. Cosas que te tocará iterar (y consejos)

### Hyperparams de YOLOv8s

Los defaults en `configs/deep_yolo.yaml` son sensatos pero **no están afinados** sobre nuestros datos. Iterar sobre:

- `epochs` — 100 es generoso; con early stopping (`patience=20` sobre mAP@0.5 val) se corta antes. Si converge en 30-40 épocas, baja para acelerar.
- `freeze_backbone_epochs: 5` — opcional. Útil cuando los datos de fine-tune son pocos vs el dominio pretrained.
- `lr0: 0.01` — clásico SGD. Si diverges en las primeras épocas, baja a 0.001.
- `augmentations.*` — el hsv_v=0.4 es agresivo (productos pueden quedar muy oscuros/quemados). Si ves fallos sistemáticos por iluminación, súbelo. Si ves overfit, baja mosaic/mixup.

### Confusables esperados

Las triadas confusables (4 manzanas, 3 pasta Reggia, 2 Coca-Cola, etc.) son donde YOLOv8s debe brillar sobre el clásico. La matriz de confusión te dirá si lo está haciendo.

Atención específica a:
- **Manzanas**: forma idéntica, distinguen color/textura. Pertinente para juzgar si YOLO usa color signaling bien.
- **Pasta Reggia**: packaging azul idéntico, distinta forma de pasta a través del plástico. Reto fino.
- **Coca-Cola light vs regular**: misma lata, diferencia es solo el color principal (rojo vs plata). Si YOLO falla esto, algo va mal con augmentations HSV.

### Reproducibilidad

`seed=42` global en todos los scripts. Si quieres comparar dos runs, fija el seed y los hyperparams.

Ultralytics no garantiza reproducibilidad bit-a-bit con CUDA por algunos kernels no-determinísticos. mAP variará ±0.5 entre runs idénticos. Normal.

### Métricas durante entreno

ultralytics imprime mAP@0.5 y mAP@0.5:0.95 por época. Guarda los TensorBoard logs (en `data/processed/yolo_runs/yolov8s_d2s20/`). Útiles para el informe.

---

## 9. Cuándo decir "listo"

Tu parte (H6) está terminada cuando:

- [ ] `yolo_best.pt` entrenado y subido a Drive (`processed/yolo_best.pt`).
- [ ] `yolo_test.json` generado en Drive (`predictions/yolo_test.json`) con todas las detecciones sobre el split test.
- [ ] Hyperparams iterados (al menos 1 round de tuning sobre los defaults).
- [ ] Métricas finales documentadas en algún notebook tipo `notebooks/06_yolo_results.ipynb` (créalo): mAP@0.5, mAP@0.5:0.95, matriz confusión 20×20, ejemplos cualitativos.

Tras eso, H7 (eval framework común) compara tu yolo_test.json contra mi classical_test.json sobre el mismo test.json.

---

## 10. Si te encallas

1. Revisa los logs del script — todos imprimen `[setup]`, `[infer]`, etc. con detalles.
2. Verifica que `data/processed/yolo/dataset.yaml` apunta a paths existentes (`cat dataset.yaml`).
3. Si ultralytics se queja de imágenes corruptas, recorre `data/d2s/images/` con `find . -name "*.jpg" -size 0` para detectar archivos vacíos.
4. Si nada va, escríbeme. No te quemes.

---

¡A meterle!
