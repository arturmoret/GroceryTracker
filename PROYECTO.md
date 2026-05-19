# Proyecto ABP — Detección de productos en MVTec D2S: Visión Clásica vs Deep Learning

> **Briefing para futuras sesiones de Claude Code.** Estado a fecha del último commit. Leer entero antes de tocar código.

---

## 0. Contexto del proyecto

- **Estudiante**: Artur Moret, Escola d'Enginyeria (UAB).
- **Asignatura**: ABP de Visión por Computador.
- **Modalidad**: Individual.
- **Hardware**: sin GPU local. Entrenamiento en **Google Colab free / Kaggle / segundo PC Fedora**.
- **Entregables**:
  - Código + repositorio limpio.
  - Informe extenso con metodología y resultados.
  - Presentación.
  - Notebooks de desarrollo.
  - Demo webcam en vivo: nice-to-have, no obligatorio.

## 1. Objetivo

Diseñar, implementar y evaluar **dos pipelines paralelos** que detecten el mismo conjunto de productos:

- **Pipeline A — Visión por Computador clásica** (sin deep learning).
- **Pipeline B — Deep Learning** (YOLOv8s).

Ambos pipelines reciben las mismas imágenes, predicen bbox + clase sobre el mismo conjunto de clases y se evalúan con las mismas métricas sobre el mismo split de test. **El verdadero entregable académico es la comparativa cuantitativa y cualitativa entre paradigmas**, no batir SOTA.

**Caso de uso**: usuario muestra un producto a una webcam y el sistema lo detecta + clasifica. **No es detección en estantería real de supermercado** — escena típica = uno o pocos productos sobre superficie clara.

## 2. Decisiones cerradas

| Decisión | Valor | Razón |
|---|---|---|
| Dataset | **MVTec D2S** (público) | 60 SKUs específicos europeos, bbox + máscaras, train con objetos sueltos sobre superficie clara (~webcam-like), val con escenas cluttered. Académicamente sólido. |
| Tarea | **Detección (bbox + clase)** | Justifica mAP/IoU. El sistema dibuja caja alrededor del producto y lo clasifica. |
| Nº de clases | **20** | Subset de D2S elegido para maximizar diversidad y triadas confusables (ver §3). |
| Preprocesado | **Gray World WB + CLAHE Lab + Bilateral denoise** | Handcrafted, ON por defecto. Aplica antes de Selective Search + features. |
| Pipeline A | **Selective Search + (HOG + SIFT/BoVW) + AdditiveChi2Sampler + LinearSVC OvR + Hard Neg Mining** | Detección clásica honesta sin caer en heurísticas de color. |
| Pipeline B | **YOLOv8s** (descartado Faster R-CNN) | Decisión del alumno. Simplicidad sobre rigor adicional. |
| Robustez | **Sí incluida** (pendiente H8) | Test perturbado (ruido, blur, iluminación). Comparativa A vs B. |
| Gestión deps | **`uv`** | Venv + lockfile + install rápido. Un solo `uv sync`. |
| Configs | **YAML + `argparse`** | Sin Hydra. Defaults en YAML, overrides básicos por CLI. |
| Ejecución cara | **Google Colab + Drive** (workflow primario) | PC del alumno demasiado lento. Pipeline H4-H5+infer corren en Colab CPU con checkpoint reanudable cada 100 imgs. |

## 3. Las 20 clases (verificadas en D2S, EDA H1)

Subset de MVTec D2S elegido para diversidad de forma + triadas/pares confusables que enriquezcan el análisis de errores. Nombres exactos de la categoría D2S (`name` en `categories` del JSON COCO):

**Manzanas (4 — color/textura distintos, misma forma)**:
1. `apple_braeburn_bundle`
2. `apple_golden_delicious`
3. `apple_granny_smith`
4. `apple_red_boskoop`

**Pasta Reggia (3 — mismo packaging azul, distinta forma visible)**:
5. `pasta_reggia_elicoidali`
6. `pasta_reggia_fusilli`
7. `pasta_reggia_spaghetti`

**Coca-Cola (2 — misma lata, color rojo vs plata)**:
8. `coca_cola_05`
9. `coca_cola_light_05`

**Corny (2 — misma marca/forma, distinto sabor)**:
10. `corny_nussvoll`
11. `corny_schoko_banane`

**Uvas (2 — verde vs morada)**:
12. `grapes_green_sugraone_seedless`
13. `grapes_sweet_celebration_seedless`

**Tomates (2 — variantes muy similares)**:
14. `vine_tomatoes`
15. `roma_vine_tomatoes`

**Clementinas (2 — presentación distinta: malla vs sueltas)**:
16. `clementine`
17. `clementine_single`

**Producto singleton (3 — formas claramente distintas, anchor de control)**:
18. `carrot`
19. `cucumber`
20. `kiwi`

**Triadas/pares confusables intencionales**: 4 manzanas, 3 pasta Reggia, 2 Coca-Cola, 2 Corny, 2 uvas, 2 tomates, 2 clementinas.

**Hipótesis comparativa**: el pipeline clásico va a confundir variantes dentro de cada triada/par (HOG falla con frutas — forma similar; BoVW falla con pasta — packaging idéntico, contenido visible solo por translucencia). YOLOv8s debería distinguir mejor. **Esa diferencia ES el resultado interesante del proyecto.**

**Mínimos verificados** (`configs/classes.yaml`):
- train: ≥60 muestras por clase (rango real ~60-270).
- test: ≥60 muestras por clase (rango real ~90-360 antes del split val/test 30/70).

## 4. Pipeline A — Visión por Computador Clásica (implementado, H1-H5)

### 4.1. Arquitectura end-to-end

```
imagen
  → Preprocesado (Gray World WB → CLAHE Lab → Bilateral denoise)
  → Selective Search "fast" (resize max 640px, ≤300 proposals)
  → por cada proposal: HOG (64×64 gris) + SIFT/BoVW(K=300) → concatenar [HOG | BoVW]
  → AdditiveChi2Sampler (sample_steps=2 en Colab / 1 en local) + LinearSVC OvR (20 SVMs binarios)
  → por proposal: argmax score sobre clases target + threshold sobre decision_function
  → NMS por clase (IoU > 0.5)
  → bboxes finales con etiqueta y confianza
```

### 4.2. Componentes técnicos

#### Preprocesado handcrafted (`classical/preprocessing.py`)

Aplicado a CADA imagen antes de Selective Search + feature extraction. Asegura consistencia train↔inferencia.

1. **Gray World White Balance** — corrige tinte de iluminación (`mean_per_channel → scale`).
2. **CLAHE** sobre canal L de Lab — iguala contraste local (`clipLimit=2.0`, `tileGridSize=(8,8)`).
3. **Bilateral filter** — suaviza ruido preservando bordes (`d=7`, `sigmaColor=50`, `sigmaSpace=50`).

Orden justificado: WB primero (opera sobre la imagen tal cual), CLAHE sobre colores ya neutros, denoise al final porque CLAHE amplifica ruido en zonas oscuras.

Activable/desactivable globalmente o por paso en `configs/classical.yaml: preprocessing`.

#### Selective Search (`classical/proposals.py`)

Region proposal sin clase. `cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()`.

- Modo `fast` (~1000 proposals) o `quality` (~2000).
- Cap a 300 por imagen tras Selective Search (orden = score interno SS).
- Resize previo a max 640px lado largo (`resize_for_proposals`); proposals se reescalan al tamaño original (`scale_proposals`).

#### HOG (`classical/descriptors/hog.py`)

`skimage.feature.hog` sobre crop redimensionado a **64×64** en escala de grises.

- 9 orientaciones, celdas 8×8 px, bloques 2×2 celdas, normalización L2-Hys.
- Dim resultante: 1764 (= `hog_dim(64,64)`).
- Funciona bien con objetos rígidos (cajas, latas). Sufre con bolsas deformables.

#### SIFT (`classical/descriptors/sift.py`)

`cv2.SIFT_create()`. Singleton cacheado para no re-instanciar.

- Detect + descriptor en escala de grises.
- Vector 128-D por keypoint.
- **Optimización clave**: SIFT se extrae UNA VEZ sobre la imagen completa; en `features.py` los keypoints se filtran por bbox de proposal. ~10× speedup vs ejecutar SIFT por crop.

#### BoVW (`classical/descriptors/bovw.py`)

Codebook = `sklearn.cluster.KMeans` (NO `MiniBatchKMeans` — cuelgues silenciosos en algunas builds Windows).

- **K=300** centroides (no 1500). Entrenado por `scripts/train_codebook.py` sobre SIFTs de 300 imgs train (cap 400 desc/img).
- `init='random'` (más rápido que k-means++).
- `max_iter=30`, `n_init=1`.
- Encoding (`encode_bovw`): histograma bincount sobre asignaciones, L2 normalizado.
- Dim: 300.

#### Feature final por proposal (`classical/features.py`)

Concatenación `[HOG (1764) | BoVW (300)]` = vector 2064-D, clip ≥0, L2-normalizado. Compatible con AdditiveChi2Sampler (requiere no-negativo).

#### Clasificador (`classical/classifier.py`)

Aproximación Vedaldi-Zisserman al kernel χ². Más rápida y memoria-eficiente que `SVC(kernel='precomputed')` con matriz Gram completa.

1. **AdditiveChi2Sampler** (`sklearn.kernel_approximation`) con `sample_steps=1` (local, memoria mínima) o `sample_steps=2` (Colab, calidad mejor — dim ×3).
2. **OneVsRestClassifier(LinearSVC)** sobre la expansión.
3. `n_jobs=1` por defecto (memoria-safe en Mac 8GB; cada worker copia la matriz expandida). `n_jobs=-1` solo si ≥16 GB libres (Colab tira de esto).
4. `C=1.0`, `dual='auto'`, `max_iter=5000`.

#### Etiquetado (`classical/labeling.py`)

Para construir el training set:
- `BACKGROUND = 0`. Target classes = 1..20 (renumeradas en `prepare_splits.py`).
- IoU≥0.5 con alguna GT → positivo, label = clase de la GT.
- IoU<0.3 con TODAS las GT → negativo (BACKGROUND).
- Entre 0.3 y 0.5 → ignorado (`-1`, no se entrena con él).

#### NMS (`classical/nms.py`)

NMS independiente por clase, IoU>0.5, cap `top_k_per_image=100`.

#### Hard Negative Mining (`classical/hard_negative.py`)

Para cada imagen train:
1. Predice con SVM actual sobre proposals.
2. FP = proposals con `score > fp_score_thresh` Y `IoU < neg_iou` con todas las GT.
3. Top-K FPs (ordenados por score descendente) → al training set como BACKGROUND.
4. Re-fit SVM.
5. Repite por `hard_negative.rounds` rondas (default 2).

Estado persistido en `.hardneg_state.json` (completed_rounds) y checkpoint parcial en `classical_features.hardneg_partial.npz`.

### 4.3. Checkpoint reanudable (todos los stages)

Cada script soporta resume automático tras corte (Colab muere, kernel cae, OOM):

- **`build_training_features`**: vuelca `data/processed/classical_features.npz` con `X`, `y`, `processed_image_ids` cada 100 imgs nuevas. Al relanzar, salta image_ids ya en el set.
- **`mine_hard_negatives`**: análogo, vuelca `classical_features.hardneg_partial.npz`. + state JSON con rondas completadas → salta a la siguiente.
- **`run_classical_infer`**: flush incremental del JSON cada 100 imgs. Al relanzar, carga el JSON existente, marca `image_id`s ya predichos, los salta.

`checkpoint_every: 100` configurable en `classical.yaml`. Flags `--rebuild-features`, `--reset`, `--rebuild` para forzar desde cero.

### 4.4. Honestidad académica

Pipeline A va a perder en mAP contra YOLOv8s. **Eso ES el resultado interesante**: cuantificar la brecha y entender dónde el clásico se rompe.

Hipótesis de qué falla:
- Triadas confusables (manzanas, pasta, Coca-Cola Light vs Regular) → HOG capta forma común, BoVW capta marca común, ambos fallan en la variante.
- Frutas sin packaging → SIFT pobre (poca textura distintiva), HOG inestable (forma variable por orientación).
- Selective Search puede no proponer el bbox correcto para frutas pequeñas o muy juntas.

## 5. Pipeline B — Deep Learning (pendiente H6)

### 5.1. Modelo

**YOLOv8s pretrained COCO**, fine-tune sobre 20 clases.

Justificación:
- YOLOv8n: demasiado pequeño, riesgo de no superar al clásico claramente.
- YOLOv8m/l: pesados para Colab gratis (sesiones se cortan).
- **YOLOv8s = sweet spot**. Entrena en T4 free en ~2-3h.
- Single-shot: ~10 ms/img en T4, compatible con demo webcam.

### 5.2. Augmentations (dominio retail)

- Mosaic (default YOLOv8).
- Mixup ligero (alpha 0.1).
- HSV jitter moderado.
- Flip horizontal SÍ, vertical NO.
- Rotación ±15°.
- Random scale + crop.
- Cutout/erasing ligero.

### 5.3. Entrenamiento

- Transfer learning desde pesos COCO.
- Freeze backbone 5 épocas, luego unfreeze todo.
- 50-100 épocas, early stopping sobre `mAP@0.5 val`.
- Batch 16 (YOLOv8s en T4).
- SGD momentum 0.937, weight decay 5e-4, LR cosine + warmup.
- Semillas fijas.

### 5.4. Implementación (scaffolding listo, lo lleva colaborador)

Estado: archivos creados, parámetros con defaults razonables, listo para entrenar. Lo conduce un compañero — onboarding en [`ONBOARDING_DL.md`](ONBOARDING_DL.md).

- ✅ `configs/deep_yolo.yaml` — hyperparams + augmentations (HSV jitter, mosaic, mixup, fliplr=0.5, flipud=0).
- ✅ `src/grocery_detection/data/coco_to_yolo.py` — convierte splits COCO a formato YOLO (txt label por imagen + split lists).
- ✅ `src/grocery_detection/deep/train.py` — wrapper sobre `ultralytics.YOLO`.
- ✅ `src/grocery_detection/deep/infer.py` — wrapper de inferencia.
- ✅ `src/grocery_detection/deep/export_coco.py` — Results → COCO results JSON (alineado con eval común).
- ✅ `scripts/prepare_yolo_dataset.py` — CLI: convierte splits COCO → estructura YOLO (idempotente).
- ✅ `scripts/run_deep_train.py` — CLI: fine-tune.
- ✅ `scripts/run_deep_infer.py` — CLI: inferencia + export COCO.
- ✅ `notebooks/colab_yolo_train.ipynb` — Colab entrenamiento end-to-end con Drive symlink.
- ✅ `notebooks/colab_yolo_infer.ipynb` — Colab inferencia + export.

## 6. Framework de evaluación común (pendiente H7)

Capa única que come predicciones COCO JSON de ambos pipelines y produce comparativa.

### 6.1. Métricas obligatorias

- **mAP@0.5** y **mAP@0.5:0.95** (`pycocotools`).
- Accuracy global.
- F1 por clase + macro F1.
- IoU medio sobre TPs.
- **Matriz confusión 20×20** + clase background (21×21).
- Curvas PR por clase.
- Tiempo inferencia ms/img en igual hardware.

### 6.2. Métricas adicionales

- Tiempo entrenamiento total (clásico CPU vs DL GPU).
- Tamaño modelo on-disk.
- Análisis cualitativo: top-K fallos por clase, side-by-side A vs B.

### 6.3. Pendiente

- `src/grocery_detection/eval/metrics.py`
- `src/grocery_detection/eval/confusion.py`
- `src/grocery_detection/eval/timing.py`
- `scripts/run_evaluation.py`

## 7. Análisis de robustez (pendiente H8)

Test perturbado con:
- Ruido gaussiano σ ∈ {0.01, 0.05, 0.1}.
- Blur σ ∈ {1, 3, 5} px.
- Brillo {0.5×, 0.75×, 1.25×, 1.5×}.
- JPEG quality {70, 50, 30}.
- Oclusión cuadrado 20% area random.

Curva degradación mAP@0.5 vs nivel, por pipeline.

Pendiente: `src/grocery_detection/eval/robustness.py` + `scripts/run_robustness.py`.

## 8. Estructura de repositorio (estado actual)

```
grocery-detection/
├── README.md
├── PROYECTO.md                         # ESTE archivo
├── RUN_LOCAL_FEDORA.md                 # quickstart local Fedora
├── RUN_COLAB.md                        # quickstart Colab + Drive
├── pyproject.toml                      # uv-managed deps
├── .python-version
├── uv.lock
├── .gitignore
├── configs/
│   ├── classes.yaml                    # 20 clases finales
│   ├── data.yaml                       # paths dataset, splits
│   └── classical.yaml                  # hyperparams pipeline A (incl. preprocessing, checkpoint)
├── src/grocery_detection/
│   ├── __init__.py
│   ├── data/
│   │   ├── download_d2s.py             # extracción D2S desde .tar.xz
│   │   ├── filter_classes.py           # subset a 20 clases con IDs contiguos
│   │   ├── prepare.py                  # stratified split val/test
│   │   ├── eda.py                      # helpers para EDA notebook
│   │   └── visualize.py
│   ├── classical/
│   │   ├── preprocessing.py            # WB + CLAHE + denoise
│   │   ├── proposals.py                # Selective Search
│   │   ├── descriptors/
│   │   │   ├── hog.py
│   │   │   ├── sift.py
│   │   │   └── bovw.py
│   │   ├── features.py                 # extract_features_batch (HOG + BoVW)
│   │   ├── labeling.py                 # IoU-based pos/neg/ignore
│   │   ├── training_set.py             # build_training_features con checkpoint
│   │   ├── classifier.py               # AdditiveChi2 + LinearSVC OvR
│   │   ├── hard_negative.py            # mining loop con resume
│   │   ├── nms.py
│   │   ├── iou.py
│   │   └── pipeline.py                 # detect() end-to-end
│   └── utils/
│       ├── config.py                   # carga YAML + repo_root()
│       └── seed.py                     # seeding global reproducible
├── scripts/
│   ├── setup.ps1 / setup.sh            # uv install + uv sync
│   ├── prepare_d2s.py                  # extracción D2S
│   ├── prepare_splits.py               # 20-class filter + val/test split
│   ├── train_codebook.py               # KMeans K=300 sobre SIFTs train
│   ├── test_classical_tiny.py          # smoke test 5+3 imgs
│   ├── run_classical_train.py          # H4: build features + SVM
│   ├── run_classical_hard_neg.py       # H5: mining + refit
│   ├── run_classical_infer.py          # inference test → COCO JSON
│   ├── import_colab_svm.py             # wrap artifact Colab → ClassicalSVM
│   ├── run_overnight.ps1 / .sh         # train → hardneg → infer chain
│   ├── colab_helper.py                 # helpers para notebooks Colab
│   ├── generate_overview_pdf.py        # PDF H1-H5
│   └── generate_onboarding_pdf.py      # PDF colaboradores
├── notebooks/
│   ├── 00_dataset_eda.ipynb            # H1 — exploración D2S, conteos
│   ├── 01_class_selection.ipynb        # H2 — criterios + verificación 20 clases
│   ├── 02_classical_dev.ipynb          # H3 — demo componentes
│   ├── 03_training_visualization.ipynb # proposals + labels + features
│   ├── 04_classical_results.ipynb      # métricas + confusión sobre test
│   ├── 05_preprocessing_viz.ipynb      # ablation del preprocesado
│   ├── colab_build_features.ipynb      # build features en Colab (H4)
│   ├── colab_train_svm.ipynb           # SVM fit en Colab (calidad full)
│   ├── colab_hard_neg.ipynb            # mining en Colab (H5)
│   └── colab_infer.ipynb               # inferencia en Colab
└── reports/
    ├── H1-H5_overview.pdf              # snapshot resultados
    ├── onboarding_colaboradores.pdf
    ├── figures/                        # generadas por notebooks/eval
    ├── tables/
    └── predictions/
        └── classical_test.json         # output de inferencia (gitignored)
```

### Principios

- Configs en YAML, sin hardcode.
- CLI con `argparse` simple. Scripts imprimen `[boot]` antes de imports pesados para feedback rápido.
- Semillas fijas globales (`utils.seed.set_seed`).
- `pyproject.toml` con `uv`, lockfile (`uv.lock`) en git.
- Notebooks como notebooks de desarrollo (exploración / visualización), no entry-point.

## 9. Hoja de ruta

| Hito | Descripción | Estado |
|---|---|---|
| **H1** | Setup repo + descarga D2S + EDA | ✅ `prepare_d2s.py` + `00_dataset_eda.ipynb`. 20 clases verificadas ≥60 train / ≥60 val. |
| **H2** | Selección 20 clases + splits filtrados | ✅ `prepare_splits.py` + `01_class_selection.ipynb`. Splits en `data/processed/`. |
| **H3** | Componentes pipeline clásico + codebook | ✅ `classical/proposals.py`, `descriptors/{hog,sift,bovw}.py`, `features.py`, `labeling.py`, `preprocessing.py`, `train_codebook.py`. Codebook K=300. |
| **H4** | Pipeline A — training + inferencia | ✅ `training_set.py` + `classifier.py` + `pipeline.py` + scripts CLI. Checkpoint reanudable cada 100 imgs. |
| **H5** | Pipeline A — hard negative mining | ✅ `hard_negative.py` + `run_classical_hard_neg.py`. Resume a 2 niveles (ronda + intra-ronda). |
| **Colab workflow** | Ejecutar H4-H5 en Colab | ✅ `scripts/colab_helper.py` + 4 notebooks (`colab_build_features`, `colab_train_svm`, `colab_hard_neg`, `colab_infer`). |
| **H6** | Pipeline B — YOLOv8s fine-tune | 🛠️ scaffolding completo (configs + deep/ + scripts + notebooks Colab). Lo lleva colaborador. Ver `ONBOARDING_DL.md`. |
| **H7** | Framework evaluación común | ⏳ pendiente. |
| **H8** | Análisis de robustez | ⏳ pendiente. |
| **H9** | Demo webcam | ⏳ opcional. |
| **H10** | Informe + presentación | ⏳ paralelo a H6-H8. |

## 10. Stack técnico

### Lenguaje y entorno
- Python 3.11.
- Gestor: **`uv`** (`uv sync --all-extras`).

### Dependencias

**Core**: numpy, scipy, pandas, opencv-contrib-python (necesario para `ximgproc` + SIFT), pillow, pyyaml.

**Pipeline A**: scikit-learn (KMeans, LinearSVC, AdditiveChi2Sampler), scikit-image (HOG).

**Pipeline B** (pendiente H6): torch, torchvision, ultralytics.

**Eval** (pendiente H7): pycocotools, matplotlib, seaborn.

**UX**: tqdm, rich, requests.

**Dev**: pytest, ruff, ipykernel, jupyter, notebook.

### Hardware
- Desarrollo local: Windows / Mac / Fedora.
- Entrenamiento pipeline A: Colab CPU (T4 free no aporta — Selective Search/SIFT/HOG son CPU-only) o PC local lo suficientemente rápido.
- Entrenamiento pipeline B: Colab GPU T4 free.
- Dataset D2S: subido por usuario a Drive `MyDrive/grocery-detection/raw/`.

## 11. Reproducibilidad

- Semilla global `seed=42` en cada script vía `utils.seed.set_seed` (random + numpy + torch + cv2).
- Splits guardados como JSON fijo en `data/processed/`. No se resamplea por ejecución.
- `uv.lock` fija versiones de todas las deps.
- Comandos exactos para reproducir cada stage documentados en `README.md` + `RUN_LOCAL_FEDORA.md` + `RUN_COLAB.md`.
- Checkpoint reanudable: cualquier corte se recupera sin perder más de N imágenes (default N=100).
- **Escritura atómica** (`utils/atomic.py`: `atomic_savez_compressed`, `atomic_write_json`, `atomic_write_pickle`) en todos los puntos de persistencia. Un corte mid-flush no deja archivos corruptos, solo basura `.tmp` (limpiable a mano).
- **Drive como almacenamiento primario en Colab**: `link_processed_to_drive()` / `link_predictions_to_drive()` symlinkan los directorios de output a `MyDrive/grocery-detection/{processed,predictions}/`. Las escrituras del pipeline caen directas a Drive — el reanude tras corte de Colab es transparente.

## 12. Pendientes / decisiones abiertas

- [x] ~~Symlink `data/processed/` ↔ `MyDrive/grocery-detection/processed/` en `colab_helper.py` (`link_processed_to_drive`, `link_predictions_to_drive`)~~ ✅ implementado.
- [x] ~~Atomic write (`tmp + rename`) en todas las escrituras persistentes (`utils/atomic.py` + integrado en `training_set.py`, `hard_negative.py`, `classifier.py`, `bovw.py`, `run_classical_infer.py`)~~ ✅ implementado.
- [ ] H6 — YOLOv8s scaffolding.
- [ ] H7 — Framework evaluación común + unit tests sobre métricas.
- [ ] H8 — Módulo de perturbaciones.
- [ ] H9 — Demo webcam.
- [ ] **Hyperparams pipeline A iterables** (en notebooks 02/03/04):
  - K del codebook (300 actual, puede subirse).
  - `proposals.max_per_image` (300 actual).
  - `classifier.C`, `sample_steps`.
  - `score_thresh` y `nms_iou` de inferencia.
- [ ] **Métrica primaria** para informe: probablemente mAP@0.5 + matriz confusión.

## 13. Riesgos conocidos

| Riesgo | Mitigación |
|---|---|
| Colab desconecta mid-run | Mitigado: `data/processed/` y `reports/predictions/` symlinkados a Drive (`link_*_to_drive`) + escritura atómica → cada checkpoint cae directo a Drive y un corte mid-flush no corrompe el archivo. Pérdida máxima por corte: ≤100 imgs. Adicionalmente: keep-alive JS en navegador, considerar Colab Pro para sesiones largas. |
| PC local incapaz de correr H4 | Alternativa Colab consolidada con `RUN_COLAB.md`. Segundo PC Fedora documentado en `RUN_LOCAL_FEDORA.md`. |
| Selective Search lento | `mode: fast` + `max_per_image: 300` + resize `max_side: 640`. Cachear proposals si fuese necesario. |
| Pipeline A clasifica casi todo como background | Hard neg mining + ajuste `score_thresh`. |
| Diferencia A vs B insuficiente | Triadas confusables — el contraste va a aparecer ahí. Análisis robustez refuerza. |
| Tamaño D2S excede Drive free (15 GB) | D2S ocupa ~6-7 GB en `.tar.xz`. Tras extraer en `/content` (volátil) se borra al terminar sesión. Drive solo guarda raw archives + processed artifacts (~200 MB). |

## 14. Qué debe hacer la siguiente sesión de Claude Code

**Estado actual**: H1-H5 completos. Workflow Colab consolidado. Docs `RUN_LOCAL_FEDORA.md` y `RUN_COLAB.md` listos.

### Acción inmediata posible

Opciones por orden de impacto:

1. **Symlink-Drive fix** + atomic write — robustez Colab. ~20 min.
2. **H6 — YOLOv8s scaffolding** (configs, conversion COCO→YOLO, train/infer wrappers, notebook Colab). ~45 min.
3. **H7 — Framework evaluación** + unit tests sobre métricas con datos sintéticos. ~60 min.
4. **H8 — Robustness perturbations module**. ~30 min.

### Estilo de trabajo del alumno

- Estudiante UAB, conoce todo el stack (Python, NumPy, OpenCV, scikit-learn, PyTorch).
- Quiere que Claude **dirija las decisiones técnicas** y proponga con justificación.
- Espera **trade-offs explícitos** sobre cualquier decisión no obvia.
- Sin diplomacia innecesaria. Si una opción es claramente mejor, decirlo.
- PC del curro lento + segundo PC Fedora + Colab free → workflow primario en Colab.
- Trabaja solo, sin presión de plazo comunicada.

### Cosas que NO hay que volver a hacer

- Decidir clases (cerradas en `configs/classes.yaml`).
- Cambiar pipeline A (consolidado y entrenable).
- Migrar a otro dataset (D2S confirmado).

### Cosas que SÍ pueden cambiar todavía

- Hyperparams del pipeline A (K, C, thresholds) — iterar en notebooks tras ver resultados.
- Familia DL en H6 si YOLOv8s da problemas (fallback YOLOv8n o RT-DETR).
- Estructura del informe — pendiente hasta H10.

---

**Fin del briefing.** Actualizar este documento conforme avancen los hitos.
