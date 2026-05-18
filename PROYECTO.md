# Proyecto ABP — Detección de productos de Mercadona: Visión Clásica vs Deep Learning

> **Briefing para futuras sesiones de Claude Code.** Este documento contiene todo el contexto, decisiones tomadas, justificación técnica, estructura de repositorio, hoja de ruta y pendientes. Leer entero antes de tocar código.

---

## 0. Contexto del proyecto

- **Estudiante**: Artur Moret, Escola d'Enginyeria (UAB).
- **Asignatura**: ABP de Visión por Computador.
- **Modalidad**: Individual.
- **Hardware disponible**: sin GPU local. Todo entrenamiento en **Google Colab / Kaggle (T4 gratis)**.
- **Entregables**:
  - Código + repositorio limpio.
  - Informe extenso con metodología y resultados.
  - Presentación.
  - Notebooks de desarrollo (para inspeccionar evolución).
  - Demo webcam en vivo: nice-to-have, no obligatorio.

## 1. Objetivo

Diseñar, implementar y evaluar **dos pipelines paralelos** que detecten el mismo conjunto de productos en imágenes:

- **Pipeline A — Visión por Computador clásica** (sin redes neuronales profundas).
- **Pipeline B — Deep Learning** (YOLOv8s).

Ambos reciben las mismas imágenes, predicen las mismas etiquetas + bounding boxes, y se comparan con las mismas métricas sobre el mismo split de test. El **verdadero entregable académico es la comparativa cuantitativa y cualitativa entre paradigmas**, no batir el estado del arte.

**Caso de uso final**: usuario muestra un producto a una webcam y el sistema lo detecta + clasifica. **No es detección en estantería de supermercado** — escena típica = un producto presentado a cámara, fondo libre o ligeramente cluttered.

## 2. Decisiones cerradas (con razonamiento)

| Decisión | Valor | Razón |
|---|---|---|
| Dataset | **MVTec D2S** (público) | 60 SKUs específicos europeos, bbox + máscaras, train con objetos sueltos sobre superficie clara (~webcam-like), test con escenas cluttered. Académicamente sólido, paper citable, splits estándar. |
| Tarea | **Detección (bbox + clase)** | Justifica las métricas mAP/IoU pedidas por el alumno. El sistema dibuja caja alrededor del producto y lo clasifica. |
| Nº de clases | **14** | Subset D2S × Mercadona. Productos verificables en Mercadona España. Mantiene triadas confusables para análisis de errores. |
| Pipeline A | **Selective Search + (HOG + SIFT/BoVW) + SVM χ²** | Detección clásica honesta sin caer en heurísticas de color. Descrito en sección 4. |
| Pipeline B | **Solo YOLOv8s** (descartado Faster R-CNN) | Decisión del alumno. Simplicidad sobre rigor adicional. |
| Robustez | **Sí incluida** | Test perturbado (ruido, blur, iluminación). Análisis de degradación A vs B = oro académico. |
| Gestión deps | **`uv`** | Gestiona venv + lockfile + install rápido. Un solo `uv sync`. |
| Configs | **YAML + dataclasses + `argparse`** | Sin Hydra. Estándar Python, cero magia. Defaults en YAML, overrides básicos por CLI. |

## 3. Las 14 clases (provisional, a confirmar tras EDA H1)

Subset de MVTec D2S filtrado a productos verificables en Mercadona España. Diseño con **triadas confusables** para enriquecer análisis de errores.

**Cereales Kellogg's** (cajas grandes — confusables mismo packaging marca):
1. Kellogg's Cornflakes
2. Kellogg's Choco Krispies
3. Kellogg's Special K
4. Kellogg's Frosties
5. Kellogg's Froot Loops
6. Kellogg's Smacks

**Chocolatinas** (barritas — forma similar, color de envoltorio distinto):

7. Mars
8. Snickers
9. Twix

**Chuches Haribo** (bolsas flexibles — packaging Haribo característico, contenido distinto):

10. Haribo Goldbären (Ositos de Oro)
11. Haribo Tropi Frutti
12. Haribo Smurfs (Pitufos)
13. Haribo Pasta Basta

**Bebidas**:

14. Red Bull 250ml lata

**Cobertura de forma**: 6 cajas cartón, 3 barritas, 4 bolsas flexibles, 1 lata. Diversidad razonable.

**Triadas/confusables intencionales**:
- 6 Kellogg's con mismo logo/marca → distinguir variante por etiqueta interior.
- 4 Haribo con misma marca + tipo de bolsa → contenido y color secundario cambian.
- 3 chocolatinas marcas distintas → mismo formato barrita.

**Hipótesis comparativa**: pipeline clásico va a confundir variantes dentro de triadas (BoVW capta presencia de logo Kellogg's pero falla en distinguir "Choco Krispies" de "Special K"). YOLOv8s va a distinguir variantes con mayor fidelidad. **Esa diferencia ES el resultado interesante del proyecto.**

**Acción H1**: tras descargar D2S, verificar conteos por clase (mínimo aceptable ~150 train / ~30 test por clase). Si alguna queda corta, sustituir desde D2S o reducir nº de clases.

## 4. Pipeline A — Visión por Computador Clásica

### 4.1. Arquitectura end-to-end

```
imagen
  → Selective Search (region proposals, ~2000 bboxes candidatos)
  → por cada proposal: extraer HOG + SIFT(BoVW histograma) → concatenar features
  → SVM χ² multi-clase (one-vs-rest, 14 SVMs) → score por clase + score "background"
  → filtrar proposals con max score < threshold
  → NMS por clase (IoU > 0.5)
  → bboxes finales con etiqueta y confianza
```

### 4.2. Componentes técnicos (con explicación pedagógica para el informe)

#### Selective Search
Algoritmo de **region proposal sin clase**.
- Sobre-segmenta imagen (Felzenszwalb) → ~500 regiones pequeñas.
- Itera mergeando regiones vecinas por similitud (color, textura, tamaño, fill).
- Cada nivel de merge produce candidatos bbox.
- Output: ~2000 bboxes por imagen, cada uno potencialmente conteniendo un objeto.
- Implementación: `cv2.ximgproc.createSelectiveSearchSegmentation()` (OpenCV contrib).

#### HOG (Histogram of Oriented Gradients)
Descriptor de **forma/silueta global**.
- Gradientes por pixel (Sobel).
- División en celdas 8×8 px, histograma de 9 orientaciones por celda.
- Normalización por bloques 2×2 celdas (robustez a iluminación).
- Concatenación → vector ~3780-D para 64×128 px.
- Funciona bien con objetos rígidos (cajas, latas). Sufre con bolsas deformables.

#### SIFT (Scale-Invariant Feature Transform)
Detector + descriptor de **keypoints locales**.
- Detecta puntos de interés en multi-escala (Difference of Gaussians).
- Por keypoint: parche 16×16 dividido en 4×4 sub-celdas con histograma 8 orientaciones cada una → vector 128-D.
- Invariante a rotación, escala, parcial a iluminación.
- Captura logos, letras, patrones distintivos del packaging.
- Implementación: `cv2.SIFT_create()`.

#### BoVW (Bag of Visual Words)
Convierte un conjunto variable de SIFTs en vector de longitud fija.
1. Extraer SIFT de todas las imágenes train → millones de descriptores 128-D.
2. **k-means** sobre todos esos descriptores con K=1500 → 1500 centroides = "vocabulario visual".
3. Para imagen/proposal nuevo: extraer SIFTs, asignar cada uno al centroide más cercano, contar → histograma de longitud 1500.
4. Normalizar L2.
- Análogo a bag-of-words de NLP, pero con parches visuales.
- Resultado = "firma" de qué texturas/patrones contiene la región, sin importar dónde ni cuántos.
- K=1500 elegido por nº de clases mayor (14); con 5 clases sería 500-1000.

#### Feature final por proposal
Concatenación: `[HOG (3780-D), BoVW (1500-D)]` → vector ~5280-D, normalizado L2.

#### SVM con kernel χ²
- SVM separa clases maximizando margen.
- Kernel χ² adecuado para histogramas: `K(x,y) = exp(-γ · Σᵢ (xᵢ-yᵢ)² / (xᵢ+yᵢ))`.
- Penaliza más diferencias en bins pequeños, donde están las palabras visuales raras y discriminativas.
- Multi-clase: **one-vs-rest** → 14 SVMs binarios (clase vs resto + background).
- Implementación: `sklearn.svm.SVC(kernel=chi2_kernel_precomputed)` o `kernel='precomputed'` con matriz Gram precomputada.

#### Hard negative mining
Pipeline propenso a falsos positivos sobre fondo.
1. Train inicial sobre proposals positivos (IoU≥0.5 con GT) + sample random de negativos.
2. Inferencia sobre train: extraer proposals clasificados como producto pero con IoU<0.3 con GT = falsos positivos.
3. Añadir esos al set de negativos.
4. Reentrenar.
- Ciclo 2-3 veces. Mejora precision sin tocar arquitectura.

#### NMS (Non-Maximum Suppression)
- Ordenar predicciones por confianza descendente.
- Coger top, marcar como definitiva.
- Descartar las que solapan IoU>0.5 con ella.
- Repetir.
- Output: 1 bbox por instancia real.

### 4.3. Honestidad académica

Pipeline A **va a perder en mAP contra YOLOv8s**. Eso ES el resultado interesante; el objetivo no es ganarle a YOLO, es **cuantificar la brecha y entender dónde el clásico se rompe**.

Hipótesis de qué falla:
- Bolsas Haribo (forma deformable) → HOG inestable.
- Triadas Kellogg's → BoVW capta marca pero falla variante.
- Iluminación variable → SIFT relativamente robusto, color histogram (si se añade) falla.

## 5. Pipeline B — Deep Learning

### 5.1. Modelo

**YOLOv8s pretrained en COCO**, fine-tune sobre 14 clases.

Justificación:
- YOLOv8n: demasiado pequeño, riesgo de no superar claramente al clásico.
- YOLOv8m/l: pesados para Colab gratis (sesiones se cortan).
- **YOLOv8s = sweet spot**. Entrena en T4 en ~2-3h para 14 clases.
- Single-shot detector: real-time inference (~10 ms/imagen en T4) — compatible con demo webcam.

Faster R-CNN descartado por decisión del alumno (simplicidad).

### 5.2. Augmentations (sensatas para dominio retail)

- **Mosaic** (default YOLOv8): combina 4 imágenes en una.
- **Mixup ligero**: alpha 0.1.
- **HSV jitter** moderado: H±0.015, S±0.7, V±0.4 (iluminación supermercado vs casa).
- **Flip horizontal sí, vertical NO**: productos no aparecen invertidos en uso real.
- **Rotación pequeña**: ±15°.
- **Random scale + crop**.
- **Erasing/cutout ligero**: simula occlusión parcial.

### 5.3. Estrategia de entrenamiento

- Transfer learning desde pesos COCO.
- Freeze backbone 5 épocas iniciales, luego unfreeze todo.
- 50-100 épocas, **early stopping** sobre `mAP@0.5 val`.
- Batch 16 (YOLOv8s en T4 cabe).
- Optimizer: SGD con momentum 0.937, weight decay 5e-4 (defaults YOLOv8).
- LR cosine schedule, warmup 3 épocas.
- Semillas fijas para reproducibilidad.

### 5.4. Implementación

Librería: `ultralytics` (oficial YOLOv8). API:
```python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.train(data='configs/d2s_mercadona.yaml', epochs=100, ...)
```
Salida: predicciones en formato YOLO → conversión a COCO JSON para evaluación común.

## 6. Framework de evaluación común

Capa única que come predicciones en formato **COCO JSON** de ambos pipelines y produce comparativa.

### 6.1. Métricas obligatorias

- **mAP@0.5** y **mAP@0.5:0.95** (`pycocotools`).
- **Accuracy global**.
- **F1 por clase** + macro F1.
- **IoU medio sobre verdaderos positivos**.
- **Matriz de confusión 14×14** (asignación greedy por IoU≥0.5, clase background como 15ª fila/columna).
- **Curvas Precision-Recall por clase**.
- **Tiempo de inferencia por imagen** (ms), medido en igual hardware:
  - CPU local (laptop del alumno).
  - GPU Colab T4.

### 6.2. Métricas adicionales (valor añadido)

- **Tiempo de entrenamiento total**: pipeline A no usa GPU; pipeline B sí. Justifica trade-off coste-beneficio.
- **Tamaño de modelo on-disk**:
  - Pipeline A: codebook k-means + 14 SVMs (.pkl) → ~10-50 MB estimado.
  - Pipeline B: YOLOv8s pesos → ~22 MB.
- **Análisis cualitativo**: top-K fallos por clase, visualizaciones side-by-side de predicciones A vs B sobre las mismas imágenes (para informe).

### 6.3. Análisis de robustez (sección 7) — métricas extra sobre test perturbado

## 7. Análisis de robustez

Test set adicional aplicando perturbaciones controladas al test original. Para cada perturbación, recalcular todas las métricas y comparar degradación A vs B.

### Perturbaciones

- **Ruido gaussiano**: σ ∈ {0.01, 0.05, 0.1} sobre imagen normalizada.
- **Blur**: kernel gaussiano σ ∈ {1, 3, 5} px.
- **Iluminación**: multiplicación brillo {0.5×, 0.75×, 1.25×, 1.5×}.
- **JPEG compression artifacts**: quality ∈ {70, 50, 30}.
- **Oclusión parcial**: cuadrado negro 20% área en posición random.

### Reporte

- Curva de degradación mAP@0.5 vs nivel de perturbación, por pipeline.
- Hipótesis a verificar: DL más robusto a perturbaciones leves, posiblemente más frágil a oclusión.

## 8. Estructura de repositorio

```
grocery-detection/
├── README.md                        # quick start: instalación + ejemplo
├── PROYECTO.md                      # ESTE archivo (briefing completo)
├── pyproject.toml                   # uv-managed deps
├── .python-version
├── uv.lock
├── .gitignore
├── configs/
│   ├── classes.yaml                 # 14 clases finales
│   ├── data.yaml                    # paths dataset, splits
│   ├── classical.yaml               # hyperparams pipeline A
│   ├── deep_yolo.yaml               # hyperparams pipeline B
│   ├── eval.yaml                    # métricas y thresholds
│   └── robustness.yaml              # niveles de perturbación
├── src/grocery_detection/
│   ├── __init__.py
│   ├── data/
│   │   ├── download_d2s.py          # descarga MVTec D2S
│   │   ├── filter_classes.py        # subset a 14 clases
│   │   ├── prepare.py               # splits, conversión a COCO JSON
│   │   └── visualize.py             # bboxes overlay para EDA
│   ├── classical/
│   │   ├── proposals.py             # Selective Search
│   │   ├── descriptors/
│   │   │   ├── hog.py
│   │   │   ├── sift.py
│   │   │   └── bovw.py              # codebook k-means + asignación
│   │   ├── codebook.py              # entrenamiento vocabulario visual
│   │   ├── classifier.py            # SVM χ² OvR, RF baseline
│   │   ├── hard_negative.py         # mining loop
│   │   ├── nms.py
│   │   └── pipeline.py              # detect() end-to-end
│   ├── deep/
│   │   ├── train.py                 # wrapper ultralytics
│   │   ├── infer.py
│   │   └── export_coco.py           # predicciones YOLO → COCO JSON
│   ├── eval/
│   │   ├── metrics.py               # mAP, IoU, F1, accuracy
│   │   ├── confusion.py             # matriz confusión con asignación greedy
│   │   ├── pr_curves.py
│   │   ├── timing.py                # benchmark inferencia
│   │   ├── robustness.py            # generación test perturbado + eval
│   │   └── compare.py               # tablas + figuras comparativas
│   ├── demo/
│   │   └── webcam.py                # demo en vivo (opcional)
│   └── utils/
│       ├── coco_format.py
│       ├── config.py                # carga YAML + dataclasses
│       └── seed.py
├── notebooks/
│   ├── 00_dataset_eda.ipynb         # exploración D2S, conteos, ejemplos
│   ├── 01_class_selection.ipynb     # filtrado + verificación 14 clases
│   ├── 02_classical_dev.ipynb       # desarrollo iterativo pipeline A
│   ├── 03_deep_dev.ipynb            # desarrollo iterativo pipeline B
│   ├── 04_comparison.ipynb          # comparativa final
│   ├── 05_robustness.ipynb          # análisis robustez
│   └── 06_error_analysis.ipynb      # qualitative deep dive en fallos
├── scripts/                         # entry points reproducibles
│   ├── run_classical_train.py
│   ├── run_classical_infer.py
│   ├── run_deep_train.py
│   ├── run_deep_infer.py
│   ├── run_evaluation.py
│   └── run_robustness.py
├── tests/                           # unit tests sobre métricas y conversiones
│   ├── test_metrics.py
│   ├── test_coco_format.py
│   └── test_nms.py
└── reports/
    ├── figures/                     # PNG/SVG generados, export-ready
    ├── tables/                      # CSV/Markdown generados
    └── predictions/                 # JSON COCO de cada pipeline + perturbation
```

### Principios

- Configs en YAML, **sin hardcode** de paths/hyperparams en código.
- CLI con `argparse` simple en cada script.
- Semillas fijas para reproducibilidad.
- Tests unitarios sobre métricas y conversiones de formato (no sobre training).
- Notebooks **solo para exploración y desarrollo iterativo**, no como entry point de producción.
- `pyproject.toml` con `uv`, lockfile en git.

## 9. Hoja de ruta — Hitos

| Hito | Descripción | Salida concreta |
|---|---|---|
| **H1** | Setup repo + descarga D2S + EDA | Repo inicializado con `uv`, D2S descargado, notebook `00_dataset_eda.ipynb` con conteos por clase y ejemplos visuales |
| **H2** | Selección 14 clases definitiva | `configs/classes.yaml` cerrado, splits train/val/test fijos en COCO JSON, `01_class_selection.ipynb` documenta criterios |
| **H3** | Pipeline A — proposals + descriptores | Selective Search funcionando, extracción HOG+SIFT+BoVW probada en notebook |
| **H4** | Pipeline A — entrenamiento + inferencia | `run_classical_train.py` produce SVMs entrenados, `run_classical_infer.py` produce predicciones COCO JSON sobre test |
| **H5** | Pipeline A — hard negative mining | Mejora medida sobre baseline (notebook con before/after) |
| **H6** | Pipeline B — YOLOv8s fine-tune | Modelo entrenado en Colab, predicciones COCO JSON sobre test |
| **H7** | Framework evaluación común | `run_evaluation.py` genera tablas + figuras comparativas autogeneradas a partir de los dos JSONs |
| **H8** | Análisis de robustez | Test perturbado generado, métricas recalculadas, curvas de degradación |
| **H9** | Demo webcam (opcional) | `webcam.py` carga ambos modelos, live inference |
| **H10** | Informe + presentación | Drafts con figuras/tablas auto-generadas referenciadas |

## 10. Stack técnico

### Lenguaje y entorno
- Python 3.11+
- Gestor: **`uv`** (`uv init`, `uv add`, `uv sync`).

### Dependencias principales

**Core**:
- `numpy`, `scipy`, `pandas`
- `opencv-contrib-python` (necesita contrib para `ximgproc`/SIFT/Selective Search)
- `pillow`
- `pyyaml`

**Pipeline A**:
- `scikit-learn` (k-means, SVM, RandomForest)
- `scikit-image` (HOG)

**Pipeline B**:
- `torch`, `torchvision`
- `ultralytics`

**Eval**:
- `pycocotools`
- `matplotlib`, `seaborn`
- `tqdm`

**Demo webcam**:
- `opencv-contrib-python` (ya incluido)

**Dev**:
- `pytest`
- `ruff` (lint+format)
- `ipykernel` (notebooks)

### Hardware
- Desarrollo local: laptop Windows 11, sin GPU. Solo para pipeline A (CPU) y código.
- Entrenamiento pipeline B: **Google Colab** (T4 free) o **Kaggle Notebooks**.
- Dataset D2S montado en Drive para acceso desde Colab.

## 11. Reproducibilidad

- **Semilla global** en cada script: `seed.py` con `random`, `numpy`, `torch`, `cv2.setRNGSeed`.
- Splits guardados como JSON fijo en `configs/data.yaml` (no resampleados en cada run).
- Versiones de deps fijadas en `uv.lock`.
- Comandos exactos para reproducir cada hito documentados en `README.md`.

## 12. Pendientes / decisiones abiertas

- [ ] **Verificación final de las 14 clases**: tras descargar D2S en H1, confirmar que cada clase tiene ≥150 muestras train y ≥30 test. Sustituir las que no.
- [ ] **Hyperparams pipeline A**: K del codebook BoVW (1500 propuesto), γ del kernel χ², threshold de confianza, threshold NMS. Iterar en notebook 02.
- [ ] **Threshold de score** para filtrar proposals en pipeline A: empezar con 0.5, calibrar por curva PR.
- [ ] **Estrategia background class** para SVM OvR: ¿clase explícita o solo "ninguna clase con score > threshold"?
- [ ] **Demo webcam**: si se hace, decidir UI (puro OpenCV vs Streamlit/Gradio).
- [ ] **Métrica primaria** para informe: probablemente mAP@0.5, pero confirmar tras ver resultados.

## 13. Riesgos conocidos

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Alguna clase D2S tiene pocas muestras | Media | Sustituir clase desde lista de candidatas D2S-Mercadona alternativas; o reducir a 12-13 clases |
| Selective Search lento en imágenes grandes | Alta | Resize de input a 640×480 antes de SS; cachear proposals en disco |
| Colab desconecta durante entreno YOLOv8s | Media | Checkpoints cada 10 épocas, guardar en Drive |
| Pipeline A clasifica casi todo como background | Media | Hard negative mining + ajuste threshold de score |
| Diferencia A vs B no es suficientemente clara | Baja | Confiar en triadas confusables — el contraste va a aparecer ahí; reforzar con análisis de robustez |
| Tamaño D2S excede cuota Drive gratis | Baja | D2S no es enorme; alternativa: dataset solo en Colab local storage por sesión |

## 14. Qué debe hacer la siguiente sesión de Claude Code

**Estado al cerrar este briefing**: plan aprobado, **nada implementado todavía**. Punto de partida = H1.

### Acción inmediata

1. Confirmar con el alumno que sigue de acuerdo con todo lo de este documento (lectura rápida).
2. Inicializar repo:
   ```powershell
   uv init grocery-detection
   cd grocery-detection
   uv add numpy scipy pandas opencv-contrib-python pillow pyyaml scikit-learn scikit-image torch torchvision ultralytics pycocotools matplotlib seaborn tqdm
   uv add --dev pytest ruff ipykernel
   ```
3. Crear estructura de directorios según sección 8.
4. Descargar MVTec D2S (`src/grocery_detection/data/download_d2s.py`):
   - URL oficial: https://www.mvtec.com/company/research/datasets/mvtec-d2s
   - Requiere aceptar licencia; el script debe instruir al usuario, no descargar automáticamente sin consentimiento.
5. Notebook `00_dataset_eda.ipynb`: cargar D2S, contar instancias por clase, verificar que las 14 propuestas tienen muestras suficientes, mostrar ejemplos.
6. Reportar al alumno conteos reales y proponer ajustes a las 14 clases si hace falta.

### No hacer en H1

- **No** entrenar nada todavía.
- **No** crear código de pipeline A ni B.
- **No** crear scripts de evaluación.
- Solo: setup + EDA + verificación de viabilidad de las 14 clases.

### Estilo de trabajo preferido por el alumno

- Es estudiante UAB, conoce todo el stack (Python, NumPy, OpenCV, scikit-learn, PyTorch, TF/Keras).
- Quiere que Claude **dirija las decisiones técnicas** y proponga.
- Espera **justificaciones** de las decisiones (trade-offs explícitos).
- Quiere ser **crítico** con las propuestas — sin diplomacia innecesaria si una opción es claramente mejor.
- Trabaja solo, sin presión de plazos comunicada.
- Sin GPU local — entrenamiento siempre va a Colab/Kaggle.

---

**Fin del briefing.** Volver a leer si surge duda. Actualizar este documento conforme avancen los hitos.
