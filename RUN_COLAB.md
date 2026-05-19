# Correr el pipeline clásico (H1-H5) en Google Colab

Guía paso a paso para ejecutar todo el pipeline clásico sin tocar tu PC local (más allá de subir archivos a Google Drive).

> Si en algún momento un paso "no va", revisa la sección [Troubleshooting](#troubleshooting) al final.

---

## Cómo funciona el resume (importante)

Cada notebook montará `data/processed/` y `reports/predictions/` como **symlinks directos a Drive** vía `link_processed_to_drive()` / `link_predictions_to_drive()`. Consecuencias:

- Cada checkpoint del script (cada 100 imgs) se escribe **directo a `MyDrive/grocery-detection/processed/`**, no a `/content/` (volátil).
- Escrituras son **atómicas** (tmp + rename): si Colab muere a mitad del flush, el archivo final no queda corrupto.
- Si Colab desconecta: reabres el notebook, **Run All**, los scripts detectan el checkpoint en Drive y reanudan desde la última imagen procesada.
- **Nunca pierdes más de 100 imágenes** por corte.

No hay paso de sync manual al final. La salida de cada stage ya está en Drive cuando el script termina su 100º checkpoint.

---

## 0. Setup en Google Drive (una sola vez)

Crear esta estructura en `MyDrive/`:

```
MyDrive/grocery-detection/
    raw/
    processed/
    predictions/
```

Cómo:

1. Abre https://drive.google.com
2. *New → Folder* → `grocery-detection`
3. Dentro: crea tres carpetas: `raw`, `processed`, `predictions`.

## 1. Descargar y subir MVTec D2S a Drive

D2S requiere aceptar licencia (no auto-download posible).

1. Abrir https://www.mvtec.com/company/research/datasets/mvtec-d2s
2. Rellenar formulario, aceptar términos.
3. Descargar (~6 GB):
   - `d2s_images_v*.tar.xz`
   - `d2s_annotations_v*.tar.xz`
4. Subir ambos archivos a `MyDrive/grocery-detection/raw/`.

> Drive desktop sync acelera mucho este paso si vas a estar varios GB.

## 2. Pasos baratos en local (1-5 min cada uno)

Si tu PC va MUY lento o no quieres tocarlo, **salta a la sección 3** — los notebooks Colab tienen celdas opcionales para correr esto allí también.

### En tu PC

```bash
# Windows PowerShell o Linux/macOS — clona el repo y prepara entorno:
git clone https://github.com/arturmoret/GroceryTracker.git
cd GroceryTracker

# Windows:
.\scripts\setup.ps1

# Linux/macOS:
bash scripts/setup.sh
```

Después: descarga D2S a `data/raw/` y:

```bash
uv run python scripts/prepare_d2s.py     # extrae D2S
uv run python scripts/prepare_splits.py  # genera train/val/test JSONs
uv run python scripts/train_codebook.py  # entrena codebook BoVW
```

Sube a `MyDrive/grocery-detection/processed/`:

- `data/processed/train.json`
- `data/processed/val.json`
- `data/processed/test.json`
- `data/processed/codebook.pkl`

> Drive desktop sync hace esto automático si pones `data/processed/` dentro del Drive sincronizado. Si no, sube manual desde la web.

## 3. Notebooks Colab — en este orden exacto

Abre cada notebook en https://colab.research.google.com vía **File → Open notebook → GitHub** tab, y pega `arturmoret/GroceryTracker` (deja la rama main).

Selecciona el notebook y dale **Runtime → Run all**.

### 3.1. `notebooks/colab_build_features.ipynb`

Extrae features (Selective Search + HOG + SIFT + BoVW) sobre las imágenes train.

- **Duración estimada**: 2-5 horas (depende del slot de CPU).
- **Output en Drive**: `processed/classical_features.npz` (~150 MB).
- **Checkpoint**: cada 100 imgs, directo a Drive, atómico. Robusto a cortes.

Si nunca corriste el paso 2 (los baratos en local), descomenta las dos últimas celdas del notebook (`# run_script("scripts/prepare_splits.py")` y `# run_script("scripts/train_codebook.py")`) y córrelas antes del build.

### 3.2. `notebooks/colab_train_svm.ipynb`

Entrena la SVM χ² (AdditiveChi2Sampler + LinearSVC OvR).

- **Duración estimada**: 5-15 min.
- **Output en Drive**: `processed/classical_svm.pkl`.
- Requisito: `classical_features.npz` ya en Drive (paso 3.1).

### 3.3. `notebooks/colab_hard_neg.ipynb` (opcional, recomendado)

Hard negative mining: mejora la SVM añadiendo falsos positivos como negativos.

- **Duración estimada**: 4-6 horas (2 rondas × ~2-3 h cada una).
- **Output en Drive**: `classical_features.npz` actualizado + `classical_svm.pkl` actualizado + `.hardneg_state.json`.
- **Resume a 2 niveles**: por ronda (state JSON) y dentro de ronda (checkpoint partial).

Saltable si quieres ahorrar tiempo — el pipeline funciona sin él, solo con peor precision.

### 3.4. `notebooks/colab_infer.ipynb`

Aplica el detector entrenado sobre el split test.

- **Duración estimada**: 1-2 horas.
- **Output en Drive**: `predictions/classical_test.json`.
- **Checkpoint**: cada 100 imgs vuelca el JSON en Drive (atómico). Reanuda automático.

## 4. Qué hago si Colab desconecta

Habitual. Free tier corta sesiones tras ~90 min de inactividad o ~12 h activas.

### Mitigación 1 — Keep-alive en el navegador

En la pestaña de Colab abierta:

1. F12 (abre DevTools).
2. Pestaña *Console*.
3. Pega y enter:

    ```javascript
    function KeepAlive(){
        document.querySelector("colab-connect-button")
                .shadowRoot.querySelector("#connect").click()
    }
    setInterval(KeepAlive, 60000);
    ```

Hace click virtual cada 60 s. No bulletproof pero aguanta varias horas más.

### Mitigación 2 — Resume automático

Cuando Colab muera:

1. Reabre el notebook.
2. *Runtime → Run all*.
3. Las celdas iniciales: clone repo + mount Drive + install deps + extraer D2S + `link_processed_to_drive()`.
4. Como `data/processed/` apunta directo a Drive, el checkpoint del corte anterior **ya está allí**.
5. Script detecta `processed_ids` en el `.npz` (o `image_id`s ya predichos en el JSON), salta lo hecho, sigue.

Pierdes como mucho las imágenes procesadas desde el último checkpoint (≤100).

### Mitigación 3 — Colab Pro

~10 €/mes. Sesiones de 24 h, background execution (corre sin tener la tab abierta), CPU mejor. Un mes durante el proyecto puede compensar.

## 5. Dónde quedan los artifacts

Tras completar los 4 notebooks, en `MyDrive/grocery-detection/`:

```
processed/
    train.json
    val.json
    test.json
    codebook.pkl
    classical_features.npz       ← features extraídas (H4)
    classical_svm.pkl            ← clasificador entrenado (H4 + H5)
    classical_svm_artifact.pkl   ← dict portable (intermediario de H4)
    .hardneg_state.json          ← marcador de rondas completadas (H5)
    classical_features.hardneg_partial.npz  ← intra-ronda (efímero, se borra al cerrar ronda)

predictions/
    classical_test.json          ← predicciones del detector clásico sobre test
```

`classical_test.json` es el entregable de H1-H5 y la entrada para el framework de evaluación (H7) y la comparativa con YOLOv8s (H6).

## 6. Bajar los artifacts al PC local

Cuando quieras seguir trabajando en local (notebooks de visualización, informe, etc.):

```bash
# Tras git pull en local
# Copia desde Drive a data/processed/ (manual o con Drive desktop sync)
# Luego puedes abrir los notebooks 03/04/05 y trabajar con los resultados.
```

Los `.pkl` y `.npz` son cross-platform — generados en Colab funcionan en Linux/Mac/Windows siempre que la versión de sklearn sea compatible (`uv.lock` fija versiones).

---

## Troubleshooting

### "FileNotFoundError: No existe /content/drive/MyDrive/grocery-detection/raw"

Drive no está montado o la carpeta no existe. Verifica:

1. Has corrido la celda `mount_drive()` y aceptado los permisos de Google Auth.
2. La carpeta `grocery-detection/` existe en `MyDrive/` (raíz, no en subcarpetas).

### "Faltan archivos D2S en /content/drive/MyDrive/grocery-detection/raw"

Los .tar.xz no están en Drive aún. Sube `d2s_images_v*.tar.xz` y `d2s_annotations_v*.tar.xz` a esa carpeta y re-ejecuta `setup_dataset()`.

### `link_processed_to_drive()` falla con `OSError: ... in use`

Drive FUSE puede estar bloqueando un archivo abierto. Reinicia el runtime (*Runtime → Restart runtime*), re-corre todas las celdas.

### El JSON de predicciones aparece como `classical_test.json.tmp` en Drive

El script murió a mitad de un flush atómico. El `.tmp` es basura — bórralo a mano (web Drive o `rm` en otra celda). El último JSON limpio sigue siendo el bueno; al relanzar el notebook reanudará desde ahí.

### `ImportError: No module named cv2.ximgproc`

Te toca un Colab con OpenCV no-contrib. `install_deps()` debería arreglarlo (`pip install opencv-contrib-python`). Si persiste:

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-contrib-python"], check=True)
# Reinicia el runtime: Runtime → Restart runtime
```

### El SVM en `colab_train_svm.ipynb` se queda colgado

Por defecto `n_jobs=-1` (todos los cores). Si el runtime de Colab es pequeño puede OOM. Cambia en la celda:

```python
svm = OneVsRestClassifier(
    LinearSVC(C=1.0, dual='auto', max_iter=5000, random_state=42),
    n_jobs=1,  # ← serial en lugar de paralelo
)
```

### Después de `import_colab_svm.py`, "ERROR: artifact incompleto"

El `.pkl` artifact tiene una key distinta a las esperadas (`sampler`, `svm`, `target_class_ids`). Verifica que la celda *Entrenar SVM* terminó OK y que la *Guardar artifact* corrió después.

---

## TL;DR

1. Drive: crea `MyDrive/grocery-detection/{raw,processed,predictions}/`.
2. Sube D2S `.tar.xz` a `raw/`.
3. (Local barato) `prepare_splits.py` + `train_codebook.py` → sube JSONs + codebook a `processed/`.
4. Colab en orden: `build_features.ipynb` → `train_svm.ipynb` → `hard_neg.ipynb` (opcional) → `infer.ipynb`.
5. Cada notebook llama a `link_processed_to_drive()` → escrituras directas a Drive con write atómico.
6. Salida final: `predictions/classical_test.json` en Drive.

Si Colab desconecta: keep-alive JS + *Run all* al reabrir = reanuda desde último checkpoint en Drive.
