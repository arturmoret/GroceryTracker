# Correr el pipeline clásico (H1-H5) en Google Colab

Guía paso a paso para ejecutar todo el pipeline clásico sin tocar tu PC local (más allá de subir archivos a Google Drive).

> Si en algún momento un paso "no va", revisa la sección [Troubleshooting](#troubleshooting) al final.

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

Después: descarga D2S a `data/raw/` (mismos archivos que ya tienes en Drive — o copia desde Drive si tienes desktop sync), y:

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

- **Duración estimada**: 2-5 horas (depende del slot de CPU que te toque).
- **Output en Drive**: `processed/classical_features.npz` (~150 MB).
- **Checkpoint**: cada 100 imgs. Si Colab desconecta, vuelves a abrir y *Run all* — reanuda donde quedó.

Si nunca corriste el paso 2 (los baratos en local), descomenta las dos últimas celdas del notebook (`# run_script("scripts/prepare_splits.py")` y `# run_script("scripts/train_codebook.py")`) y córrelas primero.

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
- **Checkpoint**: cada 100 imgs vuelca el JSON. Si Colab desconecta, *Run all* y reanuda.

## 4. ¿Qué hago si Colab desconecta?

Es habitual. Free tier corta sesiones tras ~90 min de inactividad o ~12 h activas.

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

Hace click virtual cada 60 s. No es bulletproof pero aguanta varias horas.

### Mitigación 2 — Reanudar manualmente

Si peta:

1. Reabre el notebook.
2. *Runtime → Run all*.
3. Las celdas iniciales recargan el state desde Drive (mount + clone + sync_from_drive).
4. El script detecta el checkpoint y salta las imágenes ya procesadas.

**Caveat actual**: el checkpoint se vuelca a `/content/` (volátil), y solo se sube a Drive al **final** del notebook. Si Colab muere antes de la última celda, lo procesado en esa sesión se pierde. Cada *Run all* recupera el estado de la **última sesión completa**.

> Fix robusto pendiente: symlinkar `data/processed/` directo a Drive para que cada checkpoint se escriba directo allí. Si quieres, dímelo y lo aplico.

### Mitigación 3 — Colab Pro

~10 €/mes. Sesiones de 24 h, background execution (¡corre sin tener la tab abierta!), CPU mejor. Un mes durante el proyecto puede compensar.

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
    .hardneg_state.json          ← marcador de rondas completadas (H5)

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

### "(skip) train.json no existe en Drive"

Falta el paso 2 — `prepare_splits.py` no se ha corrido (ni local ni en Colab). O bien:

- En local: córrelo y sube los JSON a Drive.
- En el notebook `colab_build_features.ipynb`: descomenta la celda `# run_script("scripts/prepare_splits.py")` al final.

### "(skip) codebook.pkl no existe en Drive"

Igual que el anterior pero para `train_codebook.py`.

### "El runtime de Colab se ha desconectado"

Ver sección 4. Vuelve a abrir, *Run all*, reanuda.

### "ImportError: No module named cv2.ximgproc"

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

El `.pkl` artifact tiene una key distinta a las esperadas (`sampler`, `svm`, `target_class_ids`). Verifica que la celda *3. Entrenar SVM* terminó OK y que la *4. Guardar artifact* corrió después.

---

## TL;DR

1. Drive: crea `MyDrive/grocery-detection/{raw,processed,predictions}/`.
2. Sube D2S `.tar.xz` a `raw/`.
3. (Local barato) `prepare_splits.py` + `train_codebook.py` → sube JSONs + codebook a `processed/`.
4. Colab en orden: `build_features.ipynb` → `train_svm.ipynb` → `hard_neg.ipynb` (opcional) → `infer.ipynb`.
5. Salida final: `predictions/classical_test.json` en Drive.

Si Colab desconecta: keep-alive JS + *Run all* manual al reabrir = reanuda desde último checkpoint.
