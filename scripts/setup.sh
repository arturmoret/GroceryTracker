#!/usr/bin/env bash
set -euo pipefail

echo "==> Grocery Detection - H1 setup"

# 1. uv install if missing
if ! command -v uv >/dev/null 2>&1; then
    echo "uv no encontrado. Instalando..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv instalado pero no está en PATH. Reinicia la terminal y vuelve a ejecutar."
        exit 1
    fi
fi
echo "uv: $(uv --version)"

# 2. Sync dependencies
echo "==> uv sync --all-extras"
uv sync --all-extras

# 3. Data dirs
mkdir -p data/raw data/d2s data/processed

cat <<'EOF'

==> Setup completado.

Siguientes pasos:
  1. Descargar D2S: https://www.mvtec.com/company/research/datasets/mvtec-d2s
  2. Aceptar licencia, descargar d2s_images_v1.tar.xz y d2s_annotations_v1.tar.xz
  3. Mover ambos archivos a: data/raw/
  4. Extraer:   uv run python scripts/prepare_d2s.py
  5. Abrir EDA: uv run jupyter notebook notebooks/00_dataset_eda.ipynb
EOF
