#Requires -Version 5.1
$ErrorActionPreference = "Stop"

Write-Host "==> Grocery Detection - H1 setup" -ForegroundColor Cyan

# 1. uv install if missing
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv no encontrado. Instalando..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Refresh PATH for current session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "uv instalado pero no encontrado en PATH. Reinicia la terminal y vuelve a ejecutar." -ForegroundColor Red
        exit 1
    }
}
Write-Host "uv: $(uv --version)" -ForegroundColor Green

# 2. Sync dependencies (incluye dev extras para jupyter)
Write-Host "==> uv sync --all-extras" -ForegroundColor Cyan
uv sync --all-extras

# 3. Ensure data dirs exist
$null = New-Item -ItemType Directory -Force -Path "data/raw"
$null = New-Item -ItemType Directory -Force -Path "data/d2s"
$null = New-Item -ItemType Directory -Force -Path "data/processed"

Write-Host ""
Write-Host "==> Setup completado." -ForegroundColor Green
Write-Host ""
Write-Host "Siguientes pasos:" -ForegroundColor Yellow
Write-Host "  1. Descargar D2S: https://www.mvtec.com/company/research/datasets/mvtec-d2s"
Write-Host "  2. Aceptar licencia, descargar d2s_images_v1.tar.xz y d2s_annotations_v1.tar.xz"
Write-Host "  3. Mover ambos archivos a: data/raw/"
Write-Host "  4. Extraer:   uv run python scripts/prepare_d2s.py"
Write-Host "  5. Abrir EDA: uv run jupyter notebook notebooks/00_dataset_eda.ipynb"
