# data/raw/

Aquí van los archivos descargados de MVTec D2S:

- `d2s_images_v1.tar.xz`
- `d2s_annotations_v1.tar.xz`

Descarga: https://www.mvtec.com/company/research/datasets/mvtec-d2s (requiere aceptar licencia).

Una vez ambos archivos estén en este directorio, ejecutar desde la raíz del repo:

```powershell
uv run python scripts/prepare_d2s.py
```

Esto los extraerá en `data/d2s/` con la estructura esperada.

**Los archivos `.tar.xz` están en `.gitignore` — no se suben al repo.**
