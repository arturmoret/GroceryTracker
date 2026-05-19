"""Genera el PDF de onboarding para compañeros que se suman al proyecto."""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Estilo blanco + morado
DEEP_PURPLE = HexColor("#6B21A8")
MEDIUM_PURPLE = HexColor("#9333EA")
LIGHT_PURPLE_BG = HexColor("#F5F0FA")
ACCENT_PURPLE_BORDER = HexColor("#C084FC")
DARK_TEXT = HexColor("#1F2937")
CODE_BG = HexColor("#F4F4F6")
TABLE_HEADER_BG = HexColor("#6B21A8")
TABLE_ALT_ROW = HexColor("#FAF5FF")
RULE_GRAY = HexColor("#E5E7EB")

ss = getSampleStyleSheet()

style_title = ParagraphStyle("Title", parent=ss["Title"], fontName="Helvetica-Bold",
                             fontSize=20, textColor=DEEP_PURPLE, alignment=TA_LEFT,
                             spaceAfter=4, leading=24)
style_h1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                          fontSize=14, textColor=DEEP_PURPLE, spaceBefore=12,
                          spaceAfter=6, leading=18)
style_h2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                          fontSize=11, textColor=MEDIUM_PURPLE, spaceBefore=8,
                          spaceAfter=4, leading=14)
style_body = ParagraphStyle("Body", parent=ss["BodyText"], fontName="Helvetica",
                            fontSize=10, textColor=DARK_TEXT, leading=14,
                            alignment=TA_JUSTIFY, spaceAfter=6)
style_body_tight = ParagraphStyle("BodyTight", parent=style_body, spaceAfter=2)
style_bullet = ParagraphStyle("Bullet", parent=style_body, leftIndent=14,
                              bulletIndent=2, spaceAfter=2, alignment=TA_LEFT)
style_code = ParagraphStyle("Code", parent=ss["Code"], fontName="Courier",
                            fontSize=9, textColor=DARK_TEXT, leading=12,
                            leftIndent=6, rightIndent=6, spaceBefore=2, spaceAfter=2)
style_callout_title = ParagraphStyle("CalloutTitle", parent=style_h2, fontSize=11,
                                     textColor=DEEP_PURPLE, spaceBefore=0, spaceAfter=4)
style_callout = ParagraphStyle("Callout", parent=style_body, alignment=TA_LEFT, spaceAfter=0)


def callout(title: str, body_html: str) -> Table:
    inner = [[Paragraph(title, style_callout_title)],
             [Paragraph(body_html, style_callout)]]
    t = Table(inner, colWidths=[15 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_PURPLE_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT_PURPLE_BORDER),
    ]))
    return t


def code_block(lines: list[str]) -> Table:
    txt = "<br/>".join(line.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;") for line in lines)
    t = Table([[Paragraph(txt, style_code)]], colWidths=[15 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def styled_table(header: list[str], rows: list[list[str]], col_widths: list[float]) -> Table:
    data = [header] + rows
    t = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_TEXT),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, DEEP_PURPLE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE_GRAY),
    ]
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT_ROW))
    t.setStyle(TableStyle(style))
    return t


def bullets(items: list[str]) -> list:
    return [Paragraph(f"<font color='#9333EA'>&#8226;</font>&nbsp;&nbsp;{x}", style_bullet) for x in items]


def build_story() -> list:
    story = []

    story.append(Paragraph("Onboarding — Detección de productos: Clásico vs Deep Learning", style_title))
    story.append(Paragraph(
        "<font color='#6B7280'>Bienvenidos. Doc directa para que tengáis exactamente lo mismo que tenemos "
        "y podáis empezar a aportar. ~5-10 min de lectura + ~30 min de setup.</font>",
        style_body_tight,
    ))
    story.append(Spacer(1, 4))

    # --- El proyecto en 30 segundos ---
    story.append(Paragraph("El proyecto en 30 segundos", style_h1))
    story.append(Paragraph(
        "ABP de Visión por Computador (UAB). Implementamos <b>dos pipelines paralelos de detección</b> "
        "sobre el MISMO dataset (un subset de MVTec D2S, 20 productos compatibles con Mercadona) y los comparamos:",
        style_body,
    ))
    story.extend(bullets([
        "<b>Pipeline A — Visión clásica</b>: Selective Search + descriptores HOG y SIFT/BoVW + SVM χ², con preprocesado handcrafted (CLAHE + white balance + denoise bilateral).",
        "<b>Pipeline B — Deep Learning</b>: YOLOv8s pre-entrenado en COCO, fine-tuned sobre nuestras clases.",
    ]))
    story.append(Paragraph(
        "El entregable académico no es ganar a YOLO con el clásico — es <b>cuantificar la brecha</b> y entender "
        "<b>dónde se rompe cada paradigma</b> (especialmente en triadas confusables: 4 manzanas, 3 pastas Reggia, "
        "2 Coca-Cola...).",
        style_body,
    ))

    # --- Callout: regla fundamental ---
    story.append(callout(
        "La regla fundamental",
        "Para que la comparación sea válida, los dos pipelines comparten:<br/>"
        "&nbsp;&nbsp;<b>·</b> el mismo dataset y los mismos splits train / val / test;<br/>"
        "&nbsp;&nbsp;<b>·</b> las mismas 20 etiquetas con los mismos IDs (1 a 20);<br/>"
        "&nbsp;&nbsp;<b>·</b> las mismas métricas (mAP@0.5, F1, IoU, matriz de confusión, tiempo);<br/>"
        "&nbsp;&nbsp;<b>·</b> el mismo hardware para medir tiempos (CPU local + Colab T4).<br/><br/>"
        "<b>Si tocáis algo de eso, decidlo en el grupo antes</b> — un cambio unilateral invalida toda la comparativa.",
    ))

    # --- Estado actual ---
    story.append(Paragraph("Estado actual del proyecto", style_h1))
    story.append(styled_table(
        ["Hito", "Estado", "Qué"],
        [
            ["H1", "✓ Hecho", "EDA + selección de 20 clases verificadas en D2S"],
            ["H2", "✓ Hecho", "Splits filtrados train/val/test en formato COCO"],
            ["H3", "✓ Hecho", "Pipeline clásico (proposals + descriptores + codebook BoVW)"],
            ["H4", "✓ Hecho (código)", "Entrenamiento + inferencia. Entreno full pendiente (vía Colab)"],
            ["H5", "✓ Hecho (código)", "Hard negative mining"],
            ["Preprocess", "✓ Hecho", "CLAHE + WB + denoise bilateral integrados y configurables"],
            ["H6", "Pendiente", "YOLOv8s fine-tuned en Colab T4"],
            ["H7", "Pendiente", "Framework evaluación común (mAP, F1, confusión)"],
            ["H8", "Pendiente", "Análisis de robustez (ruido, blur, iluminación, oclusión)"],
            ["H9", "Opcional", "Demo webcam"],
            ["H10", "Pendiente", "Informe + presentación"],
        ],
        col_widths=[1.8 * cm, 2.6 * cm, 10.6 * cm],
    ))

    # --- Setup paso a paso ---
    story.append(Paragraph("Setup desde 0 (Windows + PowerShell)", style_h1))

    steps = [
        ("Pre-requisitos", "Instalar <b>Git for Windows</b> (git-scm.com) y <b>VSCode</b> (code.visualstudio.com). No hace falta Python aparte — <font face='Courier'>uv</font> lo gestiona."),
        ("Clonar el repo", "Pedidle a Artur la URL del repositorio si no la tenéis ya. Ejecutar en PowerShell donde queráis el proyecto:"),
        ("Setup automático", "Instala <font face='Courier'>uv</font> si falta y sincroniza dependencias (~5-15 min, baja PyTorch ~2 GB):"),
        ("Conseguir MVTec D2S", "El dataset (~6 GB) <b>no está en git</b>. Dos opciones:<br/>&nbsp;&nbsp;<b>·</b> Pedírselo a Artur (lo tiene extraído, os pasa la carpeta <font face='Courier'>data/d2s</font>).<br/>&nbsp;&nbsp;<b>·</b> Bajarlo de <font color='#6B21A8'>https://www.mvtec.com/company/research/datasets/mvtec-d2s</font> (rellenar formulario, aceptar licencia, descargar <font face='Courier'>d2s_images_v*.tar.xz</font> y <font face='Courier'>d2s_annotations_v*.tar.xz</font>, moverlos a <font face='Courier'>data/raw/</font> y correr <font face='Courier'>uv run python scripts/prepare_d2s.py</font>).<br/>Resultado deseado: <font face='Courier'>data/d2s/images/</font> con ~21k .jpg y <font face='Courier'>data/d2s/annotations/</font> con los .json de D2S."),
        ("Generar splits filtrados (rápido, ~10s)", "Filtra D2S a las 20 clases y crea train/val/test estratificados con seed=42 (reproducible):"),
        ("Entrenar codebook BoVW (~1 min)", "Vocabulario visual de K=300 palabras. Determinista con seed=42:"),
        ("Smoke test del pipeline clásico (~2 min)", "Si imprime <font face='Courier'>Pipeline clasico funciona end-to-end sin crashes</font>, vuestro entorno está OK:"),
    ]
    cmds = [
        None,
        ["git clone <URL del repo>", "cd grocery-tracker"],
        [".\\scripts\\setup.ps1"],
        None,
        ["uv run python scripts/prepare_splits.py"],
        ["uv run python scripts/train_codebook.py"],
        ["uv run python scripts/test_classical_tiny.py"],
    ]
    for i, ((title, body_text), cmd) in enumerate(zip(steps, cmds), start=1):
        block = [
            Paragraph(f"<font color='#6B21A8'><b>{i}. {title}</b></font>", style_body_tight),
            Paragraph(body_text, style_body),
        ]
        if cmd:
            block.append(code_block(cmd))
        block.append(Spacer(1, 4))
        story.append(KeepTogether(block))

    # --- Mac/Linux nota ---
    story.append(Paragraph("Mac / Linux", style_h2))
    story.append(Paragraph(
        "Mismo flujo. Solo cambia el setup: <font face='Courier'>bash scripts/setup.sh</font> en lugar del <font face='Courier'>.ps1</font>. "
        "Si trabajáis en Mac y al correr <font face='Courier'>scripts/run_classical_train.py</font> sklearn rompe por OOM, abrir "
        "<font face='Courier'>configs/classical.yaml</font> y verificar que <font face='Courier'>classifier.n_jobs: 1</font> y "
        "<font face='Courier'>classifier.sample_steps: 1</font> (ya están así por defecto — son los ajustes memoria-safe para Macs 8GB).",
        style_body,
    ))

    # --- Notebooks ---
    story.append(Paragraph("Notebooks recomendados (para empaparse del proyecto)", style_h1))
    story.append(styled_table(
        ["Notebook", "Qué muestra"],
        [
            ["00_dataset_eda", "El dataset MVTec D2S y validación de las 20 clases"],
            ["01_class_selection", "Cómo se construyen los splits + visualizaciones"],
            ["02_classical_dev", "Cada componente del clásico (SS, HOG, SIFT, BoVW)"],
            ["03_training_visualization", "Cómo se etiquetan los proposals (positivos / negativos)"],
            ["04_classical_results", "Detecciones del modelo (necesita entreno previo)"],
            ["05_preprocessing_viz", "CLAHE + WB + denoise visualmente"],
        ],
        col_widths=[5.5 * cm, 9.5 * cm],
    ))

    # --- Lo que podéis coger ---
    story.append(Paragraph("Tareas pendientes que podéis coger", style_h1))
    story.append(styled_table(
        ["Tarea", "Qué hace falta"],
        [
            ["H6 — YOLOv8s en Colab", "Cuenta Colab. Fine-tune YOLOv8s sobre los splits ya generados (formato YOLO se exporta desde el COCO). Augmentation estándar de ultralytics (mosaic, hsv, flip H). Output: predicciones COCO JSON sobre el mismo test."],
            ["H7 — Framework eval", "Módulo nuevo en src/grocery_detection/eval/: mAP@0.5 y @0.5:0.95 (pycocotools), F1 por clase, matriz de confusión 20×20, curvas PR, timing. Recibe los 2 JSON (clásico + YOLO) y produce tablas/figuras."],
            ["H8 — Robustez", "Generar test perturbado (ruido gaussiano σ ∈ {0.01, 0.05, 0.1}, blur σ ∈ {1, 3, 5}, brillo {0.5, 0.75, 1.25, 1.5}, JPEG q ∈ {70, 50, 30}, oclusión 20%). Recalcular métricas. Curvas de degradación."],
            ["H10 — Informe", "Estructura del documento académico + figuras / tablas auto-generadas. Cuando H6 + H7 estén hechos."],
        ],
        col_widths=[5.5 * cm, 9.5 * cm],
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Sugerencia de reparto</b>: uno coge H6 (YOLO en Colab), otro coge H7 (eval framework). "
        "H7 se puede empezar en paralelo a H6 — solo necesita los JSON, que podemos mockear inicialmente para "
        "probar la lógica.",
        style_body,
    ))

    # --- Footer ---
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<font color='#9CA3AF'>Briefing técnico completo en <font face='Courier'>PROYECTO.md</font> del repo. "
        "Dudas: preguntar a Artur. <b>No tocar nada que afecte a splits, IDs de clase o métricas sin avisar.</b></font>",
        style_body_tight,
    ))

    return story


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "reports" / "onboarding_colaboradores.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Onboarding compañeros — Detección Clásico vs DL",
        author="Artur Moret",
    )
    doc.build(build_story())
    print(f"PDF generated: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
