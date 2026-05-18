"""Genera un PDF overview del proyecto para compañeros (H1-H5)."""

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

# ---------------- Colors (white + purple) ----------------
DEEP_PURPLE = HexColor("#6B21A8")
MEDIUM_PURPLE = HexColor("#9333EA")
LIGHT_PURPLE_BG = HexColor("#F5F0FA")
ACCENT_PURPLE_BORDER = HexColor("#C084FC")
DARK_TEXT = HexColor("#1F2937")
MUTED_TEXT = HexColor("#4B5563")
CODE_BG = HexColor("#F4F4F6")
TABLE_HEADER_BG = HexColor("#6B21A8")
TABLE_ALT_ROW = HexColor("#FAF5FF")
RULE_GRAY = HexColor("#E5E7EB")

# ---------------- Styles ----------------
ss = getSampleStyleSheet()

style_title = ParagraphStyle(
    "Title",
    parent=ss["Title"],
    fontName="Helvetica-Bold",
    fontSize=20,
    textColor=DEEP_PURPLE,
    alignment=TA_LEFT,
    spaceAfter=4,
    leading=24,
)

style_h1 = ParagraphStyle(
    "H1",
    parent=ss["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=14,
    textColor=DEEP_PURPLE,
    spaceBefore=14,
    spaceAfter=6,
    leading=18,
)

style_h2 = ParagraphStyle(
    "H2",
    parent=ss["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=11,
    textColor=MEDIUM_PURPLE,
    spaceBefore=10,
    spaceAfter=4,
    leading=14,
)

style_body = ParagraphStyle(
    "Body",
    parent=ss["BodyText"],
    fontName="Helvetica",
    fontSize=10,
    textColor=DARK_TEXT,
    leading=14,
    alignment=TA_JUSTIFY,
    spaceAfter=6,
)

style_body_tight = ParagraphStyle(
    "BodyTight",
    parent=style_body,
    spaceAfter=2,
)

style_bullet = ParagraphStyle(
    "Bullet",
    parent=style_body,
    leftIndent=14,
    bulletIndent=2,
    spaceAfter=2,
    alignment=TA_LEFT,
)

style_callout = ParagraphStyle(
    "Callout",
    parent=style_body,
    fontName="Helvetica",
    fontSize=10,
    textColor=DARK_TEXT,
    leading=14,
    alignment=TA_LEFT,
    spaceAfter=0,
)

style_callout_title = ParagraphStyle(
    "CalloutTitle",
    parent=style_h2,
    fontSize=11,
    textColor=DEEP_PURPLE,
    spaceBefore=0,
    spaceAfter=4,
)

style_code = ParagraphStyle(
    "Code",
    parent=ss["Code"],
    fontName="Courier",
    fontSize=9,
    textColor=DARK_TEXT,
    leading=12,
    leftIndent=6,
    rightIndent=6,
    spaceBefore=2,
    spaceAfter=2,
)

style_step_num = ParagraphStyle(
    "StepNum",
    parent=style_body,
    fontName="Helvetica-Bold",
    fontSize=10,
    textColor=DEEP_PURPLE,
    spaceAfter=0,
)


def callout(title: str, body_html: str) -> Table:
    """Light-purple box with a darker purple left border."""
    inner = [
        [Paragraph(title, style_callout_title)],
        [Paragraph(body_html, style_callout)],
    ]
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
    """Monospace code block with subtle gray background."""
    txt = "<br/>".join(line.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;") for line in lines)
    p = Paragraph(txt, style_code)
    t = Table([[p]], colWidths=[15 * cm])
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


# ---------------- Document ----------------
def build_story() -> list:
    story = []

    # Title (no portada, directo)
    story.append(Paragraph("Detección de productos: Visión Clásica vs Deep Learning", style_title))
    story.append(Paragraph(
        "<font color='#6B7280'>Estado del proyecto ABP — Hitos 1 a 5 completados (código). "
        "Pipeline clásico listo para entrenar.</font>",
        style_body_tight,
    ))
    story.append(Spacer(1, 4))

    # --- Resumen rápido ---
    story.append(Paragraph("Resumen rápido", style_h1))
    story.append(Paragraph(
        "Implementamos <b>dos pipelines paralelos de detección de objetos</b> sobre el mismo dataset "
        "de productos de supermercado:",
        style_body,
    ))
    story.extend(bullets([
        "<b>Pipeline A — Visión clásica</b>: Selective Search + descriptores manuales (HOG, SIFT/BoVW) + SVM con kernel chi².",
        "<b>Pipeline B — Deep Learning</b>: YOLOv8s pre-entrenado en COCO, fine-tuned sobre nuestras clases.",
    ]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Ambos predicen <b>las mismas etiquetas</b> sobre <b>las mismas imágenes</b> y se evalúan con "
        "<b>las mismas métricas</b>. El entregable académico no es ganar a YOLO con el clásico — es "
        "<b>cuantificar la brecha y entender dónde se rompe cada paradigma</b>.",
        style_body,
    ))

    # --- Callout: regla fundamental ---
    story.append(Spacer(1, 6))
    story.append(callout(
        "La regla fundamental: misma base de comparación",
        "Para que la diferencia entre clásico y DL sea atribuible <b>al método</b> y no a otra variable, "
        "los dos pipelines comparten:<br/>"
        "&nbsp;&nbsp;<b>·</b> El mismo dataset y los mismos splits train / val / test.<br/>"
        "&nbsp;&nbsp;<b>·</b> Las mismas 20 etiquetas con los mismos IDs (1 a 20).<br/>"
        "&nbsp;&nbsp;<b>·</b> Las mismas métricas (mAP@0.5, F1, IoU, matriz de confusión, tiempo de inferencia).<br/>"
        "&nbsp;&nbsp;<b>·</b> El mismo hardware para medir tiempos (CPU local + Colab T4).<br/><br/>"
        "Si alguna de estas cambia, no podemos concluir que un paradigma sea mejor — podría serlo solo por "
        "haber tenido más datos o un test más fácil. Este principio guía toda la arquitectura del proyecto.",
    ))

    # --- Dataset ---
    story.append(Paragraph("El dataset: MVTec D2S", style_h1))
    story.append(Paragraph(
        "Usamos <b>MVTec D2S</b> (Densely Segmented Supermarket): 21.000 imágenes con anotación bounding-box + "
        "máscara de 60 productos europeos. El train es \"catálogo\" (un objeto por imagen, fondo limpio); "
        "el val son escenas <b>cluttered</b> con varios productos juntos — el caso de uso realista.",
        style_body,
    ))
    story.append(Paragraph(
        "Filtramos a un <b>subset de 20 clases verificadas en Mercadona España</b>, organizadas en grupos "
        "<b>visualmente confusables</b>:",
        style_body,
    ))
    story.extend(bullets([
        "4 manzanas (Braeburn, Golden Delicious, Granny Smith, Red Boskoop) — misma forma, color sutil distinto.",
        "3 pasta Reggia (elicoidali, fusilli, spaghetti) — mismo packaging azul, distinta forma a través del plástico.",
        "2 Coca-Cola (clásica + light) — misma lata, color rojo vs plata.",
        "2 barritas Corny (chocolate-plátano + nueces) — mismo formato, distinto sabor.",
        "2 uvas (verde + morada), 2 tomates (vine + roma-vine), 2 clementinas (malla + sueltas).",
        "3 singletons: zanahoria, pepino, kiwi.",
    ]))
    story.append(Paragraph(
        "<b>Por qué grupos confusables y no clases muy distintas</b>: distinguir manzana de coca-cola lo hace "
        "cualquier sistema. La diferencia entre clásico y DL solo se ve en distinciones finas — variedades de "
        "manzana, formas de pasta de la misma marca, sabores Corny. <b>Ese contraste fine-grained es donde vive "
        "el valor académico del informe.</b>",
        style_body,
    ))

    # --- Qué se ha hecho ---
    story.append(Paragraph("Qué se ha hecho (H1 a H5)", style_h1))
    story.append(styled_table(
        ["Hito", "Qué", "Entregable"],
        [
            ["H1", "EDA + selección de las 20 clases", "00_dataset_eda.ipynb, classes.yaml"],
            ["H2", "Splits filtrados train / val / test", "data/processed/{train,val,test}.json"],
            ["H3", "Pipeline clásico: componentes + codebook BoVW", "classical/proposals.py, descriptors/*, codebook.pkl"],
            ["H4", "Entrenamiento + inferencia clásica", "scripts/run_classical_{train,infer}.py"],
            ["H5", "Hard negative mining", "scripts/run_classical_hard_neg.py"],
        ],
        col_widths=[1.5 * cm, 6 * cm, 7.5 * cm],
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Conteos resultantes: train <b>1650 imágenes</b> (2130 instancias, catálogo), val <b>603 imágenes</b> "
        "(1476 instancias, para early stopping), test <b>1407 imágenes</b> (3377 instancias, cluttered). "
        "Estratificado por clase dominante, seed fija 42 → reproducible.",
        style_body,
    ))

    # --- Pipeline clásico cómo funciona ---
    story.append(Paragraph("Cómo funciona el pipeline clásico (alto nivel)", style_h1))
    story.append(Paragraph("Flujo de un detector clásico sin redes neuronales:", style_body))
    story.append(code_block([
        "imagen entrada",
        "    |",
        "    v",
        "Selective Search  -->  ~300 cajas candidatas (sin clase aun)",
        "    |",
        "    v",
        "Por cada caja:",
        "    HOG          -->  silueta global (~1700 dims)",
        "    SIFT + BoVW  -->  textura agregada (~300 dims)",
        "    concat L2    -->  vector ~2000 dims",
        "    |",
        "    v",
        "20 SVMs one-vs-rest (kernel chi^2 aproximado)  -->  score por clase",
        "    |",
        "    v",
        "Filtrado por confianza  +  NMS por clase",
        "    |",
        "    v",
        "Hard negative mining (2 rondas):",
        "    detecta falsos positivos -> reentrena",
        "    |",
        "    v",
        "detecciones finales (bbox + clase + score)",
    ]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("En lenguaje natural:", style_body))
    story.extend(bullets([
        "<b>Selective Search</b> propone dónde podría haber objetos sin saber qué son (~300 cajas por imagen).",
        "<b>HOG</b> describe la silueta de cada caja (perfil de bordes).",
        "<b>SIFT + BoVW</b> describe la textura como un histograma de \"palabras visuales\" aprendidas previamente.",
        "<b>SVM chi²</b> decide para cada caja si es producto X, producto Y... o nada (background).",
        "<b>NMS</b> (Non-Maximum Suppression) elimina detecciones duplicadas que se solapan mucho.",
        "<b>Hard negative mining</b>: corremos el modelo entrenado sobre el train, detectamos las cajas que clasifica como producto pero no lo son, las añadimos al pool de negativos y reentrenamos. Repetimos 2-3 veces. Reduce mucho los falsos positivos.",
    ]))

    # --- Lo que falta ---
    story.append(Paragraph("Qué falta (H6 a H10)", style_h1))
    story.append(styled_table(
        ["Hito", "Tarea", "Dónde corre"],
        [
            ["H6", "Pipeline B: YOLOv8s fine-tuned sobre las MISMAS 20 clases", "Google Colab T4"],
            ["H7", "Framework de evaluación común (mAP, F1, IoU, confusión, tiempos)", "CPU local"],
            ["H8", "Análisis de robustez (ruido, blur, iluminación, JPEG, oclusión)", "CPU local"],
            ["H9", "Demo webcam en vivo (opcional)", "CPU local"],
            ["H10", "Informe + presentación con tablas y figuras auto-generadas", "—"],
        ],
        col_widths=[1.5 * cm, 9.5 * cm, 4 * cm],
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Hipótesis a verificar en H7-H8</b>: YOLOv8s va a ganar al clásico en mAP@0.5 con margen amplio, "
        "y la diferencia será <b>más grande en las triadas confusables</b> (manzanas, pasta Reggia) que en "
        "los singletons. En H8, esperamos que el clásico aguante mejor la oclusión que YOLO pero peor el ruido "
        "gaussiano y el blur.",
        style_body,
    ))

    # --- Tutorial setup ---
    story.append(Paragraph("Setup desde 0 en Windows", style_h1))
    story.append(Paragraph(
        "Pasos para llegar desde un PC vacío hasta el punto donde estamos ahora "
        "(pipeline clásico verificado, listo para entrenar).",
        style_body,
    ))

    steps = [
        ("Pre-requisitos", "Instalar <b>Git for Windows</b> (git-scm.com) y <b>VSCode</b> (code.visualstudio.com). No hace falta instalar Python por separado — uv lo gestiona."),
        ("Clonar el repo", "Abrir PowerShell en la carpeta donde quieras el proyecto y ejecutar:"),
        ("Setup automático", "Instala uv si falta y sincroniza dependencias (~5-15 min, baja PyTorch ~2 GB):"),
        ("Descargar MVTec D2S manualmente", "Ir a <font color='#6B21A8'>https://www.mvtec.com/company/research/datasets/mvtec-d2s</font>, rellenar formulario y aceptar licencia. Descargar los dos archivos <font face='Courier'>d2s_images_v*.tar.xz</font> (~6 GB) y <font face='Courier'>d2s_annotations_v*.tar.xz</font> (~40 MB). Moverlos a <font face='Courier'>data/raw/</font>. Si los descomprimes manualmente, mete las carpetas <font face='Courier'>images/</font> y <font face='Courier'>annotations/</font> directamente en <font face='Courier'>data/d2s/</font> y salta el paso 5."),
        ("Extraer el dataset", "Solo si tienes los .tar.xz sin descomprimir:"),
        ("Generar splits filtrados a 20 clases", "Output: data/processed/{train,val,test}.json (~10s):"),
        ("Entrenar codebook BoVW", "Vocabulario visual de K=300 palabras vía k-means sobre SIFTs de 300 imágenes train (~1 min):"),
        ("Smoke test del pipeline clásico", "Entrena con 5 imgs e infiere sobre 3 (~1-2 min). Si imprime <font face='Courier'>Pipeline clasico funciona end-to-end sin crashes</font>, todo OK:"),
    ]
    cmds = [
        None,
        ["git clone <URL del repo>", "cd grocery-tracker"],
        [".\\scripts\\setup.ps1"],
        None,
        ["uv run python scripts/prepare_d2s.py"],
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

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Entrenamiento e inferencia completos del pipeline clásico</b> (~5-8 h totales, dejar de noche):",
        style_body,
    ))
    story.append(code_block([
        "uv run python scripts/run_classical_train.py      # ~1-2 h",
        "uv run python scripts/run_classical_hard_neg.py   # ~1.5-3 h",
        "uv run python scripts/run_classical_infer.py      # ~1.5-2 h",
    ]))
    story.append(Paragraph(
        "Output final: <font face='Courier'>reports/predictions/classical_test.json</font> "
        "(predicciones en formato COCO listas para evaluación común en H7).",
        style_body,
    ))

    # Footer-ish
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<font color='#9CA3AF'>Briefing técnico completo en PROYECTO.md del repo. Detalles de cada hito en "
        "los notebooks <font face='Courier'>00_dataset_eda.ipynb</font>, "
        "<font face='Courier'>01_class_selection.ipynb</font>, "
        "<font face='Courier'>02_classical_dev.ipynb</font>.</font>",
        style_body_tight,
    ))

    return story


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "reports" / "H1-H5_overview.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Proyecto ABP — Detección Visión Clásica vs DL",
        author="Artur Moret",
    )

    doc.build(build_story())
    print(f"PDF generated: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
