"""Genera un primer diseño de póster A0 horizontal a partir de resultados ya publicados.

El contenido procede de results/analysis/informe.md y results/d23/resumen.md.
No modifica los resultados del experimento.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "poster_v0_a0.pdf"
MM = 72 / 25.4
W, H = 1189 * MM, 841 * MM

REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
pdfmetrics.registerFont(TTFont("DejaVu", REGULAR))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", BOLD))

INK = HexColor("#162A38")
MUTED = HexColor("#4C6270")
TEAL = HexColor("#0D7E7A")
TEAL_LIGHT = HexColor("#E1F2EE")
ORANGE = HexColor("#DF7046")
ORANGE_LIGHT = HexColor("#FBECE5")
BLUE = HexColor("#3968A6")
BLUE_LIGHT = HexColor("#E9F0F9")
PAPER = HexColor("#F8FAF9")
WHITE = HexColor("#FFFFFF")
LINE = HexColor("#D9E3E5")


def xy(x: float, y: float) -> tuple[float, float]:
    return x * MM, H - y * MM


def rect(c: canvas.Canvas, x: float, y: float, w: float, h: float, fill, radius: float = 0) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(fill)
    px, py = x * MM, H - (y + h) * MM
    if radius:
        c.roundRect(px, py, w * MM, h * MM, radius * MM, fill=1, stroke=0)
    else:
        c.rect(px, py, w * MM, h * MM, fill=1, stroke=0)


def line(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=LINE, width: float = 0.6) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(width * MM)
    c.line(*xy(x1, y1), *xy(x2, y2))


def txt(c: canvas.Canvas, x: float, y: float, text: str, size: float, color=INK,
        bold: bool = False, align: str = "left") -> None:
    c.setFillColor(color)
    c.setFont("DejaVu-Bold" if bold else "DejaVu", size)
    px, py = xy(x, y)
    if align == "center":
        c.drawCentredString(px, py, text)
    elif align == "right":
        c.drawRightString(px, py, text)
    else:
        c.drawString(px, py, text)


def wrapped(c: canvas.Canvas, x: float, y: float, text: str, width: float,
            size: float, leading: float, color=INK, bold: bool = False) -> float:
    font = "DejaVu-Bold" if bold else "DejaVu"
    words = text.split()
    rows: list[str] = []
    row = ""
    max_pt = width * MM
    for word in words:
        candidate = f"{row} {word}" if row else word
        if pdfmetrics.stringWidth(candidate, font, size) <= max_pt:
            row = candidate
        else:
            if row:
                rows.append(row)
            row = word
    if row:
        rows.append(row)
    for i, item in enumerate(rows):
        txt(c, x, y + i * leading / MM, item, size, color, bold)
    return y + len(rows) * leading / MM


def section(c: canvas.Canvas, x: float, y: float, number: str, title: str) -> None:
    txt(c, x, y, number, 28, TEAL, True)
    txt(c, x + 16, y, title, 28, INK, True)
    line(c, x, y + 7, x + 280, y + 7)


def pill(c: canvas.Canvas, x: float, y: float, w: float, label: str,
         bg=TEAL_LIGHT, fg=TEAL) -> None:
    rect(c, x, y, w, 16, bg, 8)
    txt(c, x + w / 2, y + 11.4, label, 17, fg, True, "center")


def main_chart(c: canvas.Canvas) -> None:
    x, y, w, h = 340, 263, 514, 302
    rect(c, x, y, w, h, WHITE, 9)
    txt(c, x + 20, y + 32, "RESULTADO PRINCIPAL", 24, TEAL, True)
    txt(c, x + 20, y + 61, "Mejora mediana del MAE", 33, INK, True)
    txt(c, x + 20, y + 82, "respecto a OLS con atributos originales", 20, MUTED)

    names = ["A  Selección", "B  Síntesis", "C  Árboles", "C  GEP"]
    colors = [BLUE, ORANGE, TEAL, HexColor("#66A7A0")]
    panels = [
        ("CALIFORNIA", [-0.01, 9.35, 7.92, 8.30], 105),
        ("DIABETES", [-1.43, 1.41, 1.28, 1.57], 195),
    ]
    zero = x + 193
    scale = 26.2
    for heading, vals, yy in panels:
        txt(c, x + 20, y + yy, heading, 22, INK, True)
        for i, (name, value, color) in enumerate(zip(names, vals, colors)):
            ry = y + yy + 18 + i * 17
            txt(c, x + 20, ry + 8, name, 20, MUTED)
            line(c, zero, ry - 3, zero, ry + 11, LINE, 0.5)
            if value >= 0:
                rect(c, zero, ry - 1, max(value * scale, 3), 10, color, 2)
            else:
                rect(c, zero + value * scale, ry - 1, -value * scale, 10, color, 2)
            txt(c, x + w - 20, ry + 8, f"{value:+.2f} %".replace(".", ","), 19,
                color if i else MUTED, True, "right")
    txt(c, x + 20, y + h - 14, "30 semillas por configuración y conjunto; valores positivos = menor error.",
        17, MUTED)


def d23_chart(c: canvas.Canvas) -> None:
    x, y, w, h = 340, 578, 514, 170
    rect(c, x, y, w, h, WHITE, 9)
    txt(c, x + 20, y + 30, "COMPROBACIÓN DE ROBUSTEZ · DIABETES", 23, TEAL, True)
    txt(c, x + 20, y + 56, "La mejora cambia al variar la partición", 27, INK, True)
    values = [1.59, -0.73, -0.61, -2.48, -1.17, 3.35]
    labels = ["2026", "2027", "2028", "2029", "2030", "2031"]
    base_y = y + 105
    line(c, x + 24, base_y, x + w - 24, base_y, MUTED, 0.8)
    for i, (label, value) in enumerate(zip(labels, values)):
        bx = x + 49 + i * 78
        bh = abs(value) * 10.5
        col = TEAL if value > 0 else ORANGE
        rect(c, bx, base_y - bh if value > 0 else base_y, 38, bh, col, 2)
        txt(c, bx + 19, y + 151, label, 17, MUTED, False, "center")
        if value < -2:
            txt(c, bx + 19, base_y + bh / 2 + 2, f"{value:+.2f}".replace(".", ","),
                15, WHITE, True, "center")
        else:
            label_y = base_y - bh - 7 if value > 0 else base_y + bh + 17
            txt(c, bx + 19, label_y, f"{value:+.2f}".replace(".", ","),
                16, col, True, "center")
    txt(c, x + 20, y + h - 5, "C árbol · mejora relativa mediana; 2027-2031: análisis posterior, 10 semillas por partición.",
        14, MUTED)


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(W, H), pageCompression=1)
    c.setTitle("Póster V0 A0 - Síntesis evolutiva de atributos")
    rect(c, 0, 0, 1189, 841, PAPER)
    rect(c, 0, 0, 1189, 13, TEAL)

    # Cabecera y mensaje central.
    txt(c, 35, 45, "IA BIOINSPIRADA · REGRESIÓN · UC3M", 20, TEAL, True)
    txt(c, 35, 100, "¿Qué aportan los atributos", 62, INK, True)
    txt(c, 35, 155, "creados por evolución?", 62, INK, True)
    txt(c, 1154, 49, "PÓSTER · V0", 21, MUTED, True, "right")
    rect(c, 35, 174, 1119, 67, TEAL, 8)
    txt(c, 57, 200, "IDEA CLAVE", 20, WHITE, True)
    txt(c, 57, 225, "Crear atributos reduce el error en California; en Diabetes la mejora no es estable.",
        31, WHITE, True)

    # Columna izquierda: pregunta, ablación y esquema de validación.
    section(c, 35, 291, "01", "PREGUNTA Y MÉTODO")
    wrapped(c, 35, 319,
            "¿Puede una búsqueda evolutiva construir variables que ayuden a predecir mejor que las originales?",
            280, 22, 30)
    txt(c, 35, 384, "Tres formas de transformar X", 24, INK, True)
    for yy, label, desc, bg, fg in [
        (399, "A · SELECCIÓN", "Elige columnas originales.", BLUE_LIGHT, BLUE),
        (445, "B · SÍNTESIS", "Conserva originales y añade nuevas.", ORANGE_LIGHT, ORANGE),
        (491, "C · AMBAS", "Selecciona y crea; menos columnas que X.", TEAL_LIGHT, TEAL),
    ]:
        rect(c, 35, yy, 285, 40, bg, 5)
        txt(c, 45, yy + 16, label, 18, fg, True)
        txt(c, 45, yy + 32, desc, 18, INK)

    section(c, 35, 562, "02", "VALIDACIÓN")
    txt(c, 35, 592, "Datos", 19, INK, True)
    line(c, 86, 587, 306, 587, TEAL, 1.2)
    pill(c, 35, 603, 116, "80 % train")
    pill(c, 170, 603, 150, "20 % validación", BLUE_LIGHT, BLUE)
    txt(c, 35, 641, "Evolución: CV de 5 particiones", 18, INK)
    txt(c, 35, 665, "Selección final: CV repetida", 18, INK)
    txt(c, 35, 697, "Validación solo al final:", 18, INK, True)
    txt(c, 35, 721, "MAE original (e) vs. transformado (e′)", 18, INK)

    # Centro: evidencia principal y robustez.
    main_chart(c)
    d23_chart(c)

    # Columna derecha: cifras, interpretación y límites.
    section(c, 876, 291, "03", "LECTURA RÁPIDA")
    rect(c, 876, 309, 278, 103, TEAL_LIGHT, 7)
    txt(c, 890, 339, "−9,35 %", 47, TEAL, True)
    wrapped(c, 890, 365, "MAE en California con B frente a la línea base.", 247, 18, 25)

    rect(c, 876, 422, 278, 96, ORANGE_LIGHT, 7)
    txt(c, 890, 451, "8 → 7", 39, ORANGE, True)
    wrapped(c, 890, 476, "columnas en California con C árbol; mejora del 7,92 %.", 247, 18, 25)

    txt(c, 876, 552, "LO QUE APRENDEMOS", 22, TEAL, True)
    wrapped(c, 876, 581,
            "La síntesis explica la mayor parte de la mejora. Seleccionar sin crear apenas ayuda a OLS.",
            273, 18, 25)
    wrapped(c, 876, 648,
            "Árboles y GEP tienen medianas próximas; GEP muestra menos dispersión en California.",
            273, 18, 25)
    rect(c, 876, 702, 278, 46, BLUE_LIGHT, 5)
    wrapped(c, 888, 721,
            "Límite: el beneficio depende del conjunto y del modelo predictivo.",
            250, 17, 23, INK, True)

    # Pie: metodología mínima y trazabilidad.
    line(c, 35, 767, 1154, 767, LINE, 1)
    txt(c, 35, 787, "DISEÑO", 16, TEAL, True)
    txt(c, 107, 787, "California: 16 512/4 128 filas train/validación · Diabetes: 354/88 · OLS · MAE · 30 semillas", 16, MUTED)
    txt(c, 35, 810, "FUENTES", 16, TEAL, True)
    txt(c, 107, 810, "docs/DISENO.md · results/analysis/informe.md · results/d23/resumen.md", 16, MUTED)
    txt(c, 1154, 810, "A0 horizontal · 1189 × 841 mm", 15, MUTED, False, "right")
    c.showPage()
    c.save()
    print(OUT)


if __name__ == "__main__":
    build()
