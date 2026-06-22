"""Generación de reportes PDF con fpdf2.

Usa la fuente core 'helvetica' (sin ficheros TTF). El español se codifica en
latin-1, que cubre tildes y signos (¿¡ñáéíóú); cualquier carácter fuera de
ese set se sustituye para evitar errores.
"""
from __future__ import annotations

# Colores semánticos (RGB) coherentes con el frontend.
COLOR_ORIGINALITY = (22, 132, 99)    # verde
COLOR_PLAGIARISM = (37, 99, 176)     # azul
COLOR_AI = (193, 122, 30)            # ámbar
COLOR_INK = (38, 34, 28)
COLOR_MUTED = (120, 113, 100)


def _safe(text: str) -> str:
    """Hace el texto seguro para la fuente core (latin-1)."""
    return (text or "").encode("latin-1", "replace").decode("latin-1")


def generate_pdf(result: dict, text: str = "", title: str | None = None) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Falta la librería 'fpdf2' (pip install fpdf2).") from e

    s = result["scores"]
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)

    # --- Encabezado ---
    pdf.set_text_color(*COLOR_INK)
    pdf.set_font("helvetica", "B", 22)
    pdf.cell(0, 12, _safe("Veraz · Reporte de originalidad"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(*COLOR_MUTED)
    pdf.cell(0, 6, _safe(title or "Análisis de texto"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, _safe(f"Generado: {result['meta'].get('analyzed_at', '')}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Tarjetas de puntaje ---
    cards = [
        ("Originalidad", s["originality"], COLOR_ORIGINALITY),
        ("Similitud / plagio", s["plagiarism"], COLOR_PLAGIARISM),
        ("Probabilidad de IA", s["ai_probability"], COLOR_AI),
    ]
    w = (pdf.w - 36 - 12) / 3
    x0 = pdf.get_x()
    y0 = pdf.get_y()
    for i, (label, val, color) in enumerate(cards):
        x = x0 + i * (w + 6)
        pdf.set_xy(x, y0)
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 26)
        pdf.cell(w, 20, _safe(f"{val}%"), border=0, align="C", fill=True,
                 new_x="LEFT", new_y="NEXT")
        pdf.set_xy(x, y0 + 20)
        pdf.set_fill_color(245, 242, 235)
        pdf.set_text_color(*COLOR_INK)
        pdf.set_font("helvetica", "", 9)
        pdf.cell(w, 8, _safe(label), border=0, align="C", fill=True)
    pdf.set_xy(x0, y0 + 32)

    # --- Resumen ---
    conf = result.get("confidence", {})
    pdf.set_text_color(*COLOR_INK)
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 9, _safe("Resumen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, _safe(result["explanation"]["summary"]),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*COLOR_MUTED)
    pdf.set_font("helvetica", "I", 9)
    pdf.multi_cell(0, 5, _safe(f"Confianza del análisis: {conf.get('level','?')} - "
                               f"{conf.get('reason','')}"),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # --- Señales de IA ---
    _section(pdf, "Señales analizadas (IA)")
    pdf.set_font("helvetica", "", 10)
    for sig in result["ai_detection"]["signals"]:
        pdf.set_text_color(*COLOR_INK)
        pdf.multi_cell(0, 5,
                       _safe(f"- {sig['label']} [{sig['severity']}, {sig['score']}%]: "
                             f"{sig['description']}"),
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # --- Recomendaciones ---
    _section(pdf, "Recomendaciones")
    pdf.set_font("helvetica", "", 10)
    for rec in result["recommendations"]:
        pdf.set_text_color(*COLOR_INK)
        pdf.multi_cell(0, 5, _safe(f"[{rec['priority'].upper()}] {rec['title']}: {rec['detail']}"),
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # --- Aviso ---
    _section(pdf, "Aviso importante")
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(*COLOR_MUTED)
    pdf.multi_cell(0, 5, _safe(result["explanation"]["disclaimer"]),
                   new_x="LMARGIN", new_y="NEXT")

    out = pdf.output()
    return bytes(out)


def _section(pdf, title: str) -> None:
    pdf.ln(2)
    pdf.set_text_color(38, 34, 28)
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 9, _safe(title), new_x="LMARGIN", new_y="NEXT")
