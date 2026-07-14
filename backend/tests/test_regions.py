"""Regiones de IA en textos largos: la detección debe cubrir la ZONA completa
donde se usó IA (no oraciones sueltas) sin invadir las partes humanas, y una
sección de IA dentro de un texto largo humano NO debe diluirse en el promedio.

Correr:  python tests/test_regions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analysis.engine import analyze  # noqa: E402
from app.config import MODEL_PATH         # noqa: E402
from app.model.trainer import load_model  # noqa: E402

# Las regiones se diseñan para operar CON el modelo (como en producción).
MODEL = load_model(MODEL_PATH)


def _analyze(text: str):
    return analyze(text, model=MODEL)

HUM1 = ("Cuando empecé la tesis pensaba que lo difícil sería la teoría. Me "
        "equivoqué de cabo a rabo. Lo difícil fue conseguir que alguien "
        "contestara las encuestas. Mandé doscientos correos y respondieron "
        "treinta y una personas. Mi directora me dijo que no me desanimara, "
        "que a ella le pasó igual en los noventa, con cartas de papel. "
        "Rediseñé el cuestionario tres veces. La segunda versión era tan larga "
        "que mi hermano se durmió probándola. Al final lo corté a doce "
        "preguntas y ahí empezó a fluir.")

IA_SEC = ("La metodología cuantitativa constituye un enfoque fundamental en la "
          "investigación social contemporánea. En primer lugar, permite obtener "
          "datos objetivos y medibles sobre el fenómeno de estudio. Además, "
          "facilita la generalización de los resultados a poblaciones más "
          "amplias. Asimismo, cabe destacar que el análisis estadístico "
          "proporciona rigor y validez a las conclusiones. Por otro lado, es "
          "importante mencionar que la triangulación de datos fortalece la "
          "confiabilidad del estudio. En consecuencia, la combinación de "
          "instrumentos resulta esencial para garantizar resultados robustos.")

HUM2 = ("Los resultados me sorprendieron bastante. Esperaba que la edad fuera "
        "el factor decisivo y resultó casi irrelevante. Lo que pesó fue otra "
        "cosa: si la persona había tenido una mala experiencia previa. Una "
        "encuestada me escribió aparte para contarme la suya. Ese testimonio no "
        "entra en las tablas pero me cambió la forma de leer los números. "
        "Discutí el hallazgo con mi directora un martes de lluvia. Me dijo: ahí "
        "tienes tu capítulo cinco. Y sí, ahí estaba.")


def test_region_covers_ai_section_not_human():
    full = HUM1 + "\n\n" + IA_SEC + "\n\n" + HUM2
    r = _analyze(full)
    regions = r["ai_detection"]["regions"]
    assert regions, "Debería detectarse al menos una región de IA."

    # Rango de caracteres real de la sección de IA.
    ia_start = full.index(IA_SEC)
    ia_end = ia_start + len(IA_SEC)

    covered = set()
    for reg in regions:
        # La región no debe salirse mucho hacia las partes humanas.
        assert reg["start"] >= ia_start - 120, "La región invade la parte humana previa."
        assert reg["end"] <= ia_end + 120, "La región invade la parte humana posterior."
        covered.update(range(reg["start"], reg["end"]))

    # Debe cubrir buena parte de la sección de IA (no solo una oración suelta).
    overlap = len(covered & set(range(ia_start, ia_end)))
    frac = overlap / (ia_end - ia_start)
    assert frac >= 0.6, f"La región cubre solo el {frac:.0%} de la sección de IA."


def test_ai_section_not_diluted_in_long_text():
    """Una sección de IA dentro de un texto humano largo debe elevar el % y
    marcar >=4 oraciones (antes se diluía a ~2 'cachitos')."""
    full = HUM1 + "\n\n" + IA_SEC + "\n\n" + HUM2
    only_human = HUM1 + "\n\n" + HUM2
    r_mix = _analyze(full)
    r_hum = _analyze(only_human)
    assert r_mix["scores"]["ai_probability"] > r_hum["scores"]["ai_probability"] + 10, \
        "La sección de IA debería elevar claramente el % frente al texto solo-humano."
    ia_sentences = [s for s in r_mix["segments"] if s["category"] in ("ia", "mixto")]
    assert len(ia_sentences) >= 4, \
        f"Debería marcar la zona completa, no cachitos (marcó {len(ia_sentences)})."


def test_pure_human_long_has_no_region():
    r = _analyze(HUM1 + "\n\n" + HUM2)
    assert not r["ai_detection"]["regions"], \
        "Un texto humano largo no debería tener regiones de IA."


if __name__ == "__main__":
    test_region_covers_ai_section_not_human()
    print("[OK] la región cubre la sección de IA sin invadir lo humano")
    test_ai_section_not_diluted_in_long_text()
    print("[OK] la sección de IA no se diluye en el texto largo")
    test_pure_human_long_has_no_region()
    print("[OK] texto humano largo sin regiones falsas")
    print("\nTODAS LAS PRUEBAS DE REGIONES PASARON [OK]")
