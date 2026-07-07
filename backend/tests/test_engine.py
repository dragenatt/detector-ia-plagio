"""Pruebas del motor de análisis. Se puede correr como script:

    python backend/tests/test_engine.py

o con pytest:

    pytest backend/tests/
"""
import sys
from pathlib import Path

# Permite importar el paquete `app` sin instalar nada.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analysis.engine import analyze  # noqa: E402

HUMANO = """Ayer terminé de leer el informe y, sinceramente, me costó.
No esperaba que el tema fuera tan denso. Pero algo quedó claro: nadie tiene
la respuesta completa. Recuerdo que mi profesora decía que dudar también es
avanzar. ¿Será cierto? Creo que sí. A veces una idea pequeña lo cambia todo.
Me quedé pensando en eso durante el viaje de vuelta a casa."""

IA = """En la actualidad, la tecnología juega un papel fundamental en la
sociedad. Además, es importante destacar que la educación es esencial para el
desarrollo de las personas. Por lo tanto, las instituciones deben adaptarse a
los cambios constantes. Asimismo, cabe destacar que la innovación resulta
evidente en el mundo actual. En conclusión, es fundamental promover el
progreso de manera constante y responsable."""

REF = {
    "name": "Enciclopedia: Fotosíntesis",
    "text": ("La fotosíntesis es el proceso por el cual las plantas convierten "
             "la luz solar en energía química almacenada en forma de glucosa."),
}
COPIADO = ("La fotosíntesis es el proceso por el cual las plantas convierten "
           "la luz solar en energía química almacenada en forma de glucosa. "
           "Este proceso resulta vital para la vida en el planeta.")


def _show(title, result):
    s = result["scores"]
    print(f"\n=== {title} ===")
    print(f"  originalidad={s['originality']}%  "
          f"plagio={s['plagiarism']}%  IA={s['ai_probability']}%  "
          f"(confianza={result['confidence']['level']})")
    print(f"  resumen: {result['explanation']['summary']}")


def test_ai_higher_than_human():
    human = analyze(HUMANO)
    ai = analyze(IA)
    _show("HUMANO", human)
    _show("IA", ai)
    assert ai["scores"]["ai_probability"] > human["scores"]["ai_probability"], \
        "El texto de IA debería tener mayor probabilidad de IA que el humano."


def test_plagiarism_detected():
    result = analyze(COPIADO, references=[REF])
    _show("COPIADO vs referencia", result)
    assert result["scores"]["plagiarism"] >= 50, "Debería detectar copia alta."
    assert any(seg["category"] in ("plagio", "mixto") for seg in result["segments"]), \
        "Debería marcar al menos una oración como coincidente."


def test_no_corpus_is_graceful():
    result = analyze(HUMANO)
    assert result["scores"]["plagiarism"] == 0
    assert result["plagiarism"]["has_corpus"] is False


def test_mixed_text_scores_between():
    """Un texto mitad IA + mitad humano debe puntuar ENTRE ambos extremos,
    no ser condenado ni absuelto por completo."""
    mixto = IA + "\n" + HUMANO
    ia_p = analyze(IA)["scores"]["ai_probability"]
    hum_p = analyze(HUMANO)["scores"]["ai_probability"]
    mix = analyze(mixto)
    mix_p = mix["scores"]["ai_probability"]
    _show("MIXTO (IA + humano)", mix)
    assert hum_p < mix_p < ia_p, \
        f"El mixto ({mix_p}%) debería quedar entre humano ({hum_p}%) e IA ({ia_p}%)."
    assert mix["ai_detection"]["mixed"]["heterogeneity"] > 0, \
        "El análisis por oración debería detectar heterogeneidad en un texto mixto."


def test_confidence_fields():
    conf = analyze(HUMANO)["confidence"]
    assert conf["level"] in ("baja", "media", "alta")
    assert "score" in conf and 0.0 <= conf["score"] <= 1.0


if __name__ == "__main__":
    test_ai_higher_than_human()
    test_plagiarism_detected()
    test_no_corpus_is_graceful()
    test_mixed_text_scores_between()
    test_confidence_fields()
    print("\nTODAS LAS PRUEBAS PASARON [OK]")
