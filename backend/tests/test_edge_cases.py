"""Casos límite del motor: entradas raras no deben romper el análisis, y los
offsets de segmentos/evidencias deben reconstruir el texto EXACTO (base del
resaltado). Correr:  python tests/test_edge_cases.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analysis.engine import analyze  # noqa: E402

REF = {"name": "Fuente", "text": ("La fotosíntesis es el proceso por el cual las "
                                  "plantas convierten la luz solar en energía química.")}


def _roundtrip(text: str, segments: list[dict]) -> bool:
    """Reconstruye el texto original con los offsets de los segmentos + huecos."""
    rebuilt = []
    cursor = 0
    for seg in sorted(segments, key=lambda s: s["start"]):
        if seg["start"] > cursor:
            rebuilt.append(text[cursor:seg["start"]])
        rebuilt.append(text[seg["start"]:seg["end"]])
        cursor = max(cursor, seg["end"])
    rebuilt.append(text[cursor:])
    return "".join(rebuilt) == text


def test_empty_text_raises():
    for bad in ("", "   ", "\n\n\t "):
        try:
            analyze(bad)
            raise AssertionError("El texto vacío debería lanzar ValueError.")
        except ValueError:
            pass


def test_tiny_inputs():
    for text in ("hola", "¿ya?", "😀", "A", "123", "...", "¡¡¡!!!"):
        r = analyze(text)  # no debe reventar
        assert 0 <= r["scores"]["ai_probability"] <= 100
        assert r["confidence"]["level"] == "baja", \
            f"Un texto mínimo debe tener confianza baja: {text!r}"


def test_weird_formats():
    cases = [
        "TODO EN MAYÚSCULAS. ESTO TAMBIÉN. ¿POR QUÉ NO?",
        "The quick brown fox jumps over the lazy dog. It rains a lot in London.",
        "<p>Hola <b>mundo</b></p>\n<ul><li>uno</li><li>dos</li></ul>",
        "# Título\n\n- item uno\n- item dos\n\n**negrita** y `código`.",
        "línea\r\notra\r\rlínea\n\n\n\nfinal",
        "palabra " * 50 + ".",  # repetición extrema
    ]
    for text in cases:
        r = analyze(text)
        assert 0 <= r["scores"]["ai_probability"] <= 100
        # Los offsets van sobre el texto NORMALIZADO que devuelve el motor.
        assert _roundtrip(r["analyzed_text"], r["segments"]), \
            f"Los offsets no reconstruyen el texto: {text[:40]!r}"


def test_offsets_roundtrip_and_evidence():
    text = ("En la actualidad, cabe destacar que la tecnología es fundamental.  \n\n"
            "La fotosíntesis es el proceso por el cual las plantas convierten la "
            "luz solar en energía química. ¡Y a mí me encanta pescar!  Fin.")
    r = analyze(text, references=[REF])
    at = r["analyzed_text"]  # offsets referidos al texto normalizado
    assert _roundtrip(at, r["segments"]), "Los offsets deben cubrir el texto exacto."
    # Toda evidencia debe apuntar EXACTAMENTE a su fragmento en el texto analizado.
    for seg in r["segments"]:
        for ev in seg.get("evidence", []):
            assert at[ev["start"]:ev["end"]] == ev["text"], \
                f"Evidencia desalineada: {ev['text']!r} vs {at[ev['start']:ev['end']]!r}"
    # La oración copiada debe traer el tramo y el fragmento de la fuente.
    plag_segs = [s for s in r["segments"] if s["category"] in ("plagio", "mixto")]
    assert plag_segs, "Debería marcar la oración copiada."
    copias = [e for s in plag_segs for e in s.get("evidence", []) if e["kind"] == "copia"]
    assert copias and copias[0]["source_fragment"], \
        "La evidencia de copia debe incluir el fragmento de la fuente."


def test_hard_linebreaks_do_not_change_result():
    """El mismo texto con saltos de línea 'duros' (copiado de PDF/Word) debe
    dar el MISMO resultado que sin ellos, y no fragmentar las oraciones."""
    clean = ("En la actualidad, la inteligencia artificial constituye un pilar "
             "fundamental. Además, cabe destacar que su impacto resulta evidente. "
             "Por lo tanto, es esencial un desarrollo responsable de la misma.")
    words = clean.split()
    wrapped = "\n".join(" ".join(words[i:i + 5]) for i in range(0, len(words), 5))
    rc, rw = analyze(clean), analyze(wrapped)
    assert rc["scores"]["ai_probability"] == rw["scores"]["ai_probability"], \
        "Los saltos de línea no deberían cambiar el porcentaje de IA."
    assert rc["meta"]["sentence_count"] == rw["meta"]["sentence_count"], \
        "Los saltos de línea no deberían fragmentar las oraciones."


def test_originality_is_intuitive():
    """Sin plagio, originalidad debe ser ~ 100 - IA (no la vieja fórmula 0.7)."""
    from app.analysis.originality import compute
    assert compute(0, 32) == 68
    assert compute(0, 0) == 99   # recortado a 99 (nunca 100)
    assert compute(50, 40) == 30  # (1-0.5)*(1-0.4)=0.30


def test_long_text_performance():
    base = ("Recuerdo aquel verano en el pueblo, cuando mi abuela cocinaba y "
            "el aire olía a leña. Aprendí a pescar con una caña vieja. ")
    text = base * 400  # ~10k palabras
    t0 = time.time()
    r = analyze(text)
    elapsed = time.time() - t0
    assert r["meta"]["word_count"] > 9000
    assert elapsed < 30, f"Análisis de 10k palabras demasiado lento: {elapsed:.1f}s"
    assert _roundtrip(r["analyzed_text"], r["segments"])


if __name__ == "__main__":
    test_empty_text_raises()
    print("[OK] vacío lanza ValueError")
    test_tiny_inputs()
    print("[OK] entradas mínimas (1 palabra, emoji, símbolos)")
    test_weird_formats()
    print("[OK] formatos raros (mayúsculas, inglés, HTML, markdown, CRLF)")
    test_offsets_roundtrip_and_evidence()
    print("[OK] offsets exactos y evidencias alineadas")
    test_hard_linebreaks_do_not_change_result()
    print("[OK] saltos de línea duros no cambian el resultado ni fragmentan")
    test_originality_is_intuitive()
    print("[OK] originalidad intuitiva (100 - IA sin plagio)")
    test_long_text_performance()
    print("[OK] 10k palabras en tiempo razonable")
    print("\nTODAS LAS PRUEBAS DE CASOS LIMITE PASARON [OK]")
