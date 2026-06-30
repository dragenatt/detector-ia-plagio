"""Genera segmentos coloreables a nivel de oración.

Cada oración recibe una categoría:
  - "plagio"   : coincide fuertemente con una fuente del corpus  (azul)
  - "ia"       : muestra varias marcas típicas de IA              (ámbar)
  - "mixto"    : ambas cosas a la vez                             (azul+ámbar)
  - "original" : sin señales destacadas                          (verde/neutral)

Los offsets (start/end) cubren el texto crudo, así el frontend reconstruye el
documento exactamente e ilumina cada tramo.
"""
from __future__ import annotations

from . import text_utils as tu


def _sentence_ai_cue(sentence: str, avg_len: float) -> tuple[float, list[str]]:
    """Puntaje local de 'parece IA' para UNA oración + razones legibles.

    Usa los mismos patrones que el detector global, adaptados a una sola
    oración, para que el resaltado ámbar coincida con lo que mide el motor.
    """
    reasons: list[str] = []
    score = 0.0
    low = sentence.lower()
    words = tu.words(sentence)
    n_words = len(words)

    if any(p in low for p in tu.GENERIC_PHRASES):
        score += 0.45
        reasons.append("contiene una frase genérica/vacía")

    conn = tu.count_occurrences(sentence, tu.CONNECTORS)
    if conn >= 2:
        score += 0.20
        reasons.append("acumula varios conectores")
    elif conn >= 1 and tu.starts_with_opener(sentence, tu.SENTENCE_OPENERS):
        score += 0.15
        reasons.append("abre con un conector")

    if tu.count_occurrences(sentence, tu.HEDGES) >= 1:
        score += 0.12
        reasons.append("usa fórmulas de cautela genéricas")

    if any(ch in sentence for ch in ("—", "–", "“", "”", "‘", "’")):
        score += 0.10
        reasons.append("usa tipografía 'pulida' (rayas o comillas curvas)")

    if avg_len and n_words > 1.5 * avg_len:
        score += 0.12
        reasons.append("es notablemente más larga que el promedio")

    has_personal = (any(w in tu.PERSONAL_MARKERS for w in words)
                    or any(p in low for p in tu.PERSONAL_PHRASES))
    if not has_personal and n_words >= 12:
        score += 0.12
        reasons.append("no incluye voz personal")

    return tu.clamp(score, 0.0, 1.0), reasons


def build_segments(text: str, avg_sentence_len: float,
                   flagged_sentences: list[dict],
                   ai_cue_threshold: float = 0.6) -> list[dict]:
    flagged_by_index = {f["index"]: f for f in flagged_sentences}
    segments = []
    for i, sent in enumerate(tu.split_sentences(text)):
        plag = flagged_by_index.get(i)
        ai_score, ai_reasons = _sentence_ai_cue(sent["text"], avg_sentence_len)
        is_ai = ai_score >= ai_cue_threshold
        is_plag = plag is not None

        if is_plag and is_ai:
            category = "mixto"
        elif is_plag:
            category = "plagio"
        elif is_ai:
            category = "ia"
        else:
            category = "original"

        reason_parts = []
        if is_plag:
            reason_parts.append(
                f"Coincide {plag['overlap']}% con «{plag['source']}»."
            )
        if is_ai:
            reason_parts.append("Señales de IA: " + ", ".join(ai_reasons) + ".")

        segments.append({
            "index": i,
            "text": sent["text"],
            "start": sent["start"],
            "end": sent["end"],
            "category": category,
            "ai_score": round(ai_score * 100),
            "plagiarism_overlap": plag["overlap"] if plag else 0,
            "reason": " ".join(reason_parts) or "Sin señales destacadas.",
        })
    return segments
