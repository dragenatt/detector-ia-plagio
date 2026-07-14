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

import re

from . import text_utils as tu


# --------------------------------------------------------------------------- #
# Evidencias exactas: QUÉ palabras de la oración disparan cada señal
# --------------------------------------------------------------------------- #

def _lexicon_re(items) -> re.Pattern:
    """Regex de alternación con límites de palabra para un léxico completo.
    Las frases más largas van primero para que ganen a sus prefijos."""
    parts = sorted({re.escape(x.strip()) for x in items if x.strip()},
                   key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)",
                      re.IGNORECASE | re.UNICODE)


# (clave, etiqueta, patrón). El orden define la prioridad visual en la UI.
_EVIDENCE_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("generica", "Frase genérica / vacía", _lexicon_re(tu.GENERIC_PHRASES)),
    ("conector", "Conector", _lexicon_re(tu.CONNECTORS)),
    ("atenuador", "Atenuador / generalidad", _lexicon_re(tu.HEDGES)),
]
_TYPOGRAPHIC_CHARS = "—–“”‘’…"
MAX_EVIDENCE_PER_SENTENCE = 12


def find_evidence(sentence_text: str, abs_start: int) -> list[dict]:
    """Fragmentos exactos (con offsets ABSOLUTOS en el documento) que
    disparan señales de IA dentro de una oración."""
    found: list[dict] = []
    for kind, label, rx in _EVIDENCE_PATTERNS:
        for m in rx.finditer(sentence_text):
            found.append({
                "kind": kind,
                "label": label,
                "text": m.group(),
                "start": abs_start + m.start(),
                "end": abs_start + m.end(),
            })
    for i, ch in enumerate(sentence_text):
        if ch in _TYPOGRAPHIC_CHARS:
            found.append({
                "kind": "tipografia",
                "label": "Tipografía 'pulida'",
                "text": ch,
                "start": abs_start + i,
                "end": abs_start + i + 1,
            })
    # Sin solapamientos (gana la evidencia más temprana/larga) y con tope.
    found.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    result: list[dict] = []
    last_end = -1
    for e in found:
        if e["start"] >= last_end:
            result.append(e)
            last_end = e["end"]
        if len(result) >= MAX_EVIDENCE_PER_SENTENCE:
            break
    return result


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
                   ai_cue_threshold: float = 0.6,
                   sentence_ai: list[dict] | None = None,
                   ai_regions: list[dict] | None = None) -> list[dict]:
    """`sentence_ai`: puntajes por ventana de ai_detection.sentence_scores().
    Si están disponibles, se mezclan con la pista local (ponderados por la
    confianza de la ventana) para que el ámbar también use lo aprendido.
    `ai_regions`: zonas contiguas con estilo de IA; toda oración dentro de una
    región se marca (la zona completa, no oraciones sueltas)."""
    flagged_by_index = {f["index"]: f for f in flagged_sentences}
    window_by_index = {s["index"]: s for s in (sentence_ai or [])}
    region_by_index: dict[int, dict] = {}
    for reg in (ai_regions or []):
        for k in reg["sentences"]:
            region_by_index[k] = reg
    segments = []
    for i, sent in enumerate(tu.split_sentences(text)):
        plag = flagged_by_index.get(i)
        ai_score, ai_reasons = _sentence_ai_cue(sent["text"], avg_sentence_len)
        threshold = ai_cue_threshold
        win = window_by_index.get(i)
        if win and win["score"] is not None and win["confidence"] > 0:
            w = 0.5 * win["confidence"]  # el modelo pesa según la señal local
            ai_score = tu.clamp((1 - w) * ai_score + w * win["score"], 0.0, 1.0)
            threshold = 0.5  # con modelo hay mejor señal: umbral estándar
            if win["score"] >= 0.6:
                ai_reasons.append(
                    f"el modelo puntúa alto su entorno ({round(win['score'] * 100)}%)")
        region = region_by_index.get(i)
        if region:
            # Dentro de una zona de IA: se marca la región COMPLETA.
            ai_score = max(ai_score, region["score"])
            ai_reasons.append(
                "está dentro de una zona contigua con estilo de IA "
                f"({round(region['score'] * 100)}% de la zona)")
        is_ai = region is not None or ai_score >= threshold
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

        # Evidencias exactas: qué palabras disparan las señales.
        evidence: list[dict] = []
        if is_plag and plag.get("match"):
            m = plag["match"]
            evidence.append({
                "kind": "copia",
                "label": f"Tramo que coincide con «{plag['source']}»",
                "text": m["text"],
                "start": m["start"],
                "end": m["end"],
                "source_fragment": m["source_fragment"],
                "matched_words": m["matched_words"],
            })
        if is_ai or is_plag:
            copy_span = (evidence[0]["start"], evidence[0]["end"]) if evidence else None
            for e in find_evidence(sent["text"], sent["text_start"]):
                # sin solaparse con el tramo de copia (la UI trocea por offsets)
                if copy_span and e["start"] < copy_span[1] and e["end"] > copy_span[0]:
                    continue
                evidence.append(e)
            evidence.sort(key=lambda e: e["start"])

        segments.append({
            "index": i,
            "text": sent["text"],
            "start": sent["start"],
            "end": sent["end"],
            "category": category,
            "ai_score": round(ai_score * 100),
            "plagiarism_overlap": plag["overlap"] if plag else 0,
            "plagiarism_source": plag["source"] if plag else None,
            "reason": " ".join(reason_parts) or "Sin señales destacadas.",
            "evidence": evidence,
        })
    return segments
