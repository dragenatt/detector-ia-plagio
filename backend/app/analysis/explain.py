"""Traduce los números del análisis a explicaciones en lenguaje simple.

Reglas de tono (clave para el usuario):
- Siempre hablar de "señales", "tendencia", "puede sugerir" — nunca acusar.
- Explicar el PORQUÉ, no solo el porcentaje.
- Distinguir señales humanas, señales de IA y señales de plagio.
"""
from __future__ import annotations

DISCLAIMER = (
    "Ningún detector de IA o plagio es 100 % confiable. Estos resultados son "
    "ESTIMACIONES probabilísticas, no pruebas. Úsalos como punto de partida "
    "para revisar el texto, nunca como una acusación definitiva."
)


def _verdict(ai_p: int, similarity: int) -> str:
    if similarity >= 60:
        return ("El texto comparte mucho contenido con las fuentes: hay un "
                "riesgo alto de copia o paráfrasis cercana.")
    if ai_p >= 66:
        return ("La redacción muestra varias señales asociadas a texto "
                "generado por IA. Conviene revisarlo con calma.")
    if ai_p >= 40:
        return ("Hay algunas señales mixtas: el texto podría combinar partes "
                "humanas con apoyo de IA.")
    return ("La redacción se parece más a la de un autor humano: ritmo "
            "variado y marcas personales presentes.")


def build(features: dict, ai_result: dict, plagiarism: dict,
          originality: int) -> dict:
    ai_p = ai_result["probability"]
    similarity = plagiarism["similarity"]

    human_signals: list[str] = []
    ai_signals: list[str] = []
    plagiarism_signals: list[str] = []

    # Señales de IA (las que dieron "media" o "alta").
    for s in ai_result["signals"]:
        if s["severity"] in ("media", "alta"):
            ai_signals.append(f"{s['label']}: {s['description']}")

    # Señales humanas (lo contrario de algunas señales de IA).
    if features["personal_voice_density"] >= 1.5:
        human_signals.append("Hay presencia clara de voz personal u opinión.")
    if features["sentence_len_std"] >= 6:
        human_signals.append("Las oraciones varían bastante en longitud (ritmo natural).")
    if features["informal_density"] > 0:
        human_signals.append("Aparecen expresiones informales o coloquiales.")
    if features.get("mattr", 0) >= 0.80:
        human_signals.append("El vocabulario es rico y variado (poca repetición de palabras).")
    if features.get("sentence_start_diversity", 1) >= 0.8:
        human_signals.append("Las oraciones empiezan de formas distintas (aperturas variadas).")
    if features.get("hedging_density", 0) < 1.0 and features.get("generic_phrase_density", 0) < 0.5:
        human_signals.append("Las afirmaciones son concretas, con pocas generalidades de relleno.")
    if not human_signals:
        human_signals.append("Se detectaron pocas marcas humanas evidentes.")

    # Señales de plagio.
    if plagiarism["has_corpus"]:
        if plagiarism["flagged_sentences"]:
            n = len(plagiarism["flagged_sentences"])
            plagiarism_signals.append(
                f"{n} oración(es) coinciden de forma notable con el corpus.")
        if plagiarism["topical_similarity"] >= 50:
            plagiarism_signals.append(
                "El vocabulario general es muy parecido al de alguna fuente.")
        if not plagiarism_signals:
            plagiarism_signals.append("No se hallaron coincidencias relevantes con el corpus.")
    else:
        plagiarism_signals.append(plagiarism.get("note") or "Sin corpus de referencia.")

    return {
        "summary": _verdict(ai_p, similarity),
        "human_signals": human_signals,
        "ai_signals": ai_signals or ["No destacan señales fuertes de IA."],
        "plagiarism_signals": plagiarism_signals,
        "disclaimer": DISCLAIMER,
    }
