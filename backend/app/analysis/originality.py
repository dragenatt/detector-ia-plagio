"""Índice de originalidad: combina plagio + IA en un único número.

Definición (transparente e intuitiva para el usuario):

  originalidad = (1 - similitud) * (1 - prob_IA) * 100

Se lee como "cuánto del texto es originalmente tuyo": lo que NO está copiado y
NO parece generado por IA. Si no hay plagio, la originalidad es simplemente
100 - IA (p. ej. IA 32 % -> originalidad 68 %), que es lo que la gente espera.
El plagio penaliza de forma multiplicativa (copiar todo -> 0 originalidad).

El resultado se recorta a [1, 99] para no transmitir certezas absolutas.
"""
from __future__ import annotations

from . import text_utils as tu


def compute(similarity: int, ai_probability: int) -> int:
    sim = tu.clamp(similarity / 100.0, 0.0, 1.0)
    ai = tu.clamp(ai_probability / 100.0, 0.0, 1.0)
    originality = (1.0 - sim) * (1.0 - ai) * 100.0
    return round(tu.clamp(originality, 1.0, 99.0))


def confidence_level(word_count: int, has_corpus: bool) -> dict:
    """Nivel de confianza del análisis según la cantidad de texto disponible."""
    if word_count < 80:
        level = "baja"
        reason = "El texto es corto (<80 palabras); las estimaciones son poco fiables."
    elif word_count < 300:
        level = "media"
        reason = "Texto de longitud media; las estimaciones son orientativas."
    else:
        level = "alta"
        reason = "Texto suficientemente largo para estimaciones más estables."
    if not has_corpus:
        reason += " Sin corpus de referencia, el plagio no se evalúa."
    return {"level": level, "reason": reason}


def confidence(word_count: int, has_corpus: bool,
               heuristic_p: float | None = None,
               model_p: float | None = None,
               probability: int | None = None,
               heterogeneity: float = 0.0) -> dict:
    """Confianza HONESTA del análisis: combina cuatro factores medibles.

    - Longitud: con poco texto, toda estimación es débil.
    - Acuerdo heurística-modelo: si los dos métodos discrepan, dudar.
    - Distancia al umbral: un 52 % es zona gris; un 95 % o un 5 %, no.
    - Heterogeneidad: un texto mitad-y-mitad merece menos certeza global.
    """
    factors: list[str] = []

    if word_count < 80:
        score = 0.35
        factors.append("el texto es corto (<80 palabras)")
    elif word_count < 300:
        score = 0.65
        factors.append("longitud media")
    else:
        score = 0.9
        factors.append("texto largo")

    if heuristic_p is not None and model_p is not None:
        agreement = 1.0 - abs(heuristic_p - model_p)
        score *= 0.7 + 0.3 * agreement
        if agreement < 0.5:
            factors.append("la heurística y el modelo discrepan")

    if probability is not None:
        distance = abs(probability - 50) / 50.0
        score *= 0.75 + 0.25 * distance
        if distance < 0.2:
            factors.append("el resultado cae en zona gris (cerca del 50 %)")

    if heterogeneity > 0.5:
        score *= 0.8
        factors.append("el texto combina partes de estilos muy distintos")

    if score < 0.45:
        level = "baja"
    elif score < 0.68:
        level = "media"
    else:
        level = "alta"

    reason = "Confianza {}: {}.".format(level, "; ".join(factors))
    if not has_corpus:
        reason += " Sin corpus de referencia, el plagio no se evalúa."
    return {"level": level, "reason": reason, "score": round(score, 2)}
