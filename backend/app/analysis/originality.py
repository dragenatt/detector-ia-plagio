"""Índice de originalidad: combina plagio + IA en un único número.

Definición (transparente y documentada para el usuario):

  originalidad = (1 - similitud)  *  (1 - 0.7 * prob_IA)   * 100

- La similitud (copia) penaliza de forma directa: si copiaste mucho, hay poca
  originalidad de autoría.
- La probabilidad de IA penaliza de forma más suave (factor 0.7) porque es una
  estimación menos certera y no equivale a "copiar".

El resultado se recorta a [1, 99] para no transmitir certezas absolutas.
"""
from __future__ import annotations

from . import text_utils as tu


def compute(similarity: int, ai_probability: int) -> int:
    sim = tu.clamp(similarity / 100.0, 0.0, 1.0)
    ai = tu.clamp(ai_probability / 100.0, 0.0, 1.0)
    originality = (1.0 - sim) * (1.0 - 0.7 * ai) * 100.0
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
