"""Orquestador del análisis. Une todos los módulos en un único resultado.

`analyze()` es la función central que usa la API web, los tests y el CLI.
No depende de FastAPI ni de ninguna librería externa: solo Python estándar.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("veraz.engine")

from . import (
    ai_detection,
    explain,
    features as features_mod,
    highlighting,
    originality as originality_mod,
    plagiarism as plagiarism_mod,
    recommendations,
)


def analyze(text: str, references: list[dict] | None = None,
            model=None, model_weight: float = 0.5) -> dict:
    """Analiza un texto completo.

    Parámetros
    ----------
    text       : el documento a analizar.
    references : lista de {"name", "text"} para comparar plagio (opcional).
    model      : clasificador entrenado con .predict_proba_one(vector) (opcional).
    model_weight: peso del modelo frente a las heurísticas (0..1).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("El texto está vacío.")

    # 1. Rasgos estilométricos.
    feats = features_mod.extract(text)

    # 2. Probabilidad del modelo entrenado (si existe).
    model_proba = None
    if model is not None:
        try:
            model_proba = float(model.predict_proba_one(features_mod.to_vector(feats)))
        except Exception:
            # Modelo incompatible (p. ej. entrenado con otro set de rasgos):
            # se sigue solo con heurísticas, pero queda registrado.
            logger.warning(
                "El modelo entrenado no pudo predecir; se usan solo heurísticas. "
                "Suele indicar un modelo viejo: reentrena con 'python train.py'.",
                exc_info=True)
            model_proba = None

    # 3. Detección de IA (señales + modelo + calibración del % final).
    final_calibration = getattr(model, "final_calibration", None) if model else None
    ai_result = ai_detection.detect(feats, model_proba, model_weight, final_calibration)

    # 3b. Análisis por oración (ventanas): puntaje local de IA con el modelo.
    sent_scores = ai_detection.sentence_scores(text, model)

    # 3c. Textos LARGOS: el análisis global se diluye (una sección de IA
    #     desaparece del promedio). Se mezcla con la media por bloques.
    if feats["_word_count"] >= ai_detection.LONG_TEXT_WORDS:
        blocks = ai_detection.block_estimate(text, model)
        if blocks and blocks["n_blocks"] >= 3:
            p = ai_result["probability"] / 100.0
            blended = 0.45 * p + 0.55 * blocks["mean"]
            ai_result["probability"] = round(
                min(max(blended, 0.03), 0.97) * 100)
            ai_result["blocks"] = {
                "mean": round(blocks["mean"], 3),
                "coverage": round(blocks["coverage"], 3),
                "n_blocks": blocks["n_blocks"],
            }

    # 3d. Corrección por heterogeneidad (documento mixto: mitad IA, mitad no).
    ai_result["probability"], mixed_info = ai_detection.adjust_for_heterogeneity(
        ai_result["probability"], sent_scores)
    ai_result["mixed"] = mixed_info

    # 3e. Regiones contiguas con estilo de IA (la ZONA completa, no cachitos).
    regions = ai_detection.detect_regions(text, sent_scores)
    region_words = sum(r["words"] for r in regions)
    ai_result["regions"] = regions
    ai_result["regions_summary"] = {
        "count": len(regions),
        "coverage": round(region_words / max(feats["_word_count"], 1), 3),
    }

    # 4. Detección de plagio.
    plag = plagiarism_mod.analyze(text, references)

    # 5. Originalidad combinada + confianza honesta.
    orig = originality_mod.compute(plag["similarity"], ai_result["probability"])
    model_p = ai_result["model_probability"]
    conf = originality_mod.confidence(
        feats["_word_count"], plag["has_corpus"],
        heuristic_p=ai_result["heuristic_probability"] / 100.0,
        model_p=(model_p / 100.0 if model_p is not None else None),
        probability=ai_result["probability"],
        heterogeneity=mixed_info["heterogeneity"])

    # 6. Resaltado por oración (heurística local + modelo + regiones).
    segments = highlighting.build_segments(
        text, feats["avg_sentence_len"], plag["flagged_sentences"],
        sentence_ai=sent_scores, ai_regions=regions)

    # 7. Explicaciones y recomendaciones.
    explanation = explain.build(feats, ai_result, plag, orig)
    recs = recommendations.build(feats, ai_result, plag)

    return {
        "scores": {
            "originality": orig,
            "plagiarism": plag["similarity"],
            "ai_probability": ai_result["probability"],
        },
        "confidence": conf,
        "ai_detection": ai_result,
        "plagiarism": plag,
        "segments": segments,
        "explanation": explanation,
        "recommendations": recs,
        "metrics": {k: v for k, v in feats.items() if not k.startswith("_")},
        "meta": {
            "word_count": feats["_word_count"],
            "sentence_count": feats["_sentence_count"],
            "char_count": feats["_char_count"],
            "model_used": ai_result["used_model"],
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        },
    }
