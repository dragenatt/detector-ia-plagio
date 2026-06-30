"""Orquestador del análisis. Une todos los módulos en un único resultado.

`analyze()` es la función central que usa la API web, los tests y el CLI.
No depende de FastAPI ni de ninguna librería externa: solo Python estándar.
"""
from __future__ import annotations

from datetime import datetime, timezone

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
            model_proba = None

    # 3. Detección de IA (señales + modelo + calibración del % final).
    final_calibration = getattr(model, "final_calibration", None) if model else None
    ai_result = ai_detection.detect(feats, model_proba, model_weight, final_calibration)

    # 4. Detección de plagio.
    plag = plagiarism_mod.analyze(text, references)

    # 5. Originalidad combinada.
    orig = originality_mod.compute(plag["similarity"], ai_result["probability"])
    conf = originality_mod.confidence_level(feats["_word_count"], plag["has_corpus"])

    # 6. Resaltado por oración.
    segments = highlighting.build_segments(
        text, feats["avg_sentence_len"], plag["flagged_sentences"])

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
