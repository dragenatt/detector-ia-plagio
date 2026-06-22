"""Detección de texto generado por IA mediante MÚLTIPLES señales explicables.

Filosofía (importante para el usuario): NO es una regla única ni una caja
negra. Calculamos varias señales independientes, cada una en [0,1] donde
1 = "muy parecido a IA". Cada señal trae su etiqueta y explicación, de modo
que el resultado siempre se puede justificar en lenguaje humano.

La probabilidad final es una combinación ponderada de las señales y,
opcionalmente, de un modelo entrenado con tu corpus (model/classifier.py).
El resultado se presenta SIEMPRE como "probabilidad / riesgo", nunca como
una acusación, y se recorta para evitar falsos 0 % o 100 %.
"""
from __future__ import annotations

from . import text_utils as tu

# Cada señal: (clave, etiqueta, peso, función(feats)->[0,1], descripción)
# Los pesos suman 1.0. Ajustables sin tocar la lógica.
SIGNALS = [
    {
        "key": "uniformity",
        "label": "Uniformidad de oraciones",
        "weight": 0.16,
        "describe": "Las oraciones tienen longitudes muy parecidas entre sí.",
    },
    {
        "key": "low_burstiness",
        "label": "Ritmo plano",
        "weight": 0.12,
        "describe": "Falta la alternancia natural entre frases cortas y largas.",
    },
    {
        "key": "connector_overuse",
        "label": "Exceso de conectores",
        "weight": 0.17,
        "describe": "Uso muy frecuente de conectores ('además', 'por lo tanto'...).",
    },
    {
        "key": "generic_phrases",
        "label": "Frases genéricas / vacías",
        "weight": 0.20,
        "describe": "Frases que suenan académicas pero aportan poco contenido.",
    },
    {
        "key": "low_personal_voice",
        "label": "Poca voz personal",
        "weight": 0.10,
        "describe": "Casi no aparece opinión o experiencia en primera persona.",
    },
    {
        "key": "repetition",
        "label": "Repetición de ideas",
        "weight": 0.13,
        "describe": "Se repiten secuencias de palabras o ideas similares.",
    },
    {
        "key": "too_clean",
        "label": "Redacción demasiado 'limpia'",
        "weight": 0.12,
        "describe": "Ausencia casi total de marcas informales o errores humanos.",
    },
]


def _signal_value(key: str, f: dict) -> float:
    """Mapea los rasgos crudos a una señal [0,1] (1 = más parecido a IA)."""
    if key == "uniformity":
        # Desviación estándar baja de longitudes -> muy uniforme.
        return 1.0 - tu.ramp(f["sentence_len_std"], 2.0, 9.0)
    if key == "low_burstiness":
        return 1.0 - tu.ramp(f["sentence_len_cv"], 0.20, 0.60)
    if key == "connector_overuse":
        return tu.ramp(f["connector_density"], 3.0, 10.0)
    if key == "generic_phrases":
        return tu.ramp(f["generic_phrase_density"], 0.3, 2.5)
    if key == "low_personal_voice":
        return 1.0 - tu.ramp(f["personal_voice_density"], 0.3, 3.0)
    if key == "repetition":
        return tu.ramp(f["repeated_trigram_ratio"], 0.02, 0.15)
    if key == "too_clean":
        # Sin marcas informales + puntuación moderada -> "demasiado pulido".
        no_informal = 1.0 if f["informal_density"] == 0 else 0.0
        return no_informal * tu.ramp(f["punct_ratio"], 0.05, 0.18)
    return 0.0


def _severity(value: float) -> str:
    if value >= 0.66:
        return "alta"
    if value >= 0.33:
        return "media"
    return "baja"


def detect(features: dict, model_proba: float | None = None,
           model_weight: float = 0.5) -> dict:
    """Calcula la probabilidad de IA y el desglose por señal.

    - features      : dict de features.extract()
    - model_proba   : probabilidad [0,1] del clasificador entrenado (o None).
    - model_weight  : cuánto pesa el modelo frente a las heurísticas (0..1).
    """
    breakdown = []
    heuristic = 0.0
    for s in SIGNALS:
        v = round(_signal_value(s["key"], features), 4)
        heuristic += s["weight"] * v
        breakdown.append({
            "key": s["key"],
            "label": s["label"],
            "value": v,                       # 0..1
            "score": round(v * 100),          # 0..100 para la UI
            "weight": s["weight"],
            "severity": _severity(v),
            "description": s["describe"],
        })

    heuristic = tu.clamp(heuristic, 0.0, 1.0)

    if model_proba is None:
        combined = heuristic
        used_model = False
    else:
        combined = model_weight * model_proba + (1 - model_weight) * heuristic
        used_model = True

    # Recorte: jamás afirmamos 0 % ni 100 % (ningún detector es infalible).
    probability = round(tu.clamp(combined, 0.03, 0.97) * 100)

    return {
        "probability": probability,            # 0..100
        "heuristic_probability": round(heuristic * 100),
        "model_probability": round(model_proba * 100) if model_proba is not None else None,
        "used_model": used_model,
        "signals": breakdown,
    }
