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

# Cada señal: (clave, etiqueta, peso, descripción). El valor [0,1] de cada una
# lo calcula `_signal_value`. Los pesos suman 1.0; ajustables sin tocar la lógica.
SIGNALS = [
    {
        "key": "generic_phrases",
        "label": "Frases genéricas / vacías",
        "weight": 0.14,
        "describe": "Frases que suenan académicas pero aportan poco contenido.",
    },
    {
        "key": "connector_overuse",
        "label": "Exceso de conectores",
        "weight": 0.10,
        "describe": "Uso muy frecuente de conectores ('además', 'por lo tanto'...).",
    },
    {
        "key": "connector_openers",
        "label": "Oraciones que abren con conector",
        "weight": 0.09,
        "describe": "Muchas frases empiezan con 'Además', 'Por otro lado', etc.",
    },
    {
        "key": "uniformity",
        "label": "Uniformidad de oraciones",
        "weight": 0.08,
        "describe": "Las oraciones tienen longitudes muy parecidas entre sí.",
    },
    {
        "key": "low_burstiness",
        "label": "Ritmo plano",
        "weight": 0.06,
        "describe": "Falta la alternancia natural entre frases cortas y largas.",
    },
    {
        "key": "repetition",
        "label": "Repetición de ideas",
        "weight": 0.07,
        "describe": "Se repiten secuencias de palabras o ideas similares.",
    },
    {
        "key": "repetitive_openings",
        "label": "Aperturas repetitivas",
        "weight": 0.08,
        "describe": "Las oraciones empiezan una y otra vez con las mismas palabras.",
    },
    {
        "key": "low_lexical_diversity",
        "label": "Vocabulario poco variado",
        "weight": 0.08,
        "describe": "Se reutilizan las mismas palabras; poca riqueza léxica.",
    },
    {
        "key": "hedging",
        "label": "Exceso de cautela / generalidades",
        "weight": 0.07,
        "describe": "Abundan atenuadores prudentes ('suele', 'en general', 'podría').",
    },
    {
        "key": "uniform_paragraphs",
        "label": "Párrafos de tamaño uniforme",
        "weight": 0.05,
        "describe": "Los párrafos tienen longitudes llamativamente parejas.",
    },
    {
        "key": "list_structure",
        "label": "Estructura de listas",
        "weight": 0.05,
        "describe": "Mucho contenido en viñetas o enumeraciones, típico de respuestas de IA.",
    },
    {
        "key": "too_clean",
        "label": "Redacción demasiado 'limpia'",
        "weight": 0.05,
        "describe": "Ausencia casi total de marcas informales o errores humanos.",
    },
    {
        "key": "typographic_polish",
        "label": "Tipografía 'pulida'",
        "weight": 0.04,
        "describe": "Uso de rayas y comillas curvas que rara vez teclea una persona.",
    },
    {
        "key": "low_personal_voice",
        "label": "Poca voz personal",
        "weight": 0.04,
        "describe": "Casi no aparece opinión o experiencia en primera persona.",
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
    # --- señales nuevas --------------------------------------------------- #
    if key == "connector_openers":
        return tu.ramp(f["sentence_opener_ratio"], 0.10, 0.45)
    if key == "repetitive_openings":
        # Poca diversidad de palabras iniciales -> aperturas repetitivas.
        return 1.0 - tu.ramp(f["sentence_start_diversity"], 0.45, 0.85)
    if key == "low_lexical_diversity":
        # MATTR alto = vocabulario variado (humano); bajo = repetitivo (IA).
        return 1.0 - tu.ramp(f["mattr"], 0.60, 0.85)
    if key == "hedging":
        return tu.ramp(f["hedging_density"], 1.0, 4.0)
    if key == "uniform_paragraphs":
        # Solo informa con 3+ párrafos; con menos no penaliza.
        if f.get("_paragraph_count", 1) < 3:
            return 0.0
        return 1.0 - tu.ramp(f["paragraph_len_cv"], 0.15, 0.60)
    if key == "list_structure":
        return tu.ramp(f["list_marker_ratio"], 0.10, 0.50)
    if key == "typographic_polish":
        return tu.ramp(f["typographic_density"], 0.30, 2.00)
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
