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

import logging
import math

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


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _apply_calibration(p: float, calibration: dict | None) -> float:
    """Reescala la probabilidad combinada con Platt: sigmoid(a*logit(p)+b)."""
    if not calibration:
        return p
    z = calibration["a"] * _logit(p) + calibration["b"]
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def heuristic_probability(features: dict) -> float:
    """Probabilidad de IA [0,1] solo por heurísticas (sin modelo)."""
    h = sum(s["weight"] * _signal_value(s["key"], features) for s in SIGNALS)
    return tu.clamp(h, 0.0, 1.0)


# Peso base del modelo en la mezcla, y "piso" de evidencia heurística.
BASE_MODEL_WEIGHT = 0.5
ADAPTIVE_FLOOR = 0.35


def _effective_weight(heuristic: float, base_weight: float) -> float:
    """Reduce el peso del modelo cuando la heurística NO ve marcas de IA.

    Evita falsos positivos: un modelo puede equivocarse con confianza por
    correlaciones espurias (p. ej. 'palabras largas') en textos académicos
    humanos. Si la estilometría no detecta señales de IA, el modelo no
    condena por sí solo (filosofía multi-señal: una sola señal no basta).
    """
    if heuristic >= ADAPTIVE_FLOOR:
        return base_weight
    return base_weight * (heuristic / ADAPTIVE_FLOOR)


def combine(heuristic: float, model_proba: float | None,
            base_weight: float = BASE_MODEL_WEIGHT) -> float:
    """Mezcla heurística + modelo con peso adaptativo (probabilidad cruda)."""
    if model_proba is None:
        return heuristic
    w = _effective_weight(heuristic, base_weight)
    return w * model_proba + (1 - w) * heuristic


# --------------------------------------------------------------------------- #
# Análisis por oración (ventanas deslizantes)
# --------------------------------------------------------------------------- #
# Una oración sola tiene poca señal estadística; usamos una ventana de la
# oración con sus vecinas (i-1, i, i+1) y graduamos la confianza según cuántas
# palabras aporta la ventana.

MIN_WINDOW_WORDS = 12   # por debajo no se puntúa (señal insuficiente)
FULL_CONF_WORDS = 60    # a partir de aquí la confianza local es máxima


def sentence_scores(text: str, model=None) -> list[dict]:
    """Puntaje de IA [0,1] por oración usando heurística + modelo sobre la
    ventana (oración ± 1 vecina). `score` es None si no hay señal suficiente.
    """
    from . import features as features_mod  # import local para evitar ciclos

    sents = tu.split_sentences(text)
    out: list[dict] = []
    model_failed = False  # loguear una sola vez, no por cada oración
    for i, s in enumerate(sents):
        window = " ".join(x["text"] for x in sents[max(0, i - 1):i + 2])
        w_words = len(tu.words(window))
        if w_words < MIN_WINDOW_WORDS:
            out.append({"index": i, "score": None, "confidence": 0.0,
                        "words": len(tu.words(s["text"]))})
            continue
        feats = features_mod.extract(window)
        h = heuristic_probability(feats)
        mp = None
        if model is not None and not model_failed:
            try:
                mp = float(model.predict_proba_one(features_mod.to_vector(feats)))
            except Exception:
                model_failed = True
                logging.getLogger("veraz.ai_detection").warning(
                    "El modelo falló al puntuar ventanas por oración; "
                    "el resaltado seguirá solo con heurísticas.", exc_info=True)
                mp = None
        out.append({
            "index": i,
            "score": round(combine(h, mp), 4),
            "confidence": round(tu.ramp(w_words, MIN_WINDOW_WORDS, FULL_CONF_WORDS), 4),
            "words": len(tu.words(s["text"])),
        })
    return out


def ai_sentence_fraction(scores: list[dict]) -> float | None:
    """Fracción del texto (ponderada por palabras y confianza) cuyas oraciones
    puntúan como IA. Es la estimación honesta para documentos MIXTOS: si la
    mitad del texto parece IA, el resultado debe rondar el 50 %, no un extremo.
    """
    num = den = 0.0
    for s in scores:
        if s["score"] is None:
            continue
        # La confianza modula el peso pero no lo domina: si no, las oraciones
        # cortas (ventana chica) desaparecerían de la fracción.
        w = s["words"] * (0.5 + 0.5 * s["confidence"])
        if w <= 0:
            continue
        # Rampa suave en vez de corte duro: una oración "casi IA" (0.5) cuenta
        # a medias; por debajo de 0.35 no cuenta y por encima de 0.65 cuenta
        # entera. Evita que el borde del umbral decida la fracción.
        num += w * tu.ramp(s["score"], 0.35, 0.65)
        den += w
    return num / den if den else None


MIN_SCORED_SENTENCES = 4  # mínimo de oraciones puntuadas para fiarse del mix


def adjust_for_heterogeneity(probability: int, scores: list[dict]) -> tuple[int, dict]:
    """Corrige el % global cuando el documento es HETEROGÉNEO (mitad IA,
    mitad humano). El análisis global promedia rasgos y cae en un extremo;
    la fracción de oraciones tipo IA es la estimación correcta en ese caso.

    heterogeneity = 4·f·(1−f): 0 en textos puros, 1 cuando f=0.5. Solo se
    ajusta si supera 0.5 (mezcla real), tirando el % hacia la fracción.
    """
    frac = ai_sentence_fraction(scores)
    scored = sum(1 for s in scores if s["score"] is not None)
    info = {
        "ai_sentence_fraction": None if frac is None else round(frac, 3),
        "heterogeneity": 0.0,
        "adjusted": False,
    }
    if frac is None or scored < MIN_SCORED_SENTENCES:
        return probability, info
    het = 4.0 * frac * (1.0 - frac)
    info["heterogeneity"] = round(het, 3)
    if het <= 0.5:
        return probability, info
    # Cuanto más mezclado, más manda la fracción por oraciones (tope 0.85
    # para no descartar del todo la vista global).
    pull = min(het, 0.85)
    p = (1 - pull) * (probability / 100.0) + pull * frac
    info["adjusted"] = True
    return round(tu.clamp(p, 0.03, 0.97) * 100), info


def detect(features: dict, model_proba: float | None = None,
           model_weight: float = 0.5, final_calibration: dict | None = None) -> dict:
    """Calcula la probabilidad de IA y el desglose por señal.

    - features         : dict de features.extract()
    - model_proba      : probabilidad [0,1] del clasificador entrenado (o None).
    - model_weight     : cuánto pesa el modelo frente a las heurísticas (0..1).
    - final_calibration: Platt {a,b} para que el % final refleje la realidad.
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

    combined = combine(heuristic, model_proba, model_weight)
    used_model = model_proba is not None

    # Calibración del valor combinado: que el "%" signifique lo que dice.
    combined = _apply_calibration(combined, final_calibration)

    # Recorte: jamás afirmamos 0 % ni 100 % (ningún detector es infalible).
    probability = round(tu.clamp(combined, 0.03, 0.97) * 100)

    return {
        "probability": probability,            # 0..100
        "heuristic_probability": round(heuristic * 100),
        "model_probability": round(model_proba * 100) if model_proba is not None else None,
        "used_model": used_model,
        "signals": breakdown,
    }
