"""Extracción de rasgos estilométricos del texto.

Produce un diccionario de números que describen el ESTILO de escritura
(no el tema). Estos rasgos alimentan dos cosas:

  1. El detector de IA por señales (ai_detection.py).
  2. El clasificador entrenable (model/classifier.py), que aprende qué
     combinaciones de rasgos son típicas de texto humano vs. IA.

`FEATURE_ORDER` fija el orden de las features para que el vector que entrena
el modelo y el que se evalúa en producción sean siempre consistentes.
"""
from __future__ import annotations

import statistics
from collections import Counter

from . import text_utils as tu

# Orden canónico de los rasgos numéricos usados por el clasificador.
FEATURE_ORDER: list[str] = [
    "avg_sentence_len",
    "sentence_len_std",
    "sentence_len_cv",
    "type_token_ratio",
    "hapax_ratio",
    "avg_word_len",
    "punct_ratio",
    "comma_density",
    "connector_density",
    "stopword_ratio",
    "repeated_trigram_ratio",
    "generic_phrase_density",
    "personal_voice_density",
    "informal_density",
    "burstiness",
    # --- rasgos nuevos (más patrones para distinguir IA de humano) ---
    "sentence_start_diversity",       # variedad de palabras al inicio de oración
    "sentence_opener_ratio",          # % de oraciones que abren con conector
    "paragraph_len_cv",               # uniformidad de longitud de párrafos
    "list_marker_ratio",              # proporción de líneas tipo lista/enumeración
    "hedging_density",                # atenuadores prudentes ("suele", "en general")
    "mattr",                          # diversidad léxica robusta (TTR media móvil)
    "repeated_bigram_ratio",          # repetición de pares de palabras
    "typographic_density",            # comillas tipográficas y rayas "pulidas"
]


def extract(text: str) -> dict:
    """Devuelve un dict con todos los rasgos + algunos conteos crudos útiles."""
    sents = tu.split_sentences(text)
    toks = tu.tokens(text)
    word_list = tu.words(text)
    n_words = len(word_list)
    n_sents = max(len(sents), 1)
    n_chars = len(text)

    # Longitud de oraciones (en palabras) -> ritmo / uniformidad.
    sent_word_counts = [len(tu.words(s["text"])) for s in sents] or [0]
    avg_sentence_len = tu.safe_div(sum(sent_word_counts), len(sent_word_counts))
    sentence_len_std = statistics.pstdev(sent_word_counts) if len(sent_word_counts) > 1 else 0.0
    sentence_len_cv = tu.safe_div(sentence_len_std, avg_sentence_len)

    # Diversidad léxica.
    counts = Counter(word_list)
    unique = len(counts)
    type_token_ratio = tu.safe_div(unique, n_words)
    hapax = sum(1 for c in counts.values() if c == 1)
    hapax_ratio = tu.safe_div(hapax, n_words)
    avg_word_len = tu.safe_div(sum(len(w) for w in word_list), n_words)

    # Puntuación.
    punct = sum(1 for ch in text if ch in ".,;:!?¡¿—-()\"'…")
    commas = text.count(",")
    punct_ratio = tu.safe_div(punct, n_words)
    comma_density = tu.safe_div(commas, n_sents)  # comas por oración

    # Densidades léxicas por cada 100 palabras (escala estable e interpretable).
    per100 = lambda c: tu.safe_div(c * 100.0, n_words)
    connector_count = tu.count_occurrences(text, tu.CONNECTORS)
    generic_count = tu.count_occurrences(text, tu.GENERIC_PHRASES)
    personal_count = (
        sum(1 for w in word_list if w in tu.PERSONAL_MARKERS)
        + tu.count_occurrences(text, tu.PERSONAL_PHRASES)
    )
    informal_count = tu.count_occurrences(text, tu.INFORMAL_MARKERS)

    connector_density = per100(connector_count)
    generic_phrase_density = per100(generic_count)
    personal_voice_density = per100(personal_count)
    informal_density = per100(informal_count)

    stopwords = sum(1 for w in word_list if w in tu.STOPWORDS_ES)
    stopword_ratio = tu.safe_div(stopwords, n_words)
    repeated_trigram_ratio = tu.repeated_ngram_ratio(toks, 3)
    repeated_bigram_ratio = tu.repeated_ngram_ratio(toks, 2)

    # --- Patrones nuevos -------------------------------------------------- #
    # Aperturas de oración: la IA repite la primera palabra y abre muchas
    # frases con conectores ("Además, ...", "Por otro lado, ...").
    first_words = [tu.first_word(s["text"]) for s in sents]
    first_words = [w for w in first_words if w]
    sentence_start_diversity = (
        tu.safe_div(len(set(first_words)), len(first_words)) if first_words else 1.0
    )
    opener_hits = sum(1 for s in sents if tu.starts_with_opener(s["text"], tu.SENTENCE_OPENERS))
    sentence_opener_ratio = tu.safe_div(opener_hits, n_sents)

    # Uniformidad de párrafos: la IA produce párrafos de tamaño muy parecido.
    paras = tu.paragraphs(text)
    para_word_counts = [len(tu.words(p)) for p in paras]
    if len(para_word_counts) >= 2:
        p_mean = sum(para_word_counts) / len(para_word_counts)
        paragraph_len_cv = tu.safe_div(statistics.pstdev(para_word_counts), p_mean)
    else:
        paragraph_len_cv = 0.5  # neutro: con un solo párrafo no es informativo

    # Estructura de listas/enumeraciones (muy frecuente en respuestas de IA).
    non_empty_lines = [ln for ln in text.splitlines() if ln.strip()]
    list_lines = sum(1 for ln in non_empty_lines if tu.is_list_line(ln))
    list_marker_ratio = tu.safe_div(list_lines, len(non_empty_lines))

    # Atenuadores / hedging.
    hedge_count = tu.count_occurrences(text, tu.HEDGES)
    hedging_density = per100(hedge_count)

    # Diversidad léxica robusta a la longitud.
    mattr = tu.moving_avg_ttr(word_list, window=50)

    # Marcas tipográficas "pulidas": rayas y comillas curvas que la IA inserta.
    typographic_count = sum(text.count(ch) for ch in ("—", "–", "“", "”", "‘", "’", "…"))
    typographic_density = per100(typographic_count)

    # "Burstiness": qué tan irregular es el ritmo. Los humanos alternan
    # oraciones muy cortas y muy largas; la IA tiende a un ritmo plano.
    if len(sent_word_counts) > 1 and avg_sentence_len > 0:
        burstiness = tu.clamp(sentence_len_cv, 0.0, 2.0)
    else:
        burstiness = 0.0

    return {
        # rasgos del modelo
        "avg_sentence_len": round(avg_sentence_len, 4),
        "sentence_len_std": round(sentence_len_std, 4),
        "sentence_len_cv": round(sentence_len_cv, 4),
        "type_token_ratio": round(type_token_ratio, 4),
        "hapax_ratio": round(hapax_ratio, 4),
        "avg_word_len": round(avg_word_len, 4),
        "punct_ratio": round(punct_ratio, 4),
        "comma_density": round(comma_density, 4),
        "connector_density": round(connector_density, 4),
        "stopword_ratio": round(stopword_ratio, 4),
        "repeated_trigram_ratio": round(repeated_trigram_ratio, 4),
        "generic_phrase_density": round(generic_phrase_density, 4),
        "personal_voice_density": round(personal_voice_density, 4),
        "informal_density": round(informal_density, 4),
        "burstiness": round(burstiness, 4),
        # rasgos nuevos
        "sentence_start_diversity": round(sentence_start_diversity, 4),
        "sentence_opener_ratio": round(sentence_opener_ratio, 4),
        "paragraph_len_cv": round(paragraph_len_cv, 4),
        "list_marker_ratio": round(list_marker_ratio, 4),
        "hedging_density": round(hedging_density, 4),
        "mattr": round(mattr, 4),
        "repeated_bigram_ratio": round(repeated_bigram_ratio, 4),
        "typographic_density": round(typographic_density, 4),
        # conteos crudos (para explicaciones y depuración)
        "_word_count": n_words,
        "_sentence_count": len(sents),
        "_char_count": n_chars,
        "_paragraph_count": len(paras),
        "_connector_count": connector_count,
        "_generic_count": generic_count,
        "_personal_count": personal_count,
        "_informal_count": informal_count,
        "_hedge_count": hedge_count,
        "_opener_count": opener_hits,
    }


def to_vector(feats: dict) -> list[float]:
    """Convierte el dict de rasgos en un vector ordenado para el clasificador."""
    return [float(feats.get(name, 0.0)) for name in FEATURE_ORDER]
