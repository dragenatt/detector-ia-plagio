"""Detección de plagio contra un corpus de referencia local.

Dos técnicas complementarias, ambas en Python puro:

1. COINCIDENCIA TEXTUAL (n-gramas / "shingles"): mide qué fracción de los
   n-gramas del texto aparece también en alguna fuente. Es lo que mejor
   detecta copia literal y permite localizar los fragmentos coincidentes.
   -> Es la métrica principal de "similitud / plagio".

2. SIMILITUD TEMÁTICA (TF-IDF + coseno): mide parecido de vocabulario
   ponderado por rareza. Detecta paráfrasis y cercanía de tema aunque no
   haya copia literal. Se reporta como señal secundaria.

El corpus de referencia son los documentos que el usuario carga (subidos por
la interfaz o colocados en training_data/referencias y training_data/plagiado).
Más adelante esto se puede ampliar con APIs/buscadores externos (ver README).
"""
from __future__ import annotations

import math
from collections import Counter

from . import text_utils as tu

SHINGLE_N = 5           # tamaño de n-grama para copia literal
SENT_FLAG_THRESHOLD = 0.55   # solapamiento mínimo para marcar una oración


def _shingles(toks: list[str], n: int = SHINGLE_N) -> set[tuple]:
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return set(tu.ngrams(toks, n))


def _containment(a: set, b: set) -> float:
    """Fracción de los elementos de A que están en B (0..1)."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _tfidf_vectors(docs_tokens: list[list[str]]):
    """Construye vectores TF-IDF. Devuelve (vectores, idf)."""
    n = len(docs_tokens)
    df: Counter = Counter()
    for toks in docs_tokens:
        for w in set(toks):
            df[w] += 1
    idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}

    vectors = []
    for toks in docs_tokens:
        tf = Counter(toks)
        total = max(len(toks), 1)
        vec = {w: (c / total) * idf.get(w, 0.0) for w, c in tf.items()}
        vectors.append(vec)
    return vectors, idf


def _cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return tu.safe_div(dot, na * nb)


def analyze(text: str, references: list[dict] | None = None) -> dict:
    """Compara el texto contra una lista de referencias.

    references: lista de {"name": str, "text": str}.
    Devuelve similitud global, fuentes más parecidas y oraciones marcadas.
    """
    references = references or []
    input_tokens = tu.match_tokens(text)
    input_shingles = _shingles(input_tokens)

    if not references or not input_shingles:
        return {
            "similarity": 0,
            "topical_similarity": 0,
            "matches": [],
            "flagged_sentences": [],
            "has_corpus": bool(references),
            "note": ("No hay corpus de referencia cargado: la similitud no se "
                     "puede estimar. Sube documentos para comparar."
                     if not references else "Texto demasiado corto para comparar."),
        }

    # Preparar referencias.
    ref_tokens = [tu.match_tokens(r["text"]) for r in references]
    ref_shingles = [_shingles(t) for t in ref_tokens]

    # --- Similitud por coincidencia textual (containment por documento) ---
    matches = []
    for ref, rsh in zip(references, ref_shingles):
        overlap = _containment(input_shingles, rsh)
        if overlap > 0.02:
            matches.append({
                "source": ref.get("name", "fuente sin nombre"),
                "overlap": round(overlap * 100),
            })
    matches.sort(key=lambda m: m["overlap"], reverse=True)
    similarity = matches[0]["overlap"] if matches else 0

    # --- Similitud temática (TF-IDF + coseno) ---
    vectors, _idf = _tfidf_vectors([input_tokens] + ref_tokens)
    input_vec = vectors[0]
    cos_scores = [_cosine(input_vec, v) for v in vectors[1:]]
    topical = round(max(cos_scores) * 100) if cos_scores else 0

    # --- Localización de oraciones coincidentes ---
    flagged = []
    for i, sent in enumerate(tu.split_sentences(text)):
        s_tokens = tu.match_tokens(sent["text"])
        s_sh = _shingles(s_tokens)
        if not s_sh:
            continue
        # ¿qué fuente cubre mejor esta oración?
        best_src, best_ov = None, 0.0
        for ref, rsh in zip(references, ref_shingles):
            ov = _containment(s_sh, rsh)
            if ov > best_ov:
                best_ov, best_src = ov, ref.get("name", "fuente")
        if best_ov >= SENT_FLAG_THRESHOLD:
            flagged.append({
                "index": i,
                "text": sent["text"],
                "start": sent["start"],
                "end": sent["end"],
                "source": best_src,
                "overlap": round(best_ov * 100),
            })

    return {
        "similarity": similarity,           # métrica principal (0..100)
        "topical_similarity": topical,      # señal secundaria (0..100)
        "matches": matches[:5],
        "flagged_sentences": flagged,
        "has_corpus": True,
        "note": None,
    }
