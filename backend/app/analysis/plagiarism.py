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
import re
from collections import Counter

from . import text_utils as tu

SHINGLE_N = 5           # tamaño de n-grama para copia literal
SENT_FLAG_THRESHOLD = 0.55   # solapamiento mínimo para marcar una oración

_WORD_SPAN_RE = re.compile(r"\w+", re.UNICODE)


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Offsets (start, end) de cada palabra; alineados 1:1 con match_tokens."""
    return [(m.start(), m.end()) for m in _WORD_SPAN_RE.finditer(text)]


def _longest_common_run(a: list[str], b: list[str]) -> tuple[int, int, int]:
    """Racha más larga de tokens consecutivos comunes entre a y b.

    Devuelve (inicio_en_a, inicio_en_b, longitud). DP por filas con dict para
    mantenerlo liviano (a es una oración: ~30 tokens).
    """
    best_a = best_b = best_len = 0
    prev: dict[int, int] = {}
    positions: dict[str, list[int]] = {}
    for j, tok in enumerate(b):
        positions.setdefault(tok, []).append(j)
    for i, tok in enumerate(a):
        cur: dict[int, int] = {}
        for j in positions.get(tok, ()):
            run = prev.get(j - 1, 0) + 1
            cur[j] = run
            if run > best_len:
                best_len = run
                best_a = i - run + 1
                best_b = j - run + 1
        prev = cur
    return best_a, best_b, best_len


def locate_match(sentence_text: str, ref_text: str,
                 max_source_chars: int = 240) -> dict | None:
    """Localiza el tramo copiado EXACTO: qué parte de la oración coincide con
    la fuente, con offsets locales, y el fragmento correspondiente de la fuente.
    """
    s_tokens = tu.match_tokens(sentence_text)
    r_tokens = tu.match_tokens(ref_text)
    if not s_tokens or not r_tokens:
        return None
    ia, ib, length = _longest_common_run(s_tokens, r_tokens)
    if length < 3:  # menos de 3 palabras seguidas no es evidencia de copia
        return None
    s_spans = _word_spans(sentence_text)
    r_spans = _word_spans(ref_text)
    if len(s_spans) != len(s_tokens) or len(r_spans) != len(r_tokens):
        # La alineación palabra<->token falló (texto atípico): mejor no señalar
        # un tramo equivocado.
        return None
    s_start = s_spans[ia][0]
    s_end = s_spans[ia + length - 1][1]
    r_start = r_spans[ib][0]
    r_end = r_spans[ib + length - 1][1]
    source_fragment = ref_text[r_start:r_end]
    if len(source_fragment) > max_source_chars:
        source_fragment = source_fragment[:max_source_chars].rstrip() + "…"
    return {
        "start": s_start,          # offsets locales a la oración
        "end": s_end,
        "text": sentence_text[s_start:s_end],
        "matched_words": length,
        "source_fragment": source_fragment,
    }


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
        best_src, best_ov, best_ref = None, 0.0, None
        for ref, rsh in zip(references, ref_shingles):
            ov = _containment(s_sh, rsh)
            if ov > best_ov:
                best_ov, best_src, best_ref = ov, ref.get("name", "fuente"), ref
        if best_ov >= SENT_FLAG_THRESHOLD:
            entry = {
                "index": i,
                "text": sent["text"],
                "start": sent["start"],
                "end": sent["end"],
                "source": best_src,
                "overlap": round(best_ov * 100),
            }
            # Tramo copiado exacto (offsets absolutos) + fragmento de la fuente.
            match = locate_match(sent["text"], best_ref.get("text", ""))
            if match:
                entry["match"] = {
                    "start": sent["text_start"] + match["start"],
                    "end": sent["text_start"] + match["end"],
                    "text": match["text"],
                    "matched_words": match["matched_words"],
                    "source_fragment": match["source_fragment"],
                }
            flagged.append(entry)

    return {
        "similarity": similarity,           # métrica principal (0..100)
        "topical_similarity": topical,      # señal secundaria (0..100)
        "matches": matches[:5],
        "flagged_sentences": flagged,
        "has_corpus": True,
        "note": None,
    }
