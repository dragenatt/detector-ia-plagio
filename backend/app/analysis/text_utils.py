"""Utilidades de texto: tokenización, segmentación en oraciones y léxicos ES.

Todo es Python estándar (sin dependencias) para que el motor sea liviano y
fácil de testear. Pensado para español, pero funciona razonablemente en
cualquier idioma latino.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

# --------------------------------------------------------------------------- #
# Léxicos en español
# --------------------------------------------------------------------------- #

# Conectores discursivos. Su SOBREuso es una de las señales de texto "mecánico".
CONNECTORS: set[str] = {
    "además", "asimismo", "igualmente", "sin embargo", "no obstante",
    "por lo tanto", "por consiguiente", "en consecuencia", "por ende",
    "de hecho", "es decir", "por ejemplo", "en cambio", "aunque",
    "mientras que", "debido a", "gracias a", "por otra parte", "por un lado",
    "por otro lado", "finalmente", "en primer lugar", "en segundo lugar",
    "en tercer lugar", "por último", "así que", "de modo que", "de manera que",
    "con el fin de", "a fin de", "dado que", "puesto que", "ya que",
    "en definitiva", "en resumen", "en conclusión", "a su vez", "del mismo modo",
}

# Frases "académicas pero vacías": suenan formales pero aportan poco. Los
# modelos de IA tienden a abusar de ellas.
GENERIC_PHRASES: list[str] = [
    "en la actualidad", "hoy en día", "cabe destacar", "cabe mencionar",
    "cabe resaltar", "es importante mencionar", "es importante destacar",
    "es importante señalar", "en este sentido", "en tal sentido",
    "de esta manera", "de esta forma", "a lo largo de la historia",
    "en el mundo actual", "en la sociedad actual", "juega un papel fundamental",
    "juega un papel importante", "desempeña un papel", "es fundamental",
    "es esencial", "sin lugar a dudas", "no cabe duda", "cada vez más",
    "un sinfín de", "la importancia de", "a modo de conclusión",
    "en última instancia", "resulta evidente", "es menester", "es preciso señalar",
]

# Marcadores de voz personal / opinión. Su AUSENCIA aporta a la señal de IA
# (con cuidado: muchos textos académicos legítimos evitan la 1ª persona).
PERSONAL_MARKERS: set[str] = {
    "yo", "mi", "mí", "me", "conmigo", "nosotros", "nuestro", "nuestra",
    "creo", "pienso", "opino", "considero", "siento", "personalmente",
    "recuerdo", "viví", "aprendí", "noté", "sentí", "imagino", "supongo",
}
# Frases de opinión de 1ª persona (se buscan como subcadena).
PERSONAL_PHRASES: list[str] = [
    "en mi opinión", "desde mi punto de vista", "a mi juicio", "me parece",
    "me gusta", "no me gusta", "mi experiencia", "para mí", "yo creo",
    "yo pienso", "en lo personal",
]

# Marcadores informales / "errores humanos" (su presencia baja la señal de IA).
INFORMAL_MARKERS: list[str] = [
    "jaja", "jeje", "osea", "o sea", "la verdad", "obvio", "súper", "super",
    "buenísimo", "porfa", "tipo", "onda", "bueno,", "pues", "eh,", "mmm",
]

STOPWORDS_ES: set[str] = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "ante", "bajo", "con", "contra", "desde", "en", "entre", "hacia",
    "hasta", "para", "por", "según", "sin", "sobre", "tras", "y", "e", "o", "u",
    "que", "qué", "como", "cómo", "cuando", "cuándo", "donde", "dónde", "quien",
    "se", "su", "sus", "lo", "le", "les", "me", "te", "nos", "es", "son", "ser",
    "fue", "era", "eran", "ha", "han", "he", "has", "hay", "este", "esta",
    "estos", "estas", "ese", "esa", "esos", "esas", "esto", "eso", "aquel",
    "más", "muy", "ya", "no", "sí", "si", "pero", "porque", "también", "todo",
    "todos", "toda", "todas", "cada", "mismo", "misma", "tan", "tanto", "entre",
}


# --------------------------------------------------------------------------- #
# Normalización y tokenización
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # secuencias de letras (sin dígitos)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Una oración: cualquier cosa hasta un terminador (. ! ? …) inclusive.
_SENT_RE = re.compile(r"[^.!?…\n]+[.!?…]*[\n]*", re.UNICODE)


def strip_accents(text: str) -> str:
    """Quita tildes (útil para comparar n-gramas de forma robusta)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_for_match(text: str) -> str:
    """Normaliza para comparar plagio: minúsculas, sin tildes, sin puntuación."""
    text = strip_accents(text.lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    """Palabras (solo letras), en minúsculas. Conserva tildes."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def tokens(text: str) -> list[str]:
    """Tokens alfanuméricos en minúsculas (incluye números)."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def match_tokens(text: str) -> list[str]:
    """Tokens normalizados (sin tildes) para detección de plagio."""
    return normalize_for_match(text).split()


def split_sentences(text: str) -> list[dict]:
    """Divide en oraciones conservando offsets que cubren TODO el texto.

    Devuelve dicts: {"text": <recortado>, "start": int, "end": int}.
    Los offsets (start/end) cubren el texto crudo de forma contigua para que
    el frontend pueda reconstruir el texto original exactamente al resaltar.
    """
    out: list[dict] = []
    for m in _SENT_RE.finditer(text):
        raw = m.group()
        stripped = raw.strip()
        if stripped:
            out.append({"text": stripped, "start": m.start(), "end": m.end()})
    return out


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def ngrams(seq: list[str], n: int) -> list[tuple]:
    if n <= 0 or len(seq) < n:
        return []
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def count_occurrences(haystack: str, needles: Iterable[str]) -> int:
    """Cuenta apariciones (como subcadena) de cada needle en el texto."""
    low = haystack.lower()
    return sum(low.count(n) for n in needles)


def repeated_ngram_ratio(toks: list[str], n: int = 3) -> float:
    """Proporción de n-gramas repetidos: alta repetición -> texto monótono."""
    grams = ngrams(toks, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ramp(x: float, lo: float, hi: float) -> float:
    """Mapea x al rango [0,1] de forma lineal entre lo..hi (con recorte)."""
    if hi == lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)
