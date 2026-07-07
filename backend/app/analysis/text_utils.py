"""Utilidades de texto: tokenización, segmentación en oraciones y léxicos ES.

Todo es Python estándar (sin dependencias) para que el motor sea liviano y
fácil de testear. Pensado para español, pero funciona razonablemente en
cualquier idioma latino.
"""
from __future__ import annotations

import math
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
    # ampliación: conectores que la IA tiende a encadenar
    "por su parte", "en este contexto", "en relación con", "con respecto a",
    "en lo que respecta", "de igual manera", "de igual forma", "de igual modo",
    "en otras palabras", "dicho de otro modo", "como resultado", "por esta razón",
    "por tal motivo", "a raíz de", "en virtud de", "por ello", "por tanto",
    "cabe señalar que", "vale la pena mencionar", "en suma", "para concluir",
    "en síntesis", "como consecuencia", "de ahí que", "es por ello que",
}

# Conectores típicos AL INICIO de oración. La IA abre muchas frases con ellos
# ("Además, ...", "Por otro lado, ..."). Se buscan como prefijo de la oración.
SENTENCE_OPENERS: tuple[str, ...] = (
    "además", "asimismo", "igualmente", "sin embargo", "no obstante",
    "por lo tanto", "por consiguiente", "en consecuencia", "por ende",
    "de hecho", "es decir", "por ejemplo", "por otra parte", "por otro lado",
    "por su parte", "finalmente", "en primer lugar", "en segundo lugar",
    "en tercer lugar", "por último", "en conclusión", "en resumen",
    "en este sentido", "en este contexto", "de igual manera", "de igual forma",
    "como resultado", "por ello", "por tanto", "a su vez", "del mismo modo",
    "en síntesis", "para concluir", "en definitiva",
)

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
    # ampliación: muletillas de relleno habituales en texto generado
    "en el ámbito de", "en el marco de", "en términos generales",
    "desde tiempos inmemoriales", "a lo largo del tiempo", "en pocas palabras",
    "es ampliamente reconocido", "es bien sabido", "no es ningún secreto",
    "en el panorama actual", "en un mundo cada vez más", "vale la pena destacar",
    "merece la pena señalar", "es crucial entender", "es vital comprender",
    "abre un abanico de", "una amplia gama de", "una herramienta poderosa",
    "el mundo que nos rodea", "marca una diferencia", "el futuro de",
    "se ha convertido en", "ofrece numerosas ventajas", "presenta múltiples desafíos",
]

# Atenuadores / "hedging": fórmulas prudentes y genéricas de las que la IA
# abusa para no comprometerse ("suele", "en general", "podría"...).
HEDGES: list[str] = [
    "generalmente", "en general", "por lo general", "habitualmente",
    "normalmente", "comúnmente", "frecuentemente", "a menudo", "suele ",
    "suelen ", "tiende a", "tienden a", "puede llegar a", "podría", "podrían",
    "es posible que", "probablemente", "posiblemente", "en muchos casos",
    "en algunos casos", "en ciertos casos", "en cierta medida", "hasta cierto punto",
    "de alguna manera", "de algún modo", "relativamente", "aproximadamente",
    "en su mayoría", "entre otros", "entre otras cosas", "diversos factores",
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

# Palabras funcionales de alta frecuencia (perfil estilométrico clásico).
# Su distribución relativa es una "firma" del autor difícil de imitar: se usa
# como vector de rasgos individuales (frecuencia de cada una por 100 palabras).
FUNCTION_WORDS: tuple[str, ...] = (
    "de", "la", "que", "el", "en", "y", "a", "los", "se", "no",
)

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


# Línea que es un ítem de lista o enumeración ("- ...", "* ...", "1. ...", "a) ...").
_LIST_RE = re.compile(r"^\s*(?:[-*•·–—]|\d+[.)]|[a-zA-Z][.)])\s+", re.UNICODE)


def is_list_line(line: str) -> bool:
    """True si la línea empieza como un ítem de viñeta o enumeración."""
    return bool(_LIST_RE.match(line))


def first_word(sentence: str) -> str:
    """Primera palabra (solo letras, en minúsculas) de una oración."""
    m = _WORD_RE.search(sentence)
    return m.group().lower() if m else ""


def starts_with_opener(sentence: str, openers: Iterable[str]) -> bool:
    """True si la oración empieza con alguno de los conectores dados."""
    s = strip_accents(sentence.lower()).lstrip()
    for op in openers:
        op_n = strip_accents(op.lower())
        if s.startswith(op_n):
            rest = s[len(op_n):]
            if rest == "" or not rest[0].isalpha():  # respeta el límite de palabra
                return True
    return False


# Sufijos morfológicos de la prosa "administrativa" que la IA favorece:
# adverbios en -mente y nominalizaciones (-ción, -miento, -dad...).
_MENTE_RE = re.compile(r"mente$")
_NOMINAL_RE = re.compile(r"(?:cion|ciones|sion|siones|miento|mientos|dad|dades)$")


def suffix_densities(word_list: list[str]) -> tuple[float, float]:
    """(adverbios -mente, nominalizaciones) por cada 100 palabras."""
    if not word_list:
        return 0.0, 0.0
    plain = [strip_accents(w) for w in word_list]
    mente = sum(1 for w in plain if len(w) > 6 and _MENTE_RE.search(w))
    nominal = sum(1 for w in plain if len(w) > 5 and _NOMINAL_RE.search(w))
    n = len(word_list)
    return mente * 100.0 / n, nominal * 100.0 / n


def char_trigram_diversity(text: str) -> float:
    """Trigramas de caracteres únicos / totales: baja diversidad = prosa que
    recicla las mismas construcciones (patrón frecuente en texto generado)."""
    norm = normalize_for_match(text)
    if len(norm) < 3:
        return 1.0
    grams = [norm[i:i + 3] for i in range(len(norm) - 2)]
    return len(set(grams)) / len(grams)


def vocab_entropy(word_list: list[str]) -> float:
    """Entropía de Shannon del vocabulario, normalizada a [0,1].

    1 = todas las palabras se usan por igual (máxima variedad); valores bajos
    delatan un texto que insiste en las mismas palabras.
    """
    if not word_list:
        return 0.0
    counts = Counter(word_list)
    n = len(word_list)
    if len(counts) <= 1:
        return 0.0
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return h / math.log2(len(counts))


def similar_len_run_ratio(lengths: list[int], tolerance: float = 0.25) -> float:
    """Racha más larga de oraciones consecutivas de longitud similar, como
    fracción del total. Los humanos rompen el ritmo; la IA encadena oraciones
    "clonadas" en longitud."""
    if len(lengths) < 2:
        return 0.0
    best = run = 1
    for prev, cur in zip(lengths, lengths[1:]):
        if abs(cur - prev) <= tolerance * max(prev, cur, 1):
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best / len(lengths)


def moving_avg_ttr(toks: list[str], window: int = 50) -> float:
    """Type-Token Ratio de media móvil: diversidad léxica robusta a la longitud.

    El TTR simple cae cuanto más largo es el texto, lo que sesga la comparación.
    Promediar el TTR sobre ventanas de tamaño fijo lo corrige. 1 = muy diverso.
    """
    if not toks:
        return 0.0
    if len(toks) <= window:
        return len(set(toks)) / len(toks)
    ratios = []
    for i in range(len(toks) - window + 1):
        chunk = toks[i:i + window]
        ratios.append(len(set(chunk)) / window)
    return sum(ratios) / len(ratios)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ramp(x: float, lo: float, hi: float) -> float:
    """Mapea x al rango [0,1] de forma lineal entre lo..hi (con recorte)."""
    if hi == lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)
