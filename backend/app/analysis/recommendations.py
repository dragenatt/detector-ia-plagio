"""Recomendaciones para mejorar el texto SIN volverlo más artificial.

La idea es ayudar al estudiante a escribir mejor y de forma más humana, no a
"engañar al detector". Por eso las sugerencias apuntan a calidad real:
aportar voz propia, ejemplos, variedad y buenas citas.
"""
from __future__ import annotations


def build(features: dict, ai_result: dict, plagiarism: dict) -> list[dict]:
    recs: list[dict] = []

    def add(priority, title, detail):
        recs.append({"priority": priority, "title": title, "detail": detail})

    # Basadas en señales de IA concretas.
    sig = {s["key"]: s for s in ai_result["signals"]}

    if sig["generic_phrases"]["severity"] in ("media", "alta"):
        add("alta", "Reemplaza las frases de relleno",
            "Sustituye expresiones como «cabe destacar» o «hoy en día» por "
            "ideas concretas, datos o ejemplos propios. Di QUÉ destaca y POR QUÉ.")

    if sig["connector_overuse"]["severity"] in ("media", "alta"):
        add("media", "Reduce los conectores",
            "No todas las oraciones necesitan «además» o «por lo tanto». "
            "Quita conectores que no aporten lógica real y deja respirar al texto.")

    if sig["low_personal_voice"]["severity"] in ("media", "alta"):
        add("alta", "Aporta tu mirada",
            "Incluye un ejemplo de tu experiencia, una opinión argumentada o una "
            "pregunta. Una voz propia genuina vuelve el texto más humano y valioso.")

    if sig["uniformity"]["severity"] in ("media", "alta") or \
       sig["low_burstiness"]["severity"] in ("media", "alta"):
        add("media", "Varía el ritmo",
            "Alterna oraciones cortas y largas. Una frase breve después de una "
            "extensa marca el ritmo y suena más natural.")

    if sig["repetition"]["severity"] in ("media", "alta"):
        add("media", "Evita repetir ideas",
            "Detecta párrafos que dicen lo mismo con otras palabras y fusiónalos. "
            "Cada párrafo debería aportar algo nuevo.")

    # Basadas en plagio.
    if plagiarism["has_corpus"] and plagiarism["flagged_sentences"]:
        add("alta", "Cita y reescribe las coincidencias",
            "Hay fragmentos muy parecidos a las fuentes. Cítalos correctamente "
            "(APA, etc.) o reescríbelos con tus propias palabras explicando la idea.")

    if not recs:
        add("baja", "Buen trabajo de base",
            "No se detectaron problemas evidentes. Aun así, revisa citas y aporta "
            "ejemplos propios para fortalecer la originalidad.")

    order = {"alta": 0, "media": 1, "baja": 2}
    recs.sort(key=lambda r: order[r["priority"]])
    return recs
