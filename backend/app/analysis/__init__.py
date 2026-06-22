"""Motor de análisis de Veraz.

Cada módulo aísla una responsabilidad para que sea fácil de entender,
testear y, más adelante, reemplazar por un modelo más avanzado:

- text_utils      : tokenización, oraciones, léxicos en español.
- features        : extracción de rasgos estilométricos (vector de features).
- ai_detection    : combina varias señales -> probabilidad de IA (explicable).
- plagiarism      : TF-IDF + coseno + n-gramas contra un corpus de referencia.
- highlighting    : puntaje por oración -> fragmentos coloreados.
- originality     : combina plagio + IA -> índice de originalidad.
- explain         : traduce los números a lenguaje simple.
- recommendations : sugerencias para mejorar sin volver el texto artificial.
- engine          : orquesta todo y devuelve el resultado completo.
"""
