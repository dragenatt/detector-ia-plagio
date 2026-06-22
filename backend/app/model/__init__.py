"""Modelo entrenable de Veraz.

- classifier : regresión logística en Python puro (sin dependencias).
- trainer    : lee el corpus etiquetado, entrena y guarda el modelo en JSON.

El modelo NO reemplaza a las heurísticas: las complementa. El detector de IA
combina ambos (ver analysis/ai_detection.py). Así, sin datos el sistema ya
funciona, y con datos mejora.
"""
