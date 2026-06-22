"""Entrenamiento del clasificador a partir del corpus etiquetado.

Lee los textos de training_data/<etiqueta>/*.txt y entrena el detector de IA.

Mapa de etiquetas -> objetivo binario (humano=0 / IA=1):
    humano, original, plagiado  -> 0  (escritos por personas)
    ia                          -> 1  (generados por IA)
    mixto                       -> se reserva para validación (no entrena)

'plagiado' es texto humano copiado: para el EJE de IA cuenta como humano (0);
su utilidad para plagio es servir de corpus de referencia (ver plagiarism.py).

Salida: models/ai_model.json con pesos, escalador y metadatos.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

from ..analysis import features as features_mod
from .classifier import LogisticRegression, train_test_split

LABEL_TO_TARGET = {
    "humano": 0,
    "original": 0,
    "plagiado": 0,
    "ia": 1,
    # "mixto" se excluye del entrenamiento a propósito.
}

MIN_WORDS = 40  # ignora textos demasiado cortos para tener rasgos estables


def _read_texts(folder: Path) -> list[str]:
    texts = []
    for ext in ("*.txt", "*.md"):
        for fp in glob.glob(str(folder / ext)):
            try:
                txt = Path(fp).read_text(encoding="utf-8", errors="ignore").strip()
                if len(txt.split()) >= MIN_WORDS:
                    texts.append(txt)
            except OSError:
                continue
    return texts


def load_dataset(training_dir: str | Path) -> tuple[list, list, dict]:
    training_dir = Path(training_dir)
    X, y = [], []
    counts: dict[str, int] = {}
    for label, target in LABEL_TO_TARGET.items():
        folder = training_dir / label
        if not folder.exists():
            continue
        texts = _read_texts(folder)
        counts[label] = len(texts)
        for txt in texts:
            X.append(features_mod.to_vector(features_mod.extract(txt)))
            y.append(target)
    return X, y, counts


def train(training_dir: str | Path, model_out: str | Path,
          epochs: int = 800, lr: float = 0.1) -> dict:
    X, y, counts = load_dataset(training_dir)
    n_human = sum(1 for v in y if v == 0)
    n_ai = sum(1 for v in y if v == 1)

    report = {"counts": counts, "n_human": n_human, "n_ai": n_ai,
              "trained": False, "message": ""}

    if n_human < 3 or n_ai < 3:
        report["message"] = (
            "Se necesitan al menos 3 textos humanos y 3 de IA para entrenar. "
            f"Hay {n_human} humanos y {n_ai} de IA. "
            "Agrega ejemplos en training_data/ y vuelve a entrenar. "
            "Mientras tanto, el detector funciona solo con heurísticas.")
        return report

    clf = LogisticRegression(lr=lr, epochs=epochs,
                             feature_names=features_mod.FEATURE_ORDER)

    # Validación simple si hay suficientes datos.
    test_acc = None
    if n_human + n_ai >= 12:
        Xtr, ytr, Xte, yte = train_test_split(X, y, test_ratio=0.25)
        clf.fit(Xtr, ytr)
        test_acc = round(clf.accuracy(Xte, yte), 3)

    # Entrenamiento final con TODOS los datos.
    clf.fit(X, y)
    train_acc = round(clf.accuracy(X, y), 3)

    clf.meta = {
        "n_human": n_human, "n_ai": n_ai, "counts": counts,
        "train_accuracy": train_acc, "holdout_accuracy": test_acc,
        "feature_order": features_mod.FEATURE_ORDER,
    }
    clf.save(model_out)

    report.update({
        "trained": True,
        "train_accuracy": train_acc,
        "holdout_accuracy": test_acc,
        "model_path": str(model_out),
        "message": "Modelo entrenado y guardado correctamente.",
    })
    return report


def load_model(model_path: str | Path):
    if Path(model_path).exists():
        return LogisticRegression.load(model_path)
    return None
