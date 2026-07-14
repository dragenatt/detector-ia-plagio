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

import math
import random

from ..analysis import ai_detection
from ..analysis import features as features_mod
from .classifier import LogisticRegression, platt_fit

# Peso del modelo frente a la heurística usado en producción (engine/API).
MODEL_WEIGHT = 0.5


def _sigmoid(z: float) -> float:
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _calibration_error(probs: list[float], y: list[int], bins: int = 5) -> dict:
    """Brier y ECE: qué tan bien la probabilidad refleja la frecuencia real."""
    if not probs:
        return {"brier": None, "ece": None}
    brier = sum((p - yi) ** 2 for p, yi in zip(probs, y)) / len(probs)
    n = len(probs)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (p > lo or b == 0) and p <= hi]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(y[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(conf - acc)
    return {"brier": round(brier, 4), "ece": round(ece, 4)}

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


def load_dataset(training_dir: str | Path) -> tuple[list, list, dict, list]:
    training_dir = Path(training_dir)
    X, y = [], []
    feats_list: list[dict] = []
    counts: dict[str, int] = {}
    for label, target in LABEL_TO_TARGET.items():
        folder = training_dir / label
        if not folder.exists():
            continue
        texts = _read_texts(folder)
        counts[label] = len(texts)
        for txt in texts:
            feats = features_mod.extract(txt)
            feats_list.append(feats)
            X.append(features_mod.to_vector(feats))
            y.append(target)
    return X, y, counts, feats_list


def _quality_metrics(clf, X, y) -> dict:
    """Precisión, exhaustividad (recall) y F1 para la clase 'IA' (=1).

    La exactitud sola engaña con clases desbalanceadas; estas métricas dicen
    cuántas IA detecta de verdad (recall) y cuántas de sus avisos aciertan
    (precision).
    """
    tp = fp = tn = fn = 0
    for xi, yi in zip(X, y):
        p = clf.predict_one(xi)
        if p == 1 and yi == 1:
            tp += 1
        elif p == 1 and yi == 0:
            fp += 1
        elif p == 0 and yi == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision_ia": round(precision, 3),
        "recall_ia": round(recall, 3),
        "f1_ia": round(f1, 3),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _stratified_folds(y: list[int], k: int, seed: int = 42) -> list[list[int]]:
    """Reparte los índices en k folds manteniendo la proporción de clases."""
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = [i for i, v in enumerate(y) if v == cls]
        rng.shuffle(idx)
        for pos, i in enumerate(idx):
            folds[pos % k].append(i)
    return folds


def cross_validate(X: list, y: list, k: int = 5, lr: float = 0.1,
                   epochs: int = 1000, l2: float = 0.001,
                   class_weight: str | None = "balanced", seed: int = 42) -> dict:
    """Validación cruzada k-fold: cada texto se evalúa con un modelo que NO lo
    vio al entrenar. Así las métricas no dependen de la suerte de un corte.

    Devuelve el F1 por fold (media y desviación) y las métricas agregadas
    sobre las predicciones "fuera de muestra" de todos los folds.
    """
    folds = _stratified_folds(y, k, seed)
    f1_per_fold: list[float] = []
    pooled_pred: list[int] = [0] * len(y)
    for fold in folds:
        if not fold:
            continue
        test_set = set(fold)
        Xtr = [X[i] for i in range(len(X)) if i not in test_set]
        ytr = [y[i] for i in range(len(y)) if i not in test_set]
        clf = LogisticRegression(lr=lr, epochs=epochs, l2=l2)
        clf.fit(Xtr, ytr, class_weight=class_weight)
        tp = fp = fn = 0
        for i in fold:
            p = clf.predict_one(X[i])
            pooled_pred[i] = p
            if p == 1 and y[i] == 1:
                tp += 1
            elif p == 1 and y[i] == 0:
                fp += 1
            elif p == 0 and y[i] == 1:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_per_fold.append(round(f1, 3))

    n = len(f1_per_fold) or 1
    mean = sum(f1_per_fold) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in f1_per_fold) / n)

    # Métricas agregadas sobre TODAS las predicciones fuera de muestra.
    tp = sum(1 for p, yi in zip(pooled_pred, y) if p == 1 and yi == 1)
    fp = sum(1 for p, yi in zip(pooled_pred, y) if p == 1 and yi == 0)
    tn = sum(1 for p, yi in zip(pooled_pred, y) if p == 0 and yi == 0)
    fn = sum(1 for p, yi in zip(pooled_pred, y) if p == 0 and yi == 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "k": k,
        "f1_folds": f1_per_fold,
        "f1_mean": round(mean, 3),
        "f1_std": round(std, 3),
        "accuracy": round((tp + tn) / max(len(y), 1), 3),
        "precision_ia": round(prec, 3),
        "recall_ia": round(rec, 3),
        "f1_ia": round(f1, 3),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def train(training_dir: str | Path, model_out: str | Path,
          epochs: int = 600, lr: float = 0.05, l2: float = 0.01,
          class_weight: str | None = "balanced") -> dict:
    X, y, counts, feats_list = load_dataset(training_dir)
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

    clf = LogisticRegression(lr=lr, epochs=epochs, l2=l2,
                             feature_names=features_mod.FEATURE_ORDER)

    # Validación cruzada k-fold (mejor que un corte único 75/25: cada texto
    # se evalúa fuera de muestra y las métricas no dependen de la suerte).
    test_acc = None
    holdout_metrics = None
    cv = None
    if n_human + n_ai >= 12:
        cv = cross_validate(X, y, k=5, lr=lr, epochs=epochs, l2=l2,
                            class_weight=class_weight)
        test_acc = cv["accuracy"]
        holdout_metrics = {k: cv[k] for k in
                           ("precision_ia", "recall_ia", "f1_ia", "confusion")}

    # Entrenamiento final con TODOS los datos (compensando el desbalance).
    clf.fit(X, y, class_weight=class_weight)
    train_acc = round(clf.accuracy(X, y), 3)
    train_metrics = _quality_metrics(clf, X, y)

    # 1) Calibración del modelo (Platt sobre su logit).
    clf.fit_calibration(X, y)

    # 2) Calibración del % COMBINADO final (heurística + modelo), que es el
    #    valor que ve el usuario. Así un "70 %" reflejará la frecuencia real.
    def _blend(i):
        h = ai_detection.heuristic_probability(feats_list[i])
        m = clf.predict_proba_one(X[i])
        return ai_detection.combine(h, m, MODEL_WEIGHT)

    blends = [_blend(i) for i in range(len(X))]
    cal_before = _calibration_error(blends, y)
    clf.final_calibration = platt_fit([ai_detection._logit(p) for p in blends], y)
    blends_cal = [ai_detection._apply_calibration(p, clf.final_calibration) for p in blends]
    cal_after = _calibration_error(blends_cal, y)

    clf.meta = {
        "n_human": n_human, "n_ai": n_ai, "counts": counts,
        "train_accuracy": train_acc, "holdout_accuracy": test_acc,
        "class_weight": class_weight,
        "train_metrics": train_metrics, "holdout_metrics": holdout_metrics,
        "cross_validation": cv,
        "calibration_before": cal_before, "calibration_after": cal_after,
        "feature_order": features_mod.FEATURE_ORDER,
        "n_features": len(features_mod.FEATURE_ORDER),
    }
    clf.save(model_out)

    report.update({
        "trained": True,
        "train_accuracy": train_acc,
        "holdout_accuracy": test_acc,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "cross_validation": cv,
        "calibration_before": cal_before,
        "calibration_after": cal_after,
        "model_path": str(model_out),
        "message": "Modelo entrenado y guardado correctamente.",
    })
    return report


def load_model(model_path: str | Path):
    if Path(model_path).exists():
        return LogisticRegression.load(model_path)
    return None
