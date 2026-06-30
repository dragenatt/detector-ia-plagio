"""Regresión logística en Python puro (sin numpy ni sklearn).

Por qué "a mano": mantiene CERO dependencias pesadas, es fácil de entender
para un proyecto académico y se puede serializar a JSON trivialmente. Para
vectores de ~15 features y cientos de ejemplos es más que suficiente.

Incluye estandarización (z-score) de las features, que es importante porque
están en escalas muy distintas (p. ej. longitud de oración vs. ratios 0..1).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path


def _sigmoid(z: float) -> float:
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class LogisticRegression:
    def __init__(self, lr: float = 0.1, epochs: int = 800, l2: float = 0.001,
                 feature_names: list[str] | None = None):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.feature_names = feature_names or []
        self.weights: list[float] = []
        self.bias: float = 0.0
        self.mean: list[float] = []
        self.std: list[float] = []
        self.calibration: dict | None = None        # calibra el logit del modelo
        self.final_calibration: dict | None = None   # calibra el % combinado final
        self.meta: dict = {}

    # ----- estandarización ------------------------------------------------- #
    def _fit_scaler(self, X: list[list[float]]):
        n = len(X)
        d = len(X[0])
        self.mean = [sum(row[j] for row in X) / n for j in range(d)]
        self.std = []
        for j in range(d):
            var = sum((row[j] - self.mean[j]) ** 2 for row in X) / n
            self.std.append(math.sqrt(var) or 1.0)  # evita división por cero

    def _scale(self, x: list[float]) -> list[float]:
        return [(x[j] - self.mean[j]) / self.std[j] for j in range(len(x))]

    # ----- entrenamiento --------------------------------------------------- #
    def _class_weights(self, y, class_weight):
        """Devuelve (peso_clase_0, peso_clase_1).

        'balanced' compensa el desbalance del corpus (p. ej. muchos más textos
        de IA que humanos) para que la clase minoritaria no quede aplastada.
        """
        if class_weight == "balanced":
            n = len(y)
            n_pos = sum(1 for v in y if v == 1) or 1
            n_neg = (n - n_pos) or 1
            return n / (2.0 * n_neg), n / (2.0 * n_pos)
        if isinstance(class_weight, dict):
            return float(class_weight.get(0, 1.0)), float(class_weight.get(1, 1.0))
        return 1.0, 1.0

    def fit(self, X: list[list[float]], y: list[int], class_weight=None):
        if not X:
            raise ValueError("No hay datos de entrenamiento.")
        self._fit_scaler(X)
        Xs = [self._scale(row) for row in X]
        d = len(Xs[0])
        self.weights = [0.0] * d
        self.bias = 0.0
        n = len(Xs)
        w0, w1 = self._class_weights(y, class_weight)
        wsum = sum(w1 if yi == 1 else w0 for yi in y) or 1.0

        for _ in range(self.epochs):
            grad_w = [0.0] * d
            grad_b = 0.0
            for xi, yi in zip(Xs, y):
                wi = w1 if yi == 1 else w0
                pred = _sigmoid(sum(w * xj for w, xj in zip(self.weights, xi)) + self.bias)
                err = (pred - yi) * wi
                for j in range(d):
                    grad_w[j] += err * xi[j]
                grad_b += err
            for j in range(d):
                grad_w[j] = grad_w[j] / wsum + self.l2 * self.weights[j]
                self.weights[j] -= self.lr * grad_w[j]
            self.bias -= self.lr * (grad_b / wsum)
        return self

    # ----- calibración (Platt scaling) ------------------------------------- #
    def _logit(self, x: list[float]) -> float:
        """Logit crudo (z) del modelo para un vector ya en escala original."""
        xs = self._scale(x)
        return sum(w * xj for w, xj in zip(self.weights, xs)) + self.bias

    def fit_calibration(self, X: list[list[float]], y: list[int],
                        lr: float = 0.05, epochs: int = 600):
        """Platt scaling: ajusta sigmoid(a*z + b) para que la probabilidad
        refleje la frecuencia real (un "70 %" acierte ~70 % de las veces).

        Se entrena 1 dimensión (el logit del modelo) contra las etiquetas, lo
        que corrige el exceso de confianza típico de la regresión logística.
        """
        zs = [self._logit(row) for row in X]
        a, b = 1.0, 0.0
        n = len(zs) or 1
        for _ in range(epochs):
            ga = gb = 0.0
            for z, yi in zip(zs, y):
                p = _sigmoid(a * z + b)
                err = p - yi
                ga += err * z
                gb += err
            a -= lr * ga / n
            b -= lr * gb / n
        self.calibration = {"a": a, "b": b}
        return self

    # ----- predicción ------------------------------------------------------ #
    def predict_proba_one(self, x: list[float]) -> float:
        if self.mean and len(x) != len(self.mean):
            raise ValueError(
                f"El vector tiene {len(x)} rasgos pero el modelo espera "
                f"{len(self.mean)}. Reentrena el modelo (cambió el set de rasgos).")
        z = self._logit(x)
        if self.calibration:
            return _sigmoid(self.calibration["a"] * z + self.calibration["b"])
        return _sigmoid(z)

    def predict_one(self, x: list[float], threshold: float = 0.5) -> int:
        return int(self.predict_proba_one(x) >= threshold)

    def accuracy(self, X: list[list[float]], y: list[int]) -> float:
        if not X:
            return 0.0
        correct = sum(1 for xi, yi in zip(X, y) if self.predict_one(xi) == yi)
        return correct / len(X)

    # ----- persistencia ---------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "mean": self.mean,
            "std": self.std,
            "calibration": self.calibration,
            "final_calibration": self.final_calibration,
            "hyperparams": {"lr": self.lr, "epochs": self.epochs, "l2": self.l2},
            "meta": self.meta,
        }

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegression":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        hp = data.get("hyperparams", {})
        clf = cls(lr=hp.get("lr", 0.1), epochs=hp.get("epochs", 800),
                  l2=hp.get("l2", 0.001), feature_names=data.get("feature_names", []))
        clf.weights = data["weights"]
        clf.bias = data["bias"]
        clf.mean = data["mean"]
        clf.std = data["std"]
        clf.calibration = data.get("calibration")
        clf.final_calibration = data.get("final_calibration")
        clf.meta = data.get("meta", {})
        return clf


def platt_fit(logits: list[float], y: list[int],
              lr: float = 0.05, epochs: int = 600) -> dict:
    """Ajusta sigmoid(a*z + b) sobre logits dados (Platt scaling genérico)."""
    a, b = 1.0, 0.0
    n = len(logits) or 1
    for _ in range(epochs):
        ga = gb = 0.0
        for z, yi in zip(logits, y):
            p = _sigmoid(a * z + b)
            err = p - yi
            ga += err * z
            gb += err
        a -= lr * ga / n
        b -= lr * gb / n
    return {"a": a, "b": b}


def train_test_split(X, y, test_ratio=0.25, seed=42):
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return ([X[i] for i in tr], [y[i] for i in tr],
            [X[i] for i in te], [y[i] for i in te])
