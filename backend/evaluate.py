"""Evalúa el detector sobre un conjunto APARTADO (eval_data/), que NUNCA se usa
para entrenar. Así medimos la mejora de forma honesta (sin datos en muestra).

    eval_data/ia/      -> textos de IA      (se espera IA alta,  >= umbral)
    eval_data/humano/  -> textos humanos    (se espera IA baja,  <  umbral)

Reporta matriz de confusión, precisión, recall y F1 para la clase "IA", además
del error de calibración (qué tan bien el "%" refleja la probabilidad real).

Uso (desde backend/):  python evaluate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.analysis.engine import analyze          # noqa: E402
from app.config import MODEL_PATH                  # noqa: E402
from app.model.trainer import load_model           # noqa: E402

THRESHOLD = 50  # IA% a partir del cual se clasifica como "IA"
EVAL_DIR = Path(__file__).resolve().parent / "eval_data"


def _calibration_error(probs: list[float], labels: list[int], bins: int = 5) -> dict:
    """Brier score y ECE (Expected Calibration Error).

    - Brier: error cuadrático medio entre probabilidad y resultado (0 = perfecto).
    - ECE: diferencia media |confianza - acierto| por tramos de probabilidad.
      Mide si un "70 %" acierta de verdad ~70 % de las veces.
    """
    if not probs:
        return {"brier": None, "ece": None}
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)
    ece = 0.0
    n = len(probs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (p > lo or b == 0) and p <= hi]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(conf - acc)
    return {"brier": round(brier, 4), "ece": round(ece, 4)}


def run() -> None:
    model = load_model(MODEL_PATH)
    print("Modelo entrenado:", "sí" if model else "no (solo heurísticas)")
    print(f"Umbral de decisión: IA >= {THRESHOLD}%\n")

    tp = fp = tn = fn = 0
    probs: list[float] = []
    labels: list[int] = []

    for label, expected_ai in (("ia", True), ("humano", False)):
        folder = EVAL_DIR / label
        if not folder.exists():
            continue
        for fp_path in sorted(folder.glob("*.txt")):
            text = fp_path.read_text(encoding="utf-8")
            ia = analyze(text, model=model)["scores"]["ai_probability"]
            probs.append(ia / 100.0)
            labels.append(1 if expected_ai else 0)
            pred_ai = ia >= THRESHOLD
            if pred_ai and expected_ai:
                tp += 1
            elif pred_ai and not expected_ai:
                fp += 1
            elif not pred_ai and not expected_ai:
                tn += 1
            else:
                fn += 1
            ok = pred_ai == expected_ai
            print(f"  [{label:6}] IA={ia:>3}%  {'OK' if ok else '<-- FALLA'}  {fp_path.name}")

    total = tp + fp + tn + fn
    if not total:
        print("\nNo hay textos en eval_data/. Agrega ejemplos en eval_data/ia y eval_data/humano.")
        return

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    cal = _calibration_error(probs, labels)

    print("\n--- Matriz de confusión (clase positiva = IA) ---")
    print(f"  Verdaderos IA (TP):     {tp}")
    print(f"  Falsos IA (FP):         {fp}   <- humanos marcados como IA")
    print(f"  Verdaderos humano (TN): {tn}")
    print(f"  Falsos humano (FN):     {fn}   <- IA no detectada")
    print("\n--- Métricas ---")
    print(f"  Exactitud:  {round(100 * accuracy)}%")
    print(f"  Precisión:  {round(precision, 3)}   (de lo marcado como IA, cuánto acierta)")
    print(f"  Recall:     {round(recall, 3)}   (de toda la IA, cuánta detecta)")
    print(f"  F1:         {round(f1, 3)}")
    print(f"\n--- Calibración (¿el % significa lo que dice?) ---")
    print(f"  Brier: {cal['brier']}   ECE: {cal['ece']}   (más bajo = mejor)")


if __name__ == "__main__":
    run()
