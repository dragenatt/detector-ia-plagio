"""Evalúa el detector sobre un conjunto APARTADO (eval_data/), que NUNCA se usa
para entrenar. Así medimos la mejora de forma honesta (sin datos en muestra).

    eval_data/ia/      -> textos de IA      (se espera IA alta,  >= umbral)
    eval_data/humano/  -> textos humanos    (se espera IA baja,  <  umbral)

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


def run() -> None:
    model = load_model(MODEL_PATH)
    print("Modelo entrenado:", "sí" if model else "no (solo heurísticas)")
    print(f"Umbral de decisión: IA >= {THRESHOLD}%\n")

    total = correct = 0
    for label, expected_ai in (("ia", True), ("humano", False)):
        folder = EVAL_DIR / label
        for fp in sorted(folder.glob("*.txt")):
            text = fp.read_text(encoding="utf-8")
            ia = analyze(text, model=model)["scores"]["ai_probability"]
            ok = (ia >= THRESHOLD) == expected_ai
            total += 1
            correct += int(ok)
            print(f"  [{label:6}] IA={ia:>3}%  {'OK' if ok else '<-- FALLA'}  {fp.name}")

    if total:
        print(f"\nExactitud en el set apartado: {correct}/{total} = {round(100 * correct / total)}%")


if __name__ == "__main__":
    run()
