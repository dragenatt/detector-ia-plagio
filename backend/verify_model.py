"""Verificación rápida del detector tras reentrenar.

Analiza tres textos con el modelo actual y muestra la probabilidad de IA:
- el guion del usuario (que ahora SÍ está en el corpus -> "dentro de muestra")
- un guion de IA NUEVO (no está en el corpus -> prueba de generalización)
- un texto humano (no debería marcarse como IA -> control de falsos positivos)

También muestra el resultado SOLO con heurísticas (sin modelo) para comparar
cuánto aporta el modelo entrenado.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.analysis.engine import analyze            # noqa: E402
from app.config import MODEL_PATH, TRAINING_DIR     # noqa: E402
from app.model.trainer import load_model            # noqa: E402

# Guion de IA NUEVO (held-out): NO forma parte de training_data.
GUION_NUEVO_IA = """Muy buenas tardes a todos los presentes. El día de hoy tengo el agrado de
presentarles un tema de gran relevancia en la actualidad: la importancia de la
energía renovable. En primer lugar, es fundamental destacar que las energías
limpias representan el futuro de nuestro planeta. Además, cabe mencionar que
reducen de forma significativa la contaminación. Por lo tanto, resulta evidente
que debemos apostar por ellas. No se trata únicamente de cuidar el medio
ambiente, sino también de garantizar un futuro sostenible para las próximas
generaciones. En conclusión, la transición energética es una tarea de todos.
Muchas gracias por su atención."""


def show(label, text, model):
    r = analyze(text, model=model)
    s = r["scores"]
    flag = "sí" if r["meta"]["model_used"] else "no"
    print(f"  {label:<34} IA={s['ai_probability']:>3}%   orig={s['originality']:>3}%   (modelo={flag})")


def main():
    model = load_model(MODEL_PATH)
    print("Modelo cargado:", model is not None)
    if model and model.meta:
        print("  entrenado con:", model.meta.get("counts"))
        print("  precisión (train/holdout):",
              model.meta.get("train_accuracy"), "/", model.meta.get("holdout_accuracy"))

    guion_usuario = (TRAINING_DIR / "ia" / "hist2_guion_05_luis_angel_martinez_docx.txt").read_text(encoding="utf-8")
    humano = (TRAINING_DIR / "humano" / "obra_mochila_monologo.txt").read_text(encoding="utf-8")

    print("\n== CON el modelo nuevo ==")
    show("Guion del usuario (en muestra)", guion_usuario, model)
    show("Guion IA NUEVO (held-out)", GUION_NUEVO_IA, model)
    show("Texto humano (monólogo)", humano, model)

    print("\n== SOLO heurísticas (sin modelo), para comparar ==")
    show("Guion del usuario", guion_usuario, None)
    show("Guion IA NUEVO", GUION_NUEVO_IA, None)
    show("Texto humano", humano, None)


if __name__ == "__main__":
    main()
