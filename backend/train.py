"""CLI para entrenar el detector de IA desde el corpus etiquetado.

Uso (desde la carpeta backend/):
    python train.py
    python train.py --epochs 1200 --lr 0.08

Lee los textos de training_data/<etiqueta>/ y guarda models/ai_model.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import MODEL_PATH, TRAINING_DIR   # noqa: E402
from app.model.trainer import train               # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el detector de IA de Veraz.")
    parser.add_argument("--data", default=str(TRAINING_DIR),
                        help="Carpeta con el corpus etiquetado.")
    parser.add_argument("--out", default=str(MODEL_PATH),
                        help="Ruta de salida del modelo (JSON).")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    print("Entrenando con el corpus en:", args.data)
    report = train(args.data, args.out, epochs=args.epochs, lr=args.lr)

    print("\n--- Resultado del entrenamiento ---")
    print("Conteo por etiqueta:", report.get("counts"))
    print(f"Textos humanos: {report.get('n_human')}  |  Textos IA: {report.get('n_ai')}")
    if report.get("trained"):
        print(f"Exactitud (entrenamiento): {report.get('train_accuracy')}")
        cv = report.get("cross_validation")
        if cv:
            print(f"Validación cruzada (k={cv['k']}): "
                  f"F1 = {cv['f1_mean']} ± {cv['f1_std']}  por fold: {cv['f1_folds']}")
            print(f"  fuera de muestra -> exactitud: {cv['accuracy']}  "
                  f"precisión: {cv['precision_ia']}  recall: {cv['recall_ia']}")
        elif report.get("holdout_accuracy") is not None:
            print(f"Exactitud (validación):    {report.get('holdout_accuracy')}")
        hm = report.get("holdout_metrics") or report.get("train_metrics")
        if hm and not cv:
            print(f"Detección de IA -> precisión: {hm['precision_ia']}  "
                  f"recall: {hm['recall_ia']}  F1: {hm['f1_ia']}")
        cb, ca = report.get("calibration_before"), report.get("calibration_after")
        if cb and ca:
            print(f"Calibración (ECE) antes: {cb['ece']}  ->  después: {ca['ece']}  "
                  "(más bajo = el % refleja mejor la realidad)")
        print(f"Modelo guardado en: {report.get('model_path')}")
    print("\n" + report.get("message", ""))


if __name__ == "__main__":
    main()
