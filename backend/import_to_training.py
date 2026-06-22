"""Promueve un análisis del historial al corpus de entrenamiento.

Cuando el detector se equivoca (p. ej. marca como "original" un texto que sabes
que es de IA), este script copia ese texto del historial a la carpeta de
entrenamiento que le indiques, para que el modelo aprenda de ese caso.

Uso (desde la carpeta backend/):
    python import_to_training.py <id> <etiqueta>
Ejemplos:
    python import_to_training.py 2 ia        # el análisis 2 era de IA
    python import_to_training.py 7 humano

Etiquetas válidas: humano, ia, mixto, original, plagiado, referencias.
Después, reentrena con:  python train.py
"""
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.config import DB_PATH, TRAINING_DIR  # noqa: E402

VALID = {"humano", "ia", "mixto", "original", "plagiado", "referencias"}


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:50] or "item"


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[2] not in VALID:
        print(__doc__)
        print("Etiquetas válidas:", ", ".join(sorted(VALID)))
        return
    analysis_id, label = int(sys.argv[1]), sys.argv[2]

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT title, text FROM analyses WHERE id=?",
                       (analysis_id,)).fetchone()
    con.close()
    if not row:
        print(f"No existe ningún análisis con id={analysis_id}.")
        return

    folder = TRAINING_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"hist{analysis_id}_{_slug(row['title'])}.txt"
    out.write_text(row["text"], encoding="utf-8")
    print(f"OK -> {out}  ({len(row['text'].split())} palabras, etiqueta '{label}')")


if __name__ == "__main__":
    main()
