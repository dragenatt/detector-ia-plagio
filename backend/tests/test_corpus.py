"""Ciclo de aprendizaje supervisado por el usuario: añadir ejemplo etiquetado ->
aparece en el conteo -> reentrenar -> el modelo lo tiene en cuenta.

Usa un directorio de corpus TEMPORAL para no ensuciar el real. Correr:
    python tests/test_corpus.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HUMANO = ("Ayer me quede hasta tarde con el trabajo y casi no duermo, la verdad. "
          "Mi hermana me ayudo con la parte dificil de las encuestas y al final "
          "quedo bien, aunque el punto tres todavia no me convence del todo. "
          "Manana lo reviso con la profe temprano antes de la clase de historia.")
IA = ("En la actualidad, cabe destacar que la tecnologia constituye un pilar "
      "fundamental de la sociedad moderna. Ademas, resulta evidente que su impacto "
      "es significativo en multiples sectores de la economia. Por lo tanto, es "
      "esencial promover su desarrollo responsable y sostenible en el tiempo.")


def test_add_example_appears_in_counts_and_trains():
    from app.model import trainer as T
    tmp = Path(tempfile.mkdtemp())
    try:
        # Semilla mínima para poder entrenar (3+ por clase).
        for cls, txt in (("humano", HUMANO), ("ia", IA)):
            d = tmp / cls
            d.mkdir(parents=True)
            for i in range(4):
                (d / f"seed_{i}.txt").write_text(txt + f" Variante {i} con mas palabras.",
                                                 encoding="utf-8")

        _, _, counts0, _ = T.load_dataset(tmp)
        base_human = counts0.get("humano", 0)

        # El usuario añade un ejemplo humano en la subcarpeta usuario/ (recursivo).
        udir = tmp / "humano" / "usuario"
        udir.mkdir(parents=True)
        (udir / "user_nuevo.txt").write_text(
            HUMANO + " Este es un ejemplo nuevo que agrega el usuario a mano.",
            encoding="utf-8")

        _, _, counts1, _ = T.load_dataset(tmp)
        assert counts1["humano"] == base_human + 1, \
            "El ejemplo del usuario (subcarpeta) debe contarse en su clase."

        # Reentrenar incluye el ejemplo sin romper.
        model_out = tmp / "model.json"
        report = T.train(tmp, model_out)
        assert report["trained"] is True
        assert model_out.exists()
        assert report["n_human"] == counts1["humano"] + counts1.get("original", 0) + counts1.get("plagiado", 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_train_needs_minimum_examples():
    from app.model import trainer as T
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "humano").mkdir(parents=True)
        (tmp / "humano" / "a.txt").write_text(HUMANO, encoding="utf-8")
        report = T.train(tmp, tmp / "m.json")
        assert report["trained"] is False
        assert "al menos" in report["message"].lower()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_add_example_appears_in_counts_and_trains()
    print("[OK] añadir ejemplo -> aparece en conteo -> reentrena y lo incluye")
    test_train_needs_minimum_examples()
    print("[OK] entrenar avisa si faltan ejemplos, sin romper")
    print("\nTODAS LAS PRUEBAS DE CORPUS PASARON [OK]")
