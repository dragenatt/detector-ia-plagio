"""Análisis por lotes: varios documentos a la vez + plagio cruzado entre ellos.
Usa TestClient de FastAPI. Correr:  python tests/test_batch.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IA = ("En la actualidad, cabe destacar que la tecnologia juega un papel "
      "fundamental. Ademas, es importante destacar que la educacion resulta "
      "esencial. Por lo tanto, es fundamental promover la innovacion constante.")
HUM = ("Ayer me quede hasta tarde con el trabajo y casi no duermo. La verdad el "
       "tema me costo un monton, pero mi hermana me ayudo. Al final quedo bien, "
       "aunque el punto tres no me convence. Manana lo reviso con la profe.")
COPION = ("Ayer me quede hasta tarde con el trabajo y casi no duermo. La verdad "
          "el tema me costo un monton, pero mi hermana me ayudo. Al final quedo bien.")


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    # Context manager -> dispara el lifespan (carga modelo, init db).
    return TestClient(app)


def test_batch_and_cross_plagiarism():
    with _client() as c:
        files = [
            ("files", ("ia.txt", IA.encode(), "text/plain")),
            ("files", ("humano.txt", HUM.encode(), "text/plain")),
            ("files", ("copion.txt", COPION.encode(), "text/plain")),
        ]
        r = c.post("/api/analyze/batch", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["count"] == 3 and data["failed"] == 0
    by = {row["name"]: row for row in data["rows"]}
    assert by["ia.txt"]["ai_probability"] > by["humano.txt"]["ai_probability"]
    # El plagio cruzado debe emparejar humano.txt y copion.txt entre sí.
    assert by["copion.txt"]["cross_match"] is not None
    assert by["copion.txt"]["cross_match"]["source"] == "humano.txt"


def test_batch_reports_bad_file_without_crashing():
    with _client() as c:
        files = [
            ("files", ("bueno.txt", HUM.encode(), "text/plain")),
            ("files", ("malo.exe", b"x", "application/octet-stream")),
            ("files", ("vacio.txt", b"   ", "text/plain")),
        ]
        r = c.post("/api/analyze/batch", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["failed"] == 2 and data["count"] == 1
    bad = {row["name"]: row for row in data["rows"] if not row["ok"]}
    assert "malo.exe" in bad and bad["malo.exe"]["error"]
    assert "vacio.txt" in bad


def test_health_ok_without_lifespan():
    """Aun sin disparar el lifespan, /api/health no debe romper (getattr)."""
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)  # sin 'with': no corre el lifespan
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "model_loaded" in r.json()


if __name__ == "__main__":
    try:
        import fastapi  # noqa: F401
    except ImportError:
        print("SKIP: FastAPI no instalado en este entorno.")
        sys.exit(0)
    test_batch_and_cross_plagiarism()
    print("[OK] lote analiza varios y detecta copia cruzada")
    test_batch_reports_bad_file_without_crashing()
    print("[OK] archivos invalidos se reportan sin romper el lote")
    print("\nTODAS LAS PRUEBAS DE LOTE PASARON [OK]")
