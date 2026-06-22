"""Arranque cómodo del servidor de desarrollo.

    python run.py

Equivale a `uvicorn app.main:app --port 8000`. Para autorecarga al editar
código instala watchfiles (`pip install watchfiles`) y usa:
    uvicorn app.main:app --reload --port 8000
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
