"""Configuración central: rutas y constantes del backend."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # .../backend/app
BACKEND_DIR = BASE_DIR.parent                        # .../backend

TRAINING_DIR = BACKEND_DIR / "training_data"
MODELS_DIR = BACKEND_DIR / "models"
MODEL_PATH = MODELS_DIR / "ai_model.json"
DB_PATH = BACKEND_DIR / "veraz.db"

# Carpetas cuyos .txt se usan como corpus de referencia para el plagio,
# además de los documentos que el usuario suba por la interfaz.
REFERENCE_FOLDERS = [TRAINING_DIR / "referencias", TRAINING_DIR / "plagiado"]

# CORS: orígenes del frontend de desarrollo (Vite).
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
