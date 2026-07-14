"""API de Veraz (FastAPI).

Expone el motor de análisis, el historial, el corpus de referencia, el
entrenamiento del modelo y la generación de reportes PDF.

Ejecutar (desde la carpeta backend/):
    uvicorn app.main:app --reload --port 8000
o simplemente:
    python run.py
"""
from __future__ import annotations

import glob
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("veraz.api")

from . import config, db, extract, report
from .analysis.engine import analyze as run_analysis
from .model.trainer import load_model, train as train_model
from .schemas import AnalyzeRequest, ReferenceIn


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al arrancar: inicializa la base de datos y carga el modelo si existe.
    db.init_db()
    app.state.model = load_model(config.MODEL_PATH)
    yield


def _current_model():
    """Modelo cargado, tolerante a que el arranque (lifespan) no haya corrido."""
    return getattr(app.state, "model", None)


app = FastAPI(title="Veraz API", version="0.1.0",
              description="Estimación de originalidad, plagio y uso de IA.",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Cualquier error inesperado queda LOGUEADO con contexto (ruta y tipo),
    y el cliente recibe un mensaje claro en vez de un 500 mudo."""
    logger.error("Error no manejado en %s %s: %s",
                 request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": ("Error interno al procesar la petición. "
                            "Revisa los logs del servidor para el detalle.")},
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _folder_references() -> list[dict]:
    """Lee documentos de referencia desde las carpetas configuradas."""
    refs = []
    for folder in config.REFERENCE_FOLDERS:
        for ext in ("*.txt", "*.md"):
            for fp in glob.glob(str(Path(folder) / ext)):
                try:
                    txt = Path(fp).read_text(encoding="utf-8", errors="ignore")
                    if txt.strip():
                        refs.append({"name": Path(fp).name, "text": txt})
                except OSError:
                    continue
    return refs


def _all_references() -> list[dict]:
    """Corpus de referencia = carpetas + documentos subidos por el usuario."""
    refs = _folder_references()
    for r in db.list_references(include_text=True):
        refs.append({"name": r["name"], "text": r["text"]})
    return refs


def _analyze_text(text: str, req: AnalyzeRequest) -> dict:
    if not text or not text.strip():
        raise HTTPException(400, "El texto está vacío.")
    model = _current_model() if req.use_model else None
    try:
        result = run_analysis(text, references=_all_references(),
                              model=model, model_weight=req.model_weight)
    except ValueError as e:
        raise HTTPException(400, str(e))
    analysis_id = db.save_analysis(result, text, req.title) if req.save else None
    return {"id": analysis_id, **result}


# --------------------------------------------------------------------------- #
# Análisis
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _current_model() is not None}


@app.post("/api/analyze")
def analyze_text_endpoint(req: AnalyzeRequest) -> dict:
    return _analyze_text(req.text, req)


@app.post("/api/analyze/batch")
async def analyze_batch_endpoint(files: List[UploadFile] = File(...),
                                 use_model: bool = True) -> dict:
    """Analiza VARIOS archivos de una vez (p. ej. los trabajos de una clase).

    Devuelve una fila por archivo (IA%, plagio, originalidad, confianza) y
    detecta además plagio CRUZADO: cada trabajo se compara contra los demás
    del lote para descubrir si se copiaron entre sí.
    """
    if not files:
        raise HTTPException(400, "No se recibió ningún archivo.")
    if len(files) > 60:
        raise HTTPException(400, "Máximo 60 archivos por lote.")

    # 1. Extraer texto de cada archivo (los que fallen se reportan, no rompen).
    docs: list[dict] = []
    for f in files:
        name = f.filename or "sin nombre"
        ext = Path(name).suffix.lower()
        row = {"name": name, "ok": False, "error": None}
        if ext not in config.ALLOWED_EXTENSIONS:
            row["error"] = f"Formato no soportado ({ext or 'desconocido'})."
            docs.append(row)
            continue
        data = await f.read()
        if len(data) > config.MAX_UPLOAD_BYTES:
            row["error"] = "Supera el límite de 5 MB."
            docs.append(row)
            continue
        try:
            text = extract.extract_text(name, data)
        except Exception as e:
            logger.warning("Lote: fallo al leer %r: %s", name, e, exc_info=True)
            row["error"] = f"No se pudo leer: {e}"
            docs.append(row)
            continue
        if not (text or "").strip():
            row["error"] = "Sin texto extraíble."
            docs.append(row)
            continue
        row["ok"] = True
        row["text"] = text
        docs.append(row)

    valid = [d for d in docs if d["ok"]]
    corpus_refs = _all_references()
    model = _current_model() if use_model else None

    # 2. Analizar cada documento; como referencia de plagio, el corpus global
    #    MÁS los otros documentos del lote (para detectar copia entre ellos).
    results = []
    for i, d in enumerate(valid):
        others = [{"name": o["name"], "text": o["text"]}
                  for j, o in enumerate(valid) if j != i]
        try:
            r = run_analysis(d["text"], references=corpus_refs + others,
                             model=model)
        except ValueError as e:
            results.append({"name": d["name"], "ok": False, "error": str(e)})
            continue
        # ¿con qué OTRO archivo del lote coincide más? (plagio cruzado)
        other_names = {o["name"] for o in others}
        cross = [m for m in r["plagiarism"]["matches"] if m["source"] in other_names]
        results.append({
            "name": d["name"],
            "ok": True,
            "words": r["meta"]["word_count"],
            "ai_probability": r["scores"]["ai_probability"],
            "plagiarism": r["scores"]["plagiarism"],
            "originality": r["scores"]["originality"],
            "confidence": r["confidence"]["level"],
            "cross_match": cross[0] if cross else None,
        })

    failed = [{"name": d["name"], "ok": False, "error": d["error"]}
              for d in docs if not d["ok"]]
    return {"count": len(valid), "failed": len(failed),
            "rows": results + failed}


@app.post("/api/analyze/file")
async def analyze_file_endpoint(file: UploadFile = File(...),
                                use_model: bool = True,
                                save: bool = True) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Formato no soportado: {ext or 'desconocido'}.")
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "El archivo supera el límite de 5 MB.")
    try:
        text = extract.extract_text(file.filename, data)
    except Exception as e:
        logger.warning("Fallo al extraer texto de %r (%d bytes): %s",
                       file.filename, len(data), e, exc_info=True)
        raise HTTPException(422, f"No se pudo leer el archivo: {e}")
    if not (text or "").strip():
        raise HTTPException(422, ("El archivo no contiene texto extraíble "
                                  "(¿es un PDF escaneado o un archivo vacío?)."))
    req = AnalyzeRequest(text=text, title=file.filename,
                         use_model=use_model, save=save)
    result = _analyze_text(text, req)
    result["extracted_text"] = text
    return result


# --------------------------------------------------------------------------- #
# Historial
# --------------------------------------------------------------------------- #

@app.get("/api/history")
def history() -> List[dict]:  # List[...] (no list[...]) por compat. con Python 3.8
    return db.list_analyses()


@app.get("/api/history/{analysis_id}")
def history_detail(analysis_id: int) -> dict:
    row = db.get_analysis(analysis_id)
    if not row:
        raise HTTPException(404, "Análisis no encontrado.")
    return row


@app.delete("/api/history/{analysis_id}")
def history_delete(analysis_id: int) -> dict:
    db.delete_analysis(analysis_id)
    return {"deleted": analysis_id}


@app.delete("/api/history")
def history_clear() -> dict:
    db.clear_analyses()
    return {"cleared": True}


# --------------------------------------------------------------------------- #
# Reportes PDF
# --------------------------------------------------------------------------- #

@app.get("/api/report/{analysis_id}")
def report_pdf(analysis_id: int) -> Response:
    row = db.get_analysis(analysis_id)
    if not row or not row.get("payload"):
        raise HTTPException(404, "Análisis no encontrado.")
    try:
        pdf_bytes = report.generate_pdf(row["payload"], row.get("text", ""), row.get("title"))
    except Exception as e:
        logger.error("Fallo generando el PDF del análisis %s: %s",
                     analysis_id, e, exc_info=True)
        raise HTTPException(500, "No se pudo generar el reporte PDF de este análisis.")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="veraz-reporte-{analysis_id}.pdf"'},
    )


# --------------------------------------------------------------------------- #
# Corpus de referencia (para plagio)
# --------------------------------------------------------------------------- #

@app.get("/api/references")
def references_list() -> dict:
    return {
        "uploaded": db.list_references(),
        "from_folders": [r["name"] for r in _folder_references()],
    }


@app.post("/api/references")
def references_add(ref: ReferenceIn) -> dict:
    ref_id = db.add_reference(ref.name, ref.text)
    return {"id": ref_id, "name": ref.name}


@app.post("/api/references/file")
async def references_add_file(file: UploadFile = File(...)) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Formato no soportado: {ext or 'desconocido'}.")
    data = await file.read()
    try:
        text = extract.extract_text(file.filename, data)
    except Exception as e:
        logger.warning("Fallo al extraer referencia de %r: %s",
                       file.filename, e, exc_info=True)
        raise HTTPException(422, f"No se pudo leer el archivo: {e}")
    if not (text or "").strip():
        raise HTTPException(422, "El archivo no contiene texto extraíble.")
    ref_id = db.add_reference(file.filename, text)
    return {"id": ref_id, "name": file.filename}


@app.delete("/api/references/{ref_id}")
def references_delete(ref_id: int) -> dict:
    db.delete_reference(ref_id)
    return {"deleted": ref_id}


# --------------------------------------------------------------------------- #
# Entrenamiento del modelo
# --------------------------------------------------------------------------- #

@app.get("/api/model")
def model_status() -> dict:
    model = _current_model()
    return {
        "loaded": model is not None,
        "meta": getattr(model, "meta", None) if model else None,
    }


@app.post("/api/train")
def train_endpoint() -> dict:
    result = train_model(config.TRAINING_DIR, config.MODEL_PATH)
    if result.get("trained"):
        app.state.model = load_model(config.MODEL_PATH)  # recarga en caliente
    return result


# --------------------------------------------------------------------------- #
# Frontend compilado (un solo servicio)
# --------------------------------------------------------------------------- #
# Si existe frontend/dist (tras `npm run build`), la interfaz se sirve desde el
# propio backend en http://localhost:8000. Así Veraz funciona como UN ÚNICO
# servicio (sin Node en tiempo de ejecución), ideal para el lanzador de
# escritorio. Se monta al final para no tapar las rutas /api.
_DIST_DIR = config.BACKEND_DIR.parent / "frontend" / "dist"
if (_DIST_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="frontend")
else:
    # Si la interfaz no está compilada, en vez de un críptico 404
    # ({"detail":"Not Found"}) damos una explicación clara de qué hacer.
    from fastapi.responses import HTMLResponse

    @app.get("/", response_class=HTMLResponse)
    def _frontend_no_compilado() -> str:
        return (
            "<html><head><meta charset='utf-8'><title>Veraz</title></head>"
            "<body style='font-family:sans-serif;max-width:640px;margin:60px auto;"
            "line-height:1.6;color:#222'>"
            "<h1>Veraz · interfaz no compilada</h1>"
            "<p>La API está funcionando, pero la interfaz web "
            "(<code>frontend/dist</code>) no se encuentra.</p>"
            "<p>Para generarla necesitas <b>Node 18+</b> y ejecutar:</p>"
            "<pre style='background:#f4f4f4;padding:12px;border-radius:8px'>"
            "cd frontend\nnpm install\nnpm run build</pre>"
            "<p>Luego vuelve a iniciar el lanzador. La documentación de la API "
            "sigue disponible en <a href='/docs'>/docs</a>.</p>"
            "</body></html>"
        )
