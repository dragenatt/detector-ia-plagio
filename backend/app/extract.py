"""Extracción de texto desde archivos subidos (.txt, .md, .pdf, .docx).

Las dependencias (pypdf, python-docx) se importan de forma perezosa para que
el resto del backend funcione aunque no estén instaladas.
"""
from __future__ import annotations

import io
from pathlib import Path


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()

    if ext in (".txt", ".md"):
        for enc in ("utf-8", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="ignore")

    if ext == ".pdf":
        return _extract_pdf(data)

    if ext == ".docx":
        return _extract_docx(data)

    raise ValueError(f"Formato no soportado: {ext}")


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Falta la librería 'pypdf' (pip install pypdf).") from e
    reader = PdfReader(io.BytesIO(data))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts).strip()


def _extract_docx(data: bytes) -> str:
    try:
        import docx  # python-docx
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Falta la librería 'python-docx' (pip install python-docx).") from e
    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs).strip()
