"""Persistencia en SQLite: historial de análisis y corpus de referencia.

Se usa sqlite3 de la librería estándar (sin ORM) para mantenerlo simple y
sin dependencias. Dos tablas:

  analyses    : cada análisis realizado (para el historial y los reportes).
  references  : documentos de referencia cargados por el usuario (para plagio).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                title TEXT,
                word_count INTEGER,
                originality INTEGER,
                plagiarism INTEGER,
                ai_probability INTEGER,
                summary TEXT,
                text TEXT,
                payload TEXT
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS refs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                name TEXT NOT NULL,
                text TEXT NOT NULL
            )""")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Historial de análisis
# --------------------------------------------------------------------------- #

def save_analysis(result: dict, text: str, title: str | None) -> int:
    s = result["scores"]
    title = title or _auto_title(text)
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO analyses
               (created_at, title, word_count, originality, plagiarism,
                ai_probability, summary, text, payload)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (_now(), title, result["meta"]["word_count"], s["originality"],
             s["plagiarism"], s["ai_probability"],
             result["explanation"]["summary"], text, json.dumps(result, ensure_ascii=False)),
        )
        return cur.lastrowid


def _auto_title(text: str) -> str:
    snippet = " ".join(text.split())[:48]
    return (snippet + "…") if len(snippet) == 48 else (snippet or "Análisis")


def list_analyses(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            """SELECT id, created_at, title, word_count, originality,
                      plagiarism, ai_probability, summary
               FROM analyses ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_analysis(analysis_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["payload"] = json.loads(d["payload"]) if d["payload"] else None
    return d


def delete_analysis(analysis_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))


def clear_analyses() -> None:
    with _conn() as c:
        c.execute("DELETE FROM analyses")


# --------------------------------------------------------------------------- #
# Corpus de referencia
# --------------------------------------------------------------------------- #

def add_reference(name: str, text: str) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO refs (created_at, name, text) VALUES (?,?,?)",
                        (_now(), name, text))
        return cur.lastrowid


def list_references(include_text: bool = False) -> list[dict]:
    cols = "id, created_at, name" + (", text" if include_text else "")
    with _conn() as c:
        rows = c.execute(f"SELECT {cols} FROM refs ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def delete_reference(ref_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM refs WHERE id=?", (ref_id,))
