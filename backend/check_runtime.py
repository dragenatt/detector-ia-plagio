"""Comprobaciones de arranque para los lanzadores de Veraz."""
from __future__ import annotations

import importlib.util
import sys

REQUIRED_MODULES = (
    "fastapi",
    "uvicorn",
    "pydantic",
    "multipart",
    "pypdf",
    "docx",
    "fpdf",
)


def check_python_version() -> int:
    if sys.version_info < (3, 8):
        print(
            "[ERROR] Veraz necesita Python 3.8 o superior. "
            f"Version actual: {sys.version.split()[0]}",
            file=sys.stderr,
        )
        return 1
    return 0


def check_dependencies() -> int:
    missing: list[str] = []
    for module_name in REQUIRED_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            missing.append(module_name)
    if missing:
        print(
            "[ERROR] Faltan dependencias: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"--version", "--deps"}:
        print("Uso: python check_runtime.py --version|--deps", file=sys.stderr)
        return 2
    if sys.argv[1] == "--version":
        return check_python_version()
    return check_dependencies()


if __name__ == "__main__":
    raise SystemExit(main())
