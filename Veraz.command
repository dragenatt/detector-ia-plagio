#!/usr/bin/env bash
# ====================================================================
#  Veraz - Lanzador para macOS (y Linux)
#  Doble clic para iniciar la aplicacion. Prepara el entorno la
#  primera vez (entorno virtual, dependencias e interfaz) y abre el
#  navegador en http://127.0.0.1:8000
# ====================================================================
set -e

trap 'echo; echo "[ERROR] Veraz no pudo iniciar. Revisa el mensaje anterior."; read -r -p "Pulsa Enter para cerrar..."' ERR

# Ir a la carpeta del script (asi funciona con doble clic)
cd "$(dirname "$0")"

echo
echo " ========================================"
echo "   Veraz - Detector de IA y plagio"
echo " ========================================"
echo

# --- 1. Buscar Python -------------------------------------------------
PY=""
if command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
fi
if [ -z "$PY" ]; then
    echo "[ERROR] No se encontro Python. Instala Python 3.8 o superior desde"
    echo "        https://www.python.org/downloads/ (o 'brew install python')."
    read -r -p "Pulsa Enter para cerrar..."
    exit 1
fi

cd backend

# --- 2. Crear el entorno virtual la primera vez ----------------------
if [ ! -x ".venv/bin/python" ]; then
    echo "[1/3] Creando entorno virtual..."
    "$PY" -m venv .venv
fi
VENV_PY=".venv/bin/python"

# Verificar version de Python dentro del entorno virtual (FastAPI requiere 3.8+).
if ! "$VENV_PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "[ERROR] Veraz necesita Python 3.8 o superior."
    echo "        Borra la carpeta backend/.venv despues de instalar una version compatible"
    echo "        y vuelve a ejecutar este archivo."
    read -r -p "Pulsa Enter para cerrar..."
    exit 1
fi

# --- 3. Instalar dependencias (solo la primera vez) -----------------
if [ ! -f ".venv/.deps_ok" ]; then
    echo "[2/3] Instalando dependencias (puede tardar un momento)..."
    "$VENV_PY" -m pip install --upgrade pip >/dev/null
    "$VENV_PY" -m pip install -r requirements.txt
    "$VENV_PY" -c 'import fastapi, uvicorn, pydantic, multipart, pypdf, docx, fpdf'
    echo "ok" > ".venv/.deps_ok"
fi

# Si el marcador existe pero el entorno quedo incompleto, reinstalar.
if ! "$VENV_PY" -c 'import fastapi, uvicorn, pydantic, multipart, pypdf, docx, fpdf' >/dev/null 2>&1; then
    echo "[2/3] Reparando dependencias incompletas..."
    rm -f ".venv/.deps_ok"
    "$VENV_PY" -m pip install -r requirements.txt
    echo "ok" > ".venv/.deps_ok"
fi

# --- 4. Compilar la interfaz si no existe frontend/dist -------------
if [ ! -f "../frontend/dist/index.html" ]; then
    if command -v npm >/dev/null 2>&1; then
        echo "[3/3] Compilando la interfaz (solo la primera vez)..."
        ( cd ../frontend && npm install && npm run build )
    else
        echo "[AVISO] No se encontro Node/npm: no se pudo compilar la interfaz."
        echo "        La API funcionara, pero la pagina web no estara disponible."
        echo "        Instala Node 18+ desde https://nodejs.org y vuelve a ejecutar."
    fi
fi

# --- 4b. Entrenar el detector la primera vez (si no hay modelo) -----
if [ ! -f "models/ai_model.json" ]; then
    echo "[*] Entrenando el detector con el corpus incluido (solo la primera vez)..."
    "$VENV_PY" train.py || echo "[AVISO] No se pudo entrenar; el detector usara solo heuristicas."
fi

# --- 5. Abrir el navegador y arrancar el servidor ------------------
URL="http://127.0.0.1:8000"
echo
echo " Iniciando Veraz en $URL"
echo " (Pulsa Ctrl+C o cierra esta ventana para detener la aplicacion)"
echo

# Abrir el navegador tras un breve retardo, en segundo plano
(
    sleep 3
    if command -v open >/dev/null 2>&1; then
        open "$URL"            # macOS
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL"        # Linux
    fi
) >/dev/null 2>&1 &

exec "$VENV_PY" run.py
