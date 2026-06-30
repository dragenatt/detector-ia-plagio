@echo off
REM ====================================================================
REM  Veraz - Lanzador para Windows
REM  Doble clic para iniciar la aplicacion. Prepara el entorno la
REM  primera vez (entorno virtual, dependencias e interfaz) y abre el
REM  navegador en http://127.0.0.1:8000
REM ====================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ========================================
echo    Veraz - Detector de IA y plagio
echo  ========================================
echo.

REM --- 1. Buscar Python -------------------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [ERROR] No se encontro Python. Instala Python 3.10+ desde
    echo         https://www.python.org/downloads/ y marca "Add to PATH".
    pause
    exit /b 1
)

cd backend

REM --- 2. Crear el entorno virtual la primera vez ----------------------
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creando entorno virtual...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)
set "VENV_PY=.venv\Scripts\python.exe"

REM --- 3. Instalar dependencias (solo la primera vez) -----------------
if not exist ".venv\.deps_ok" (
    echo [2/3] Instalando dependencias (puede tardar un momento)...
    "%VENV_PY%" -m pip install --upgrade pip >nul
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Fallo la instalacion de dependencias.
        pause
        exit /b 1
    )
    echo ok> ".venv\.deps_ok"
)

REM --- 4. Compilar la interfaz si no existe frontend\dist -------------
if not exist "..\frontend\dist\index.html" (
    where npm >nul 2>nul
    if errorlevel 1 (
        echo [AVISO] No se encontro Node/npm: no se pudo compilar la interfaz.
        echo         La API funcionara, pero la pagina web no estara disponible.
        echo         Instala Node 18+ desde https://nodejs.org y vuelve a ejecutar.
    ) else (
        echo [3/3] Compilando la interfaz (solo la primera vez)...
        pushd ..\frontend
        call npm install
        call npm run build
        popd
    )
)

REM --- 4b. Entrenar el detector la primera vez (si no hay modelo) ----
if not exist "models\ai_model.json" (
    echo [*] Entrenando el detector con el corpus incluido (solo la primera vez)...
    "%VENV_PY%" train.py
)

REM --- 5. Iniciar el servidor y abrir el navegador -------------------
echo.
echo  Iniciando Veraz en http://127.0.0.1:8000
echo  (Cierra esta ventana para detener la aplicacion)
echo.
start "" http://127.0.0.1:8000
"%VENV_PY%" run.py

endlocal
