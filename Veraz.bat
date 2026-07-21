@echo off
REM ====================================================================
REM  Veraz - Lanzador para Windows
REM  Doble clic para iniciar la aplicacion. Prepara el entorno la
REM  primera vez (entorno virtual, dependencias e interfaz) y abre el
REM  navegador en http://127.0.0.1:8000
REM ====================================================================
setlocal enabledelayedexpansion

REM Al hacer doble clic, Windows ejecuta los .bat en una ventana que puede
REM cerrarse al terminar. Relanzamos una vez con cmd /k para que cualquier
REM error quede visible y el usuario pueda copiarlo.
if /i not "%~1"=="--inner" (
    start "Veraz" cmd /k ""%~f0" --inner"
    exit /b
)

cd /d "%~dp0"
set "VERAZ_LOG=%~dp0veraz-arranque.log"
echo [%date% %time%] Iniciando Veraz > "%VERAZ_LOG%"

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
    echo [ERROR] No se encontro Python. Instala Python 3.8 o superior desde
    echo         https://www.python.org/downloads/ y marca "Add to PATH".
    echo         (Windows 8 o Windows 10 antiguo: usa Python 3.8.10, que si los soporta).
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

REM Verificar version de Python dentro del entorno virtual (FastAPI requiere 3.8+).
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)"
if errorlevel 1 (
    echo [ERROR] Veraz necesita Python 3.8 o superior.
    echo         Borra la carpeta backend\.venv despues de instalar una version compatible
    echo         y vuelve a ejecutar este archivo.
    pause
    exit /b 1
)

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
    "%VENV_PY%" -c "import fastapi, uvicorn, pydantic, multipart, pypdf, docx, fpdf"
    if errorlevel 1 (
        echo [ERROR] Las dependencias se instalaron, pero no se pueden importar.
        echo         Borra backend\.venv y vuelve a ejecutar Veraz.bat.
        pause
        exit /b 1
    )
    echo ok> ".venv\.deps_ok"
)

REM Si el marcador existe pero el entorno quedo incompleto, reinstalar.
"%VENV_PY%" -c "import fastapi, uvicorn, pydantic, multipart, pypdf, docx, fpdf" >nul 2>nul
if errorlevel 1 (
    echo [2/3] Reparando dependencias incompletas...
    del /q ".venv\.deps_ok" >nul 2>nul
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] No se pudieron reparar las dependencias.
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
    if errorlevel 1 (
        echo [AVISO] No se pudo entrenar; Veraz usara solo heuristicas.
    )
)

REM --- 5. Iniciar el servidor y abrir el navegador -------------------
echo.
echo  Iniciando Veraz en http://127.0.0.1:8000
echo  (Cierra esta ventana para detener la aplicacion)
echo.
start "" http://127.0.0.1:8000
"%VENV_PY%" run.py

echo.
echo El servidor se detuvo. Esta ventana queda abierta para mostrar errores.
echo Log guardado en: "%VERAZ_LOG%"
echo Si necesitas cerrar, escribe exit y pulsa Enter.
endlocal
