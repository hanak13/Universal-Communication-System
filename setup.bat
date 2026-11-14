@echo off
REM ── UCS Setup ──────────────────────────────────────────────────
REM Run this from any terminal:  setup.bat
REM Creates the venv, installs deps, and runs the test suite.

cd /d "%~dp0"
echo === UCS Setup ===
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating Python 3.12 venv...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create venv. Is Python 3.12 on PATH?
        exit /b 1
    )
) else (
    echo [1/3] Venv already exists.
)

echo [2/3] Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip -q
.venv\Scripts\python.exe -m pip install -r requirements.txt -q

echo [3/3] Running tests...
.venv\Scripts\python.exe -m pytest tests/ -v
if errorlevel 1 (
    echo.
    echo TESTS FAILED
    exit /b 1
)

echo.
echo ===================================================
echo   All good! Run the app with:
echo   .venv\Scripts\streamlit.exe run app.py
echo ===================================================
