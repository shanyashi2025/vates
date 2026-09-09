@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "VENV_STREAM=%~dp0.venv\Scripts\streamlit.exe"

REM Create the project venv from the system Python if it does not exist
if not exist "%VENV_PY%" (
    echo Setting up virtual environment...
    python -m venv ".venv"
    if errorlevel 1 (
        echo [ERROR] Could not create virtual environment. Make sure Python is installed and on PATH.
        pause
        exit /b 1
    )
)

REM Install dependencies if streamlit is not present in the venv
if not exist "%VENV_STREAM%" (
    echo Installing dependencies...
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Could not install dependencies. Check your internet connection.
        pause
        exit /b 1
    )
)

echo Starting app...
"%VENV_STREAM%" run app.py
if errorlevel 1 (
    echo [ERROR] Streamlit exited with an error.
    pause
)
endlocal