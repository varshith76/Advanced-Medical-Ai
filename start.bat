@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo Creating virtual environment...
    py -3 -m venv .venv
)

echo Installing Python dependencies...
"%VENV_PYTHON%" -m pip install -r requirements.txt

echo.
echo Starting backend API...
start "Backend" cmd /k ""%VENV_PYTHON%" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo Starting frontend...
start "Frontend" cmd /k ""%VENV_PYTHON%" -m streamlit run frontend/app.py --server.port 8501 --server.headless true"

echo.
echo Project started.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:8501
pause
