@echo off
REM Double-click to start Sketch2Shape (run setup_windows.bat once first).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" http://127.0.0.1:7860
python app.py
pause
