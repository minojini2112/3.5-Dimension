@echo off
REM One-shot setup for Windows + NVIDIA GPU (e.g. RTX 2050 4 GB). Run from the project folder.
REM Needs Python 3.10, 3.11 or 3.12 (PyTorch 2.4 has no wheels for 3.13+). Uses the "py" launcher to find one.
cd /d "%~dp0"
set PY=
for %%v in (3.11 3.12 3.10) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set PY=py -%%v
  )
)
if not defined PY (
  echo.
  echo ERROR: Python 3.11 not found. Install it from https://www.python.org/downloads/release/python-3119/
  echo        "Windows installer (64-bit)", then run this script again. Your other Python can stay installed.
  exit /b 1
)
echo Using %PY%
if exist .venv (
  .venv\Scripts\python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
  if errorlevel 1 (
    echo Removing old .venv made with an unsupported Python version...
    rmdir /s /q .venv
  )
)
if not exist .venv %PY% -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 || goto :err
pip install -r requirements.txt || goto :err
REM Fetches TripoSR source + all model weights (~6 GB). app.py would also do this on first start.
python download_weights.py || goto :err
python -c "import torch;print('CUDA available:', torch.cuda.is_available())"
echo.
echo Done. Double-click start_app.bat to run.
goto :eof
:err
echo Setup failed, see the message above.
exit /b 1
