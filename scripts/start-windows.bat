@echo off

call git pull --recurse-submodules

set VENVDIR=%CD%\venv
set PYTHON=%VENVDIR%\Scripts\Python.exe
set ACCELERATE=%VENVDIR%\Scripts\accelerate.exe

set HUGGINGFACE_TOKEN=
set DISCORD_BOT_TOKEN=
set LOG_LEVEL=INFO

call %VENVDIR%\Scripts\activate

start http://localhost:5003/

if EXIST %ACCELERATE% (
  %ACCELERATE% launch main.py
) ELSE (
  %PYTHON% main.py
)

pause
exit /b