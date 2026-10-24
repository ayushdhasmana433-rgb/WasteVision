@echo off
title WasteVision
cd /d "%~dp0"
set "UV=%LOCALAPPDATA%\hermes\bin\uv.exe"
if not exist "%UV%" set "UV="
if "%UV%"=="" (where uv >nul 2>nul && set "UV=uv")

if not exist ".venv\Scripts\python.exe" (
  echo First run: setting up a private Python environment for WasteVision. This takes a few minutes, once.
  if defined UV (
    "%UV%" venv .venv --python 3.11 || goto :fail
  ) else (
    py -3.11 -m venv .venv 2>nul || python -m venv .venv || goto :fail
  )
)
".venv\Scripts\python.exe" -c "import tensorflow, keras, gradio" 2>nul || (
  echo Installing TensorFlow, Keras and Gradio ...
  if defined UV (
    "%UV%" pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :fail
  ) else (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
  )
)

echo Starting WasteVision. Your browser opens by itself when the model is ready.
echo Leave this window open while you use the app. Close it to stop.
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 goto :fail
goto :eof

:fail
echo.
echo Something went wrong. Check the error message above.
pause
