@echo off
cd /d "%~dp0"
python servidor_interfaz.py
if errorlevel 1 (
  echo.
  echo No se pudo abrir la interfaz.
  echo Instale las librerias con: python -m pip install -r requirements.txt
  pause
)
