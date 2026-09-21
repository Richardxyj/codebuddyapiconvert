@echo off
setlocal
title CodeBuddy API Convert (127.0.0.1:8787)

set "ROOT=%~dp0"
set "PYTHON=%PYTHON%"

if not defined PYTHON (
  where python >nul 2>&1
  if errorlevel 1 (
    echo Python was not found in PATH. Set PYTHON to the interpreter to use.
    pause
    exit /b 1
  )
  set "PYTHON=python"
)

netstat -ano | findstr ":8787" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo CodeBuddy API Convert is already running on port 8787.
  echo Health check: http://127.0.0.1:8787/health
  timeout /t 8 >nul
  exit /b 0
)

cd /d "%ROOT%"
echo Starting CodeBuddy API Convert on http://127.0.0.1:8787 ...
echo Keep this window open while using the service. Press Ctrl+C to stop.
echo.
"%PYTHON%" -m core.converter --desensitize --log converter.log

echo.
echo Converter exited. Check converter.log for details.
pause
