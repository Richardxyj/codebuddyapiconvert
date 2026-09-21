@echo off
setlocal
title CodeBuddy API Convert - stop
set "FOUND=0"

for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8787" ^| findstr "LISTENING"') do (
  taskkill /PID %%p /F >nul 2>&1
  set "FOUND=1"
)

if "%FOUND%"=="1" (
  echo CodeBuddy API Convert stopped.
) else (
  echo CodeBuddy API Convert is not running.
)
timeout /t 5 >nul
