@echo off
title PhishDec Multi-Agent Platform Launcher
echo ========================================================
echo         PHISHDEC MULTI-AGENT CYBERSECURITY PLATFORM     
echo ========================================================
echo.
echo [1/2] Starting Flask Backend API on port 5050...
start "PhishDec Backend API (:5050)" cmd /k "cd /d %~dp0 && python api/app.py"

echo [2/2] Starting Vite Frontend on port 5173...
start "PhishDec React UI (:5173)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both servers are starting up!
echo Frontend will be accessible at: http://localhost:5173/
echo Backend API will be accessible at: http://127.0.0.1:5050/
echo.
echo Opening browser in 3 seconds...
timeout /t 3 >nul
start http://localhost:5173/
pause
