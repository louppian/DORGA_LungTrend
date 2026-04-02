@echo off
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7860 "') do taskkill /PID %%a /F >nul 2>&1
call conda activate torch_ev
python "%~dp0app.py"
pause
