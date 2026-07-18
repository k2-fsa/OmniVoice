@echo off
cd /d "%~dp0"
.venv\Scripts\python -m omnivoice.cli.demo --ip 127.0.0.1 --port 8001
pause
