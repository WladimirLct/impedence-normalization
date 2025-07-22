@echo off
echo Starting YESPRO-35 application...
powershell -ExecutionPolicy Bypass -Command "uv run app/main.py"
pause