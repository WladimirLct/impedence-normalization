@echo off
echo Starting YESPRO-35 application...
cd "../" 
@REM \ Change to the root of the project so the user can see tmp/ and normalizations/ folders
powershell -ExecutionPolicy Bypass -Command "uv run app/main.py"
pause