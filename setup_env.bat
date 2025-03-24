@echo off
title COMP9517 CV Group Project - Windows Setup
echo.
echo 🧪 COMP9517 CV Group Project - Virtual Environment Setup (Windows)
echo --------------------------------------------------------------
echo.

REM Check for Python
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Python is not installed or not added to PATH.
    pause
    exit /b
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

IF NOT EXIST venv\Scripts\activate.bat (
    echo ❌ Failed to create virtual environment.
    pause
    exit /b
)

REM Activate and install requirements
echo Activating virtual environment and installing requirements...
call venv\Scripts\activate.bat
pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Error installing packages.
    pause
    exit /b
)

echo.
echo ✅ Setup complete!
echo To activate your environment in the future, run:
echo     venv\Scripts\activate
echo.
pause
