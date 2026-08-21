@echo off
REM Brain Hemorrhage Detection System - Windows Setup Script
REM This script will install Python and all required dependencies

setlocal enabledelayedexpansion

echo.
echo ================================================================================
echo Brain Hemorrhage Detection System - Setup
echo ================================================================================
echo.

REM Check if Python is installed
echo Checking for Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please download and install Python from:
    echo https://www.python.org/downloads/
    echo.
    echo During installation, make sure to:
    echo 1. Check "Add Python to PATH"
    echo 2. Choose "Install Now" or customize installation
    echo 3. Click "Disable path length limit" at the end
    echo.
    echo After installing Python, run this script again.
    pause
    exit /b 1
)

echo ✓ Python found!
for /f "tokens=*" %%i in ('python --version') do echo   Version: %%i
echo.

REM Upgrade pip
echo Installing/Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)
echo ✓ Pip upgraded successfully!
echo.

REM Install requirements
echo Installing dependencies (this may take a few minutes)...
echo This includes: TensorFlow, OpenCV, Flask, and more...
echo.
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)
echo ✓ Dependencies installed successfully!
echo.

REM Create model
echo Creating pre-trained model...
python model/train_model.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to create model
    pause
    exit /b 1
)
echo ✓ Model created successfully!
echo.

REM Success
echo ================================================================================
echo Setup Complete!
echo ================================================================================
echo.
echo You can now run the application with:
echo   python app.py
echo.
echo Then open http://localhost:5000 in your web browser
echo.
pause
