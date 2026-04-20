@echo off
setlocal enabledelayedexpansion
title RII Pipeline - Build installer
echo.
echo =========================================================
echo   RII Pipeline - Windows installer builder
echo =========================================================
echo.

:: Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo Tick "Add Python to PATH" during install, then re-run BUILD.bat.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo Using %%v

:: Step 1 - install Python dependencies
echo.
echo [1/3] Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)

:: Step 2 - build the standalone app folder via PyInstaller
echo.
echo [2/3] Building standalone application folder...
python build_windows.py
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed.
    pause
    exit /b 1
)

:: Step 3 - build the installer via Inno Setup (optional but recommended)
echo.
echo [3/3] Building Windows installer (Inno Setup)...
set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
    echo [WARN]  Inno Setup 6 not found. Skipping installer build.
    echo         Install from https://jrsoftware.org/isdl.php and re-run.
    echo         The portable app folder is still available at dist\RII_Pipeline\
    echo         You can zip and distribute that folder as-is.
    pause
    exit /b 0
)
"%ISCC%" installer.iss
if errorlevel 1 (
    echo [ERROR] Inno Setup compilation failed.
    pause
    exit /b 1
)

echo.
echo =========================================================
echo   Build complete!
echo =========================================================
echo   Portable folder:  dist\RII_Pipeline\
echo   Installer .exe:   Output\RII_Pipeline_Setup_2.0.exe
echo.
echo   Double-click the installer to install the app, or zip
echo   the dist folder for a no-install portable distribution.
echo =========================================================
pause
