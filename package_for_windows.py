#!/usr/bin/env python3
"""Assemble a self-contained `windows_build_kit/` folder.

Run on Linux (or Windows). Produces a folder you can copy to any Windows 11 PC
and run the included BUILD.bat to get an installer .exe.

    python3 package_for_windows.py

Output folder: windows_build_kit/
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KIT = ROOT / "windows_build_kit"

# Files to copy verbatim into the kit (relative to ROOT)
FILES_TO_COPY = [
    "rii_pipeline.py",
    "rii_pipeline_v2.py",
    "rii_pipeline_v3.py",
    "config.py",
    "convert_map.py",
    "build_windows.py",
    "installer.iss",
    "requirements.txt",
    "icon.png",
    "README_WINDOWS.md",
    "README.md",
]

# Directories to copy (relative to ROOT). Python cache and editor junk are
# skipped by the ignore filter below.
DIRS_TO_COPY = [
    "core",
    "gui",
    "cross_section",
    "src",
    "color_scale",
    "map",
]


_IGNORE_NAMES = {
    "__pycache__", ".DS_Store", ".pytest_cache", ".mypy_cache",
    ".vscode", ".idea", ".git", "node_modules", "build", "dist",
    "Output", "windows_build_kit",
}
_IGNORE_SUFFIXES = (".pyc", ".pyo", ".db", ".log", ".tmp")


def _ignore(_path, names):
    """shutil.copytree ignore filter — skip caches, editor junk, and large
    generated artifacts that shouldn't travel with the source kit."""
    return [
        n for n in names
        if n in _IGNORE_NAMES
        or n.endswith(_IGNORE_SUFFIXES)
    ]


def copy_tree(name: str) -> bool:
    src = ROOT / name
    if not src.exists():
        print(f"  - skip   {name} (not found)")
        return False
    dst = KIT / name
    if dst.exists():
        shutil.rmtree(dst)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=_ignore)
    else:
        shutil.copy2(src, dst)
    print(f"  + copied {name}")
    return True


def write_text(relpath: str, content: str) -> None:
    path = KIT / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    # Windows-friendly: CRLF line endings so Notepad renders correctly
    path.write_text(content.replace("\n", "\r\n"), encoding="utf-8")
    print(f"  + wrote  {relpath}")


BUILD_BAT = r"""@echo off
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
"""

INSTALL_DEPS_BAT = r"""@echo off
title RII Pipeline - Install dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)
echo.
echo Done.
pause
"""

RUN_DEV_BAT = r"""@echo off
title RII Pipeline - Run from source
cd /d "%~dp0"
python rii_pipeline.py
pause
"""

README_FIRST = """RII Pipeline - Windows Build Kit
================================

This folder contains everything needed to build and install the RII
Pipeline on Windows 10 or 11. No ROS, no Linux tools required.


QUICK START (one-time setup)
----------------------------

1. Install Python 3.10 or 3.11 (64-bit):
   https://www.python.org/downloads/windows/
   -> IMPORTANT: tick "Add Python to PATH" during install.

2. Install Inno Setup 6 (needed only for the installer step):
   https://jrsoftware.org/isdl.php
   Accept defaults.

3. Double-click BUILD.bat in this folder.
   It will:
     - install Python dependencies (numpy, PyQt5, etc.)
     - build the standalone app via PyInstaller
     - build the Windows installer (RII_Pipeline_Setup_2.0.exe)

   The first run takes 5-10 minutes. Re-runs are faster.


BUILD OUTPUT
------------

After BUILD.bat completes:

  dist\\RII_Pipeline\\RII_Pipeline.exe
     --> The portable app. Zip this folder to distribute without an
         installer. Users unzip and double-click the .exe.

  Output\\RII_Pipeline_Setup_2.0.exe
     --> The proper installer. Users double-click this, click Next a
         few times, and the app is installed to Program Files with a
         Start Menu shortcut and .riiproj file association.


OTHER BATCH FILES
-----------------

  INSTALL_DEPS.bat
     Install just the Python dependencies. Useful if you only want to
     run from source without building an installer.

  RUN_DEV.bat
     Launch the app directly from source (no build step). Requires
     INSTALL_DEPS.bat to have been run first. Good for development
     and quick smoke tests.


FULL DOCUMENTATION
------------------

See README_WINDOWS.md in this folder for:
  - detailed build troubleshooting
  - code-signing guidance (SmartScreen)
  - per-user vs. all-users install
  - GitHub Actions CI workflow
  - distribution options


FIRST-TIME SMARTSCREEN WARNING
-------------------------------

The installer is unsigned. When a user runs it for the first time,
Windows SmartScreen may show a blue screen saying:

   "Windows protected your PC"

Tell the user to click "More info" then "Run anyway". This is only
on the first download of each version. For production deployments,
obtain a code-signing certificate (~$100/year) - see README_WINDOWS.md.
"""


def main() -> int:
    if KIT.exists():
        print(f"Removing existing kit at {KIT}")
        shutil.rmtree(KIT)
    KIT.mkdir(parents=True)
    print(f"Creating Windows build kit at: {KIT}\n")

    print("Copying files…")
    for name in FILES_TO_COPY:
        copy_tree(name)
    print()

    print("Copying directories…")
    for name in DIRS_TO_COPY:
        copy_tree(name)
    print()

    print("Writing Windows helper scripts…")
    write_text("BUILD.bat", BUILD_BAT)
    write_text("INSTALL_DEPS.bat", INSTALL_DEPS_BAT)
    write_text("RUN_DEV.bat", RUN_DEV_BAT)
    write_text("README_FIRST.txt", README_FIRST)
    print()

    # Compute total size
    total = 0
    files = 0
    for root, _dirs, filenames in os.walk(KIT):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(root, f))
                files += 1
            except OSError:
                pass
    print("=" * 58)
    print(f"  Kit ready: {KIT}")
    print(f"  {files} files, {total / (1024 * 1024):.1f} MB")
    print()
    print("  Next steps:")
    print(f"    1. Copy the entire '{KIT.name}' folder to a Windows 11 PC.")
    print(f"    2. Read README_FIRST.txt inside the folder.")
    print(f"    3. Double-click BUILD.bat.")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    sys.exit(main())
