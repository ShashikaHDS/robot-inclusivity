@echo off
REM ──────────────────────────────────────────────────────────────────────
REM Robot Inclusivity Index (RII) — Windows Launch Script
REM
REM Usage:
REM   launch.bat                          Launch the GUI
REM   launch.bat convert --in map.pcd     Convert 3D PCD to 2D Nav2 map
REM   launch.bat analyze --in map.pcd     Analyze Z distribution
REM ──────────────────────────────────────────────────────────────────────

cd /d "%~dp0"

if "%1"=="convert" (
    shift
    python convert_map.py %*
) else if "%1"=="analyze" (
    shift
    python convert_map.py --analyze-z %*
) else (
    python rii_pipeline.py %*
)
