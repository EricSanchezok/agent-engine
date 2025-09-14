@echo off
REM Run Arxiv Sync Service Once
REM This script runs the Arxiv sync service once for testing

echo Running Arxiv Sync Service once...
echo This will sync the current week's papers and then exit.
echo.

cd /d "%~dp0"
..\..\run.bat run_once.py

pause
