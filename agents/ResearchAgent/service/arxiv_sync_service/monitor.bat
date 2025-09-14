@echo off
REM Monitor script for Arxiv Sync Service
REM This script provides monitoring and status information

echo Monitoring Arxiv Sync Service...
echo.

cd /d "%~dp0"
..\..\run.bat monitor.py

pause
