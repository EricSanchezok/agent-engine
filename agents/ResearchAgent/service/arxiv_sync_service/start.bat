@echo off
REM Start script for Arxiv Sync Service
REM This script starts the continuous Arxiv sync service

echo Starting Arxiv Sync Service...
echo This service will run continuously and sync arXiv papers every 15 minutes.
echo Press Ctrl+C to stop the service.
echo.

cd /d "%~dp0"
..\..\run.bat start_service.py

pause
