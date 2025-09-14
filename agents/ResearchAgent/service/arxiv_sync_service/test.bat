@echo off
REM Test script for Arxiv Sync Service
REM This script tests the service configuration and basic functionality

echo Testing Arxiv Sync Service...
echo This will validate configuration and test basic functionality.
echo.

cd /d "%~dp0"
..\..\run.bat test_service.py

pause
