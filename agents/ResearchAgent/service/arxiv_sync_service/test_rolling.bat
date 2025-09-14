@echo off
REM Test script for rolling 7-day logic
REM This script tests the new rolling 7-day logic

echo Testing Rolling 7-Day Logic...
echo This will verify that the rolling window logic works correctly.
echo.

cd /d "%~dp0"
..\..\..\..\run.bat test_rolling_logic.py

pause
