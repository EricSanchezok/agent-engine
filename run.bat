@echo off
setlocal enabledelayedexpansion

:: Set the absolute path of the project root directory (auto-detect)
set PROJECT_ROOT=%~dp0
set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

:: Set PYTHONPATH environment variable
set PYTHONPATH=%PROJECT_ROOT%

:: Check if parameters are provided
if "%~1"=="" (
    echo Usage: run.bat ^<python_file_path^>
    echo Example: run.bat test\test.py
    exit /b 1
)

:: Build the complete file path
set SCRIPT_PATH=%PROJECT_ROOT%\%~1

:: Check if the file exists
if not exist "%SCRIPT_PATH%" (
    echo Error: File does not exist - %SCRIPT_PATH%
    echo Please check if the file path is correct
    exit /b 1
)

:: Display run information
echo ========================================
echo Project Root: %PROJECT_ROOT%
echo PYTHONPATH: %PYTHONPATH%
echo Running Script: %SCRIPT_PATH%
echo ========================================
echo.

:: Run Python script
uv run "%SCRIPT_PATH%"

:: Check run result
if %ERRORLEVEL% neq 0 (
    echo.
    echo Script execution failed, error code: %ERRORLEVEL%
) else (
    echo.
    echo Script execution completed
)

endlocal