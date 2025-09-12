@echo off
echo 🎯 Agent Engine Package Builder
echo ================================

REM Check if pyproject.toml exists
if not exist "pyproject.toml" (
    echo ❌ pyproject.toml not found. Please run this script from the project root.
    pause
    exit /b 1
)

REM Clean build directories
echo.
echo 🧹 Cleaning build directories...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "agent_engine.egg-info" rmdir /s /q "agent_engine.egg-info"

REM Sync dependencies
echo.
echo 📦 Syncing dependencies...
uv sync --extra opts
if errorlevel 1 (
    echo ❌ Failed to sync dependencies
    pause
    exit /b 1
)

REM Build package
echo.
echo 🔨 Building package...
uv build
if errorlevel 1 (
    echo ❌ Package build failed
    pause
    exit /b 1
)

REM Check package
echo.
echo 🔍 Checking package...
twine check dist/*
if errorlevel 1 (
    echo ❌ Package check failed
    pause
    exit /b 1
)

echo.
echo ✅ Package built successfully!
echo.
echo 📋 Package files in dist/:
dir dist

echo.
echo 📋 Next steps:
echo 1. Test the package locally:
echo    pip install dist/agent_engine-*.whl
echo 2. Upload to Test PyPI:
echo    twine upload --repository testpypi dist/*
echo 3. Upload to PyPI:
echo    twine upload dist/*

pause
