@echo off
echo ==========================================================
echo       Tokenizer Design Lab - Developer Setup
echo ==========================================================
echo.
echo This script installs the required pre-commit hooks to ensure 
echo code formatting (black, isort, etc.) is applied locally 
echo before any commits are made.
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Find the git repository root
for /f "delims=" %%i in ('git rev-parse --show-toplevel 2^>nul') do set REPO_ROOT=%%i

if "%REPO_ROOT%"=="" (
    echo ERROR: Not in a Git repository. Please clone the repo first.
    pause
    exit /b 1
)

echo [1/3] Installing pre-commit via pip...
pip install --upgrade pre-commit --quiet

echo [2/3] Installing Git pre-commit hooks...
cd /D "%REPO_ROOT%"
pre-commit install

echo.
echo [3/3] Running hooks on all files once to ensure baseline compliance...
pre-commit run --all-files

echo.
echo ==========================================================
echo     Setup Complete! You are ready to develop and commit.
echo     The hooks will now run automatically on every 'git commit'.
echo ==========================================================
pause
