
"""Pipeline verification script."""

import subprocess
import sys
import venv
from pathlib import Path

def print_step(step_num, message):
    print(f"\n\033[0;32m[{step_num}/4] {message}\033[0m")

def run_command(command, cwd=None, shell=True):
    try:
        subprocess.check_call(command, cwd=cwd, shell=shell)
    except subprocess.CalledProcessError:
        print(f"\n\033[0;31m✗ Command failed: {command}\033[0m")
        sys.exit(1)

def main():
    print("\033[0;32m=== Starting Pipeline Verification ===\033[0m")
    
    # 1. Check/Create Environment
    print_step(1, "Checking Python environment...")
    
    # Check if we are in a venv
    in_venv = sys.prefix != sys.base_prefix
    
    if in_venv:
        print("Using existing virtual environment.")
        python_exe = sys.executable
        pip_exe = "pip"
    else:
        # Check for local .venv directory
        venv_path = Path(".venv")
        if not venv_path.exists():
            print("No .venv found. Creating one...")
            venv.create(venv_path, with_pip=True)
            
        print("Using local .venv...")
        # Determine executable paths based on OS
        if sys.platform == "win32":
            python_exe = str(venv_path / "Scripts" / "python.exe")
            pip_exe = str(venv_path / "Scripts" / "pip.exe")
        else:
            python_exe = str(venv_path / "bin" / "python")
            pip_exe = str(venv_path / "bin" / "pip")

    # 2. Install Package
    print_step(2, "Installing package in editable mode...")
    run_command(f"{python_exe} -m pip install --upgrade pip")
    # Install with dev dependencies using the specific python executable
    run_command(f"{python_exe} -m pip install -e \".[dev]\"")
    # Fallback to manual install if [dev] isn't set up
    run_command(f"{python_exe} -m pip install pytest pyarrow pyyaml pandas")

    # 3. Run Unit Tests
    print_step(3, "Running Unit Tests...")
    run_command(f"{python_exe} -m pytest tests/ -v")
    print("\033[0;32m✓ All tests passed\033[0m")

    # 4. Run Smoke Test
    print_step(4, "Running Smoke Test (basic_usage.py)...")
    run_command(f"{python_exe} examples/basic_usage.py")
    print("\033[0;32m✓ Smoke test passed\033[0m")

    print("\n\033[0;32m=== ✅ PIPELINE VERIFIED SUCCESSFULLY ===\033[0m")
    print("You are ready to push to git.")

if __name__ == "__main__":
    main()
