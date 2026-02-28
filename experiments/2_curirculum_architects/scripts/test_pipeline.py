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
        print(f"\n\033[0;31m[!] Command failed: {command}\033[0m")
        sys.exit(1)


def find_project_root():
    """Find the project root looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    # check if current is root (e.g. script in root)
    if (current / "pyproject.toml").exists():
        return current
    # check parent (e.g. script in scripts/)
    if (current.parent / "pyproject.toml").exists():
        return current.parent
    # recursive search up
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find pyproject.toml in any parent directory")


def main():
    print("\033[0;32m=== Starting Pipeline Verification ===\033[0m")

    # 0. Set Working Directory to Project Root
    try:
        project_root = find_project_root()
        print(f"Project root detected: {project_root}")
        import os

        os.chdir(project_root)
    except Exception as e:
        print(f"\033[0;31m✗ Error: {e}\033[0m")
        sys.exit(1)

    # 1. Check/Create Environment
    print_step(1, "Checking Python environment...")

    # Check if we are in a venv
    in_venv = sys.prefix != sys.base_prefix

    if in_venv:
        print("Using existing virtual environment.")
        python_exe = sys.executable
        # pip_exe = "pip"
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
            # pip_exe = str(venv_path / "Scripts" / "pip.exe")
        else:
            python_exe = str(venv_path / "bin" / "python")
            # pip_exe = str(venv_path / "bin" / "pip")

    # 2. Install Package
    print_step(2, "Installing package in editable mode...")
    run_command(f"{python_exe} -m pip install --upgrade pip")
    # Install with dev dependencies using the specific python executable
    run_command(f'{python_exe} -m pip install -e ".[dev]"')
    # Fallback to manual install if [dev] isn't set up
    run_command(f"{python_exe} -m pip install pytest pyarrow pyyaml pandas")

    # 3. Run Unit Tests
    print_step(3, "Running Unit Tests...")
    run_command(f"{python_exe} -m pytest tests/ -v")
    print("\033[0;32m[OK] All tests passed\033[0m")

    # 4. Run Smoke Test
    print_step(4, "Running Smoke Test (basic_usage.py)...")
    run_command(f"{python_exe} examples/basic_usage.py")
    print("\033[0;32m[OK] Smoke test passed\033[0m")

    print("\n\033[0;32m=== [DONE] PIPELINE VERIFIED SUCCESSFULLY ===\033[0m")
    print("You are ready to push to git.")


if __name__ == "__main__":
    main()
