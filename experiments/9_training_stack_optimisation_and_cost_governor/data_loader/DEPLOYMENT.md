# SPDL DataLoader Project - Dependency Management
This project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible Python dependency management and virtual environments.

This project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible Python dependency management and virtual environments.

**Required dependencies:**
- spdl, spdl_core, spdl_io, spdl-dataloader, pyarrow, numpy, torch, typing_extensions, pyyaml

1. **Install uv (if not already installed):**
   ```sh
   pip install uv
   # or
   brew install uv
   ```

2. **Create and activate the virtual environment:**
   ```sh
   cd data_loader
   uv venv
   source .venv/bin/activate
   uv venv
   source .venv/bin/activate
3. **Install all dependencies:**
   ```sh
   uv pip sync requirements.uv.txt
   ```

4. **Run tests:**
   ```sh
   bash run_test.sh
   ```

## Notes
- All dependencies are listed in `requirements.uv.txt` (no requirements.txt).
- The `.venv` directory is used for the local environment.
- If you add or update dependencies, update `requirements.uv.txt` and re-run `uv pip sync requirements.uv.txt`.
- For more info, see: https://github.com/astral-sh/uv
