"""Shared utilities for the contamination scanner."""

import subprocess
from pathlib import Path


def get_git_info() -> dict[str, str]:
    """Return the current git commit hash and repository cleanliness.

    Runs ``git rev-parse HEAD`` and ``git status --porcelain`` relative to
    this file's location, so it always refers to the scanner repo regardless
    of the caller's working directory.

    Returns:
        Dict with two keys:

        - ``commit`` — full 40-character SHA, or ``"unknown"`` if git is
          unavailable or the project is not inside a repository.
        - ``dirty`` — ``"true"`` if there are uncommitted changes (staged
          or unstaged), ``"false"`` otherwise, or ``"unknown"`` on error.
    """
    repo_dir = str(__file__)
    # Run git commands from the module's directory, not the file path itself.
    repo_cwd = Path(repo_dir).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        return {"commit": commit, "dirty": str(bool(dirty_output)).lower()}
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return {"commit": "unknown", "dirty": "unknown"}


def normalize(text: str) -> str:
    """Normalize text for consistent comparison across the pipeline.

    Lowercases, strips leading/trailing whitespace, and collapses
    internal whitespace (tabs, newlines, multiple spaces) to single spaces.

    Args:
        text: Raw input text of any type (will be coerced to str).

    Returns:
        Normalized text string.
    """
    return " ".join(str(text).lower().strip().split())
