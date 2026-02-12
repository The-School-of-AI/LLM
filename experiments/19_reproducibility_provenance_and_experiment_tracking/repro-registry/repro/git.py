# repro/git.py
import subprocess


def _git(cmd: str) -> str:
    return (
        subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL).decode().strip()
    )


def get_commit_hash() -> str:
    return _git("git rev-parse HEAD")


def is_repo_dirty() -> bool:
    return bool(_git("git status --porcelain"))


def get_repo_url() -> str:
    return _git("git config --get remote.origin.url")
