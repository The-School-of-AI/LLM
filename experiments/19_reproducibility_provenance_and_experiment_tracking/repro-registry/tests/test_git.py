import subprocess

import pytest
from repro.git import _git, get_commit_hash, get_repo_url, is_repo_dirty

# Test _git() happy path


def test_git_executes_command_and_returns_string(monkeypatch):
    def mock_check_output(cmd, stderr):
        assert cmd == ["git", "rev-parse", "HEAD"]
        return b"abc123\n"

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    result = _git("git rev-parse HEAD")
    assert result == "abc123"


# Test get_commit_hash()


def test_get_commit_hash(monkeypatch):
    monkeypatch.setattr("repro.git._git", lambda cmd: "deadbeef1234567890")

    assert get_commit_hash() == "deadbeef1234567890"


# Test is_repo_dirty() → clean repo


def test_is_repo_dirty_false_when_clean(monkeypatch):
    monkeypatch.setattr("repro.git._git", lambda cmd: "")

    assert is_repo_dirty() is False


# Test is_repo_dirty() → dirty repo


def test_is_repo_dirty_true_when_dirty(monkeypatch):
    monkeypatch.setattr("repro.git._git", lambda cmd: " M repro/git.py")

    assert is_repo_dirty() is True


# Test get_repo_url()


def test_get_repo_url(monkeypatch):
    monkeypatch.setattr("repro.git._git", lambda cmd: "https://github.com/org/repo.git")

    assert get_repo_url() == "https://github.com/org/repo.git"


# (Optional but strong) Test Git failure propagates


def test_git_command_failure_raises(monkeypatch):
    def mock_check_output(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    with pytest.raises(subprocess.CalledProcessError):
        _git("git rev-parse HEAD")
