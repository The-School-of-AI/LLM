import json
import subprocess
from pathlib import Path

import repro.env as env

# Test _pip_freeze() success case


def test_pip_freeze_success(monkeypatch):
    def mock_check_output(cmd):
        assert cmd == ["pip", "freeze"]
        return b"numpy==1.26.0\npandas==2.1.0\n"

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    result = env._pip_freeze()

    assert result == ["numpy==1.26.0", "pandas==2.1.0"]


# Test _pip_freeze() failure case


def test_pip_freeze_failure_returns_empty_list(monkeypatch):
    def mock_check_output(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    result = env._pip_freeze()

    assert result == []


# Test capture_env() creates directories and file


def test_capture_env_creates_output_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(env.platform, "platform", lambda: "TestOS")
    monkeypatch.setattr(env.platform, "python_version", lambda: "3.11.0")
    monkeypatch.setattr(env.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env.platform, "processor", lambda: "Intel")
    monkeypatch.setattr(env, "_pip_freeze", lambda: ["pkg==1.0"])

    output_path = tmp_path / "a/b/env.json"

    env.capture_env(output_path)

    assert output_path.exists()


# Test JSON content is correct and stable


def test_capture_env_writes_expected_json(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(env.platform, "platform", lambda: "Linux-Test")
    monkeypatch.setattr(env.platform, "python_version", lambda: "3.10.5")
    monkeypatch.setattr(env.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(env.platform, "processor", lambda: "ARM")
    monkeypatch.setattr(env, "_pip_freeze", lambda: ["a==1.0", "b==2.0"])

    output_path = tmp_path / "env.json"

    env.capture_env(output_path)

    data = json.loads(output_path.read_text())

    assert data == {
        "os": "Linux-Test",
        "python": "3.10.5",
        "machine": "arm64",
        "processor": "ARM",
        "pip_freeze": ["a==1.0", "b==2.0"],
    }


# Test JSON formatting (indentation)


def test_capture_env_json_is_pretty_printed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(env.platform, "platform", lambda: "OS")
    monkeypatch.setattr(env.platform, "python_version", lambda: "3.9")
    monkeypatch.setattr(env.platform, "machine", lambda: "m")
    monkeypatch.setattr(env.platform, "processor", lambda: "p")
    monkeypatch.setattr(env, "_pip_freeze", lambda: [])

    output_path = tmp_path / "env.json"
    env.capture_env(output_path)

    text = output_path.read_text()
    assert "\n  " in text  # indentation present
