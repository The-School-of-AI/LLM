import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import repro.registry as registry
from repro.manifest import RunManifest
from repro.registry import RunContext

# Test RunContext.save_output() writes JSON


def test_save_output_writes_json(tmp_path: Path):
    s3 = MagicMock()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    rc = RunContext("run1", run_dir, s3)

    rc.save_output("metrics/result.json", {"acc": 0.9})

    out = run_dir / "metrics/result.json"
    assert out.exists()

    data = json.loads(out.read_text())
    assert data == {"acc": 0.9}


# Test save_output() rejects unsupported types


def test_save_output_rejects_unsupported_type(tmp_path: Path):
    rc = RunContext("run1", tmp_path, MagicMock())

    with pytest.raises(ValueError):
        rc.save_output("x.txt", "not allowed")


# Test RunContext.finalize() uploads all files


def test_finalize_uploads_all_files(tmp_path: Path):
    s3 = MagicMock()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "a.txt").write_text("a")
    (run_dir / "sub").mkdir()
    (run_dir / "sub/b.txt").write_text("b")

    rc = RunContext("run1", run_dir, s3)
    rc.finalize()

    uploaded_keys = {call.args[1] for call in s3.upload_file.call_args_list}

    assert uploaded_keys == {"a.txt", "sub/b.txt"}


# Test start_training_run() happy path (core test)


def test_start_training_run(monkeypatch, tmp_path: Path):
    # Redirect TMP_ROOT
    monkeypatch.setattr(registry, "TMP_ROOT", tmp_path)

    # Deterministic run id
    monkeypatch.setattr(registry, "generate_training_run_id", lambda: "run123")

    # Stub external side effects
    monkeypatch.setattr(registry, "freeze_config", lambda c, p: "cfg_hash")
    monkeypatch.setattr(registry, "capture_env", lambda p: None)
    monkeypatch.setattr(registry, "capture_seeds", lambda s, p: None)

    monkeypatch.setattr(registry, "get_repo_url", lambda: "repo_url")
    monkeypatch.setattr(registry, "get_commit_hash", lambda: "commit_hash")
    monkeypatch.setattr(registry, "is_repo_dirty", lambda: False)

    mock_s3 = MagicMock()
    monkeypatch.setattr(
        registry,
        "ImmutableS3Writer",
        lambda bucket, prefix: mock_s3,
    )

    rc = registry.start_training_run(
        config={"a": 1},
        seed=42,
        bucket="test-bucket",
    )

    assert isinstance(rc, RunContext)
    assert rc.run_id == "run123"
    assert rc.run_dir.exists()
    assert rc.s3 is mock_s3

    manifest_path = rc.run_dir / "manifest.json"
    assert manifest_path.exists()

    manifest = RunManifest.model_validate_json(manifest_path.read_text())

    assert manifest.run_id == "run123"
    assert manifest.pipeline == "training"
    assert manifest.config_hash == "cfg_hash"
    assert manifest.seed == 42
    assert manifest.git.repo_url == "repo_url"
    assert manifest.git.commit_hash == "commit_hash"
    assert manifest.git.dirty is False
    assert manifest.status == "STARTED"


# Test: run directory already exists → failure


def test_start_training_run_fails_if_run_dir_exists(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(registry, "TMP_ROOT", tmp_path)
    monkeypatch.setattr(registry, "generate_training_run_id", lambda: "run123")

    existing = tmp_path / "runs" / "run123"
    existing.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        registry.start_training_run(config={}, seed=1)
