from datetime import datetime, timezone

import pytest
from repro.manifest import GitInfo, RunManifest

# Test GitInfo creation (happy path)


def test_gitinfo_creation():
    git = GitInfo(
        repo_url="https://github.com/org/repo.git",
        commit_hash="abc123",
        dirty=False,
    )

    assert git.repo_url == "https://github.com/org/repo.git"
    assert git.commit_hash == "abc123"
    assert git.dirty is False


# Test RunManifest creation with nested GitInfo


def test_runmanifest_creation():
    git = GitInfo(
        repo_url="https://github.com/org/repo.git",
        commit_hash="deadbeef",
        dirty=True,
    )

    manifest = RunManifest(
        run_id="run_001",
        pipeline="training",
        created_at=datetime.now(timezone.utc),
        git=git,
        config_hash="cfg123",
        seed=42,
        status="STARTED",
    )

    assert manifest.run_id == "run_001"
    assert manifest.pipeline == "training"
    assert manifest.git.commit_hash == "deadbeef"
    assert manifest.seed == 42
    assert manifest.coreset_run_id is None


# Test nested dict → GitInfo auto-parsing


def test_runmanifest_accepts_git_as_dict():
    manifest = RunManifest(
        run_id="run_002",
        pipeline="eval",
        created_at=datetime.now(timezone.utc),
        git={
            "repo_url": "https://example.com/repo.git",
            "commit_hash": "abc",
            "dirty": False,
        },
        config_hash="cfg456",
        seed=1,
        status="COMPLETED",
    )

    assert isinstance(manifest.git, GitInfo)
    assert manifest.git.dirty is False


# Test datetime parsing from ISO string


def test_created_at_accepts_iso_string():
    manifest = RunManifest(
        run_id="run_003",
        pipeline="pipeline",
        created_at="2026-02-08T10:30:00Z",
        git={
            "repo_url": "url",
            "commit_hash": "hash",
            "dirty": False,
        },
        config_hash="cfg",
        seed=0,
        status="FAILED",
    )

    assert isinstance(manifest.created_at, datetime)


# Test optional coreset_run_id


def test_coreset_run_id_optional():
    manifest = RunManifest(
        run_id="run_004",
        pipeline="pipeline",
        created_at=datetime.now(timezone.utc),
        git={
            "repo_url": "url",
            "commit_hash": "hash",
            "dirty": False,
        },
        config_hash="cfg",
        seed=10,
        coreset_run_id="coreset_001",
        status="STARTED",
    )

    assert manifest.coreset_run_id == "coreset_001"


# Test missing required fields → validation error


def test_runmanifest_missing_required_field_raises():
    with pytest.raises(Exception):
        RunManifest(
            run_id="run_005",
            pipeline="pipeline",
            # created_at missing
            git={
                "repo_url": "url",
                "commit_hash": "hash",
                "dirty": False,
            },
            config_hash="cfg",
            seed=1,
            status="STARTED",
        )


# Test wrong type → validation error


def test_runmanifest_invalid_seed_type_raises():
    with pytest.raises(Exception):
        RunManifest(
            run_id="run_006",
            pipeline="pipeline",
            created_at=datetime.now(timezone.utc),
            git={
                "repo_url": "url",
                "commit_hash": "hash",
                "dirty": False,
            },
            config_hash="cfg",
            seed="not-an-int",
            status="STARTED",
        )


# Test serialization round-trip (important for manifests)


def test_runmanifest_json_roundtrip():
    manifest = RunManifest(
        run_id="run_007",
        pipeline="train",
        created_at=datetime.now(timezone.utc),
        git={
            "repo_url": "url",
            "commit_hash": "hash",
            "dirty": False,
        },
        config_hash="cfg",
        seed=123,
        status="COMPLETED",
    )

    data = manifest.model_dump()
    restored = RunManifest(**data)

    assert restored == manifest
