# test_run_id.py
# Basic tests: prefix & structure

import re
import uuid
from datetime import datetime, timezone

from repro.ids import generate_coreset_run_id, generate_run_id, generate_training_run_id

RUN_ID_PATTERN = re.compile(r"^(?P<prefix>\w+)_(\d{8})_(\d{6})_(?P<rand>[0-9a-f]{6})$")


def test_generate_run_id_structure():
    run_id = generate_run_id("test")

    match = RUN_ID_PATTERN.match(run_id)
    assert match is not None
    assert match.group("prefix") == "test"


# Test convenience wrappers


def test_generate_training_run_id_prefix():
    run_id = generate_training_run_id()
    assert run_id.startswith("run_")


def test_generate_coreset_run_id_prefix():
    run_id = generate_coreset_run_id()
    assert run_id.startswith("coreset_")


# Validate UUID suffix length & charset


def test_uuid_suffix_is_hex_and_correct_length():
    run_id = generate_run_id("run")
    suffix = run_id.split("_")[-1]

    assert len(suffix) == 6
    assert all(c in "0123456789abcdef" for c in suffix)


# Uniqueness sanity test (non-deterministic but useful)


def test_run_id_uniqueness():
    ids = {generate_run_id("run") for _ in range(1000)}
    assert len(ids) == 1000


# Best practice: deterministic test using monkeypatch


def test_generate_run_id_deterministic(monkeypatch):
    fixed_time = datetime(2026, 2, 8, 10, 30, 0, tzinfo=timezone.utc)
    fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")

    monkeypatch.setattr(
        "repro.ids.datetime",
        type(
            "MockDateTime",
            (),
            {"now": lambda *_: fixed_time},
        ),
    )

    monkeypatch.setattr(
        "repro.ids.uuid.uuid4",
        lambda: fixed_uuid,
    )

    run_id = generate_run_id("run")

    assert run_id == "run_20260208_103000_123456"
