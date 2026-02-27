"""Test that the public API is importable."""


def test_opus_public_imports():
    from llm.opus import (
        OpusConfig,
        OpusDataSelector,
        OpusSelector,
        SelectionResult,
        GhostCollector,
        CountSketchProjector,
        AdamWPreconditionerView,
    )

    assert OpusConfig is not None
    assert OpusDataSelector is not None
    assert OpusSelector is not None
    assert SelectionResult is not None
    assert GhostCollector is not None
    assert CountSketchProjector is not None
    assert AdamWPreconditionerView is not None
