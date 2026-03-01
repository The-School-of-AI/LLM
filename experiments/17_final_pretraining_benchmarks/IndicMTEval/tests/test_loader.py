import json
import pytest
from pathlib import Path
from benchmark_indic_mt_eval.data.loader import (
    MTSample,
    load_language_data,
    load_benchmark_data,
    GITHUB_RAW_BASE,
)


SAMPLE_JSONL_ROW = {
    "src": "The cat sat on the mat.",
    "ref": "बिल्ली चटाई पर बैठी।",
    "translation": "बिल्ली मैट पर बैठ गई।",
    "mqm_norm_score": "0.76",
    "da_norm_score": "0.72",
    "adequacy_score": "19",
    "fluency_score": "25",
    "full_score": "19",
    "completion": [],
    "prompt": "...",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_mt_sample_fields():
    s = MTSample(
        source="hello",
        hypothesis="नमस्ते",
        reference="नमस्कार",
        human_score=0.8,
        language="hi",
        adequacy_score=20.0,
        fluency_score=22.0,
        full_score=17.0,
    )
    assert s.source == "hello"
    assert s.human_score == 0.8


def test_load_language_data_from_local(tmp_path):
    rows = [SAMPLE_JSONL_ROW] * 5
    _write_jsonl(tmp_path / "Hin_test.jsonl", rows)
    samples = load_language_data("hi", "test", data_dir=str(tmp_path))
    assert len(samples) == 5
    assert samples[0].source == "The cat sat on the mat."
    assert samples[0].hypothesis == "बिल्ली मैट पर बैठ गई।"
    assert samples[0].human_score == 0.76
    assert samples[0].language == "hi"


def test_load_language_data_with_limit(tmp_path):
    rows = [SAMPLE_JSONL_ROW] * 20
    _write_jsonl(tmp_path / "Hin_test.jsonl", rows)
    samples = load_language_data("hi", "test", data_dir=str(tmp_path), max_samples=5)
    assert len(samples) == 5


def test_load_benchmark_data_multiple_langs(tmp_path):
    for prefix, code in [("Hin", "hi"), ("Tam", "ta")]:
        _write_jsonl(tmp_path / f"{prefix}_test.jsonl", [SAMPLE_JSONL_ROW] * 3)
    data = load_benchmark_data(
        languages=["hi", "ta"], split="test", data_dir=str(tmp_path)
    )
    assert set(data.keys()) == {"hi", "ta"}
    assert len(data["hi"]) == 3
    assert len(data["ta"]) == 3


def test_load_language_data_invalid_lang():
    with pytest.raises(ValueError, match="Unknown language"):
        load_language_data("xx", "test")
