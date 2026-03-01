"""End-to-end integration tests using local synthetic data."""

import json
import pytest
from pathlib import Path

from benchmark_indic_mt_eval.config import BenchmarkConfig
from benchmark_indic_mt_eval.runner import run_benchmark


def _create_synthetic_data(data_dir: Path) -> None:
    """Create minimal synthetic JSONL files for testing."""
    for prefix in ["Hin", "Tam"]:
        filepath = data_dir / f"{prefix}_test.jsonl"
        rows = []
        for i in range(15):
            score = round(0.3 + 0.05 * i, 2)
            rows.append(
                json.dumps(
                    {
                        "src": f"Source sentence number {i}.",
                        "ref": f"Reference translation number {i}.",
                        "translation": (
                            f"Reference translation number {i}."
                            if i < 7
                            else f"Wrong output {i} bad."
                        ),
                        "mqm_norm_score": str(score),
                        "da_norm_score": str(score),
                        "adequacy_score": str(int(score * 25)),
                        "fluency_score": str(int(score * 25)),
                        "full_score": str(int(score * 20)),
                        "completion": [],
                        "prompt": "...",
                    },
                    ensure_ascii=False,
                )
            )
        filepath.write_text("\n".join(rows) + "\n", encoding="utf-8")


class TestEndToEnd:
    def test_verify_flow(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _create_synthetic_data(data_dir)

        output_path = tmp_path / "results.json"
        config = BenchmarkConfig.from_dict(
            {
                "data": {
                    "languages": ["hi"],
                    "split": "test",
                    "max_samples": 10,
                    "data_dir": str(data_dir),
                },
                "metrics": {"metrics": ["bleu", "chrf"]},
                "run": {
                    "output": str(output_path),
                    "levels": ["segment"],
                },
            }
        )

        results = run_benchmark(config)

        assert "config" in results
        assert "results" in results
        assert "summary" in results
        assert "segment_level" in results["results"]
        assert "hi" in results["results"]["segment_level"]
        assert "bleu" in results["results"]["segment_level"]["hi"]
        assert "pearson" in results["results"]["segment_level"]["hi"]["bleu"]
        assert "kendall_tau" in results["results"]["segment_level"]["hi"]["bleu"]

        assert output_path.exists()
        with open(output_path) as f:
            saved = json.load(f)
        assert saved["config"]["languages"] == ["hi"]

    def test_multi_language_flow(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _create_synthetic_data(data_dir)

        output_path = tmp_path / "results.json"
        config = BenchmarkConfig.from_dict(
            {
                "data": {
                    "languages": ["hi", "ta"],
                    "split": "test",
                    "data_dir": str(data_dir),
                },
                "metrics": {"metrics": ["bleu"]},
                "run": {
                    "output": str(output_path),
                    "levels": ["segment", "system"],
                },
            }
        )

        results = run_benchmark(config)

        assert "hi" in results["results"]["segment_level"]
        assert "ta" in results["results"]["segment_level"]
        assert "hi" in results["results"]["system_level"]

        assert "segment_level_avg" in results["summary"]
        assert "bleu" in results["summary"]["segment_level_avg"]

    def test_system_level_correlation(self, tmp_path):
        """Verify system-level produces valid correlation values."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _create_synthetic_data(data_dir)

        output_path = tmp_path / "results.json"
        config = BenchmarkConfig.from_dict(
            {
                "data": {
                    "languages": ["hi"],
                    "split": "test",
                    "data_dir": str(data_dir),
                },
                "metrics": {"metrics": ["bleu", "ter"]},
                "run": {
                    "output": str(output_path),
                    "levels": ["system"],
                },
            }
        )

        results = run_benchmark(config)
        sys_results = results["results"]["system_level"]["hi"]
        for metric in ["bleu", "ter"]:
            assert -1.0 <= sys_results[metric]["pearson"] <= 1.0
            assert -1.0 <= sys_results[metric]["kendall_tau"] <= 1.0
