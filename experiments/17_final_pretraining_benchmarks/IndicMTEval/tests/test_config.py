import pytest
from benchmark_indic_mt_eval.config import (
    BenchmarkConfig,
    DataConfig,
    MetricConfig,
    RunConfig,
    load_config,
)


def test_default_config():
    cfg = BenchmarkConfig()
    assert cfg.data.languages == ["hi", "ta", "mr", "ml", "gu"]
    assert cfg.data.split == "test"
    assert cfg.data.max_samples is None
    assert cfg.metrics.metrics == ["bleu", "chrf", "ter"]
    assert cfg.metrics.device == "auto"
    assert cfg.run.levels == ["segment", "system"]
    assert cfg.run.output == "results.json"


def test_config_from_dict():
    d = {
        "data": {"languages": ["hi"], "max_samples": 10},
        "metrics": {"metrics": ["bleu"]},
    }
    cfg = BenchmarkConfig.from_dict(d)
    assert cfg.data.languages == ["hi"]
    assert cfg.data.max_samples == 10
    assert cfg.metrics.metrics == ["bleu"]
    assert cfg.run.output == "results.json"


def test_load_config_from_yaml(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(
        "data:\n  languages: [hi, ta]\n  max_samples: 5\nmetrics:\n  metrics: [bleu]\n"
    )
    cfg = load_config(str(yaml_file))
    assert cfg.data.languages == ["hi", "ta"]
    assert cfg.data.max_samples == 5


def test_config_all_languages_shortcut():
    cfg = BenchmarkConfig.from_dict({"data": {"languages": ["all"]}})
    assert cfg.data.languages == ["hi", "ta", "mr", "ml", "gu"]
