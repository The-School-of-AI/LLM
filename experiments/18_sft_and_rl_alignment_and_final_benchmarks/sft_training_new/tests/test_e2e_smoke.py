"""
End-to-end smoke test — runs stages 1–6 on 100 synthetic examples.

Stage 0 is skipped (fixture file used as input directly).
Uses GPT-2 tokenizer (public, no auth required) for Stage 4.

Requirements:
    pip install transformers datasets pyarrow datasketch langdetect pyyaml click

Run:
    pytest tests/test_e2e_smoke.py -v
"""
import json
import shutil
import pytest
from pathlib import Path

# Skip entire module if transformers is not installed
pytest.importorskip("transformers")
pytest.importorskip("pyarrow")

FIXTURES = Path(__file__).parent / "fixtures"
SMOKE_JSONL = FIXTURES / "smoke_100.jsonl"


@pytest.fixture()
def smoke_cfg(tmp_path):
    """Build a PipelineConfig for the smoke test using tmp_path as work/output dirs."""
    import yaml
    from pipeline.config import PipelineConfig

    # Load the smoke config template
    smoke_config_path = Path(__file__).parent.parent / "config" / "pipeline_smoke.yaml"
    with open(smoke_config_path) as f:
        raw = yaml.safe_load(f)

    raw["global"]["work_dir"]   = str(tmp_path / "work")
    raw["global"]["output_dir"] = str(tmp_path / "out")
    raw["stage1"]["input_file"] = "smoke_input.jsonl"

    cfg = PipelineConfig._from_dict(raw)

    # Copy fixture to work_dir
    work_dir = Path(cfg.globals.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(SMOKE_JSONL, work_dir / "smoke_input.jsonl")

    return cfg


def test_stage1_runs_without_error(smoke_cfg):
    from pipeline import stage1_validate
    stage1_validate.run(smoke_cfg)
    out = Path(smoke_cfg.globals.work_dir) / smoke_cfg.stage1.output_file
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) > 0, "Stage 1 output is empty"


def test_stage2_runs_without_error(smoke_cfg):
    from pipeline import stage1_validate, stage2_clean
    stage1_validate.run(smoke_cfg)
    stage2_clean.run(smoke_cfg)
    out = Path(smoke_cfg.globals.work_dir) / smoke_cfg.stage2.output_file
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) > 0, "Stage 2 output is empty"


def test_stage3_runs_and_produces_role_spans(smoke_cfg):
    from pipeline import stage1_validate, stage2_clean, stage3_template
    stage1_validate.run(smoke_cfg)
    stage2_clean.run(smoke_cfg)
    stage3_template.run(smoke_cfg)

    out = Path(smoke_cfg.globals.work_dir) / smoke_cfg.stage3.output_file
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) > 0

    rec = json.loads(lines[0])
    assert "formatted_text" in rec
    assert "role_spans" in rec
    assert isinstance(rec["role_spans"], list)
    assert len(rec["role_spans"]) >= 2


def test_stage4_produces_input_ids(smoke_cfg):
    from pipeline import stage1_validate, stage2_clean, stage3_template, stage4_tokenize
    stage1_validate.run(smoke_cfg)
    stage2_clean.run(smoke_cfg)
    stage3_template.run(smoke_cfg)
    stage4_tokenize.run(smoke_cfg)

    out = Path(smoke_cfg.globals.work_dir) / smoke_cfg.stage4.output_file
    assert out.exists()

    from pipeline.io.arrow_writer import iter_arrow
    records = list(iter_arrow(out))
    assert len(records) > 0

    rec = records[0]
    assert "input_ids" in rec
    assert "attention_mask" in rec
    assert "token_role_spans" in rec
    assert isinstance(rec["input_ids"], list)
    assert len(rec["input_ids"]) > 0


def test_stage5_labels_correct(smoke_cfg):
    from pipeline import stage1_validate, stage2_clean, stage3_template, stage4_tokenize, stage5_mask
    stage1_validate.run(smoke_cfg)
    stage2_clean.run(smoke_cfg)
    stage3_template.run(smoke_cfg)
    stage4_tokenize.run(smoke_cfg)
    stage5_mask.run(smoke_cfg)

    out = Path(smoke_cfg.globals.work_dir) / smoke_cfg.stage5.output_file
    assert out.exists()

    from pipeline.io.arrow_writer import iter_arrow
    records = list(iter_arrow(out))
    assert len(records) > 0

    IGNORE = smoke_cfg.stage5.ignore_index

    for rec in records:
        input_ids = rec.get("input_ids", [])
        labels    = rec.get("labels", [])

        assert len(labels) == len(input_ids), "labels and input_ids length mismatch"
        unmasked = [l for l in labels if l != IGNORE]
        assert len(unmasked) > 0, "All labels are masked — should have at least some assistant tokens"
        # Unmasked labels must equal their corresponding input_id values
        for i, (iid, lbl) in enumerate(zip(input_ids, labels)):
            if lbl != IGNORE:
                assert lbl == iid, f"Position {i}: label={lbl} != input_id={iid}"


def test_stage6_gate_passes(smoke_cfg):
    from pipeline import (
        stage1_validate, stage2_clean, stage3_template,
        stage4_tokenize, stage5_mask, stage6_quality,
    )
    stage1_validate.run(smoke_cfg)
    stage2_clean.run(smoke_cfg)
    stage3_template.run(smoke_cfg)
    stage4_tokenize.run(smoke_cfg)
    stage5_mask.run(smoke_cfg)
    stage6_quality.run(smoke_cfg)

    out_dir = Path(smoke_cfg.globals.output_dir)
    report_path = out_dir / smoke_cfg.stage6.report.output_json
    assert report_path.exists()

    with open(report_path) as f:
        report = json.load(f)

    assert report.get("total_examples", 0) > 0
    assert report.get("gate_passed", True) is True


def test_final_shards_exist(smoke_cfg):
    from pipeline import (
        stage1_validate, stage2_clean, stage3_template,
        stage4_tokenize, stage5_mask, stage6_quality,
    )
    stage1_validate.run(smoke_cfg)
    stage2_clean.run(smoke_cfg)
    stage3_template.run(smoke_cfg)
    stage4_tokenize.run(smoke_cfg)
    stage5_mask.run(smoke_cfg)
    stage6_quality.run(smoke_cfg)

    shard_dir = Path(smoke_cfg.globals.output_dir) / smoke_cfg.stage6.output_dir_sharded
    assert shard_dir.exists()
    shards = list(shard_dir.glob("*.arrow"))
    assert len(shards) > 0, "No Arrow shards produced"


def test_funnel_tracker_accumulates(smoke_cfg):
    from pipeline import (
        stage1_validate, stage2_clean, stage3_template,
        stage4_tokenize, stage5_mask, stage6_quality,
    )
    from pipeline.funnel_tracker import FunnelTracker
    from pathlib import Path

    funnel_path = Path(smoke_cfg.globals.output_dir) / "funnel_report.json"
    tracker = FunnelTracker(total_input=100, output_path=funnel_path)

    stage1_validate.run(smoke_cfg, tracker)
    stage2_clean.run(smoke_cfg, tracker)
    stage3_template.run(smoke_cfg, tracker)
    stage4_tokenize.run(smoke_cfg, tracker)
    stage5_mask.run(smoke_cfg, tracker)
    stage6_quality.run(smoke_cfg, tracker)
    tracker.save()

    assert funnel_path.exists()
    with open(funnel_path) as f:
        report = json.load(f)

    assert "total_input" in report
    assert "total_output" in report
    assert report["total_output"] > 0
