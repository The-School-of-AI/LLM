import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "18_sft_and_rl_alignment_and_final_benchmarks"
    / "03_evaluation"
    / "select_pretrain_checkpoint.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("select_pretrain_checkpoint", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_hf_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def workdir():
    path = Path(__file__).resolve().parent / f".tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_rank_checkpoints_keeps_unchecked_above_spiked():
    module = load_module()

    ranked = module.rank_checkpoints(
        [
            {
                "checkpoint": "step_100",
                "perplexity": 10.0,
                "benchmark_avg": 55.0,
                "stable": False,
            },
            {
                "checkpoint": "custom_name",
                "perplexity": 11.0,
                "benchmark_avg": 50.0,
                "stable": None,
            },
        ]
    )

    assert [item["checkpoint"] for item in ranked] == ["custom_name", "step_100"]
    assert module._stability_label(ranked[0]["stable"]) == "UNKNOWN"
    assert module._stability_label(ranked[1]["stable"]) == "SPIKE"


def test_check_stability_skips_bad_rows(workdir: Path):
    module = load_module()
    loss_log = workdir / "loss.csv"
    loss_log.write_text(
        "step,loss\n"
        "90,1.1\n"
        "91,\n"
        "oops,1.3\n"
        "95,not-a-number\n"
        "100,1.0\n",
        encoding="utf-8",
    )

    stable, note = module.check_stability(str(loss_log), checkpoint_step=100, window=20)

    assert stable is True
    assert "stability unchecked" in note.lower()


def test_check_stability_rejects_missing_required_columns(workdir: Path):
    module = load_module()
    loss_log = workdir / "loss.csv"
    loss_log.write_text("global_step,train_loss\n100,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="columns: step, loss"):
        module.check_stability(str(loss_log), checkpoint_step=100)


def test_main_creates_output_parent_and_preserves_unknown_stability(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
):
    module = load_module()
    checkpoint = workdir / "checkpoint_candidate"
    tokenizer_dir = workdir / "tokenizer"
    val_data = workdir / "val.jsonl"
    loss_log = workdir / "loss.csv"
    output_json = workdir / "nested" / "checkpoint_ranking.json"

    write_hf_checkpoint(checkpoint)
    tokenizer_dir.mkdir()
    val_data.write_text(json.dumps({"text": "hello world"}) + "\n", encoding="utf-8")
    loss_log.write_text("step,loss\n100,1.0\n", encoding="utf-8")

    class FakeTokenizer:
        eos_token_id = 1

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        type(
            "T",
            (),
            {
                "AutoTokenizer": type(
                    "AutoTokenizer",
                    (),
                    {
                        "from_pretrained": staticmethod(
                            lambda *args, **kwargs: FakeTokenizer()
                        )
                    },
                )
            },
        ),
    )
    monkeypatch.setattr(module, "_load_val_texts_from_jsonl", lambda path: ["hello world"])
    monkeypatch.setattr(module, "build_val_token_blocks", lambda texts, tokenizer, max_tokens=None: "blocks")
    monkeypatch.setattr(module, "_lm_eval_available", lambda: False)
    monkeypatch.setattr(module, "compute_perplexity", lambda **kwargs: 7.5)

    argv = [
        "select_pretrain_checkpoint.py",
        "--checkpoints",
        str(checkpoint),
        "--tokenizer_path",
        str(tokenizer_dir),
        "--val_data",
        str(val_data),
        "--loss_log",
        str(loss_log),
        "--output_json",
        str(output_json),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    ranked = payload["ranked_checkpoints"]
    assert output_json.exists()
    assert ranked[0]["stable"] is None
    assert ranked[0]["stability_note"].startswith("Step not inferred")


def test_main_exits_cleanly_for_missing_val_data(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
):
    module = load_module()
    checkpoint = workdir / "step_100"
    tokenizer_dir = workdir / "tokenizer"
    output_json = workdir / "out.json"

    write_hf_checkpoint(checkpoint)
    tokenizer_dir.mkdir()

    argv = [
        "select_pretrain_checkpoint.py",
        "--checkpoints",
        str(checkpoint),
        "--tokenizer_path",
        str(tokenizer_dir),
        "--val_data",
        str(workdir / "missing.jsonl"),
        "--output_json",
        str(output_json),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 1
