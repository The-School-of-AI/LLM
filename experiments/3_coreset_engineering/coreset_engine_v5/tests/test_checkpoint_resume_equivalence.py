import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_pipeline_yaml(
    *,
    path: Path,
    curriculum_yaml_path: Path,
    output_coreset_path: Path,
    output_manifest_path: Path,
    stage_target_tokens: int,
) -> None:
    cfg = {
        "curriculum": {
            "curriculum_yaml_path": str(curriculum_yaml_path),
            "freeze_curriculum": True,
            "enforce_rolling_window": True,
            "rolling_window_tolerance": 0.03,
            "deterministic_seed": 42,
            "seed_scope": ["sampling", "shuffling", "stage_transition"],
        },
        "io": {
            "output_coreset_path": str(output_coreset_path),
            "output_manifest_path": str(output_manifest_path),
            "output_index_format": "jsonl",
            "num_parallel_loaders": 1,
        },
        # Keep stages minimal so the test runs fast.
        "stages": {
            "1B": {
                "stage_name": "1B",
                "target_tokens": int(stage_target_tokens),
            }
        },
        # Keep dedup/diversity enabled by default; the key regression here is
        # deterministic equivalence across crash+resume.
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_builder(
    *,
    config_path: Path,
    curriculum_path: Path,
    input_path: Path,
    checkpoint_dir: Path,
    total_input_tokens_estimate: int,
    crash_after_batch: int | None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Make cross-process hash iteration deterministic.
    env.setdefault("PYTHONHASHSEED", "0")

    if crash_after_batch is not None:
        env["CORESET_SIMULATE_CRASH_AFTER_CHECKPOINT_STAGE"] = "1B"
        env["CORESET_SIMULATE_CRASH_AFTER_CHECKPOINT_BATCH"] = str(
            int(crash_after_batch)
        )
    else:
        env.pop("CORESET_SIMULATE_CRASH_AFTER_CHECKPOINT_STAGE", None)
        env.pop("CORESET_SIMULATE_CRASH_AFTER_CHECKPOINT_BATCH", None)

    cmd = [
        sys.executable,
        str(_repo_root() / "coreset_builder.py"),
        "--config",
        str(config_path),
        "--curriculum",
        str(curriculum_path),
        "--input-path",
        str(input_path),
        "--input-format",
        "jsonl",
        "--batch-size",
        "20",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--total-input-tokens-estimate",
        str(int(total_input_tokens_estimate)),
        "--stages",
        "1B",
        "--stage-target-scale",
        "1.0",
    ]

    return subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _read_selected_chunk_ids(
    output_coreset_path: Path, stage_name: str = "1B"
) -> set[str]:
    stage_dir = output_coreset_path / stage_name
    parquet_parts = sorted(stage_dir.glob("selected_indices_part_*.parquet"))
    jsonl_parts = sorted(stage_dir.glob("selected_indices_part_*.jsonl"))
    parts = parquet_parts or jsonl_parts
    if not parts:
        nearby = sorted(
            p.relative_to(output_coreset_path)
            for p in output_coreset_path.rglob("*")
            if p.is_file()
        )
        preview = "\n".join(str(p) for p in nearby[:50])
        raise AssertionError(
            f"No parquet parts found under {stage_dir}. "
            f"Files under output_coreset_path (first 50):\n{preview}"
        )

    out: set[str] = set()
    for p in parts:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
            if "chunk_id" not in df.columns:
                raise AssertionError(f"Missing chunk_id column in {p}")
            out.update(df["chunk_id"].astype(str).tolist())
        elif p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    cid = row.get("chunk_id")
                    if cid is not None:
                        out.add(str(cid))
        else:
            raise AssertionError(f"Unsupported part file type: {p}")
    return out


def test_clean_vs_resume_equivalence_tiny_dataset(tmp_path: Path) -> None:
    """Regression test: crash+resume must match a clean run.

    This specifically guards against regressions in checkpoint state restoration
    (engine internal state + deterministic ordering in language enforcement).
    """

    curriculum_path = _repo_root() / "config" / "curriculum.yaml"
    assert curriculum_path.exists(), "Expected repo curriculum.yaml to exist"

    # Create a tiny synthetic dataset that spans multiple batches.
    # Mostly English, a couple of Indic rows with lower band_score so they tend
    # not to be selected unless language policy adds them.
    rows: list[dict] = []
    for i in range(60):
        is_indic = i in {5, 15, 23}
        rows.append(
            {
                "chunk_id": f"c{i:03d}",
                "dataset_id": "ds",
                "token_count_estimate": 1,
                "domain": "web",
                "language": "hi" if is_indic else "en",
                "band": "B0",
                "band_score": 0.0 if is_indic else 1.0,
            }
        )

    input_path = tmp_path / "tiny.jsonl"
    _write_jsonl(input_path, rows)
    total_tokens = sum(int(r["token_count_estimate"]) for r in rows)

    # Stage target is smaller than total input so selection must make choices.
    stage_target_tokens = 60

    # Clean run
    clean_out = tmp_path / "out_clean" / "coresets"
    clean_man = tmp_path / "out_clean" / "manifests"
    clean_cfg = tmp_path / "pipeline_clean.yaml"
    clean_chk = tmp_path / "chk_clean"
    _write_pipeline_yaml(
        path=clean_cfg,
        curriculum_yaml_path=curriculum_path,
        output_coreset_path=clean_out,
        output_manifest_path=clean_man,
        stage_target_tokens=stage_target_tokens,
    )
    clean_res = _run_builder(
        config_path=clean_cfg,
        curriculum_path=curriculum_path,
        input_path=input_path,
        checkpoint_dir=clean_chk,
        total_input_tokens_estimate=total_tokens,
        crash_after_batch=None,
    )
    assert (
        clean_res.returncode == 0
    ), f"Clean run failed (rc={clean_res.returncode})\nSTDOUT:\n{clean_res.stdout}\nSTDERR:\n{clean_res.stderr}"

    # Crash after checkpoint batch 1, then resume to completion
    resume_out = tmp_path / "out_resume" / "coresets"
    resume_man = tmp_path / "out_resume" / "manifests"
    resume_cfg = tmp_path / "pipeline_resume.yaml"
    resume_chk = tmp_path / "chk_resume"
    _write_pipeline_yaml(
        path=resume_cfg,
        curriculum_yaml_path=curriculum_path,
        output_coreset_path=resume_out,
        output_manifest_path=resume_man,
        stage_target_tokens=stage_target_tokens,
    )

    crash_res = _run_builder(
        config_path=resume_cfg,
        curriculum_path=curriculum_path,
        input_path=input_path,
        checkpoint_dir=resume_chk,
        total_input_tokens_estimate=total_tokens,
        crash_after_batch=1,
    )
    assert (
        crash_res.returncode != 0
    ), "Crash simulation did not terminate the process as expected"

    resumed_res = _run_builder(
        config_path=resume_cfg,
        curriculum_path=curriculum_path,
        input_path=input_path,
        checkpoint_dir=resume_chk,
        total_input_tokens_estimate=total_tokens,
        crash_after_batch=None,
    )
    assert (
        resumed_res.returncode == 0
    ), f"Resumed run failed (rc={resumed_res.returncode})\nSTDOUT:\n{resumed_res.stdout}\nSTDERR:\n{resumed_res.stderr}"

    clean_ids = _read_selected_chunk_ids(clean_out, "1B")
    resumed_ids = _read_selected_chunk_ids(resume_out, "1B")

    assert clean_ids == resumed_ids, (
        "Selected chunk sets differ between clean and crash+resume runs. "
        f"Only-in-clean={sorted(clean_ids - resumed_ids)[:10]} "
        f"Only-in-resume={sorted(resumed_ids - clean_ids)[:10]}"
    )
