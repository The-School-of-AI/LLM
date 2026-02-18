from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run(tool: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(tool), *args],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("ext", ["jsonl", "parquet"])
def test_check_indices_determinism_smoke(tmp_path: Path, ext: str) -> None:
    pyarrow = pytest.importorskip("pyarrow")  # noqa: F841
    import pandas as pd

    tool = (
        Path(__file__).resolve().parents[1] / "tools" / "check_indices_determinism.py"
    )

    out_a = tmp_path / "output_a" / "coresets" / "1B"
    out_b = tmp_path / "output_b" / "coresets" / "1B"
    out_a.mkdir(parents=True)
    out_b.mkdir(parents=True)

    rel = f"selected_indices_part_shard000_batch000001.{ext}"
    file_a = out_a / rel
    file_b = out_b / rel

    df = pd.DataFrame(
        {
            "chunk_id": ["c1", "c2", "c3"],
            "domain": ["web", "web", "books"],
            "band": ["B0", "B1", "B2"],
        }
    )

    if ext == "jsonl":
        file_a.write_text(
            "\n".join(df.to_json(orient="records", lines=True).splitlines()) + "\n",
            encoding="utf-8",
        )
        file_b.write_text(
            "\n".join(df.to_json(orient="records", lines=True).splitlines()) + "\n",
            encoding="utf-8",
        )
    else:
        df.to_parquet(file_a, index=False)
        df.to_parquet(file_b, index=False)

    ok = _run(tool, "--output-dirs", str(out_a.parents[1]), str(out_b.parents[1]))
    assert ok.returncode == 0, ok.stderr + "\n" + ok.stdout

    # Now introduce a change.
    df2 = df.copy()
    df2.loc[1, "band"] = "B5"

    if ext == "jsonl":
        file_b.write_text(
            "\n".join(df2.to_json(orient="records", lines=True).splitlines()) + "\n",
            encoding="utf-8",
        )
    else:
        df2.to_parquet(file_b, index=False)

    bad = _run(
        tool,
        "--output-dirs",
        str(out_a.parents[1]),
        str(out_b.parents[1]),
        "--show-first-diff",
    )
    assert bad.returncode == 2
    assert "mismatch" in (bad.stdout + bad.stderr).lower()


@pytest.mark.parametrize("ext", ["jsonl", "parquet"])
def test_check_indices_determinism_ignores_row_order(tmp_path: Path, ext: str) -> None:
    pyarrow = pytest.importorskip("pyarrow")  # noqa: F841
    import pandas as pd

    tool = (
        Path(__file__).resolve().parents[1] / "tools" / "check_indices_determinism.py"
    )

    out_a = tmp_path / "output_a" / "coresets" / "1B"
    out_b = tmp_path / "output_b" / "coresets" / "1B"
    out_a.mkdir(parents=True)
    out_b.mkdir(parents=True)

    rel = f"selected_indices_part_shard000_batch000001.{ext}"
    file_a = out_a / rel
    file_b = out_b / rel

    df = pd.DataFrame(
        {
            "chunk_id": ["c1", "c2", "c3", "c4"],
            "domain": ["web", "web", "books", "web"],
            "band": ["B0", "B1", "B2", "B2"],
        }
    )

    df_shuffled = df.iloc[::-1].reset_index(drop=True)

    if ext == "jsonl":
        file_a.write_text(
            "\n".join(df.to_json(orient="records", lines=True).splitlines()) + "\n",
            encoding="utf-8",
        )
        file_b.write_text(
            "\n".join(df_shuffled.to_json(orient="records", lines=True).splitlines())
            + "\n",
            encoding="utf-8",
        )
    else:
        df.to_parquet(file_a, index=False)
        df_shuffled.to_parquet(file_b, index=False)

    ok = _run(tool, "--output-dirs", str(out_a.parents[1]), str(out_b.parents[1]))
    assert ok.returncode == 0, ok.stderr + "\n" + ok.stdout
