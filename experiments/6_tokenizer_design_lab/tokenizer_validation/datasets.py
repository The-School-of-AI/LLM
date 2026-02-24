from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq


@dataclass(frozen=True)
class SampledText:
    text: str
    source: str
    meta: dict[str, Any]


def _reservoir_sample(iterable: Iterable[SampledText], k: int, rng: random.Random) -> list[SampledText]:
    out: list[SampledText] = []
    for i, item in enumerate(iterable):
        if i < k:
            out.append(item)
            continue
        j = rng.randint(0, i)
        if j < k:
            out[j] = item
    return out


def iter_parquet_text(path: Path, *, text_col: str = "text", language_col: str = "language") -> Iterator[SampledText]:
    table = pq.ParquetFile(path)
    cols = [text_col]
    if language_col:
        cols.append(language_col)

    for batch in table.iter_batches(columns=cols, batch_size=8192):
        texts = batch.column(batch.schema.get_field_index(text_col)).to_pylist()
        langs = None
        if language_col in batch.schema.names:
            langs = batch.column(batch.schema.get_field_index(language_col)).to_pylist()

        for idx, t in enumerate(texts):
            if t is None:
                continue
            meta: dict[str, Any] = {}
            if langs is not None:
                meta["language"] = langs[idx]
            yield SampledText(text=str(t), source="pretraining", meta=meta)


def iter_jsonl_text(path: Path, *, text_field: str = "text") -> Iterator[SampledText]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get(text_field)
            if not isinstance(t, str) or not t:
                continue
            meta = {k: v for k, v in obj.items() if k != text_field}
            yield SampledText(text=t, source="golden", meta=meta)


def iter_sft_txt(paths: list[Path]) -> Iterator[SampledText]:
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                yield SampledText(text=line, source="sft", meta={"file": str(p)})


def sample_pretraining(path: Path, *, n: int, seed: int) -> list[SampledText]:
    rng = random.Random(seed)
    return _reservoir_sample(iter_parquet_text(path), n, rng)


def sample_golden(path: Path, *, n: int, seed: int) -> list[SampledText]:
    rng = random.Random(seed)
    return _reservoir_sample(iter_jsonl_text(path), n, rng)


def sample_sft(paths: list[Path], *, n: int, seed: int) -> list[SampledText]:
    rng = random.Random(seed)
    return _reservoir_sample(iter_sft_txt(paths), n, rng)
