import os
import torch
import pyarrow.parquet as pq
from spdl.pipeline import PipelineBuilder
from common import BATCH_SIZE, NUM_THREADS, PREFETCH_BUFFER, TOKENS_COLUMN

def load_tokens_from_local_parquet(file_path):
    table = pq.read_table(
        file_path,
        columns=[TOKENS_COLUMN],
        memory_map=True,
        use_threads=True
    )
    tokens = torch.tensor(table[TOKENS_COLUMN].to_pylist())
    return tokens

def build_pipeline(parquet_shards):
    return (
        PipelineBuilder()
        .add_source(parquet_shards)
        .pipe(
            load_tokens_from_local_parquet,
            concurrency=NUM_THREADS
        )
        .aggregate(BATCH_SIZE)
        .add_sink(PREFETCH_BUFFER)
        .build(num_threads=NUM_THREADS)
    )

class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=1)
