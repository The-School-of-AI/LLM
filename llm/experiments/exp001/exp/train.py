from dataclasses import asdict, dataclass
from typing import Any

import deepspeed
import torch
import yaml
from exp.distributed import set_seed
from exp.model import build_kronecker_vocab, build_model
from exp.opus import AdamWPreconditionerView, RandomInDistributionProxyProvider
from exp.proxy_dataset import ProxyDatasetConfig, get_proxy_dataloader

from llm.data import get_dataloaders, get_tokenizer
from llm.profiler import PipelineProfiler
from llm.utils import print_rank_0


@dataclass
class DataConfig:
    max_length: int
    dataset_name: str | None = None
    dataset_config: str | None = None
    tokenized_dataset_path: str | None = None
    dataset_cache_dir: str | None = None
    local_nvme_cache_dir: str | None = None
    require_local_nvme: bool = False
    block_sizes: list[int] | None = None
    block_size_counts: dict[Any, Any] | None = None
    domain_column: str | None = None
    concat_across_domains: bool = False
    drop_remainder: bool = True
    num_workers: int = 12
    tokenize_num_proc: int | None = None


@dataclass
class OpusConfig:
    strict_shard_preconditioner: bool


@dataclass
class Config:
    seed: int
    deepspeed_config: str
    tokenizer_dir: str
    profiler_output_dir: str
    data: DataConfig
    proxy: ProxyDatasetConfig
    opus: OpusConfig


class Trainer:
    def __init__(self, local_rank: int, c: Config):
        with open(c.deepspeed_config, "r") as f:
            ds_config = yaml.safe_load(f)
        set_seed(c.seed)
        self.pipe = PipelineProfiler(rank=local_rank, output_dir=c.profiler_output_dir)

        with self.pipe.stage("tokenizer_load"):
            print_rank_0("loading tokenizer")
            tokenizer = get_tokenizer(c.tokenizer_dir)

        with self.pipe.stage("data_load"):
            print_rank_0("loading dataloaders")
            batch_size = ds_config["train_micro_batch_size_per_gpu"]
            self.train_loader, _, _, _ = get_dataloaders(
                tokenizer=tokenizer, batch_size=batch_size, **asdict(c.data)
            )
            proxy_loader = get_proxy_dataloader(
                tokenizer=tokenizer,
                config=c.proxy,
                seed=c.seed,
            )
            self.proxy_provider = RandomInDistributionProxyProvider(proxy_loader)

        with self.pipe.stage("kronecker_vocab_build"):
            print_rank_0("building kronecker embeddings")
            bpe_vocab, k_embed = build_kronecker_vocab(tokenizer)

        with self.pipe.stage("model_build"):
            print_rank_0("building model")
            model = build_model(
                embedding_type="kronecker", bpe_vocab=bpe_vocab, k_embed=k_embed
            )

        with self.pipe.stage("model_to_bf16"):
            print_rank_0("casting model to bfloat16")
            model = model.to(dtype=torch.bfloat16)

        self.engine: deepspeed.DeepSpeedEngine
        with self.pipe.stage("deepspeed_init"):
            self.engine, self.optimizer, self.lr_scheduler, _ = deepspeed.initialize(
                config_params=ds_config, model=model
            )

        self.preconditioner_view = AdamWPreconditionerView(
            self.optimizer, strict_shard_only=c.opus.strict_shard_preconditioner
        )

        print_rank_0(f"ZeRo Stage: {self.engine.zero_optimization_stage()}")

    def train(self):
        pass
