from typing import Any

import deepspeed
import torch.nn as nn
from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from llm.aws import S3Config
from llm.checkpoint import S3CheckpointManager
from llm.config import (
    CheckpointConfig,
    DataConfig,
    LossSpikeConfig,
    ModelConfig,
    ObservabilityConfig,
    TrainingConfig,
)
from llm.data import get_dataloaders, get_tokenizer
from llm.logger import Logger
from llm.logger.noop import NoOpLogger
from llm.models import (
    KroneckerConfig,
    KroneckerEmbeddings,
    Model1B,
)
from llm.models import ModelConfig as Model1BConfig
from llm.profiler import BaseStepProfiler, NoOpStepProfiler, StepProfiler


def build_tokenizer(cfg: DataConfig) -> TokenizersBackend:
    return get_tokenizer(cfg.tokenizer_dir)


def build_dataloaders(
    cfg: DataConfig,
    batch_size: int,
    tokenizer,
) -> tuple[
    DataLoader,
    DistributedSampler | None,
    DataLoader,
    DistributedSampler | None,
    DataLoader,
    DistributedSampler | None,
]:
    train_loader, eval_loader, test_loader, dataset_info = get_dataloaders(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=cfg.max_length,
        tokenized_dataset_path=cfg.tokenized_dataset_path,
        dataset_cache_dir=cfg.dataset_cache_dir,
        local_nvme_cache_dir=cfg.local_nvme_cache_dir,
        require_local_nvme=cfg.require_local_nvme,
        pack_into_blocks=cfg.pack_into_blocks,
        block_sizes=cfg.block_sizes,
        block_size_counts=cfg.block_size_counts,
        domain_column=cfg.domain_column,
        concat_across_domains=cfg.concat_across_domains,
        drop_remainder=cfg.drop_remainder,
        num_workers=cfg.num_workers,
        tokenize_num_proc=cfg.tokenize_num_proc,
    )

    return (
        train_loader,
        dataset_info["train_sampler"],
        eval_loader,
        dataset_info["eval_sampler"],
        test_loader,
        dataset_info["test_sampler"],
    )


def build_model(model_cfg: ModelConfig, tokenizer: TokenizersBackend) -> nn.Module:
    bpe_vocab: list[str] | None = None
    k_embed: KroneckerEmbeddings | None = None
    if model_cfg.embedding_type == "kronecker":
        bpe_vocab, k_embed = _build_kronecker_vocab(tokenizer)

    match model_cfg.variant:
        case "1b_reversible":
            return Model1B(
                config=Model1BConfig(**model_cfg.overrides),  # type: ignore
                embedding_type=model_cfg.embedding_type,
                bpe_vocab=bpe_vocab,
                pf_codec=k_embed,
            )

        case _:
            raise ValueError(f"unknown model variant: {model_cfg.variant}")


def _build_kronecker_vocab(
    tokenizer: TokenizersBackend,
) -> tuple[list[str], KroneckerEmbeddings]:
    # Use len(tokenizer) to include special tokens (pad, eos, etc.)
    vocab_size = len(tokenizer)
    bpe_vocab = []
    for i in range(vocab_size):
        try:
            token = tokenizer.decode([i])
            bpe_vocab.append(token if token else f"<unk_{i}>")
        except Exception:
            bpe_vocab.append(f"<unk_{i}>")

    # Create Kronecker codec
    pf_config = KroneckerConfig(
        CHAR_DIM=256,
        POS_DIM=32,
        D=8192,
        length_normalize=True,
        truncate_long_words=True,
    )

    return bpe_vocab, KroneckerEmbeddings(pf_config)


def build_deepspeed(model: nn.Module, ds_config: dict[str, Any]) -> DeepSpeedEngine:
    engine, _, _, _ = deepspeed.initialize(config_params=ds_config, model=model)
    return engine


def build_checkpoint_manager(
    cfg: CheckpointConfig, checkpoint_output_dir: str
) -> S3CheckpointManager | None:
    match cfg.backend:
        case "s3":
            ckpt_config = cfg.s3
            assert (
                ckpt_config is not None
            ), "S3 checkpoint configuration must be present if checkpoint backend type is s3"
            return S3CheckpointManager(
                S3Config(
                    bucket_name=ckpt_config.bucket,
                    s3_prefix=ckpt_config.prefix,
                    region=ckpt_config.region,
                    local_checkpoint_dir=checkpoint_output_dir,
                    keep_last_n_checkpoints=ckpt_config.keep_last_n,
                    cleanup_after_upload=ckpt_config.cleanup_after_upload,
                )
            )
        case "none":
            return None

    raise ValueError(f"unknown checkpoint manager backend: {cfg.backend}")


def build_step_profiler(
    cfg: TrainingConfig,
    rank: int = 0,
) -> BaseStepProfiler:
    """
    Build a step profiler from training config.

    Returns a real StepProfiler when profile_steps is non-empty, or a
    NoOpStepProfiler (zero overhead) when profiling is disabled.

    Args:
        cfg  : Training configuration supplying profile_steps.
        rank : Local rank of this process (only rank-0 writes output).
    """
    if not cfg.profile_steps:
        return NoOpStepProfiler()

    return StepProfiler(
        rank=rank,
        profile_steps=set(cfg.profile_steps),
    )


def build_loss_spike_detector(
    cfg: LossSpikeConfig,
) -> "LossSpikeDetector | None":
    """Build a LossSpikeDetector from config, or None if disabled."""
    if not cfg.enabled:
        return None

    from llm.loss_spike_recovery import LossSpikeDetector

    return LossSpikeDetector(
        window_size=cfg.window_size,
        z_threshold=cfg.z_threshold,
        min_spike_ratio=cfg.min_spike_ratio,
        min_abs_delta=cfg.min_abs_delta,
        cooldown_steps=cfg.cooldown_steps,
    )


def build_observability(
    cfg: ObservabilityConfig,
    run_id: str,
    rank: int,
) -> Logger:
    match cfg.backend:
        case "none":
            return NoOpLogger()
        case _:
            raise RuntimeError(f"observability backend {cfg.backend} not supported")
