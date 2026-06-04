"""
Main entry point for 1.5B Non-Reversible Model training with DeepSpeed ZeRO-1.

Architecture:
- 1.513B parameters, 100% dense (no MoE)
- 131,072 vocabulary (2^17 - TSAI BPE tokenizer)
- 4096 hidden size, 8 layers: DDDGDDDG (6 DeltaNet + 2 GSA)
- Kronecker byte-level embeddings (8192 -> 4096 projection)
- Multi-Token Prediction (2 heads)
- Multi-Head Composition (4 streams, Sinkhorn)
- Non-reversible: standard sequential forward pass
- ZeRO-1: optimizer state partitioning only (~9GB/GPU on 8xA100-40GB)

Usage:
    deepspeed --num_gpus=8 main.py --config ../configs/train_1b_nonrev_z1.yaml
"""

import argparse
import json
import os
import time as import_time
import warnings
from typing import Any, Dict

# Runtime defaults
if "TORCHDYNAMO_DISABLE" not in os.environ:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Suppress deprecated pynvml FutureWarning
warnings.filterwarnings(
    "ignore",
    message=".*pynvml package is deprecated.*",
    category=FutureWarning,
)

import torch
import yaml

import deepspeed
from lightninglm.aws.config import S3Config
from lightninglm.checkpointing.checkpoint import S3CheckpointManager
from lightninglm.components.spot_checkpoint import SpotCheckpointOrchestrator
from lightninglm.data.bin_idx_dataloader import build_bin_idx_dataloader
from lightninglm.data.curriculum_dataloader_v2 import build_curriculum_v2_dataloader
from lightninglm.data.data import get_dataloaders, get_tokenizer
from lightninglm.kernels import HAS_TRITON
from lightninglm.models.recurrence_model_1b_non_rev import (
    KroneckerConfig,
    KroneckerEmbeddings,
    Model1B,
    ModelConfig,
)
from lightninglm.models.recurrence_model_3b_moe import (
    KroneckerConfig as KroneckerConfig_3B,
)
from lightninglm.models.recurrence_model_3b_moe import (
    KroneckerEmbeddings as KroneckerEmbeddings_3B,
)
from lightninglm.models.recurrence_model_3b_moe import Model3B as Model3B_Rev
from lightninglm.models.recurrence_model_3b_moe import ModelConfig as ModelConfig_3B_Rev
from lightninglm.models.recurrence_model_8b_moe import (
    KroneckerConfig as KroneckerConfig_8B,
)
from lightninglm.models.recurrence_model_8b_moe import (
    KroneckerEmbeddings as KroneckerEmbeddings_8B,
)
from lightninglm.models.recurrence_model_8b_moe import Model8B as Model8B_Rev
from lightninglm.models.recurrence_model_8b_moe import ModelConfig as ModelConfig_8B_Rev
from lightninglm.training.staged_lr import StagedCosineScheduler
from lightninglm.training.train import evaluate, train_epoch
from lightninglm.utils.profiler import PipelineProfiler
from lightninglm.utils.utils import print_rank_0, set_seed


class Config:
    """Configuration object from YAML."""

    def __init__(self, config_dict: Dict[str, Any], config_path: str = None):
        # Data configuration
        self.dataset_name = config_dict["data"]["dataset_name"]
        self.dataset_config = config_dict["data"]["dataset_config"]
        self.max_length = config_dict["data"]["max_length"]
        self.tokenized_dataset_path = config_dict["data"].get("tokenized_dataset_path")
        self.dataset_cache_dir = config_dict["data"].get("dataset_cache_dir")
        self.local_nvme_cache_dir = config_dict["data"].get("local_nvme_cache_dir")
        self.require_local_nvme = config_dict["data"].get("require_local_nvme", False)
        self.pack_into_blocks = config_dict["data"].get("pack_into_blocks", False)
        self.block_sizes = config_dict["data"].get("block_sizes")
        self.block_size_counts = config_dict["data"].get("block_size_counts")
        self.domain_column = config_dict["data"].get("domain_column")
        self.concat_across_domains = config_dict["data"].get(
            "concat_across_domains", False
        )
        self.drop_remainder = config_dict["data"].get("drop_remainder", True)
        self.num_workers = config_dict["data"].get("num_workers", 8)
        self.tokenize_num_proc = config_dict["data"].get("tokenize_num_proc")

        self.loader_type = config_dict["data"].get("loader_type", "hf")

        # Curriculum v2 configuration
        self.curriculum_config_path = config_dict["data"].get("curriculum_config_path")
        self.curriculum_stage = config_dict["data"].get("curriculum_stage", "1B")
        self.curriculum_mode = config_dict["data"].get("curriculum_mode", "combined")
        self.manifest_dir = config_dict["data"].get("manifest_dir")

        def _resolve_path(path_value):
            if not path_value:
                return path_value
            path_value = os.path.expanduser(os.path.expandvars(path_value))
            if os.path.isabs(path_value):
                return path_value
            if config_path:
                base = os.path.dirname(os.path.abspath(config_path))
                return os.path.abspath(os.path.join(base, path_value))
            return os.path.abspath(path_value)

        self.shard_dir = _resolve_path(config_dict["data"].get("shard_dir"))
        self.eval_shard_dir = _resolve_path(config_dict["data"].get("eval_shard_dir"))
        if self.curriculum_config_path:
            self.curriculum_config_path = _resolve_path(self.curriculum_config_path)
        if self.manifest_dir:
            self.manifest_dir = _resolve_path(self.manifest_dir)
        self.tokenizer_dir = _resolve_path(config_dict["data"].get("tokenizer_dir"))
        self.validate_tokenizer = config_dict["data"].get("validate_tokenizer", True)

        # Training configuration
        self.num_epochs = config_dict["training"]["num_epochs"]
        self.max_train_steps = config_dict["training"]["max_train_steps"]
        self.max_train_seconds = config_dict["training"].get("max_train_seconds", 600)
        self.max_eval_steps = config_dict["training"]["max_eval_steps"]
        self.log_interval = config_dict["training"]["log_interval"]
        self.seed = config_dict["training"]["seed"]
        self.require_fused_kernels = config_dict["training"].get(
            "require_fused_kernels", False
        )
        self.metrics_jsonl_path = config_dict["training"].get(
            "metrics_jsonl_path", "./logs/metrics.jsonl"
        )
        self.enable_system_metrics = config_dict["training"].get(
            "enable_system_metrics", False
        )
        self.init_model_path = config_dict["training"].get("init_model_path")
        self.max_chunk_gb = config_dict["training"].get("max_chunk_gb", 8.0)
        self.lr_schedule_path = _resolve_path(
            config_dict["training"].get("lr_schedule_path")
        )
        self.lr_stage = config_dict["training"].get("lr_stage")

        _ps = config_dict["training"].get("profile_steps", [])
        self.profile_steps: set = set(_ps) if _ps else set()

        # DeepSpeed configuration
        self.deepspeed_config = config_dict["deepspeed"]["config_path"]
        self.local_rank = config_dict["deepspeed"]["local_rank"]

        with open(self.deepspeed_config, "r") as f:
            deepspeed_cfg = json.load(f)
        self.batch_size = deepspeed_cfg.get("train_micro_batch_size_per_gpu", 1)

        # Model configuration
        self.model_name = config_dict["model"].get("model_name", "1b_nonrev")
        self.model_variant = config_dict["model"].get("model_variant", "non_reversible")
        self.embedding_type = config_dict["model"].get("embedding_type", "kronecker")
        self.moe_t4_enabled = bool(config_dict["model"].get("moe_t4_enabled", False))
        self.moe_t4_dispatcher = str(
            config_dict["model"].get("moe_t4_dispatcher", "deepep")
        )
        self.moe_expert_parallel_size = int(
            config_dict["model"].get("moe_expert_parallel_size", 1)
        )

        # Checkpoint configuration
        self.output_dir = config_dict["checkpoint"]["output_dir"]
        self.save_checkpoint = config_dict["checkpoint"]["save_checkpoint"]
        self.checkpoint_interval = config_dict["checkpoint"]["checkpoint_interval"]
        self.keep_last_n_checkpoints = config_dict["checkpoint"][
            "keep_last_n_checkpoints"
        ]
        self.resume_from_checkpoint = config_dict["checkpoint"][
            "resume_from_checkpoint"
        ]
        self.resume_step = config_dict["checkpoint"]["resume_step"]

        # S3 configuration
        self.use_s3 = config_dict["s3"]["enabled"]
        self.s3_bucket = config_dict["s3"]["bucket"]
        self.s3_prefix = config_dict["s3"]["prefix"]
        self.s3_region = config_dict["s3"]["region"]
        self.cleanup_after_upload = config_dict["s3"]["cleanup_after_upload"]

        # Generation configuration
        self.test_generation = config_dict["generation"]["test_generation"]
        self.generation_prompt = config_dict["generation"]["generation_prompt"]

        # Spot checkpoint configuration
        _spot = config_dict.get("spot_checkpoint", {})
        self.spot_checkpoint_enabled = _spot.get("enabled", True)
        self.spot_checkpoint_interval_seconds = _spot.get(
            "checkpoint_interval_seconds", 3600
        )
        self.spot_poll_interval = _spot.get("spot_poll_interval", 5.0)
        self.spot_keep_last_n_local = _spot.get("keep_last_n_local", 3)
        self.spot_log_max_bytes = _spot.get("log_max_bytes", 50 * 1024 * 1024)
        self.spot_metadata_url = _spot.get("spot_metadata_url")

        # Observability configuration
        _obs = config_dict.get("observability", {})
        self.observability_enabled = _obs.get("enabled", True)
        self.observability_log_dir = _obs.get("log_dir", "/tmp/training_logs")
        self.observability_metrics_port = _obs.get("metrics_port", 8000)
        self.observability_skip_vector_check = _obs.get("skip_vector_check", True)
        self.observability_system_metrics_interval = _obs.get(
            "system_metrics_interval", 5.0
        )

        # OPUS configuration
        _opus = config_dict.get("opus", {})
        self.opus_enabled = _opus.get("enabled", False)
        self.opus_score_seq_len = _opus.get("score_seq_len", 512)
        self.opus_sketch_dim = _opus.get("sketch_dim", 8192)
        self.opus_selection_ratio = _opus.get("selection_ratio", 0.5)
        self.opus_temperature = _opus.get("temperature", 1.0)
        self.opus_n_proxy = _opus.get("n_proxy", 8)
        self.opus_score_layer_stride = _opus.get("score_layer_stride", 1)
        self.opus_proxy_type = _opus.get("proxy_type", "random_in_distribution")
        self.opus_proxy_token_path = _resolve_path(_opus.get("proxy_token_path"))
        self.opus_aon_per_step = _opus.get("aon_per_step", 8)
        self.opus_candidates_per_step = _opus.get("candidates_per_step", 24)


def load_config(config_path: str = "config.yaml") -> Config:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict, config_path=config_path)


def parse_args():
    parser = argparse.ArgumentParser(description="1B Non-Reversible Training - ZeRO-1")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank (set by DeepSpeed)"
    )
    return parser.parse_args()


def validate_precision_policy(
    ds_config: Dict[str, Any], model_dtype: torch.dtype
) -> None:
    bf16_enabled = bool(ds_config.get("bf16", {}).get("enabled", False))
    fp16_enabled = bool(ds_config.get("fp16", {}).get("enabled", False))
    if bf16_enabled and fp16_enabled:
        raise ValueError("Invalid DeepSpeed config: both bf16 and fp16 are enabled.")
    if model_dtype == torch.bfloat16 and not bf16_enabled:
        raise ValueError("Model dtype is bfloat16 but DeepSpeed bf16 is disabled.")
    if model_dtype == torch.float16 and not fp16_enabled:
        raise ValueError("Model dtype is float16 but DeepSpeed fp16 is disabled.")


def validate_kernel_policy(require_fused_kernels: bool) -> None:
    if require_fused_kernels and not HAS_TRITON:
        raise RuntimeError(
            f"training.require_fused_kernels=true but Triton kernels are unavailable (HAS_TRITON={HAS_TRITON})."
        )


def main():
    _total_start = import_time.time()
    cmd_args = parse_args()
    args = load_config(cmd_args.config)
    if cmd_args.local_rank != -1:
        args.local_rank = cmd_args.local_rank
    set_seed(args.seed)

    _pipe_out = (
        os.path.dirname(args.metrics_jsonl_path)
        if args.metrics_jsonl_path
        else "results/run"
    )
    pipe = PipelineProfiler(
        rank=args.local_rank if args.local_rank >= 0 else 0, output_dir=_pipe_out
    )

    print_rank_0("=" * 80)
    print_rank_0("1.5B Non-Reversible Model Training — DeepSpeed ZeRO-1")
    print_rank_0("=" * 80)
    print_rank_0(f"Configuration File: {cmd_args.config}")

    # Library versions
    import einops as _einops
    import fla as _fla
    import triton as _triton

    print_rank_0(f"  PyTorch:        {torch.__version__}")
    print_rank_0(f"  Triton:         {_triton.__version__}")
    print_rank_0(f"  DeepSpeed:      {deepspeed.__version__}")
    print_rank_0(f"  FLA:            {_fla.__version__}")
    try:
        import importlib.metadata as _meta

        _fla_dist = _meta.distribution("flash-linear-attention")
        _fla_url = next(
            (u for u in (_fla_dist.metadata.get_all("Download-URL") or []) if u), None
        )
        if _fla_url is None:
            _fla_url = (
                str(_fla_dist._path) if hasattr(_fla_dist, "_path") else "unknown"
            )
        print_rank_0(f"  FLA source:     {_fla_url}")
    except Exception:
        pass
    print_rank_0(f"  Einops:         {_einops.__version__}")
    print_rank_0(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print_rank_0(f"  CUDA Devices:   {torch.cuda.device_count()}")

    print_rank_0("\nConfiguration:")
    # ── Tmpfs checkpoint redirect ────────────────────────────────────────────
    # If /dev/shm (tmpfs) is available with enough space, redirect checkpoint
    # writes there.  This eliminates NVMe contention between the S3 upload
    # thread and the dataloader workers, since uploads read from RAM instead.
    _tmpfs_path = "/dev/shm/checkpoints"
    _original_output_dir = args.output_dir
    if os.path.isdir("/dev/shm"):
        try:
            _shm_stat = os.statvfs("/dev/shm")
            _shm_free_gb = (_shm_stat.f_bavail * _shm_stat.f_frsize) / (1024**3)
            # Need ~60GB headroom: 3 checkpoints × ~19GB each
            if _shm_free_gb > 80:
                args.output_dir = _tmpfs_path
                os.makedirs(_tmpfs_path, exist_ok=True)
                print_rank_0(
                    f"  Checkpoint dir: {_tmpfs_path} (tmpfs, {_shm_free_gb:.0f}GB free)"
                )
                print_rank_0(
                    f"  Original dir:   {_original_output_dir} (NVMe fallback)"
                )
            else:
                print_rank_0(
                    f"  /dev/shm has only {_shm_free_gb:.0f}GB free — using NVMe for checkpoints"
                )
        except Exception as _e:
            print_rank_0(f"  tmpfs check failed ({_e}) — using NVMe for checkpoints")
    else:
        print_rank_0("  /dev/shm not available — using NVMe for checkpoints")

    print_rank_0("  Model: 1B Non-Reversible Dense (1.513B params)")
    print_rank_0(f"  Embedding Type: {args.embedding_type}")
    print_rank_0(f"  Dataset: {args.dataset_name}/{args.dataset_config}")
    print_rank_0(f"  Loader Type: {args.loader_type}")
    print_rank_0(f"  DeepSpeed Config: {args.deepspeed_config}")
    print_rank_0(f"  Batch Size/GPU: {args.batch_size}")
    print_rank_0(f"  Max Length: {args.max_length}")
    print_rank_0(f"  DataLoader Workers: {args.num_workers}/GPU")
    print_rank_0(f"  Epochs: {args.num_epochs}")
    print_rank_0(f"  Max Train Steps: {args.max_train_steps}")
    print_rank_0(f"  Checkpoint Interval: {args.checkpoint_interval} steps")
    print_rank_0(f"  Output Directory: {args.output_dir}")
    print_rank_0(f"  Seed: {args.seed}")
    if args.use_s3:
        print_rank_0(f"  S3: s3://{args.s3_bucket}/{args.s3_prefix}")
    if args.resume_from_checkpoint:
        print_rank_0(f"  Resume From: {args.resume_from_checkpoint}")
    print_rank_0("=" * 80)

    # ========================================
    # Step 0.5: Read DeepSpeed Config
    # ========================================
    print_rank_0("\n[0.5/5] Reading DeepSpeed configuration...")
    with pipe.stage("deepspeed_config_read"):
        with open(args.deepspeed_config, "r") as f:
            deepspeed_config = json.load(f)
        validate_kernel_policy(args.require_fused_kernels)

        micro_batch_size = deepspeed_config.get("train_micro_batch_size_per_gpu", 1)
        gradient_accumulation_steps = deepspeed_config.get(
            "gradient_accumulation_steps", 1
        )
        train_batch_size = deepspeed_config.get("train_batch_size", None)

        print_rank_0(f"  train_micro_batch_size_per_gpu: {micro_batch_size}")
        print_rank_0(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")

        import torch.distributed as dist

        num_gpus = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        expected_global_batch = (
            micro_batch_size * gradient_accumulation_steps * num_gpus
        )

        if train_batch_size is not None:
            print_rank_0(f"  train_batch_size (from config): {train_batch_size}")
            if train_batch_size != expected_global_batch:
                print_rank_0(
                    f"  WARNING: train_batch_size ({train_batch_size}) != "
                    f"micro_batch ({micro_batch_size}) x accum ({gradient_accumulation_steps}) x "
                    f"gpus ({num_gpus}) = {expected_global_batch}"
                )
        else:
            print_rank_0(f"  Calculated global batch size: {expected_global_batch}")

        zero_stage = deepspeed_config.get("zero_optimization", {}).get("stage", 0)
        print_rank_0(f"  ZeRO Stage: {zero_stage}")
        print_rank_0(f"  Using micro_batch_size_per_gpu={micro_batch_size}")

    # ========================================
    # Step 1: Load Tokenizer & Data
    # ========================================
    print_rank_0("\n[1/5] Loading tokenizer and data...")
    with pipe.stage("tokenizer_load"):
        print_rank_0("  Using TSAI 131K Tokenizer (2^17 = 131,072 tokens)")
        tokenizer = get_tokenizer()

    dataset_info = {}
    if args.loader_type == "curriculum_v2":
        with pipe.stage("data_load"):
            if not args.shard_dir:
                raise ValueError(
                    "data.loader_type=curriculum_v2 requires data.shard_dir in config."
                )
            if not args.curriculum_config_path:
                raise ValueError(
                    "data.loader_type=curriculum_v2 requires data.curriculum_config_path."
                )
            if not args.manifest_dir:
                raise ValueError(
                    "data.loader_type=curriculum_v2 requires data.manifest_dir."
                )

            import torch.distributed as _dist

            _rank, _ws = (0, 1)
            if _dist.is_available() and _dist.is_initialized():
                _rank, _ws = _dist.get_rank(), _dist.get_world_size()
            else:
                _rank = int(os.environ.get("RANK", "0"))
                _ws = int(os.environ.get("WORLD_SIZE", "1"))

            print_rank_0(
                f"  [curriculum_v2] stage={args.curriculum_stage}, mode={args.curriculum_mode}"
            )
            print_rank_0(f"  [curriculum_v2] shard_root={args.shard_dir}")
            print_rank_0(f"  [curriculum_v2] manifest_dir={args.manifest_dir}")
            print_rank_0(
                f"  [curriculum_v2] curriculum_config={args.curriculum_config_path}"
            )

            train_loader = build_curriculum_v2_dataloader(
                shard_root=args.shard_dir,
                manifest_dir=args.manifest_dir,
                curriculum_path=args.curriculum_config_path,
                stage=args.curriculum_stage,
                batch_size=micro_batch_size,
                seq_len=args.max_length,
                rank=_rank,
                world_size=_ws,
                mode=args.curriculum_mode,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            print_rank_0(
                f"  [curriculum_v2] Train loader ready (seq_len={args.max_length})"
            )

            # Curriculum doesn't have a separate eval set
            eval_loader = None
            test_loader = None
            print_rank_0(
                "  [curriculum_v2] No eval loader — validation skipped for curriculum mode."
            )

    elif args.loader_type == "bin_idx":
        with pipe.stage("data_load"):
            if not args.shard_dir:
                raise ValueError(
                    "data.loader_type=bin_idx requires data.shard_dir in config."
                )

            print_rank_0(f"  [bin_idx] Loading train shards from: {args.shard_dir}")
            train_loader = build_bin_idx_dataloader(
                shard_dir=args.shard_dir,
                batch_size=micro_batch_size,
                tokenizer=tokenizer,
                tokenizer_dir=args.tokenizer_dir,
                seq_len=args.max_length,
                num_workers=args.num_workers,
                validate_tokenizer=args.validate_tokenizer,
            )
            print_rank_0(f"  [bin_idx] Train loader ready (seq_len={args.max_length})")

            if args.eval_shard_dir:
                print_rank_0(
                    f"  [bin_idx] Loading eval shards from: {args.eval_shard_dir}"
                )
                eval_loader = build_bin_idx_dataloader(
                    shard_dir=args.eval_shard_dir,
                    batch_size=micro_batch_size,
                    tokenizer=tokenizer,
                    tokenizer_dir=args.tokenizer_dir,
                    seq_len=args.max_length,
                    num_workers=args.num_workers,
                    validate_tokenizer=args.validate_tokenizer,
                )
                test_loader = eval_loader
                print_rank_0("  [bin_idx] Eval/test loader ready")
            else:
                eval_loader = None
                test_loader = None
                print_rank_0("  [bin_idx] No eval_shard_dir — validation/test skipped.")
    else:
        with pipe.stage("data_load"):
            train_loader, eval_loader, test_loader, dataset_info = get_dataloaders(
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                tokenizer=tokenizer,
                batch_size=micro_batch_size,
                max_length=args.max_length,
                tokenized_dataset_path=args.tokenized_dataset_path,
                dataset_cache_dir=args.dataset_cache_dir,
                local_nvme_cache_dir=args.local_nvme_cache_dir,
                require_local_nvme=args.require_local_nvme,
                pack_into_blocks=args.pack_into_blocks,
                block_sizes=args.block_sizes,
                block_size_counts=args.block_size_counts,
                domain_column=args.domain_column,
                concat_across_domains=args.concat_across_domains,
                drop_remainder=args.drop_remainder,
                num_workers=args.num_workers,
                tokenize_num_proc=args.tokenize_num_proc,
            )
            print_rank_0(f"  Train batches: {len(train_loader)}")
            print_rank_0(f"  Eval batches: {len(eval_loader)}")
            print_rank_0(f"  Test batches: {len(test_loader)}")

    # ========================================
    # Step 2: Load Model
    # ========================================
    is_3b = args.model_name == "3bmoe"
    is_8b = args.model_name == "8bmoe"

    if is_8b:
        print_rank_0("\n[2/5] Loading 8B MoE Model (reversible)...")
        config = ModelConfig_8B_Rev()
        config.moe_backend = "auto"
        config.require_fused_moe_kernel = False
        config.allow_moe_vectorized_fallback = True
        config.moe_t4_enabled = args.moe_t4_enabled
        config.moe_t4_dispatcher = args.moe_t4_dispatcher
        config.moe_expert_parallel_size = args.moe_expert_parallel_size
        _KroneckerConfig = KroneckerConfig_8B
        _KroneckerEmbeddings = KroneckerEmbeddings_8B
    elif is_3b:
        print_rank_0("\n[2/5] Loading 3B MoE Model (reversible)...")
        config = ModelConfig_3B_Rev()
        config.moe_backend = "auto"
        config.require_fused_moe_kernel = False
        config.allow_moe_vectorized_fallback = True
        config.moe_t4_enabled = args.moe_t4_enabled
        config.moe_t4_dispatcher = args.moe_t4_dispatcher
        config.moe_expert_parallel_size = args.moe_expert_parallel_size
        _KroneckerConfig = KroneckerConfig_3B
        _KroneckerEmbeddings = KroneckerEmbeddings_3B
    else:
        print_rank_0("\n[2/5] Loading 1B Non-Reversible Dense Model...")
        config = ModelConfig()
        config.moe_backend = "auto"
        config.require_fused_moe_kernel = False
        config.allow_moe_vectorized_fallback = True
        config.moe_t4_enabled = False
        config.moe_t4_dispatcher = "deepep"
        config.moe_expert_parallel_size = 1
        _KroneckerConfig = KroneckerConfig
        _KroneckerEmbeddings = KroneckerEmbeddings

    vocab_size = len(tokenizer)
    config.vocab_size = vocab_size
    print_rank_0(
        f"  vocab_size: {vocab_size:,} (tokenizer.vocab_size={tokenizer.vocab_size})"
    )

    bpe_vocab = None
    pf_codec = None

    with pipe.stage("kronecker_vocab_build"):
        if args.embedding_type == "kronecker":
            print_rank_0("  Setting up Kronecker Product Embeddings (byte-level)...")
            bpe_vocab = []
            for i in range(vocab_size):
                try:
                    token = tokenizer.decode([i])
                    bpe_vocab.append(token if token else f"<unk_{i}>")
                except Exception:
                    bpe_vocab.append(f"<unk_{i}>")

            pf_config = _KroneckerConfig(
                CHAR_DIM=256,
                POS_DIM=32,
                D=8192,
                length_normalize=True,
                truncate_long_words=True,
            )
            pf_codec = _KroneckerEmbeddings(pf_config)
            print_rank_0("  Kronecker: POS_DIM=32 x CHAR_DIM=256 = D=8192")
        else:
            print_rank_0("  Using Standard Embeddings")

    with pipe.stage("model_build"):
        if is_8b:
            model = Model8B_Rev(
                config=config,
                embedding_type=args.embedding_type,
                bpe_vocab=bpe_vocab,
                pf_codec=pf_codec,
            )
        elif is_3b:
            model = Model3B_Rev(
                config=config,
                embedding_type=args.embedding_type,
                bpe_vocab=bpe_vocab,
                pf_codec=pf_codec,
            )
        else:
            model = Model1B(
                config=config,
                embedding_type=args.embedding_type,
                bpe_vocab=bpe_vocab,
                pf_codec=pf_codec,
            )

    with pipe.stage("model_to_bf16"):
        print_rank_0("  Casting model to bfloat16...")
        model = model.to(dtype=torch.bfloat16)

    if args.init_model_path:
        with pipe.stage("init_weights_load"):
            init_model_path = os.path.expanduser(args.init_model_path)
            if not os.path.exists(init_model_path):
                raise FileNotFoundError(
                    f"training.init_model_path does not exist: {init_model_path}"
                )
            print_rank_0(f"  Loading init weights from: {init_model_path}")
            init_payload = torch.load(init_model_path, map_location="cpu")
            init_state_dict = (
                init_payload["state_dict"]
                if isinstance(init_payload, dict) and "state_dict" in init_payload
                else init_payload
            )
            model.load_state_dict(init_state_dict, strict=True)
            print_rank_0("  Init weights loaded")

    validate_precision_policy(deepspeed_config, next(model.parameters()).dtype)
    print_rank_0(
        f"  Model created successfully ({sum(p.numel() for p in model.parameters()) / 1e9:.3f}B params)"
    )

    # ========================================
    # Step 3: Initialize DeepSpeed
    # ========================================
    print_rank_0("\n[3/5] Initializing DeepSpeed...")
    with pipe.stage("deepspeed_init"):
        with open(args.deepspeed_config, "r") as f:
            ds_config = json.load(f)
        model_engine, optimizer, _, _ = deepspeed.initialize(
            config_params=ds_config,
            model=model,
            model_parameters=model.parameters(),
        )
    print_rank_0(f"  ZeRO Stage: {model_engine.zero_optimization_stage()}")

    # ========================================
    # Step 3.1: Initialize LR Scheduler
    # ========================================
    lr_scheduler = None
    if args.lr_schedule_path and args.lr_stage:
        print_rank_0("\n[3.1/5] Initializing Staged LR Scheduler...")
        tokens_per_step = (
            deepspeed_config.get("train_batch_size", micro_batch_size * num_gpus)
            * args.max_length
        )
        lr_scheduler = StagedCosineScheduler(
            config_path=args.lr_schedule_path,
            stage_name=args.lr_stage,
            tokens_per_step=tokens_per_step,
        )
        # Apply weight decay from stage config to optimizer
        lr_scheduler.apply_weight_decay(model_engine.optimizer)
        # Set initial LR (step 0)
        lr_scheduler.step(model_engine.optimizer, 0)
        print_rank_0(
            f"  tokens_per_step = {tokens_per_step:,} "
            f"(batch={deepspeed_config.get('train_batch_size', 'auto')} x seq={args.max_length})"
        )
    else:
        print_rank_0(
            "\n[3.1/5] No lr_schedule_path/lr_stage — using DeepSpeed built-in scheduler"
        )

    # ========================================
    # Step 3.15: Initialize OPUS (if enabled)
    # ========================================
    opus_components = None
    opus_loader = None
    aon_loader = None

    if args.opus_enabled:
        print_rank_0("\n[3.15/5] Initializing OPUS data selection...")

        if args.loader_type != "curriculum_v2":
            raise ValueError(
                "opus.enabled=true requires data.loader_type=curriculum_v2"
            )

        from lightninglm.opus import (
            AdamWPreconditionerView,
            BenchProxyProvider,
            CountSketchProjector,
            OpusSelector,
            RandomInDistributionProxyProvider,
        )

        _opus_device = model_engine.device

        # 1. AdamW preconditioner view (read-only from optimizer state)
        opus_preconditioner = AdamWPreconditionerView(model_engine.optimizer)
        print_rank_0("  Preconditioner: AdamW view (frozen v_hat geometry)")

        # 2. CountSketch projector
        opus_sketcher = CountSketchProjector(
            sketch_dim=args.opus_sketch_dim,
            seed=args.seed,
        )
        print_rank_0(f"  Sketcher: CountSketch dim={args.opus_sketch_dim}")

        # 3. Selector (Boltzmann sampling with Gumbel-max trick)
        opus_selector = OpusSelector(
            selection_ratio=args.opus_selection_ratio,
            temperature=args.opus_temperature,
            seed=args.seed,
        )
        print_rank_0(
            f"  Selector: ρ={args.opus_selection_ratio}, τ={args.opus_temperature}"
        )

        # 4. Proxy provider
        if args.opus_proxy_type == "bench" and args.opus_proxy_token_path:
            opus_proxy = BenchProxyProvider(args.opus_proxy_token_path)
            print_rank_0(f"  Proxy: BenchProxy from {args.opus_proxy_token_path}")
        else:
            # Use in-distribution proxy from D1-D4 (same pools as candidates)
            # A separate loader with independent shard shuffling provides
            # an unbiased proxy direction without cross-contaminating the candidate stream.
            _proxy_loader = build_curriculum_v2_dataloader(
                shard_root=args.shard_dir,
                manifest_dir=args.manifest_dir,
                curriculum_path=args.curriculum_config_path,
                stage=args.curriculum_stage,
                batch_size=args.opus_n_proxy,
                seq_len=args.max_length,
                rank=_rank,
                world_size=_ws,
                mode="opus_candidates",
                seed=args.seed + 3000,  # Independent seed for proxy stream
                num_workers=2,
            )
            opus_proxy = RandomInDistributionProxyProvider(_proxy_loader)
            print_rank_0(
                f"  Proxy: RandomInDistribution (D1-D4, k={args.opus_n_proxy})"
            )

        # 5. OPUS candidate loader (D1-D4 only, large batch for scoring)
        #    Score many candidates at once for amortization:
        #    buffer = score_batch * selection_ratio, drain candidates_per_step/step
        #    → amortize over (buffer / candidates_per_step) steps
        # OPUS candidates per GPU — model-dependent (MTP off during scoring)
        # 1B dense:      36 cand → 18 sel → drain 3/step → 1:6   (peak 68.7G)
        # 3B MoE (rev):  60 cand → 30 sel → drain 3/step → 1:10  (reversible = less VRAM)
        # 8B MoE (rev):  60 cand → 30 sel → drain 3/step → 1:10
        if is_3b or is_8b:
            _opus_cand_per_gpu = 60
        else:
            _opus_cand_per_gpu = 36
        opus_loader = build_curriculum_v2_dataloader(
            shard_root=args.shard_dir,
            manifest_dir=args.manifest_dir,
            curriculum_path=args.curriculum_config_path,
            stage=args.curriculum_stage,
            batch_size=_opus_cand_per_gpu,
            seq_len=args.max_length,
            rank=_rank,
            world_size=_ws,
            mode="opus_candidates",
            seed=args.seed + 1000,  # Different seed for independent sampling
            num_workers=args.num_workers,
        )
        print_rank_0(
            f"  OPUS loader: mode=opus_candidates, batch_size={_opus_cand_per_gpu}/GPU"
        )

        # 6. AON loader (Always-ON: benchmark train + Indic guaranteed)
        aon_loader = build_curriculum_v2_dataloader(
            shard_root=args.shard_dir,
            manifest_dir=args.manifest_dir,
            curriculum_path=args.curriculum_config_path,
            stage=args.curriculum_stage,
            batch_size=args.opus_aon_per_step,
            seq_len=args.max_length,
            rank=_rank,
            world_size=_ws,
            mode="always_on",
            seed=args.seed + 2000,
            num_workers=2,
        )
        print_rank_0(
            f"  AON loader: mode=always_on, batch_size={args.opus_aon_per_step}"
        )

        # Compute amortization: how many train steps per OPUS scoring
        _selected_per_gpu = int(_opus_cand_per_gpu * args.opus_selection_ratio)
        _drain_per_step = args.opus_candidates_per_step
        _amortized_steps = max(1, _selected_per_gpu // _drain_per_step)
        _waste = _selected_per_gpu - (_amortized_steps * _drain_per_step)
        print_rank_0(
            f"  Amortization: score {_opus_cand_per_gpu}/GPU → select {_selected_per_gpu} "
            f"→ drain {_drain_per_step}/step → {_amortized_steps} steps "
            f"({_waste} waste)"
        )

        opus_components = {
            "preconditioner": opus_preconditioner,
            "sketcher": opus_sketcher,
            "selector": opus_selector,
            "proxy": opus_proxy,
            "n_proxy": args.opus_n_proxy,
            "score_seq_len": args.opus_score_seq_len,
            "score_layer_stride": args.opus_score_layer_stride,
            "candidates_per_step": args.opus_candidates_per_step,
            "aon_per_step": args.opus_aon_per_step,
        }
        print_rank_0("  OPUS initialization complete")

    # ========================================
    # Step 3.2: Initialize Observability
    # ========================================
    training_ops = None
    if args.observability_enabled:
        print_rank_0("\n[3.2/5] Initializing Observability (TrainingOps)...")
        try:
            from components import TrainingOps

            _rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            _run_id = os.environ.get("RUN_ID") or f"1b_nonrev_{int(import_time.time())}"
            training_ops = TrainingOps(
                run_id=_run_id,
                rank=_rank,
                log_dir=args.observability_log_dir,
                metrics_port=args.observability_metrics_port,
                skip_vector_check=args.observability_skip_vector_check,
                system_metrics_interval=args.observability_system_metrics_interval,
            )
            print_rank_0(f"  TrainingOps initialized (run_id={_run_id})")
        except Exception as e:
            print_rank_0(f"  TrainingOps init failed: {e}")
            print_rank_0("  Training will continue without observability.")
            training_ops = None

    # ========================================
    # Step 3.5: Initialize Checkpoint Manager
    # ========================================
    checkpoint_manager = None
    if args.use_s3:
        print_rank_0("\n[3.5/5] Initializing S3 Checkpoint Manager...")
        if not args.s3_bucket:
            raise ValueError("s3.bucket is required when s3.enabled is true")
        with pipe.stage("checkpoint_manager_init"):
            s3_config = S3Config(
                bucket_name=args.s3_bucket,
                s3_prefix=args.s3_prefix,
                region=args.s3_region,
                local_checkpoint_dir=args.output_dir,
                keep_last_n_checkpoints=args.keep_last_n_checkpoints,
                cleanup_after_upload=args.cleanup_after_upload,
            )
            checkpoint_manager = S3CheckpointManager(s3_config)
        print_rank_0("  S3 Checkpoint Manager initialized")

    # ========================================
    # Step 3.55: Initialize Spot Checkpoint Orchestrator
    # ========================================
    spot_orchestrator = None
    if args.spot_checkpoint_enabled:
        print_rank_0("\n[3.55/5] Initializing Spot Checkpoint Orchestrator...")
        # Allow overriding the metadata URL for testing
        if args.spot_metadata_url:
            import lightninglm.components.spot_checkpoint as _sc_mod

            _sc_mod.SpotTerminationListener.METADATA_URL = args.spot_metadata_url
            _sc_mod.SpotTerminationListener.TOKEN_URL = (
                args.spot_metadata_url.rsplit("/latest/", 1)[0] + "/latest/api/token"
            )
            print_rank_0(f"  Spot metadata URL override: {args.spot_metadata_url}")

        spot_orchestrator = SpotCheckpointOrchestrator(
            checkpoint_interval_seconds=args.spot_checkpoint_interval_seconds,
            s3_bucket=args.s3_bucket if args.use_s3 else None,
            s3_prefix=args.s3_prefix if args.use_s3 else "training/checkpoints",
            s3_region=args.s3_region if args.use_s3 else "us-east-1",
            local_checkpoint_dir=args.output_dir,
            metrics_jsonl_path=args.metrics_jsonl_path,
            spot_poll_interval=args.spot_poll_interval,
            log_max_bytes=args.spot_log_max_bytes,
            keep_last_n_local=args.spot_keep_last_n_local,
        )
        spot_orchestrator.start_all()
        print_rank_0(
            f"  Periodic checkpoint: every {args.spot_checkpoint_interval_seconds}s"
        )
        print_rank_0(f"  Spot listener: polling every {args.spot_poll_interval}s")
        print_rank_0("  On-demand: SIGUSR1 or Ctrl+C (x2 to abort)")
        print_rank_0(
            f"  Log rotation: every {args.spot_log_max_bytes // (1024*1024)}MB"
        )

    # ========================================
    # Step 3.6: Resume from Checkpoint
    # ========================================
    start_epoch = 0
    start_step = 0
    global_step = 0

    if args.resume_from_checkpoint:
        print_rank_0("\n[3.6/5] Resuming from checkpoint...")
        with pipe.stage("checkpoint_resume"):
            try:
                if checkpoint_manager:
                    resume_step = args.resume_step if args.resume_step else 0
                    client_state = checkpoint_manager.load_checkpoint(
                        model_engine, step=resume_step, tag=args.resume_from_checkpoint
                    )
                else:
                    from lightninglm.training.train import load_checkpoint

                    client_state = load_checkpoint(
                        model_engine, args.output_dir, tag=args.resume_from_checkpoint
                    )
                if client_state:
                    start_epoch = client_state.get("epoch", 0)
                    start_step = client_state.get("step", 0)
                    global_step = client_state.get("global_step", 0)
                    print_rank_0(
                        f"  Resumed from epoch {start_epoch}, step {start_step}, global_step {global_step}"
                    )

                    # Restore LR scheduler offset from checkpoint
                    _saved_lr_state = client_state.get("lr_scheduler_state")
                    if lr_scheduler is not None and _saved_lr_state:
                        _saved_stage = _saved_lr_state.get("stage_name")
                        _saved_offset = _saved_lr_state.get("stage_step_offset", 0)
                        if _saved_stage == lr_scheduler.stage_name:
                            # Same stage: restore the offset
                            lr_scheduler.stage_step_offset = _saved_offset
                            print_rank_0(
                                f"  LR scheduler: same stage '{_saved_stage}', offset={_saved_offset}"
                            )
                        else:
                            # Stage changed (e.g. 1B -> WU_3B): new stage starts at current global_step
                            lr_scheduler.stage_step_offset = global_step
                            print_rank_0(
                                f"  LR scheduler: stage changed '{_saved_stage}' -> '{lr_scheduler.stage_name}', "
                                f"offset set to {global_step}"
                            )
                        # Re-apply LR at the correct position
                        _resumed_lr = lr_scheduler.step(
                            model_engine.optimizer, global_step
                        )
                        print_rank_0(
                            f"  LR scheduler: resumed at global_step={global_step}, lr={_resumed_lr:.2e}"
                        )

                    # Resume curriculum dataloader shard state if available
                    _saved_shard_state = client_state.get("shard_state")
                    if (
                        _saved_shard_state
                        and hasattr(train_loader, "dataset")
                        and hasattr(train_loader.dataset, "resume_from_state")
                    ):
                        train_loader.dataset.resume_from_state(_saved_shard_state)
                        print_rank_0(
                            "  Curriculum shard state restored from checkpoint"
                        )
                else:
                    print_rank_0("  No client state found, starting fresh")
            except Exception as e:
                print_rank_0(f"  Failed to resume: {e}")
                print_rank_0("  Starting training from scratch...")

    # ========================================
    # Step 4: Training
    # ========================================
    print_rank_0("\n[4/5] Training...")
    print_rank_0(f"  Checkpoint interval: {args.checkpoint_interval} steps")
    print_rank_0(f"  Starting from epoch {start_epoch}, global step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler = dataset_info.get("train_sampler")
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        print_rank_0(f"\n{'=' * 80}")
        print_rank_0(f"Epoch {epoch + 1}/{args.num_epochs}")
        print_rank_0(f"{'=' * 80}")

        epoch_start_step = start_step if epoch == start_epoch else 0

        with pipe.stage(f"epoch_{epoch}_train"):
            avg_loss, global_step, train_stats = train_epoch(
                model_engine,
                train_loader,
                epoch,
                max_steps=args.max_train_steps,
                max_train_seconds=args.max_train_seconds,
                log_interval=args.log_interval,
                enable_system_metrics=args.enable_system_metrics,
                checkpoint_interval=args.checkpoint_interval,
                output_dir=args.output_dir,
                checkpoint_manager=checkpoint_manager,
                start_step=epoch_start_step,
                global_step=global_step,
                metrics_jsonl_path=args.metrics_jsonl_path,
                max_chunk_gb=args.max_chunk_gb,
                profile_steps=args.profile_steps if args.profile_steps else None,
                profile_output_dir=(
                    os.path.dirname(args.metrics_jsonl_path)
                    if args.metrics_jsonl_path
                    else None
                ),
                training_ops=training_ops,
                spot_orchestrator=spot_orchestrator,
                lr_scheduler=lr_scheduler,
                opus_components=opus_components,
                opus_loader=opus_loader,
                aon_loader=aon_loader,
            )

        eval_loss = None
        eval_perplexity = None
        if eval_loader is not None:
            print_rank_0("\nEvaluating on validation set...")
            with pipe.stage(f"epoch_{epoch}_eval"):
                eval_loss, eval_perplexity = evaluate(
                    model_engine,
                    eval_loader,
                    phase="Validation",
                    max_steps=args.max_eval_steps,
                    metrics_jsonl_path=args.metrics_jsonl_path,
                )
        else:
            print_rank_0("\nSkipping validation (no eval loader configured).")

        # Save epoch checkpoint
        if checkpoint_manager or args.save_checkpoint:
            epoch_tag = f"epoch{epoch}_end"
            print_rank_0("\nSaving end-of-epoch checkpoint...")
            client_state = {
                "epoch": epoch + 1,
                "step": 0,
                "global_step": global_step,
                "avg_loss": avg_loss,
                "eval_loss": eval_loss,
                "eval_perplexity": eval_perplexity,
                "lr_scheduler_state": (
                    lr_scheduler.state_dict() if lr_scheduler is not None else None
                ),
            }
            with pipe.stage(f"epoch_{epoch}_checkpoint_save"):
                if checkpoint_manager:
                    checkpoint_manager.save_checkpoint(
                        model_engine,
                        step=global_step,
                        tag=epoch_tag,
                        client_state=client_state,
                    )
                else:
                    from lightninglm.training.train import save_checkpoint

                    save_checkpoint(model_engine, args.output_dir, tag=epoch_tag)

    # ========================================
    # Step 5: Summary (Charter format)
    # ========================================
    _total_seconds = import_time.time() - _total_start
    print_rank_0("\n" + "=" * 80)
    print_rank_0("Training Complete!")
    print_rank_0("=" * 80)

    # Charter-format summary — parseable by grep
    _lr = train_stats.get("learning_rate", 0) if train_stats else 0
    _train_secs = train_stats.get("training_seconds", 0) if train_stats else 0
    _tok_s = train_stats.get("tokens_per_sec", 0) if train_stats else 0
    _n_steps = train_stats.get("num_steps", 0) if train_stats else 0
    _peak_vram = train_stats.get("peak_vram_mb", 0) if train_stats else 0
    print_rank_0("---")
    print_rank_0(f"loss:             {avg_loss:.6f}")  # NTP loss (not combined)
    print_rank_0(f"training_seconds: {_train_secs:.1f}")
    print_rank_0(f"total_seconds:    {_total_seconds:.1f}")
    print_rank_0(f"tokens/s:         {int(_tok_s)}")
    print_rank_0(f"num_steps:        {_n_steps}")
    print_rank_0(f"peak_vram_mb:     {int(_peak_vram)}")
    print_rank_0(f"LR:               {_lr:.1e}")
    print_rank_0("---")

    if args.save_checkpoint or checkpoint_manager:
        print_rank_0(f"Checkpoints: {args.output_dir}")
        if checkpoint_manager:
            print_rank_0(f"S3: s3://{args.s3_bucket}/{args.s3_prefix}")
    print_rank_0("=" * 80)

    # Wait for any pending S3 uploads before shutting down
    if checkpoint_manager is not None:
        try:
            checkpoint_manager.wait_for_uploads()
        except Exception as e:
            print_rank_0(f"wait_for_uploads error: {e}")

    # Shutdown observability
    if training_ops is not None:
        try:
            training_ops.shutdown()
        except Exception as e:
            print_rank_0(f"TrainingOps shutdown error: {e}")

    # Shutdown spot checkpoint orchestrator
    if spot_orchestrator is not None:
        try:
            spot_orchestrator.shutdown()
        except Exception as e:
            print_rank_0(f"SpotCheckpointOrchestrator shutdown error: {e}")

    pipe.write_report()
    pipe.write_jsonl()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
