"""
Main entry point for DeepSpeed training with 1B Model.

This script trains a 1.513B parameter dense model with:
- TSAI 131K Tokenizer (2^17 vocab, byte-level Kronecker embeddings)
- Hybrid Gated DeltaNet + Gated Sparse Attention architecture
- Reversible midpoint integration for memory efficiency
- Dense FFN (no MoE)
- 256k context length target with YARN RoPE scaling
- Memory Stream Recurrence for infinite-length documents

Supports ZeRO Stage 2/3, S3 checkpointing, and resume from checkpoint.

Usage:
    # Run with default config.yaml
    deepspeed main.py

    # Run with custom config file
    deepspeed main.py --config config/my_config.yaml

    # Multi-GPU training (specify number of GPUs)
    deepspeed --num_gpus=4 main.py

Configuration:
    Edit config.yaml to customize:
    - Dataset, batch size, epochs, max sequence length
    - Embedding type (kronecker or standard)
    - Checkpoint intervals and S3 settings
    - DeepSpeed configuration file path
    - Resume from checkpoint settings
"""

import argparse
import os
import warnings
from typing import Any, Dict

# Set CUDA allocator to use expandable segments (reduces fragmentation)
# Must be set before any CUDA operations. Prevents spurious OOM when
# reserved-but-fragmented VRAM can't satisfy large contiguous allocations.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Suppress deprecated pynvml FutureWarning emitted inside torch.cuda
warnings.filterwarnings(
    "ignore",
    message=".*pynvml package is deprecated.*",
    category=FutureWarning,
)

import json  # noqa: E402

import deepspeed  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from aws.config import S3Config  # noqa: E402
from src.checkpoint import S3CheckpointManager  # noqa: E402
from src.data import get_dataloaders, get_tokenizer  # noqa: E402
from src.models.recurrence_model_1b import (  # noqa: E402
    KroneckerConfig,
    KroneckerEmbeddings,
    Model1B,
    ModelConfig,
)
from src.prefetch_loader import PrefetchDataLoader  # noqa: E402
from src.train import evaluate, generate_text, train_epoch  # noqa: E402
from src.utils import print_rank_0, set_seed  # noqa: E402


def validate_precision_policy(ds_config: dict, model_dtype: torch.dtype):
    """Validate that DeepSpeed precision config is compatible with model dtype.

    The reversible midpoint backward recomputes forward outside any autocast
    context.  If DeepSpeed bf16/fp16 is enabled *and* the model is already
    pre-cast, the double-cast causes dtype mismatches and silent correctness bugs.
    """
    bf16_enabled = ds_config.get("bf16", {}).get("enabled", False)
    fp16_enabled = ds_config.get("fp16", {}).get("enabled", False)

    if model_dtype == torch.bfloat16 and bf16_enabled:
        raise ValueError(
            "Precision mismatch: model is pre-cast to bf16 but DeepSpeed bf16.enabled=true. "
            "This wraps forward in autocast which breaks reversible backward. "
            "Set bf16.enabled=false in your DeepSpeed config."
        )
    if model_dtype == torch.bfloat16 and fp16_enabled:
        raise ValueError(
            "Precision mismatch: model is pre-cast to bf16 but DeepSpeed fp16.enabled=true. "
            "Mixed fp16/bf16 will cause dtype errors. "
            "Set fp16.enabled=false in your DeepSpeed config."
        )
    if model_dtype == torch.float16 and bf16_enabled:
        raise ValueError(
            "Precision mismatch: model is fp16 but DeepSpeed bf16.enabled=true. "
            "Set bf16.enabled=false or cast model to bf16."
        )


class Config:
    """Configuration object that mimics argparse Namespace for compatibility."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        # Data configuration
        self.dataset_name = config_dict["data"]["dataset_name"]
        self.dataset_config = config_dict["data"]["dataset_config"]
        self.max_length = config_dict["data"]["max_length"]
        self.num_workers = config_dict["data"].get(
            "num_workers", 8
        )  # Default to 8 for p4d.24xlarge
        self.streaming = config_dict["data"].get(
            "streaming", False
        )  # Enable HF streaming for large datasets
        self.tokenized_dataset_path = config_dict["data"].get(
            "tokenized_dataset_path", None
        )

        # SPDL Configuration
        self.use_dataloader = config_dict["data"].get("use_dataloader", False)
        self.shard_dir = config_dict["data"].get("shard_dir", None)
        self.shard_manifest_path = config_dict["data"].get("shard_manifest_path", None)

        # GPU prefetch configuration
        self.prefetch_to_gpu = config_dict["data"].get("prefetch_to_gpu", False)
        self.prefetch_depth = config_dict["data"].get("prefetch_depth", 2)

        # Training configuration
        self.num_epochs = config_dict["training"]["num_epochs"]
        self.max_train_steps = config_dict["training"]["max_train_steps"]
        self.max_eval_steps = config_dict["training"]["max_eval_steps"]
        self.log_interval = config_dict["training"]["log_interval"]
        self.seed = config_dict["training"]["seed"]
        self.enable_system_metrics = config_dict["training"].get(
            "enable_system_metrics", False
        )
        self.require_fused_kernels = config_dict["training"].get(
            "require_fused_kernels", False
        )

        # DeepSpeed configuration
        self.deepspeed_config = config_dict["deepspeed"]["config_path"]
        self.local_rank = config_dict["deepspeed"]["local_rank"]

        # Load batch size from DeepSpeed config
        with open(self.deepspeed_config, "r") as f:
            deepspeed_cfg = json.load(f)
        self.batch_size = deepspeed_cfg.get("train_micro_batch_size_per_gpu", 1)

        # Model configuration
        self.embedding_type = config_dict["model"].get(
            "embedding_type", "kronecker"
        )  # "kronecker" or "standard"
        self._model_overrides = config_dict.get(
            "model", {}
        )  # Store full model dict for ModelConfig overrides

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
        self.generation_interval = config_dict["generation"].get(
            "generation_interval", None
        )  # Generate a sample every N optimizer steps (None = disabled)


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Config object with all training parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def parse_args():
    """Parse minimal command line arguments (only config file path)."""
    parser = argparse.ArgumentParser(
        description="DeepSpeed Training Template - Configuration via YAML"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by DeepSpeed launcher)",
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    # Parse command line args (only --config and --local_rank)
    cmd_args = parse_args()

    # Load configuration from YAML
    args = load_config(cmd_args.config)

    # Override local_rank if provided via command line (DeepSpeed launcher sets this)
    if cmd_args.local_rank != -1:
        args.local_rank = cmd_args.local_rank

    # Set random seed for reproducibility
    set_seed(args.seed)

    print_rank_0("=" * 80)
    print_rank_0("1B Dense Model Training - DeepSpeed + Reversible Architecture")
    print_rank_0("=" * 80)
    print_rank_0(f"Configuration File: {cmd_args.config}")
    print_rank_0(f"DeepSpeed Version: {deepspeed.__version__}")
    print_rank_0(f"PyTorch Version: {torch.__version__}")
    print_rank_0(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print_rank_0(f"CUDA Devices: {torch.cuda.device_count()}")
    print_rank_0("\nConfiguration:")
    print_rank_0("  Model: 1B Dense (1.513B params, 100% active)")
    print_rank_0("  Tokenizer: TSAI 131K (2^17 = 131,072 vocab)")
    print_rank_0(f"  Embedding Type: {args.embedding_type}")
    print_rank_0(f"  Dataset: {args.dataset_name}/{args.dataset_config}")
    print_rank_0(f"  DeepSpeed Config: {args.deepspeed_config}")
    print_rank_0(f"  Batch Size: {args.batch_size}")
    print_rank_0(f"  Max Length: {args.max_length}")
    print_rank_0(f"  DataLoader Workers: {args.num_workers} per GPU")
    print_rank_0(f"  Epochs: {args.num_epochs}")
    print_rank_0(f"  Checkpoint Interval: Every {args.checkpoint_interval} steps")
    print_rank_0(f"  Output Directory: {args.output_dir}")
    print_rank_0(f"  Random Seed: {args.seed}")
    if args.use_s3:
        print_rank_0("  S3 Enabled: Yes")
        print_rank_0(f"  S3 Bucket: {args.s3_bucket}")
        print_rank_0(f"  S3 Prefix: {args.s3_prefix}")
    if args.resume_from_checkpoint:
        print_rank_0(f"  Resume From: {args.resume_from_checkpoint}")
    print_rank_0("=" * 80)

    # ========================================
    # Step 0.5: Read DeepSpeed Config to Get Batch Size
    # ========================================
    print_rank_0("\n[0.5/5] Reading DeepSpeed configuration...")
    with open(args.deepspeed_config, "r") as f:
        deepspeed_config = json.load(f)

    # Extract batch size from DeepSpeed config
    micro_batch_size = deepspeed_config.get("train_micro_batch_size_per_gpu", 1)
    gradient_accumulation_steps = deepspeed_config.get("gradient_accumulation_steps", 1)

    print_rank_0(f"  DeepSpeed Config: {args.deepspeed_config}")
    print_rank_0(f"  train_micro_batch_size_per_gpu: {micro_batch_size}")
    print_rank_0(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")

    # Do NOT inject train_batch_size here — distributed is not initialized yet,
    # Let DeepSpeed auto-compute it:
    #   train_batch_size = micro_batch × grad_accum × world_size
    # by ensuring train_batch_size is absent from the config.
    if "train_batch_size" in deepspeed_config:
        print_rank_0(
            f"  Removing explicit train_batch_size={deepspeed_config['train_batch_size']} "
            f"from config — DeepSpeed will auto-compute it after dist init."
        )
        del deepspeed_config["train_batch_size"]

    print_rank_0(f"  Using micro_batch_size_per_gpu={micro_batch_size} for DataLoader")

    # ========================================
    # Step 1: Load Tokenizer & Data
    # ========================================
    print_rank_0("\n[1/5] Loading tokenizer and data...")
    print_rank_0("  Using TSAI 131K Tokenizer (2^17 = 131,072 tokens)")
    tokenizer = get_tokenizer()  # Loads from src/tokenizer/
    train_loader, eval_loader, test_loader, dataset_info = get_dataloaders(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        batch_size=micro_batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        tokenized_dataset_path=args.tokenized_dataset_path,
        streaming=args.streaming,
        use_spdl=args.use_dataloader,
        shard_manifest_path=args.shard_manifest_path,
    )
    is_streaming = dataset_info.get("streaming", False)
    if is_streaming:
        print_rank_0("  Streaming mode: dataset size unknown (infinite iterator)")
    else:
        print_rank_0(f"  Train batches: {len(train_loader)}")
        print_rank_0(f"  Eval batches: {len(eval_loader)}")
        print_rank_0(f"  Test batches: {len(test_loader)}")

    # ========================================
    # Step 2: Load Model (1B Dense Model)
    # ========================================
    print_rank_0("\n[2/5] Loading model...")
    print_rank_0(
        "  Creating 1B Dense Model with memory-efficient reversible architecture..."
    )

    # Create model configuration
    config = ModelConfig()

    # ── Apply YAML model overrides to ModelConfig ─────────────────────────
    # This lets YAML presets (config_4k_throughput.yaml, etc.) control
    # model behavior: GSA sparsity budget, sequence length, etc.
    _model_override_keys = [
        "max_seq_len",
        "gsa_k_base",
        "gsa_k_min",
        "gsa_k_max",
    ]
    for key in _model_override_keys:
        yaml_val = args._model_overrides.get(key)
        if yaml_val is not None:
            old_val = getattr(config, key, None)
            setattr(config, key, yaml_val)
            print_rank_0(f"  Model override: {key} = {old_val} → {yaml_val}")

    # Update vocab size to match tokenizer
    # Use len(tokenizer) to include special tokens (pad, eos, etc.)
    vocab_size = len(tokenizer)
    config.vocab_size = vocab_size
    print_rank_0(f"  Updated model vocab_size to match tokenizer: {vocab_size:,}")
    print_rank_0(
        f"    (tokenizer.vocab_size={tokenizer.vocab_size}, len(tokenizer)={len(tokenizer)})"
    )

    # Prepare vocabulary for Kronecker embeddings if needed
    bpe_vocab = None
    pf_codec = None

    if args.embedding_type == "kronecker":
        print_rank_0("  Setting up Kronecker Product Embeddings (byte-level)...")

        # Extract vocabulary words from tokenizer
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
        pf_codec = KroneckerEmbeddings(pf_config)

        print_rank_0("  Kronecker embeddings: POS_DIM=32 x CHAR_DIM=256 = D=8192")
    else:
        print_rank_0("  Using Standard Embeddings")

    # Create the 1B model
    model = Model1B(
        config=config,
        embedding_type=args.embedding_type,
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    )

    print_rank_0(
        "  Casting model to bfloat16 to avoid autocast dtype mismatches in reversible backward pass..."
    )
    model = model.to(dtype=torch.bfloat16)
    # ----------------------

    print_rank_0("  Model cast to bfloat16")
    print_rank_0("  Model created successfully")

    # ========================================
    # Step 2.5: Preflight checks
    # ========================================
    # Precision policy validation — hard-fail on mismatch
    model_dtype = next(model.parameters()).dtype
    validate_precision_policy(deepspeed_config, model_dtype)
    print_rank_0("  Precision policy validated (no autocast conflict)")

    # Kernel fail-fast — abort if required fused kernels are missing
    if args.require_fused_kernels:
        from src.models.recurrence_model_1b import HAS_FLA, HAS_TRITON

        missing = []
        if not HAS_TRITON:
            missing.append("Triton")
        if not HAS_FLA:
            missing.append("fla (flash-linear-attention)")
        if missing:
            raise RuntimeError(
                f"require_fused_kernels=true but missing: {', '.join(missing)}. "
                f"Install the required packages or set require_fused_kernels: false."
            )
        print_rank_0("  Kernel fail-fast passed: Triton and fla available")

    # ========================================
    # Step 3: Initialize DeepSpeed
    # ========================================
    print_rank_0("\n[3/5] Initializing DeepSpeed...")

    # Use the deepspeed_config dict we already loaded and patched (with computed train_batch_size)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config_params=deepspeed_config,
        model=model,
        model_parameters=model.parameters(),
    )

    print_rank_0(f"ZeRO Stage: {model_engine.zero_optimization_stage()}")

    # ========================================
    # Step 3.5: Initialize Checkpoint Manager
    # ========================================
    checkpoint_manager = None
    if args.use_s3:
        print_rank_0("\n[3.5/5] Initializing S3 Checkpoint Manager...")
        if not args.s3_bucket:
            raise ValueError("--s3_bucket is required when --use_s3 is enabled")

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
    # Step 3.6: Resume from Checkpoint
    # ========================================
    start_epoch = 0
    start_step = 0
    global_step = 0

    if args.resume_from_checkpoint:
        print_rank_0("\n[3.6/5] Resuming from checkpoint...")
        try:
            if checkpoint_manager:
                # Use S3CheckpointManager for resume
                resume_step = args.resume_step if args.resume_step else 0
                client_state = checkpoint_manager.load_checkpoint(
                    model_engine, step=resume_step, tag=args.resume_from_checkpoint
                )
            else:
                # Use local checkpoint loading
                from src.train import load_checkpoint

                client_state = load_checkpoint(
                    model_engine, args.output_dir, tag=args.resume_from_checkpoint
                )

            # Restore training state from client_state
            if client_state:
                start_epoch = client_state.get("epoch", 0)
                start_step = client_state.get("step", 0)
                global_step = client_state.get("global_step", 0)
                print_rank_0(
                    f"  ✓ Resumed from epoch {start_epoch}, step {start_step}, global_step {global_step}"
                )
            else:
                print_rank_0("  ⚠️  No client state found, starting fresh")
        except Exception as e:
            print_rank_0(f"  ❌ Failed to resume from checkpoint: {e}")
            print_rank_0("  Starting training from scratch...")

    # ========================================
    # Step 3.7: Wrap DataLoaders with GPU prefetcher
    # ========================================
    if args.prefetch_to_gpu and torch.cuda.is_available():
        device = model_engine.device
        depth = args.prefetch_depth
        print_rank_0(
            f"\n[3.7/5] Wrapping DataLoaders with PrefetchDataLoader "
            f"(device={device}, depth={depth})..."
        )
        train_loader = PrefetchDataLoader(
            train_loader, device=device, prefetch_depth=depth
        )
        eval_loader = PrefetchDataLoader(
            eval_loader, device=device, prefetch_depth=depth
        )
        test_loader = PrefetchDataLoader(
            test_loader, device=device, prefetch_depth=depth
        )

    # ========================================
    # Step 4: Training
    # ========================================
    print_rank_0("\n[4/5] Training...")
    print_rank_0(f"Checkpoint interval: Every {args.checkpoint_interval} steps")
    print_rank_0(f"Starting from epoch {start_epoch}, global step {global_step}")

    # Running total tokens processed — accumulates across epochs
    total_tokens_processed = 0

    for epoch in range(start_epoch, args.num_epochs):
        print_rank_0(f"\n{'=' * 80}")
        print_rank_0(f"Epoch {epoch + 1}/{args.num_epochs}")
        print_rank_0(f"{'=' * 80}")

        # Set epoch on DistributedSampler for deterministic shuffling per epoch
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Determine if we need to skip steps (only for first resumed epoch)
        epoch_start_step = start_step if epoch == start_epoch else 0

        # Train
        avg_loss, global_step, total_tokens_processed = train_epoch(
            model_engine,
            train_loader,
            epoch,
            max_steps=args.max_train_steps,
            log_interval=args.log_interval,
            enable_system_metrics=args.enable_system_metrics,
            checkpoint_interval=args.checkpoint_interval,
            output_dir=args.output_dir,
            checkpoint_manager=checkpoint_manager,
            start_step=epoch_start_step,
            global_step=global_step,
            shard_tracker=dataset_info.get("shard_tracker"),
            tokenizer=tokenizer,
            generation_interval=args.generation_interval,
            start_total_tokens=total_tokens_processed,
        )

        # Evaluate on validation set
        print_rank_0("\nEvaluating on validation set...")
        eval_loss, eval_perplexity = evaluate(
            model_engine,
            eval_loader,
            phase="Validation",
            max_steps=args.max_eval_steps,
            output_dir=args.output_dir,
            global_step=global_step,
        )

        # Save epoch checkpoint
        if checkpoint_manager or args.save_checkpoint:
            epoch_tag = f"epoch{epoch}_end"
            print_rank_0("\nSaving end-of-epoch checkpoint...")

            client_state = {
                "epoch": epoch + 1,  # Next epoch to start from
                "step": 0,
                "global_step": global_step,
                "avg_loss": avg_loss,
                "eval_loss": eval_loss,
                "eval_perplexity": eval_perplexity,
            }

            if checkpoint_manager:
                checkpoint_manager.save_checkpoint(
                    model_engine,
                    step=global_step,
                    tag=epoch_tag,
                    client_state=client_state,
                )
            else:
                from src.train import save_checkpoint

                save_checkpoint(model_engine, args.output_dir, tag=epoch_tag)

    # ========================================
    # Step 5: Final Evaluation and Testing
    # ========================================
    print_rank_0("\n[5/5] Final Evaluation...")

    # Evaluate on test set
    print_rank_0("\nEvaluating on test set...")
    test_loss, test_perplexity = evaluate(
        model_engine,
        test_loader,
        phase="Test",
        max_steps=args.max_eval_steps,
        output_dir=args.output_dir,
        global_step=global_step,
    )

    # Test text generation
    if args.test_generation:
        print_rank_0("\nTesting text generation...")
        generate_text(model_engine, tokenizer, prompt=args.generation_prompt)

    # Save final checkpoint
    if args.save_checkpoint or checkpoint_manager:
        print_rank_0("\nSaving final checkpoint...")

        client_state = {
            "epoch": args.num_epochs,
            "step": 0,
            "global_step": global_step,
            "test_loss": test_loss,
            "test_perplexity": test_perplexity,
            "training_complete": True,
        }

        if checkpoint_manager:
            checkpoint_manager.save_checkpoint(
                model_engine, step=global_step, tag="final", client_state=client_state
            )
            # Wait for all uploads to complete
            print_rank_0("\nWaiting for S3 uploads to complete...")
            checkpoint_manager.wait_for_uploads()

            # Cleanup old checkpoints
            if args.keep_last_n_checkpoints > 0:
                print_rank_0("\nCleaning up old checkpoints...")
                checkpoint_manager.cleanup_old_checkpoints()
        else:
            from src.train import save_checkpoint

            save_checkpoint(model_engine, args.output_dir, tag="final")

    # Summary
    print_rank_0("\n" + "=" * 80)
    print_rank_0("Training Complete!")
    print_rank_0("=" * 80)
    print_rank_0(f"Final Test Loss: {test_loss:.4f}")
    print_rank_0(f"Final Test Perplexity: {test_perplexity:.4f}")
    print_rank_0(f"Total Global Steps: {global_step}")
    if args.save_checkpoint or checkpoint_manager:
        print_rank_0(f"Checkpoint saved to: {args.output_dir}")
        if checkpoint_manager:
            print_rank_0(f"S3 Bucket: s3://{args.s3_bucket}/{args.s3_prefix}")
    print_rank_0("=" * 80)

    # Cleanup
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
