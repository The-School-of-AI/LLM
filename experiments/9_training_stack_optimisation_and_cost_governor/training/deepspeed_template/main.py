"""
Main entry point for DeepSpeed training.

This script initializes the model, loads data, and runs training with DeepSpeed.
Supports both ZeRO Stage 2 and Stage 3 configurations, S3 checkpointing, and resume.

All configuration is loaded from config.yaml by default.

Usage:
    # Run with default config.yaml
    deepspeed main.py
    
    # Run with custom config file
    deepspeed main.py --config config/my_config.yaml
    
    # Multi-GPU training (specify number of GPUs)
    deepspeed --num_gpus=4 main.py

Configuration:
    Edit config.yaml to customize:
    - Dataset, batch size, epochs
    - Model and tokenizer selection
    - Checkpoint intervals and S3 settings
    - DeepSpeed configuration file path
    - Resume from checkpoint settings
"""

import argparse
import os
from typing import Any, Dict

import deepspeed
import torch
import yaml

from src.checkpoint import S3CheckpointManager
from src.data import get_dataloaders, get_tokenizer
from src.model import get_model, get_qwen2_moe_model
from src.train import evaluate, generate_text, train_epoch
from src.utils import print_rank_0, set_seed
from aws.config import S3Config


class Config:
    """Configuration object that mimics argparse Namespace for compatibility."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        # Data configuration
        self.dataset_name = config_dict['data']['dataset_name']
        self.dataset_config = config_dict['data']['dataset_config']
        self.batch_size = config_dict['data']['batch_size']
        self.max_length = config_dict['data']['max_length']
        
        # Training configuration
        self.num_epochs = config_dict['training']['num_epochs']
        self.max_train_steps = config_dict['training']['max_train_steps']
        self.max_eval_steps = config_dict['training']['max_eval_steps']
        self.log_interval = config_dict['training']['log_interval']
        self.seed = config_dict['training']['seed']
        
        # DeepSpeed configuration
        self.deepspeed_config = config_dict['deepspeed']['config_path']
        self.local_rank = config_dict['deepspeed']['local_rank']
        
        # Model configuration
        self.tokenizer_name = config_dict['model'].get('tokenizer_name', 'Qwen/Qwen2.5-0.5B')
        self.model_name = config_dict['model'].get('model_name', 'distilgpt2')
        
        # Checkpoint configuration
        self.output_dir = config_dict['checkpoint']['output_dir']
        self.save_checkpoint = config_dict['checkpoint']['save_checkpoint']
        self.checkpoint_interval = config_dict['checkpoint']['checkpoint_interval']
        self.keep_last_n_checkpoints = config_dict['checkpoint']['keep_last_n_checkpoints']
        self.resume_from_checkpoint = config_dict['checkpoint']['resume_from_checkpoint']
        self.resume_step = config_dict['checkpoint']['resume_step']
        
        # S3 configuration
        self.use_s3 = config_dict['s3']['enabled']
        self.s3_bucket = config_dict['s3']['bucket']
        self.s3_prefix = config_dict['s3']['prefix']
        self.s3_region = config_dict['s3']['region']
        self.cleanup_after_upload = config_dict['s3']['cleanup_after_upload']
        
        # Generation configuration
        self.test_generation = config_dict['generation']['test_generation']
        self.generation_prompt = config_dict['generation']['generation_prompt']


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
    
    with open(config_path, 'r') as f:
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
    print_rank_0("DeepSpeed Training Template")
    print_rank_0("=" * 80)
    print_rank_0(f"Configuration File: {cmd_args.config}")
    print_rank_0(f"DeepSpeed Version: {deepspeed.__version__}")
    print_rank_0(f"PyTorch Version: {torch.__version__}")
    print_rank_0(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print_rank_0(f"CUDA Devices: {torch.cuda.device_count()}")
    print_rank_0("\nConfiguration:")
    print_rank_0(f"  Dataset: {args.dataset_name}/{args.dataset_config}")
    print_rank_0(f"  DeepSpeed Config: {args.deepspeed_config}")
    print_rank_0(f"  Batch Size: {args.batch_size}")
    print_rank_0(f"  Max Length: {args.max_length}")
    print_rank_0(f"  Epochs: {args.num_epochs}")
    print_rank_0(f"  Checkpoint Interval: Every {args.checkpoint_interval} steps")
    print_rank_0(f"  Output Directory: {args.output_dir}")
    print_rank_0(f"  Random Seed: {args.seed}")
    if args.use_s3:
        print_rank_0(f"  S3 Enabled: Yes")
        print_rank_0(f"  S3 Bucket: {args.s3_bucket}")
        print_rank_0(f"  S3 Prefix: {args.s3_prefix}")
    if args.resume_from_checkpoint:
        print_rank_0(f"  Resume From: {args.resume_from_checkpoint}")
    print_rank_0("=" * 80)

    # ========================================
    # Step 1: Load Data
    # ========================================
    print_rank_0("\n[1/5] Loading data...")
    tokenizer = get_tokenizer(args.tokenizer_name)
    train_loader, eval_loader, test_loader, _ = get_dataloaders(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print_rank_0(f"  Train batches: {len(train_loader)}")
    print_rank_0(f"  Eval batches: {len(eval_loader)}")
    print_rank_0(f"  Test batches: {len(test_loader)}")

    # ========================================
    # Step 2: Load Model
    # ========================================
    print_rank_0("\n[2/5] Loading model...")
    model = get_qwen2_moe_model(print_info=True)
    # model = get_model(args.model_name, print_info=True)

    # ========================================
    # Step 3: Initialize DeepSpeed
    # ========================================
    print_rank_0("\n[3/5] Initializing DeepSpeed...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters()
    )
    print_rank_0("  DeepSpeed engine initialized")
    print_rank_0(f"  Device: {model_engine.device}")
    print_rank_0(f"  Global Rank: {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}")
    print_rank_0(f"  World Size: {torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1}")

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
                    model_engine,
                    step=resume_step,
                    tag=args.resume_from_checkpoint
                )
            else:
                # Use local checkpoint loading
                from src.train import load_checkpoint
                client_state = load_checkpoint(
                    model_engine,
                    args.output_dir,
                    tag=args.resume_from_checkpoint
                )
            
            # Restore training state from client_state
            if client_state:
                start_epoch = client_state.get('epoch', 0)
                start_step = client_state.get('step', 0)
                global_step = client_state.get('global_step', 0)
                print_rank_0(f"  ✓ Resumed from epoch {start_epoch}, step {start_step}, global_step {global_step}")
            else:
                print_rank_0("  ⚠️  No client state found, starting fresh")
        except Exception as e:
            print_rank_0(f"  ❌ Failed to resume from checkpoint: {e}")
            print_rank_0("  Starting training from scratch...")

    # ========================================
    # Step 4: Training
    # ========================================
    print_rank_0("\n[4/5] Training...")
    print_rank_0(f"Checkpoint interval: Every {args.checkpoint_interval} steps")
    print_rank_0(f"Starting from epoch {start_epoch}, global step {global_step}")
    
    for epoch in range(start_epoch, args.num_epochs):
        print_rank_0(f"\n{'='*80}")
        print_rank_0(f"Epoch {epoch + 1}/{args.num_epochs}")
        print_rank_0(f"{'='*80}")

        # Determine if we need to skip steps (only for first resumed epoch)
        epoch_start_step = start_step if epoch == start_epoch else 0

        # Train
        avg_loss, global_step = train_epoch(
            model_engine,
            train_loader,
            epoch,
            max_steps=args.max_train_steps,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            output_dir=args.output_dir,
            checkpoint_manager=checkpoint_manager,
            start_step=epoch_start_step,
            global_step=global_step,
        )

        # Evaluate on validation set
        print_rank_0("\nEvaluating on validation set...")
        eval_loss, eval_perplexity = evaluate(
            model_engine, eval_loader, phase="Validation", max_steps=args.max_eval_steps
        )
        
        # Save epoch checkpoint
        if checkpoint_manager or args.save_checkpoint:
            epoch_tag = f"epoch{epoch}_end"
            print_rank_0(f"\nSaving end-of-epoch checkpoint...")
            
            client_state = {
                'epoch': epoch + 1,  # Next epoch to start from
                'step': 0,
                'global_step': global_step,
                'avg_loss': avg_loss,
                'eval_loss': eval_loss,
                'eval_perplexity': eval_perplexity,
            }
            
            if checkpoint_manager:
                checkpoint_manager.save_checkpoint(
                    model_engine,
                    step=global_step,
                    tag=epoch_tag,
                    client_state=client_state
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
        model_engine, test_loader, phase="Test", max_steps=args.max_eval_steps
    )

    # Test text generation
    if args.test_generation:
        print_rank_0("\nTesting text generation...")
        generate_text(model_engine, tokenizer, prompt=args.generation_prompt)

    # Save final checkpoint
    if args.save_checkpoint or checkpoint_manager:
        print_rank_0("\nSaving final checkpoint...")
        
        client_state = {
            'epoch': args.num_epochs,
            'step': 0,
            'global_step': global_step,
            'test_loss': test_loss,
            'test_perplexity': test_perplexity,
            'training_complete': True,
        }
        
        if checkpoint_manager:
            checkpoint_manager.save_checkpoint(
                model_engine,
                step=global_step,
                tag="final",
                client_state=client_state
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
