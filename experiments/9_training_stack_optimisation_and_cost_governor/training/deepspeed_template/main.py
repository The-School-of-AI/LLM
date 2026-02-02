"""
Main entry point for DeepSpeed training.

This script initializes the model, loads data, and runs training with DeepSpeed.
Supports both ZeRO Stage 2 and Stage 3 configurations, S3 checkpointing, and resume.

Usage:
    # Basic multi-GPU training with Stage 2
    deepspeed main.py --deepspeed_config config/deepspeed/zero-2.json
    
    # With custom settings and checkpoint interval
    deepspeed --num_gpus=4 main.py --deepspeed_config config/deepspeed/zero-2.json \
                                    --num_epochs 3 \
                                    --batch_size 16 \
                                    --checkpoint_interval 50
    
    # With S3 checkpointing (auto-uploads to S3)
    deepspeed main.py --deepspeed_config config/deepspeed/zero-2.json \
                      --use_s3 \
                      --s3_bucket my-training-bucket \
                      --s3_prefix experiments/training-run-1 \
                      --checkpoint_interval 100
    
    # Resume from local checkpoint
    deepspeed main.py --deepspeed_config config/deepspeed/zero-2.json \
                      --resume_from_checkpoint epoch0_step50
    
    # Resume from S3 checkpoint (auto-downloads from S3)
    deepspeed main.py --deepspeed_config config/deepspeed/zero-2.json \
                      --use_s3 \
                      --s3_bucket my-training-bucket \
                      --s3_prefix experiments/training-run-1 \
                      --resume_from_checkpoint epoch0_step100 \
                      --resume_step 100
    
    # S3 with cleanup (delete local checkpoints after upload)
    deepspeed main.py --deepspeed_config config/deepspeed/zero-2.json \
                      --use_s3 \
                      --s3_bucket my-training-bucket \
                      --cleanup_after_upload \
                      --keep_last_n_checkpoints 2
"""

import argparse
import os

import deepspeed
import torch
from src.checkpoint import S3CheckpointManager
from src.data import get_dataloaders, get_tokenizer
from src.model import get_model, get_qwen2_moe_model
from src.train import evaluate, generate_text, train_epoch
from src.utils import print_rank_0, set_seed
from config.aws.config import S3Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DeepSpeed Training Template")

    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Dataset name from HuggingFace datasets",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--max_length", type=int, default=128, help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximum training steps per epoch (for debugging)",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=None,
        help="Maximum evaluation steps (for debugging)",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Log every N steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # DeepSpeed arguments
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="config/deepspeed/zero-2-moe.json",
        help="Path to DeepSpeed configuration file",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--save_checkpoint",
        action="store_true",
        help="Save model checkpoint after training",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save checkpoint every N steps during training (default: 50)",
    )
    
    # Resume arguments
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint (tag name, e.g., 'step_100' or 'epoch0_step50')",
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=None,
        help="Step number to resume from (used with resume_from_checkpoint)",
    )
    
    # S3 arguments
    parser.add_argument(
        "--use_s3",
        action="store_true",
        help="Enable S3 checkpoint upload/download",
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        default=None,
        help="S3 bucket name for checkpoints",
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="training/checkpoints",
        help="S3 prefix/folder path for checkpoints",
    )
    parser.add_argument(
        "--s3_region",
        type=str,
        default="us-east-1",
        help="AWS region for S3",
    )
    parser.add_argument(
        "--cleanup_after_upload",
        action="store_true",
        help="Delete local checkpoints after successful S3 upload",
    )
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=3,
        help="Number of local checkpoints to keep",
    )

    # Generation arguments
    parser.add_argument(
        "--test_generation",
        action="store_true",
        default=True,
        help="Test text generation after training",
    )
    parser.add_argument(
        "--generation_prompt",
        type=str,
        default="The history of artificial intelligence begins with",
        help="Prompt for text generation",
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    print_rank_0("=" * 80)
    print_rank_0("DeepSpeed Training Template")
    print_rank_0("=" * 80)
    print_rank_0(f"DeepSpeed Version: {deepspeed.__version__}")
    print_rank_0(f"PyTorch Version: {torch.__version__}")
    print_rank_0(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print_rank_0(f"CUDA Devices: {torch.cuda.device_count()}")
    print_rank_0("\nConfiguration:")
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
    # Use Qwen2 tokenizer if using custom Qwen2 model, otherwise use model_name
    tokenizer_name = "Qwen/Qwen2.5-0.5B"
    # tokenizer_name = "distilgpt2"
    tokenizer = get_tokenizer(tokenizer_name)
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
