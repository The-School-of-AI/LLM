"""
Main entry point for DeepSpeed training.

This script initializes the model, loads data, and runs training with DeepSpeed.
Supports both ZeRO Stage 2 and Stage 3 configurations.

Usage:
    # Multi-GPU training with Stage 2 (uses all available GPUs)
    deepspeed main.py --deepspeed_config config/deepspeed/zero-2.json
    
    # Multi-GPU training with specific number of GPUs
    deepspeed --num_gpus=4 main.py --deepspeed_config config/deepspeed/zero-2.json
    
    # Stage 3 (optimizer + parameters + gradients partitioning + CPU offload)
    deepspeed main.py --deepspeed_config config/deepspeed/zero-3.json
    
    # With custom settings
    deepspeed --num_gpus=4 main.py --deepspeed_config config/deepspeed/zero-2.json \
                                    --model_name distilgpt2 \
                                    --num_epochs 3 \
                                    --batch_size 16 \
                                    --max_length 256
    
    # Single GPU training (for testing)
    python main.py --deepspeed_config config/deepspeed/zero-2.json
"""

import argparse

import deepspeed
import torch
from src.data import get_dataloaders, get_tokenizer
from src.model import get_model, get_qwen2_moe_model
from src.train import evaluate, generate_text, save_checkpoint, train_epoch
from src.utils import print_rank_0, set_seed


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
    print_rank_0(f"  Random Seed: {args.seed}")
    print_rank_0("=" * 80)

    # ========================================
    # Step 1: Load Data
    # ========================================
    print_rank_0("\n[1/5] Loading data...")
    # Use Qwen2 tokenizer if using custom Qwen2 model, otherwise use model_name
    tokenizer_name = "Qwen/Qwen2.5-0.5B"
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
    # Step 4: Training
    # ========================================
    print_rank_0("\n[4/5] Training...")
    for epoch in range(args.num_epochs):
        print_rank_0(f"\n{'='*80}")
        print_rank_0(f"Epoch {epoch + 1}/{args.num_epochs}")
        print_rank_0(f"{'='*80}")

        # Train
        train_epoch(
            model_engine,
            train_loader,
            epoch,
            max_steps=args.max_train_steps,
            log_interval=args.log_interval,
        )

        # Evaluate on validation set
        print_rank_0("\nEvaluating on validation set...")
        eval_loss, eval_perplexity = evaluate(
            model_engine, eval_loader, phase="Validation", max_steps=args.max_eval_steps
        )

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

    # Save checkpoint
    if args.save_checkpoint:
        print_rank_0("\nSaving checkpoint...")
        save_checkpoint(model_engine, args.output_dir, tag="final")

    # Summary
    print_rank_0("\n" + "=" * 80)
    print_rank_0("Training Complete!")
    print_rank_0("=" * 80)
    print_rank_0(f"Final Test Loss: {test_loss:.4f}")
    print_rank_0(f"Final Test Perplexity: {test_perplexity:.4f}")
    if args.save_checkpoint:
        print_rank_0(f"Checkpoint saved to: {args.output_dir}")
    print_rank_0("=" * 80)

    # Cleanup
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
