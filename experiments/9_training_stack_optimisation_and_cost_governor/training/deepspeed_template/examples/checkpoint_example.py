"""
Example: Using S3CheckpointManager in Training.

This example demonstrates how to integrate the S3CheckpointManager
into your DeepSpeed training pipeline.
"""

import os
import deepspeed
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

# Import checkpoint manager and config
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from aws.config import S3Config
from src.checkpoint import S3CheckpointManager


def create_model():
    """Create a sample Qwen2 MoE model."""
    config = Qwen2Config(
        vocab_size=151936,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=1280,
        max_position_embeddings=1024,
        num_experts=8,
        num_experts_per_tok=2,
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    model = Qwen2ForCausalLM(config)
    model.gradient_checkpointing_enable()
    return model


def main():
    """Main training loop with S3 checkpointing."""
    
    # ========================================
    # Step 1: Configure S3 Checkpointing
    # ========================================
    
    # Option 1: Create config manually
    s3_config = S3Config(
        bucket_name='my-training-bucket',
        s3_prefix='moe-training/experiment-001',
        local_checkpoint_dir='./checkpoints',
        region='us-east-1',
        keep_last_n_checkpoints=3,
        verbose=True
    )
    
    # Option 2: Load from environment variables
    # s3_config = S3Config.from_env(
    #     bucket_name='my-training-bucket',  # Override specific values
    #     s3_prefix='moe-training/experiment-001'
    # )
    
    # Option 3: Use preset configuration
    # from aws.config import get_default_config
    # s3_config = get_default_config('development')
    # s3_config.bucket_name = 'my-training-bucket'
    
    # Initialize checkpoint manager
    checkpoint_mgr = S3CheckpointManager(s3_config)
    
    # ========================================
    # Step 2: Setup Model and DeepSpeed
    # ========================================
    
    model = create_model()
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="deepspeed/zero-2-moe.json",
    )
    
    global_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # ========================================
    # Step 3: Training Loop with Checkpointing
    # ========================================
    
    if global_rank == 0:
        print("\n" + "="*70)
        print("Starting Training with S3 Checkpointing")
        print("="*70 + "\n")
    
    num_steps = 1000
    checkpoint_interval = 100
    cleanup_interval = 200
    
    for step in range(1, num_steps + 1):
        # Generate dummy batch
        input_ids = torch.randint(
            0, 151936, 
            (8, 512),  # batch_size=8, seq_len=512
            device=model_engine.device
        )
        labels = input_ids.clone()
        
        # Forward pass
        outputs = model_engine(input_ids=input_ids, labels=labels)
        
        # Backward pass
        model_engine.backward(outputs.loss)
        
        # Update weights
        model_engine.step()
        
        # Log progress
        if step % 10 == 0 and global_rank == 0:
            mem = torch.cuda.memory_allocated(local_rank) / 1e9
            print(f"Step {step:4d} | Loss: {outputs.loss.item():.4f} | "
                  f"GPU Memory: {mem:.2f}GB")
        
        # Save checkpoint periodically
        if step % checkpoint_interval == 0:
            # Option 1: Simple checkpoint
            checkpoint_mgr.save_checkpoint(model_engine, step=step)
            
            # Option 2: Checkpoint with custom state
            # checkpoint_mgr.save_checkpoint(
            #     model_engine,
            #     step=step,
            #     client_state={
            #         'step': step,
            #         'epoch': step // 100,
            #         'best_loss': outputs.loss.item(),
            #         'learning_rate': optimizer.param_groups[0]['lr']
            #     }
            # )
        
        # Cleanup old checkpoints periodically
        if step % cleanup_interval == 0:
            checkpoint_mgr.cleanup_old_checkpoints()
    
    # ========================================
    # Step 4: Wait for All Uploads to Complete
    # ========================================
    
    if global_rank == 0:
        print("\n" + "="*70)
        print("Training Complete - Waiting for Uploads")
        print("="*70 + "\n")
    
    checkpoint_mgr.wait_for_uploads()
    
    # ========================================
    # Step 5: Final Cleanup
    # ========================================
    
    checkpoint_mgr.cleanup_old_checkpoints(keep_last_n=2)
    
    if global_rank == 0:
        print("\n" + "="*70)
        print("✅ All Done!")
        print("="*70)
        
        # List available checkpoints in S3
        available = checkpoint_mgr.list_available_checkpoints()
        print(f"\nCheckpoints in S3: {available}")
        
        latest_step = checkpoint_mgr.get_latest_checkpoint_step()
        print(f"Latest checkpoint: step_{latest_step}")


def example_resume_training():
    """Example: Resume training from a checkpoint."""
    
    # Setup config and checkpoint manager
    s3_config = S3Config(
        bucket_name='my-training-bucket',
        s3_prefix='moe-training/experiment-001',
        region='us-east-1'
    )
    checkpoint_mgr = S3CheckpointManager(s3_config)
    
    # Create model and DeepSpeed engine
    model = create_model()
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="deepspeed/zero-2-moe.json",
    )
    
    # Find latest checkpoint
    latest_step = checkpoint_mgr.get_latest_checkpoint_step()
    
    if latest_step is not None:
        print(f"Resuming from step {latest_step}")
        
        # Load checkpoint
        client_state = checkpoint_mgr.load_checkpoint(
            model_engine,
            step=latest_step
        )
        
        # Extract training state
        start_step = client_state.get('step', 0) + 1
        epoch = client_state.get('epoch', 0)
        
        print(f"Loaded checkpoint: step={start_step}, epoch={epoch}")
        
        # Continue training from start_step...
    else:
        print("No checkpoint found, starting fresh training")
        start_step = 0


def example_with_environment_variables():
    """
    Example: Configure using environment variables.
    
    Set these environment variables before running:
        export S3_BUCKET_NAME=my-training-bucket
        export S3_PREFIX=moe-training/experiment-001
        export S3_REGION=us-east-1
        export LOCAL_CHECKPOINT_DIR=./checkpoints
        export KEEP_LAST_N_CHECKPOINTS=3
        export AWS_ACCESS_KEY_ID=your-access-key  # Optional
        export AWS_SECRET_ACCESS_KEY=your-secret-key  # Optional
    """
    
    # Load config from environment
    s3_config = S3Config.from_env()
    
    # Override specific values if needed
    s3_config.verbose = True
    s3_config.log_upload_progress = True
    
    # Initialize checkpoint manager
    _ = S3CheckpointManager(s3_config)
    
    # Rest of training code...


if __name__ == "__main__":
    # Run main training example
    main()
    
    # Or run resume example
    # example_resume_training()
    
    # Or run with environment variables
    # example_with_environment_variables()
