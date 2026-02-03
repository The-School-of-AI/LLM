# S3 Checkpoint System - Integration Guide

This guide shows how to integrate the S3 checkpoint system with your existing DeepSpeed training code.

## Quick Integration (5 minutes)

### Step 1: Add Command Line Arguments

Update `main.py` to add checkpoint-related arguments:

```python
def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Training Template")
    
    # ... existing arguments ...
    
    # Add these checkpoint arguments
    parser.add_argument(
        "--s3_bucket",
        type=str,
        default=None,
        help="S3 bucket name for checkpoint storage (enables S3 uploads if provided)"
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="training/checkpoints",
        help="S3 prefix for checkpoints"
    )
    parser.add_argument(
        "--s3_region",
        type=str,
        default="us-east-1",
        help="AWS region for S3"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--keep_checkpoints",
        type=int,
        default=3,
        help="Number of local checkpoints to keep"
    )
    
    return parser.parse_args()
```

### Step 2: Initialize Checkpoint Manager in main()

Add this after DeepSpeed initialization in `main.py`:

```python
def main():
    args = parse_args()
    
    # ... existing setup code ...
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters()
    )
    
    # ========================================
    # NEW: Initialize S3 Checkpoint Manager
    # ========================================
    checkpoint_mgr = None
    if args.s3_bucket:
        from aws.config import S3Config
        from src.checkpoint import S3CheckpointManager
        
        s3_config = S3Config(
            bucket_name=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            region=args.s3_region,
            local_checkpoint_dir=args.output_dir,
            keep_last_n_checkpoints=args.keep_checkpoints,
            verbose=True
        )
        
        checkpoint_mgr = S3CheckpointManager(s3_config)
        print_rank_0("✓ S3 Checkpoint Manager initialized")
    
    # ... rest of training code ...
```

### Step 3: Update Training Loop

Modify the training loop in `main.py`:

```python
# ========================================
# Step 4: Training
# ========================================
print_rank_0("\n[4/5] Training...")

global_step = 0  # Track global step across epochs

for epoch in range(args.num_epochs):
    print_rank_0(f"\n{'='*80}")
    print_rank_0(f"Epoch {epoch + 1}/{args.num_epochs}")
    print_rank_0(f"{'='*80}")
    
    # Train
    global_step = train_epoch(
        model_engine,
        train_loader,
        epoch,
        global_step=global_step,  # Pass global step
        checkpoint_mgr=checkpoint_mgr,  # Pass checkpoint manager
        checkpoint_interval=args.checkpoint_interval,
        max_steps=args.max_train_steps,
        log_interval=args.log_interval,
    )
    
    # Evaluate
    print_rank_0("\nEvaluating on validation set...")
    eval_loss, eval_perplexity = evaluate(
        model_engine, eval_loader, phase="Validation", max_steps=args.max_eval_steps
    )
    
    # Save epoch checkpoint
    if checkpoint_mgr:
        checkpoint_mgr.save_checkpoint(
            model_engine,
            step=global_step,
            client_state={
                'epoch': epoch + 1,
                'step': global_step,
                'eval_loss': eval_loss,
                'eval_perplexity': eval_perplexity
            },
            tag=f"epoch_{epoch + 1}"
        )

# ... existing evaluation code ...

# NEW: Wait for S3 uploads before exiting
if checkpoint_mgr:
    print_rank_0("\nWaiting for S3 uploads to complete...")
    checkpoint_mgr.wait_for_uploads()
    checkpoint_mgr.cleanup_old_checkpoints()
```

### Step 4: Update train.py

Modify `train_epoch()` to support checkpointing:

```python
def train_epoch(
    model_engine, 
    train_loader, 
    epoch, 
    global_step=0,  # NEW
    checkpoint_mgr=None,  # NEW
    checkpoint_interval=100,  # NEW
    max_steps=None, 
    log_interval=10
):
    """
    Train the model for one epoch.
    
    Args:
        model_engine: DeepSpeed model engine
        train_loader: DataLoader for training data
        epoch: Current epoch number
        global_step: Global step counter across epochs
        checkpoint_mgr: S3CheckpointManager instance (optional)
        checkpoint_interval: Save checkpoint every N steps
        max_steps: Maximum number of steps per epoch (None for full epoch)
        log_interval: Log every N steps
    
    Returns:
        Updated global_step counter
    """
    model_engine.train()
    total_loss = 0
    steps = 0
    
    progress_bar = tqdm(
        train_loader, 
        desc=f"Epoch {epoch}",
        disable=not is_main_process()
    )
    
    for i, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(model_engine.device)
        attention_mask = batch["attention_mask"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)
        
        # Forward pass
        outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        model_engine.backward(loss)
        
        # Update weights
        model_engine.step()
        
        # Track metrics
        total_loss += loss.item()
        steps += 1
        global_step += 1  # Increment global step
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Log periodically
        if i % log_interval == 0:
            print_rank_0(
                f"Epoch {epoch}, Step {i}, Global Step {global_step}, "
                f"Loss: {loss.item():.4f}"
            )
        
        # NEW: Checkpoint periodically
        if checkpoint_mgr and global_step % checkpoint_interval == 0:
            checkpoint_mgr.save_checkpoint(
                model_engine,
                step=global_step,
                client_state={
                    'epoch': epoch,
                    'step': global_step,
                    'loss': loss.item(),
                    'avg_loss': total_loss / steps
                }
            )
            
            # Cleanup old checkpoints every few saves
            if global_step % (checkpoint_interval * 2) == 0:
                checkpoint_mgr.cleanup_old_checkpoints()
        
        # Early stopping for demo/debugging
        if max_steps is not None and i >= max_steps:
            break
    
    avg_loss = total_loss / steps
    print_rank_0(f"Epoch {epoch} - Training Average Loss: {avg_loss:.4f}")
    
    return global_step  # Return updated global step
```

## Usage Examples

### Basic Usage

```bash
# Train with S3 checkpointing
deepspeed main.py \
    --deepspeed_config deepspeed/zero-2-moe.json \
    --s3_bucket my-training-checkpoints \
    --s3_prefix experiments/my-model \
    --checkpoint_interval 100 \
    --keep_checkpoints 3
```

### With Environment Variables

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_DEFAULT_REGION=us-east-1

# Train
deepspeed main.py \
    --deepspeed_config deepspeed/zero-2-moe.json \
    --s3_bucket my-training-checkpoints \
    --s3_prefix experiments/my-model
```

### Without S3 (Local Only)

```bash
# Don't specify --s3_bucket to disable S3 uploads
deepspeed main.py \
    --deepspeed_config deepspeed/zero-2-moe.json \
    --output_dir ./checkpoints
```

## Resume Training

### Option 1: Add Resume Logic to main.py

```python
def main():
    args = parse_args()
    
    # ... setup code ...
    
    # Initialize checkpoint manager
    checkpoint_mgr = None
    start_epoch = 0
    global_step = 0
    
    if args.s3_bucket:
        from aws.config import S3Config
        from src.checkpoint import S3CheckpointManager
        
        s3_config = S3Config(
            bucket_name=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            region=args.s3_region,
            local_checkpoint_dir=args.output_dir,
            keep_last_n_checkpoints=args.keep_checkpoints
        )
        
        checkpoint_mgr = S3CheckpointManager(s3_config)
        
        # Check for existing checkpoints
        if args.resume_from_latest:
            latest_step = checkpoint_mgr.get_latest_checkpoint_step()
            
            if latest_step:
                print_rank_0(f"Resuming from step {latest_step}")
                
                client_state = checkpoint_mgr.load_checkpoint(
                    model_engine,
                    step=latest_step
                )
                
                start_epoch = client_state.get('epoch', 0)
                global_step = client_state.get('step', 0)
                
                print_rank_0(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Start training from start_epoch
    for epoch in range(start_epoch, args.num_epochs):
        global_step = train_epoch(
            model_engine,
            train_loader,
            epoch,
            global_step=global_step,
            checkpoint_mgr=checkpoint_mgr,
            checkpoint_interval=args.checkpoint_interval,
            max_steps=args.max_train_steps,
            log_interval=args.log_interval,
        )
```

### Option 2: Add Resume Argument

```python
def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Training Template")
    
    # ... existing arguments ...
    
    parser.add_argument(
        "--resume_from_latest",
        action="store_true",
        help="Resume training from latest checkpoint in S3"
    )
    parser.add_argument(
        "--resume_from_step",
        type=int,
        default=None,
        help="Resume training from specific checkpoint step"
    )
    
    return parser.parse_args()
```

## Complete Modified Files

### Modified main.py (Key Changes)

```python
# At the top, add import
from aws.config import S3Config
from src.checkpoint import S3CheckpointManager

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Training Template")
    
    # ... existing arguments ...
    
    # Checkpoint arguments
    parser.add_argument("--s3_bucket", type=str, default=None)
    parser.add_argument("--s3_prefix", type=str, default="training/checkpoints")
    parser.add_argument("--s3_region", type=str, default="us-east-1")
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--keep_checkpoints", type=int, default=3)
    parser.add_argument("--resume_from_latest", action="store_true")
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # ... setup and data loading ...
    
    model = get_qwen2_moe_model(print_info=True)
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters()
    )
    
    # Initialize checkpoint manager
    checkpoint_mgr = None
    start_epoch = 0
    global_step = 0
    
    if args.s3_bucket:
        s3_config = S3Config(
            bucket_name=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            region=args.s3_region,
            local_checkpoint_dir=args.output_dir,
            keep_last_n_checkpoints=args.keep_checkpoints,
            verbose=True
        )
        
        checkpoint_mgr = S3CheckpointManager(s3_config)
        print_rank_0("✓ S3 Checkpoint Manager initialized")
        
        # Resume if requested
        if args.resume_from_latest:
            latest_step = checkpoint_mgr.get_latest_checkpoint_step()
            if latest_step:
                client_state = checkpoint_mgr.load_checkpoint(model_engine, latest_step)
                start_epoch = client_state.get('epoch', 0)
                global_step = client_state.get('step', 0)
                print_rank_0(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training loop
    print_rank_0("\n[4/5] Training...")
    for epoch in range(start_epoch, args.num_epochs):
        print_rank_0(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        global_step = train_epoch(
            model_engine,
            train_loader,
            epoch,
            global_step=global_step,
            checkpoint_mgr=checkpoint_mgr,
            checkpoint_interval=args.checkpoint_interval,
            max_steps=args.max_train_steps,
            log_interval=args.log_interval,
        )
        
        # Evaluation
        eval_loss, eval_perplexity = evaluate(
            model_engine, eval_loader, phase="Validation", max_steps=args.max_eval_steps
        )
        
        # Save epoch checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.save_checkpoint(
                model_engine,
                step=global_step,
                client_state={
                    'epoch': epoch + 1,
                    'step': global_step,
                    'eval_loss': eval_loss,
                    'eval_perplexity': eval_perplexity
                },
                tag=f"epoch_{epoch + 1}"
            )
    
    # Final evaluation
    test_loss, test_perplexity = evaluate(
        model_engine, test_loader, phase="Test", max_steps=args.max_eval_steps
    )
    
    if args.test_generation:
        generate_text(model_engine, tokenizer, prompt=args.generation_prompt)
    
    # Wait for uploads
    if checkpoint_mgr:
        print_rank_0("\nWaiting for S3 uploads...")
        checkpoint_mgr.wait_for_uploads()
        checkpoint_mgr.cleanup_old_checkpoints()
    
    print_rank_0(f"\n{'='*80}")
    print_rank_0("Training Complete!")
    print_rank_0(f"Final Test Loss: {test_loss:.4f}")
    print_rank_0(f"Final Test Perplexity: {test_perplexity:.4f}")
    print_rank_0(f"{'='*80}")

if __name__ == "__main__":
    main()
```

## Testing

### Test Local Checkpointing (No S3)

```bash
deepspeed --num_gpus=2 main.py \
    --deepspeed_config deepspeed/zero-2-moe.json \
    --num_epochs 1 \
    --max_train_steps 50
```

### Test S3 Checkpointing

```bash
# Make sure AWS credentials are set
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Run with S3
deepspeed --num_gpus=2 main.py \
    --deepspeed_config deepspeed/zero-2-moe.json \
    --s3_bucket my-test-bucket \
    --s3_prefix test/checkpoint-test \
    --checkpoint_interval 10 \
    --num_epochs 1 \
    --max_train_steps 50
```

### Verify S3 Upload

```bash
# Check S3 bucket
aws s3 ls s3://my-test-bucket/test/checkpoint-test/

# Should see directories like:
# step_10/
# step_20/
# step_30/
# ...
```

## Troubleshooting Integration

### Import Errors

```bash
# If you get import errors, make sure the package structure is correct:
pip install -e .

# Or add to PYTHONPATH:
export PYTHONPATH="${PYTHONPATH}:/path/to/deepspeed_template"
```

### Boto3 Not Found

```bash
pip install boto3 botocore
```

### S3 Permission Errors

```bash
# Check AWS credentials
aws sts get-caller-identity

# Check S3 access
aws s3 ls s3://my-bucket/

# Grant necessary permissions (S3 bucket policy):
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "s3:PutObject",
      "s3:GetObject",
      "s3:ListBucket",
      "s3:DeleteObject"
    ],
    "Resource": [
      "arn:aws:s3:::my-bucket/*",
      "arn:aws:s3:::my-bucket"
    ]
  }]
}
```

## Next Steps

After integration:

1. Test on a small dataset first
2. Monitor S3 upload progress and costs
3. Set up S3 lifecycle policies for cost optimization
4. Configure CloudWatch alerts for upload failures
5. Document your specific S3 bucket and prefix conventions

## Support

For issues or questions:
- Check `docs/CHECKPOINT_SYSTEM.md` for detailed documentation
- See `examples/checkpoint_example.py` for more examples
- Review CloudWatch logs for S3 API errors
