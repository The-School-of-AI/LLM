# S3 Checkpoint System Documentation

## Overview

The S3 Checkpoint System provides **non-blocking checkpoint management** with automatic S3 upload for DeepSpeed training. It works seamlessly across single-GPU, multi-GPU, and multi-node training configurations.

## Key Features

✅ **Non-blocking**: Checkpoints upload to S3 in background threads  
✅ **Universal**: Auto-detects single-node vs multi-node setups  
✅ **Efficient**: One uploader per node, intelligent file distribution  
✅ **Robust**: Retry logic with exponential backoff  
✅ **Progress tracking**: Detailed logging and upload statistics  
✅ **Checkpoint management**: Automatic cleanup of old checkpoints  

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Process                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ GPU 0      │  │ GPU 1      │  │ GPU 2      │  ...   │
│  │ (Rank 0)   │  │ (Rank 1)   │  │ (Rank 2)   │        │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘        │
│         │                │                │              │
│         └────────────────┴────────────────┘              │
│                          │                               │
│              All ranks save locally                      │
│                          │                               │
│         ┌────────────────▼────────────────┐             │
│         │   Local Checkpoint Directory    │             │
│         │      ./checkpoints/step_1000/   │             │
│         └────────────────┬────────────────┘             │
│                          │                               │
│              Only local_rank=0 uploads                   │
│                          │                               │
│         ┌────────────────▼────────────────┐             │
│         │    Background Upload Thread     │             │
│         │  (Non-blocking, per-node)       │             │
│         └────────────────┬────────────────┘             │
│                          │                               │
│                          ▼                               │
│         ┌─────────────────────────────────┐             │
│         │          AWS S3 Bucket           │             │
│         │  s3://bucket/prefix/step_1000/   │             │
│         └─────────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install boto3 botocore
```

### 2. Configure AWS Credentials

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your-key-id
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_DEFAULT_REGION=us-east-1

# Option 3: IAM role (for EC2/SageMaker)
# No configuration needed - uses instance role
```

### 3. Create S3 Bucket

```bash
aws s3 mb s3://my-training-checkpoints
```

### 4. Basic Usage

```python
from aws.config import S3Config
from src.checkpoint import S3CheckpointManager

# Configure
config = S3Config(
    bucket_name='my-training-checkpoints',
    s3_prefix='experiments/my-model',
    region='us-east-1'
)

# Initialize
checkpoint_mgr = S3CheckpointManager(config)

# During training
for step in range(1000):
    # ... training code ...
    
    if step % 100 == 0:
        checkpoint_mgr.save_checkpoint(model_engine, step=step)

# Wait for uploads to complete
checkpoint_mgr.wait_for_uploads()
```

## Configuration

### S3Config Options

```python
S3Config(
    # Required
    bucket_name='my-bucket',           # S3 bucket name
    s3_prefix='training/experiment',   # S3 prefix (folder path)
    
    # Optional
    region='us-east-1',                # AWS region
    local_checkpoint_dir='./checkpoints',  # Local storage
    keep_last_n_checkpoints=3,         # Local cleanup threshold
    max_retries=3,                     # Upload retry attempts
    verbose=True,                      # Enable detailed logging
)
```

### Configuration Methods

#### Method 1: Direct Configuration

```python
config = S3Config(
    bucket_name='my-bucket',
    s3_prefix='experiments/exp-001',
    region='us-west-2',
    keep_last_n_checkpoints=5
)
```

#### Method 2: Environment Variables

```bash
export S3_BUCKET_NAME=my-bucket
export S3_PREFIX=experiments/exp-001
export S3_REGION=us-west-2
export KEEP_LAST_N_CHECKPOINTS=5
```

```python
config = S3Config.from_env()
```

#### Method 3: Preset Configurations

```python
from aws.config import get_default_config

# Development preset
config = get_default_config('development')

# Production preset
config = get_default_config('production')

# Test preset
config = get_default_config('test')
```

## Usage Patterns

### Basic Checkpointing

```python
# Save checkpoint every N steps
if step % checkpoint_interval == 0:
    checkpoint_mgr.save_checkpoint(model_engine, step=step)
```

### Checkpoint with Custom State

```python
checkpoint_mgr.save_checkpoint(
    model_engine,
    step=step,
    client_state={
        'step': step,
        'epoch': epoch,
        'best_loss': best_loss,
        'learning_rate': current_lr,
        'custom_metadata': {...}
    }
)
```

### Loading Checkpoints

```python
# Load specific checkpoint
client_state = checkpoint_mgr.load_checkpoint(model_engine, step=1000)

# Get latest checkpoint automatically
latest_step = checkpoint_mgr.get_latest_checkpoint_step()
if latest_step:
    client_state = checkpoint_mgr.load_checkpoint(model_engine, latest_step)
    start_step = client_state['step'] + 1
```

### Checkpoint Management

```python
# List available checkpoints
checkpoints = checkpoint_mgr.list_available_checkpoints()
print(f"Available: {checkpoints}")  # ['step_100', 'step_200', 'step_300']

# Get latest checkpoint step
latest = checkpoint_mgr.get_latest_checkpoint_step()
print(f"Latest: {latest}")  # 300

# Cleanup old local checkpoints
checkpoint_mgr.cleanup_old_checkpoints(keep_last_n=2)
```

## Multi-Node Training

The checkpoint system automatically detects multi-node setups and optimizes accordingly:

### Single-Node Behavior
- All checkpoint files uploaded by rank 0
- Flat S3 structure: `s3://bucket/prefix/step_100/`

### Multi-Node Behavior
- Each node uploads its own files
- Node-organized structure: `s3://bucket/prefix/step_100/node_0/`, `node_1/`, etc.
- Shared metadata uploaded by node 0 only
- Automatic file distribution to avoid duplicate uploads

### Launch Multi-Node Training

```bash
# Node 0
deepspeed --num_gpus=8 --num_nodes=4 --node_rank=0 \
    --master_addr=node0 --master_port=29500 \
    main.py --deepspeed_config deepspeed/zero-2-moe.json

# Node 1
deepspeed --num_gpus=8 --num_nodes=4 --node_rank=1 \
    --master_addr=node0 --master_port=29500 \
    main.py --deepspeed_config deepspeed/zero-2-moe.json

# ... (nodes 2, 3)
```

## Integration with Existing Code

### Minimal Integration

```python
# At the top of main.py
from aws.config import S3Config
from src.checkpoint import S3CheckpointManager

# In main() function, before training loop
s3_config = S3Config(
    bucket_name=os.getenv('S3_BUCKET_NAME', 'my-bucket'),
    s3_prefix=f'experiments/{args.experiment_name}',
    keep_last_n_checkpoints=args.keep_checkpoints
)
checkpoint_mgr = S3CheckpointManager(s3_config)

# In training loop
if step % args.checkpoint_interval == 0:
    checkpoint_mgr.save_checkpoint(model_engine, step=step)
    checkpoint_mgr.cleanup_old_checkpoints()

# After training
checkpoint_mgr.wait_for_uploads()
```

### Full Integration with train.py

```python
# Update train_epoch() signature
def train_epoch(
    model_engine,
    train_loader,
    epoch,
    checkpoint_mgr=None,  # Add this
    checkpoint_interval=100,
    max_steps=None,
    log_interval=10
):
    # ... training code ...
    
    # Add checkpointing
    if checkpoint_mgr and steps % checkpoint_interval == 0:
        checkpoint_mgr.save_checkpoint(
            model_engine,
            step=steps,
            client_state={
                'epoch': epoch,
                'step': steps,
                'loss': avg_loss
            }
        )
```

## Best Practices

### 1. Checkpoint Frequency

```python
# Too frequent: increases S3 costs and overhead
checkpoint_interval = 10  # ❌ Every 10 steps

# Recommended: balance between safety and cost
checkpoint_interval = 100  # ✅ Every 100 steps
checkpoint_interval = 500  # ✅ For large models

# Consider training time:
# If training takes 10 hours, checkpointing every 30 minutes is reasonable
steps_per_hour = total_steps / training_hours
checkpoint_interval = steps_per_hour // 2  # Every 30 minutes
```

### 2. Local Cleanup

```python
# Keep enough checkpoints for safety
keep_last_n_checkpoints = 3  # ✅ Good default

# Cleanup regularly to save disk space
if step % (checkpoint_interval * 2) == 0:
    checkpoint_mgr.cleanup_old_checkpoints()
```

### 3. Wait for Uploads

```python
# Always wait before exiting
try:
    # Training loop
    for step in range(num_steps):
        # ... training ...
        if step % checkpoint_interval == 0:
            checkpoint_mgr.save_checkpoint(model_engine, step)
finally:
    # Ensure uploads complete even if training crashes
    checkpoint_mgr.wait_for_uploads()
```

### 4. Error Handling

```python
try:
    checkpoint_mgr.save_checkpoint(model_engine, step)
except Exception as e:
    print(f"Checkpoint failed but continuing training: {e}")
    # Don't crash training due to checkpoint failures
```

### 5. Cost Optimization

```python
# Use intelligent lifecycle policies
config = S3Config(
    bucket_name='my-bucket',
    s3_prefix='experiments/exp-001',
    keep_last_n_checkpoints=2,  # Keep only 2 locally
    cleanup_after_upload=False,  # Keep local copy as backup
)

# Set S3 lifecycle rules to move old checkpoints to cheaper storage:
# - Transition to S3 Infrequent Access after 30 days
# - Transition to Glacier after 90 days
# - Delete after 1 year
```

## Monitoring

### Upload Progress

The checkpoint manager provides detailed logging:

```
[Node 0, Rank 0] 💾 Saved locally in 2.34s: step_100
[Node 0, Rank 0] 📤 Queued for S3 upload: step 100
[Node 0, Rank 0] ⬆️  Starting upload: step 100
[Node 0, Rank 0] ✅ Uploaded step 100: 156 files, 2.3GB in 45.2s (52.1 MB/s)
```

### S3 Verification

```bash
# List checkpoints in S3
aws s3 ls s3://my-bucket/experiments/exp-001/

# Check checkpoint size
aws s3 ls --summarize --human-readable --recursive \
    s3://my-bucket/experiments/exp-001/step_1000/

# Download checkpoint manually
aws s3 cp --recursive \
    s3://my-bucket/experiments/exp-001/step_1000/ \
    ./checkpoints/step_1000/
```

### CloudWatch Metrics

Monitor S3 API calls and costs in CloudWatch:
- `PutObject` requests (uploads)
- `GetObject` requests (downloads)
- `StorageBytes` (total storage)

## Troubleshooting

### Issue: "Bucket not found" error

```python
# Solution: Create bucket or check permissions
aws s3 mb s3://my-bucket
aws s3api put-bucket-versioning --bucket my-bucket --versioning-configuration Status=Enabled
```

### Issue: Slow uploads

```python
# Solution: Increase concurrency and chunk size
config = S3Config(
    bucket_name='my-bucket',
    s3_prefix='experiments/exp-001',
    multipart_threshold=50 * 1024 * 1024,  # 50MB
    multipart_chunksize=25 * 1024 * 1024,  # 25MB chunks
    max_concurrency=20  # More concurrent uploads
)
```

### Issue: Upload failures

```python
# Solution: Increase retries and backoff
config = S3Config(
    bucket_name='my-bucket',
    s3_prefix='experiments/exp-001',
    max_retries=5,
    retry_backoff_base=3  # 3^attempt seconds
)
```

### Issue: High S3 costs

```python
# Solution 1: Reduce checkpoint frequency
checkpoint_interval = 500  # Instead of 100

# Solution 2: Use S3 lifecycle policies
# Create lifecycle rule in AWS Console or CLI:
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-bucket \
    --lifecycle-configuration file://lifecycle.json

# lifecycle.json:
{
  "Rules": [{
    "Id": "archive-old-checkpoints",
    "Prefix": "experiments/",
    "Status": "Enabled",
    "Transitions": [
      {"Days": 30, "StorageClass": "STANDARD_IA"},
      {"Days": 90, "StorageClass": "GLACIER"}
    ]
  }]
}
```

## S3 Structure

### Single-Node

```
s3://bucket/prefix/
├── step_100/
│   ├── mp_rank_00_model_states.pt
│   ├── mp_rank_01_model_states.pt
│   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   ├── latest
│   └── global_step.txt
├── step_200/
└── step_300/
```

### Multi-Node

```
s3://bucket/prefix/
├── step_100/
│   ├── node_0/
│   │   ├── mp_rank_00_model_states.pt
│   │   ├── mp_rank_01_model_states.pt
│   │   ├── latest
│   │   └── global_step.txt
│   ├── node_1/
│   │   ├── mp_rank_02_model_states.pt
│   │   └── mp_rank_03_model_states.pt
│   └── node_2/
│       ├── mp_rank_04_model_states.pt
│       └── mp_rank_05_model_states.pt
```

## API Reference

### S3Config

```python
S3Config(
    bucket_name: str,              # Required: S3 bucket name
    s3_prefix: str,                # Required: S3 prefix/folder
    region: str = "us-east-1",     # AWS region
    local_checkpoint_dir: str = "./checkpoints",
    keep_last_n_checkpoints: int = 3,
    max_retries: int = 3,
    retry_backoff_base: int = 2,
    verbose: bool = True
)
```

### S3CheckpointManager

```python
# Initialization
checkpoint_mgr = S3CheckpointManager(config: S3Config)

# Save checkpoint
checkpoint_mgr.save_checkpoint(
    model_engine,                  # DeepSpeed engine
    step: int,                     # Training step
    client_state: dict = None,     # Custom state
    tag: str = None                # Custom tag (default: f"step_{step}")
)

# Load checkpoint
client_state = checkpoint_mgr.load_checkpoint(
    model_engine,
    step: int,
    tag: str = None
)

# Wait for uploads
checkpoint_mgr.wait_for_uploads()

# Cleanup
checkpoint_mgr.cleanup_old_checkpoints(keep_last_n: int = None)

# Query checkpoints
checkpoints = checkpoint_mgr.list_available_checkpoints()  # Returns list
latest_step = checkpoint_mgr.get_latest_checkpoint_step()  # Returns int or None
```

## Examples

See `examples/checkpoint_example.py` for complete working examples:
- Basic training with checkpointing
- Resume from checkpoint
- Multi-node training
- Environment variable configuration
- Custom client state

## Performance Tips

1. **Use appropriate checkpoint intervals**: Balance safety vs overhead
2. **Enable multipart uploads**: Automatically enabled for files > 100MB
3. **Use instance storage**: Store checkpoints on fast NVMe for multi-node
4. **Monitor upload queue**: Call `wait_for_uploads()` periodically
5. **Use S3 Transfer Acceleration**: For cross-region uploads
6. **Consider S3 Intelligent-Tiering**: Automatic cost optimization

## License

This checkpoint system is part of the DeepSpeed training template.
