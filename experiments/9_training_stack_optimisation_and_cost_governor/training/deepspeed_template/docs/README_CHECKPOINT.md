# S3 Non-Blocking Checkpoint System

> **Production-ready checkpoint management with automatic S3 upload for DeepSpeed training**

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install boto3 botocore

# 2. Configure AWS
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_DEFAULT_REGION=us-east-1

# 3. Run training with S3 checkpointing
deepspeed main.py \
    --deepspeed_config config/deepspeed/zero-2-moe.json \
    --s3_bucket my-training-bucket \
    --s3_prefix experiments/my-model \
    --checkpoint_interval 100
```

## ✨ Features

- ✅ **Non-blocking uploads** - Training continues while checkpoints upload to S3
- ✅ **Universal** - Auto-detects single-GPU, multi-GPU, and multi-node setups
- ✅ **Efficient** - One background thread per node, intelligent file distribution
- ✅ **Robust** - Automatic retry with exponential backoff
- ✅ **Progress tracking** - Detailed logging and upload statistics
- ✅ **Checkpoint management** - Automatic cleanup of old local checkpoints
- ✅ **Resume training** - Load from latest or specific checkpoint

## 📁 Project Structure

```
deepspeed_template/
├── config/
│   └── aws/
│       ├── __init__.py
│       └── config.py              # AWS/S3 configuration
├── src/
│   └── checkpoint.py              # S3CheckpointManager implementation
├── docs/
│   ├── CHECKPOINT_SYSTEM.md       # Detailed documentation
│   └── INTEGRATION_GUIDE.md       # Integration instructions
├── examples/
│   └── checkpoint_example.py      # Complete working examples
├── test/
│   └── test_checkpoint.py         # Unit tests
└── README_CHECKPOINT.md           # This file
```

## 🔧 Configuration

### Method 1: Command Line Arguments

```bash
deepspeed main.py \
    --s3_bucket my-training-bucket \
    --s3_prefix experiments/my-model \
    --s3_region us-east-1 \
    --checkpoint_interval 100 \
    --keep_checkpoints 3
```

### Method 2: Environment Variables

```bash
export S3_BUCKET_NAME=my-training-bucket
export S3_PREFIX=experiments/my-model
export S3_REGION=us-east-1
export KEEP_LAST_N_CHECKPOINTS=3

deepspeed main.py --deepspeed_config config/deepspeed/zero-2-moe.json
```

### Method 3: Python Code

```python
from config.aws.config import S3Config
from src.checkpoint import S3CheckpointManager

config = S3Config(
    bucket_name='my-training-bucket',
    s3_prefix='experiments/my-model',
    region='us-east-1',
    keep_last_n_checkpoints=3
)

checkpoint_mgr = S3CheckpointManager(config)
```

## 💡 Usage Examples

### Basic Training with Checkpointing

```python
from config.aws.config import S3Config
from src.checkpoint import S3CheckpointManager

# Setup
config = S3Config(
    bucket_name='my-bucket',
    s3_prefix='training/exp-001'
)
checkpoint_mgr = S3CheckpointManager(config)

# Training loop
for step in range(1000):
    # ... training code ...
    
    if step % 100 == 0:
        checkpoint_mgr.save_checkpoint(model_engine, step=step)

# Wait for uploads
checkpoint_mgr.wait_for_uploads()
```

### Resume from Latest Checkpoint

```python
# Find latest checkpoint
latest_step = checkpoint_mgr.get_latest_checkpoint_step()

if latest_step:
    # Load checkpoint
    client_state = checkpoint_mgr.load_checkpoint(model_engine, latest_step)
    start_step = client_state['step'] + 1
    print(f"Resuming from step {start_step}")
else:
    start_step = 0
    print("Starting fresh training")

# Continue training
for step in range(start_step, 1000):
    # ... training ...
```

### Save with Custom State

```python
checkpoint_mgr.save_checkpoint(
    model_engine,
    step=step,
    client_state={
        'epoch': epoch,
        'step': step,
        'best_loss': best_loss,
        'learning_rate': current_lr,
        'custom_data': {...}
    }
)
```

## 🏗️ Architecture

```
Training Process (Multi-GPU/Multi-Node)
    │
    ├─► All ranks save locally (DeepSpeed requirement)
    │   └─► ./checkpoints/step_1000/
    │
    └─► Only local_rank=0 uploads to S3 (one per node)
        └─► Background thread (non-blocking)
            └─► s3://bucket/prefix/step_1000/
```

### Single-Node Setup
- All GPUs save checkpoint locally
- Rank 0 uploads everything to S3
- Flat S3 structure: `s3://bucket/prefix/step_100/`

### Multi-Node Setup
- All GPUs save checkpoint locally
- Each node's rank 0 uploads its files
- Organized structure: `s3://bucket/prefix/step_100/node_0/`, `node_1/`, etc.
- Automatic file distribution (no duplicates)

## 📊 Performance

### Upload Performance
- **Typical throughput**: 50-100 MB/s per node
- **Non-blocking**: Training continues during upload
- **Parallel uploads**: Multiple files uploaded concurrently
- **Multi-node**: Each node uploads independently

### Storage Efficiency
- **Automatic cleanup**: Keeps only N most recent local checkpoints
- **S3 lifecycle**: Configure automatic archival/deletion
- **Incremental**: Only changed files uploaded (with proper tagging)

## 🔍 Monitoring

### Training Output

```
[Node 0, Rank 0] 💾 Saved locally in 2.34s: step_100
[Node 0, Rank 0] 📤 Queued for S3 upload: step 100
[Node 0, Rank 0] ⬆️  Starting upload: step 100
[Node 0, Rank 0] ✅ Uploaded step 100: 156 files, 2.3GB in 45.2s (52.1 MB/s)
```

### S3 Verification

```bash
# List checkpoints
aws s3 ls s3://my-bucket/experiments/my-model/

# Check checkpoint size
aws s3 ls --summarize --human-readable --recursive \
    s3://my-bucket/experiments/my-model/step_1000/

# Download checkpoint
aws s3 cp --recursive \
    s3://my-bucket/experiments/my-model/step_1000/ \
    ./checkpoints/step_1000/
```

### Python API

```python
# List available checkpoints
checkpoints = checkpoint_mgr.list_available_checkpoints()
print(f"Available: {checkpoints}")  # ['step_100', 'step_200', ...]

# Get latest checkpoint
latest = checkpoint_mgr.get_latest_checkpoint_step()
print(f"Latest: step_{latest}")  # step_300
```

## 🧪 Testing

```bash
# Run all tests
pytest test/test_checkpoint.py -v

# Run specific test
pytest test/test_checkpoint.py::test_s3_config -v

# Test with coverage
pytest test/test_checkpoint.py --cov=src.checkpoint --cov-report=html
```

## 📖 Documentation

- **[CHECKPOINT_SYSTEM.md](docs/CHECKPOINT_SYSTEM.md)** - Complete system documentation
- **[INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)** - Step-by-step integration guide
- **[checkpoint_example.py](examples/checkpoint_example.py)** - Working code examples

## 🔧 Advanced Configuration

### S3Config Options

```python
config = S3Config(
    # Required
    bucket_name='my-bucket',
    s3_prefix='training/exp',
    
    # Optional
    region='us-east-1',
    local_checkpoint_dir='./checkpoints',
    keep_last_n_checkpoints=3,
    max_retries=3,
    retry_backoff_base=2,
    
    # AWS credentials (optional - uses default chain)
    aws_access_key_id=None,
    aws_secret_access_key=None,
    
    # Performance tuning
    multipart_threshold=100 * 1024 * 1024,  # 100MB
    multipart_chunksize=50 * 1024 * 1024,   # 50MB
    max_concurrency=10,
    
    # Logging
    verbose=True,
    log_upload_progress=True
)
```

### Preset Configurations

```python
from config.aws.config import get_default_config

# Development: Verbose logging, fewer checkpoints
dev_config = get_default_config('development')

# Production: Less logging, more checkpoints
prod_config = get_default_config('production')

# Test: Minimal checkpoints
test_config = get_default_config('test')
```

## 🚨 Troubleshooting

### Issue: Slow uploads
**Solution**: Increase concurrency and chunk size
```python
config.max_concurrency = 20
config.multipart_chunksize = 25 * 1024 * 1024
```

### Issue: Upload failures
**Solution**: Increase retries
```python
config.max_retries = 5
config.retry_backoff_base = 3
```

### Issue: High S3 costs
**Solution**: Reduce checkpoint frequency and use lifecycle policies
```python
checkpoint_interval = 500  # Instead of 100
config.keep_last_n_checkpoints = 2
```

### Issue: Bucket not found
**Solution**: Create bucket with versioning
```bash
aws s3 mb s3://my-bucket
aws s3api put-bucket-versioning \
    --bucket my-bucket \
    --versioning-configuration Status=Enabled
```

## 💰 Cost Optimization

### S3 Lifecycle Policy Example

```json
{
  "Rules": [{
    "Id": "archive-old-checkpoints",
    "Prefix": "experiments/",
    "Status": "Enabled",
    "Transitions": [
      {"Days": 30, "StorageClass": "STANDARD_IA"},
      {"Days": 90, "StorageClass": "GLACIER"}
    ],
    "Expiration": {"Days": 365}
  }]
}
```

Apply with:
```bash
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-bucket \
    --lifecycle-configuration file://lifecycle.json
```

### Cost Estimation

For a model with 7B parameters (28GB checkpoint):
- **S3 Standard**: $0.023/GB/month × 28GB = ~$0.64/month per checkpoint
- **After 30 days (IA)**: $0.0125/GB/month × 28GB = ~$0.35/month
- **After 90 days (Glacier)**: $0.004/GB/month × 28GB = ~$0.11/month

## 🤝 Contributing

Issues and PRs welcome! Please see the main project README.

## 📝 License

Same as the main DeepSpeed training template project.

## 🙏 Acknowledgments

Based on best practices from:
- DeepSpeed checkpoint system
- AWS S3 best practices
- Distributed training patterns

---

**Need help?** Check the documentation:
- 📚 [Full Documentation](docs/CHECKPOINT_SYSTEM.md)
- 🔧 [Integration Guide](docs/INTEGRATION_GUIDE.md)
- 💻 [Code Examples](examples/checkpoint_example.py)
