# S3 Checkpoint System - Quick Reference

## 🚀 Setup (5 minutes)

```bash
# 1. Install
pip install boto3 botocore

# 2. Configure AWS
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_DEFAULT_REGION=us-east-1

# 3. Create bucket
aws s3 mb s3://my-training-bucket

# 4. Verify setup
python scripts/verify_s3_setup.py --bucket my-training-bucket
```

## 💻 Basic Usage

### Training with S3 Checkpoints

```python
from aws.config import S3Config
from src.checkpoint import S3CheckpointManager

# Initialize
config = S3Config(
    bucket_name='my-bucket',
    s3_prefix='training/exp-001',
    region='us-east-1'
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

### Command Line - Training with Checkpoints

**Basic Training with Periodic Checkpoints:**
```bash
deepspeed main.py --deepspeed_config deepspeed/zero-2.json \
                  --checkpoint_interval 50 \
                  --output_dir ./checkpoints
```

**Training with S3 Upload:**
```bash
deepspeed main.py --deepspeed_config deepspeed/zero-2.json \
                  --use_s3 \
                  --s3_bucket my-training-bucket \
                  --s3_prefix experiments/run-1 \
                  --checkpoint_interval 100
```

**Resume from Local Checkpoint:**
```bash
deepspeed main.py --deepspeed_config deepspeed/zero-2.json \
                  --resume_from_checkpoint epoch0_step100 \
                  --output_dir ./checkpoints
```

**Resume from S3 Checkpoint:**
```bash
deepspeed main.py --deepspeed_config deepspeed/zero-2.json \
                  --use_s3 \
                  --s3_bucket my-training-bucket \
                  --s3_prefix experiments/run-1 \
                  --resume_from_checkpoint epoch0_step100 \
                  --resume_step 100
```

### Legacy/Old Format

```bash
deepspeed main.py \
    --deepspeed_config deepspeed/zero-2-moe.json \
    --s3_bucket my-training-bucket \
    --s3_prefix experiments/my-model \
    --checkpoint_interval 100
```

## 📝 Common Operations

### Save Checkpoint

```python
# Simple save
checkpoint_mgr.save_checkpoint(model_engine, step=100)

# With custom state
checkpoint_mgr.save_checkpoint(
    model_engine,
    step=100,
    client_state={'epoch': 2, 'best_loss': 0.5}
)

# Custom tag
checkpoint_mgr.save_checkpoint(
    model_engine,
    step=100,
    tag='best_model'
)
```

### Load Checkpoint

```python
# Load specific step
client_state = checkpoint_mgr.load_checkpoint(model_engine, step=100)

# Load latest
latest_step = checkpoint_mgr.get_latest_checkpoint_step()
if latest_step:
    client_state = checkpoint_mgr.load_checkpoint(model_engine, latest_step)
```

### List Checkpoints

```python
# List all available
checkpoints = checkpoint_mgr.list_available_checkpoints()
print(checkpoints)  # ['step_100', 'step_200', 'step_300']

# Get latest step
latest = checkpoint_mgr.get_latest_checkpoint_step()
print(f"Latest: {latest}")  # 300
```

### Cleanup

```python
# Cleanup old local checkpoints (keep last 3)
checkpoint_mgr.cleanup_old_checkpoints(keep_last_n=3)

# Wait for all uploads
checkpoint_mgr.wait_for_uploads()
```

## ⚙️ Configuration Options

### S3Config Parameters

```python
S3Config(
    bucket_name='my-bucket',              # Required
    s3_prefix='training/exp',             # Required
    region='us-east-1',                   # Default: us-east-1
    local_checkpoint_dir='./checkpoints', # Default: ./checkpoints
    keep_last_n_checkpoints=3,            # Default: 3
    max_retries=3,                        # Default: 3
    verbose=True                          # Default: True
)
```

### Environment Variables

```bash
export S3_BUCKET_NAME=my-bucket
export S3_PREFIX=training/exp
export S3_REGION=us-east-1
export LOCAL_CHECKPOINT_DIR=./checkpoints
export KEEP_LAST_N_CHECKPOINTS=3
```

```python
config = S3Config.from_env()
```

### Presets

```python
from aws.config import get_default_config

config = get_default_config('development')  # Verbose, fewer checkpoints
config = get_default_config('production')   # Less verbose, more checkpoints
config = get_default_config('test')         # Minimal checkpoints
```

## 🎯 Training Integration

### Update main.py

```python
# Add arguments
parser.add_argument("--s3_bucket", type=str, default=None)
parser.add_argument("--s3_prefix", type=str, default="training/checkpoints")
parser.add_argument("--checkpoint_interval", type=int, default=100)

# Initialize manager
checkpoint_mgr = None
if args.s3_bucket:
    config = S3Config(
        bucket_name=args.s3_bucket,
        s3_prefix=args.s3_prefix
    )
    checkpoint_mgr = S3CheckpointManager(config)

# Training loop
for step in range(num_steps):
    # ... training ...
    if checkpoint_mgr and step % args.checkpoint_interval == 0:
        checkpoint_mgr.save_checkpoint(model_engine, step=step)

# Cleanup
if checkpoint_mgr:
    checkpoint_mgr.wait_for_uploads()
```

## 🐛 Common Issues

### Slow Uploads

```python
config = S3Config(
    bucket_name='my-bucket',
    s3_prefix='training',
    max_concurrency=20,               # Increase
    multipart_chunksize=25*1024*1024  # Smaller chunks
)
```

### Upload Failures

```python
config = S3Config(
    bucket_name='my-bucket',
    s3_prefix='training',
    max_retries=5,          # More retries
    retry_backoff_base=3    # Longer backoff
)
```

### High Costs

```python
# Reduce checkpoint frequency
checkpoint_interval = 500  # Instead of 100

# Keep fewer local checkpoints
config.keep_last_n_checkpoints = 2

# Add S3 lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-bucket \
    --lifecycle-configuration file://lifecycle.json
```

## 🔍 Debugging

### Check S3 Upload

```bash
# List checkpoints
aws s3 ls s3://my-bucket/training/

# Check size
aws s3 ls --summarize --human-readable --recursive \
    s3://my-bucket/training/step_1000/

# Download
aws s3 cp --recursive \
    s3://my-bucket/training/step_1000/ \
    ./checkpoints/step_1000/
```

### Monitor Progress

```python
# Check active uploads
print(f"Active uploads: {len(checkpoint_mgr.active_uploads)}")

# Wait and monitor
checkpoint_mgr.wait_for_uploads()
```

### Verify Setup

```bash
python scripts/verify_s3_setup.py --bucket my-bucket --region us-east-1
```

## 📊 Performance Tips

| Tip | Code |
|-----|------|
| **Checkpoint frequency** | `checkpoint_interval = 500` (balance safety vs overhead) |
| **Local cleanup** | `cleanup_old_checkpoints(keep_last_n=2)` |
| **Concurrent uploads** | `config.max_concurrency = 20` |
| **Chunk size** | `config.multipart_chunksize = 25*1024*1024` |
| **Wait periodically** | Call `wait_for_uploads()` every few epochs |

## 🚀 Multi-Node Training

```bash
# Node 0
deepspeed --num_gpus=8 --num_nodes=4 --node_rank=0 \
    --master_addr=node0 --master_port=29500 \
    main.py --s3_bucket my-bucket

# Node 1
deepspeed --num_gpus=8 --num_nodes=4 --node_rank=1 \
    --master_addr=node0 --master_port=29500 \
    main.py --s3_bucket my-bucket

# Each node uploads its own files automatically
```

## 📖 Full Documentation

- **[CHECKPOINT_SYSTEM.md](CHECKPOINT_SYSTEM.md)** - Complete documentation
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Integration steps
- **[../examples/checkpoint_example.py](../examples/checkpoint_example.py)** - Code examples

## 🆘 Quick Help

```bash
# Test S3 setup
python scripts/verify_s3_setup.py --bucket my-bucket

# Run tests
pytest test/test_checkpoint.py -v

# Check AWS credentials
aws sts get-caller-identity

# Create S3 bucket
aws s3 mb s3://my-bucket --region us-east-1
```

## 🔗 Useful Commands

```bash
# Install dependencies
pip install boto3 botocore

# Configure AWS CLI
aws configure

# Test S3 access
aws s3 ls s3://my-bucket/

# Monitor S3 costs
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-02-01 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --filter file://s3-filter.json

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket my-bucket \
    --versioning-configuration Status=Enabled
```

---

**Questions?** See [CHECKPOINT_SYSTEM.md](CHECKPOINT_SYSTEM.md) for detailed docs.
