# Ideal Architecture: Multi-GPU Training with S3 → Instance Store → GPU Pipeline

## Setup
- **Single node** with multiple GPUs (**p5en.48xlarge** with **8x H200 GPUs**)
- **Data source**: dataset in S3
- **Local storage**: Instance store (8x 3.84TB NVMe SSDs = **~30TB total**)
- **Task**: Stream data from S3 → Cache on NVMe → Feed GPUs for training

### p5en.48xlarge Specifications
- **GPUs**: 8x NVIDIA H200 (141GB HBM3e each = **1.1TB total GPU memory**)
- **CPU**: 192 vCPUs (Intel Xeon Scalable, 4th gen)
- **System RAM**: **2TB DDR5**
- **Instance Storage**: 8x 3.84TB NVMe SSD (**~30TB total**)
- **Network**: **3200 Gbps EFA** (Elastic Fabric Adapter)
- **GPU-GPU**: **NVLink 4.0** at 900 GB/s per GPU
- **Cost**: ~$98/hour (on-demand), ~$30/hour (spot)

---

## The Ideal Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         AWS EC2 Instance                        │
│  (p5en.48xlarge: 8x H200 + 30TB NVMe)                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Step 1: Background Prefetch to NVMe                         ││
│  │                                                             ││
│  │   S3 Bucket (Dataset)                                       ││
│  │         ↓                                                   ││
│  │   [SPDL Pipeline: Download + Decompress]                    ││
│  │         ↓                                                   ││
│  │   /mnt/nvme/dataset_cache/  (Instance Store: 30TB)          ││
│  │   - Automatic LRU eviction                                  ││
│  │   - Keeps hot data local                                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Step 2: Multi-GPU Data Loading with DDP                     ││
│  │                                                             ││
│  │   Process 0 (H200-0) ← DistributedSampler (shard 0)         ││
│  │   Process 1 (H200-1) ← DistributedSampler (shard 1)         ││
│  │   Process 2 (H200-2) ← DistributedSampler (shard 2)         ││
│  │         ...                                                 ││
│  │   Process 7 (H200-7) ← DistributedSampler (shard 7)         ││
│  │                                                             ││
│  │   Each process:                                             ││
│  │   1. Reads from NVMe cache (ultra fast, ~60GB/s)            ││
│  │   2. Preprocesses data (tokenization)                       ││
│  │   3. Prefetches 4-6 batches ahead                           ││
│  │   4. Transfers to GPU asynchronously                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Step 3: Distributed Training (DDP)                          ││
│  │                                                              ││
│  │   H200-0: Forward → Backward → AllReduce (NVLink 4.0)      ││
│  │   H200-1: Forward → Backward → AllReduce (NVLink 4.0)      ││
│  │   H200-2: Forward → Backward → AllReduce (NVLink 4.0)      ││
│  │         ...                                                  ││
│  │   H200-7: Forward → Backward → AllReduce (NVLink 4.0)      ││
│  │                                                              ││
│  │   NCCL handles gradient synchronization (900GB/s)           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Implementation

### Step 1: Setup Instance Store (NVMe)

First, configure the EC2 instance to use NVMe storage.

#### A. Check Available NVMe Drives

```bash
# List NVMe drives
lsblk

# Example output on p5en.48xlarge:
# nvme1n1    259:0    0   3.5T  0 disk
# nvme2n1    259:1    0   3.5T  0 disk
# nvme3n1    259:2    0   3.5T  0 disk
# nvme4n1    259:3    0   3.5T  0 disk
# nvme5n1    259:4    0   3.5T  0 disk
# nvme6n1    259:5    0   3.5T  0 disk
# nvme7n1    259:6    0   3.5T  0 disk
# nvme8n1    259:7    0   3.5T  0 disk
# (8 x 3.84TB = 30.7TB total)
```

#### B. Create RAID0 for Maximum Performance

```bash
#!/bin/bash
# raid_setup.sh - Run this once when instance starts

# Install mdadm for RAID management
sudo yum install -y mdadm  # Amazon Linux
# sudo apt install -y mdadm  # Ubuntu

# Find all NVMe instance store volumes
NVME_DISKS=($(lsblk -d -n -p -o NAME,SIZE | grep nvme | grep -v nvme0n1 | awk '{print $1}'))

echo "Found ${#NVME_DISKS[@]} NVMe drives: ${NVME_DISKS[@]}"

# Create RAID0 array (stripes data across all drives for max speed)
sudo mdadm --create --verbose /dev/md0 \
    --level=0 \
    --raid-devices=${#NVME_DISKS[@]} \
    ${NVME_DISKS[@]}

# Format with XFS (better for large files than ext4)
sudo mkfs.xfs -f /dev/md0

# Create mount point
sudo mkdir -p /mnt/nvme

# Mount the RAID array
sudo mount /dev/md0 /mnt/nvme

# Set permissions
sudo chmod 777 /mnt/nvme

# Verify
df -h /mnt/nvme
```

**Expected Result:**
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/md0         30T   33M   30T   1% /mnt/nvme
```

**Performance:** RAID0 across 8 drives = **~60 GB/s read, ~30 GB/s write** (2x faster than p4d)

---

### Step 2: Intelligent Caching Strategy

Use a two-tier caching approach:

#### Option A: Mountpoint for S3 with NVMe Cache (Recommended)

AWS's Mountpoint automatically caches S3 data to local NVMe.

```bash
# Install Mountpoint for S3
wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.rpm
sudo yum install -y ./mount-s3.rpm

# Mount S3 bucket with aggressive caching
mount-s3 \
    --cache /mnt/nvme/s3-cache \
    --metadata-ttl indefinite \
    --max-threads 64 \
    data-bucket \
    /mnt/data

# Now /mnt/data looks like a local filesystem
# But data is cached to /mnt/nvme/s3-cache automatically
```

**Benefits:**
- Automatic LRU cache management
- Transparent to training code
- First access downloads from S3, subsequent accesses read from NVMe
- AWS-optimized for S3 throughput

#### Option B: Manual Caching with SPDL

If you want more control, implement custom caching:

```python
# cache_manager.py
import os
import hashlib
import boto3
from pathlib import Path
from typing import Optional

class NVMeCacheManager:
    """
    Manages local NVMe cache for S3 files
    - Downloads to NVMe on first access
    - Serves from NVMe on subsequent access
    - LRU eviction when space runs low
    """
    
    def __init__(
        self,
        cache_dir: str = '/mnt/nvme/data_cache',
        max_cache_size_gb: float = 28000,  # 28TB for 30TB drive
        min_free_gb: float = 2000  # Keep 2TB free
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size_gb * 1024**3
        self.min_free = min_free_gb * 1024**3
        self.s3_client = boto3.client('s3')
    
    def get_cache_path(self, s3_uri: str) -> Path:
        """Generate local cache path from S3 URI"""
        # Hash the S3 URI to get filename
        uri_hash = hashlib.sha256(s3_uri.encode()).hexdigest()[:16]
        # Preserve some directory structure for browsing
        bucket, key = s3_uri.replace('s3://', '').split('/', 1)
        return self.cache_dir / bucket / uri_hash
    
    def get_file(self, s3_uri: str) -> Path:
        """
        Get file from cache or download from S3
        Returns: Path to local file on NVMe
        """
        cache_path = self.get_cache_path(s3_uri)
        
        # Check if already cached
        if cache_path.exists():
            # Update access time for LRU
            cache_path.touch()
            return cache_path
        
        # Download from S3
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        bucket, key = s3_uri.replace('s3://', '').split('/', 1)
        
        print(f"Downloading {s3_uri} to cache...")
        self.s3_client.download_file(bucket, key, str(cache_path))
        
        # Evict old files if needed
        self._evict_if_needed()
        
        return cache_path
    
    def _evict_if_needed(self):
        """Evict least recently used files if cache is full"""
        # Check available space
        stat = os.statvfs(self.cache_dir)
        free_space = stat.f_bavail * stat.f_frsize
        
        if free_space < self.min_free:
            # Get all cached files sorted by access time
            cached_files = []
            for f in self.cache_dir.rglob('*'):
                if f.is_file():
                    cached_files.append((f.stat().st_atime, f.stat().st_size, f))
            
            cached_files.sort()  # Oldest access time first
            
            # Delete oldest files until we have enough space
            freed = 0
            for atime, size, filepath in cached_files:
                if free_space + freed > self.min_free:
                    break
                filepath.unlink()
                freed += size
                print(f"Evicted: {filepath.name} (freed {size/1024**3:.2f} GB)")


# Usage in data loading pipeline
cache_manager = NVMeCacheManager()

def download_and_cache(s3_uri: str) -> bytes:
    """Download from S3 with caching to NVMe"""
    local_path = cache_manager.get_file(s3_uri)
    with open(local_path, 'rb') as f:
        return f.read()
```

---

### Step 3: Multi-GPU Data Loading with DDP

#### A. Setup Distributed Training

```python
# train_distributed.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed():
    """Initialize distributed training"""
    # Get rank and world size from environment
    # (automatically set by torchrun)
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU communication
        init_method='env://'
    )
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()
```

#### B. Create Sharded Data Source

Each GPU process should only see its shard of the data.

```python
# distributed_data_source.py
from typing import Iterator

class ShardedDataSource:
    """
    S3 source that shards data across GPUs
    Each GPU gets every Nth file where N = world_size
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str,
        rank: int,
        world_size: int,
        cache_manager: NVMeCacheManager
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.rank = rank
        self.world_size = world_size
        self.cache_manager = cache_manager
        self.s3_client = boto3.client('s3')
    
    def __iter__(self) -> Iterator[str]:
        """Yield S3 URIs for this GPU's shard"""
        file_index = 0
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith(('.jsonl.gz', '.jsonl.zst')):
                    # Round-robin assignment: GPU i gets files where idx % world_size == i
                    if file_index % self.world_size == self.rank:
                        s3_uri = f"s3://{self.bucket}/{key}"
                        yield s3_uri
                    file_index += 1
```

#### C. Build Per-GPU SPDL Pipeline

```python
# gpu_pipeline.py
from spdl.dataloader import PipelineBuilder

def create_gpu_pipeline(
    bucket: str,
    prefix: str,
    rank: int,
    world_size: int,
    batch_size: int = 16,
    prefetch_factor: int = 4
):
    """
    Create SPDL pipeline for one GPU
    Each GPU has its own pipeline loading its shard
    """
    
    cache_manager = NVMeCacheManager()
    
    # Create sharded source
    source = ShardedDataSource(bucket, prefix, rank, world_size, cache_manager)
    
    def load_from_cache(s3_uri: str) -> bytes:
        """Load file from NVMe cache (or download if not cached)"""
        local_path = cache_manager.get_file(s3_uri)
        with open(local_path, 'rb') as f:
            return f.read()
    
    # Build pipeline
    pipeline = (
        PipelineBuilder()
        # Source: This GPU's shard of files
        .add_source(source)
        
        # Stage 1: Load from cache/S3
        # High concurrency OK since we're reading from NVMe (fast)
        .pipe(
            load_from_cache,
            concurrency=8,
            output_order="input"
        )
        
        # Stage 2: Decompress
        .pipe(
            decompress_and_parse,
            concurrency=4
        )
        
        # Stage 3: Tokenize
        .pipe(
            tokenize,
            concurrency=4
        )
        
        # Stage 4: Batch
        .aggregate(batch_size)
        
        # Stage 5: Collate to tensors
        .pipe(collate_fn)
        
        # Stage 6: Prefetch buffer (keep GPU fed)
        .add_sink(prefetch_factor)
        
        .build(num_threads=16)
    )
    
    return pipeline
```

---

### Step 4: Distributed Training Loop

```python
# main_train.py
import torch
from transformers import GPT2LMHeadModel
from tqdm import tqdm

def train_distributed(
    model,
    bucket: str,
    prefix: str,
    num_epochs: int = 3,
    batch_size_per_gpu: int = 16
):
    """Main distributed training function"""
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    
    # Move model to GPU
    model = model.to(local_rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Create data pipeline for this GPU
    pipeline = create_gpu_pipeline(
        bucket=bucket,
        prefix=prefix,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size_per_gpu,
        prefetch_factor=4
    )
    
    # Training loop
    try:
        with pipeline.auto_stop():
            for epoch in range(num_epochs):
                model.train()
                
                # Only rank 0 shows progress bar
                iterator = tqdm(pipeline, desc=f"Epoch {epoch+1}") if rank == 0 else pipeline
                
                for step, batch in enumerate(iterator):
                    # Move to GPU (non-blocking for overlap)
                    input_ids = batch['input_ids'].to(local_rank, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(local_rank, non_blocking=True)
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    
                    loss = outputs.loss
                    
                    # Backward pass (DDP automatically syncs gradients)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Log only from rank 0
                    if rank == 0 and step % 100 == 0:
                        print(f"Step {step}, Loss: {loss.item():.4f}")
                
                # Save checkpoint (only rank 0)
                if rank == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f'/mnt/nvme/checkpoints/checkpoint_epoch_{epoch}.pt')
                
                # Synchronize all processes
                dist.barrier()
    
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    # Model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Train
    train_distributed(
        model=model,
        bucket='data-bucket',
        prefix='data/v1_7/',
        num_epochs=3,
        batch_size_per_gpu=32  # Effective batch size = 32 * 8 = 256
                                # H200 has 141GB vs A100's 80GB
    )
```

---

### Step 5: Launch Training

```bash
# launch_training.sh

# Set environment for multi-GPU with EFA
export NCCL_DEBUG=INFO  # For debugging NCCL issues
export FI_PROVIDER=efa  # Use EFA provider (not IB like p4d)
export FI_EFA_USE_DEVICE_RDMA=1  # Enable RDMA with EFA
export NCCL_PROTO=simple  # Recommended for EFA
export NCCL_P2P_DISABLE=0  # Enable peer-to-peer GPU transfers via NVLink

# Launch with torchrun (PyTorch 1.9+)
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    main_train.py

# Alternative: Old style with torch.distributed.launch
# python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --nnodes=1 \
#     --node_rank=0 \
#     main_train.py
```

---

## Performance Optimization Tips

### 1. **NVMe RAID Configuration**

```bash
# Check RAID performance
sudo mdadm --detail /dev/md0

# Benchmark NVMe speed
sudo fio --name=seqread --rw=read --bs=1M --size=10G --numjobs=8 --filename=/mnt/nvme/test
# Expected: ~60 GB/s sequential read (8 drives)

sudo fio --name=randread --rw=randread --bs=4k --size=10G --numjobs=16 --filename=/mnt/nvme/test
# Expected: ~4M IOPS (8 drives)
```

### 2. **Prefetch Tuning**

```python
# Adjust prefetch based on bottleneck

# If GPU utilization < 90%:
prefetch_factor = 6  # More buffering (H200s are faster, need more data)

# If running out of memory:
prefetch_factor = 3  # Less buffering

# Monitor GPU utilization
watch -n 1 nvidia-smi
```

### 3. **Worker Tuning**

```python
# For NVMe cache (ultra-fast local I/O on p5en):
download_workers = 12   # More workers for 8-drive RAID
preprocess_workers = 8  # More CPU power available (192 vCPUs)

# vs. for S3 direct (network bound):
download_workers = 20  # Higher for network I/O with EFA
preprocess_workers = 6 # Moderate CPU needed
```

### 4. **Enable Mixed Precision (FP16/BF16/FP8)**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop - BF16 (recommended for H200)
with autocast(dtype=torch.bfloat16):
    outputs = model(input_ids, attention_mask, labels=input_ids)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**H200 FP8 Support (Experimental):**
```python
# H200 supports FP8 for even faster training
# Requires torch 2.4+ and transformer_engine
from transformer_engine.pytorch import fp8_autocast

with fp8_autocast(enabled=True):
    outputs = model(input_ids, attention_mask, labels=input_ids)
    loss = outputs.loss
# Can provide 2x speedup over BF16 with minimal accuracy loss
```

---

## Expected Performance

### Typical Timings (p5en.48xlarge, 8x H200):

| Metric | S3 Direct | S3 → NVMe Cache | Improvement |
|--------|-----------|-----------------|-------------|
| **First epoch** | 120 min | 120 min | None (cold cache) |
| **Second epoch** | 120 min | **25 min** | **4.8x faster** |
| **Steady state** | 120 min | **25 min** | **4.8x faster** |
| **GPU utilization** | 65-75% | **98%+** | Much better |
| **Effective throughput** | 6k samples/s | **28k samples/s** | **4.7x faster** |

**Note:** H200s are ~30% faster than A100s, so baseline is already faster than p4d.

### Cache Hit Rate:
- **Epoch 1**: 0% (downloading everything)
- **Epoch 2**: 85-95% (most data cached)
- **Epoch 3+**: 98%+ (stable cache)

### H200 vs A100 Advantages:
- **141GB vs 80GB** memory → 75% larger batches or longer sequences
- **HBM3e vs HBM2e** → 30% higher memory bandwidth (4.8 TB/s vs 3.35 TB/s)
- **FP8 support** → 2x faster training with minimal accuracy loss
- **NVLink 4.0** → 50% faster GPU-GPU communication (900 GB/s vs 600 GB/s)

---

## Key Advantages of This Architecture

✅ **Eliminates S3 latency** after first epoch (data on local NVMe)
✅ **Maximizes GPU utilization** (GPUs never wait for data)
✅ **Automatic cache management** (LRU eviction handles space)
✅ **No code changes** between epochs (transparent caching)
✅ **Perfect for iterative training** (multiple epochs over same data)
✅ **Cost effective** (instance store is free, no extra EBS costs)

---

## Alternative: Pre-download Strategy

If dataset fits entirely in instance store (~28TB usable), pre-download everything:

```bash
#!/bin/bash
# predownload_dataset.sh

# Parallel download from S3 to NVMe (use all available bandwidth)
aws s3 sync \
    s3://data-bucket/data/v1_7/ \
    /mnt/nvme/data/ \
    --only-show-errors \
    --no-progress \
    --request-payer requester  # If data requires

# Then train directly from /mnt/nvme/data/
```

**When to use:**
- Dataset < 28TB (p5en has 4x more storage than p4d!)
- Training for many epochs (10+)
- Want predictable first epoch speed
- Full Dataset is ~9TB compressed, easily fits!

---

## p5en.48xlarge Specific Optimizations

### 1. **Leverage 30TB Storage**
```python
# Can cache multiple datasets simultaneously
cache_structure = {
    '/mnt/nvme/data/': '9TB',
    '/mnt/nvme/pile/': '8TB', 
    '/mnt/nvme/c4/': '5TB',
    '/mnt/nvme/checkpoints/': '5TB',
    '/mnt/nvme/temp/': '3TB'
}
# Total: 30TB - still room to spare!
```

### 2. **Utilize 2TB System RAM**
```python
# Can keep more in memory than p4d (1.1TB)
# Enable aggressive OS caching
import os
os.system("echo 3 > /proc/sys/vm/drop_caches")  # Clear cache
os.system("echo 90 > /proc/sys/vm/vfs_cache_pressure")  # Aggressive caching
```

### 3. **EFA Network Configuration**
```bash
# Verify EFA is working
fi_info -p efa

# Test EFA bandwidth between processes
efa_test -r 0  # Run on rank 0
efa_test -r 1 -s <rank0_ip>  # Run on rank 1

# Expected: ~400 Gbps per connection
```

### 4. **H200-Specific Settings**
```python
# Enable TF32 for matrix operations (H200 optimized)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use Flash Attention 2 (optimized for H200)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'gpt2',
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
```

---

## Summary

**Ideal Setup for p5en.48xlarge (8x H200):**

1. ✅ **RAID0 across 8 NVMe drives** → 60 GB/s read, 30TB storage
2. ✅ **Intelligent cache layer** → S3 → NVMe → H200s
3. ✅ **Data sharding across 8 GPUs** → Round-robin file assignment
4. ✅ **Independent pipelines** → One per GPU process
5. ✅ **Aggressive prefetching** → 5-6 batches ahead (H200s are fast!)
6. ✅ **DDP with NVLink 4.0** → 900 GB/s gradient sync
7. ✅ **BF16 or FP8 training** → Maximize H200 performance

**Expected Result:** 98%+ GPU utilization, **~28k samples/s** throughput, **4.8x faster** than S3 direct after first epoch.

**Cost Efficiency:**
- On-demand: ~$98/hour (~$2.45/hour per H200)
- Spot: ~$30/hour (~$0.75/hour per H200) ← **Recommended**
- With spot, cost per training run is dramatically lower

**Pro Tip:** The 30TB storage means you can cache entire training corpus plus checkpoints without ever running out of space!
