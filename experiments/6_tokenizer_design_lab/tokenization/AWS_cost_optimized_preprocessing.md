# AWS Cost-Optimized Preprocessing Guide

## Overview

This guide covers how to tokenize and shard 4 TB of multi-source training data on AWS at minimal cost. The preprocessing is a **one-time CPU-bound job** — no GPUs needed.

---

## Cost Summary (TL;DR)

| Approach | Instance | $/hr | Hours | Compute | Storage | S3 | **Total** |
|----------|----------|------|-------|---------|---------|-----|-----------|
| **⭐ Recommended** | c7i.24xlarge **Spot** | $1.22 | ~12 | $15 | $11 | $0 | **~$26** |
| ARM alternative | c7g.16xlarge **Spot** | $0.70 | ~16 | $11 | $11 | $0 | **~$22** |
| Safe (no interruption) | c7i.24xlarge **On-Demand** | $4.08 | ~12 | $49 | $11 | $0 | **~$60** |
| Auto-scaled | AWS Batch + Spot fleet | ~$2.50 | ~14 | $35 | $15 | $0 | **~$50** |

**Compare to tokenizing during training on P5en.48xlarge: ~$2,800+ wasted per run.**

---

## Instance Selection

### Why CPU-Only (No GPU Needed)?

Tokenization is **pure CPU work** (BPE encoding). GPUs are useless for this task:

```
Tokenization speed per core:
  BPE (Rust-based, HuggingFace fast tokenizer): ~100K-200K tokens/sec/core
  
On c7i.24xlarge (96 vCPUs):
  ~96 × 100K = ~9.6 million tokens/sec
  
For 2 trillion tokens (4 TB raw → ~2T tokens after tokenization):
  2T / 9.6M = ~208,000 seconds ≈ ~58 hours single-threaded
  With 90 workers (leaving 6 for I/O): 58 / 90 ≈ ~0.64 hours for tokenization
  
  BUT: I/O overhead, parquet parsing, and memory management add ~10× overhead
  Realistic estimate: ~8-12 hours total
```

### Recommended Instance: c7i.24xlarge Spot

| Spec | Value |
|------|-------|
| **vCPUs** | 96 (Intel 4th Gen Xeon) |
| **RAM** | 192 GB |
| **Network** | 37.5 Gbps (for S3 downloads) |
| **On-Demand price** | ~$4.08/hr |
| **Spot price** | ~$1.22/hr (70% savings) |
| **Interruption frequency** | Low (<5% for c7i family) |

### Why Not Smaller Instances?

```
c7i.24xlarge (96 vCPU):  12 hours × $1.22/hr = $14.64  ← CHEAPEST
c7i.12xlarge (48 vCPU):  24 hours × $0.61/hr = $14.64  ← Same cost, 2× longer
c7i.4xlarge  (16 vCPU):  72 hours × $0.20/hr = $14.40  ← Same cost, 6× longer, higher spot risk

Rule: Larger instance = faster = less time for spot interruption = same cost
      → Always pick the largest instance that spot offers
```

### Alternative: Graviton (ARM) for Extra Savings

```
c7g.16xlarge (64 vCPU ARM):
  Spot: ~$0.70/hr
  Time: ~16 hours
  Cost: $11.20

  Caveat: Ensure all Python dependencies support ARM
  (numpy, pyarrow, transformers — all support ARM ✅)
```

---

## Storage Strategy

### Option A: EBS gp3 (Recommended)

```
Raw data:        4 TB  (downloaded from S3)
Temporary files: 4 TB  (tokenized .npy files)
Final shards:    4 TB  (uniform .npy shards)
Headroom:        2 TB
─────────────────────
Total EBS:       14 TB needed → round to 16 TB

Cost: 16 TB × $0.08/GB/month × (12 hours / 720 hours)
    = 16,000 × $0.08 × 0.017
    = ~$22

Performance:
  gp3 baseline:  3,000 IOPS, 125 MB/s
  gp3 provisioned: 16,000 IOPS, 1,000 MB/s (+$6/hr for max throughput)
  
  For our workload: baseline is fine (CPU is the bottleneck, not disk)
```

### Option B: Instance Store NVMe (If Available)

Some instance types include free NVMe storage:

```
i3en.24xlarge:  8 × 7.5 TB NVMe = 60 TB (free!)
  On-Demand: $10.85/hr
  Spot:      ~$3.25/hr
  12 hours × $3.25 = $39 (no EBS cost)
  
c7id.24xlarge: 2 × 1.9 TB NVMe = 3.8 TB (free but tight)
  May need EBS for overflow
```

**Verdict**: Stick with c7i.24xlarge + gp3 EBS. Simpler and cheaper overall.

### Option C: Stream Processing (No Local Storage)

```
S3 → Process in streaming mode → Write directly to S3

Pro:  No EBS needed ($0 storage cost)
Con:  Much more complex, harder to resume, S3 PUT costs add up
      Also slower — random S3 reads for re-sharding phase

Not recommended for 4 TB scale.
```

---

## Network & S3 Transfer Costs

```
S3 → EC2 (same region):  FREE ✅  (no data transfer charge)
EC2 → S3 (upload):       FREE ✅  (uploads to S3 are free)
S3 storage:              $0.023/GB/month

Raw data on S3:     4 TB × $0.023/GB = ~$92/month
Final shards on S3: 4 TB × $0.023/GB = ~$92/month

S3 API costs:
  GET requests (downloading raw data):  ~$0.40 per 1M requests
  PUT requests (uploading shards):      ~$5.00 per 1M requests
  Estimated: ~$2 for our volume

IMPORTANT: Launch in the SAME REGION as your S3 bucket!
```

---

## Step-by-Step Execution Plan

### Step 0: Prepare the Preprocessing Script

Upload the preprocessing code to S3 or put it in a Docker image:

```bash
# Pack the scripts
cd /path/to/deepspeed_template
tar -czf preprocess-scripts.tar.gz scripts/preprocess/ scripts/preprocess_data.py requirements-preprocess.txt

# Upload to S3
aws s3 cp preprocess-scripts.tar.gz s3://your-bucket/code/preprocess-scripts.tar.gz
```

### Step 1: Launch Spot Instance

```bash
# Request a persistent spot instance (auto-restarts after interruption)
aws ec2 run-instances \
  --instance-type c7i.24xlarge \
  --instance-market-options '{
    "MarketType": "spot",
    "SpotOptions": {
      "SpotInstanceType": "persistent",
      "InstanceInterruptionBehavior": "stop"
    }
  }' \
  --block-device-mappings '[{
    "DeviceName": "/dev/xvda",
    "Ebs": {
      "VolumeSize": 16000,
      "VolumeType": "gp3",
      "Iops": 6000,
      "Throughput": 500,
      "DeleteOnTermination": true
    }
  }]' \
  --image-id ami-0abcdef1234567890 \
  --key-name your-key \
  --iam-instance-profile Name=your-s3-access-role \
  --region us-east-1 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=preprocess-tokenize}]'
```

**Key settings:**
- `SpotInstanceType: persistent` — AWS will restart the instance after interruption
- `InstanceInterruptionBehavior: stop` — stop (not terminate!) on interruption so EBS data persists
- `DeleteOnTermination: true` — auto-cleanup EBS when done
- **IAM role** — must have S3 read/write access

### Step 2: Setup Environment

```bash
#!/bin/bash
# Run on the instance after SSH

# System packages
sudo yum update -y
sudo yum install -y python3.11 python3.11-pip git htop tmux

# Python packages
pip3.11 install numpy pyarrow transformers tokenizers boto3 tqdm

# Download preprocessing code
aws s3 cp s3://your-bucket/code/preprocess-scripts.tar.gz /home/ec2-user/
cd /home/ec2-user && tar -xzf preprocess-scripts.tar.gz

# Create data directories
sudo mkdir -p /data/{raw,tmp,shards}
sudo chown -R ec2-user:ec2-user /data
```

### Step 3: Download Raw Data from S3

```bash
# Run in tmux (survives SSH disconnect)
tmux new -s download

# Parallel download from S3 (use all 37.5 Gbps bandwidth)
aws s3 sync s3://your-bucket/dolma/     /data/raw/dolma/     --only-show-errors &
aws s3 sync s3://your-bucket/sangraha/  /data/raw/sangraha/  --only-show-errors &
aws s3 sync s3://your-bucket/ncert/     /data/raw/ncert/     --only-show-errors &
aws s3 sync s3://your-bucket/indicnlp/  /data/raw/indicnlp/  --only-show-errors &
wait

echo "Download complete!"
# Time: ~30 min for 4 TB on 37.5 Gbps connection
```

### Step 4: Run Preprocessing

```bash
# Run in tmux (survives SSH disconnect AND spot interruption recovery)
tmux new -s preprocess

python3.11 scripts/preprocess_data.py \
  --raw-dir /data/raw \
  --tmp-dir /data/tmp \
  --output-dir /data/shards \
  --tokenizer "Qwen/Qwen2-7B" \
  --shard-size 500000000 \
  --workers 90 \
  --progress-file /data/progress.json \
  --mix-ratios "dolma:0.55,sangraha-verified:0.20,sangraha-unverified:0.10,indicnlp:0.10,ncert:0.03,sangraha-synthetic:0.02" \
  2>&1 | tee /data/preprocess.log

# Time: ~8-12 hours
# The script saves progress after each file → safe to resume after interruption
```

### Step 5: Upload Final Shards to S3

```bash
# Upload shards as they're created (or after completion)
aws s3 sync /data/shards/ s3://your-bucket/training-shards/ \
  --only-show-errors \
  --storage-class STANDARD

echo "Upload complete!"
# Time: ~30 min for 4 TB
```

### Step 6: Verify and Cleanup

```bash
# Verify shard count
aws s3 ls s3://your-bucket/training-shards/ | wc -l
# Expected: ~4000 shards (2T tokens / 500M per shard)

# Verify a random shard
python3.11 -c "
import numpy as np
data = np.load('/data/shards/shard-00042.npy')
print(f'Shape: {data.shape}, dtype: {data.dtype}')
print(f'Tokens: {len(data):,}')
print(f'First 20: {data[:20]}')
print(f'Size: {data.nbytes / 1e9:.2f} GB')
"

# Terminate instance (auto-deletes EBS)
# Do NOT do this until you've verified the S3 upload!
aws ec2 terminate-instances --instance-ids i-0123456789abcdef0
```

---

## Handling Spot Interruptions

The preprocessing script is designed to handle spot interruptions gracefully:

```
Spot Interruption → AWS sends 2-min warning → Instance STOPS (not terminates)
                                                     ↓
                                              EBS volume PERSISTS
                                                     ↓
                                        AWS restarts instance when capacity returns
                                                     ↓
                                     Script reads progress.json → skips completed files
                                                     ↓
                                              Resumes from where it left off

Key mechanisms:
  1. progress.json tracks completed files (saved after each file)
  2. Atomic writes (tmp + rename) prevent corrupted output
  3. EBS persists across stop/start cycles
  4. Persistent spot type auto-restarts the instance
```

### User Data Script (Auto-Resume on Restart)

Add this as the instance's **user data** to auto-resume after spot restart:

```bash
#!/bin/bash
# This runs automatically when the instance starts/restarts

# Check if preprocessing was in progress
if [ -f /data/progress.json ]; then
    echo "Resuming preprocessing after spot interruption..."
    cd /home/ec2-user
    
    # Re-activate environment and resume
    nohup python3.11 scripts/preprocess_data.py \
      --raw-dir /data/raw \
      --tmp-dir /data/tmp \
      --output-dir /data/shards \
      --tokenizer "Qwen/Qwen2-7B" \
      --shard-size 500000000 \
      --workers 90 \
      --progress-file /data/progress.json \
      --mix-ratios "dolma:0.55,sangraha-verified:0.20,sangraha-unverified:0.10,indicnlp:0.10,ncert:0.03,sangraha-synthetic:0.02" \
      >> /data/preprocess.log 2>&1 &
fi
```

---

## Cost Optimization Checklist

| Optimization | Savings | How |
|---|---|---|
| **Use Spot instances** | 60-70% off compute | `SpotInstanceType: persistent` |
| **Same-region as S3** | $0 transfer fees | Launch in same region as your bucket |
| **gp3 over gp2** | 20% cheaper storage | gp3 is default, always cheaper |
| **Delete EBS after** | $0 ongoing storage | `DeleteOnTermination: true` |
| **Large instance** | Same cost, less risk | Faster = less time exposed to spot interruption |
| **S3 Intelligent-Tiering** | ~30% off cold data | For raw data you won't access often |
| **Resume support** | $0 wasted on reruns | progress.json prevents re-tokenizing |
| **Streaming upload** | Reduced peak disk | Upload shards as created, delete tmp files |

---

## Monitoring During Preprocessing

### Watch Progress

```bash
# Watch the log
tail -f /data/preprocess.log

# Monitor CPU utilization (should be near 100% on all cores)
htop

# Monitor disk space
watch -n 30 'df -h /data'

# Monitor I/O
iostat -x 5
```

### Cost Tracking

```bash
# Check instance uptime (to estimate current cost)
uptime

# Check spot price (verify you're still getting a good deal)
aws ec2 describe-spot-price-history \
  --instance-types c7i.24xlarge \
  --start-time $(date -u +"%Y-%m-%dT%H:%M:%S") \
  --product-descriptions "Linux/UNIX" \
  --region us-east-1
```

---

## Alternative: AWS Batch (For Repeated Preprocessing)

If you plan to re-run preprocessing multiple times (different tokenizers, updated data), AWS Batch automates everything:

```
AWS Batch Setup:
  1. Create a Docker image with preprocessing code + dependencies
  2. Push to ECR (Elastic Container Registry)
  3. Create Batch Compute Environment (Spot, c7i family)
  4. Create Job Definition (container, resource requirements)
  5. Submit Job → AWS Batch handles everything

Pros:
  + Fully automated (no SSH, no manual steps)
  + Auto-retry on spot interruption
  + Logs go to CloudWatch automatically
  + Easy to parameterize (different tokenizers, shard sizes)

Cons:
  - Setup overhead (~1-2 hours first time)
  - Docker image build time
  - Slightly more expensive ($50 vs $26)

Verdict: Use simple EC2 for one-time job, AWS Batch if you'll iterate.
```

---

## Final Cost Breakdown

```
┌──────────────────────────────────────────────────────────────────┐
│                    TOTAL PREPROCESSING COST                      │
│                                                                  │
│  Compute (c7i.24xlarge Spot × 12 hours)                         │
│    $1.22/hr × 12 hr                            = $14.64         │
│                                                                  │
│  Storage (16 TB gp3 EBS × 12 hours)                             │
│    16,000 GB × $0.08/GB/month × (12/720)       = $21.33         │
│                                                                  │
│  S3 storage (final shards, ongoing)                             │
│    4,000 GB × $0.023/GB/month                  = $92.00/month   │
│                                                                  │
│  S3 API calls                                                   │
│    GET + PUT requests                           = ~$2.00         │
│                                                                  │
│  Data transfer (same region)                                    │
│    S3 ↔ EC2                                    = $0.00          │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│  ONE-TIME preprocessing cost:             ~$38                  │
│  ONGOING S3 storage:                      ~$92/month            │
│                                                                  │
│  vs. tokenizing during training:          ~$2,800+ WASTED       │
│  ROI on first training run:               73× return            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```
