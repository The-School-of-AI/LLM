# AWS Batch Deployment for Coreset Engine v5

This directory contains the configuration and scripts to run the Coreset Selection Engine as an **AWS Batch Array Job**, where each array index maps to one shard.

## How It Works

```
shard.sh (local)                    AWS Batch Array Job
────────────────────────            ─────────────────────────────────────
for SHARD_ID in 0..N-1:            arrayProperties.size = N
  python coreset_builder.py   →     Each container runs batch_entrypoint.sh
    --shard-id $SHARD_ID              AWS_BATCH_JOB_ARRAY_INDEX → --shard-id
    --num-shards N                    NUM_SHARDS env var         → --num-shards
    --input-path s3://...             S3_INPUT_PATH env var      → --input-path
    ...                               ...
```

## Files

| File | Purpose |
|------|---------|
| `../Dockerfile` | Container image for the engine |
| `../batch_entrypoint.sh` | Per-container entrypoint (replaces `shard.sh` inner loop) |
| `deploy_infra_and_run.sh` | **Automated Full Deployment** (IAM, Batch, ECR, Job) |
| `job_definition.json` | AWS Batch Job Definition template |
| `submit_job.sh` | CLI helper to submit the Array Job (manual) |

## Deployment & Execution Workflow

There are two primary scripts depending on your situation:

### Use Case 1: First-time Setup, New Region, or Code Changes
If you are running in a new AWS account, a new region, or you have modified the engine code, use the full deployment script. It provisions all IAM roles, Batch infrastructure, builds the Docker image, and submits the job.

**The Golden Rule:** Always run this script first in any new environment.

```bash
# From the project root:
./aws_batch/deploy_infra_and_run.sh \
  --bucket <my-bucket> \
  --input s3://<bucket>/path/to/data/ \
  --shards 8
```

### Use Case 2: Infrastructure Exists & Code is Unchanged
If you have already run the deployment script and just want to process a new dataset or resume a run, use the submission script. It skips the infrastructure and Docker overhead, making it much faster.

```bash
# From the project root:
./aws_batch/submit_job.sh \
  --bucket <my-bucket> \
  --input s3://<bucket>/new-dataset/ \
  --shards 8
```

---

## Technical Reference

### Script Arguments

Both scripts support the following flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--bucket` | **Required**. S3 bucket for outputs and checkpoints. | - |
| `--input` | **Required**. S3 URI to the input dataset shards. | - |
| `--shards` | Number of parallel shards (AWS Batch Array Size). | `8` |
| `--stages` | Space-separated list of stages to run. | `"1B 3B 8B 70B"` |
| `--region` | AWS Region to deploy/run in. | (detected) |
| `--tokens` | Total tokens estimate for the dataset. | `4523096944` |
| `--resume` | Set to `true` to continue from existing S3 checkpoints. | `false` |

### Infrastructure provisioned by `deploy_infra_and_run.sh`:
- **IAM**: Creates `coreset-engine-batch-role` with S3, ECR, and CloudWatch permissions.
- **ECR**: Creates `coreset-engine` repository.
- **Batch**: 
  - Fargate Compute Environment (`coreset-engine-fargate`)
  - Job Queue (`coreset-engine-queue`)
  - Job Definition (`coreset-engine`)
- **Networking**: Automatically discovers and uses the **Default VPC** and its subnets.
- **Logs**: Creates CloudWatch Log Group `/aws/batch/coreset-engine`.

## Manual Steps (Advanced)

If you prefer to manage infrastructure manually (e.g., via Terraform), you can still use the underlying components:

## Monitoring & Operations

### View Logs
```bash
# General Tail
aws logs tail /aws/batch/coreset-engine --follow

# Per-shard logs (Shard 0)
LOG_STREAM=$(aws batch describe-jobs --jobs <JOB_ID>:0 --query "jobs[0].attempts[-1].container.logStreamName" --output text)
aws logs tail /aws/batch/coreset-engine --log-stream-names $LOG_STREAM --follow
```

### Checkpoints & Resuming
Checkpoints are written to `s3://${S3_BUCKET}/coreset-checkpoints/shard###/`. To resume a failed run:
1. Resubmit the job with `RESUME=true` in environment overrides.
2. Each shard container will automatically discover its own last checkpoint from S3 and proceed.

### Merging Outputs
After all shards complete, merge the per-shard part files into the final manifests:
```bash
python tools/merge_selected_indices.py \
  --coreset-root s3://my-data-bucket/coreset-output \
  --stages 1B
```

## IAM Permissions Required
The execution role requires:
- `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`
- `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`, `ecr:GetAuthorizationToken`
- `logs:CreateLogStream`, `logs:PutLogEvents`
