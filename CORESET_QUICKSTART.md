# Coreset Engineering - Quick Start Guide

Get the coreset engineering pipeline running on AWS in 5 minutes.

## What You're Deploying

A production ML pipeline that:
1. Ingests raw training data (diverse sources, modalities)
2. Deduplicates and scores data by difficulty
3. Samples curriculum-aligned subsets for 4 training phases
4. Generates AWS S3 manifests for trainer consumption
5. Validates correctness and generates audit reports

**Architecture**: GitHub → Docker → AWS ECR → ECS Fargate + S3 + Lambda

## Prerequisites (2 minutes)

### 1. AWS Setup
```bash
# Create IAM role (ask your AWS admin or follow CloudFormation template)
# Get your account ID
aws sts get-caller-identity | grep Account

# Save these as GitHub secrets
# Repository Settings → Secrets and variables → Actions
# Add:
#   AWS_ACCOUNT_ID = "123456789012"
#   AWS_ROLE_TO_ASSUME = "arn:aws:iam::123456789012:role/github-actions-staging"
#   AWS_ROLE_TO_ASSUME_PROD = "arn:aws:iam::123456789012:role/github-actions-prod"
```

### 2. Create ECR Repository
```bash
aws ecr create-repository \
    --repository-name coreset-engineering \
    --region us-east-1
```

## Deploy (3 minutes)

### Option A: Automatic (Recommended)

```bash
# Push to develop branch → auto-deploys to staging
git commit -m "feat: update coreset pipeline"
git push origin develop

# Push to main branch → auto-deploys to production
git push origin main
```

**Behind the scenes:**
- GitHub Actions runs tests
- Builds Docker image
- Pushes to ECR
- Updates ECS service
- Validates deployment

**View progress**: Repository → Actions tab

### Option B: Manual

```bash
# Build locally
cd experiments/3_coreset_engineering
docker build -t coreset-engineering .

# Tag and push
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

docker tag coreset-engineering:latest \
    $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/coreset-engineering:latest

docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/coreset-engineering:latest

# Deploy
aws ecs update-service \
    --cluster coreset-staging \
    --service coreset-builder \
    --force-new-deployment \
    --region $REGION
```

## Run Tests (30 seconds)

```bash
cd experiments/3_coreset_engineering

# All tests
uv run pytest tests/ -v

# Quick tests only
uv run pytest tests/ -m "not slow" -v

# With coverage
uv run pytest tests/ --cov=src/coreset_engine
```

## Monitor Deployment

### GitHub Actions
```
Repository → Actions → "Coreset Engineering - Build, Test & Deploy"
```

### CloudWatch Logs
```bash
aws logs tail /ecs/coreset-staging --follow

# Search for specific stage
aws logs filter-log-events \
    --log-group-name /ecs/coreset-staging \
    --filter-pattern "1B stage"
```

### AWS Console
```
ECS → Clusters → coreset-staging → Services → coreset-builder
```

## Verify Success

### Check S3 Output
```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="llm-coreset-artifacts-$ACCOUNT_ID-staging"

# List manifests
aws s3 ls s3://$BUCKET/manifests/1B/
aws s3 ls s3://$BUCKET/manifests/3B/
aws s3 ls s3://$BUCKET/manifests/8B/
aws s3 ls s3://$BUCKET/manifests/70B/

# View manifest
aws s3 cp s3://$BUCKET/manifests/1B/manifest.json - | jq '.' | head -50
```

### Run Validation Lambda
```bash
aws lambda invoke \
    --function-name coreset-staging-validation \
    --payload '{"check_type":"comprehensive"}' \
    response.json

cat response.json | jq '.'
```

## Usage

### Execute Pipeline on AWS

```bash
# Run task on demand
aws ecs run-task \
    --cluster coreset-staging \
    --task-definition coreset-builder-staging \
    --launch-type FARGATE \
    --network-configuration 'awsvpcConfiguration={
        subnets=[subnet-xxxxx],
        securityGroups=[sg-xxxxx],
        assignPublicIp=ENABLED
    }' \
    --region us-east-1
```

### Monitor Execution
```bash
# Get task ID from above output
TASK_ARN="arn:aws:ecs:us-east-1:123456789012:task/coreset-staging/xxxxx"

# Watch logs
aws logs tail /ecs/coreset-staging --follow

# Check task status
aws ecs describe-tasks --cluster coreset-staging --tasks $TASK_ARN
```

## Download Results

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="llm-coreset-artifacts-$ACCOUNT_ID-staging"

# Download all manifests
aws s3 sync s3://$BUCKET/manifests/ ./coreset_manifests/

# Download audit reports
aws s3 sync s3://$BUCKET/audits/ ./coreset_audits/
```

## Troubleshooting

### Tests Fail in GitHub Actions
```
→ Check: Repository → Actions → Logs
→ Common: Missing dependencies in pyproject.toml
→ Fix: Update pyproject.toml and push again
```

### Docker Build Fails
```
→ Check: Docker build locally first
→ docker build -f experiments/3_coreset_engineering/Dockerfile .
→ Fix: Check file paths and PYTHON_VERSION variable
```

### ECS Task Fails
```bash
# Check logs
aws logs tail /ecs/coreset-staging --follow

# Check task definition
aws ecs describe-task-definition \
    --task-definition coreset-builder-staging | jq '.taskDefinition'

# Check service
aws ecs describe-services \
    --cluster coreset-staging \
    --services coreset-builder
```

### Validation Lambda Fails
```bash
# View logs
aws logs tail /aws/lambda/coreset-staging-validation --follow

# Check manifest exists
aws s3 ls s3://llm-coreset-artifacts-{account}-staging/manifests/
```

## Next Steps

1. **Read Full Documentation**
   - [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) - Detailed deployment guide
   - [tests/3_coreset_engineering/TEST_GUIDE.md](../../tests/3_coreset_engineering/TEST_GUIDE.md) - Testing guide

2. **Customize for Your Data**
   - Update `configs/curriculum.yaml` with your stage definitions
   - Modify `scripts/generate_mock_data.py` for your data format
   - Adjust `src/coreset_engine/scoring/` for your difficulty metrics

3. **Monitor Production**
   - Set up CloudWatch alarms for task failures
   - Configure SNS notifications to Slack
   - Review metrics dashboard weekly

4. **Optimize Performance**
   - Profile with larger datasets (100k+ samples)
   - Tune ECS task CPU/memory allocation
   - Consider Fargate Spot for non-critical runs (70% cost savings)

## Support

**Stuck?**
1. Check logs: `aws logs tail /ecs/coreset-staging --follow`
2. Review [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) troubleshooting section
3. Check GitHub Actions logs: Repository → Actions tab
4. Contact: Team 3 - Coreset Engineering

**Contributing?**
1. Create feature branch: `git checkout -b feature/my-feature`
2. Add/update tests: `uv run pytest tests/`
3. Push: `git push origin feature/my-feature`
4. Create PR with test results
5. Minimum 2 approvals required before merge
