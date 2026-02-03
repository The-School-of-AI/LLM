# Deployment Checklist & Verification

## ✓ All Files Created Successfully

The following files have been created and are ready for use:

### GitHub Actions & CI/CD
- ✓ [`.github/workflows/coreset-deploy.yml`](.github/workflows/coreset-deploy.yml) - Main pipeline
- ✓ [`.github/workflows/template-deploy.yml`](.github/workflows/template-deploy.yml) - Reusable template
- ✓ [`.github/copilot-instructions.md`](.github/copilot-instructions.md) - AI agent guidance

### Docker & AWS Infrastructure
- ✓ [`experiments/3_coreset_engineering/Dockerfile`](experiments/3_coreset_engineering/Dockerfile)
- ✓ [`experiments/3_coreset_engineering/aws/cloudformation.yaml`](experiments/3_coreset_engineering/aws/cloudformation.yaml)
- ✓ [`experiments/3_coreset_engineering/aws/lambda_validator.py`](experiments/3_coreset_engineering/aws/lambda_validator.py)

### Tests & Configuration
- ✓ [`tests/3_coreset_engineering/test_builder_regression.py`](tests/3_coreset_engineering/test_builder_regression.py) - 28 regression tests
- ✓ [`tests/3_coreset_engineering/test_e2e_integration.py`](tests/3_coreset_engineering/test_e2e_integration.py) - 14 integration tests
- ✓ [`tests/conftest.py`](tests/conftest.py) - Pytest configuration

### Documentation
- ✓ [`experiments/3_coreset_engineering/AWS_DEPLOYMENT.md`](experiments/3_coreset_engineering/AWS_DEPLOYMENT.md)
- ✓ [`tests/3_coreset_engineering/TEST_GUIDE.md`](tests/3_coreset_engineering/TEST_GUIDE.md)
- ✓ [`CORESET_QUICKSTART.md`](CORESET_QUICKSTART.md)
- ✓ [`CORESET_IMPLEMENTATION_SUMMARY.md`](CORESET_IMPLEMENTATION_SUMMARY.md)
- ✓ [`FILES_CREATED.md`](FILES_CREATED.md)

---

## Pre-Deployment Steps

### 1. Update Dependencies (if not already done)

**File**: `experiments/3_coreset_engineering/pyproject.toml`

Ensure these are in dev-dependencies:
```toml
[tool.uv]
dev-dependencies = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
]
```

### 2. Configure GitHub Secrets

**Location**: Repository Settings → Secrets and variables → Actions

Required secrets:
```
AWS_ACCOUNT_ID              # e.g., 123456789012
AWS_ROLE_TO_ASSUME          # e.g., arn:aws:iam::123456789012:role/github-actions-staging
AWS_ROLE_TO_ASSUME_PROD     # e.g., arn:aws:iam::123456789012:role/github-actions-prod
```

Optional for other teams:
```
ECR_REPOSITORY              # e.g., coreset-engineering
ECS_CLUSTER_NAME            # e.g., coreset-staging
ECS_SERVICE_NAME            # e.g., coreset-builder
```

### 3. AWS Account Setup (One-time)

```bash
# Create IAM roles for GitHub Actions
# (Ask AWS admin or use provided CloudFormation template)

# Create ECR repository
aws ecr create-repository \
    --repository-name coreset-engineering \
    --region us-east-1

# Create S3 buckets (CloudFormation will do this)
# Manual creation (optional):
aws s3 mb s3://llm-coreset-artifacts-$(aws sts get-caller-identity --query Account --output text)-staging
aws s3 mb s3://llm-coreset-artifacts-$(aws sts get-caller-identity --query Account --output text)-production
```

### 4. Create CloudFormation Stack

```bash
# Staging environment
aws cloudformation create-stack \
    --stack-name coreset-staging \
    --template-body file://experiments/3_coreset_engineering/aws/cloudformation.yaml \
    --parameters \
        ParameterKey=Environment,ParameterValue=staging \
        ParameterKey=ECRImageUri,ParameterValue=<account>.dkr.ecr.us-east-1.amazonaws.com/coreset-engineering:latest \
    --capabilities CAPABILITY_NAMED_IAM \
    --region us-east-1

# Production environment
aws cloudformation create-stack \
    --stack-name coreset-production \
    --template-body file://experiments/3_coreset_engineering/aws/cloudformation.yaml \
    --parameters \
        ParameterKey=Environment,ParameterValue=production \
        ParameterKey=ECRImageUri,ParameterValue=<account>.dkr.ecr.us-east-1.amazonaws.com/coreset-engineering:latest \
    --capabilities CAPABILITY_NAMED_IAM \
    --region us-east-1
```

### 5. Test Locally

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/3_coreset_engineering -v

# With coverage
uv run pytest tests/ --cov=src/coreset_engine --cov-report=html

# Check linting
uv run ruff check experiments/3_coreset_engineering

# Check types
uv run mypy experiments/3_coreset_engineering --ignore-missing-imports
```

---

## Deployment Steps

### Option 1: Automatic Deployment (Recommended)

```bash
# Commit changes
git add .

git commit -m "feat: add GitHub pipeline and regression tests

- Add coreset-deploy.yml workflow
- Add 42 regression tests (28 unit + 14 integration)
- Add CloudFormation infrastructure template
- Add Lambda validator for post-deploy checks
- Add comprehensive documentation"

# Push to trigger automatic deployment
git push origin develop     # → Deploys to staging
# OR
git push origin main        # → Deploys to production
```

**Monitor deployment**:
```bash
# GitHub Actions
echo "View progress at: https://github.com/The-School-of-AI/LLM/actions"

# CloudWatch Logs
aws logs tail /ecs/coreset-staging --follow
```

### Option 2: Manual Deployment (if needed)

```bash
# Build image
cd experiments/3_coreset_engineering
docker build -t coreset-engineering:latest .

# Push to ECR
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

docker tag coreset-engineering:latest \
    $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/coreset-engineering:latest

docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/coreset-engineering:latest

# Update ECS service
aws ecs update-service \
    --cluster coreset-staging \
    --service coreset-builder \
    --force-new-deployment \
    --region $REGION

# Wait for stability
aws ecs wait services-stable \
    --cluster coreset-staging \
    --services coreset-builder \
    --region $REGION

# Verify
aws logs tail /ecs/coreset-staging --follow
```

---

## Post-Deployment Verification

### 1. Verify Tests Pass

```bash
# Check GitHub Actions
# Repository → Actions → Coreset Engineering - Build, Test & Deploy

# Expected: ✓ test job passes with >80% coverage
```

### 2. Verify Docker Build

```bash
# Check ECR
aws ecr describe-images \
    --repository-name coreset-engineering \
    --region us-east-1

# Expected: Image with tag matching commit SHA
```

### 3. Verify ECS Deployment

```bash
# Check service status
aws ecs describe-services \
    --cluster coreset-staging \
    --services coreset-builder \
    --region us-east-1 | jq '.services[0].status'

# Expected: ACTIVE

# Check running tasks
aws ecs list-tasks \
    --cluster coreset-staging \
    --service-name coreset-builder \
    --region us-east-1

# Expected: At least 1 task running
```

### 4. Verify S3 Artifacts

```bash
# Check artifact bucket
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="llm-coreset-artifacts-$ACCOUNT-staging"

aws s3 ls $BUCKET/

# Expected: manifests/, audits/, logs/ directories
```

### 5. Verify Lambda Validation

```bash
# Run validation Lambda
aws lambda invoke \
    --function-name coreset-staging-validation \
    --payload '{"manifest_path":"manifests/1B/manifest.json"}' \
    response.json \
    --region us-east-1

cat response.json | jq '.'

# Expected: status: "PASS"
```

---

## Troubleshooting

### Tests Fail in GitHub Actions

**Check logs**:
```bash
# Repository → Actions → Coreset Engineering - Build, Test & Deploy
# → Click "test" job → View logs
```

**Common issues**:
1. Missing imports → Update imports in test files
2. Fixture errors → Check conftest.py paths
3. Mock data issues → Verify temp_dirs fixture

**Fix**:
```bash
# Run locally first
uv run pytest tests/3_coreset_engineering -v

# Fix issues, then push
git add .
git commit -m "fix: resolve test failures"
git push origin feature-branch
```

### Docker Build Fails

**Check logs**:
```bash
# GitHub Actions → build job → View logs
```

**Common issues**:
1. File not found → Check COPY paths in Dockerfile
2. Dependency missing → Update pyproject.toml
3. Python version issue → Check FROM python:3.12 base

**Fix**:
```bash
# Test build locally
docker build -f experiments/3_coreset_engineering/Dockerfile .

# Fix Dockerfile, then push
git add experiments/3_coreset_engineering/Dockerfile
git commit -m "fix: update Dockerfile"
git push
```

### ECS Deployment Fails

**Check logs**:
```bash
# CloudWatch
aws logs tail /ecs/coreset-staging --follow

# ECS console
aws ecs describe-tasks \
    --cluster coreset-staging \
    --tasks <task-arn> \
    --region us-east-1 | jq '.tasks[0].stoppedReason'
```

**Common issues**:
1. Port not open → Check security group
2. Memory exceeded → Increase ECS task memory
3. Missing environment vars → Check CloudFormation parameters
4. IAM permissions → Verify task execution role

**Fix**:
```bash
# Update CloudFormation if needed
aws cloudformation update-stack \
    --stack-name coreset-staging \
    --template-body file://experiments/3_coreset_engineering/aws/cloudformation.yaml \
    --parameters ParameterKey=TaskMemory,ParameterValue=4096 \
    --capabilities CAPABILITY_NAMED_IAM

# Check status
aws cloudformation describe-stacks --stack-name coreset-staging
```

### Lambda Validation Fails

**Check logs**:
```bash
# CloudWatch Logs
aws logs tail /aws/lambda/coreset-staging-validation --follow

# View full validation report
aws lambda invoke \
    --function-name coreset-staging-validation \
    --payload '{"check_type":"comprehensive"}' \
    response.json
cat response.json | jq '.body'
```

**Common issues**:
1. Manifest doesn't exist → Run coreset builder
2. Schema invalid → Check manifest.json structure
3. Distribution mismatch → Verify curriculum.yaml

**Fix**:
```bash
# Validate manifest manually
aws s3 cp s3://$BUCKET/manifests/1B/manifest.json - | jq '.' | head -50

# If invalid, re-run coreset builder with correct config
```

---

## Monitoring & Alerts

### Set Up CloudWatch Alarms

```bash
# Task count alarm (already in CloudFormation)
aws cloudwatch describe-alarms \
    --alarm-names coreset-task-count-staging \
    --region us-east-1

# Add custom alarms as needed
aws cloudwatch put-metric-alarm \
    --alarm-name coreset-validation-failures \
    --metric-name ValidationStatus \
    --namespace CoresetEngineering \
    --statistic Average \
    --period 300 \
    --threshold 0.5 \
    --comparison-operator LessThanThreshold \
    --evaluation-periods 1
```

### View Metrics

```bash
# Validation metrics
aws cloudwatch get-metric-statistics \
    --namespace CoresetEngineering \
    --metric-name ValidationStatus \
    --start-time 2026-02-01T00:00:00Z \
    --end-time 2026-02-02T00:00:00Z \
    --period 3600 \
    --statistics Average

# ECS metrics
aws cloudwatch get-metric-statistics \
    --namespace AWS/ECS \
    --metric-name TaskCount \
    --dimensions Name=ClusterName,Value=coreset-staging \
    --start-time 2026-02-01T00:00:00Z \
    --end-time 2026-02-02T00:00:00Z \
    --period 3600 \
    --statistics Average
```

---

## Success Criteria

All of the following should be true:

- [ ] GitHub Actions workflow runs on every push
- [ ] All 42 tests pass (28 regression + 14 integration)
- [ ] Code coverage >80%
- [ ] Docker image builds and pushes to ECR
- [ ] ECS service updates with new image
- [ ] Tasks start and pass health checks
- [ ] S3 manifests are generated
- [ ] Lambda validation confirms manifest integrity
- [ ] CloudWatch logs show successful execution
- [ ] Team can fetch manifests for training

---

## Next Steps After Deployment

### 1. Monitor First Week

- Watch CloudWatch logs daily
- Check alarm triggers
- Monitor cost (should stay <$50/month)
- Verify manifests are usable by training team

### 2. Optimize

- Adjust ECS task CPU/memory based on actual usage
- Consider Fargate Spot for non-critical runs
- Enable S3 lifecycle policies for old manifests

### 3. Scale

- Plan for 1B+ sample datasets
- Consider distributed bucketing/sampling
- Add streaming manifest support

### 4. Integrate

- Coordinate with Team 10 (training) on manifest format
- Align with Team 2 (curriculum) on config schema
- Plan tokenizer integration with Team 6

---

## Support

**Documentation**:
- [AWS_DEPLOYMENT.md](experiments/3_coreset_engineering/AWS_DEPLOYMENT.md) - Detailed guide
- [TEST_GUIDE.md](tests/3_coreset_engineering/TEST_GUIDE.md) - Testing reference
- [CORESET_QUICKSTART.md](CORESET_QUICKSTART.md) - Quick reference
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - For AI agents

**Contact**:
- Slack: #team-3-coreset-engineering
- GitHub: Issues with label `coreset-engineering`
- Code owners: @The-School-of-AI/llm-repo-owners

---

## Summary

You now have a **production-ready** CI/CD pipeline with:

✓ 42 comprehensive regression tests  
✓ GitHub Actions automated deployment  
✓ AWS infrastructure as code (CloudFormation)  
✓ Post-deployment validation (Lambda)  
✓ Complete documentation  
✓ Copilot instructions for AI agents  

**Ready to deploy?** Follow the "Deployment Steps" section above.
