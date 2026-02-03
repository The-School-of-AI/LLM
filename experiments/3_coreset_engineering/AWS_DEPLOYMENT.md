# Coreset Engineering - AWS Deployment Guide

This guide covers deploying the coreset engineering pipeline to AWS.

## Architecture Overview

```
GitHub Actions (CI/CD)
    ↓
Build Docker Image → Push to ECR
    ↓
Deploy to AWS:
  - ECS Fargate (Task execution)
  - S3 (Artifact storage)
  - Lambda (Validation)
  - CloudWatch (Monitoring)
```

## Prerequisites

1. **AWS Account Setup**
   - AWS account with appropriate permissions
   - IAM roles configured (see CloudFormation template)
   - ECR repository created: `coreset-engineering`

2. **GitHub Secrets**
   Set these in repository settings → Secrets and variables → Actions:
   
   ```
   AWS_ACCOUNT_ID          # Your AWS account ID
   AWS_ROLE_TO_ASSUME      # Staging environment role
   AWS_ROLE_TO_ASSUME_PROD # Production environment role
   ```

3. **Local Development**
   ```bash
   # Install AWS CLI
   pip install awscli
   
   # Configure credentials
   aws configure
   ```

## Deployment Process

### 1. Automatic Deployment via GitHub Actions

Pushes to `main` or `develop` branches automatically trigger:

```bash
# Trigger workflow
git push origin develop  # → Staging deployment
git push origin main     # → Production deployment
```

**Workflow stages:**
- Test (unit + regression tests)
- Build (Docker image)
- Deploy to staging/production

### 2. Manual Deployment

```bash
# Build image locally
cd experiments/3_coreset_engineering
docker build -t coreset-engineering:latest .

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account_id>.dkr.ecr.us-east-1.amazonaws.com

docker tag coreset-engineering:latest <account_id>.dkr.ecr.us-east-1.amazonaws.com/coreset-engineering:latest
docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/coreset-engineering:latest

# Update ECS service
aws ecs update-service \
    --cluster coreset-staging \
    --service coreset-builder \
    --force-new-deployment \
    --region us-east-1
```

### 3. CloudFormation Stack Creation

```bash
# Deploy infrastructure
aws cloudformation create-stack \
    --stack-name coreset-staging \
    --template-body file://experiments/3_coreset_engineering/aws/cloudformation.yaml \
    --parameters \
        ParameterKey=Environment,ParameterValue=staging \
        ParameterKey=ECRImageUri,ParameterValue=<account_id>.dkr.ecr.us-east-1.amazonaws.com/coreset-engineering:latest \
    --capabilities CAPABILITY_NAMED_IAM \
    --region us-east-1
```

## Running the Pipeline

### Via CLI

```bash
# Execute coreset builder
aws ecs run-task \
    --cluster coreset-staging \
    --task-definition coreset-builder-staging \
    --launch-type FARGATE \
    --network-configuration awsvpcConfiguration="{subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
    --overrides '{
        "containerOverrides": [{
            "name": "coreset-builder",
            "environment": [
                {"name": "CONFIG_PATH", "value": "s3://bucket/curriculum.yaml"},
                {"name": "DATA_PATH", "value": "s3://bucket/input/"},
                {"name": "OUTPUT_PATH", "value": "s3://bucket/output/"}
            ]
        }]
    }' \
    --region us-east-1
```

### Via Console

1. ECS → Clusters → `coreset-staging` → Services → `coreset-builder`
2. Click "Update service" → Force new deployment
3. Monitor task execution in CloudWatch Logs

## Monitoring & Validation

### CloudWatch Logs

```bash
# View logs
aws logs tail /ecs/coreset-staging --follow

# Search for errors
aws logs filter-log-events \
    --log-group-name /ecs/coreset-staging \
    --filter-pattern "ERROR"
```

### CloudWatch Metrics

```bash
# View custom metrics
aws cloudwatch get-metric-statistics \
    --namespace CoresetEngineering \
    --metric-name ValidationStatus \
    --dimensions Name=Environment,Value=staging \
    --start-time 2026-02-01T00:00:00Z \
    --end-time 2026-02-02T00:00:00Z \
    --period 3600 \
    --statistics Average
```

### Lambda Validation

Automatically triggered post-deployment:

```bash
# Manual invocation
aws lambda invoke \
    --function-name coreset-staging-validation \
    --payload '{"manifest_path":"manifests/1B/manifest.json","check_type":"comprehensive"}' \
    response.json \
    --region us-east-1

cat response.json
```

## S3 Artifact Structure

```
s3://llm-coreset-artifacts-{account}-{env}/
├── manifests/
│   ├── 1B/
│   │   └── manifest.json
│   ├── 3B/
│   │   └── manifest.json
│   ├── 8B/
│   │   └── manifest.json
│   └── 70B/
│       └── manifest.json
├── audits/
│   ├── band_distribution_1B.png
│   ├── band_distribution_3B.png
│   ├── modality_distribution_70B.png
│   └── difficulty_histogram.png
└── logs/
    └── pipeline_execution_TIMESTAMP.log
```

## Regression Testing

Tests run automatically in CI/CD pipeline. To run locally:

```bash
# Unit tests
cd experiments/3_coreset_engineering
uv run pytest tests/test_builder_regression.py -v

# Integration tests
uv run pytest tests/test_e2e_integration.py -v

# Coverage report
uv run pytest tests/ --cov=src/coreset_engine --cov-report=html
open htmlcov/index.html
```

## Troubleshooting

### Task Fails to Start

```bash
# Check task definition
aws ecs describe-task-definition --task-definition coreset-builder-staging

# Check service logs
aws logs describe-log-streams --log-group-name /ecs/coreset-staging
```

### Validation Fails

```bash
# Check Lambda logs
aws logs tail /aws/lambda/coreset-staging-validation --follow

# Re-run validation
aws lambda invoke \
    --function-name coreset-staging-validation \
    --payload '{"check_type":"comprehensive"}' \
    response.json
```

### S3 Upload Issues

```bash
# Verify bucket permissions
aws s3 ls s3://llm-coreset-artifacts-{account}-staging/ --recursive

# Test write access
aws s3 cp test.txt s3://llm-coreset-artifacts-{account}-staging/test.txt
```

## Cost Optimization

- **ECS**: Use Fargate Spot for non-critical runs (70% savings)
- **S3**: Lifecycle policies to move old manifests to Glacier after 30 days
- **Lambda**: Configure memory appropriately (validation: 256MB should suffice)

## Security

- All S3 buckets are private (block public access)
- Task execution role has minimal required permissions (principle of least privilege)
- ECR images scanned for vulnerabilities
- Secrets managed via AWS Secrets Manager (not GitHub secrets for sensitive data)

## Support

- **GitHub Actions Issues**: Check workflow logs → Actions tab
- **AWS Issues**: CloudWatch Logs → /ecs/coreset-{env}
- **Validation Failures**: Lambda execution logs → /aws/lambda/coreset-{env}-validation
