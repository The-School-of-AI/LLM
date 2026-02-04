# AWS S3 Upload Guide

This guide will help you upload the Sangraha and IndicCorp V2 datasets to AWS S3.

## 📋 Prerequisites

1. **AWS Account** with S3 access
2. **AWS CLI** installed and configured
3. **Python 3.8+** with required packages

## 🔧 Setup

### 1. Install AWS CLI (if not already installed)

**Windows:**
```powershell
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
```

**Linux/Mac:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### 2. Configure AWS Credentials

```bash
aws configure
```

You'll be prompted for:
- **AWS Access Key ID**: Your access key
- **AWS Secret Access Key**: Your secret key
- **Default region**: e.g., `us-east-1`
- **Default output format**: `json`

### 3. Create S3 Bucket

```bash
# Replace 'my-llm-datasets' with your bucket name
aws s3 mb s3://my-llm-datasets --region us-east-1
```

### 4. Update Configuration

Edit `config.yml` and update:

```yaml
storage:
  mode: s3          # Use 's3' for S3 only, 'both' for local + S3

aws:
  region: us-east-1
  s3_bucket: my-llm-datasets      # YOUR BUCKET NAME HERE
  s3_prefix: hf
```

## 🚀 Running the Upload

### Test Mode (Recommended First)

Downloads limited data for testing:

```bash
python download.py
```

This will upload:
- **Sangraha**: 10,000 rows × 3 languages = ~30,000 records
- **IndicCorp V2**: 50,000 Hindi records

**Expected S3 structure:**
```
s3://my-llm-datasets/
└── hf/
    ├── sangraha/
    │   ├── hin_Deva/
    │   │   ├── part-00000.jsonl
    │   │   └── part-00001.jsonl
    │   ├── hin_Latn/
    │   │   └── part-00000.jsonl
    │   └── tam_Taml/
    │       └── part-00000.jsonl
    └── indiccorp_v2/
        └── hin/
            ├── part-00000.jsonl
            ├── part-00001.jsonl
            ├── part-00002.jsonl
            ├── part-00003.jsonl
            ├── part-00004.jsonl
            └── part-00005.jsonl
```

### Full Mode (Production)

To download complete datasets, edit `config.yml`:

```yaml
mode: full        # Change from 'test' to 'full'

datasets:
  sangraha:
    test_limit:
      type: none  # Download everything
  
  indiccorp_v2:
    test_limit:
      type: none  # Download everything
```

Then run:
```bash
python download.py
```

⚠️ **Warning**: Full datasets are VERY large (100s of GB). Monitor your:
- AWS S3 storage costs
- Data transfer costs
- Disk space (if using `mode: both`)

## 💰 Cost Estimation

### S3 Storage Costs (us-east-1)
- **Standard Storage**: $0.023 per GB/month
- **PUT Requests**: $0.005 per 1,000 requests

### Estimated Costs (Test Mode)
- **Data size**: ~500 MB
- **Monthly storage**: ~$0.01
- **Upload requests**: ~$0.00

### Estimated Costs (Full Mode)
- **Sangraha**: ~50 GB
- **IndicCorp V2**: ~200 GB
- **Total**: ~250 GB
- **Monthly storage**: ~$5.75
- **One-time upload**: ~$0.02

## 🔍 Verify Upload

### Check S3 bucket contents:
```bash
aws s3 ls s3://my-llm-datasets/hf/ --recursive --human-readable
```

### Download a sample file:
```bash
aws s3 cp s3://my-llm-datasets/hf/sangraha/hin_Deva/part-00000.jsonl ./test.jsonl
```

### Count total files:
```bash
aws s3 ls s3://my-llm-datasets/hf/ --recursive | wc -l
```

## 🐛 Troubleshooting

### Error: "Unable to locate credentials"
```bash
# Check credentials
aws configure list

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### Error: "Access Denied"
- Check IAM permissions for S3 access
- Ensure bucket policy allows PutObject

### Error: "Bucket does not exist"
```bash
# Create bucket
aws s3 mb s3://my-llm-datasets --region us-east-1
```

### Slow uploads
- Use `mode: local` first, then sync to S3:
```bash
aws s3 sync ./data/downloaded_datasets s3://my-llm-datasets/hf/
```

## 📊 Monitoring

### Watch progress in real-time:
```bash
watch -n 5 'aws s3 ls s3://my-llm-datasets/hf/ --recursive | wc -l'
```

### Check bucket size:
```bash
aws s3 ls s3://my-llm-datasets --recursive --summarize --human-readable
```

## 🔒 Security Best Practices

1. **Use IAM roles** instead of access keys when possible
2. **Enable bucket versioning** for data protection
3. **Enable encryption** at rest
4. **Set lifecycle policies** to manage costs
5. **Use VPC endpoints** for private access

### Enable encryption:
```bash
aws s3api put-bucket-encryption \
  --bucket my-llm-datasets \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'
```

## 📝 Next Steps

After uploading to S3, you can:
1. **Process data** using AWS EMR or Glue
2. **Train models** using SageMaker
3. **Share data** with team via S3 presigned URLs
4. **Archive old versions** to Glacier for cost savings

## 🆘 Support

- AWS S3 Documentation: https://docs.aws.amazon.com/s3/
- boto3 Documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
- HuggingFace Datasets: https://huggingface.co/docs/datasets/
