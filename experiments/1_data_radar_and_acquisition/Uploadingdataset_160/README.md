# Dataset Download Tool

A flexible tool to download HuggingFace datasets (Sangraha & IndicCorp V2) to local storage and/or AWS S3 with comprehensive timing tracking.

## 📋 Features

✅ **Multiple Storage Modes**: Download to local, S3, or both  
✅ **Test & Full Modes**: Test with limited data or download complete datasets  
✅ **Runtime Configuration**: Override settings via command-line arguments  
✅ **Comprehensive Timing**: Track time for each step and total execution  
✅ **Progress Tracking**: Real-time progress with record counts  
✅ **Error Handling**: Graceful error handling with detailed logging  
✅ **Flexible Configuration**: YAML-based configuration with CLI overrides

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Download test data locally
python download.py --storage local --mode test

# Upload to S3
python download.py --storage s3 --s3-bucket YOUR-BUCKET --mode test

# Download locally AND upload to S3
python download.py --storage both --s3-bucket YOUR-BUCKET --mode test
```

## 📚 Datasets

This tool downloads three high-quality datasets for Indic languages and educational content:

### 1. **Sangraha** - AI4Bharat Synthetic Dataset
- **Languages**: Hindi (Devanagari), Hindi (Latin), Tamil
- **Test Mode**: 10,000 rows per language (30,000 total)
- **Full Mode**: Complete dataset (~50 GB)
- **Source**: [ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)

### 2. **IndicCorp V2** - Hindi Corpus
- **Language**: Hindi (Devanagari)
- **Test Mode**: 50,000 rows
- **Full Mode**: Complete dataset (~200 GB)
- **Source**: [ai4bharat/IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2)

### 3. **NCERT** - Educational Textbooks (Grades 6-12)
- **Content**: NCERT textbooks across multiple subjects
- **Subjects**: Science, Mathematics, Social Studies, Languages, etc.
- **Grades**: 6th to 12th standard
- **Test Mode**: 10,000 rows
- **Full Mode**: Complete dataset (~5 GB)
- **Source**: [ParthKadam2003/NCERT_Dataset](https://huggingface.co/datasets/ParthKadam2003/NCERT_Dataset)

## 🎯 Command-Line Options

### Available Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--storage` | Storage mode: `local`, `s3`, or `both` | `--storage both` |
| `--s3-bucket` | S3 bucket name | `--s3-bucket my-bucket` |
| `--region` | AWS region | `--region us-east-1` |
| `--mode` | Download mode: `test` or `full` | `--mode full` |
| `--config` | Config file path | `--config custom.yml` |
| `--datasets` | Specific datasets to download | `--datasets sangraha ncert` |

### Usage Examples

#### Selective Dataset Download

```bash
# Download only NCERT dataset
python download.py --datasets ncert --storage local --mode test

# Download only Sangraha dataset
python download.py --datasets sangraha --storage local --mode test

# Download multiple specific datasets
python download.py --datasets sangraha ncert --storage local --mode test

# Download only IndicCorp V2 to S3
python download.py --datasets indiccorp_v2 --storage s3 --s3-bucket my-bucket --mode full
```

#### Test Mode Examples

```bash
# Test data locally only (all datasets)
python download.py --storage local --mode test

# Test data to S3 only
python download.py --storage s3 --s3-bucket ai4bharat-data --mode test

# Test data to both local and S3
python download.py --storage both --s3-bucket ai4bharat-data --mode test
```

#### Full Mode Examples

```bash
# Full datasets locally (⚠️ ~255 GB)
python download.py --storage local --mode full

# Full datasets to S3 (recommended)
python download.py --storage s3 --s3-bucket ai4bharat-data --mode full

# Full datasets to both (⚠️ requires lots of disk space)
python download.py --storage both --s3-bucket ai4bharat-data --mode full
```

#### Advanced Examples

```bash
# Download only NCERT with custom region
python download.py --datasets ncert --storage s3 --s3-bucket my-bucket --region us-west-2 --mode full

# Use custom config file
python download.py --config prod-config.yml --mode full

# See all options
python download.py --help
```

## 📁 Output Structure

### Local Storage

```
./data/downloaded_datasets/
├── sangraha/
│   ├── hin_Deva/
│   │   ├── part-00000.jsonl
│   │   └── part-00001.jsonl
│   ├── hin_Latn/
│   │   ├── part-00000.jsonl
│   │   └── part-00001.jsonl
│   └── tam_Taml/
│       ├── part-00000.jsonl
│       └── part-00001.jsonl
├── indiccorp_v2/
│   └── hin/
│       ├── part-00000.jsonl
│       ├── part-00001.jsonl
│       ├── part-00002.jsonl
│       ├── part-00003.jsonl
│       ├── part-00004.jsonl
│       └── part-00005.jsonl
└── ncert/
    ├── part-00000.jsonl
    ├── part-00001.jsonl
    └── ...
```

### S3 Storage

```
s3://YOUR-BUCKET/
└── hf/
    ├── sangraha/
    │   ├── hin_Deva/
    │   ├── hin_Latn/
    │   └── tam_Taml/
    ├── indiccorp_v2/
    │   └── hin/
    └── ncert/
```

## ⚙️ Configuration

### config.yml

```yaml
# Storage configuration
storage:
  mode: local       # local | s3 | both
  local_dir: ./data/downloaded_datasets
  
aws:
  region: us-east-1
  s3_bucket: my-llm-datasets
  s3_prefix: hf

# Mode: 'test' applies limits below, 'full' downloads entire datasets
mode: test        # test | full

datasets:
  sangraha:
    repo: ai4bharat/sangraha
    subset: synthetic
    languages:
      - hin_Deva
      - hin_Latn
      - tam_Taml
    test_limit:
      type: rows
      value: 10000

  indiccorp_v2:
    repo: ai4bharat/IndicCorpV2
    name: indiccorp_v2
    split: hin_Deva
    test_limit:
      type: rows
      value: 50000

  ncert:
    repo: ParthKadam2003/NCERT_Dataset
    split: train
    test_limit:
      type: rows
      value: 10000
```

**Note**: Command-line arguments override config.yml values.

## 🔧 AWS Setup

### 1. Configure AWS Credentials

```bash
aws configure
```

Enter:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., `us-east-1`)
- Default output format (`json`)

### 2. Create S3 Bucket (if needed)

```bash
aws s3 mb s3://YOUR-BUCKET-NAME --region us-east-1
```

### 3. Verify Access

```bash
aws s3 ls s3://YOUR-BUCKET-NAME/
```

## 📊 Expected Output

```
============================================================
🚀 Dataset Download Tool
============================================================
🔧 Using storage mode from command line: local
🔧 Using download mode from command line: test

⚙️  Mode: test
💾 Storage: local
📁 Local dir: ./data/downloaded_datasets

============================================================
🚀 Processing: sangraha
============================================================
  📥 Loading Sangraha dataset...
  🌐 Language: hin_Deva
  💾 Saved: ./data/downloaded_datasets\sangraha/hin_Deva\part-00000.jsonl
  📊 Progress: 10,000 records
  ✅ hin_Deva: 10,000 records, 2 shards
  🌐 Language: hin_Latn
  💾 Saved: ./data/downloaded_datasets\sangraha/hin_Latn\part-00000.jsonl
  ✅ hin_Latn: 10,000 records, 2 shards
  🌐 Language: tam_Taml
  💾 Saved: ./data/downloaded_datasets\sangraha/tam_Taml\part-00000.jsonl
  ✅ tam_Taml: 10,000 records, 2 shards
⏱️  Time: 59.02s

============================================================
🚀 Processing: indiccorp_v2
============================================================
  📥 Loading IndicCorp V2 (split: hin_Deva)...
  💾 Saved: ./data/downloaded_datasets\indiccorp_v2/hin\part-00000.jsonl
  📊 Progress: 10,000 records
  💾 Saved: ./data/downloaded_datasets\indiccorp_v2/hin\part-00001.jsonl
  📊 Progress: 20,000 records
  ✅ Total: 50,000 records, 6 shards
⏱️  Time: 7.66s

============================================================
📊 SUMMARY
============================================================
✅ sangraha: 59.02s
✅ indiccorp_v2: 7.66s

⏱️  TOTAL TIME: 66.68s
============================================================
🎉 Done!
```

## 💡 Recommended Workflows

### Workflow 1: Test First, Then Full

```bash
# 1. Test with small data locally
python download.py --storage local --mode test

# 2. Verify data
dir .\data\downloaded_datasets

# 3. Upload test data to S3
aws s3 sync .\data\downloaded_datasets s3://YOUR-BUCKET/Testing/

# 4. Download full datasets to S3 (when ready)
python download.py --storage s3 --s3-bucket YOUR-BUCKET --mode full
```

### Workflow 2: Direct to S3

```bash
# Download full datasets directly to S3 (no local storage needed)
python download.py --storage s3 --s3-bucket YOUR-BUCKET --mode full
```

### Workflow 3: Local Backup + S3

```bash
# Download and backup to both local and S3
python download.py --storage both --s3-bucket YOUR-BUCKET --mode full
```

## 📈 Data Sizes

### Test Mode
- **Sangraha**: ~5 MB (30,000 records)
- **IndicCorp V2**: ~10 MB (50,000 records)
- **NCERT**: ~2 MB (10,000 records)
- **Total**: ~17 MB
- **Time**: ~1-2 minutes

### Full Mode
- **Sangraha**: ~50 GB (all languages, complete dataset)
- **IndicCorp V2**: ~200 GB (Hindi complete corpus)
- **NCERT**: ~5 GB (complete educational content)
- **Total**: ~255 GB
- **Time**: Several hours (depends on internet speed)

## 🐛 Troubleshooting

### Error: "Unable to locate credentials"
```bash
# Configure AWS credentials
aws configure
```

### Error: "Invalid bucket name"
- Bucket names cannot contain slashes or special characters
- Use only lowercase letters, numbers, hyphens, and dots
- Example: `my-bucket-name` ✅, `my-bucket/path` ❌

### Error: "Access Denied"
- Check IAM permissions for S3 access
- Ensure bucket policy allows PutObject

### Slow Downloads
- Use `--storage s3` to skip local storage
- Check internet connection
- Consider running on AWS EC2 for faster S3 uploads

## 📝 Additional Resources

- **AWS S3 Guide**: See `AWS_S3_GUIDE.md` for detailed AWS setup
- **HuggingFace Datasets**: https://huggingface.co/docs/datasets/
- **Sangraha Dataset**: https://huggingface.co/datasets/ai4bharat/sangraha
- **IndicCorp V2**: https://huggingface.co/datasets/ai4bharat/IndicCorpV2

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `AWS_S3_GUIDE.md` for AWS-specific issues
3. Verify your config.yml is properly formatted
4. Check AWS credentials and permissions

## 📄 License

MIT License - Feel free to use and modify as needed.
