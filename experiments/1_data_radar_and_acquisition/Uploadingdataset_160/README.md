# Dataset Download Tool

A flexible tool to download HuggingFace datasets to local storage and/or upload to S3 with comprehensive timing tracking.

## Features

✅ **Multiple Storage Modes**: Download to local, S3, or both  
✅ **Test & Full Modes**: Test with limited data or download complete datasets  
✅ **Comprehensive Timing**: Track time for each step and total execution  
✅ **Progress Tracking**: Real-time progress with record counts  
✅ **Error Handling**: Graceful error handling with detailed logging  
✅ **Flexible Configuration**: YAML-based configuration for easy customization

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### Storage Modes

Edit `config.yml` to set your storage preference:

```yaml
storage:
  mode: local       # Options: local | s3 | both
  local_dir: ./data/downloaded_datasets
```

- **`local`**: Download datasets to local filesystem only
- **`s3`**: Upload datasets to S3 only (requires AWS credentials)
- **`both`**: Download locally AND upload to S3

### Download Modes

#### Test Mode (Limited Download)
For testing with a subset of data:

```yaml
mode: test        # Use test limits defined below

datasets:
  sangraha:
    repo: ai4bharat/sangraha
    subset: synthetic
    split: train
    test_limit:
      type: percent
      value: 1        # Download only 1% of data
    s3_path: sangraha
    local_path: sangraha
```

**Test Limit Types:**
- `type: percent` - Download a percentage of the dataset
- `type: rows` - Download a specific number of rows
- `type: none` - Download the entire dataset (ignores mode setting)

#### Full Mode (Complete Download)
For downloading entire datasets:

```yaml
mode: full        # Download complete datasets

datasets:
  sangraha:
    repo: ai4bharat/sangraha
    subset: synthetic
    split: train
    test_limit:
      type: none      # Download full dataset
    s3_path: sangraha
    local_path: sangraha
```

**Note**: When `mode: full`, all test limits are ignored (except `type: none` which is always respected).

### AWS Configuration

If using S3 storage, configure your AWS settings:

```yaml
aws:
  region: us-east-1
  s3_bucket: my-llm-datasets
  s3_prefix: hf
```

Make sure you have AWS credentials configured:
```bash
aws configure
# OR set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

## Dataset Configuration Examples

### Example 1: Percentage-based Limit
```yaml
sangraha:
  repo: ai4bharat/sangraha
  subset: synthetic
  split: train
  test_limit:
    type: percent
    value: 1        # 1% for testing
  s3_path: sangraha
  local_path: sangraha
```

### Example 2: Row-based Limit
```yaml
indiccorp_v2:
  repo: ai4bharat/IndicCorpV2
  name: hin_Deva
  split: train
  test_limit:
    type: rows
    value: 50000    # First 50,000 rows
  s3_path: indiccorp_v2/hin
  local_path: indiccorp_v2/hin
```

### Example 3: Full Dataset
```yaml
dolma:
  repo: allenai/dolma
  name: v1_6-sample
  split: train
  test_limit:
    type: none      # Download everything
  s3_path: dolma/v1_6_sample
  local_path: dolma/v1_6_sample
```

## Usage

### Run the Download Script

```bash
python download.py
```

### Output Example

```
🔧 Configuration:
   Storage mode: local
   Local directory: ./data/downloaded_datasets
   Mode: full
   S3 bucket: my-llm-datasets/hf
   Start time: 2026-01-31 13:16:00

============================================================
🚀 Processing dataset: sangraha
============================================================
📥 Loading from: ai4bharat/sangraha
   Subset/Config: synthetic
   ⏱️  Load time: 2.34s
📊 Processing records (shard size: 10,000)...
💾 Saved locally: ./data/downloaded_datasets/sangraha/part-00000.jsonl
   ⏱️  Save time: 0.45s
   Progress: 10,000 records processed
💾 Saved locally: ./data/downloaded_datasets/sangraha/part-00001.jsonl
   ⏱️  Save time: 0.43s
   Progress: 20,000 records processed

✅ Completed sangraha
   Total records: 25,432
   Total shards: 3
   ⏱️  Dataset time: 5m 23.45s

============================================================
📊 SUMMARY
============================================================
✅ sangraha:
   Records: 25,432
   Shards: 3
   Time: 5m 23.45s
✅ indiccorp_v2:
   Records: 1,234,567
   Shards: 124
   Time: 2h 15m 34.12s

============================================================
⏱️  TOTAL TIME: 2h 20m 57.57s
   End time: 2026-01-31 15:36:57
============================================================
🎉 All datasets processed!
============================================================
```

## Timing Features

The script tracks timing at multiple levels:

1. **Load Time**: Time to load dataset from HuggingFace
2. **Save Time**: Time to save each shard (local/S3)
3. **Dataset Time**: Total time per dataset
4. **Total Time**: Complete execution time

All times are formatted as:
- `< 60s`: `23.45s`
- `< 1h`: `5m 23.45s`
- `≥ 1h`: `2h 15m 34.12s`

## File Structure

After running, your local directory will look like:

```
./data/downloaded_datasets/
├── sangraha/
│   ├── part-00000.jsonl
│   ├── part-00001.jsonl
│   └── part-00002.jsonl
├── indiccorp_v2/
│   └── hin/
│       ├── part-00000.jsonl
│       └── ...
└── dolma/
    └── v1_6_sample/
        ├── part-00000.jsonl
        └── ...
```

## Tips

### For Testing
1. Set `mode: test`
2. Use small `test_limit` values
3. Use `storage.mode: local` to avoid S3 costs

### For Production
1. Set `mode: full`
2. Set all `test_limit.type: none`
3. Use `storage.mode: both` for redundancy
4. Monitor disk space and S3 costs

### Performance
- Each shard contains 10,000 records
- Adjust `shard_size` in code if needed
- Use streaming mode for large datasets (already enabled)

## Troubleshooting

**Issue**: AWS credentials not found  
**Solution**: Run `aws configure` or set environment variables

**Issue**: Out of disk space  
**Solution**: Increase disk space or use `storage.mode: s3`

**Issue**: Dataset not found  
**Solution**: Check repo name and subset/name in config

**Issue**: Slow downloads  
**Solution**: Check internet connection, use smaller test limits first

## License

MIT
