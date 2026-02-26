# AWS Glue Scripts Documentation

This folder contains scripts and documentation for processing datasets using AWS Glue as part of the dataset normalization pipeline.

## Overview

The scripts in this directory are designed to process large-scale datasets using AWS Glue, PySpark, and related tools. The main processing scripts are:

- `DolmaProcessing.py`
- `NCRTProcessing.py`
- `SangrahaProcessing.py`

Each script is tailored for specific dataset normalization and transformation tasks.

## Running AWS Glue Scripts

### Prerequisites
- AWS account with permissions to run Glue jobs
- S3 buckets for input and output data
- IAM roles with necessary Glue and S3 permissions
- Python environment with required dependencies (see `requirements.txt`)

### Parameters
Each script accepts parameters that control its behavior. Common parameters include:
- `--input_path`: S3 path to the input dataset
- `--output_path`: S3 path for the processed output
- `--temp_dir`: S3 path for temporary Glue job data
- `--job_name`: Name of the Glue job
- Additional script-specific parameters (see script docstrings)

### Example Command
To run a Glue script with parameters:

```
aws glue start-job-run \
  --job-name <GlueJobName> \
  --arguments '{"--input_path":"s3://your-bucket/input/", "--output_path":"s3://your-bucket/output/", "--temp_dir":"s3://your-bucket/temp/"}'
```

Or, for local testing with PySpark:

```
spark-submit DolmaProcessing.py \
  --input_path s3://your-bucket/input/ \
  --output_path s3://your-bucket/output/ \
  --temp_dir s3://your-bucket/temp/ \
  --job_name test_job
```

## Script Details
- **DolmaProcessing.py**: Handles normalization and transformation for Dolma datasets.
- **NCRTProcessing.py**: Processes NCRT datasets with custom logic.
- **SangrahaProcessing.py**: Specialized for Sangraha dataset workflows.

Each script contains detailed docstrings and comments for further guidance.

## Additional Resources
- [AWS Glue Documentation](https://docs.aws.amazon.com/glue/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)

For questions or issues, please contact the project maintainers.
