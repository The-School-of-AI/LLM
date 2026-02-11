# Professional Data Processing & Deduplication Pipeline

This repository contains a high-performance, modular pipeline for processing, transforming, and deduplicating LLM pretraining datasets stored in Amazon S3. It is designed to handle millions of records efficiently using a parallel Producer-Consumer architecture.

---

## 📁 Project Structure

```text
curriculum_data_processing/
├── config/
│   └── config.yaml           # Centralized configuration (S3, Schema, Logging)
├── src/
│   ├── logger.py            # Logging setup (UTF-8, Windows compatible)
│   ├── s3_utils.py          # S3 filesystem interaction and discovery
│   └── data_processor.py     # Core Parallel Producer-Consumer logic
├── main.py                   # CLI Entry point with .env support
├── .env                      # AWS Credentials (Access Key, Secret, Region)
├── requirements.txt          # Python dependencies
└── .gitignore                # Git exclusions
```

---

## 🛡️ Global Deduplication Strategy

The deduplication in this pipeline is **strictly Global**. It ensures that every record in your final output is unique, regardless of which source or band it originated from.

- **Single Set**: The pipeline maintains one master list of hashes (`seen_hashes`) for the entire execution in RAM.
- **Cross-Everything**: Whether a record comes from `source=arxiv` in `band=B0` or `source=books` in `band=B5`, if the hash has been seen before *anywhere* in the current run, it is dropped.
- **Keep-First Policy**: The very first time a hash is encountered, the record is kept and the hash is added to the global set. All subsequent occurrences are discarded immediately.
- **Result**: The final output contains only globally unique records across your entire processed dataset.

---

## 🔄 Execution Logic & Process

The pipeline follows a sophisticated 4-phase process designed to minimize I/O wait times and maximize CPU occupancy.

1. **Discovery**: The `s3_utils` module crawls the S3 bucket using the provided prefix. It looks for a structure like `source=<SOURCE_NAME>/bands/band=<BAND_NAME>/` and returns an optimized map of every valid "bucket-prefix" combination.
2. **Production (Parallel)**: Using `concurrent.futures.ThreadPoolExecutor`, the pipeline spawns workers to stream Parquet files directly from S3, perform column renaming, metadata injection, and record filtering.
3. **Consumption (Sequential)**: A single thread receives transformed records, performs the **Global Dedup Check**, and writes to the `RecordWriter`.
4. **Finalization**: Once workers finish, all file handles are closed, and a comprehensive audit report is printed.

---

## 🛠️ Setup & Requirements

Follow these steps to set up the pipeline on your local machine or an AWS EC2 instance:

### 1. Environment Preparation
It is recommended to use a virtual environment to manage dependencies:
```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies
Run the following command to install all necessary libraries (PyArrow, Boto3, python-dotenv, etc.):
```bash
pip install -r requirements.txt
```

### 3. AWS Credentials Configuration
Create a file named `.env` in the project root folder. The pipeline will automatically load these keys at runtime:
```text
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

### 4. Verify Configuration
Open `config/config.yaml` and ensure the following match your S3 dataset:
- `bucket`: The name of your S3 bucket.
- `base_prefix`: The path to the root of your curriculum data.

---

## 📖 Command Line Arguments

The pipeline provides several command-line arguments that allow you to override settings in `config.yaml` without modifying the file.

### 🔍 Argument Descriptions

- **`--config`**: Path to the YAML configuration file. Use this if you have multiple configuration profiles (e.g., `config_prod.yaml`, `config_test.yaml`).
- **`--bands`**: A space-separated list of bands to process (e.g., `B0 B1 B5`). If not provided, the pipeline automatically discovers all bands available in the S3 bucket.
- **`--output-dir`**: Specifies the local directory where processed files will be saved. The directory will be created automatically if it doesn't exist.
- **`--output-mode`**: Determines how records are organized in the output:
    - `per_band`: (Default) Creates a separate file for each band.
    - `single`: Merges all processed records into one large `output.jsonl` file.
    - `sharded`: Splits the output into multiple files based on the `--shard-size`.
- **`--shard-size`**: The number of records to include per output file. When the limit is reached, the pipeline rotates to a new file (e.g., `shard_0.jsonl`, `shard_1.jsonl`).
- **`--workers`**: The number of parallel threads used to download and transform data from S3. Increasing this can significantly speed up processing on multi-core systems.
- **`--log-level`**: Sets the minimum logging level to display. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`.

### 📋 Argument Reference Table

| Argument | Full Flag | Default |
| :--- | :--- | :--- |
| Config Path | `--config <path>` | `config/config.yaml` |
| Targeted Bands | `--bands <B1 B2 ...>` | All bands found in S3 |
| Output Directory| `--output-dir <path>` | `output/` |
| Output Mode | `--output-mode <mode>` | `per_band` |
| Shard Size | `--shard-size <int>` | No Sharding |
| Parallel Workers| `--workers <int>` | System CPU Count |
| Log Level | `--log-level <LEVEL>` | `INFO` |

### 🚀 Usage Examples

#### 1. Quick Test (Single Band, Small Batch)
Process only band `B0` and shard the output every 500 records for easy manual review.
```bash
python main.py --bands B0 --output-mode per_band --shard-size 500
```

#### 2. Full Merged Output
Combine every source and every band into one massive single `output.jsonl` file.
```bash
python main.py --output-mode single
```

#### 3. High-Performance Server Run
Optimized for 16-core machines with detailed debugging logs enabled.
```bash
python main.py --workers 16 --log-level DEBUG
```

#### 4. Sharded Global Output
Process all data and split the output into chunks of 100,000 records each, regardless of band.
```bash
python main.py --output-mode sharded --shard-size 100000
```

---

## 📊 Sample Output Record

```json
{
  "source_doc_id": "part-00000-57c53cda-106f-4746-bd71-552be0a7d864.c000.zstd.parquet",
  "source_url": "s3://t2-datacurriculum-353/processed_dataset/curriculum_data/source=books/bands/band=B0/",
  "band": "B0",
  "chunk_id": "064bbeb9-e655-4b19-86c4-5b0e47a189d9",
  "source": "books",
  "domain": "literature",
  "language": "en",
  "band_score": 0.5951465457335166,
  "difficulty_score": 0.09526189810928569,
  "word_count": 1076,
  "token_count_estimate": 1398
}
```

---

## 📜 Statistics Report
At the end of each run, the pipeline logs a summary:
- **Total rows read/kept/dropped**: Audit of the Global Dedup performance.
- **Duplicate rate %**: Percentage of rows removed from the total.
- **Per-source breakdown**: Detailed stats showing how many records were kept from each source.
- **Word count impact**: The total number of words retained in your final processed corpus.
