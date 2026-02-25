# Dataset Download and Normalization: Detailed Report

This README provides a comprehensive guide for downloading and normalizing datasets for LLM training, referencing all relevant README files and process documentation in the `1_data_radar_and_acquisition` directory.

---

## 1. Dataset Download Approaches

### Sangraha
- **Streaming Approach:**
  - The Sangraha dataset is downloaded from Hugging Face and streamed directly to AWS S3 using the provided Python script (`sangraha_download.py`).
  - This approach avoids local disk usage, making it ideal for large datasets and limited local storage.
  - Usage and configuration details are in the [`sangraha_download/README.md`](../dataset_download_scripts/sangraha_download/README.md).
  - Example usage:
    ```bash
    uv run python sangraha_download.py --category verified --lang hin
    ```
  - The script lists files in the Hugging Face repo, streams each file, and uploads to S3. AWS credentials and S3 bucket must be configured.

### Dolma
- **Script-Based Batch Download:**
  - Due to the very large size of Dolma (~4TB), a different approach is used.
  - Dolma is downloaded in batches using shell scripts (`download_dolma_dataset.sh` and `download_dolma_all_dataset_batch.sh`).
  - Data is first downloaded to local AWS EBS storage, then uploaded to S3.
  - This method is necessary to handle the high volume and to allow for resumable, auditable downloads.
  - Usage and configuration details are in the [`dolma_download/README.md`](../dataset_download_scripts/dolma_download/README.md).

### NCERT
- Downloaded from Hugging Face and uploaded to S3 using a similar approach as Sangraha (see scripts and documentation in the relevant folders).

---

## 2. Centralized Storage: AWS S3
- All datasets are stored in AWS S3, serving as the central data lake for downstream processing.

---

## 3. Dataset Normalization Pipeline (AWS Glue)
- Once data is downloaded to S3, normalization is performed using AWS Glue jobs.
- Glue job scripts are located in the `dataset_normalization_pipeline` folder.
- See `AWS_GLUE_README.md` for detailed instructions and parameter options.
- **Default S3 paths and parameters are pre-configured** in the Glue scripts, but users can override them as needed (e.g., to change input/output locations or job names).
- Glue jobs transform all datasets into a unified schema and Parquet format, adding metadata and ensuring consistency.

---

## 4. Common Output Schema
- All normalized datasets follow a common schema, as described in `dataset_normalization.md`:

| Column    | Type    | Description                                 |
|-----------|---------|---------------------------------------------|
| id        | string  | Unique record identifier                    |
| hash      | string  | SHA-256 hash of text content                |
| dataset   | string  | Source dataset name                        |
| domain    | string  | Content domain (web, literature, education) |
| source    | string (nullable) | Source identifier (for Dolma)         |
| text      | string  | Main text content                          |
| language  | string  | Full language name                         |
| metadata  | string (JSON) | Additional fields as JSON string        |
| added     | string (nullable) | ISO timestamp when added              |
| created   | string (nullable) | ISO timestamp of creation             |
| version   | string (nullable) | Dataset version                       |

---

## 5. End-to-End Workflow Summary
1. **Download datasets** using the appropriate script/approach for each dataset (streaming for Sangraha, batch script for Dolma, etc.).
2. **Store all raw data in AWS S3.**
3. **Trigger AWS Glue jobs** (see `dataset_normalization_pipeline` and `AWS_GLUE_README.md`) to normalize and transform data into the common schema.
4. **Result:** All datasets are available in a unified, analysis-ready Parquet format in S3.

---

For further details, see the individual README files in each dataset's download folder and the AWS Glue documentation in the normalization pipeline directory.
