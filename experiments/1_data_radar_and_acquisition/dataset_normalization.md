# End-to-End Dataset Preparation for LLM Model Training

Preparing high-quality datasets is a critical step in training Large Language Models (LLMs). This documentation outlines the end-to-end process, architecture, and schema used for dataset preparation, focusing on three main models: Dolma, Sangraha, and NCERT.

## Finalized Dataset Strategy

The following datasets were selected for LLM training:
- **Dolma**
- **Sangraha**
- **NCERT**

## High-Level Architecture Diagram

```mermaid
graph TD
    A[Hugging Face Datasets]
    A --> B[Sangraha Download Component]
    A --> C[Dolma Download Component]
    A --> D[NCERT Download Component]
    B --> E[AWS S3 (Direct Stream)]
    C --> F[Local Storage (AWS EBS)]
    F --> G[Dolma Upload Component]
    G --> E
    D --> E
    E --> H[AWS Glue Dataset Normalization Pipeline]
    H --> I[Normalized Parquet Files (Common Schema)]
```

## Dataset Download & Ingestion

- **Sangraha**: Downloaded from Hugging Face and streamed directly to AWS S3 using the Sangraha download component.
- **Dolma**: Downloaded from Hugging Face to local AWS EBS storage, then uploaded to AWS S3 using the Dolma upload component.
- **NCERT**: Downloaded from Hugging Face and uploaded to AWS S3.

## AWS S3 as Central Data Lake

All datasets are stored in AWS S3, making them accessible for downstream processing and normalization.

## Dataset Normalization Pipeline

AWS Glue jobs are triggered for each dataset to normalize the data into a unified schema and file format. The pipeline ensures consistency and adds additional metadata fields.

- **File Format:** Parquet
- **Schema:** Common across all datasets

## Parquet Schema

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

## Summary

This pipeline ensures that all datasets are consistently formatted, enriched with metadata, and ready for LLM model training. The use of AWS Glue and Parquet enables scalable, efficient processing and storage.

For further details, refer to the dataset_normalization_pipeline and AWS Glue job documentation in this repository.
