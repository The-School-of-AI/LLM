"""Dataset downloader - Downloads records from Dolma, Sangraha, IndicCorp and NCERT datasets."""


import argparse
import logging
import os
import sys
from common import set_data_dir, S3Storage, SCOPE_CHUNK_SIZES
import ncert
import dolma
import sangraha
import indiccorp

# Setup logging

# Set log directory in the current working directory
LOG_DIR = os.path.join(os.getcwd(), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'dataset_pipeline.log')
os.makedirs(LOG_DIR, exist_ok=True)

# Set up per-dataset log file
def set_dataset_log_file(dataset):
    dataset_log_file = os.path.join(LOG_DIR, f"{dataset}.log")
    file_handler = RotatingFileHandler(
        dataset_log_file, maxBytes=100*1024*1024, backupCount=5
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    # Remove all handlers associated with the root logger object (to avoid duplicate logs)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler
        ]
    )

# Scope configuration: maps scope name to number of records
SCOPE_CONFIG = {
    "test": 10,
    "validate": 10000,
    "pre-prod": 100000,
    "production": None,  # Full dataset
}

# Available datasets
DATASETS = ["dolma", "sangraha", "indiccorp", "ncert"]

# Supported output formats
FORMATS = ["json", "parquet"]

# Supported storage types
STORAGE_TYPES = ["local", "s3"]


def main():
    parser = argparse.ArgumentParser(description="Download records from various datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        required=True,
        help="Dataset to download from: 'dolma', 'sangraha', 'indiccorp', or 'ncert'"
    )
    parser.add_argument(
        "--dolma-urls-file",
        type=str,
        default=None,
        help="Path to local file containing Dolma data URLs (one per line). If provided, overrides default Dolma URLs."
    )
    parser.add_argument(
        "--scope",
        type=str,
        choices=["test", "validate", "pre-prod", "production"],
        default="test",
        help="Scope of download: 'test' (10), 'validate' (10000), 'pre-prod' (100000), 'production' (full)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume download from previous progress if interrupted"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Language code for sangraha (e.g., 'hin') or indiccorp (e.g., 'hi')"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of records per output file (default: auto based on scope)"
    )
    # Removed --subject argument for NCERT - downloads all subjects
    parser.add_argument(
        "--format",
        type=str,
        choices=FORMATS,
        default="json",
        help="Output file format: 'json' or 'parquet' (default: json)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        choices=STORAGE_TYPES,
        default="local",
        help="Storage type: 'local' or 's3' (default: local)"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket name (required if --storage=s3)"
    )
    parser.add_argument(
        "--s3-region",
        type=str,
        default="us-east-1",
        help="AWS region for S3 bucket (default: us-east-1)"
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="datasets",
        help="S3 key prefix/folder (default: datasets)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".data",
        help="Output directory path for local storage (default: .data)"
    )
    args = parser.parse_args()
    
    # Validate S3 arguments
    if args.storage == "s3" and not args.s3_bucket:
        parser.error("--s3-bucket is required when --storage=s3")
    
    # Set the output directory for local storage
    set_data_dir(args.output_dir)
    
    # Initialize S3 storage if needed
    s3_storage = None
    if args.storage == "s3":
        s3_storage = S3Storage(
            bucket_name=args.s3_bucket,
            region=args.s3_region,
            prefix=args.s3_prefix,
        )
    
    if args.scope == "production":
        num_records = None
    else:
        num_records = SCOPE_CONFIG[args.scope]
    # Auto-select chunk size based on scope if not provided
    chunk_size = args.chunk_size if args.chunk_size is not None else SCOPE_CHUNK_SIZES[args.scope]
    resume_msg = " (resume mode)" if args.resume else ""
    storage_msg = f"s3://{args.s3_bucket}/{args.s3_prefix}" if args.storage == "s3" else "local"
    logging.info(f"Running with dataset={args.dataset}, scope={args.scope}, records={num_records or 'full'}{resume_msg}")
    logging.info(f"Chunk size: {chunk_size} records per file, format: {args.format}, storage: {storage_msg}")
    
    if args.dataset == "dolma":
        try:
            count = dolma.download(
                num_records, resume=args.resume, chunk_size=chunk_size,
                output_format=args.format, storage_type=args.storage, s3_storage=s3_storage,
                urls_file=args.dolma_urls_file
            )
            logging.info(f"Downloaded {count} Dolma records")
        except KeyboardInterrupt:
            print("\nDownload interrupted. Use --resume to continue later.")
        except Exception as e:
            logging.error(f"Error downloading Dolma: {e}")
    
    elif args.dataset == "sangraha":
        lang = args.lang or "hin"
        try:
            count = sangraha.download(
                num_records, lang=lang, resume=args.resume, chunk_size=chunk_size,
                output_format=args.format, storage_type=args.storage, s3_storage=s3_storage
            )
            logging.info(f"Downloaded {count} Sangraha records")
        except KeyboardInterrupt:
            print("\nDownload interrupted. Use --resume to continue later.")
        except Exception as e:
            logging.error(f"Error downloading Sangraha: {e}")
    
    elif args.dataset == "indiccorp":
        lang = args.lang or "hi"
        try:
            count = indiccorp.download(
                num_records, lang=lang, resume=args.resume, chunk_size=chunk_size,
                output_format=args.format, storage_type=args.storage, s3_storage=s3_storage
            )
            logging.info(f"Downloaded {count} IndicCorp records")
        except KeyboardInterrupt:
            print("\nDownload interrupted. Use --resume to continue later.")
        except Exception as e:
            logging.error(f"Error downloading IndicCorp: {e}")
    
    elif args.dataset == "ncert":
        try:
            count = ncert.download(
                num_records, resume=args.resume, chunk_size=chunk_size,
                output_format=args.format, storage_type=args.storage, s3_storage=s3_storage
            )
            logging.info(f"Downloaded {count} NCERT records")
        except KeyboardInterrupt:
            print("\nDownload interrupted. Use --resume to continue later.")
        except Exception as e:
            logging.error(f"Error downloading NCERT: {e}")


if __name__ == "__main__":
    main()
