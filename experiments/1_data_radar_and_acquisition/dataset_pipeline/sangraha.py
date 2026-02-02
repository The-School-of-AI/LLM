"""Sangraha dataset downloader with streaming support."""

import os
import torch
from typing import Optional

from datasets import load_dataset
from huggingface_hub import login

from common import (
    ChunkedWriter,
    DEFAULT_CHUNK_SIZE,
    OutputFormat,
    S3Storage,
    StorageType,
    clear_progress,
    compute_hash,
    get_download_path,
    load_progress,
    save_progress,
)

# Authenticate with HuggingFace if token is available
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

# Dataset identifier
DATASET_NAME = "Sangraha"

# Available type tags for filtering
TYPES = ["web", "pdf", "ocr"]

# Available language splits
LANGUAGES = [
    "asm", "ben", "bod", "brx", "doi", "gom", "guj", "hin", "kan", "kas",
    "mai", "mal", "mar", "mni", "nep", "ori", "pan", "san", "sat", "snd",
    "tam", "tel", "urd"
]

# Language code to full name mapping
LANGUAGE_NAMES = {
    "asm": "Assamese",
    "ben": "Bengali",
    "bod": "Tibetan",
    "brx": "Bodo",
    "doi": "Dogri",
    "gom": "Konkani",
    "guj": "Gujarati",
    "hin": "Hindi",
    "kan": "Kannada",
    "kas": "Kashmiri",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mni": "Manipuri",
    "nep": "Nepali",
    "ori": "Odia",
    "pan": "Punjabi",
    "san": "Sanskrit",
    "sat": "Santali",
    "snd": "Sindhi",
    "tam": "Tamil",
    "tel": "Telugu",
    "urd": "Urdu",
}

# Progress update interval
PROGRESS_INTERVAL = 1000


def download(
    num_records: int = 10,
    lang: str = "hin",
    resume: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    output_format: OutputFormat = "json",
    storage_type: StorageType = "local",
    s3_storage: Optional[S3Storage] = None,
) -> int:
    """Download records from Sangraha dataset.
    
    Args:
        num_records: Number of records to download
        lang: Language split (e.g., 'hin', 'mar', 'ben', 'tam', etc.)
        resume: If True, resume from previous progress
        chunk_size: Number of records per output file
        output_format: Output format ('json' or 'parquet')
        storage_type: Storage type ('local' or 's3')
        s3_storage: S3Storage instance (required if storage_type='s3')
        
    Returns:
        Total number of records downloaded
    """
    # Check for existing progress if resume mode
    skip_count = 0
    
    if resume:
        progress = load_progress("sangraha", lang, output_format, s3_storage)
        if progress:
            skip_count = progress.get("downloaded_count", 0)
            print(f"Resuming from record {skip_count}, targeting {num_records} total records")
            
            # If we already have enough records, return
            if skip_count >= num_records:
                print(f"Already have {skip_count} records, nothing to download")
                return skip_count
    
    if num_records is not None:
        print(f"--- Downloading {num_records - skip_count} records from Sangraha ({lang}) ---")
    else:
        print(f"--- Downloading all available records from Sangraha ({lang}) ---")
    print(f"Chunk size: {chunk_size} records per file, format: {output_format}")
    
    # Initialize chunked writer
    writer = ChunkedWriter(
        dataset="sangraha",
        subfolder=lang,
        base_filename="records",
        chunk_size=chunk_size,
        use_custom_serializer=False,
        resume_from_record=skip_count,
        output_format=output_format,
        storage_type=storage_type,
        s3_storage=s3_storage,
    )
    
    dataset = load_dataset("ai4bharat/sangraha", "verified", split=lang, streaming=True)
    
    # Use DataLoader with prefetching for faster iteration
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=10,
        collate_fn=passthrough_collate,
    )

    count = 0
    lang_name = LANGUAGE_NAMES.get(lang, lang)

    try:
        for batch in dataloader:
            example = batch[0]
            # Skip already downloaded records
            if count < skip_count:
                count += 1
                continue
                
            if num_records is not None and count >= num_records:
                break
                
            text = example.get("text", "")
            record = {
                "id": example.get("doc_id"),
                "hash": compute_hash(text),
                "dataset": DATASET_NAME,
                "domain": "web",
                "source": None,
                "text": text,
                "language": lang_name,
                "metadata": {
                    "language_code": lang,
                    "type": example.get("type", "unknown"),
                },
                "added": None,
                "created": None,
                "version": None,
            }
            writer.add_record(record)
            count += 1
            
            # Print progress
            if count % PROGRESS_INTERVAL == 0:
                save_progress("sangraha", lang, count, num_records, output_format, s3_storage)
                stats = writer.get_stats()
                print(f"Progress: {count}/{num_records} records, {stats['files_written']} files written")
    
    except (KeyboardInterrupt, Exception) as e:
        print(f"\nInterrupted: {e}")
        writer.flush()
        save_progress("sangraha", lang, count, num_records, output_format, s3_storage)
        print(f"Progress saved. Run with --resume to continue from record {count}")
        raise
    
    # Finalize writing
    writer.finalize()
    clear_progress("sangraha", lang, output_format, s3_storage)
    
    return count


def passthrough_collate(batch):
    return batch
