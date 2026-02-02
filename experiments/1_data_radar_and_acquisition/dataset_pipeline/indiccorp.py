"""IndicCorp v2 dataset downloader with streaming support."""

from typing import Optional

from datasets import load_dataset

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

# Dataset identifier
DATASET_NAME = "IndicCorp_v2"

# Available languages (file codes in the dataset)
# Format: code -> (file_pattern, language_name)
LANGUAGES = {
    "as": ("as", "Assamese"),
    "bd": ("bd", "Bodo"),
    "bn": ("bn", "Bengali"),
    "dg": ("dg", "Dogri"),
    "en": ("en", "English"),
    "gom": ("gom", "Konkani"),
    "gu": ("gu", "Gujarati"),
    "hi": ("hi-1", "Hindi"),  # Hindi has multiple files: hi-1, hi-2, hi-3
    "kha": ("kha", "Khasi"),
    "kn": ("kn", "Kannada"),
    "ks": ("ks", "Kashmiri"),
    "mai": ("mai", "Maithili"),
    "ml": ("ml", "Malayalam"),
    "mni": ("mni", "Manipuri"),
    "mr": ("mr", "Marathi"),
    "ne": ("ne", "Nepali"),
    "or": ("or", "Odia"),
    "pa": ("pa", "Punjabi"),
    "sa": ("sa", "Sanskrit"),
    "sat": ("sat", "Santali"),
    "sd": ("sd", "Sindhi"),
    "ta": ("ta", "Tamil"),
    "te": ("te", "Telugu"),
    "ur": ("ur", "Urdu"),
}

# Progress update interval
PROGRESS_INTERVAL = 1000


def download(
    num_records: int = 10,
    lang: str = "hi",
    resume: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    output_format: OutputFormat = "json",
    storage_type: StorageType = "local",
    s3_storage: Optional[S3Storage] = None,
) -> int:
    """Download records from IndicCorp v2 dataset.
    
    Args:
        num_records: Number of records to download
        lang: Language code (e.g., 'hi', 'ta', 'bn', etc.)
        resume: If True, resume from previous progress
        chunk_size: Number of records per output file
        output_format: Output format ('json' or 'parquet')
        storage_type: Storage type ('local' or 's3')
        s3_storage: S3Storage instance (required if storage_type='s3')
        
    Returns:
        Total number of records downloaded
    """
    # Validate language
    if lang not in LANGUAGES:
        available = ", ".join(sorted(LANGUAGES.keys()))
        raise ValueError(f"Invalid language '{lang}'. Available: {available}")
    
    file_code, lang_name = LANGUAGES[lang]
    
    # Check for existing progress if resume mode
    skip_count = 0
    
    if resume:
        progress = load_progress("indiccorp", lang, output_format, s3_storage)
        if progress:
            skip_count = progress.get("downloaded_count", 0)
            print(f"Resuming from record {skip_count}, targeting {num_records} total records")
            
            # If we already have enough records, return
            if skip_count >= num_records:
                print(f"Already have {skip_count} records, nothing to download")
                return skip_count
    
    print(f"--- Downloading {num_records - skip_count} records from IndicCorp v2 ({lang_name}) ---")
    print(f"Chunk size: {chunk_size} records per file, format: {output_format}")
    
    # Initialize chunked writer
    writer = ChunkedWriter(
        dataset="indiccorp",
        subfolder=lang,
        base_filename="records",
        chunk_size=chunk_size,
        use_custom_serializer=False,
        resume_from_record=skip_count,
        output_format=output_format,
        storage_type=storage_type,
        s3_storage=s3_storage,
    )
    
    # Load dataset using text file
    dataset = load_dataset(
        "ai4bharat/IndicCorpV2",
        data_files=f"data/{file_code}.txt",
        split="train",
        streaming=True
    )
    
    count = 0
    try:
        for example in dataset:
            # Skip already downloaded records
            if count < skip_count:
                count += 1
                continue
                
            if count >= num_records:
                break
            
            # Text files have a single 'text' column per line
            text = example.get("text", "")
            
            record = {
                "id": f"{lang}_{count}",
                "hash": compute_hash(text),
                "dataset": DATASET_NAME,
                "domain": "web",
                "source": None,
                "text": text,
                "language": lang_name,
                "metadata": {
                    "language_code": lang,
                },
                "added": None,
                "created": None,
                "version": "v2",
            }
            writer.add_record(record)
            count += 1
            
            # Print progress
            if count % PROGRESS_INTERVAL == 0:
                save_progress("indiccorp", lang, count, num_records, output_format, s3_storage)
                stats = writer.get_stats()
                print(f"Progress: {count}/{num_records} records, {stats['files_written']} files written")
    
    except (KeyboardInterrupt, Exception) as e:
        print(f"\nInterrupted: {e}")
        writer.flush()
        save_progress("indiccorp", lang, count, num_records, output_format, s3_storage)
        print(f"Progress saved. Run with --resume to continue from record {count}")
        raise
    
    # Finalize writing
    writer.finalize()
    clear_progress("indiccorp", lang, output_format, s3_storage)
    
    return count
