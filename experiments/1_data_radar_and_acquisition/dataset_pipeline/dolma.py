"""Dolma v1.7 dataset downloader with streaming support."""

import os
from typing import Optional

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download, login

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

# Progress update interval
PROGRESS_INTERVAL = 1000

# Dataset identifier
DATASET_NAME = "dolma_v1.7"

# Available source tags for filtering
SOURCES = [
    "books", "c4-filtered", "cc_en_head", "cc_en_middle", "cc_en_tail",
    "cc_news_head", "cc_news_middle", "cc_news_tail", "falcon-refinedweb-filtered",
    "pes2o", "proof_pile_2-algebraic_stack", "proof_pile_2-open_web_math",
    "reddit", "redpajama-arxiv", "redpajama-stackexchange", "starcoder",
    "tulu_flan", "wiki", "wikiref_megawika"
]

# Domain mapping for sources
DOMAIN_MAP = {
    "books": "literature",
    "gutenberg": "literature",
    "c4-filtered": "web",
    "cc_en_head": "web",
    "cc_en_middle": "web",
    "cc_en_tail": "web",
    "cc_news_head": "news",
    "cc_news_middle": "news",
    "cc_news_tail": "news",
    "falcon-refinedweb-filtered": "web",
    "pes2o": "science",
    "proof_pile_2-algebraic_stack": "math",
    "proof_pile_2-open_web_math": "math",
    "reddit": "social",
    "redpajama-arxiv": "science",
    "redpajama-stackexchange": "qa",
    "starcoder": "code",
    "tulu_flan": "instruction",
    "wiki": "encyclopedia",
    "wikiref_megawika": "encyclopedia",
}


def get_domain(source: str) -> str:
    """Get domain for a source.
    
    Args:
        source: The source identifier
        
    Returns:
        Domain string
    """
    if source in DOMAIN_MAP:
        return DOMAIN_MAP[source]
    # Try partial matching for compound sources
    for key, domain in DOMAIN_MAP.items():
        if key in source or source in key:
            return domain
    return "other"


def passthrough_collate(batch):
    """Pass through batch items without conversion - handles datetime objects."""
    return batch


def download(
    num_records: int = 10,
    resume: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    output_format: OutputFormat = "json",
    storage_type: StorageType = "local",
    s3_storage: Optional[S3Storage] = None,
) -> int:
    """Download records from Dolma v1.7 dataset.
    
    Args:
        num_records: Number of records to download
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
        progress = load_progress("dolma", "_all", output_format, s3_storage)
        if progress:
            skip_count = progress.get("downloaded_count", 0)
            print(f"Resuming from record {skip_count}, targeting {num_records} total records")
            
            if num_records is not None and skip_count >= num_records:
                print(f"Already have {skip_count} records, nothing to download")
                return skip_count
    
    if num_records is not None:
        print(f"--- Downloading {num_records - skip_count} records from Dolma v1.7 ---")
    else:
        print(f"--- Downloading all available records from Dolma v1.7 ---")
    print(f"Chunk size: {chunk_size} records per file, format: {output_format}")
    
    # Fetch the list of URLs for v1.7
    urls_path = hf_hub_download(
        repo_id="allenai/dolma", 
        filename="urls/v1_7.txt", 
        repo_type="dataset"
    )
    with open(urls_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # Load using the json builder since the data files are .json.gz
    dataset = load_dataset("json", data_files=urls[:5], split="train", streaming=True)
    
    # Use DataLoader with prefetching for faster iteration
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=10,
        collate_fn=passthrough_collate,
    )
    
    # Dictionary to hold ChunkedWriter instances per source
    # Note: For dolma, we track per-source record counts separately
    writers: dict[str, ChunkedWriter] = {}
    source_counts: dict[str, int] = {}
    
    count = 0
    try:
        for batch in dataloader:
            # Extract single example from batch (batch_size=1, custom collate returns list)
            example = batch[0]
            # Skip already downloaded records
            if count < skip_count:
                count += 1
                # Track source counts during skip phase for proper resume
                source = example.get("source", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
                continue
                
            if num_records is not None and count >= num_records:
                break
            
            source = example.get("source", "unknown")
            
            # Create writer for this source if not exists
            if source not in writers:
                # Resume from the count of records we skipped for this source
                resume_count = source_counts.get(source, 0)
                writers[source] = ChunkedWriter(
                    dataset="dolma",
                    subfolder=source,
                    base_filename="records",
                    chunk_size=chunk_size,
                    use_custom_serializer=True,
                    resume_from_record=resume_count,
                    output_format=output_format,
                    storage_type=storage_type,
                    s3_storage=s3_storage,
                )
            
            text = example.get("text", "")
            record = {
                "id": example.get("id"),
                "hash": compute_hash(text),
                "dataset": DATASET_NAME,
                "domain": get_domain(source),
                "source": source,
                "text": text,
                "language": "English",
                "metadata": example.get("metadata"),
                "added": example.get("added"),
                "created": example.get("created"),
                "version": example.get("version"),
            }
            writers[source].add_record(record)
            count += 1
            
            # Print progress
            if count % PROGRESS_INTERVAL == 0:
                save_progress("dolma", "_all", count, num_records, output_format, s3_storage)
                total_files = sum(w.get_stats()["files_written"] for w in writers.values())
                print(f"Progress: {count}/{num_records} records, {total_files} files written across {len(writers)} sources")
    
    except (KeyboardInterrupt, Exception) as e:
        print(f"\nInterrupted: {e}")
        # Flush all writers
        for writer in writers.values():
            writer.flush()
        save_progress("dolma", "_all", count, num_records, output_format, s3_storage)
        print(f"Progress saved. Run with --resume to continue from record {count}")
        raise
    
    # Finalize all writers
    for writer in writers.values():
        writer.finalize()
    
    clear_progress("dolma", "_all", output_format, s3_storage)
    
    return count
