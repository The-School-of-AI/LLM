"""NCERT dataset downloader with streaming support."""

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
DATASET_NAME = "Ncert"

# Available subjects (based on NCERT curriculum)
SUBJECTS = [
    "physics", "chemistry", "biology", "mathematics",
    "history", "geography", "civics", "economics",
    "english", "hindi"
]

# Progress update interval
PROGRESS_INTERVAL = 1000


def format_text(example: dict) -> str:
    """Format NCERT example fields into a structured text.
    
    Args:
        example: Dictionary containing NCERT dataset fields
        
    Returns:
        Formatted text combining all fields
    """
    parts = []
    
    # Topic
    topic = example.get("Topic", "")
    if topic:
        parts.append(f"### Topic: {topic}")
    
    # Explanation
    explanation = example.get("Explanation", "")
    if explanation:
        parts.append(f"\n### Explanation:\n{explanation}")
    
    # Question
    question = example.get("Question", "")
    if question:
        parts.append(f"\n### Question:\n{question}")
    
    # Answer
    answer = example.get("Answer", "")
    if answer:
        parts.append(f"\n### Answer:\n{answer}")
    
    # Metadata section
    metadata_parts = []
    
    difficulty = example.get("Difficulty", "")
    if difficulty:
        metadata_parts.append(f"Difficulty: {difficulty}")
    
    student_level = example.get("StudentLevel", "")
    if student_level:
        metadata_parts.append(f"Student Level: {student_level}")
    
    subject = example.get("subject", "")
    if subject:
        metadata_parts.append(f"Subject: {subject}")
    
    grade = example.get("grade", "")
    if grade:
        metadata_parts.append(f"Grade: {grade}")
    
    estimated_time = example.get("EstimatedTime", "")
    if estimated_time:
        metadata_parts.append(f"Estimated Time: {estimated_time} minutes")
    
    prerequisites = example.get("Prerequisites", "")
    if prerequisites:
        metadata_parts.append(f"Prerequisites: {prerequisites}")
    
    if metadata_parts:
        parts.append("\n### Metadata:\n" + "  \n".join(metadata_parts))
    
    return "\n".join(parts)


def download(
    num_records: int = 10,
    resume: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    output_format: OutputFormat = "json",
    storage_type: StorageType = "local",
    s3_storage: Optional[S3Storage] = None,
) -> int:
    """Download records from NCERT dataset.
    
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
    subfolder = "_all"
    
    # Check for existing progress if resume mode
    skip_count = 0
    
    if resume:
        progress = load_progress("ncert", subfolder, output_format, s3_storage)
        if progress:
            skip_count = progress.get("downloaded_count", 0)
            print(f"Resuming from record {skip_count}, targeting {num_records} total records")
            
            # If we already have enough records, return
            if skip_count >= num_records:
                print(f"Already have {skip_count} records, nothing to download")
                return skip_count
    
    print(f"--- Downloading {num_records - skip_count} records from NCERT ---")
    print(f"Chunk size: {chunk_size} records per file, format: {output_format}")
    
    # Initialize chunked writer
    writer = ChunkedWriter(
        dataset="ncert",
        subfolder=subfolder,
        base_filename="records",
        chunk_size=chunk_size,
        use_custom_serializer=False,
        resume_from_record=skip_count,
        output_format=output_format,
        storage_type=storage_type,
        s3_storage=s3_storage,
    )
    
    # Load NCERT dataset
    dataset = load_dataset(
        "KadamParth/Ncert_dataset",
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
            
            # Extract fields from the dataset
            example_subject = example.get("subject", "unknown")
            grade = example.get("grade", "unknown")
            
            # Format text by combining all fields
            formatted_text = format_text(example)
            
            record = {
                "id": f"ncert_{count}",
                "hash": compute_hash(formatted_text),
                "dataset": DATASET_NAME,
                "domain": "education",
                "source": None,
                "text": formatted_text,
                "language": "English",
                "metadata": {
                    "subject": example_subject,
                    "grade": grade,
                    "topic": example.get("Topic", ""),
                    "difficulty": example.get("Difficulty", ""),
                    "student_level": example.get("StudentLevel", ""),
                    "question_type": example.get("QuestionType", ""),
                    "question_complexity": example.get("QuestionComplexity", ""),
                    "estimated_time": example.get("EstimatedTime", ""),
                    "prerequisites": example.get("Prerequisites", ""),
                    "source_type": "textbook",
                },
                "added": None,
                "created": None,
                "version": None,
            }
            writer.add_record(record)
            count += 1
            
            # Print progress
            if count % PROGRESS_INTERVAL == 0:
                save_progress("ncert", subfolder, count, num_records, output_format, s3_storage)
                stats = writer.get_stats()
                print(f"Progress: {count}/{num_records} records, {stats['files_written']} files written")
    
    except (KeyboardInterrupt, Exception) as e:
        print(f"\nInterrupted: {e}")
        writer.flush()
        save_progress("ncert", subfolder, count, num_records, output_format, s3_storage)
        print(f"Progress saved. Run with --resume to continue from record {count}")
        raise
    
    # Finalize writing
    writer.finalize()
    clear_progress("ncert", subfolder, output_format, s3_storage)
    
    return count
