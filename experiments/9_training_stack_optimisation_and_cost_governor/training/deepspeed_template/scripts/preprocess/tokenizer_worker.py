"""
Parallel tokenization worker — Phase 1 of the preprocessing pipeline.

This module handles the CPU-bound tokenization step. Each source file is
tokenized independently, producing a temporary .npy file of uint32 token IDs.
Multiple files are processed in parallel across all available CPU cores.

Architecture:
    1. Main process discovers all source files across all data sources.
    2. Files are assigned to a multiprocessing Pool.
    3. Each worker loads the tokenizer, reads documents from the file,
       tokenizes them, and writes a tmp-*.npy file.
    4. Progress is tracked in a JSON file for resumability (spot instance restart).

Output format:
    Each tmp-*.npy file is a 1-D array of uint32 token IDs, with EOS tokens
    separating documents. These are temporary — the sharder (Phase 2) reads
    them and re-shards into uniform-size final .npy files.
"""

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TokenizeTask:
    """A single file to tokenize."""

    file_path: str
    source_name: str  # e.g., "dolma", "sangraha-verified", "ncert", "indicnlp"
    output_path: str  # e.g., /data/tmp/tmp-dolma-00042.npy
    task_id: int  # For progress tracking


@dataclass
class TokenizeResult:
    """Result from tokenizing a single file."""

    task_id: int
    file_path: str
    output_path: str
    num_tokens: int
    num_documents: int
    elapsed_seconds: float
    success: bool
    error: Optional[str] = None


class TokenizerWorker:
    """
    Tokenizes documents from a single source file and writes a .npy file.

    This class is designed to be instantiated once per worker process.
    It loads the tokenizer ONCE (to avoid repeated loading overhead)
    and reuses it across all documents in its assigned file.
    """

    def __init__(
        self,
        tokenizer_name: str,
        eos_token_id: Optional[int] = None,
        max_doc_tokens: int = 100_000,
    ):
        """
        Args:
            tokenizer_name: HuggingFace tokenizer name or path.
            eos_token_id: Explicit EOS token ID. If None, uses tokenizer's default.
            max_doc_tokens: Maximum tokens per document (truncation safety).
        """
        self.tokenizer_name = tokenizer_name
        self.max_doc_tokens = max_doc_tokens
        self._tokenizer = None
        self._eos_token_id = eos_token_id

    @property
    def tokenizer(self):
        """Lazy-load tokenizer (loaded once per worker process)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=True,  # Use Rust-based tokenizer for speed
                trust_remote_code=True,
            )
            logger.info(f"Loaded tokenizer: {self.tokenizer_name} (vocab_size={self._tokenizer.vocab_size})")

            if self._eos_token_id is None:
                self._eos_token_id = self._tokenizer.eos_token_id
                if self._eos_token_id is None:
                    # Fallback: use the last token in vocab as EOS
                    self._eos_token_id = self._tokenizer.vocab_size - 1
                    logger.warning(
                        f"Tokenizer has no EOS token. Using vocab_size-1 = {self._eos_token_id}"
                    )
        return self._tokenizer

    @property
    def eos_token_id(self) -> int:
        # Trigger tokenizer load if needed
        _ = self.tokenizer
        return self._eos_token_id

    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize a single document's text.

        Returns:
            List of token IDs (without special tokens — we add EOS ourselves).
        """
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_doc_tokens,
        )
        return token_ids

    def tokenize_file(self, task: TokenizeTask) -> TokenizeResult:
        """
        Tokenize all documents in a single source file.

        Reads the file using the appropriate reader, tokenizes each document,
        appends EOS tokens between documents, and writes the result as a .npy file.

        Args:
            task: A TokenizeTask describing the file to process.

        Returns:
            TokenizeResult with statistics.
        """
        start_time = time.time()

        try:
            from scripts.preprocess.readers import get_reader_for_source

            reader = get_reader_for_source(task.source_name)
            all_token_ids = []
            num_documents = 0

            for text, language, source_tag in reader.read_file(task.file_path):
                token_ids = self.tokenize_text(text)

                if not token_ids:
                    continue

                # Append document tokens + EOS separator
                all_token_ids.extend(token_ids)
                all_token_ids.append(self.eos_token_id)
                num_documents += 1

            if all_token_ids:
                # Write as uint32 numpy array (4 bytes per token)
                token_array = np.array(all_token_ids, dtype=np.uint32)

                # Ensure output directory exists
                os.makedirs(os.path.dirname(task.output_path), exist_ok=True)

                # Atomic write: write to temp file, then rename
                tmp_file = task.output_path + ".tmp"
                np.save(tmp_file, token_array)
                os.replace(tmp_file, task.output_path)

                num_tokens = len(all_token_ids)
            else:
                num_tokens = 0

            elapsed = time.time() - start_time
            return TokenizeResult(
                task_id=task.task_id,
                file_path=task.file_path,
                output_path=task.output_path,
                num_tokens=num_tokens,
                num_documents=num_documents,
                elapsed_seconds=elapsed,
                success=True,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error tokenizing {task.file_path}: {e}")
            return TokenizeResult(
                task_id=task.task_id,
                file_path=task.file_path,
                output_path=task.output_path,
                num_tokens=0,
                num_documents=0,
                elapsed_seconds=elapsed,
                success=False,
                error=str(e),
            )


# ─── Worker function for multiprocessing ──────────────────────────────────

# Global worker instance (one per process, initialized once)
_worker: Optional[TokenizerWorker] = None


def _init_worker(tokenizer_name: str, eos_token_id: Optional[int], max_doc_tokens: int):
    """Process initializer — creates a TokenizerWorker once per worker process."""
    global _worker
    _worker = TokenizerWorker(tokenizer_name, eos_token_id, max_doc_tokens)
    # Force tokenizer load during init
    _ = _worker.tokenizer
    logger.info(f"Worker {os.getpid()} initialized with tokenizer {tokenizer_name}")


def _process_task(task: TokenizeTask) -> TokenizeResult:
    """Process a single tokenization task (called in worker process)."""
    global _worker
    return _worker.tokenize_file(task)


# ─── Progress tracking ───────────────────────────────────────────────────

class ProgressTracker:
    """
    Tracks tokenization progress for resumability.

    Saves a JSON file after each completed task so that if the spot instance
    is interrupted, we can skip already-completed files on restart.
    """

    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.completed: Dict[int, dict] = {}  # task_id -> result info
        self._load()

    def _load(self):
        """Load existing progress from disk."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                self.completed = {int(k): v for k, v in data.get("completed", {}).items()}
            logger.info(f"Resumed progress: {len(self.completed)} tasks already completed")

    def _save(self):
        """Save progress to disk."""
        with open(self.progress_file, "w") as f:
            json.dump(
                {
                    "completed": self.completed,
                    "total_tokens": sum(r["num_tokens"] for r in self.completed.values()),
                    "total_documents": sum(r["num_documents"] for r in self.completed.values()),
                },
                f,
                indent=2,
            )

    def is_completed(self, task_id: int) -> bool:
        """Check if a task was already completed in a previous run."""
        return task_id in self.completed

    def mark_completed(self, result: TokenizeResult):
        """Mark a task as completed and save progress."""
        self.completed[result.task_id] = {
            "file_path": result.file_path,
            "output_path": result.output_path,
            "num_tokens": result.num_tokens,
            "num_documents": result.num_documents,
            "elapsed_seconds": result.elapsed_seconds,
        }
        self._save()


# ─── Main parallel tokenization function ─────────────────────────────────

def parallel_tokenize(
    tasks: List[TokenizeTask],
    tokenizer_name: str,
    num_workers: int = None,
    eos_token_id: Optional[int] = None,
    max_doc_tokens: int = 100_000,
    progress_file: Optional[str] = None,
) -> List[TokenizeResult]:
    """
    Tokenize multiple files in parallel using a process pool.

    This is Phase 1 of the preprocessing pipeline. Each source file is
    tokenized independently, producing a temporary .npy file.

    Args:
        tasks: List of TokenizeTasks (one per source file).
        tokenizer_name: HuggingFace tokenizer name or local path.
        num_workers: Number of parallel worker processes. Defaults to CPU count - 2.
        eos_token_id: Explicit EOS token ID. If None, uses tokenizer's default.
        max_doc_tokens: Max tokens per document (truncation safety).
        progress_file: Path to save progress JSON (for spot instance resume).

    Returns:
        List of TokenizeResults with statistics for each file.
    """
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 2)

    # ─── Resume support ───────────────────────────────────────────────
    tracker = ProgressTracker(progress_file) if progress_file else None

    # Filter out already-completed tasks
    pending_tasks = tasks
    if tracker:
        pending_tasks = [t for t in tasks if not tracker.is_completed(t.task_id)]
        if len(tasks) - len(pending_tasks) > 0:
            logger.info(f"Skipping {len(tasks) - len(pending_tasks)} already-completed tasks")

    if not pending_tasks:
        logger.info("All tasks already completed!")
        return []

    logger.info(
        f"Starting parallel tokenization: "
        f"{len(pending_tasks)} files, {num_workers} workers, "
        f"tokenizer={tokenizer_name}"
    )

    results: List[TokenizeResult] = []
    total_tokens = 0
    total_docs = 0
    start_time = time.time()

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer_name, eos_token_id, max_doc_tokens),
    ) as pool:
        # Submit all tasks
        futures = {pool.submit(_process_task, task): task for task in pending_tasks}

        # Process results as they complete
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            if result.success:
                total_tokens += result.num_tokens
                total_docs += result.num_documents

                if tracker:
                    tracker.mark_completed(result)

                # Log progress every 10 files or on completion
                if (i + 1) % 10 == 0 or (i + 1) == len(pending_tasks):
                    elapsed = time.time() - start_time
                    files_per_sec = (i + 1) / elapsed
                    tokens_per_sec = total_tokens / elapsed
                    eta_seconds = (len(pending_tasks) - i - 1) / files_per_sec if files_per_sec > 0 else 0

                    logger.info(
                        f"  Progress: {i + 1}/{len(pending_tasks)} files "
                        f"({total_tokens:,} tokens, {total_docs:,} docs) "
                        f"| {tokens_per_sec:,.0f} tok/s "
                        f"| ETA: {eta_seconds / 3600:.1f}h"
                    )
            else:
                logger.error(f"  FAILED: {result.file_path} — {result.error}")

    # ─── Final summary ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    logger.info(
        f"\n{'='*60}\n"
        f"Phase 1 Complete: Parallel Tokenization\n"
        f"{'='*60}\n"
        f"  Files processed:  {successful} succeeded, {failed} failed\n"
        f"  Total tokens:     {total_tokens:,}\n"
        f"  Total documents:  {total_docs:,}\n"
        f"  Total time:       {elapsed:.1f}s ({elapsed/3600:.2f}h)\n"
        f"  Throughput:       {total_tokens/elapsed:,.0f} tokens/sec\n"
        f"{'='*60}"
    )

    return results


# ─── Helper: create tasks from source directories ────────────────────────

def create_tasks_from_sources(
    source_dirs: Dict[str, str],
    output_dir: str,
) -> List[TokenizeTask]:
    """
    Discover all source files and create TokenizeTasks.

    Args:
        source_dirs: Mapping of source_name -> directory_path.
                     e.g., {"dolma": "/data/raw/dolma", "ncert": "/data/raw/ncert"}
        output_dir: Directory for temporary tokenized .npy files.

    Returns:
        List of TokenizeTasks, one per source file.
    """
    from scripts.preprocess.readers import discover_source_files

    tasks = []
    task_id = 0

    for source_name, dir_path in sorted(source_dirs.items()):
        files = discover_source_files(source_name, dir_path)
        logger.info(f"  {source_name}: {len(files)} files in {dir_path}")

        for file_path in files:
            # Create a unique output filename
            # e.g., tmp-dolma-00042.npy
            output_filename = f"tmp-{source_name}-{task_id:06d}.npy"
            output_path = os.path.join(output_dir, output_filename)

            tasks.append(
                TokenizeTask(
                    file_path=file_path,
                    source_name=source_name,
                    output_path=output_path,
                    task_id=task_id,
                )
            )
            task_id += 1

    logger.info(f"Created {len(tasks)} tokenization tasks from {len(source_dirs)} sources")
    return tasks
