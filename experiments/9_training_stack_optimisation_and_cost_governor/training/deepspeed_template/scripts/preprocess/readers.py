"""
Source-specific document readers.

Each reader yields (text, language, source_tag) tuples from a specific data format.
This normalizes all 4 data sources into a uniform interface for the tokenizer.

Supported sources:
    - Dolma:     JSONL.gz files (AI2's web corpus)
    - Sangraha:  Parquet files (AI4Bharat's Indic corpus — verified, unverified, synthetic)
    - NCERT:     CSV / plain text files (Indian educational textbooks)
    - IndicNLP:  Plain text files organized by language (AI4Bharat monolingual corpus)
"""

import csv
import gzip
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Type alias: (document_text, language_code, source_tag)
Document = Tuple[str, str, str]


# ─── Minimum document length (characters) to skip very short/empty docs ───
MIN_DOC_LENGTH = 50


class BaseReader(ABC):
    """Abstract base class for all source readers."""

    def __init__(self, min_doc_length: int = MIN_DOC_LENGTH):
        self.min_doc_length = min_doc_length
        self._files_read = 0
        self._docs_yielded = 0
        self._docs_skipped = 0

    @abstractmethod
    def read_file(self, file_path: str) -> Generator[Document, None, None]:
        """Yield (text, language, source_tag) from a single file."""
        ...

    def read_directory(self, dir_path: str, extensions: Optional[List[str]] = None) -> Generator[Document, None, None]:
        """
        Recursively read all matching files from a directory.

        Args:
            dir_path: Root directory to scan.
            extensions: File extensions to match (e.g., [".jsonl.gz", ".jsonl"]).
                        If None, reads all files.
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        files = sorted(dir_path.rglob("*"))
        for f in files:
            if not f.is_file():
                continue
            if extensions and not any(str(f).endswith(ext) for ext in extensions):
                continue

            logger.info(f"Reading: {f}")
            self._files_read += 1
            for doc in self.read_file(str(f)):
                yield doc

        logger.info(
            f"  {self.__class__.__name__}: "
            f"{self._files_read} files, "
            f"{self._docs_yielded} docs yielded, "
            f"{self._docs_skipped} docs skipped (too short)"
        )

    def _filter(self, text: str, language: str, source_tag: str) -> Optional[Document]:
        """Apply minimum length filter. Returns the doc tuple or None."""
        if not text or len(text.strip()) < self.min_doc_length:
            self._docs_skipped += 1
            return None
        self._docs_yielded += 1
        return (text.strip(), language, source_tag)

    @staticmethod
    def discover_files(dir_path: str, extensions: List[str]) -> List[str]:
        """
        Discover all files with matching extensions in a directory.
        Returns sorted list of absolute paths.
        Used by the parallel tokenizer to split work across workers.
        """
        dir_path = Path(dir_path)
        files = []
        for f in sorted(dir_path.rglob("*")):
            if f.is_file() and any(str(f).endswith(ext) for ext in extensions):
                files.append(str(f.resolve()))
        return files


class DolmaReader(BaseReader):
    """
    Reader for Dolma dataset (AI2).

    Format: JSONL.gz (gzip-compressed JSON lines)
    Each line: {"id": "...", "text": "...", "source": "common-crawl", "metadata": {...}}

    Dolma is primarily English but may contain other languages.
    """

    SOURCE_TAG = "dolma"
    EXTENSIONS = [".jsonl.gz", ".jsonl.zst", ".jsonl"]

    def read_file(self, file_path: str) -> Generator[Document, None, None]:
        opener = gzip.open if file_path.endswith(".gz") else open

        try:
            with opener(file_path, "rt", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON at {file_path}:{line_num}")
                        continue

                    text = record.get("text", "")
                    # Dolma records may have a source field (common-crawl, c4, stack, etc.)
                    sub_source = record.get("source", "unknown")
                    source_tag = f"{self.SOURCE_TAG}-{sub_source}"

                    doc = self._filter(text, "en", source_tag)
                    if doc:
                        yield doc

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")


class SangrahaReader(BaseReader):
    """
    Reader for Sangraha dataset (AI4Bharat).

    Format: Parquet files
    Columns typically include: text, language/lang, source, doc_id
    Three tiers: verified, unverified, synthetic

    Requires: pyarrow
    """

    SOURCE_TAG = "sangraha"
    EXTENSIONS = [".parquet"]

    def __init__(self, tier: str = "verified", min_doc_length: int = MIN_DOC_LENGTH):
        """
        Args:
            tier: One of "verified", "unverified", "synthetic"
        """
        super().__init__(min_doc_length)
        self.tier = tier

    def read_file(self, file_path: str) -> Generator[Document, None, None]:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for reading Sangraha parquet files: pip install pyarrow")

        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()

            # Sangraha columns: text, language (or lang), source
            text_col = "text" if "text" in df.columns else None
            lang_col = next((c for c in df.columns if c in ("language", "lang")), None)

            if text_col is None:
                logger.error(f"No 'text' column found in {file_path}. Columns: {list(df.columns)}")
                return

            for _, row in df.iterrows():
                text = str(row.get(text_col, ""))
                lang = str(row.get(lang_col, "unknown")) if lang_col else "unknown"
                source_tag = f"{self.SOURCE_TAG}-{self.tier}"

                doc = self._filter(text, lang, source_tag)
                if doc:
                    yield doc

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")


class NCERTReader(BaseReader):
    """
    Reader for NCERT textbook dataset.

    Supports:
        - CSV files with 'text' column (or similar: 'content', 'passage')
        - Plain text files (one document per file or paragraph-separated)

    NCERT content is bilingual (Hindi + English), high quality, curated educational text.
    """

    SOURCE_TAG = "ncert"
    EXTENSIONS = [".csv", ".txt", ".md"]

    def read_file(self, file_path: str) -> Generator[Document, None, None]:
        if file_path.endswith(".csv"):
            yield from self._read_csv(file_path)
        else:
            yield from self._read_text(file_path)

    def _read_csv(self, file_path: str) -> Generator[Document, None, None]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)

                # Find the text column
                text_col = None
                lang_col = None
                if reader.fieldnames:
                    for candidate in ["text", "content", "passage", "Text", "Content", "Passage"]:
                        if candidate in reader.fieldnames:
                            text_col = candidate
                            break
                    for candidate in ["language", "lang", "Language"]:
                        if candidate in reader.fieldnames:
                            lang_col = candidate
                            break

                if text_col is None:
                    # If no known text column, use the first column
                    text_col = reader.fieldnames[0] if reader.fieldnames else None
                    logger.warning(f"No standard text column in {file_path}, using '{text_col}'")

                for row in reader:
                    text = str(row.get(text_col, ""))
                    lang = str(row.get(lang_col, "hi")) if lang_col else "hi"  # Default Hindi for NCERT

                    doc = self._filter(text, lang, self.SOURCE_TAG)
                    if doc:
                        yield doc

        except Exception as e:
            logger.error(f"Error reading CSV {file_path}: {e}")

    def _read_text(self, file_path: str) -> Generator[Document, None, None]:
        """Read plain text file. Split on double newlines (paragraphs) into documents."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Split into paragraphs/sections (double newline as separator)
            sections = content.split("\n\n")

            # Detect language from file path (heuristic)
            lang = "hi" if any(x in file_path.lower() for x in ["hindi", "_hi", "/hi/"]) else "en"

            for section in sections:
                doc = self._filter(section, lang, self.SOURCE_TAG)
                if doc:
                    yield doc

        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")


class IndicNLPReader(BaseReader):
    """
    Reader for AI4Bharat IndicNLP Corpus.

    Format: Plain text files, one sentence per line, organized by language directory.
    Structure:
        indicnlp/
        ├── hi/
        │   └── hi_corpus.txt
        ├── ta/
        │   └── ta_corpus.txt
        └── ...

    The corpus is pre-tokenized using IndicNLP tokenizer but we treat it as raw text
    since we'll re-tokenize with our chosen tokenizer (e.g., Qwen2).
    """

    SOURCE_TAG = "indicnlp"
    EXTENSIONS = [".txt"]

    # ISO 639-1 language codes for Indian languages
    LANG_DIRS = {
        "as", "bn", "gu", "hi", "kn", "ml", "mr", "or",
        "pa", "ta", "te", "ur", "en", "ne", "sa", "sd",
        "si", "bh", "doi", "kok", "mai", "mni", "sat",
    }

    def read_file(self, file_path: str) -> Generator[Document, None, None]:
        """Read a single text file. Detect language from directory name or filename."""
        # Try to detect language from path
        lang = self._detect_language(file_path)

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                # Accumulate sentences into documents (every N sentences = one doc)
                buffer = []
                sentences_per_doc = 20  # Group ~20 sentences into a document

                for line in f:
                    line = line.strip()
                    if not line:
                        # Empty line = paragraph boundary, flush buffer
                        if buffer:
                            text = " ".join(buffer)
                            doc = self._filter(text, lang, self.SOURCE_TAG)
                            if doc:
                                yield doc
                            buffer = []
                        continue

                    buffer.append(line)

                    if len(buffer) >= sentences_per_doc:
                        text = " ".join(buffer)
                        doc = self._filter(text, lang, self.SOURCE_TAG)
                        if doc:
                            yield doc
                        buffer = []

                # Flush remaining
                if buffer:
                    text = " ".join(buffer)
                    doc = self._filter(text, lang, self.SOURCE_TAG)
                    if doc:
                        yield doc

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    def _detect_language(self, file_path: str) -> str:
        """Detect language code from file path or directory structure."""
        parts = Path(file_path).parts
        for part in reversed(parts):
            # Check directory name (e.g., "hi", "ta", "bn")
            if part.lower() in self.LANG_DIRS:
                return part.lower()
            # Check filename prefix (e.g., "hi_corpus.txt")
            for lang in self.LANG_DIRS:
                if part.lower().startswith(f"{lang}_") or part.lower().startswith(f"{lang}-"):
                    return lang
        return "unknown"


# ─── Factory function ───────────────────────────────────────────────────────

def get_reader_for_source(source_name: str, **kwargs) -> BaseReader:
    """
    Factory function to get the appropriate reader for a data source.

    Args:
        source_name: One of "dolma", "sangraha-verified", "sangraha-unverified",
                     "sangraha-synthetic", "ncert", "indicnlp"

    Returns:
        An instance of the appropriate reader class.
    """
    readers = {
        "dolma": DolmaReader,
        "sangraha-verified": lambda **kw: SangrahaReader(tier="verified", **kw),
        "sangraha-unverified": lambda **kw: SangrahaReader(tier="unverified", **kw),
        "sangraha-synthetic": lambda **kw: SangrahaReader(tier="synthetic", **kw),
        "ncert": NCERTReader,
        "indicnlp": IndicNLPReader,
    }

    if source_name not in readers:
        raise ValueError(
            f"Unknown source: '{source_name}'. "
            f"Available sources: {list(readers.keys())}"
        )

    creator = readers[source_name]
    if callable(creator) and not isinstance(creator, type):
        return creator(**kwargs)
    return creator(**kwargs)


# ─── Helper: discover all files for a source ────────────────────────────────

def discover_source_files(source_name: str, dir_path: str) -> List[str]:
    """
    Discover all files in a directory matching the expected extensions for a source.

    Returns:
        Sorted list of absolute file paths.
    """
    extension_map = {
        "dolma": DolmaReader.EXTENSIONS,
        "sangraha-verified": SangrahaReader.EXTENSIONS,
        "sangraha-unverified": SangrahaReader.EXTENSIONS,
        "sangraha-synthetic": SangrahaReader.EXTENSIONS,
        "ncert": NCERTReader.EXTENSIONS,
        "indicnlp": IndicNLPReader.EXTENSIONS,
    }

    extensions = extension_map.get(source_name)
    if extensions is None:
        raise ValueError(f"Unknown source: '{source_name}'")

    return BaseReader.discover_files(dir_path, extensions)
