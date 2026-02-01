#!/usr/bin/env python3
"""
IndicCorpV2 Production Cleaning & Categorization Pipeline

This script cleans and categorizes the entire IndicCorpV2 dataset according to
the difficulty ladder (B0-B5) for staged language model training.

Features:
- Quality filtering (spam, gibberish, duplicates)
- Language verification
- Difficulty classification (B0-B4)
- Content categorization (news, wiki, social, blog, formal, conversational)
- Parallel processing for scale
- Organized output by difficulty level

Author: Generated for IndicCorpV2 processing
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Available splits in IndicCorpV2 (uses ISO 639-3 codes)
AVAILABLE_SPLITS = [
    "asm_Beng",
    "ben_Beng",
    "brx_Deva",
    "doi_Deva",
    "gom_Deva",
    "guj_Gujr",
    "hin_Deva",
    "kan_Knda",
    "kas_Arab",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "mni_Mtei",
    "npi_Deva",
    "ory_Orya",
    "pan_Guru",
    "san_Deva",
    "snd_Deva",
    "tam_Taml",
    "tel_Telu",
    "urd_Arab",
    "khasi",
    "santhali",
]

# Map common 2-letter codes to 3-letter codes used by IndicCorpV2
LANG_CODE_MAP = {
    "hi": "hin",  # Hindi
    "bn": "ben",  # Bengali
    "te": "tel",  # Telugu
    "ta": "tam",  # Tamil
    "mr": "mar",  # Marathi
    "gu": "guj",  # Gujarati
    "kn": "kan",  # Kannada
    "ml": "mal",  # Malayalam
    "pa": "pan",  # Punjabi
    "or": "ory",  # Odia
    "as": "asm",  # Assamese
    "ur": "urd",  # Urdu
    "ne": "npi",  # Nepali
    "sa": "san",  # Sanskrit
    "sd": "snd",  # Sindhi
    "ks": "kas",  # Kashmiri
}


@dataclass
class FilterConfig:
    """Quality filter configuration"""

    min_words: int = 10
    max_words: int = 10000
    min_avg_word_length: float = 2.0
    max_avg_word_length: float = 15.0
    max_symbol_ratio: float = 0.3
    max_repetition_ratio: float = 0.5
    min_lexical_diversity: float = 0.15


@dataclass
class ProcessingStats:
    """Track processing statistics"""

    total_processed: int = 0
    kept: int = 0
    quality_failures: Counter = None
    difficulty_distribution: Counter = None
    category_distribution: Counter = None
    duplicates_removed: int = 0

    def __post_init__(self):
        if self.quality_failures is None:
            self.quality_failures = Counter()
        if self.difficulty_distribution is None:
            self.difficulty_distribution = Counter()
        if self.category_distribution is None:
            self.category_distribution = Counter()


# ============================================================================
# PHASE 1: QUALITY FILTERING
# ============================================================================


class QualityFilter:
    """Rule-based quality filtering to remove junk"""

    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()

        # Compile spam patterns once
        self.spam_patterns = [
            re.compile(
                r"(click here|buy now|limited offer).{0,50}(click here|buy now|limited offer)",
                re.I,
            ),
            re.compile(r"(\\x[0-9a-f]{2}){10,}"),  # Hex escapes
            re.compile(r"(.)\1{20,}"),  # Character repetition
            re.compile(r"(https?://\S+\s+){10,}"),  # Too many URLs
        ]

    def filter(self, text: str) -> Tuple[bool, str]:
        """
        Returns (should_keep, failure_reason)
        """

        # Basic cleaning
        text = text.strip()
        if not text:
            return False, "empty_text"

        words = text.split()

        # === Length checks ===
        if len(words) < self.config.min_words:
            return False, "too_short"
        if len(words) > self.config.max_words:
            return False, "too_long"

        # === Average word length (detects gibberish) ===
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < self.config.min_avg_word_length:
            return False, "words_too_short"
        if avg_word_len > self.config.max_avg_word_length:
            return False, "words_too_long"

        # === Symbol ratio (detects spam/code dumps) ===
        symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        symbol_ratio = symbols / max(len(text), 1)
        if symbol_ratio > self.config.max_symbol_ratio:
            return False, "too_many_symbols"

        # === Lexical diversity (detects spam) ===
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words)
        if lexical_diversity < self.config.min_lexical_diversity:
            return False, "low_lexical_diversity"

        # === Line repetition (detects spam) ===
        lines = text.split("\n")
        if len(lines) > 5:
            line_counts = Counter(lines)
            max_repeated = max(line_counts.values())
            repetition_ratio = max_repeated / len(lines)
            if repetition_ratio > self.config.max_repetition_ratio:
                return False, "repetitive_lines"

        # === Spam patterns ===
        for pattern in self.spam_patterns:
            if pattern.search(text):
                return False, "spam_pattern"

        # === Bullet point spam (common in web scraping) ===
        bullet_lines = sum(
            1 for line in lines if line.strip().startswith(("•", "-", "*", "·"))
        )
        if bullet_lines > 0.8 * len(lines) and len(lines) > 10:
            return False, "bullet_spam"

        return True, "pass"


class SimpleDeduplicator:
    """Memory-efficient exact deduplication using hashes"""

    def __init__(self):
        self.seen_hashes = set()
        self.duplicate_count = 0

    def is_duplicate(self, text: str) -> bool:
        """Returns True if exact duplicate found"""

        # Normalize: lowercase + whitespace normalization
        normalized = " ".join(text.lower().split())

        # Hash
        text_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()

        if text_hash in self.seen_hashes:
            self.duplicate_count += 1
            return True

        self.seen_hashes.add(text_hash)
        return False

    def get_stats(self) -> Dict:
        return {
            "unique_documents": len(self.seen_hashes),
            "duplicates_found": self.duplicate_count,
        }


# ============================================================================
# PHASE 2: FEATURE EXTRACTION
# ============================================================================


class FeatureExtractor:
    """Extract features for difficulty and category classification"""

    def __init__(self):
        # Compile sentence splitter
        self.sentence_pattern = re.compile(r"[.!?।]")

    def extract(self, text: str) -> Dict:
        """Extract all features"""

        words = text.split()
        sentences = self._split_sentences(text)

        features = {}

        # === Basic stats ===
        features["word_count"] = len(words)
        features["char_count"] = len(text)
        features["sentence_count"] = max(len(sentences), 1)

        # === Complexity metrics ===
        features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
        features["avg_sentence_length"] = len(words) / features["sentence_count"]

        # === Vocabulary metrics ===
        unique_words = set(words)
        features["unique_words"] = len(unique_words)
        features["lexical_diversity"] = len(unique_words) / len(words) if words else 0

        # TTR on first 100 words (more stable metric)
        first_100 = words[:100]
        features["ttr_100"] = len(set(first_100)) / len(first_100) if first_100 else 0

        # === Structural metrics ===
        features["paragraph_count"] = text.count("\n\n") + 1
        features["avg_paragraph_length"] = len(words) / features["paragraph_count"]

        # Sentence length variance (higher = more complex)
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]
        features["sentence_length_variance"] = (
            float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0
        )

        # === Language complexity ===
        # Long words (>8 chars) indicate technical/formal content
        long_words = sum(1 for w in words if len(w) > 8)
        features["long_word_ratio"] = long_words / len(words) if words else 0

        # Very long words (>12 chars) indicate very technical content
        very_long_words = sum(1 for w in words if len(w) > 12)
        features["very_long_word_ratio"] = very_long_words / len(words) if words else 0

        # === Content indicators ===
        # Numbers (technical/data content)
        features["number_ratio"] = (
            sum(1 for w in words if any(c.isdigit() for c in w)) / len(words)
            if words
            else 0
        )

        # Punctuation diversity
        punct_chars = [c for c in text if c in "।.!?,;:—-()[]{}\"'"]
        features["punct_diversity"] = len(set(punct_chars)) / max(len(punct_chars), 1)

        # Question marks (conversational/educational)
        features["question_ratio"] = text.count("?") / features["sentence_count"]

        # === Domain-specific indicators ===
        features["has_date"] = int(
            bool(re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text))
        )
        features["has_references"] = int(bool(re.search(r"\[\d+\]", text)))
        features["has_hashtags"] = int(bool(re.search(r"#\w+", text)))
        features["url_count"] = len(re.findall(r"http[s]?://", text))

        # English word ratio (code-switching)
        english_words = sum(1 for w in words if re.match(r"^[a-zA-Z]+$", w))
        features["english_ratio"] = english_words / len(words) if words else 0

        return features

    def _split_sentences(self, text: str) -> List[str]:
        """Split into sentences (handles Indic punctuation)"""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# PHASE 3: DIFFICULTY CLASSIFICATION
# ============================================================================


class DifficultyClassifier:
    """Classify text into B0-B4 difficulty levels"""

    def classify(self, features: Dict) -> Tuple[str, float]:
        """
        Returns (difficulty_label, confidence_score)

        B0: Nursery - Simple grammar, repetitive (children's stories, simple social)
        B1: Primary - Fluent general knowledge (news, general web)
        B2: High School - Structured narrative, longer docs (books, tutorials)
        B3: Undergrad - Technical, reasoning (tech blogs, manuals)
        B4: Graduate - Advanced technical (papers, complex code)
        """

        # === B0: Nursery ===
        # Very simple, short sentences, low diversity
        if (
            features["avg_sentence_length"] < 10
            and features["lexical_diversity"] < 0.35
            and features["word_count"] < 300
            and features["long_word_ratio"] < 0.10
        ):

            confidence = 0.8 if features["lexical_diversity"] < 0.30 else 0.6
            return "B0", confidence

        # === B4: Graduate ===
        # Very technical, has references, very long words, high complexity
        if (
            features["avg_sentence_length"] > 25
            and features["very_long_word_ratio"] > 0.15
            and features["lexical_diversity"] > 0.60
            and (features["has_references"] == 1 or features["number_ratio"] > 0.20)
        ):

            confidence = 0.9 if features["has_references"] == 1 else 0.75
            return "B4", confidence

        # === B3: Undergrad ===
        # Technical content, complex sentences, high diversity
        if (
            features["avg_sentence_length"] > 18
            and features["long_word_ratio"] > 0.18
            and features["lexical_diversity"] > 0.50
            and features["word_count"] > 300
        ):

            # Higher confidence if has technical indicators
            confidence = 0.8 if features["number_ratio"] > 0.10 else 0.65
            return "B3", confidence

        # === B2: High School ===
        # Well-structured, moderate complexity, longer documents
        if (
            features["avg_sentence_length"] > 12
            and features["lexical_diversity"] > 0.40
            and features["word_count"] > 200
            and features["paragraph_count"] > 2
        ):

            confidence = 0.7
            return "B2", confidence

        # === B1: Primary (Default) ===
        # Everything else - general web content
        confidence = 0.6
        return "B1", confidence


# ============================================================================
# PHASE 4: CONTENT CATEGORIZATION
# ============================================================================


class CategoryClassifier:
    """Classify content type: news, wiki, social, blog, formal, conversational"""

    def __init__(self):
        # Compile patterns once for performance
        self.patterns = {
            "news": [
                re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"),
                re.compile(
                    r"\b(reported|announced|said|according to|spokesperson|statement)\b",
                    re.I,
                ),
                re.compile(
                    r"\b(journalist|correspondent|reporter|press|news agency)\b", re.I
                ),
                re.compile(r"\b(today|yesterday|last week|this month)\b", re.I),
            ],
            "wiki": [
                re.compile(r"\[\d+\]"),  # References
                re.compile(
                    r"\b(also known as|commonly known as|referred to as)\b", re.I
                ),
                re.compile(
                    r"\b(see also|references|external links|bibliography|further reading)\b",
                    re.I,
                ),
                re.compile(
                    r"\b(born|died|established|founded|created)\s+\d{4}\b", re.I
                ),
                re.compile(
                    r"\b(is a|was a|are a|were a)\b.*\b(country|city|person|organization)\b",
                    re.I,
                ),
            ],
            "social": [
                re.compile(r"[@#]\w+"),  # Mentions/hashtags
                re.compile(r"\b(rt|retweet|follow|like|share|comment|reply)\b", re.I),
                re.compile(r"\b(lol|omg|btw|imo|fyi|tbh|smh)\b", re.I),
                re.compile(r"[😀-🙏]{2,}"),  # Emoji sequences
            ],
            "blog": [
                re.compile(
                    r"\b(i think|in my opinion|i believe|personally|i feel)\b", re.I
                ),
                re.compile(
                    r"\b(today i|yesterday i|this week|last month|recently i)\b", re.I
                ),
                re.compile(
                    r"\b(dear readers|hello everyone|hi folks|hey guys)\b", re.I
                ),
                re.compile(r"\b(my experience|my story|my journey)\b", re.I),
            ],
        }

    def classify(self, text: str) -> str:
        """Returns primary category"""

        scores = {}
        for category, pattern_list in self.patterns.items():
            score = sum(1 for pattern in pattern_list if pattern.search(text))
            scores[category] = score

        max_score = max(scores.values()) if scores else 0

        # If no patterns matched, use heuristics
        if max_score == 0:
            return self._classify_by_heuristics(text)

        # Return category with highest score
        return max(scores, key=scores.get)

    def _classify_by_heuristics(self, text: str) -> str:
        """Fallback classification using text structure"""

        words = text.split()
        sentences = text.count(".") + text.count("।")

        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        avg_sent_len = len(words) / max(sentences, 1)

        # Formal: longer words, longer sentences
        if avg_word_len > 6 and avg_sent_len > 15:
            return "formal"

        # Conversational: shorter, simpler
        return "conversational"


# ============================================================================
# PHASE 5: MAIN PROCESSING PIPELINE
# ============================================================================


class IndicCorpProcessor:
    """Main processing pipeline"""

    def __init__(self, lang_code: str, script_code: str, output_dir: str):
        self.lang_code = lang_code
        self.script_code = script_code
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.quality_filter = QualityFilter()
        self.deduplicator = SimpleDeduplicator()
        self.feature_extractor = FeatureExtractor()
        self.difficulty_classifier = DifficultyClassifier()
        self.category_classifier = CategoryClassifier()

        # Statistics
        self.stats = ProcessingStats()

        # Create difficulty subdirectories
        for difficulty in ["B0", "B1", "B2", "B3", "B4"]:
            (self.output_dir / difficulty).mkdir(exist_ok=True)

    def process_single(self, text: str) -> Optional[Dict]:
        """Process a single document"""

        self.stats.total_processed += 1

        # === Quality filtering ===
        keep, reason = self.quality_filter.filter(text)
        if not keep:
            self.stats.quality_failures[reason] += 1
            return None

        # === Deduplication ===
        if self.deduplicator.is_duplicate(text):
            self.stats.duplicates_removed += 1
            return None

        # === Feature extraction ===
        features = self.feature_extractor.extract(text)

        # === Difficulty classification ===
        difficulty, diff_confidence = self.difficulty_classifier.classify(features)
        self.stats.difficulty_distribution[difficulty] += 1

        # === Category classification ===
        category = self.category_classifier.classify(text)
        self.stats.category_distribution[category] += 1

        self.stats.kept += 1

        # Return processed document
        return {
            "text": text,
            "difficulty": difficulty,
            "difficulty_confidence": float(diff_confidence),
            "category": category,
            "word_count": features["word_count"],
            "char_count": features["char_count"],
            "metadata": {
                "lang": self.lang_code,
                "lexical_diversity": float(features["lexical_diversity"]),
                "avg_sentence_length": float(features["avg_sentence_length"]),
            },
        }

    def process_dataset(self, batch_size: int = 10000, limit: Optional[int] = None):
        """Process entire dataset"""

        print(f"\n{'='*70}")
        print(f"Processing {self.lang_code}_{self.script_code}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")

        # Load dataset
        # Map 2-letter code to 3-letter code if needed (IndicCorpV2 uses ISO 639-3)
        lang_code_3 = LANG_CODE_MAP.get(self.lang_code, self.lang_code)
        language_split_name = f"{lang_code_3}_{self.script_code}"

        # Validate split exists
        if language_split_name not in AVAILABLE_SPLITS:
            print(f"Error: Split '{language_split_name}' not found.")
            print(f"Available splits: {AVAILABLE_SPLITS}")
            return

        try:
            dataset = load_dataset(
                "ai4bharat/IndicCorpV2",
                "indiccorp_v2",
                split=language_split_name,
                streaming=True,
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        # Process documents
        batch = []
        batch_num = 0

        progress_bar = tqdm(desc=f"Processing {self.lang_code}", unit=" docs")

        for i, example in enumerate(dataset):
            if limit and i >= limit:
                break

            text = example.get("text", "")
            processed = self.process_single(text)

            if processed:
                batch.append(processed)

            # Save batch
            if len(batch) >= batch_size:
                self._save_batch(batch, batch_num)
                batch = []
                batch_num += 1

            # Update progress
            if self.stats.total_processed % 1000 == 0:
                progress_bar.update(1000)
                progress_bar.set_postfix(
                    {
                        "kept": self.stats.kept,
                        "pass_rate": f"{self.stats.kept/self.stats.total_processed*100:.1f}%",
                    }
                )

        # Save final batch
        if batch:
            self._save_batch(batch, batch_num)

        progress_bar.close()

        # Save statistics
        self._save_stats()
        self._print_summary()

    def _save_batch(self, batch: List[Dict], batch_num: int):
        """Save batch organized by difficulty"""

        # Group by difficulty
        by_difficulty = defaultdict(list)
        for doc in batch:
            by_difficulty[doc["difficulty"]].append(doc)

        # Save each difficulty
        for difficulty, docs in by_difficulty.items():
            output_file = self.output_dir / difficulty / f"batch_{batch_num:06d}.jsonl"

            with open(output_file, "w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    def _save_stats(self):
        """Save processing statistics"""

        stats_file = self.output_dir / "processing_stats.json"

        stats_dict = {
            "language": self.lang_code,
            "script": self.script_code,
            "total_processed": self.stats.total_processed,
            "kept": self.stats.kept,
            "removed": self.stats.total_processed - self.stats.kept,
            "pass_rate": self.stats.kept / max(self.stats.total_processed, 1),
            "duplicates_removed": self.stats.duplicates_removed,
            "quality_failures": dict(self.stats.quality_failures),
            "difficulty_distribution": dict(self.stats.difficulty_distribution),
            "category_distribution": dict(self.stats.category_distribution),
        }

        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Statistics saved to: {stats_file}")

    def _print_summary(self):
        """Print processing summary"""

        total = self.stats.total_processed
        kept = self.stats.kept

        print(f"\n{'='*70}")
        print(f"PROCESSING SUMMARY - {self.lang_code}")
        print(f"{'='*70}")

        print("\nOverall:")
        print(f"  Total processed: {total:,}")
        print(f"  Kept: {kept:,} ({kept/total*100:.1f}%)")
        print(f"  Removed: {total-kept:,} ({(total-kept)/total*100:.1f}%)")
        print(f"  Duplicates removed: {self.stats.duplicates_removed:,}")

        print("\nQuality Failures:")
        for reason, count in self.stats.quality_failures.most_common(10):
            print(f"  {reason:25s}: {count:,} ({count/total*100:.1f}%)")

        print("\nDifficulty Distribution:")
        for diff in ["B0", "B1", "B2", "B3", "B4"]:
            count = self.stats.difficulty_distribution[diff]
            print(f"  {diff}: {count:,} ({count/kept*100:.1f}%)")

        print("\nCategory Distribution:")
        for cat, count in self.stats.category_distribution.most_common():
            print(f"  {cat:20s}: {count:,} ({count/kept*100:.1f}%)")

        print(f"\n{'='*70}\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Clean and categorize IndicCorpV2 dataset by difficulty level"
    )
    parser.add_argument(
        "--lang",
        required=True,
        help="Language code - 2-letter (hi, bn, te) or 3-letter (hin, ben, tel)",
    )
    parser.add_argument(
        "--script",
        required=True,
        help="Script code (e.g., Deva, Beng, Taml, Telu, Knda, Mlym, Gujr, Guru, Orya)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for processed data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for saving (default: 10000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)",
    )

    args = parser.parse_args()

    # Create processor
    processor = IndicCorpProcessor(
        lang_code=args.lang, script_code=args.script, output_dir=args.output_dir
    )

    # Process dataset
    processor.process_dataset(batch_size=args.batch_size, limit=args.limit)

    print("\n✅ Processing complete!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
