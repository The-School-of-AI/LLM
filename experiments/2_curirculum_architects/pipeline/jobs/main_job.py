# T2 Curriculum Band Assignment — Main Job (v7.1, EMR Serverless)
# Covers: large-scale web/book/code sources (RedPajama, FineWeb, Dolma, Sangraha, arXiv, etc.)
# Band range: B0–B5 (full)
# Methodology: docs/band_assignment_methodology.md | Version history: docs/CHANGELOG.md


import argparse
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# =========================================================================
# CONFIGURATION
# =========================================================================

VERSION = "7.1-EMR-SERVERLESS"
INPUT_BASE_DEFAULT = (
    "s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"
)
OUTPUT_BASE_DEFAULT = "s3://t2-datacurriculum-353/processed_dataset/curriculum_data/"
REPORT_BASE = "s3://t2-datacurriculum-353/processed_dataset/stats"

BANDS = ["B0", "B1", "B2", "B3", "B4", "B5"]

# Band probability parameters (from original)
BAND_CENTERS = {"B0": 0.05, "B1": 0.20, "B2": 0.35, "B3": 0.55, "B4": 0.75, "B5": 0.90}
WIDTH = 0.20
EPS = 0.15  # Minimum probability threshold

# =========================================================================
# KEYWORD LISTS (Fast Statistical Indicators)
# =========================================================================

# Code indicators (top discriminative keywords)
CODE_KEYWORDS = [
    "def ",
    "function ",
    "class ",
    "import ",
    "return ",
    "const ",
    "let ",
    "var ",
    "public ",
    "private ",
    "void ",
    "int ",
    "string ",
    "if (",
    "for (",
    "while (",
    "malloc",
    "sizeof",
    "iostream",
    "namespace",
]

# Math/reasoning indicators
MATH_KEYWORDS = [
    "theorem",
    "lemma",
    "proof",
    "corollary",
    "equation",
    "integral",
    "derivative",
    "polynomial",
    "qed",
    "iff",
    "proposition",
]

# Reasoning connectives
REASONING_KEYWORDS = [
    "therefore",
    "thus",
    "hence",
    "consequently",
    "because",
    "since",
    "implies",
    "follows that",
    "we conclude",
    "as a result",
]

# Agentic indicators (B5 markers)
AGENTIC_KEYWORDS = [
    "execute",
    "invoke",
    "call",
    "orchestrate",
    "delegate",
    "workflow",
    "pipeline",
    "task",
    "step",
    "action",
    "tool use",
    "agent",
]

# Chain-of-thought indicators (B3-B5)
COT_KEYWORDS = [
    "let's think",
    "step by step",
    "first",
    "second",
    "next",
    "finally",
    "breaking down",
    "analyzing",
]

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================


def parse_args():
    """Parse command line arguments (replaces Glue getResolvedOptions)."""
    parser = argparse.ArgumentParser(
        description="T2 Fast Curriculum Calculator - EMR Serverless"
    )
    parser.add_argument("--SOURCE", required=True, help="Source folder name (required)")
    parser.add_argument(
        "--INPUT_BASE", default=INPUT_BASE_DEFAULT, help="Input base path"
    )
    parser.add_argument(
        "--OUTPUT_BASE", default=OUTPUT_BASE_DEFAULT, help="Output base path"
    )
    parser.add_argument(
        "--ESTIMATED_SIZE_GB",
        type=float,
        default=None,
        help="Estimated data size in GB",
    )
    return parser.parse_args()


def safe_divide(numerator, denominator, default=0.0):
    """Safe division."""
    if not isinstance(numerator, F.Column):
        numerator = F.lit(numerator)
    if not isinstance(denominator, F.Column):
        denominator = F.lit(denominator)
    if not isinstance(default, F.Column):
        default = F.lit(default)
    return F.when(denominator > 0, numerator / denominator).otherwise(default)


# =========================================================================
# ARXIV CLEANING (Optimized)
# =========================================================================


def clean_arxiv_optimized(df):
    """
    Optimized ArXiv structural cleaner.
    Only applied when source contains 'arxiv'.
    Fast single-pass regex for LaTeX artifact removal.
    """
    print("Applying optimized ArXiv cleaning...")

    # 1. Remove bibliography from end (last match)
    ref_pattern = r"(?s)(.*)\n(\\section\*?\{References\}|\\section\*?\{Bibliography\}|\\begin\{thebibliography\}|\nReferences\n|\nBibliography\n)"
    df = df.withColumn("text", F.regexp_replace(F.col("text"), ref_pattern, "$1"))

    # 2. Single-pass removal: figures, LaTeX commands, citations
    remove_pattern = (
        r"(?s:\\begin\{(figure|tikzpicture)\}.*?\\end\{\1\})|"  # Figures
        r"\\(section|subsection|subsubsection|label|caption)\*?\{.*?\}|"  # Structure commands
        r"\\cite\{.*?\}"  # Citations
    )
    df = df.withColumn("text", F.regexp_replace(F.col("text"), remove_pattern, ""))

    return df


# =========================================================================
# ADAPTIVE SAMPLING
# =========================================================================


def create_adaptive_sample(df):
    """
    Create adaptive text sample based on document size.
    - Short docs (<1K chars): Use full text
    - Medium docs (1K-10K): Use first 3K chars
    - Long docs (10K-50K): Use first 10K chars
    - Very long docs (>50K): Use first 15K chars
    """
    print("Creating adaptive text samples...")

    df = df.withColumn("char_length", F.length(F.col("text")))

    # Adaptive sample size
    df = df.withColumn(
        "sample_size",
        F.when(F.col("char_length") < 1000, F.col("char_length"))
        .when(F.col("char_length") < 10000, F.lit(5000))
        .when(F.col("char_length") < 50000, F.lit(15000))
        .otherwise(F.lit(25000)),
    )

    # Use expr for dynamic substring
    df = df.withColumn(
        "text_sample", F.expr("substring(text, 1, cast(sample_size as int))")
    )

    # Drop sample_size - no longer needed
    df = df.drop("sample_size")

    return df


# =========================================================================
# FAST METRICS (No Regex, Pure String Operations)
# =========================================================================


def compute_basic_stats(df):
    """Basic statistics (very fast)."""
    print("Computing basic statistics...")

    df = df.withColumn("byte_length", F.length(F.encode(F.col("text"), "utf-8")))

    # Word and line count
    df = df.withColumn(
        "word_count", F.greatest(F.size(F.split(F.col("text"), r"\s+")) - 1, F.lit(1))
    )
    df = df.withColumn(
        "line_count", F.greatest(F.size(F.split(F.col("text"), "\n")) - 1, F.lit(1))
    )

    # Token estimate (source-aware)
    df = df.withColumn(
        "token_count_estimate",
        F.when(
            F.col("source").startswith("sangraha_"),
            (F.col("word_count") * 1.8).cast("int"),
        ).otherwise((F.col("word_count") * 1.3).cast("int")),
    )

    # Fertility estimate
    df = df.withColumn(
        "fertility_estimate",
        safe_divide(F.col("char_length"), F.col("token_count_estimate"), 1.0),
    )

    return df


def compute_character_stats(df):
    """Character-level statistics (fast)."""
    print("Computing character statistics...")

    text_col = F.col("text_sample")
    char_len = F.length(text_col)

    # Punctuation ratio - using translate instead of regex
    punct_chars = ".,;:()[]{}!?"
    df = df.withColumn(
        "punct_count", char_len - F.length(F.translate(text_col, punct_chars, ""))
    )
    df = df.withColumn("punct_ratio", safe_divide(F.col("punct_count"), char_len))

    # Digit ratio - using translate for 0-9
    df = df.withColumn(
        "digit_count", char_len - F.length(F.translate(text_col, "0123456789", ""))
    )
    df = df.withColumn("digit_ratio", safe_divide(F.col("digit_count"), char_len))

    # Special chars (code indicator) - using translate
    special_chars = "{}=&|<>"
    df = df.withColumn(
        "special_count", char_len - F.length(F.translate(text_col, special_chars, ""))
    )
    df = df.withColumn("special_ratio", safe_divide(F.col("special_count"), char_len))

    # Uppercase ratio (code/acronyms) - using SQL expression
    df = df.withColumn(
        "upper_count",
        F.expr(
            "length(text_sample) - length(regexp_replace(text_sample, '[A-Z]', ''))"
        ),
    )
    df = df.withColumn("upper_ratio", safe_divide(F.col("upper_count"), char_len))

    # Drop raw count columns - only ratios needed
    df = df.drop("punct_count", "digit_count", "special_count", "upper_count")

    return df


def compute_word_stats(df):
    """Word-level statistics (fast array operations)."""
    print("Computing word statistics...")

    df = df.withColumn("words", F.split(F.col("text_sample"), r"\s+"))
    # Unique word ratio (vocabulary diversity)
    df = df.withColumn("unique_words", F.size(F.array_distinct(F.col("words"))))
    df = df.withColumn(
        "unique_token_ratio", safe_divide(F.col("unique_words"), F.col("word_count"))
    )

    # Average word length
    df = df.withColumn("all_words_concat", F.concat_ws("", F.col("words")))
    df = df.withColumn("total_word_chars", F.length(F.col("all_words_concat")))
    df = df.withColumn(
        "avg_word_length", safe_divide(F.col("total_word_chars"), F.col("word_count"))
    )
    df = df.drop("all_words_concat")

    # Compression ratio proxy
    df = df.withColumn(
        "compression_ratio", safe_divide(F.col("byte_length"), F.col("char_length"))
    )

    # Drop word-related intermediates
    df = df.drop("words", "unique_words", "total_word_chars")

    return df


def compute_keyword_scores(df):
    """Fast keyword-based scoring (simple contains, no regex)."""
    print("Computing keyword scores...")

    text_lower = F.lower(F.col("text_sample"))

    # Code keywords
    code_hits = sum(
        [F.when(text_lower.contains(kw), 1).otherwise(0) for kw in CODE_KEYWORDS]
    )
    df = df.withColumn("code_keyword_count", code_hits)

    # Math keywords
    math_hits = sum(
        [F.when(text_lower.contains(kw), 1).otherwise(0) for kw in MATH_KEYWORDS]
    )
    df = df.withColumn("math_keyword_count", math_hits)

    # Reasoning keywords
    reasoning_hits = sum(
        [F.when(text_lower.contains(kw), 1).otherwise(0) for kw in REASONING_KEYWORDS]
    )
    df = df.withColumn("reasoning_keyword_count", reasoning_hits)

    # Agentic keywords (B5 markers)
    agentic_hits = sum(
        [F.when(text_lower.contains(kw), 1).otherwise(0) for kw in AGENTIC_KEYWORDS]
    )
    df = df.withColumn("agentic_keyword_count", agentic_hits)

    # CoT keywords
    cot_hits = sum(
        [F.when(text_lower.contains(kw), 1).otherwise(0) for kw in COT_KEYWORDS]
    )
    df = df.withColumn("cot_keyword_count", cot_hits)

    # Drop text_sample - all text processing complete
    df = df.drop("text_sample")

    return df


# =========================================================================
# COMPOSITE SCORES (Aligned with Band Definitions)
# =========================================================================


def compute_composite_scores(df):
    """
    Compute composite scores aligned with band definitions.
    """
    print("Computing composite scores...")

    # Code score (B1=trivial, B3=meaningful, B4=hard)
    df = df.withColumn(
        "code_score",
        (
            F.col("special_ratio") * 30
            + F.col("digit_ratio") * 20
            + F.col("code_keyword_count") * 5
            + F.when(F.col("avg_word_length") > 8, 5).otherwise(0)
        ).cast("int"),
    )

    # Math score (B4 indicator)
    df = df.withColumn(
        "math_score",
        (
            F.col("math_keyword_count") * 6
            + F.col("digit_ratio") * 15
            + F.col("special_ratio") * 8
        ).cast("int"),
    )

    # Reasoning score (B3-B4 indicator)
    df = df.withColumn(
        "reasoning_score",
        (F.col("reasoning_keyword_count") * 6 + F.col("upper_ratio") * 8).cast("int"),
    )

    # Agentic score (B5 indicator)
    df = df.withColumn(
        "agentic_score", (F.col("agentic_keyword_count") * 6).cast("int")
    )

    # CoT score (B3-B5, but capped per B5 definition)
    df = df.withColumn(
        "cot_score",
        (F.col("cot_keyword_count") * 5 + F.col("reasoning_keyword_count") * 2).cast(
            "int"
        ),
    )

    # Boolean flags
    df = df.withColumn("has_code", F.col("code_score") >= 10)
    df = df.withColumn("has_math", F.col("math_score") >= 8)
    df = df.withColumn("has_reasoning", F.col("reasoning_score") >= 6)
    df = df.withColumn("has_agentic", F.col("agentic_score") >= 8)
    df = df.withColumn("has_cot", F.col("cot_score") >= 10)

    # Drop raw keyword counts
    df = df.drop(
        "code_keyword_count",
        "math_keyword_count",
        "reasoning_keyword_count",
        "agentic_keyword_count",
        "cot_keyword_count",
    )

    return df


def compute_difficulty_score(df):
    """
    Simple difficulty score.
    Maps to band centers: B0=0.05, B1=0.20, B2=0.35, B3=0.55, B4=0.75, B5=0.90
    """
    print("Computing difficulty score...")

    # Normalize components
    df = df.withColumn(
        "vocab_component", F.least(F.col("unique_token_ratio") * 2.5, F.lit(1.0))
    )

    df = df.withColumn(
        "length_component", F.least((F.col("avg_word_length") - 4) / 6, F.lit(1.0))
    )

    df = df.withColumn(
        "structure_component", F.least(F.col("punct_ratio") * 3, F.lit(1.0))
    )

    df = df.withColumn(
        "specialty_component",
        F.least(
            (F.col("code_score") + F.col("math_score") + F.col("reasoning_score")) / 60,
            F.lit(1.0),
        ),
    )

    # Weighted difficulty
    df = df.withColumn(
        "difficulty_score",
        (
            F.col("vocab_component") * 0.25
            + F.col("length_component") * 0.25
            + F.col("structure_component") * 0.20
            + F.col("specialty_component") * 0.30
        ),
    )

    # Drop difficulty component columns
    df = df.drop(
        "vocab_component",
        "length_component",
        "structure_component",
        "specialty_component",
    )

    return df


# =========================================================================
# PROBABILISTIC BANDING (From Original Design)
# =========================================================================


def assign_curriculum_bands_probabilistic(df):
    """
    Assign probabilistic curriculum bands.
    Returns band probabilities (band_p_B0 through band_p_B5) and final band.
    """
    print("Assigning probabilistic curriculum bands...")

    # Initialize weights for each band (Gaussian around centers)
    for band in BANDS:
        center = BAND_CENTERS[band]
        df = df.withColumn(
            f"_w_{band}",
            F.greatest(
                F.lit(0.0),
                F.lit(1.0)
                - F.abs(F.col("difficulty_score") - F.lit(center)) / F.lit(WIDTH),
            ),
        )

    # Content-based nudges (from band definitions)

    # B1: Trivial code only
    df = df.withColumn(
        "_w_B1",
        F.when(
            (F.col("code_score") > 0) & (F.col("code_score") < 15),
            F.col("_w_B1") + 0.05,
        ).otherwise(F.col("_w_B1")),
    )

    # B2: Intro technical, implicit reasoning
    df = df.withColumn(
        "_w_B2",
        F.when(
            (F.col("code_score") >= 15) & (F.col("code_score") < 25),
            F.col("_w_B2") + 0.08,
        )
        .when(F.col("reasoning_score") >= 5, F.col("_w_B2") + 0.05)
        .otherwise(F.col("_w_B2")),
    )

    # B3: Meaningful code, multi-step reasoning
    df = df.withColumn(
        "_w_B3",
        F.when(F.col("code_score") >= 25, F.col("_w_B3") + 0.10)
        .when(F.col("reasoning_score") >= 8, F.col("_w_B3") + 0.08)
        .when(F.col("cot_score") >= 10, F.col("_w_B3") + 0.05)
        .otherwise(F.col("_w_B3")),
    )

    # B4: Math, proofs, hard code
    df = df.withColumn(
        "_w_B4",
        F.when(F.col("math_score") >= 12, F.col("_w_B4") + 0.15)
        .when(F.col("code_score") >= 40, F.col("_w_B4") + 0.12)
        .when(F.col("reasoning_score") >= 12, F.col("_w_B4") + 0.10)
        .otherwise(F.col("_w_B4")),
    )

    # B5: Agentic traces, advanced reasoning (but CoT capped)
    df = df.withColumn(
        "_w_B5",
        F.when(F.col("agentic_score") >= 8, F.col("_w_B5") + 0.20)
        .when(F.col("math_score") >= 20, F.col("_w_B5") + 0.12)
        .when(
            (F.col("reasoning_score") >= 15) & (F.col("code_score") >= 30),
            F.col("_w_B5") + 0.10,
        )
        .otherwise(F.col("_w_B5")),
    )

    # Normalize to probabilities
    df = df.withColumn("_total_weight", sum([F.col(f"_w_{b}") for b in BANDS]))
    df = df.withColumn(
        "_total_weight",
        F.when(F.col("_total_weight") > 0, F.col("_total_weight")).otherwise(
            F.lit(1.0)
        ),
    )

    for band in BANDS:
        df = df.withColumn(
            f"band_p_{band}", F.col(f"_w_{band}") / F.col("_total_weight")
        )

    # Assign final band (lowest credible probability)
    df = df.withColumn(
        "band",
        F.when(F.col("band_p_B0") >= EPS, "B0")
        .when(F.col("band_p_B1") >= EPS, "B1")
        .when(F.col("band_p_B2") >= EPS, "B2")
        .when(F.col("band_p_B3") >= EPS, "B3")
        .when(F.col("band_p_B4") >= EPS, "B4")
        .otherwise("B5"),
    )

    df = df.withColumn("assigned_band", F.col("band"))

    # Cleanup temp columns
    for band in BANDS:
        df = df.drop(f"_w_{band}")
    df = df.drop("_total_weight")

    return df


# =========================================================================
# ENHANCED QUALITY FILTERS (Stage 1 & 2 - Fast)
# =========================================================================


def compute_noise_metrics(df):
    """
    Compute fast noise detection metrics.
    Optimized for speed with minimal regex.
    """
    print("Computing noise detection metrics...")

    # Whitespace ratio - count space, tab, newline, carriage return
    text_len = F.length(F.col("text"))
    df = df.withColumn(
        "whitespace_count",
        text_len - F.length(F.translate(F.col("text"), " \t\n\r", "")),
    )
    df = df.withColumn(
        "whitespace_ratio", safe_divide(F.col("whitespace_count"), F.col("char_length"))
    )

    # URL detection (using regexp_count for reliability)
    df = df.withColumn("url_count", F.expr("regexp_count(text, 'https?://')"))
    df = df.withColumn(
        "url_ratio", safe_divide(F.col("url_count"), F.col("word_count"))
    )

    # Boilerplate keywords (fast contains check)
    boilerplate_keywords = [
        "cookie policy",
        "privacy policy",
        "terms of service",
        "all rights reserved",
        "subscribe to",
        "sign up",
    ]
    boilerplate_hits = sum(
        [
            F.when(F.lower(F.col("text")).contains(kw), 1).otherwise(0)
            for kw in boilerplate_keywords
        ]
    )
    df = df.withColumn("boilerplate_count", boilerplate_hits)
    df = df.withColumn(
        "boilerplate_ratio",
        safe_divide(F.col("boilerplate_count"), F.lit(len(boilerplate_keywords))),
    )

    # Thread markers (fast simple pattern)
    thread_keywords = [">>", "replied to", "in response to", "re:"]
    thread_hits = sum(
        [F.when(F.col("text").contains(kw), 1).otherwise(0) for kw in thread_keywords]
    )
    df = df.withColumn("thread_marker_count", thread_hits)

    # Drop text column - no longer needed (major memory savings)
    df = df.drop("text", "whitespace_count")

    return df


def apply_enhanced_quality_filters(df):
    """
    Apply Stage 1 & 2 rejection rules (optimized for speed).
    Target: ~95-98% pass-through rate.
    """
    print("Applying enhanced quality filters...")

    # Initialize rejection tracking
    df = df.withColumn("is_rejected", F.lit(False))
    df = df.withColumn("rejection_reason", F.lit(""))
    df = df.withColumn("rejection_level", F.lit(None).cast(IntegerType()))

    # ===== STAGE 1: Physical & Basic Corruption =====

    # Rule 1.1: Byte length < 50
    cond = F.col("byte_length") < 50
    df = df.withColumn(
        "is_rejected", F.when(cond, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond & (F.col("rejection_reason") == ""), "too_short_bytes").otherwise(
            F.col("rejection_reason")
        ),
    )
    df = df.withColumn(
        "rejection_level",
        F.when(cond & F.col("rejection_level").isNull(), 1).otherwise(
            F.col("rejection_level")
        ),
    )

    # Rule 1.2: Char length < 20
    cond = F.col("char_length") < 20
    df = df.withColumn(
        "is_rejected", F.when(cond, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond & (F.col("rejection_reason") == ""), "too_short_chars").otherwise(
            F.col("rejection_reason")
        ),
    )
    df = df.withColumn(
        "rejection_level",
        F.when(cond & F.col("rejection_level").isNull(), 1).otherwise(
            F.col("rejection_level")
        ),
    )

    # Rule 1.3: Token count < 10
    cond = F.col("token_count_estimate") < 10
    df = df.withColumn(
        "is_rejected", F.when(cond, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond & (F.col("rejection_reason") == ""), "too_short_tokens").otherwise(
            F.col("rejection_reason")
        ),
    )
    df = df.withColumn(
        "rejection_level",
        F.when(cond & F.col("rejection_level").isNull(), 1).otherwise(
            F.col("rejection_level")
        ),
    )

    # ===== STAGE 2: Noise & Spam =====

    # Rule 2.1: Repetitive template (all same words)
    cond = (F.col("unique_token_ratio") < 0.01) & (F.col("word_count") > 200)
    df = df.withColumn(
        "is_rejected", F.when(cond, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(
            cond & (F.col("rejection_reason") == ""), "repetitive_template"
        ).otherwise(F.col("rejection_reason")),
    )
    df = df.withColumn(
        "rejection_level",
        F.when(cond & F.col("rejection_level").isNull(), 2).otherwise(
            F.col("rejection_level")
        ),
    )

    # Rule 2.2: Excessive whitespace
    cond = F.col("whitespace_ratio") > 0.95
    df = df.withColumn(
        "is_rejected", F.when(cond, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(
            cond & (F.col("rejection_reason") == ""), "excessive_whitespace"
        ).otherwise(F.col("rejection_reason")),
    )
    df = df.withColumn(
        "rejection_level",
        F.when(cond & F.col("rejection_level").isNull(), 2).otherwise(
            F.col("rejection_level")
        ),
    )

    # Rule 2.3: Link spam
    cond = (F.col("url_ratio") > 0.7) & (F.col("url_count") > 50)
    df = df.withColumn(
        "is_rejected", F.when(cond, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond & (F.col("rejection_reason") == ""), "link_spam").otherwise(
            F.col("rejection_reason")
        ),
    )
    df = df.withColumn(
        "rejection_level",
        F.when(cond & F.col("rejection_level").isNull(), 2).otherwise(
            F.col("rejection_level")
        ),
    )

    # Rule 2.4: Boilerplate spam
    cond = F.col("boilerplate_ratio") > 0.50
    df = df.withColumn(
        "is_rejected", F.when(cond, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond & (F.col("rejection_reason") == ""), "boilerplate_spam").otherwise(
            F.col("rejection_reason")
        ),
    )
    df = df.withColumn(
        "rejection_level",
        F.when(cond & F.col("rejection_level").isNull(), 2).otherwise(
            F.col("rejection_level")
        ),
    )

    # Rule 2.5: Thread fragment
    cond = (F.col("thread_marker_count") > 5) & (F.col("token_count_estimate") < 200)
    df = df.withColumn(
        "is_rejected", F.when(cond, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(
            cond & (F.col("rejection_reason") == ""), "orphaned_thread_fragment"
        ).otherwise(F.col("rejection_reason")),
    )
    df = df.withColumn(
        "rejection_level",
        F.when(cond & F.col("rejection_level").isNull(), 2).otherwise(
            F.col("rejection_level")
        ),
    )

    return df


# =========================================================================
# OUTPUT PREPARATION
# =========================================================================


def prepare_output_columns(df, include_rejection=False):
    """
    Prepare output columns matching V5 schema.
    Includes uuid, file_path for tracking, and maintains column order.
    """
    # Core columns (with tracking fields)
    core_cols = [
        "uuid",
        "id",
        "file_path",
        "source",
        "domain",
        "hash",
        "language",
        "metadata",
    ]

    # Band columns
    band_cols = [
        "assigned_band",
        "band_p_B0",
        "band_p_B1",
        "band_p_B2",
        "band_p_B3",
        "band_p_B4",
        "band_p_B5",
        "band",
        "difficulty_score",
    ]

    # Score columns
    score_cols = [
        "has_code",
        "has_cot",
        "has_reasoning",
        "has_agentic",
        "agentic_score",
        "cot_score",
        "reasoning_score",
        "code_score",
        "math_score",
    ]

    # Metric columns
    metric_cols = [
        "byte_length",
        "word_count",
        "unique_token_ratio",
        "compression_ratio",
        "token_count_estimate",
        "fertility_estimate",
    ]

    # Rejection columns (when applicable)
    rejection_cols = ["is_rejected", "rejection_reason", "rejection_level"]

    # Build select list
    select_cols = core_cols + band_cols + score_cols + metric_cols

    if include_rejection:
        select_cols += rejection_cols

    # Only select columns that exist in dataframe
    existing_cols = [c for c in select_cols if c in df.columns]
    return df.select(*existing_cols)


# =========================================================================
# MAIN PIPELINE
# =========================================================================


def main():
    args = parse_args()

    input_base = args.INPUT_BASE
    output_base = args.OUTPUT_BASE
    source_filter = args.SOURCE

    input_path = f"{input_base}/source={source_filter}"
    output_bands = f"{output_base}/source={source_filter}/bands"
    output_rejected = f"{output_base}/source={source_filter}/rejections"
    output_stats = f"{REPORT_BASE}/source={source_filter}"

    # Initialize Spark (EMR Serverless compatible)
    print("=" * 60)
    print(f"T2 Fast Curriculum Calculator v{VERSION}")
    print("=" * 60)
    print(f"Processing: {source_filter}")
    print(f"Input: {input_path}")
    print(f"Output: {output_base}")
    print("=" * 60)

    spark = SparkSession.builder.appName(
        f"T2_Fast_Curriculum_{source_filter}"
    ).getOrCreate()

    # Optimized Spark config
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.files.maxPartitionBytes", "268435456")
    spark.conf.set("spark.default.parallelism", "200")

    print(f"Spark version: {spark.version}")

    # Read data
    print("Reading input data...")
    df = spark.read.parquet(input_path)

    # Add tracking fields (uuid and file_path)
    print("Adding tracking metadata...")
    df = df.withColumn("uuid", F.expr("uuid()"))
    prefix_to_remove = f"{input_base}/"
    df = df.withColumn("file_path", F.input_file_name())
    df = df.withColumn(
        "file_path", F.regexp_replace(F.col("file_path"), prefix_to_remove, "")
    )

    # # Not applied as the text column is not used by t3
    # # Apply ArXiv cleaning if applicable (before processing)
    # if source_filter and "arxiv" in source_filter.lower():
    #     print("ArXiv source detected - applying optimized LaTeX cleaning...")
    #     df = clean_arxiv_optimized(df)

    # Process pipeline
    print("Processing metrics pipeline...")
    start_time = datetime.now()

    df = create_adaptive_sample(df)
    df = compute_basic_stats(df)  # Uses text, keeps it for noise metrics

    # Compute noise metrics early (needs full text for URL/boilerplate detection)
    df = compute_noise_metrics(df)  # Drops text at the end

    # Continue with text_sample-based processing
    df = compute_character_stats(df)
    df = compute_word_stats(df)
    df = compute_keyword_scores(df)  # Drops text_sample at the end
    df = compute_composite_scores(df)
    df = compute_difficulty_score(df)

    # Apply quality filters
    df = apply_enhanced_quality_filters(df)

    # Final cleanup: Drop noise detection intermediates used only for rejection rules
    print("Dropping final noise detection intermediates...")
    df = df.drop(
        "url_count",
        "url_ratio",
        "whitespace_ratio",
        "boilerplate_count",
        "boilerplate_ratio",
        "thread_marker_count",
    )

    # Only assign bands to non-rejected documents
    df = assign_curriculum_bands_probabilistic(df)

    # Split accepted/rejected
    rejected = df.filter(F.col("is_rejected"))
    accepted = df.filter(~F.col("is_rejected"))

    # Prepare outputs
    rejected_out = prepare_output_columns(rejected, include_rejection=True)
    accepted_out = prepare_output_columns(accepted, include_rejection=False)

    # Write rejections
    print(f"\nWriting rejections to {output_rejected}")
    rejected_out.write.mode("overwrite").option("compression", "zstd").parquet(
        output_rejected
    )

    # Rejection statistics report
    # rejection_stats = rejected_out.select("source", "token_count_estimate") \
    #     .groupBy("source") \
    #     .agg(
    #         F.sum("token_count_estimate").alias("total_tokens_estimated"),
    #         F.count("*").alias("record_count")
    #     )

    # rejection_stats.write.mode("overwrite").option("header", "true") \
    #     .csv(f"{output_stats}/rejections")

    # Write bands
    print(f"Writing bands to {output_bands}")
    accepted_out.write.mode("overwrite").partitionBy("band").option(
        "compression", "zstd"
    ).parquet(output_bands)

    # Generate detailed statistics
    print("\nGenerating band statistics...")

    # band_stats = accepted_out.select("assigned_band", "source", "token_count_estimate") \
    #     .groupBy("assigned_band", "source") \
    #     .agg(
    #         F.sum("token_count_estimate").alias("total_tokens_estimated"),
    #         F.count("*").alias("record_count")
    #     ) \
    #     .orderBy("assigned_band", "source")

    # band_stats.write.mode("overwrite").option("header", "true").csv(f"{output_stats}/bands")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Duration: {duration:.1f} seconds")
    print("Output structure:")
    print(f"  - Bands: {output_bands}/band=<B0-B5>/")
    print(f"  - Rejections: {output_rejected}/")
    print(f"  - Statistics: {output_stats}/")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
