# T2 Curriculum Band Assignment — Student Data Job (v7.1, EMR Serverless)
# Covers: ERAv4 student-generated Q&A drills + Samvaad conversation
# Band range: B0–B2 (tight, foundational)
# Methodology: docs/band_assignment_methodology.md | Version history: docs/CHANGELOG.md


import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# =========================================================================
# VERSION & PATHS
# =========================================================================

VERSION = "1.0-STUDENT"
INPUT_BASE_DEFAULT = (
    "s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"
)
OUTPUT_BASE_DEFAULT = "s3://t2-datacurriculum-353/processed_dataset/curriculum_data/"
REPORT_BASE = "s3://t2-datacurriculum-353/processed_dataset/stats"

BANDS = ["B0", "B1", "B2", "B3", "B4", "B5"]

# Probabilistic banding constants (same as v7.1)
BAND_CENTERS = {"B0": 0.05, "B1": 0.20, "B2": 0.35, "B3": 0.55, "B4": 0.75, "B5": 0.90}
WIDTH = 0.20
EPS = 0.15

# =========================================================================
# SOURCE REGISTRY
# =========================================================================
# Each entry: (substring_match, floor_idx, ceil_idx, description, domain_override)
# floor/ceil are 0-based band indices (B0=0 … B5=5).
# domain_override: if not None, replaces T1's domain tag in output.
#
# Band rationale:
#   erav4_lang    → spelling/phonics/literacy Q&A, B0-B1
#                   B0 for simple letter drills; B1 for vocabulary/reading items
#                   domain=language_literacy
#   erav4_math    → elementary math Q&A, B0-B2
#                   B0 for basic arithmetic; B1-B2 for word problems/reasoning
#                   domain=education
#   erav4_pattern → pattern recognition Q&A, B0-B2
#                   B0 for simple sequences; B1-B2 for multi-step pattern tasks
#                   domain=education
#   samvaad       → everyday conversation, B0-B2
#                   B0 for simple exchanges; B1-B2 for richer conversations
#                   domain=conversation
#   (fallback)    → unknown source → B0-B1, domain preserved from T1
#
# Most-specific entries first; first substring match wins.

SOURCE_REGISTRY = [
    # ── GROUP 1: LANGUAGE-LITERACY Q&A ───────────────────────────────────
    ("erav4_lang", 0, 1, "language_literacy"),
    # ── GROUP 2: MATH Q&A ─────────────────────────────────────────────────
    ("erav4_math", 0, 2, "education"),
    # ── GROUP 3: PATTERN RECOGNITION Q&A ─────────────────────────────────
    ("erav4_pattern", 0, 2, "education"),
    # ── SAMVAAD: EVERYDAY CONVERSATION ───────────────────────────────────
    ("samvaad", 0, 2, "conversation"),
]

# =========================================================================
# LANGUAGE-AWARE TOKEN ESTIMATION
# =========================================================================
# Maps `language` column value → bytes-per-token divisor.
# Source: rough empirical estimates for LLaMA-2 style tokenizers.
#   English (ASCII):      1 byte/char,  4 chars/token → ~4 bytes/token
#   Devanagari (hi, mr):  3 bytes/char, 2 chars/token → ~6 bytes/token
#   Bengali (bn, as):     3 bytes/char, 2.2 chars/token → ~6.5 bytes/token
#   Tamil (ta):           3 bytes/char, 1.8 chars/token → ~5.5 bytes/token
#   Telugu/Kannada (te,kn): 3 bytes/char, 2 chars/token → ~6 bytes/token
#   Malayalam (ml):       3 bytes/char, 1.8 chars/token → ~5.5 bytes/token
#   Gujarati (gu):        3 bytes/char, 2.2 chars/token → ~6.5 bytes/token
#   Gurmukhi/Punjabi (pa): 3 bytes/char, 2 chars/token → ~6 bytes/token
#   Odia (or):            3 bytes/char, 2 chars/token → ~6 bytes/token
# Conservative defaults: Indic → 6.0, unknown → 4.5

LANG_BYTES_PER_TOKEN = {
    "en": 4.0,
    "hi": 6.0,
    "mr": 6.0,
    "bn": 6.5,
    "as": 6.5,
    "ta": 5.5,
    "te": 6.0,
    "kn": 6.0,
    "ml": 5.5,
    "gu": 6.5,
    "pa": 6.0,
    "or": 6.0,
}
DEFAULT_BYTES_PER_TOKEN = 4.5  # conservative for mixed/unknown


# Build a Spark column expression for the language-aware divisor
def build_lang_divisor_expr():
    """Returns a Column expression: language → bytes_per_token divisor."""
    expr = F.lit(DEFAULT_BYTES_PER_TOKEN)
    for lang_code, divisor in LANG_BYTES_PER_TOKEN.items():
        expr = F.when(F.col("language") == lang_code, F.lit(divisor)).otherwise(expr)
    return expr


# =========================================================================
# INDIC LANGUAGE SETS
# =========================================================================
INDIC_LANGS = {"hi", "mr", "bn", "as", "ta", "te", "kn", "ml", "gu", "pa", "or"}


def build_is_indic_expr():
    """Returns Column: 1 if language is Indic, 0 otherwise."""
    expr = F.lit(0)
    for lang in INDIC_LANGS:
        expr = F.when(F.col("language") == lang, F.lit(1)).otherwise(expr)
    return expr


# =========================================================================
# KEYWORD LISTS (lightweight — no regex)
# =========================================================================

# Conversation markers (samvaad-style)
CONV_MARKERS = [
    "user:",
    "human:",
    "assistant:",
    "<|user|>",
    "<|assistant|>",
    "system:",
    "[inst]",
    "[/inst]",
]

# Q&A phonics format markers — boost Q&A recognition
QA_MARKERS = [
    "spelling",
    "वर्तनी",
    "বানান",
    "எழுத்து",
    "స్పెల్లింగ్",
    "जोडणी",
    "ਸਪੈਲਿੰਗ",
    "ക്ഷരം",
    "ಸ್ಪೆಲಿಂಗ್",
    "শব্দেৰ",
    "ओडिশা",
    "বানান",
    "สปเปลลิง",
]

# Letters/sounds vocabulary (very high-frequency short words in answers)
LETTER_SIGNALS = [
    "a, b",
    "b, c",
    "c, d",  # English letter sequences
    "क, म",
    "घ, र",
    "प, ा",  # Hindi letter sequences
    "ব, ই",
    "ঘ, র",  # Bengali
    "வ, ீ",
    "ந, ீ",  # Tamil
    "ఇ, ల",
    "న, ీ",  # Telugu
    "ಮ, ನ",
    "ನ, ೀ",  # Kannada
    "ഘ, ര",
    "വ, ീ",  # Malayalam
    "ઘ, ર",
    "પ, ા",  # Gujarati
    "घ, र",
    "प, ा",  # Marathi (Devanagari)
    "ਘ, ਰ",
    "ਪ, ਾ",  # Punjabi
    "ব, হ",
    "ঘ, ৰ",  # Odia / Assamese
]

# =========================================================================
# HELPERS
# =========================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="T2 Student-Generated Curriculum Calculator"
    )
    p.add_argument("--SOURCE", required=True, help="Source folder name")
    p.add_argument("--INPUT_BASE", default=INPUT_BASE_DEFAULT)
    p.add_argument("--OUTPUT_BASE", default=OUTPUT_BASE_DEFAULT)
    p.add_argument("--ESTIMATED_SIZE_GB", type=float, default=None)
    return p.parse_args()


def safe_divide(num, den, default=0.0):
    if not isinstance(num, F.Column):
        num = F.lit(num)
    if not isinstance(den, F.Column):
        den = F.lit(den)
    if not isinstance(default, F.Column):
        default = F.lit(default)
    return F.when(den > 0, num / den).otherwise(default)


def band_name_to_idx(col):
    return (
        F.when(col == "B0", 0)
        .when(col == "B1", 1)
        .when(col == "B2", 2)
        .when(col == "B3", 3)
        .when(col == "B4", 4)
        .otherwise(5)
    )


def idx_to_band_name(col):
    return (
        F.when(col == 0, "B0")
        .when(col == 1, "B1")
        .when(col == 2, "B2")
        .when(col == 3, "B3")
        .when(col == 4, "B4")
        .otherwise("B5")
    )


# =========================================================================
# PIPELINE STAGES
# =========================================================================


def create_text_sample(df):
    """
    Cap text for metric computation.
    Student-gen records are very short (usually <500 chars) so we rarely
    need to truncate, but keep the guard for samvaad and future use.
    """
    df = df.withColumn("byte_length", F.length(F.encode(F.col("text"), "UTF-8")))
    df = df.withColumn("char_length", F.length(F.col("text")))

    return df


def compute_basic_stats(df):
    """Word count, unique token ratio, compression proxy."""
    df = df.withColumn("word_count", F.size(F.split(F.col("text"), r"\s+")))
    df = df.withColumn(
        "unique_token_ratio",
        safe_divide(
            F.size(F.array_distinct(F.split(F.col("text"), r"\s+"))),
            F.col("word_count"),
            default=0.0,
        ),
    )
    # Compression proxy: high ratio = repetitive (Q&A drills repeat patterns)
    df = df.withColumn(
        "compression_ratio",
        safe_divide(F.col("char_length"), F.col("byte_length"), default=1.0),
    )
    return df


def compute_qa_signals(df):
    """
    Detect Q&A format structure specific to student-generated data.
    Uses split() — no regex.
    """
    # Count Q?A pairs by '?' separator
    df = df.withColumn(
        "qa_pair_count", F.greatest(F.lit(0), F.size(F.split(F.col("text"), r"\?")) - 1)
    )

    # Count Devanagari danda '।' — marks Indic Q?A pair boundaries
    df = df.withColumn("danda_count", F.size(F.split(F.col("text"), "।")) - 1)

    # Is this primarily Q&A phonics format?
    # Heuristic: has at least one '?' and answer contains comma-separated chars
    df = df.withColumn("comma_count", F.size(F.split(F.col("text"), ",")) - 1)

    # QA density: qa_pairs / word_count — high for phonics drills
    df = df.withColumn(
        "qa_density",
        safe_divide(F.col("qa_pair_count"), F.col("word_count"), default=0.0),
    )

    # is_indic: language-based flag
    df = df.withColumn("is_indic", build_is_indic_expr())

    return df


def detect_content_signals(df):
    """
    Keyword scoring.  Keeps only columns needed for downstream computation;
    drops text_sample at end of this stage.
    """
    text_lower = F.lower(F.col("text"))

    # Count QA format markers (spelling, वर्तनी, etc.)
    qa_marker_count = F.lit(0)
    for kw in QA_MARKERS:
        qa_marker_count = qa_marker_count + F.when(
            text_lower.contains(kw), 1
        ).otherwise(0)
    df = df.withColumn("qa_marker_count", qa_marker_count.cast(IntegerType()))

    # Count letter-answer sequences (comma-separated single chars)
    letter_signal_count = F.lit(0)
    for sig in LETTER_SIGNALS:
        letter_signal_count = letter_signal_count + F.when(
            F.col("text").contains(sig), 1
        ).otherwise(0)
    df = df.withColumn("letter_signal_count", letter_signal_count.cast(IntegerType()))

    # Conversation markers (for samvaad)
    conv_count = F.lit(0)
    for kw in CONV_MARKERS:
        conv_count = conv_count + F.when(text_lower.contains(kw), 1).otherwise(0)
    df = df.withColumn("conv_marker_count", conv_count.cast(IntegerType()))

    # Code signal (should be zero for this data; include for schema completeness)
    df = df.withColumn("code_score", F.lit(0).cast(IntegerType()))

    # Math signal (low for this data)
    df = df.withColumn("math_score", F.lit(0).cast(IntegerType()))

    # CoT / reasoning / agentic (all zero for this data)
    df = df.withColumn("cot_score", F.lit(0).cast(IntegerType()))
    df = df.withColumn("reasoning_score", F.lit(0).cast(IntegerType()))
    df = df.withColumn("agentic_score", F.lit(0).cast(IntegerType()))

    # has_* flags
    df = df.withColumn("has_code", F.lit(False))
    df = df.withColumn("has_cot", F.lit(False))
    df = df.withColumn("has_reasoning", F.lit(False))
    df = df.withColumn("has_agentic", F.lit(False))

    df = df.drop("text")
    return df


def compute_language_aware_token_estimate(df):
    """
    Token count estimate using per-language bytes-per-token divisor.
    Replaces the single byte_length/4.0 formula from v7.1 with a
    language-aware version so Indic scripts are correctly estimated.

    fertility_estimate is back-computed as token_count / (byte_length/4)
    so it tracks the relative over/under compared to the naive English formula.
    This keeps the column meaningful for downstream analysis.
    """
    lang_divisor = build_lang_divisor_expr()
    df = df.withColumn("_lang_divisor", lang_divisor)

    df = df.withColumn(
        "token_count_estimate",
        F.greatest(
            F.lit(1),
            (
                safe_divide(F.col("byte_length"), F.col("_lang_divisor"), default=1.0)
            ).cast(IntegerType()),
        ),
    )

    # fertility_estimate: tokens per English-equivalent byte-chunk
    # fertility = token_count_estimate / (byte_length / 4.0)
    # → For English this gives ~1.0; for Indic ~0.5-0.7 (fewer tokens per byte
    #   because each Indic char is 3 bytes but only ~2 tokens)
    df = df.withColumn(
        "fertility_estimate",
        safe_divide(
            F.col("token_count_estimate"),
            safe_divide(F.col("byte_length"), F.lit(4.0), default=1.0),
            default=1.0,
        ),
    )

    df = df.drop("_lang_divisor")
    return df


def compute_difficulty_score(df):
    """
    Difficulty score covering all four source types.

    Target difficulty ranges (maps to bands via BAND_CENTERS):
      erav4_lang (literacy drills)    → ~0.05-0.20  (B0-B1)
        - simple letter/spelling Q&A lands at ~0.05 (B0)
        - vocabulary/reading items with varied vocab reach ~0.15-0.20 (B1)
      erav4_math (math Q&A)           → ~0.10-0.35  (B0-B2)
        - basic arithmetic at ~0.10-0.15 (B0)
        - word problems with more text/vocab push to ~0.25-0.35 (B1-B2)
      erav4_pattern (pattern Q&A)     → ~0.10-0.35  (B0-B2)
        - simple sequences at ~0.10 (B0)
        - multi-step pattern reasoning at ~0.30-0.35 (B1-B2)
      samvaad (everyday conversation) → ~0.20-0.35  (B1-B2)
        - conversational text has higher vocab diversity + length

    Components:
      vocab_component    (0-1): unique_token_ratio — low for repetitive drills,
                                higher for math word problems / conversation
      length_component   (0-1): document length — short drills are low,
                                longer explanations/conversations are higher
      qa_density_factor  (0-1): low Q&A density → text-heavy → harder
                                pure phonics drills have very high density → lower
      language_component (0-1): Indic orthography adds processing complexity;
                                non-English Indic content scores slightly higher
      conv_bonus         (0-1): free conversation (samvaad) → raised difficulty

    Weights balance so the registry floor/ceiling does final enforcement.
    """
    # vocab_component: 0 = completely repetitive, 1 = all unique words
    df = df.withColumn(
        "vocab_component", F.least(F.col("unique_token_ratio"), F.lit(1.0))
    )

    # length_component: normalise char_length; cap at 1000 chars
    # (math word problems and samvaad can be longer than spelling drills)
    df = df.withColumn(
        "length_component",
        F.least(safe_divide(F.col("char_length"), F.lit(1000.0)), F.lit(1.0)),
    )

    # qa_density_factor: low QA density → more free text → harder
    # High QA density (pure drills) contributes low difficulty.
    # Invert: (1 - min(qa_density, 1.0)) * 0.5
    df = df.withColumn(
        "qa_density_factor",
        F.greatest(F.lit(0.0), F.lit(1.0) - F.least(F.col("qa_density"), F.lit(1.0)))
        * 0.5,
    )

    # language_component: Indic orthography → slightly higher
    # en: 0.05, Indic: 0.15
    df = df.withColumn(
        "language_component",
        F.when(F.col("is_indic") == 1, F.lit(0.15)).otherwise(F.lit(0.05)),
    )

    # conversation_bonus: free conversation markers → raised difficulty
    # Caps at 0.25 — even rich samvaad stays within B0-B2 registry ceiling
    df = df.withColumn(
        "conversation_bonus",
        F.least(
            safe_divide(F.col("conv_marker_count"), F.lit(5.0), default=0.0),
            F.lit(0.25),
        ),
    )

    # Weighted sum — registry floor/ceiling does final band enforcement
    df = df.withColumn(
        "difficulty_score",
        F.col("vocab_component") * 0.30  # strongest signal: vocab diversity
        + F.col("length_component") * 0.20  # document length
        + F.col("qa_density_factor") * 0.20  # free text vs drill density
        + F.col("language_component") * 0.10  # script complexity
        + F.col("conversation_bonus") * 0.20,  # conversation lift (samvaad)
    )
    return df


def apply_quality_filters(df):
    """
    Reject genuinely bad records.  Student-gen data is high quality so
    thresholds are relaxed — we mostly want to catch empty/corrupted rows.
    """
    df = df.withColumn(
        "reject_reason",
        F.when(F.col("char_length") < 5, F.lit("too_short"))
        .when(F.col("word_count") < 1, F.lit("no_words"))
        .otherwise(F.lit(None).cast("string")),
    )
    df = df.withColumn("is_rejected", F.col("reject_reason").isNotNull())
    return df


def assign_curriculum_bands_probabilistic(df):
    """
    Same probabilistic banding logic as v7.1.
    Gaussian weights centred at each band's difficulty centroid.
    """

    def gauss(center):
        exponent = -0.5 * ((F.col("difficulty_score") - center) / WIDTH) ** 2
        return F.exp(exponent)

    raw_weights = {b: gauss(BAND_CENTERS[b]) for b in BANDS}

    # normalise
    total = sum(raw_weights.values())

    probs = {}
    for b in BANDS:
        col_name = f"band_p_{b}"
        df = df.withColumn(col_name, raw_weights[b] / total)
        probs[b] = col_name

    # Assign to highest-probability band where probability > EPS
    band_expr = F.lit("B0")  # default fallback
    for b in reversed(BANDS):
        band_expr = F.when(F.col(probs[b]) > EPS, F.lit(b)).otherwise(band_expr)

    df = df.withColumn("band", band_expr)
    return df


def apply_source_band_clamp(df):
    """
    Clamp raw `band` to the source-specific floor/ceiling from SOURCE_REGISTRY.
    The clamped result is written to `assigned_band`.
    The raw `band` column is preserved for transparency.

    Domain override is also applied here: sets `domain` col to the
    source-specific value if present in the registry.
    """
    src_lower = F.lower(F.col("source"))

    floor_expr = F.lit(0)  # default: B0 floor
    ceil_expr = F.lit(2)  # default: B2 ceiling (safe default for unknowns)
    domain_expr = F.col("domain")  # preserve T1 domain by default

    # Build WHEN chain; reversed so first entry in list wins
    for prefix, floor_idx, ceil_idx, domain_override in reversed(SOURCE_REGISTRY):
        matches = src_lower.contains(prefix)
        floor_expr = F.when(matches, F.lit(floor_idx)).otherwise(floor_expr)
        ceil_expr = F.when(matches, F.lit(ceil_idx)).otherwise(ceil_expr)
        if domain_override:
            domain_expr = F.when(matches, F.lit(domain_override)).otherwise(domain_expr)

    df = df.withColumn("_src_floor", floor_expr)
    df = df.withColumn("_src_ceil", ceil_expr)
    df = df.withColumn("_band_idx", band_name_to_idx(F.col("band")))
    df = df.withColumn(
        "_clamped_idx",
        F.greatest(
            F.col("_src_floor"), F.least(F.col("_src_ceil"), F.col("_band_idx"))
        ),
    )
    df = df.withColumn("assigned_band", idx_to_band_name(F.col("_clamped_idx")))
    df = df.withColumn("domain", domain_expr)  # apply domain override
    df = df.drop("_src_floor", "_src_ceil", "_band_idx", "_clamped_idx")
    return df


# =========================================================================
# MAIN PIPELINE
# =========================================================================


def run_pipeline(df):
    df = create_text_sample(df)
    df = compute_basic_stats(df)
    df = compute_qa_signals(df)
    df = detect_content_signals(df)  # drops text
    df = compute_language_aware_token_estimate(df)
    df = compute_difficulty_score(df)
    df = apply_quality_filters(df)
    df = assign_curriculum_bands_probabilistic(df)
    df = apply_source_band_clamp(df)  # clamp + domain override
    return df


def select_output_columns(df):
    """
    Identical output schema to v7.1-EMR-SERVERLESS.
    Student-specific intermediate columns are kept during processing
    but dropped here so downstream T3 jobs see no changes.
    """
    # ── Exact v7.1 schema ────────────────────────────────────────────────
    core_cols = [
        "id",
        "text",
        "source",
        "added",
        "created",
        "metadata",
        "domain",
        "language",
    ]
    band_cols = [
        "band",
        "assigned_band",
        "difficulty_score",
        "band_p_B0",
        "band_p_B1",
        "band_p_B2",
        "band_p_B3",
        "band_p_B4",
        "band_p_B5",
    ]
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
    size_cols = [
        "byte_length",
        "word_count",
        "unique_token_ratio",
        "compression_ratio",
        "token_count_estimate",
        "fertility_estimate",
    ]
    meta_cols = ["is_rejected", "reject_reason"]

    all_output = core_cols + band_cols + score_cols + size_cols + meta_cols

    existing = set(df.columns)
    select = [c for c in all_output if c in existing]
    return df.select(select)


# =========================================================================
# ENTRY POINT
# =========================================================================


def main():
    args = parse_args()
    source = args.SOURCE
    input_base = args.INPUT_BASE.rstrip("/")
    output_base = args.OUTPUT_BASE.rstrip("/")
    input_path = f"{input_base}/{source}/"

    spark = (
        SparkSession.builder.appName(f"T2_StudentCurriculum_{source}_v{VERSION}")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print(f"T2 Student-Generated Curriculum  v{VERSION}")
    print(f"Source : {source}")
    print(f"Input  : {input_path}")
    print(f"Output : {output_base}")
    print("=" * 60)

    # ── Read ─────────────────────────────────────────────────────────────
    df = spark.read.parquet(input_path)

    required_cols = {"id", "text", "source"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    # Add missing optional columns with sensible defaults.
    # domain default is "education" — the SOURCE_REGISTRY will override
    # it to the correct per-source domain (language_literacy / conversation).
    for col_name, default in [
        ("domain", "education"),
        ("language", "en"),
        ("added", ""),
        ("created", ""),
        ("metadata", "{}"),
    ]:
        if col_name not in df.columns:
            df = df.withColumn(col_name, F.lit(default))

    # Filter empty text
    df = df.filter(F.col("text").isNotNull() & (F.length(F.col("text")) > 0))

    # ── Process ──────────────────────────────────────────────────────────
    df = run_pipeline(df)

    # ── Split accepted vs rejected ────────────────────────────────────────
    df_good = df.filter(~F.col("is_rejected"))
    df_rejected = df.filter(F.col("is_rejected"))

    # ── Select output schema ──────────────────────────────────────────────
    df_good = select_output_columns(df_good)
    df_rejected = select_output_columns(df_rejected)

    # ── Write — same path convention as v7.1 ─────────────────────────────
    # v7.1 pattern: output_base/band=<B>/  (partitioned by assigned_band)
    output_bands = f"{output_base}"
    output_rejected = f"{output_base}/rejected/source={source}/"

    (df_good.write.mode("overwrite").partitionBy("assigned_band").parquet(output_bands))

    (df_rejected.write.mode("overwrite").parquet(output_rejected))

    print(f"Output written to: {output_bands}")
    spark.stop()


if __name__ == "__main__":
    main()
