# T2 Curriculum Band Assignment — Curated Datasets Job (v7.1, EMR Serverless)
# Covers: HuggingFace curated instruction/preference/math/code datasets
# Band range: source-clamped per dataset (see source_clamp_ranges in script)
# Methodology: docs/band_assignment_methodology.md | Version history: docs/CHANGELOG.md


import argparse
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# =========================================================================
# VERSION & PATHS
# =========================================================================

VERSION = "1.0-CURATED"
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
# Each entry: (substring_to_match_in_source, floor_idx, ceil_idx, description)
# Matching uses F.lower(source).contains(key)  — so "gsm8k" matches "gsm8k_train".
# Entries are ordered most-specific first; the first match wins.
# floor_idx / ceil_idx are 0-based band indices (B0=0 … B5=5).
#
# Rationale per dataset:
#   samvaad_hi   → everyday Hindi conversation   → B0-B2
#   smoltalk     → general SFT chit-chat         → B1-B3
#   perfectblend → mixed quality SFT             → B2-B4
#   orpo_dpo     → preference/alignment mix      → B2-B4
#   ultrafeedback→ binarised preferences         → B2-B4
#   infinity_pref→ early preference data         → B2-B4
#   lmarena/arena→ arena winning responses       → B2-B4
#   helpsteer    → multi-attr instruction follow → B3-B5
#   nemotron_post→ diverse post-training         → B3-B5
#   megascience  → multi-domain science text     → B3-B5
#   ling_coder   → coding SFT                    → B3-B5
#   gsm8k        → grade-school math + steps     → B3-B4  (cap at B4)
#   nemotron_math→ expert math                   → B4-B5
#   ultradata    → advanced math (L3 only spec)  → B4-B5
#   skywork      → scientific/math reward        → B4-B5
#   hardgen      → hard generation tasks         → B4-B5
#   teichai/     → Claude 4.5 opus reasoning     → B4-B5
#   high_reason  →   (alt name match)            → B4-B5

SOURCE_REGISTRY = [
    # ── MATH ─────────────────────────────────────────────────────────────
    ("nemotron_math", 4, 5, "Nemotron-Math-v2 (expert math)"),
    ("ultradata", 4, 5, "UltraData-Math (advanced math, L3)"),
    ("skywork", 4, 5, "Skywork Reward Preference (sci/math)"),
    # ── HARD REASONING ───────────────────────────────────────────────────
    ("hardgen", 4, 5, "HardGen (hard generation)"),
    ("teichai", 4, 5, "TeichAI Claude 4.5 high-reasoning"),
    ("high_reasoning", 4, 5, "High-reasoning traces"),
    ("claude_4", 4, 5, "Claude 4.x reasoning traces"),
    # ── MATH WITH STEP SOLUTIONS (capped at B4) ──────────────────────────
    ("gsm8k", 3, 4, "GSM8K (grade-school math, capped B4)"),
    # ── CODE ─────────────────────────────────────────────────────────────
    ("ling_coder", 3, 5, "Ling-Coder-SFT (coding instruction)"),
    ("coder_sft", 3, 5, "Generic coding SFT"),
    # ── SCIENCE ──────────────────────────────────────────────────────────
    ("megascience", 3, 5, "MegaScience (multi-domain science)"),
    # ── INSTRUCTION / HELPFULNESS ────────────────────────────────────────
    ("helpsteer", 3, 5, "HelpSteer3 (multi-attribute instruction)"),
    ("nemotron_post", 3, 5, "Nemotron Post-Training (diverse)"),
    # ── GENERAL SFT / PREFERENCE ─────────────────────────────────────────
    ("perfectblend", 2, 4, "open-perfectblend (mixed SFT)"),
    ("orpo_dpo", 2, 4, "ORPO-DPO mix"),
    ("ultrafeedback", 2, 4, "UltraFeedback binarised preferences"),
    ("infinity_prefer", 2, 4, "Infinity-Preference"),
    ("lmarena", 2, 4, "lmarena arena human preference"),
    ("arena_prefer", 2, 4, "Arena preference (alt name)"),
    ("arena_human", 2, 4, "Arena human preference (alt name)"),
    # ── GENERAL CONVERSATION ─────────────────────────────────────────────
    ("smoltalk", 1, 3, "SmolTalk2 (general conversation SFT)"),
    # ── HINDI CONVERSATION ───────────────────────────────────────────────
    ("samvaad", 0, 2, "Samvaad-HI (Hindi everyday conversation)"),
]

# =========================================================================
# KEYWORD LISTS
# =========================================================================

# ── Retained from v7.1 ───────────────────────────────────────────────────

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

# ── NEW: Conversation / instruction structure ────────────────────────────
# Signals that the text is instruction-following or multi-turn conversation.
# Presence lifts toward B2-B3 (instruction requires more than raw prose).
CONV_MARKERS = [
    "human:",
    "assistant:",
    "<|user|>",
    "<|assistant|>",
    "[inst]",
    "[/inst]",
    "### instruction",
    "### response",
    "user:",
    "<human>",
    "<bot>",
    "system:",
    "<|system|>",
    "### input:",
    "### output:",
    "below is an instruction",
]

# ── NEW: Structured step reasoning ──────────────────────────────────────
# Indicates multi-step problem solving.  Strong B3-B4 signal.
STEP_KEYWORDS = [
    "step 1:",
    "step 2:",
    "step 3:",
    "step 1.",
    "step 2.",
    "step one",
    "step two",
    "step three",
    "let me think",
    "<thinking>",
    "</thinking>",
    "let's work through",
    "let me break",
    "let me first",
]

# ── NEW: LaTeX / formal math notation ───────────────────────────────────
# Presence indicates formal mathematics.  Strong B4-B5 signal.
LATEX_KEYWORDS = [
    "\\frac{",
    "\\sum_{",
    "\\int_",
    "\\sqrt{",
    "\\begin{",
    "\\end{align",
    "\\mathbb{",
    "^{2}",
    "^{n}",
    "_{i}",
    "_{n}",
    "\\alpha",
    "\\beta",
    "\\theta",
    "\\lambda",
    "\\mathcal",
    "\\mathbf",
    "\\vec{",
]

# ── NEW: Science / academic vocabulary ──────────────────────────────────
# Signals structured academic / scientific content.  B3-B5 signal.
SCIENCE_KEYWORDS = [
    "hypothesis",
    "methodology",
    "experiment",
    "we propose",
    "we present",
    "abstract:",
    "introduction:",
    "conclusion:",
    "evaluation:",
    "analysis shows",
    "results indicate",
    "literature review",
    "dataset",
    "benchmark",
]

# =========================================================================
# HELPERS
# =========================================================================


def parse_args():
    p = argparse.ArgumentParser(description="T2 Curated Datasets Curriculum Calculator")
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
    """Column expression: band string → 0-5 integer index."""
    return (
        F.when(col == "B0", 0)
        .when(col == "B1", 1)
        .when(col == "B2", 2)
        .when(col == "B3", 3)
        .when(col == "B4", 4)
        .otherwise(5)
    )


def idx_to_band_name(col):
    """Column expression: 0-5 integer index → band string."""
    return (
        F.when(col == 0, "B0")
        .when(col == 1, "B1")
        .when(col == 2, "B2")
        .when(col == 3, "B3")
        .when(col == 4, "B4")
        .otherwise("B5")
    )


# =========================================================================
# ADAPTIVE SAMPLING (same as v7.1)
# =========================================================================


def create_adaptive_sample(df):
    """Create a capped text sample for cheap per-row metric computation."""
    df = df.withColumn("char_length", F.length(F.col("text")))
    df = df.withColumn(
        "sample_size",
        F.when(F.col("char_length") < 1000, F.col("char_length"))
        .when(F.col("char_length") < 10000, F.lit(5000))
        .when(F.col("char_length") < 50000, F.lit(15000))
        .otherwise(F.lit(25000)),
    )
    df = df.withColumn(
        "text_sample", F.expr("substring(text, 1, cast(sample_size as int))")
    )
    df = df.drop("sample_size")
    return df


# =========================================================================
# FAST METRICS  (regex-free, same as v7.1 — no changes)
# =========================================================================


def compute_basic_stats(df):
    df = df.withColumn("byte_length", F.length(F.encode(F.col("text"), "utf-8")))
    df = df.withColumn(
        "word_count", F.greatest(F.size(F.split(F.col("text"), r"\s+")) - 1, F.lit(1))
    )
    df = df.withColumn(
        "line_count", F.greatest(F.size(F.split(F.col("text"), "\n")) - 1, F.lit(1))
    )
    df = df.withColumn(
        "token_count_estimate",
        F.when(
            F.col("source").startswith("sangraha_"),
            (F.col("word_count") * 1.8).cast("int"),
        ).otherwise((F.col("word_count") * 1.3).cast("int")),
    )
    df = df.withColumn(
        "fertility_estimate",
        safe_divide(F.col("char_length"), F.col("token_count_estimate"), 1.0),
    )
    return df


def compute_character_stats(df):
    text_col = F.col("text_sample")
    char_len = F.length(text_col)

    df = df.withColumn(
        "punct_count", char_len - F.length(F.translate(text_col, ".,;:()[]{}!?", ""))
    )
    df = df.withColumn("punct_ratio", safe_divide(F.col("punct_count"), char_len))

    df = df.withColumn(
        "digit_count", char_len - F.length(F.translate(text_col, "0123456789", ""))
    )
    df = df.withColumn("digit_ratio", safe_divide(F.col("digit_count"), char_len))

    df = df.withColumn(
        "special_count", char_len - F.length(F.translate(text_col, "{}=&|<>", ""))
    )
    df = df.withColumn("special_ratio", safe_divide(F.col("special_count"), char_len))

    df = df.withColumn(
        "upper_count",
        F.expr(
            "length(text_sample) - length(regexp_replace(text_sample, '[A-Z]', ''))"
        ),
    )
    df = df.withColumn("upper_ratio", safe_divide(F.col("upper_count"), char_len))

    df = df.drop("punct_count", "digit_count", "special_count", "upper_count")
    return df


def compute_word_stats(df):
    df = df.withColumn("words", F.split(F.col("text_sample"), r"\s+"))
    df = df.withColumn("unique_words", F.size(F.array_distinct(F.col("words"))))
    df = df.withColumn(
        "unique_token_ratio", safe_divide(F.col("unique_words"), F.col("word_count"))
    )

    df = df.withColumn("all_words_concat", F.concat_ws("", F.col("words")))
    df = df.withColumn("total_word_chars", F.length(F.col("all_words_concat")))
    df = df.withColumn(
        "avg_word_length", safe_divide(F.col("total_word_chars"), F.col("word_count"))
    )
    df = df.drop("all_words_concat")

    df = df.withColumn(
        "compression_ratio", safe_divide(F.col("byte_length"), F.col("char_length"))
    )
    df = df.drop("words", "unique_words", "total_word_chars")
    return df


# =========================================================================
# KEYWORD SCORES  (expanded with new keyword sets)
# =========================================================================


def compute_keyword_scores(df):
    """
    Compute hit counts for all keyword lists.
    Uses simple string.contains() — no regex.
    Drops text_sample when done (major memory saving).
    """
    text_lower = F.lower(F.col("text_sample"))

    def hit_sum(kw_list):
        return sum(F.when(text_lower.contains(kw), 1).otherwise(0) for kw in kw_list)

    # ── Retained from v7.1 ───────────────────────────────────────────────
    df = df.withColumn("code_keyword_count", hit_sum(CODE_KEYWORDS))
    df = df.withColumn("math_keyword_count", hit_sum(MATH_KEYWORDS))
    df = df.withColumn("reasoning_keyword_count", hit_sum(REASONING_KEYWORDS))
    df = df.withColumn("agentic_keyword_count", hit_sum(AGENTIC_KEYWORDS))
    df = df.withColumn("cot_keyword_count", hit_sum(COT_KEYWORDS))

    # ── New: curated-dataset signals ─────────────────────────────────────
    df = df.withColumn("conv_marker_count", hit_sum(CONV_MARKERS))  # instruction format
    df = df.withColumn("step_keyword_count", hit_sum(STEP_KEYWORDS))  # structured steps
    df = df.withColumn("latex_keyword_count", hit_sum(LATEX_KEYWORDS))  # formal math
    df = df.withColumn("science_kw_count", hit_sum(SCIENCE_KEYWORDS))  # academic vocab

    df = df.drop("text_sample")
    return df


# =========================================================================
# COMPOSITE SCORES
# =========================================================================


def compute_composite_scores(df):
    """
    Map keyword counts to interpretable integer scores.
    New scores added on top of v7.1 without modifying existing ones.
    """
    # ── Retained from v7.1 ───────────────────────────────────────────────
    df = df.withColumn(
        "code_score",
        (
            F.col("special_ratio") * 30
            + F.col("digit_ratio") * 20
            + F.col("code_keyword_count") * 5
            + F.when(F.col("avg_word_length") > 8, 5).otherwise(0)
        ).cast("int"),
    )
    df = df.withColumn(
        "math_score",
        (
            F.col("math_keyword_count") * 6
            + F.col("digit_ratio") * 15
            + F.col("special_ratio") * 8
        ).cast("int"),
    )
    df = df.withColumn(
        "reasoning_score",
        (F.col("reasoning_keyword_count") * 6 + F.col("upper_ratio") * 8).cast("int"),
    )
    df = df.withColumn(
        "agentic_score", (F.col("agentic_keyword_count") * 6).cast("int")
    )
    df = df.withColumn(
        "cot_score",
        (F.col("cot_keyword_count") * 5 + F.col("reasoning_keyword_count") * 2).cast(
            "int"
        ),
    )

    # Boolean flags (v7.1)
    df = df.withColumn("has_code", F.col("code_score") >= 10)
    df = df.withColumn("has_math", F.col("math_score") >= 8)
    df = df.withColumn("has_reasoning", F.col("reasoning_score") >= 6)
    df = df.withColumn("has_agentic", F.col("agentic_score") >= 8)
    df = df.withColumn("has_cot", F.col("cot_score") >= 10)

    # ── NEW: curated dataset scores ───────────────────────────────────────
    # conv_score: instruction/conversation format presence → lifts to B2-B3
    df = df.withColumn("conv_score", (F.col("conv_marker_count") * 4).cast("int"))

    # step_score: structured step-by-step reasoning → lifts to B3-B4
    df = df.withColumn(
        "step_score",
        (F.col("step_keyword_count") * 5 + F.col("cot_keyword_count") * 2).cast("int"),
    )

    # latex_score: LaTeX / formal math notation → lifts to B4-B5
    df = df.withColumn(
        "latex_score",
        (F.col("latex_keyword_count") * 6 + F.col("math_keyword_count") * 2).cast(
            "int"
        ),
    )

    # science_score: academic/science vocabulary → lifts to B3-B5
    df = df.withColumn("science_score", (F.col("science_kw_count") * 4).cast("int"))

    # Drop raw keyword counts
    df = df.drop(
        "code_keyword_count",
        "math_keyword_count",
        "reasoning_keyword_count",
        "agentic_keyword_count",
        "cot_keyword_count",
        "conv_marker_count",
        "step_keyword_count",
        "latex_keyword_count",
        "science_kw_count",
    )
    return df


# =========================================================================
# DIFFICULTY SCORE
# =========================================================================


def compute_difficulty_score(df):
    """
    Continuous difficulty in [0, 1] mapping to band centers
    B0=0.05 … B5=0.90.

    New datasets are curated and typically denser with signals, so the
    specialty component now incorporates latex_score and science_score.
    """
    df = df.withColumn(
        "vocab_component", F.least(F.col("unique_token_ratio") * 2.5, F.lit(1.0))
    )
    df = df.withColumn(
        "length_component", F.least((F.col("avg_word_length") - 4) / 6, F.lit(1.0))
    )
    df = df.withColumn(
        "structure_component", F.least(F.col("punct_ratio") * 3, F.lit(1.0))
    )
    # Expanded specialty: includes latex + science alongside code/math/reasoning
    df = df.withColumn(
        "specialty_component",
        F.least(
            (
                F.col("code_score")
                + F.col("math_score")
                + F.col("reasoning_score")
                + F.col("latex_score")
                + F.col("science_score")
            )
            / 80.0,
            F.lit(1.0),
        ),
    )
    df = df.withColumn(
        "difficulty_score",
        F.col("vocab_component") * 0.20
        + F.col("length_component") * 0.20
        + F.col("structure_component") * 0.20
        + F.col("specialty_component")
        * 0.40,  # specialty weighted higher for curated data
    )
    df = df.drop(
        "vocab_component",
        "length_component",
        "structure_component",
        "specialty_component",
    )
    return df


# =========================================================================
# NOISE METRICS + QUALITY FILTERS  (identical to v7.1)
# =========================================================================


def compute_noise_metrics(df):
    text_len = F.length(F.col("text"))
    df = df.withColumn(
        "whitespace_count",
        text_len - F.length(F.translate(F.col("text"), " \t\n\r", "")),
    )
    df = df.withColumn(
        "whitespace_ratio", safe_divide(F.col("whitespace_count"), F.col("char_length"))
    )
    df = df.withColumn("url_count", F.expr("regexp_count(text, 'https?://')"))
    df = df.withColumn(
        "url_ratio", safe_divide(F.col("url_count"), F.col("word_count"))
    )

    boilerplate_kws = [
        "cookie policy",
        "privacy policy",
        "terms of service",
        "all rights reserved",
        "subscribe to",
        "sign up",
    ]
    bp_hits = sum(
        F.when(F.lower(F.col("text")).contains(k), 1).otherwise(0)
        for k in boilerplate_kws
    )
    df = df.withColumn("boilerplate_count", bp_hits)
    df = df.withColumn(
        "boilerplate_ratio",
        safe_divide(F.col("boilerplate_count"), F.lit(len(boilerplate_kws))),
    )

    thread_kws = [">>", "replied to", "in response to", "re:"]
    th_hits = sum(F.when(F.col("text").contains(k), 1).otherwise(0) for k in thread_kws)
    df = df.withColumn("thread_marker_count", th_hits)

    df = df.drop("text", "whitespace_count")
    return df


def apply_quality_filters(df):
    df = df.withColumn("is_rejected", F.lit(False))
    df = df.withColumn("rejection_reason", F.lit(""))
    df = df.withColumn("rejection_level", F.lit(None).cast(IntegerType()))

    rules = [
        # (condition_col_expr, rejection_reason, stage)
        (F.col("byte_length") < 50, "too_short_bytes", 1),
        (F.col("char_length") < 20, "too_short_chars", 1),
        (F.col("token_count_estimate") < 10, "too_short_tokens", 1),
        (
            (F.col("unique_token_ratio") < 0.01) & (F.col("word_count") > 200),
            "repetitive_template",
            2,
        ),
        (F.col("whitespace_ratio") > 0.95, "excessive_whitespace", 2),
        ((F.col("url_ratio") > 0.7) & (F.col("url_count") > 50), "link_spam", 2),
        (F.col("boilerplate_ratio") > 0.50, "boilerplate_spam", 2),
        (
            (F.col("thread_marker_count") > 5) & (F.col("token_count_estimate") < 200),
            "orphaned_thread_fragment",
            2,
        ),
    ]
    for cond, reason, level in rules:
        df = df.withColumn(
            "is_rejected", F.when(cond, True).otherwise(F.col("is_rejected"))
        )
        df = df.withColumn(
            "rejection_reason",
            F.when(cond & (F.col("rejection_reason") == ""), reason).otherwise(
                F.col("rejection_reason")
            ),
        )
        df = df.withColumn(
            "rejection_level",
            F.when(cond & F.col("rejection_level").isNull(), level).otherwise(
                F.col("rejection_level")
            ),
        )
    return df


# =========================================================================
# PROBABILISTIC BANDING  (v7.1 logic + new nudges for curated signals)
# =========================================================================


def assign_curriculum_bands_probabilistic(df):
    """
    Gaussian-weight probabilistic banding around difficulty_score,
    with content-based weight nudges.

    New nudges added for conv, step, latex, science signals on top of
    the existing v7.1 nudges.  The 'band' column is the pre-clamp result;
    'assigned_band' is set later by apply_source_band_clamp().
    """
    # Gaussian weights
    for b in BANDS:
        center = BAND_CENTERS[b]
        df = df.withColumn(
            f"_w_{b}",
            F.greatest(
                F.lit(0.0),
                F.lit(1.0)
                - F.abs(F.col("difficulty_score") - F.lit(center)) / F.lit(WIDTH),
            ),
        )

    # ── v7.1 nudges ──────────────────────────────────────────────────────
    df = df.withColumn(
        "_w_B1",
        F.when(
            (F.col("code_score") > 0) & (F.col("code_score") < 15),
            F.col("_w_B1") + 0.05,
        ).otherwise(F.col("_w_B1")),
    )
    df = df.withColumn(
        "_w_B2",
        F.when(
            (F.col("code_score") >= 15) & (F.col("code_score") < 25),
            F.col("_w_B2") + 0.08,
        )
        .when(F.col("reasoning_score") >= 5, F.col("_w_B2") + 0.05)
        .otherwise(F.col("_w_B2")),
    )
    df = df.withColumn(
        "_w_B3",
        F.when(F.col("code_score") >= 25, F.col("_w_B3") + 0.10)
        .when(F.col("reasoning_score") >= 8, F.col("_w_B3") + 0.08)
        .when(F.col("cot_score") >= 10, F.col("_w_B3") + 0.05)
        .otherwise(F.col("_w_B3")),
    )
    df = df.withColumn(
        "_w_B4",
        F.when(F.col("math_score") >= 12, F.col("_w_B4") + 0.15)
        .when(F.col("code_score") >= 40, F.col("_w_B4") + 0.12)
        .when(F.col("reasoning_score") >= 12, F.col("_w_B4") + 0.10)
        .otherwise(F.col("_w_B4")),
    )
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

    # ── NEW nudges: curated dataset signals ───────────────────────────────
    # Conversation format → lifts B2-B3 slightly (instruction content)
    df = df.withColumn(
        "_w_B2",
        F.when(F.col("conv_score") >= 4, F.col("_w_B2") + 0.06).otherwise(
            F.col("_w_B2")
        ),
    )
    df = df.withColumn(
        "_w_B3",
        F.when(F.col("conv_score") >= 8, F.col("_w_B3") + 0.06).otherwise(
            F.col("_w_B3")
        ),
    )

    # Step reasoning → lifts B3-B4
    df = df.withColumn(
        "_w_B3",
        F.when(F.col("step_score") >= 5, F.col("_w_B3") + 0.08).otherwise(
            F.col("_w_B3")
        ),
    )
    df = df.withColumn(
        "_w_B4",
        F.when(F.col("step_score") >= 12, F.col("_w_B4") + 0.08).otherwise(
            F.col("_w_B4")
        ),
    )

    # LaTeX / formal math → lifts B4-B5
    df = df.withColumn(
        "_w_B4",
        F.when(F.col("latex_score") >= 6, F.col("_w_B4") + 0.12).otherwise(
            F.col("_w_B4")
        ),
    )
    df = df.withColumn(
        "_w_B5",
        F.when(F.col("latex_score") >= 14, F.col("_w_B5") + 0.10).otherwise(
            F.col("_w_B5")
        ),
    )

    # Science vocab → lifts B3-B4
    df = df.withColumn(
        "_w_B3",
        F.when(F.col("science_score") >= 8, F.col("_w_B3") + 0.06).otherwise(
            F.col("_w_B3")
        ),
    )

    # Normalize
    total_w = sum(F.col(f"_w_{b}") for b in BANDS)
    df = df.withColumn(
        "_total_weight", F.when(total_w > 0, total_w).otherwise(F.lit(1.0))
    )
    for b in BANDS:
        df = df.withColumn(f"band_p_{b}", F.col(f"_w_{b}") / F.col("_total_weight"))

    # Assign raw band (lowest credible probability wins)
    df = df.withColumn(
        "band",
        F.when(F.col("band_p_B0") >= EPS, "B0")
        .when(F.col("band_p_B1") >= EPS, "B1")
        .when(F.col("band_p_B2") >= EPS, "B2")
        .when(F.col("band_p_B3") >= EPS, "B3")
        .when(F.col("band_p_B4") >= EPS, "B4")
        .otherwise("B5"),
    )

    # assigned_band will be set by source clamp below; initialise to same
    df = df.withColumn("assigned_band", F.col("band"))

    for b in BANDS:
        df = df.drop(f"_w_{b}")
    df = df.drop("_total_weight")
    return df


# =========================================================================
# SOURCE-AWARE BAND CLAMPING  (key innovation for curated datasets)
# =========================================================================


def apply_source_band_clamp(df):
    """
    Override `assigned_band` with a version clamped to the source's
    floor/ceiling from SOURCE_REGISTRY.

    `band` (the raw probabilistic result) is preserved unchanged so the
    full distribution (band_p_B0 … band_p_B5) remains interpretable.

    Matching: case-insensitive substring of the `source` column.
    Most-specific entries in SOURCE_REGISTRY should be listed first.
    """
    print("Applying source-aware band clamping...")

    # Build WHEN chain for floor and ceiling (first match wins)
    src_lower = F.lower(F.col("source"))
    floor_expr = F.lit(0)  # default: no floor
    ceil_expr = F.lit(5)  # default: no ceiling
    for prefix, floor_idx, ceil_idx, _ in reversed(SOURCE_REGISTRY):
        floor_expr = F.when(src_lower.contains(prefix), F.lit(floor_idx)).otherwise(
            floor_expr
        )
        ceil_expr = F.when(src_lower.contains(prefix), F.lit(ceil_idx)).otherwise(
            ceil_expr
        )

    df = df.withColumn("_src_floor", floor_expr)
    df = df.withColumn("_src_ceil", ceil_expr)
    df = df.withColumn("_band_idx", band_name_to_idx(F.col("band")))

    # clamp: max(floor, min(ceil, raw_idx))
    df = df.withColumn(
        "_clamped_idx",
        F.greatest(
            F.col("_src_floor"), F.least(F.col("_src_ceil"), F.col("_band_idx"))
        ),
    )
    df = df.withColumn("assigned_band", idx_to_band_name(F.col("_clamped_idx")))
    df = df.drop("_src_floor", "_src_ceil", "_band_idx", "_clamped_idx")
    return df


# =========================================================================
# OUTPUT PREPARATION  (identical schema to v7.1)
# =========================================================================


def prepare_output_columns(df, include_rejection=False):
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
        # new curated-dataset scores
        "conv_score",
        "step_score",
        "latex_score",
        "science_score",
    ]
    metric_cols = [
        "byte_length",
        "word_count",
        "unique_token_ratio",
        "compression_ratio",
        "token_count_estimate",
        "fertility_estimate",
    ]
    rejection_cols = ["is_rejected", "rejection_reason", "rejection_level"]

    select_cols = core_cols + band_cols + score_cols + metric_cols
    if include_rejection:
        select_cols += rejection_cols

    existing = [c for c in select_cols if c in df.columns]
    return df.select(*existing)


# =========================================================================
# MAIN
# =========================================================================


def main():
    args = parse_args()
    source = args.SOURCE
    input_base = args.INPUT_BASE
    output_base = args.OUTPUT_BASE

    input_path = f"{input_base}/source={source}"
    output_bands = f"{output_base}/source={source}/bands"
    output_rejected = f"{output_base}/source={source}/rejections"

    print("=" * 60)
    print(f"T2 Curated Datasets Curriculum Calculator v{VERSION}")
    print("=" * 60)
    print(f"Source  : {source}")
    print(f"Input   : {input_path}")
    print(f"Output  : {output_base}")
    print("=" * 60)

    # Show registry entry for this source (informational)
    src_lower = source.lower()
    matched = next(
        ((pf, fl, cl, desc) for pf, fl, cl, desc in SOURCE_REGISTRY if pf in src_lower),
        None,
    )
    if matched:
        pf, fl, cl, desc = matched
        print(f"Registry: [{BANDS[fl]}-{BANDS[cl]}]  {desc}")
    else:
        print("Registry: no match → default [B0-B5] (full range)")
    print("=" * 60)

    spark = SparkSession.builder.appName(f"T2_Curated_{source}").getOrCreate()

    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.files.maxPartitionBytes", "268435456")
    spark.conf.set("spark.default.parallelism", "200")

    print(f"Spark {spark.version}")

    # ── Read ──────────────────────────────────────────────────────────────
    print("Reading input…")
    df = spark.read.parquet(input_path)

    df = df.withColumn("uuid", F.expr("uuid()"))
    df = df.withColumn("file_path", F.input_file_name())
    df = df.withColumn(
        "file_path", F.regexp_replace(F.col("file_path"), f"{input_base}/", "")
    )

    # ── Pipeline ──────────────────────────────────────────────────────────
    start = datetime.now()
    print("Running pipeline…")

    df = create_adaptive_sample(df)
    df = compute_basic_stats(df)  # keeps full text for noise metrics
    df = compute_noise_metrics(df)  # drops text column
    df = compute_character_stats(df)
    df = compute_word_stats(df)
    df = compute_keyword_scores(df)  # drops text_sample
    df = compute_composite_scores(df)
    df = compute_difficulty_score(df)
    df = apply_quality_filters(df)

    # Drop noise intermediates no longer needed
    df = df.drop(
        "url_count",
        "url_ratio",
        "whitespace_ratio",
        "boilerplate_count",
        "boilerplate_ratio",
        "thread_marker_count",
    )

    df = assign_curriculum_bands_probabilistic(df)
    df = apply_source_band_clamp(df)  # ← source-aware floor/ceiling

    # ── Split ─────────────────────────────────────────────────────────────
    rejected = df.filter(F.col("is_rejected"))
    accepted = df.filter(~F.col("is_rejected"))

    rejected_out = prepare_output_columns(rejected, include_rejection=True)
    accepted_out = prepare_output_columns(accepted, include_rejection=False)

    # ── Write ─────────────────────────────────────────────────────────────
    print(f"Writing rejections → {output_rejected}")
    rejected_out.write.mode("overwrite").option("compression", "zstd").parquet(
        output_rejected
    )

    print(f"Writing bands → {output_bands}")
    accepted_out.write.mode("overwrite").partitionBy("band").option(
        "compression", "zstd"
    ).parquet(output_bands)

    duration = (datetime.now() - start).total_seconds()
    print("=" * 60)
    print(f"Done in {duration:.1f}s")
    print(f"  Bands     : {output_bands}/band=<B0-B5>/")
    print(f"  Rejections: {output_rejected}/")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
