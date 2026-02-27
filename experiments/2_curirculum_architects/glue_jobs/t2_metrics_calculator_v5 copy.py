"""
T2 Metrics Calculator V5.0 - Robust Modality Scoring + Probabilistic Banding (Optimized)
========================================================================================
Purpose: Compute quality metrics, assign probabilistic curriculum bands, and reject extreme noise.
Strategy: 2-stage noise rejection → Robust V5 modality scoring → Probabilistic banding.
Optimization: Aggressive compute reduction (removed generic metrics), pure Spark SQL, AQE.

CHANGELOG V5.0 (2026-02-08):
- MAJOR RESTRUCTURE:
  * Removed Stage 3 Rejections entirely (filtering is now Stage 1 & 2 only).
  * Simplified Output Structure: `bands/` (passed) and `rejections/` (failed).
  * Simplified Partitioning: Only by `band` or `rejection_level` (removed domain partitions).
  * Added `text` column to all outputs for downstream debugging/analysis.

- METRIC REMOVALS (Compute Optimization):
  * Removed: risky_tld_count, sentence_boundary_coherence, html_tag_density, non_printable_ratio.
  * Removed: punctuation_density, dependency_depth_estimate, num_numeric_tokens.
  * Removed: citation_count, step_indicator_count, ellipsis_count, dialogue_turn_count.
  * Reason: High compute cost or low signal-to-noise ratio for curriculum sorting.

- NEW V5 PATTERNS:
  * Implemented `compute_robust_modality_scores`: Multi-signal detection for Agentic, CoT, Reasoning, Code, Math.
  * Implemented `assign_curriculum_band_probabilistic`: Uses robust scores for safer band nudging.
  * Difficulty Score: Updated to use V5 signals, removed dependencies on deleted metrics.

- OPTIMIZATIONS:
  * Single-pass regex extraction for all modality signals.
  * Removed tiktoken approximations (using word_count proxies).
  * Broadcast variable for high-value keywords (retained from V4).
  * AQE enabled for automatic skew handling.

Usage (Glue Flex):
    aws glue start-job-run --job-name T2_metrics_calculator_v5 \
        --worker-type G.2X --number-of-workers 20 --execution-class FLEX \
        --arguments '{"--INPUT_BASE":"s3://...", "--OUTPUT_BASE":"s3://..."}'
"""

import sys
import re
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
from datetime import datetime

# =========================================================================
# CONFIGURATION & CONSTANTS
# =========================================================================

VERSION = "5.0"
INPUT_BASE_DEFAULT = "s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"
OUTPUT_BASE_DEFAULT = "s3://t1-dataacquisition-datasets/processed_dataset/curriculum_data"
INTERMEDIATE_BASE_DEFAULT = "s3://t1-dataacquisition-datasets/processed_dataset/checkpoints"

# Band ordering
BANDS = ["B0", "B1", "B2", "B3", "B4", "B5"]
BAND_CENTERS = {"B0": 0.05, "B1": 0.20, "B2": 0.35, "B3": 0.55, "B4": 0.75, "B5": 0.90}
WIDTH = 0.20
EPS = 0.15
MAX_SINGLE_BAND_MASS = 0.85

# -------------------------------------------------------------------------
# V5 PATTERN DEFINITIONS (Regex)
# -------------------------------------------------------------------------

# Pattern 1: AGENTIC CONTENT
AGENTIC_STRUCTURAL_PATTERN = r'''(?x)
    (?:(?:Step\s+\d+|Task\s+\d+):\s*(?:Call|Execute|Run|Use|Invoke)\s+\w+)|
    (?:(?:tool|function|api)_(?:use|call|invoke)\s*\()|
    (?:\[(?:PLAN|ACTION|TOOL|STEP)\s*\d*\])|
    (?:Thought\s*\d*:\s*.{10,}Action\s*\d*:)
'''
AGENTIC_VOCAB_PATTERN = r'\b(?:execute|invoke|call|dispatch|orchestrate|coordinate|delegate|subgoal|subtask|decompose|breakdown|workflow|pipeline)\b'

# Pattern 2: CHAIN-OF-THOUGHT
COT_EXPLICIT_PATTERN = r'''(?x)
    (?:Let's\s+think\s+(?:step[- ]by[- ]step|through\s+this|carefully|systematically))|
    (?:\[(?:REASONING|THINKING|ANALYSIS)\])|
    (?:I\s+(?:need\s+to|should|must)\s+(?:think\s+about|consider|analyze))
'''
COT_REASONING_CONNECTIVES = r'\b(?:therefore|thus|hence|because|since|this\s+means|which\s+implies)\b'
EDUCATIONAL_MARKER_PATTERN = r"(?i)###\s*(?:Explanation|Question|Answer|Topic|Metadata|Prerequisites):"

# Pattern 3: FORMAL REASONING
FORMAL_REASONING_PATTERN = r'''(?x)
    (?:Proof:|Theorem:|Lemma:|Corollary:)|
    (?:Q\.E\.D\.|∎|□)|
    (?:(?:By|Using)\s+(?:induction|contradiction|construction))|
    (?:It\s+follows\s+that|We\s+can\s+deduce|This\s+implies)
'''
# Simplified Math Symbols (removed rare unicode to save regex compile time if needed, but keeping standard set)
MATH_SYMBOLS_PATTERN = r'[∀∃∈∉⊂⊆∪∩∅⇒⇔∧∨¬→↔⊢⊨≡≠≤≥±∓∞∑∏∫√]'

# Pattern 4: TABLE STRUCTURES
MARKDOWN_TABLE_SEPARATOR = r'\|[-:]+\|[-:]+\|'
TABLE_ROW_PATTERN = r'\|(?:\s*\w+\s*\|){2,}'
TABLE_HEADER_KEYWORDS = r'(?i)\b(?:name|id|value|type|date|count|total|column|field|description)\b'

# Pattern 5: CODE WITH COMMENTS
CODE_COMMENT_SYNTAX = r'''(?x)
    (?:^[ \t]*(?://|#)\s+\w+)|
    (?:/\*.*?\*/)|
    (?:(?:"""|\'\'\').{30,}?(?:"""|\'\'\''))
'''
CODE_KEYWORDS_PATTERN = r'\b(?:function|def|class|return|import|from|const|let|var|if|else|for|while|try|catch|public|private)\b'

# Pattern 6: Q&A
QA_PAIR_PATTERN = r'''(?x)
    (?:Q(?:uestion)?|Query)\s*\d*[:.]?\s*.{20,}?\?\s+A(?:nswer)?[:.]?\s*.{30,}|
    (?:^|\n)(?:Q|Question):\s*.{20,}?\?\s+(?:A|Answer):\s*.{30,}
'''
QA_ANSWER_MARKERS = r'\?\s+(?:The\s+answer\s+is|It\s+is\s+because|Yes|No|In\s+summary)'

# Pattern 7: CODE (Multi-language)
PYTHON_SYNTAX = r'(?:^|\n)(?:def|class|import|from\s+\w+\s+import)\s+\w+'
JAVASCRIPT_SYNTAX = r'(?:function\s+\w+\s*\(|const|let|var)\s+\w+\s*=|=>'
JAVA_CPP_SYNTAX = r'(?:public|private|protected|#include|int\s+main)'
CODE_STRUCTURE = r'^\s{2,}\S'
CODE_SYNTAX_CHARS = r'[;{}()\[\]]'
CAMEL_SNAKE_CASE = r'\b[a-z]+[A-Z]\w+\b|\b[a-z]+_[a-z_]+\b'

# Pattern 8: MATH CONTENT
# OPTIMIZATION: Removed full unicode range for regex performance, focusing on high-signal chars
EQUATION_PATTERN = r'[a-z]\s*[+\-*/=]\s*[a-z0-9]|[a-z]\^[0-9]|\([a-z0-9\s+\-*/]+\)\s*='
LATEX_COMMANDS = r'\\(?:frac|sum|prod|int|lim|infty|sqrt|cdot|times|begin\{equation)'
MATH_TERMINOLOGY = r'\b(?:theorem|lemma|proof|equation|derivative|integral|matrix|vector|polynomial)\b'

# Other Patterns (Retained from V2 for Stage 2)
BOILERPLATE_PATTERN = r"(?i)(cookie policy|privacy policy|terms of service|all rights reserved|© copyright|click here|subscribe to|sign up|newsletter|unsubscribe|contact us|about us|follow us on|accept cookies|manage preferences)"
THREAD_MARKER_PATTERN = r"(>>|replied to:|in response to|re:|replying to|quote from|responding to)"
# Removed RISKY_TLD_PATTERN
URL_PATTERN = r"https?://[^\s]+"
CODE_FENCE_PATTERN = r"```|~~~"
HEADING_PATTERN = r"^#+\s+|^[A-Z][^\n]{5,50}$"

# Metadata Extraction Patterns
METADATA_DIFFICULTY_PATTERN = r"(?i)Difficulty:\s*(\w+)"
METADATA_GRADE_PATTERN = r"(?i)Grade:\s*(\d+)"
METADATA_LEVEL_PATTERN = r"(?i)Student Level:\s*(\w+)"

# High-value keywords for cheap complexity detection (Broadcast)
HIGH_VALUE_KEYWORDS = [
    "hypothesis", "methodology", "empirical", "theorem", "lemma", "corollary",
    "ontology", "epistemology", "phenomenology", "hermeneutics", "dialectic",
    "paradigm", "heuristic", "algorithm", "optimization", "convergence",
    "heterogeneous", "homogeneous", "isotropic", "anisotropic", "stochastic",
    "deterministic", "asymptotic", "parametric", "nonparametric", "multivariate",
    "eigenvalue", "eigenvector", "gradient", "jacobian", "hessian",
    "pathogenesis", "etiology", "pharmacokinetics", "pharmacodynamics", "metabolism",
    "carcinogenesis", "immunology", "cytology", "histology", "morphology",
    "jurisprudence", "adjudication", "litigation", "jurisdiction", "precedent",
    "appellant", "respondent", "plaintiff", "defendant", "indictment",
    "polynomial", "exponential", "logarithmic", "trigonometric", "hyperbolic",
    "differential", "integral", "derivative", "convolution", "fourier",
    "bayesian", "frequentist", "likelihood", "posterior", "prior",
    "polymorphism", "encapsulation", "inheritance", "abstraction", "concurrency",
    "parallelism", "distributed", "synchronization", "mutex", "semaphore",
    "recursion", "memoization", "backtracking", "hashing", "traversal",
    "syllogism", "tautology", "contradiction", "axiom", "inference",
    "deduction", "induction", "abduction", "fallacy", "proposition",
    "elasticity", "equilibrium", "arbitrage", "volatility", "derivative",
    "amortization", "depreciation", "valuation", "liquidity", "solvency",
    "thermodynamics", "kinetics", "dynamics", "statics", "mechanics",
    "electromagnetic", "semiconductor", "transistor", "amplifier", "oscillator",
    "stoichiometry", "titration", "catalysis", "synthesis", "hydrolysis",
    "oxidation", "reduction", "equilibrium", "entropy", "enthalpy",
    "quantum", "relativity", "spacetime", "superposition", "entanglement",
    "hamiltonian", "lagrangian", "schrodinger", "heisenberg", "maxwell",
    "dichotomy", "juxtaposition", "ubiquitous", "ephemeral", "perpetual",
    "ambiguous", "arbitrary", "intrinsic", "extrinsic", "implicit",
    "explicit", "analogous", "congruent", "disparate", "heterogeneous",
    "therefore", "consequently", "nevertheless", "notwithstanding", "furthermore",
    "moreover", "conversely", "alternatively", "analogously", "presumably"
]


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def get_glue_args():
    """Parse Glue job arguments."""
    args = getResolvedOptions(sys.argv, [])
    
    optional_args = {}
    if '--INPUT_BASE' in sys.argv:
        optional_args['INPUT_BASE'] = getResolvedOptions(sys.argv, ['INPUT_BASE'])['INPUT_BASE']
    else:
        optional_args['INPUT_BASE'] = INPUT_BASE_DEFAULT
        
    if '--OUTPUT_BASE' in sys.argv:
        optional_args['OUTPUT_BASE'] = getResolvedOptions(sys.argv, ['OUTPUT_BASE'])['OUTPUT_BASE']
    else:
        optional_args['OUTPUT_BASE'] = OUTPUT_BASE_DEFAULT

    if '--INTERMEDIATE_BASE' in sys.argv:
        optional_args['INTERMEDIATE_BASE'] = getResolvedOptions(sys.argv, ['INTERMEDIATE_BASE'])['INTERMEDIATE_BASE']
    else:
        optional_args['INTERMEDIATE_BASE'] = INTERMEDIATE_BASE_DEFAULT

    # REQUIRED: Filter by specific source for incremental processing
    if '--SOURCE' in sys.argv:
        optional_args['SOURCE'] = getResolvedOptions(sys.argv, ['SOURCE'])['SOURCE']
    else:
        print("ERROR: --SOURCE argument is required.")
        sys.exit(1)
    
    # Optional: Manual restart ID
    if '--MANUAL_RESTART' in sys.argv:
        optional_args['MANUAL_RESTART'] = getResolvedOptions(sys.argv, ['MANUAL_RESTART'])['MANUAL_RESTART']
    else:
        optional_args['MANUAL_RESTART'] = None
    
    return args, optional_args

def add_uuid_and_metadata(df):
    """Add unique ID and file path for tracking."""
    df = df.withColumn("uuid", F.expr("uuid()"))
    # remove bucket information and keep only the path after bucket
    prefix_to_remove = INPUT_BASE_DEFAULT
    df = df.withColumn("file_path", F.input_file_name())
    df = df.withColumn("file_path", F.regexp_replace(F.col("file_path"), prefix_to_remove, ""))
    return df

def checkpoint_exists(spark, s3_path):
    """Check for custom _CHECKPOINT_COMPLETE flag."""
    checkpoint_flag = f"{s3_path}/_CHECKPOINT_COMPLETE"
    try:
        spark.read.text(checkpoint_flag)
        return True
    except Exception:
        return False

def write_checkpoint_flag(spark, s3_path):
    """Write custom completion flag."""
    checkpoint_flag = f"{s3_path}/_CHECKPOINT_COMPLETE"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"Checkpoint completed successfully at {timestamp}"
    flag_df = spark.createDataFrame([(message,)], ["status"])
    flag_df.coalesce(1).write.mode("overwrite").text(checkpoint_flag)

def safe_divide(numerator, denominator, default=0.0):
    """Safe division preventing divide-by-zero."""
    from pyspark.sql import Column
    if not isinstance(numerator, Column): numerator = F.lit(numerator)
    if not isinstance(denominator, Column): denominator = F.lit(denominator)
    if not isinstance(default, Column): default = F.lit(default)
    return F.when(denominator > 0, numerator / denominator).otherwise(default)


# =========================================================================
# STAGE 1 (PRIORITY 1): PHYSICAL & BASIC CORRUPTION
# =========================================================================

def compute_stage1_metrics(df):
    """
    Stage 1: Physical properties.
    REMOVED: non_printable_ratio (compute optimization + Indic support)
    """
    print("  Computing Stage 1 metrics...")
    df = df.withColumn("byte_length", F.length(F.encode(F.col("text"), "utf-8")))
    df = df.withColumn("char_length", F.length(F.col("text")))
    df = df.withColumn("word_count", F.size(F.split(F.col("text"), r"\s+")) - 1)
    df = df.withColumn("line_count", F.size(F.split(F.col("text"), "\n")))
    
    # Token count estimate (approximate: word_count * 1.3 for English)
    # This avoids tiktoken dependency while providing reasonable proxy
    df = df.withColumn("token_count_estimate", (F.col("word_count") * 1.3).cast("int"))
    df = df.withColumn("fertility_estimate", safe_divide(F.col("char_length"), F.col("token_count_estimate"), 1.0))
    return df

def apply_stage1_rejection(df):
    """
    Apply Priority 1 rejection rules.
    REMOVED: corruption (non_printable_ratio) rule.
    """
    print("  Applying Stage 1 rejection rules...")
    
    df = df.withColumn("is_rejected", F.lit(False))
    df = df.withColumn("rejection_reason", F.lit(""))
    df = df.withColumn("rejection_level", F.lit(None).cast(IntegerType()))
    
    # Rule 1: Byte length < 50
    cond_byte = F.col("byte_length") < 50
    df = df.withColumn("is_rejected", F.when(cond_byte, True).otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason", F.when(cond_byte, "too_short_bytes").otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_level", F.when(cond_byte, 1).otherwise(F.col("rejection_level")))
    
    # Rule 2: Char length < 20
    cond_char = F.col("char_length") < 20
    df = df.withColumn("is_rejected", F.when(cond_char, True).otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason", F.when(cond_char, "too_short_chars").otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_level", F.when(cond_char, 1).otherwise(F.col("rejection_level")))
    
    # Rule 3: Token count < 10
    cond_token = F.col("token_count_estimate") < 10
    df = df.withColumn("is_rejected", F.when(cond_token, True).otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason", F.when(cond_token, "too_short_tokens").otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_level", F.when(cond_token, 1).otherwise(F.col("rejection_level")))
    
    return df.filter(F.col("is_rejected")), df.filter(~F.col("is_rejected"))


# =========================================================================
# STAGE 2 (PRIORITY 2): NOISE & SPAM
# =========================================================================

def compute_stage2_metrics(df):
    """
    Stage 2: Noise detection.
    OPTIMIZATION: Batch regex.
    REMOVED: risky_tld, html_tag_density, sentence_boundary_coherence.
    KEPT: thread_fragment_indicator (as requested).
    """
    print("  Computing Stage 2 metrics...")
    
    # Batch 1: Array ops (Optimized: Single pass for unique tokens)
    # Combined token operations to avoid intermediate column 'tokens_list'
    df = df.withColumn("unique_token_ratio", 
                       safe_divide(F.size(F.array_distinct(F.split(F.lower(F.col("text")), r"\s+"))), F.col("word_count")))
    
    # Compression proxy
    df = df.withColumn("compression_ratio", safe_divide(F.col("byte_length"), F.col("char_length")))
    
    # Batch 2: Char ops
    text_len = F.length(F.col("text"))
    df = df.withColumn("whitespace_count", text_len - F.length(F.regexp_replace(F.col("text"), r"\S", "")))
    df = df.withColumn("whitespace_ratio", safe_divide(F.col("whitespace_count"), F.col("char_length")))
    
    df = df.withColumn("alpha_count", F.length(F.regexp_replace(F.col("text"), r"[^a-zA-Z]", "")))
    df = df.withColumn("upper_count", F.length(F.regexp_replace(F.col("text"), r"[^A-Z]", "")))
    df = df.withColumn("capitalization_ratio", safe_divide(F.col("upper_count"), F.col("alpha_count")))
    
    # Batch 3: Single Pass Regex
    # Capturing: URL, Boilerplate, Thread Markers
    df = df.withColumn("_url_matches", F.expr(f"regexp_extract_all(text, '{URL_PATTERN}', 0)"))
    df = df.withColumn("url_count", F.size(F.col("_url_matches")))
    df = df.withColumn("url_ratio", safe_divide(F.col("url_count"), F.col("word_count")))
    df = df.drop("_url_matches")
    
    df = df.withColumn("_boilerplate_matches", F.expr(f"regexp_extract_all(lower(text), '{BOILERPLATE_PATTERN}', 0)"))
    df = df.withColumn("boilerplate_count", F.size(F.col("_boilerplate_matches")))
    df = df.withColumn("boilerplate_ratio", safe_divide(F.col("boilerplate_count"), F.col("word_count")))
    df = df.drop("_boilerplate_matches")
    
    df = df.withColumn("_thread_matches", F.expr(f"regexp_extract_all(text, '{THREAD_MARKER_PATTERN}', 0)"))
    df = df.withColumn("thread_marker_count", F.size(F.col("_thread_matches")))
    df = df.withColumn("thread_fragment_indicator", F.col("thread_marker_count")) # Kept as requested
    df = df.drop("_thread_matches")
    
    # Sentence estimate (kept for word/sentence ratio if needed for simple heuristic, but removed coherence check)
    df = df.withColumn("sentence_count_estimate", F.size(F.split(F.col("text"), r"[.!?]+\s+")) - 1)
    df = df.withColumn("sentence_count_estimate", F.when(F.col("sentence_count_estimate") < 1, 1).otherwise(F.col("sentence_count_estimate")))

    # Rare word estimate (kept for difficulty score)
    df = df.withColumn("rare_word_ratio_estimate",
                       F.when(
                            F.col("unique_token_ratio") > 0.4,
                            F.lit(1.0) - (F.lit(2.0) * F.col("unique_token_ratio"))
                        )
                       .otherwise(F.lit(0.1)))
    
    return df

def apply_stage2_rejection(df):
    """
    Apply Priority 2 rejection rules.
    REMOVED: risky_tld, html_tag, sentence_boundary rules.
    """
    print("  Applying Stage 2 rejection rules...")
    
    df = df.withColumn("is_rejected", F.lit(False))
    df = df.withColumn("rejection_reason", F.lit(""))
    df = df.withColumn("rejection_level", F.lit(None).cast(IntegerType()))
    
    # Rule 1: Repetitive template
    cond_rep = (F.col("unique_token_ratio") < 0.01) & (F.col("word_count") > 200)
    df = df.withColumn("is_rejected", F.when(cond_rep, True).otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason", F.when(cond_rep, "repetitive_template").otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_level", F.when(cond_rep, 2).otherwise(F.col("rejection_level")))
    
    # # Rule 3: All Caps
    # cond_caps = (F.col("capitalization_ratio") > 0.95) & (F.col("word_count") > 200)
    # df = df.withColumn("is_rejected", F.when(cond_caps, True).otherwise(F.col("is_rejected")))
    # df = df.withColumn("rejection_reason", F.when(cond_caps, "all_caps_spam").otherwise(F.col("rejection_reason")))
    # df = df.withColumn("rejection_level", F.when(cond_caps, 2).otherwise(F.col("rejection_level")))
    
    # Rule 4: Whitespace
    cond_white = F.col("whitespace_ratio") > 0.95
    df = df.withColumn("is_rejected", F.when(cond_white, True).otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason", F.when(cond_white, "excessive_whitespace").otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_level", F.when(cond_white, 2).otherwise(F.col("rejection_level")))
    
    # Rule 5: Link Spam
    cond_link = (F.col("url_ratio") > 0.7) & (F.col("url_count") > 50)
    df = df.withColumn("is_rejected", F.when(cond_link, True).otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason", F.when(cond_link, "link_spam").otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_level", F.when(cond_link, 2).otherwise(F.col("rejection_level")))
    
    # Rule 7: Boilerplate
    cond_boil = F.col("boilerplate_ratio") > 0.50
    df = df.withColumn("is_rejected", F.when(cond_boil, True).otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason", F.when(cond_boil, "boilerplate_spam").otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_level", F.when(cond_boil, 2).otherwise(F.col("rejection_level")))
    
    # Rule 8: Thread Fragment
    cond_thread = (F.col("thread_fragment_indicator") > 5) & (F.col("token_count_estimate") < 200)
    df = df.withColumn("is_rejected", F.when(cond_thread, True).otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason", F.when(cond_thread, "orphaned_thread_fragment").otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_level", F.when(cond_thread, 2).otherwise(F.col("rejection_level")))
    
    return df.filter(F.col("is_rejected")), df.filter(~F.col("is_rejected"))


# =========================================================================
# STAGE 3 (PRIORITY 3): SCORING (Modality + Banding) - NO REJECTION
# =========================================================================

def compute_robust_modality_scores(df):
    """
    V5 Robust Modality Scoring.
    Multi-signal approach for high precision.
    Optimization: Reuse basic text stats where possible. Be efficient with regex.
    """
    print("  Computing V5 Modality Scores (Optimized)...")
    
    # Pre-escape patterns for SQL injection AND Spark SQL string parsing
    # 1. Replace backslash (\) with double backslash (\\) so Spark SQL preserves them for Regex
    # 2. Replace single quote (') with two single quotes ('') to escape SQL string delimiters
    def sql_escape(pattern):
        return pattern.replace("\\", "\\\\").replace("'", "''")

    p_agentic_struct = sql_escape(AGENTIC_STRUCTURAL_PATTERN)
    p_agentic_vocab = sql_escape(AGENTIC_VOCAB_PATTERN)
    p_cot_explicit = sql_escape(COT_EXPLICIT_PATTERN)
    p_cot_connectives = sql_escape(COT_REASONING_CONNECTIVES)
    p_formal_reasoning = sql_escape(FORMAL_REASONING_PATTERN)
    p_table_row = sql_escape(TABLE_ROW_PATTERN)
    p_table_header = sql_escape(TABLE_HEADER_KEYWORDS)
    p_table_separ = sql_escape(MARKDOWN_TABLE_SEPARATOR)
    p_python = sql_escape(PYTHON_SYNTAX)
    p_js = sql_escape(JAVASCRIPT_SYNTAX)
    p_java_cpp = sql_escape(JAVA_CPP_SYNTAX)
    p_code_struct = sql_escape(CODE_STRUCTURE)
    p_code_syntax = sql_escape(CODE_SYNTAX_CHARS)
    p_camel_snake = sql_escape(CAMEL_SNAKE_CASE)
    p_equation = sql_escape(EQUATION_PATTERN)
    p_latex = sql_escape(LATEX_COMMANDS)
    p_math_term = sql_escape(MATH_TERMINOLOGY)

    # ------------------------------------------------------------------------
    # 1. Agentic
    # ------------------------------------------------------------------------
    df = df.withColumn("agentic_score",
        F.when(F.expr(f"regexp_count(text, '{p_agentic_struct}')") >= 2, F.lit(3)).otherwise(F.lit(0)) +
        F.when((F.expr(f"regexp_count(text, '{p_agentic_vocab}')") / F.greatest(F.col("word_count"), F.lit(1))) > 0.006, F.lit(2)).otherwise(F.lit(0)) +
        F.when(F.expr("regexp_count(text, '\\\\b(?:subgoal|subtask|decompose|breakdown|workflow|pipeline)\\\\b')") >= 3, F.lit(2)).otherwise(F.lit(0)) +
        F.when(F.expr("regexp_count(text, '(?:def|function)\\\\s+\\\\w+_(?:tool|call|agent)')") >= 1, F.lit(2)).otherwise(F.lit(0))
    )
    df = df.withColumn("is_agentic", F.when(F.col("agentic_score") >= 5, F.lit(1)).otherwise(F.lit(0)))

    # ------------------------------------------------------------------------
    # 2. Chain of Thought (CoT)
    # ------------------------------------------------------------------------
    df = df.withColumn("cot_score",
        F.when(F.expr(f"regexp_count(text, '{p_cot_explicit}')") >= 1, F.lit(3)).otherwise(F.lit(0)) +
        F.when((F.expr(f"regexp_count(text, '{p_cot_connectives}')") / F.greatest(F.col("word_count"), F.lit(1))) > 0.02, F.lit(2)).otherwise(F.lit(0)) +
        F.when((F.expr("regexp_count(text, '\\\\?\\\\s+.{30,}\\\\.')") >= 2) & (F.expr("regexp_count(text, '(?:So|Therefore|Thus),?\\\\s+')") >= 2), F.lit(2)).otherwise(F.lit(0)) +
        F.when(F.expr(f"regexp_count(text, '{sql_escape(EDUCATIONAL_MARKER_PATTERN)}')") >= 3, F.lit(3)).otherwise(F.lit(0))
    )
    df = df.withColumn("is_cot", F.when(F.col("cot_score") >= 5, F.lit(1)).otherwise(F.lit(0)))

    # ------------------------------------------------------------------------
    # 3. Formal Reasoning
    # ------------------------------------------------------------------------
    df = df.withColumn("reasoning_score",
        F.when(F.expr(f"regexp_count(text, '{p_formal_reasoning}')") >= 2, F.lit(4)).otherwise(F.lit(0)) +
        F.when((F.length(F.regexp_replace(F.col("text"), f"[^{MATH_SYMBOLS_PATTERN[1:-1]}]", "")) / F.greatest(F.col("char_length"), F.lit(100))) > 0.01, F.lit(3)).otherwise(F.lit(0)) +
        F.when(F.expr("regexp_count(text, '(?:If|Suppose|Assume).{10,}?,\\\\s+then')") >= 3, F.lit(2)).otherwise(F.lit(0))
    )
    df = df.withColumn("is_reasoning", F.when(F.col("reasoning_score") >= 6, F.lit(1)).otherwise(F.lit(0)))

    # ------------------------------------------------------------------------
    # 4. Table Structure
    # ------------------------------------------------------------------------
    # Optimized: Use simple line filtering. 'split' by pipe | gives N+1 elements for N pipes.
    # So size >= 3 implies >= 2 pipes. Using -1 limit to keep empty strings.
    df = df.withColumn("table_lines", F.size(F.filter(F.split(F.col("text"), '\n'), lambda line: F.size(F.split(line, r'\|', -1)) >= 3)))
    
    df = df.withColumn("table_score",
        F.when((F.col("table_lines") >= 3) & (F.col("table_lines") / F.greatest(F.col("line_count"), F.lit(1)) >= 0.5), F.lit(3)).otherwise(F.lit(0)) +
        F.when(F.expr(f"regexp_count(text, '{p_table_row}')") >= 3, F.lit(2)).otherwise(F.lit(0)) +
        F.when(F.expr(f"regexp_count(substring(text, 1, 200), '{p_table_header}')") >= 2, F.lit(2)).otherwise(F.lit(0)) +
        F.when(F.expr(f"regexp_count(text, '{p_table_separ}')") >= 1, F.lit(3)).otherwise(F.lit(0))
    )
    df = df.withColumn("is_table", F.when(F.col("table_score") >= 5, F.lit(1)).otherwise(F.lit(0)))
    df = df.drop("table_lines") # cleanup

    # ------------------------------------------------------------------------
    # 5. Code
    # ------------------------------------------------------------------------
    df = df.withColumn("code_score",
        F.when(F.expr(f"regexp_count(text, '{p_python}')") >= 2, F.lit(6)).otherwise(F.lit(0)) +
        F.when(F.expr(f"regexp_count(text, '{p_js}')") >= 2, F.lit(6)).otherwise(F.lit(0)) +
        F.when(F.expr(f"regexp_count(text, '{p_java_cpp}')") >= 2, F.lit(6)).otherwise(F.lit(0)) +
        F.when((F.expr(f"regexp_count(text, '{p_code_struct}')") >= 5) | (F.expr(f"regexp_count(text, '{p_code_syntax}')") >= 10), F.lit(3)).otherwise(F.lit(0)) +
        F.when(F.expr(f"regexp_count(text, '{p_camel_snake}')") >= 5, F.lit(2)).otherwise(F.lit(0))
    )
    df = df.withColumn("is_code", F.when(F.col("code_score") >= 10, F.lit(1)).otherwise(F.lit(0)))

    # ------------------------------------------------------------------------
    # 6. Math
    # ------------------------------------------------------------------------
    # Math symbols count (optimized: removed full unicode range check to generic match in pattern)
    df = df.withColumn("math_score",
        F.when(F.length(F.regexp_replace(F.col("text"), f"[^{MATH_SYMBOLS_PATTERN[1:-1]}]", "")) >= 5, F.lit(4)).otherwise(F.lit(0)) +
        F.when(F.expr(f"regexp_count(text, '{p_equation}')") >= 3, F.lit(4)).otherwise(F.lit(0)) +
        F.when(F.expr(f"regexp_count(text, '{p_latex}')") >= 2, F.lit(3)).otherwise(F.lit(0)) +
        F.when(F.expr(f"regexp_count(text, '{p_math_term}')") >= 2, F.lit(2)).otherwise(F.lit(0)) -
        F.when(F.expr("regexp_count(text, '\\\\b\\\\d{4}\\\\b|\\\\d{1,2}/\\\\d{1,2}/\\\\d{2,4}')") >= 5, F.lit(2)).otherwise(F.lit(0))
    )
    df = df.withColumn("is_math", F.when(F.col("math_score") >= 8, F.lit(1)).otherwise(F.lit(0)))

    return df

def compute_stage3_metrics(df):
    """
    Stage 3: Advanced signals (V5).
    Removed rejections.
    Removed expensive metrics (punctuation, dependency depth, etc.).
    Preserved code_block_count, heading_count, table_count_estimate for difficulty scoring.
    """
    print("  Computing Stage 3 metrics...")
    
    # Run V5 Modality Scoring
    df = compute_robust_modality_scores(df)
    
    # Compute base structural counts needed for difficulty score structure_density
    # Structure density = code_blocks + headings + tables
    df = df.withColumn("_code_fence_matches", F.expr(f"regexp_extract_all(text, '{CODE_FENCE_PATTERN}', 0)"))
    df = df.withColumn("code_block_count", F.size(F.col("_code_fence_matches")))
    df = df.drop("_code_fence_matches")

    df = df.withColumn("_heading_matches", F.expr(f"regexp_extract_all(text, '{HEADING_PATTERN}', 0)"))
    df = df.withColumn("heading_count", F.size(F.col("_heading_matches")))
    df = df.drop("_heading_matches")
    
    # We already computed 'is_table' and 'table_score' in modality, but structure_density uses table_count_estimate
    # We can approximate table_count from V5 score or reuse regex. Let's reuse regex for consistency with formula.
    # Note: V5 modality used regex count inside python logic.
    # Let's just do a quick regex count for this specific stat:
    df = df.withColumn("table_count_estimate", F.size(F.filter(F.split(F.col("text"), '\n'), lambda line: F.regexp_count(line, F.lit(r"\|")) >= 2)))

    # Compute additional flags needed for bands/difficulty
    df = df.withColumn("has_code", F.col("is_code") == 1)
    df = df.withColumn("has_math", F.col("is_math") == 1)
    df = df.withColumn("has_reasoning", (F.col("is_reasoning") == 1) | (F.col("is_cot") == 1))
    df = df.withColumn("has_agentic", F.col("is_agentic") == 1)
    
    # Approximation for 'research_paper' using logic from V4 or V5 signals (using reasoning+math correlation)
    # V5 design relies on specific scores. Let's create a proxy 'has_research' from reasoning and math.
    df = df.withColumn("has_research_paper", (F.col("reasoning_score") >= 4) & (F.col("math_score") >= 4)) # Proxy
    
    # Flesch Readability (Retained as it's useful for profiling, though not rejection)
    df = df.withColumn("avg_word_length", safe_divide(F.col("char_length"), F.col("word_count")))
    df = df.withColumn("syllables_per_word_estimate", F.col("avg_word_length") / 3.0)
    df = df.withColumn("words_per_sentence", safe_divide(F.col("word_count"), F.col("sentence_count_estimate")))
    df = df.withColumn("flesch_reading_ease",
                       F.lit(206.835) - (F.lit(1.015) * F.col("words_per_sentence")) - 
                       (F.lit(84.6) * F.col("syllables_per_word_estimate")))

    # ------------------------------------------------------------------------
    # 7. Metadata Signal Extraction (Ground Truth)
    # ------------------------------------------------------------------------
    # From JSON column
    df = df.withColumn("meta_diff_json", F.lower(F.get_json_object(F.col("metadata"), "$.difficulty")))
    df = df.withColumn("meta_grade_json", F.get_json_object(F.col("metadata"), "$.grade"))
    
    # From text block (embedded)
    df = df.withColumn("meta_diff_text", F.lower(F.regexp_extract(F.col("text"), METADATA_DIFFICULTY_PATTERN, 1)))
    df = df.withColumn("meta_grade_text", F.regexp_extract(F.col("text"), METADATA_GRADE_PATTERN, 1))
    df = df.withColumn("meta_level_text", F.lower(F.regexp_extract(F.col("text"), METADATA_LEVEL_PATTERN, 1)))

    # Coalesce signals
    df = df.withColumn("final_meta_diff", F.coalesce(F.col("meta_diff_json"), F.col("meta_diff_text")))
    df = df.withColumn("final_meta_grade", F.coalesce(F.col("meta_grade_json"), F.col("meta_grade_text")))

    return df


# =========================================================================
# DIFFICULTY & BANDING (V5)
# =========================================================================

def compute_difficulty_score(df, keyword_pattern):
    """
    V5 Robust Difficulty Scoring.
    Removes dependencies on deleted metrics (list markers, etc.).
    Uses V5 robustness signals.
    """
    print("  Computing V5 Difficulty Score...")
    
    # Component 1: Normalized length (cap at 10K tokens)
    df = df.withColumn("_norm_length", F.least(F.col("token_count_estimate") / 10000.0, F.lit(1.0)))
    
    # Component 2: Structural density
    # Relying on code_block, heading, table_count
    total_struct = (F.col("code_block_count") + F.col("heading_count") + F.col("table_count_estimate"))
    df = df.withColumn("_struct_density",
                       F.least(total_struct / F.greatest(F.col("line_count"), F.lit(1)) * 10.0, F.lit(1.0)))
    
    # Component 3: Reasoning difficulty (using V5 scores)
    df = df.withColumn("_reason_diff",
        F.when(F.col("is_cot") == 1, F.lit(0.15)).otherwise(F.lit(0.0)) +
        F.when(F.col("is_reasoning") == 1, F.lit(0.25)).otherwise(F.lit(0.0)) +
        F.when(F.col("is_agentic") == 1, F.lit(0.30)).otherwise(F.lit(0.0))
    )
    # Cap at 0.3
    df = df.withColumn("_reason_density", F.least(F.col("_reason_diff"), F.lit(0.3)))
    
    # Component 4: Symbol density (Math/Code V5 scores)
    df = df.withColumn("_symbol_density",
        F.when(F.col("is_math") == 1, F.lit(0.20))
        .when(F.col("is_code") == 1, F.lit(0.15))
        .otherwise(F.lit(0.0))
    )
    
    # Component 5: Rarity proxy (Broadcast keywords)
    df = df.withColumn("_high_value_matches", F.size(F.regexp_extract_all(F.lower(F.col("text")), F.lit(keyword_pattern), 0)))
    df = df.withColumn("_rarity_proxy", F.least(F.col("_high_value_matches") / 5.0, F.lit(1.0)))
    
    # Component 6: Ground Truth Metadata Nudge
    # Map qualitative difficulty to numeric
    # Easy (0.2), Medium (0.4), Hard (0.8), Expert/Advanced (0.95)
    df = df.withColumn("_meta_score_diff", 
        F.when(F.col("final_meta_diff") == "hard", F.lit(0.8))
        .when(F.col("final_meta_diff").isin("expert", "advanced"), F.lit(0.95))
        .when(F.col("final_meta_diff") == "medium", F.lit(0.45))
        .when(F.col("final_meta_diff") == "easy", F.lit(0.2))
        .otherwise(F.lit(None))
    )
    
    # Map Grade level
    # Grade 11-12 (0.8), 9-10 (0.6), 5-8 (0.4), <5 (0.2)
    df = df.withColumn("_grade_val", F.col("final_meta_grade").cast("int"))
    df = df.withColumn("_meta_score_grade",
        F.when(F.col("_grade_val") >= 11, F.lit(0.8))
        .when(F.col("_grade_val") >= 9, F.lit(0.6))
        .when(F.col("_grade_val") >= 5, F.lit(0.4))
        .when(F.col("_grade_val") > 0, F.lit(0.2))
        .otherwise(F.lit(None))
    )
    
    # Combine metadata signals (use strongest signal)
    df = df.withColumn("_metadata_base", F.greatest(F.col("_meta_score_diff"), F.col("_meta_score_grade")))
    
    # Weighted Sum (Heuristic)
    df = df.withColumn("_heuristic_score",
                       F.greatest(F.lit(0.0), F.least(
                           F.lit(0.25) * F.col("_norm_length") +
                           F.lit(0.20) * F.col("_struct_density") +
                           F.lit(0.25) * F.col("_reason_density") +
                           F.lit(0.15) * F.col("_symbol_density") +
                           F.lit(0.15) * F.col("_rarity_proxy"),
                           F.lit(1.0))))
    
    # Final Blend: If metadata exists, it carries 70% weight
    df = df.withColumn("difficulty_score",
        F.when(F.col("_metadata_base").isNotNull(), 
               (F.lit(0.7) * F.col("_metadata_base") + F.lit(0.3) * F.col("_heuristic_score")))
        .otherwise(F.col("_heuristic_score"))
    )
    
    # Cleanup
    df = df.drop("_norm_length", "_struct_density", "_reason_diff", "_reason_density", "_symbol_density", 
                 "_high_value_matches", "_rarity_proxy", "_meta_score_diff", "_grade_val", "_meta_score_grade", 
                 "_metadata_base", "_heuristic_score")
    return df

def assign_curriculum_band_probabilistic(df):
    """
    V5 Probabilistic Banding.
    Uses robust signals for content nudges.
    """
    print("  Assigning V5 Probabilistic Bands...")
    
    BAND_CENTERS = {"B0": 0.05, "B1": 0.20, "B2": 0.35, "B3": 0.55, "B4": 0.75, "B5": 0.90}
    WIDTH = 0.20
    
    # Base Weights
    for band, center in BAND_CENTERS.items():
        df = df.withColumn(f"_w_{band}",
                           F.greatest(F.lit(0.0), 
                                     F.lit(1.0) - F.abs(F.col("difficulty_score") - F.lit(center)) / F.lit(WIDTH)))
    
    # Content Nudges (Using V5 Robust Columns)
    # Code Nudges
    df = df.withColumn("_w_B3", F.when(F.col("code_score") >= 6, F.col("_w_B3") + 0.05).otherwise(F.col("_w_B3")))
    df = df.withColumn("_w_B4", F.when(F.col("code_score") >= 10, F.col("_w_B4") + 0.10).otherwise(F.col("_w_B4")))
    
    # Agentic Nudges
    df = df.withColumn("_w_B4", F.when(F.col("agentic_score") >= 5, F.col("_w_B4") + 0.10).otherwise(F.col("_w_B4")))
    df = df.withColumn("_w_B5", F.when(F.col("agentic_score") >= 7, F.col("_w_B5") + 0.15).otherwise(F.col("_w_B5")))
    
    # Research/Reasoning Nudges
    df = df.withColumn("_w_B4", F.when(F.col("reasoning_score") >= 6, F.col("_w_B4") + 0.08).otherwise(F.col("_w_B4")))
    df = df.withColumn("_w_B5", F.when(F.col("cot_score") >= 5, F.col("_w_B5") + 0.12).otherwise(F.col("_w_B5")))
    
    # Math Nudges
    df = df.withColumn("_w_B3", F.when(F.col("math_score") >= 4, F.col("_w_B3") + 0.05).otherwise(F.col("_w_B3")))
    df = df.withColumn("_w_B4", F.when(F.col("math_score") >= 8, F.col("_w_B4") + 0.08).otherwise(F.col("_w_B4")))
    
    # Normalize
    df = df.withColumn("_total_weight", sum([F.col(f"_w_{b}") for b in BANDS]))
    df = df.withColumn("_total_weight", F.when(F.col("_total_weight") > 0, F.col("_total_weight")).otherwise(F.lit(1.0)))
    
    for band in BANDS:
        df = df.withColumn(f"band_p_{band}", F.col(f"_w_{band}") / F.col("_total_weight"))
        
    # Final Band (Lowest Credible)
    df = df.withColumn("band",
                      F.when(F.col(f"band_p_{BANDS[0]}") >= EPS, BANDS[0])
                      .when(F.col(f"band_p_{BANDS[1]}") >= EPS, BANDS[1])
                      .when(F.col(f"band_p_{BANDS[2]}") >= EPS, BANDS[2])
                      .when(F.col(f"band_p_{BANDS[3]}") >= EPS, BANDS[3])
                      .when(F.col(f"band_p_{BANDS[4]}") >= EPS, BANDS[4])
                      .otherwise(BANDS[5]))

    # Cleanup
    for band in BANDS: df = df.drop(f"_w_{band}")
    df = df.drop("_total_weight")
    
    return df


def prepare_output_columns(df, include_rejection=False):
    """
    Prepare columns for output.
    Aggressively optimized: only V5 scores + bands + core data.
    Removed all deleted metrics.
    Added 'text' column as requested.
    """
    core_cols = ["uuid", "id", "file_path", "source", "domain", "text", "hash", "language", "metadata"] # Added hash, language, metadata
    
    band_cols = ["band_p_B0", "band_p_B1", "band_p_B2", "band_p_B3", "band_p_B4", "band_p_B5",
                 "band", "difficulty_score"] # renamed final_band to band
                 
    v5_score_cols = ["agentic_score", "cot_score", "reasoning_score", "code_score", "math_score", "table_score"]
    
    stage1_2_cols = ["byte_length", "word_count", "unique_token_ratio", "compression_ratio", "token_count_estimate", "fertility_estimate"]
    
    rejection_cols = ["is_rejected", "rejection_reason", "rejection_level"]
    
    select_cols = core_cols + band_cols + v5_score_cols + stage1_2_cols
    
    if include_rejection:
        select_cols += rejection_cols
        
    existing_cols = [col for col in select_cols if col in df.columns]
    return df.select(*existing_cols)


# =========================================================================
# MAIN
# =========================================================================

def main():
    args, optional_args = get_glue_args()
    input_base = optional_args['INPUT_BASE']
    output_base = optional_args['OUTPUT_BASE']
    intermediate_base = optional_args['INTERMEDIATE_BASE']
    source_filter = optional_args['SOURCE']
    manual_restart_id = optional_args.get('MANUAL_RESTART', '')

    input_base = f"{input_base}/source={source_filter}" if source_filter else input_base
    rejections_path = f"{output_base}/source={source_filter}/rejections" if source_filter else f"{output_base}/rejections"
    bands_path = f"{output_base}/source={source_filter}/bands" if source_filter else f"{output_base}/bands"
    
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    
    # === CONFIGURATION TO ENFORCE FILE SIZE ===
    # 1. Force Spark to read 256 MB chunks per task (Input Split)
    #    This naturally creates 256 MB partitions in memory without shuffling.
    spark.conf.set("spark.sql.files.maxPartitionBytes", "268435456")
    # 2. Safety net: If a shuffle DOES happen (e.g. joins/grouping), 
    #    tell AQE to aim for 256 MB partitions.
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "268435456")
    
    # Spark Flex Optimizations
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.shuffle.partitions", "2000") # Lowered for Flex/Cost
    spark.conf.set("spark.sql.parquet.compression.codec", "zstd")
    spark.conf.set("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
    
    job = Job(glueContext)
    job.init('T2_metrics_v5_optimized')
    
    # Broadcast Keywords
    keyword_pattern_str = "\\b(" + "|".join(HIGH_VALUE_KEYWORDS) + ")\\b"
    print(f"Initialized with {len(HIGH_VALUE_KEYWORDS)} high-value keywords.")
    
    # Read Data
    print("Reading Input...")
    df = spark.read.parquet(input_base).select("id", "text", "source", "domain", "hash", "language", "metadata")
    # if source_filter: df = df.filter(F.col("source") == source_filter) # moved to input file path itself to avoid unnecessary scanning
    df = add_uuid_and_metadata(df)
    
    # Determine Run ID
    job_run_id = manual_restart_id or args.get('JOB_RUN_ID') or datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Job Run ID: {job_run_id}")
    
    # ---------------------------------------------------------------------
    # STAGE 1 & 2: REJECTION (Processing + Checkpointing)
    # ---------------------------------------------------------------------
    stage2_passed_path = f"{intermediate_base}/passed_stage2/{job_run_id}"
    
    if checkpoint_exists(spark, stage2_passed_path):
        print("Loading Stage 2 Checkpoint (Restart)...")
        # Optimization: We do not load rejections on restart as they are already written/discarded
        passed_stage2 = spark.read.parquet(stage2_passed_path)
    else:
        print("Processing Stage 1 & 2...")
        # Stage 1
        df_s1 = compute_stage1_metrics(df)
        rejected_stage1, passed_stage1 = apply_stage1_rejection(df_s1)
        
        # Stage 2
        df_s2 = compute_stage2_metrics(passed_stage1)
        rejected_stage2, passed_stage2 = apply_stage2_rejection(df_s2)
        
        # Consolidate Rejections
        rejected_stage1 = prepare_output_columns(rejected_stage1, include_rejection=True)
        rejected_stage2 = prepare_output_columns(rejected_stage2, include_rejection=True)
        rejected_all = rejected_stage1.unionByName(rejected_stage2, allowMissingColumns=True)
        
        # Optimization: Write Rejections immediately effectively removing them from memory
        # We use overwrite here as requested by user context
        print(f"Writing Rejections to {rejections_path} ...")
        (rejected_all.write.mode("overwrite")
         .partitionBy("rejection_level")
         .option("compression", "zstd")
         .parquet(rejections_path))
         
        # Checkpoint Passed Data (Breaks lineage + Restart safety)
        print(f"Checkpointing Passed Data to {stage2_passed_path} ...")
        passed_stage2.write.mode("overwrite").parquet(stage2_passed_path)
        write_checkpoint_flag(spark, stage2_passed_path)
        
        # Reload to ensure clean lineage
        passed_stage2 = spark.read.parquet(stage2_passed_path)

    # ---------------------------------------------------------------------
    # STAGE 3: SCORING Only (No Rejection cleanup)
    # ---------------------------------------------------------------------
    print("Processing Stage 3 (Scoring)...")
    df_s3 = compute_stage3_metrics(passed_stage2)
    
    # Difficulty & Bands
    df_s3 = compute_difficulty_score(df_s3, keyword_pattern_str)
    df_s3 = assign_curriculum_band_probabilistic(df_s3)
    
    # ---------------------------------------------------------------------
    # OUTPUTS
    # ---------------------------------------------------------------------
    print(f"Writing band outputs to {bands_path}...")
    
    # 1. Bands (Rejections already written)
    # Structure: bands/band=Y/
    bands_out_df = prepare_output_columns(df_s3, include_rejection=False)
    (bands_out_df
     .write
     .mode("overwrite")
     .partitionBy("band")
     .option("compression", "zstd")
     .parquet(bands_path))
     
    print("Optimization Complete. Job Finished.")
    job.commit()

if __name__ == '__main__':
    main()
