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

CHANGELOG V4.2 (2026-02-08):
- CRITICAL BUG FIX: Fixed regexp_extract_all error by wrapping keyword_pattern in F.lit()
- CRITICAL BUG FIX: Fixed safe_divide() double-wrapping F.lit() values
  * Now checks if input is already a Column object before wrapping
  * Prevents F.lit(F.lit(value)) errors in metric computations
  * Handles both Column objects and scalar values correctly
- MANUAL RESTART SUPPORT: Added --MANUAL_RESTART parameter for manual job restarts
  * Usage: --MANUAL_RESTART=my_custom_restart_id
  * Overrides JOB_RUN_ID when provided
  * Useful for restarting failed jobs with a known checkpoint ID
- CHECKPOINT DETECTION: Custom flag-based checkpoint tracking
  * Creates _CHECKPOINT_COMPLETE flag file (not relying on Spark's _SUCCESS)
  * Explicitly marks successful stage completion with timestamp
  * Safe naming prevents interference with Spark internals
  * Checks if Stage 1/2 checkpoints exist before processing
  * Skips computation if checkpoint is found, loads directly from S3
  * Saves cost and time when restarting after Stage 1 or Stage 2 completion

CHANGELOG V4.1 (2026-02-08):
- CRITICAL BUG FIX: Wrapped all float/int literals in F.lit() to fix PySpark type errors
- ULTRA-PERMISSIVE Stage 2: Targeting <2% rejection rate
  * unique_token_ratio: 0.02 → 0.01 (1% threshold)
  * compression_ratio: 0.98 → 0.99 (99% threshold)
  * capitalization_ratio: 0.85 → 0.90 (90% threshold, 300 words)
  * whitespace_ratio: 0.90 → 0.95 (95% threshold)
  * url_ratio: 0.5/30 → 0.6/50 (60% ratio, 50 URLs)
  * html_tag_density: 0.15/1000 → 0.20/2000 (20% density, 2K chars)
  * boilerplate_ratio: 0.35 → 0.50 (50% threshold)
  * thread_fragment: 5 → 10 markers
  * risky_tld_count: 5 → 10
  * sentence_boundary: 0.1 → 0.05 (5% minimum)
- Philosophy: PRESERVE EVERYTHING except absolute garbage
- Expected retention: 98%+ of training data
- See T2_METRICS_V4.1_SPEC.md for detailed threshold analysis

OUTPUT GOAL: Every non-rejected document gets:
1. Quality metrics for profiling/analysis
2. Probabilistic band distribution (band_p_B0...band_p_B5) for smooth curriculum
3. Conservative final_band (lowest safe band) for training safety
4. Difficulty score for transparency

New in V4.0 (PROBABILISTIC BANDING - MoE-FRIENDLY):
- **Philosophy shift**: FROM deterministic labels TO soft probability distributions
- Added 6 columns: `band_p_B0`, `band_p_B1`, `band_p_B2`, `band_p_B3`, `band_p_B4`, `band_p_B5`
- Added `final_band`: Conservative assignment (lowest credible band, EPS=0.10)
- Added `difficulty_score`: Single scalar [0,1] from cheap heuristics
- Added `fertility_estimate`: char/token ratio (requested for analysis)
- Removed hard modality overrides: now small probability nudges only
- Optimized rare word detection: broadcast variable instead of expensive regex

PROBABILISTIC BANDING BENEFITS (70B Scale):
- Eliminates curriculum cliffs (smooth transitions between stages)
- Preserves uncertainty (mixed-difficulty samples handled naturally)
- MoE-friendly (probabilities align with expert routing dynamics)
- Enables downstream analysis without reprocessing (expected band, entropy, etc.)
- Prevents early expert collapse (no hard gates on content)
- Conservative execution (final_band = lowest safe, never upgrade on uncertainty)

BAND PROBABILITY COMPUTATION (Single Pass, Cheap):
1. Compute difficulty_score ∈ [0,1] from: length, structure, reasoning, symbols, rarity
2. Map score → band probabilities using triangular weighting (fixed centers)
3. Apply small content nudges (+0.05 to +0.15) for code/agentic/research
4. Normalize to sum=1, emit 6 probability columns
5. Select final_band = argmin(band where p(band) >= EPS)

BAND CENTERS (Fixed, No Overrides):
- B0: 0.05 (surface fluency)
- B1: 0.20 (everyday language)
- B2: 0.35 (structured knowledge)
- B3: 0.55 (reasoning begins)
- B4: 0.75 (abstraction)
- B5: 0.90 (planning/agentic)

PERFORMANCE OPTIMIZATIONS (Budget-Constrained):
- Broadcast variable for high-value keywords (~150 words) instead of expensive regex
- Column pruning: read only (id, text, source, domain) initially
- Coalesce after Stage 1 to reduce partitions without full shuffle
- fertility_estimate re-added (cheap: char_length / token_count)

New in V2.8 (TRAINING-OPTIMIZED THRESHOLDS):
- Philosophy shift: FROM "keep only high quality" TO "reject only extreme noise"
- Comprehensive threshold analysis based on Dolma/Sangraha dataset characteristics
- Expected data retention: 85-90% (up from 60-70%)
- Expected false positive rate: <5% (down from 20-30%)

PHASE 1 (Books Recovery - 90% rejection → 10% rejection):
- whitespace_ratio: 0.75 → 0.85 (accommodate chapter breaks, poetry, code)
- non_printable_ratio: 0.03 → 0.10 (support Unicode scripts, math symbols)

PHASE 2 (Precision Tuning - Reduce false positives across all domains):
- unique_token_ratio: 0.1 → 0.05 + length check (preserve repetitive but valid content)
- capitalization_ratio: 0.6 → 0.7 + word_count 50→100 (reduce title false positives)
- url_ratio: 0.3 → 0.4 + url_count check (preserve papers with citations)
- html_tag_density: 0.05 → 0.10 + length check (preserve code examples)
- boilerplate_ratio: 0.15 → 0.25 (preserve legitimate metadata)
- risky_tld_count: >0 → >3 (allow security research mentions)
- sentence_boundary_coherence: 0.5 → 0.2 (reduce code/poetry false positives)
- truncation_indicators: 2 → 4 + length check (preserve cliffhangers, multi-part articles)
- code_comment_ratio: 0.8 → 0.9 (preserve tutorial/documentation code)

See T2_REJECTION_ANALYSIS.md for detailed rationale and validation guidelines.

New in V2.7 (BOOK-FRIENDLY THRESHOLDS):
- Fixed excessive rejections of book content (1610/1738 books were rejected)
- Increased whitespace_ratio threshold: 0.6 → 0.75 (books have chapter breaks, structured layout)
- Increased non_printable_ratio threshold: 0.01 → 0.03 (books have Unicode formatting)
- Increased capitalization_ratio threshold: 0.5 → 0.6 (books have chapter titles)
- Added minimum word_count checks to capitalization and corruption rules
- Expected impact: 90%+ reduction in false book rejections

New in V2.6 (S3 WRITE FIX FOR FLEX):
- Fixed UNCLASSIFIED_ERROR: Failed to delete key intermediate data
- Changed write mode from overwrite to unique timestamped paths
- Prevents S3 deletion conflicts in Flex execution (stricter permissions)
- Timestamp format: stage1_rejected_20260208_143022 (YYYYMMDD_HHMMSS)

New in V2.5 (SPARK SQL SYNTAX FIX):
- Fixed INVALID_PARAMETER_VALUE.REGEX_GROUP_INDEX error in regexp_extract_all
- Added explicit group index parameter (idx=0) to all 23 regexp_extract_all calls
- Spark SQL requires group index even for patterns without capture groups
- Without this fix, job fails immediately at first pattern extraction

New in V2.4 (GLUE FLEX COMPATIBILITY):
- Removed ALL forbidden spark.conf.set() calls for Flex execution
- Moved spark.network.timeout and spark.sql.broadcastTimeout to CLI --conf
- Updated usage examples with complete Flex-compatible configuration
- Added required --conf parameters documentation
- Worker type, number-of-workers, execution-class now in all examples

New in V2.3 (4TB PRODUCTION OPTIMIZATION):
- Fixed CANNOT_MODIFY_CONFIG error (removed forbidden spark.memory.* configs)
- Increased shuffle partitions: 2000 → 8000 for 4TB scale (~512MB per partition)
- Enabled skew join handling for unbalanced domain/source distribution
- Added S3 fast committer (mapreduce.fileoutputcommitter v2)
- Removed 7 expensive calls (global actions eliminated for performance)
- Let AQE auto-determine partition counts instead of fixed repartition
- Cost impact: Fixes job failures + eliminates unnecessary global scans

New in V2.2 (REGEX OPTIMIZATION - CRITICAL FOR 4TB):
- Batch regex processing: Reduced string traversals from 60+ to ~10 per document
- Stage 2: Single-pass pattern extraction using regexp_extract_all (4 scans vs 15+)
- Stage 3: Single-pass modality detection (8 scans vs 25+)
- Expected 60-75% reduction in Stage 2/3 compute time
- Why it matters: For 4TB, every regexp_replace scans entire string (millions of string ops)

New in V2.1:
- Added fertility_estimate, rare_word_ratio_estimate, mtld_estimate, information_density_estimate
- Added modality detection (has_code, has_math, has_agentic, primary_modality)
- Added difficulty scoring (difficulty_score, difficulty_level L0-L5)
- Removed textbook-killing length rejections (supports documents up to 10MB+)
- Physical S3 writes instead of checkpointing (100% reliable for 4TB data)
- Partitioned outputs by (domain, source) for easy downstream joins
- Incremental processing support (--SOURCE parameter)

Data Flow:
    T1 Normalized Parquet (~4TB) 
    → Stage 1: Fast corruption/minimum length checks → Physical S3 write
    → Stage 2: Spam/boilerplate/templates → Physical S3 write
    → Stage 3: Quality/Modality/Difficulty metrics
    → Output: rejection_file.parquet + metrics_file.parquet

Usage (Glue Flex):
    aws glue start-job-run --job-name T2_metrics_calculator_v5 \
        --worker-type G.2X --number-of-workers 20 --execution-class FLEX \
        --arguments '{"--INPUT_BASE":"s3://...", "--OUTPUT_BASE":"s3://...", "--SOURCE":"redpajama-arxiv", "--ESTIMATED_SIZE_GB":"100"}'

Parameters:
    --INPUT_BASE: S3 path to input data (optional, defaults to T1 output)
    --OUTPUT_BASE: S3 path for output data (optional, defaults to T2 output)
    --SOURCE: Data source name (required, filters input path)
    --ESTIMATED_SIZE_GB: Estimated input size in GB (optional, enables partition optimization)
        - Enables dynamic partition tuning: ~50MB per processing partition, ~256MB per output file
        - Recommended for datasets 10GB-1000GB to optimize parallelism and output file size
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
OUTPUT_BASE_DEFAULT = "s3://t2-datacurriculum-353/processed_dataset/curriculum_data"
INTERMEDIATE_BASE_DEFAULT = "s3://t2-datacurriculum-353/processed_dataset/checkpoints"
REPORT_BASE = "s3://t2-datacurriculum-353/processed_dataset/stats"


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
COT_REASONING_CONNECTIVES = r'\b(?:therefore|thus|hence|consequently|as\s+a\s+result|this\s+implies|which\s+implies|it\s+follows\s+that|we\s+conclude)\b'
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
MARKDOWN_TABLE_SEPARATOR = r'\|\s*[-:]{3,}\s*\|\s*[-:]{3,}(?:\s*\|\s*[-:]{3,})*\s*\|'
TABLE_ROW_PATTERN = r'\|(?:[^\n\r\|]+\|){2,}'
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
CODE_STRUCTURE = r'(?m)^(?:\t| {4,})\S'
CODE_SYNTAX_CHARS = r'[;{}()\[\]]'
# Identifier patterns for code detection (robust across domains/languages).
# - snake_case (incl digits): foo_bar, foo2_bar3
# - SCREAMING_SNAKE_CASE: HTTP_SERVER_ERROR
# - camelCase / PascalCase with 2+ humps: parseHTTPResponse, HttpServerError

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
        # optional_args['SOURCE'] = None
    
    # Optional: Manual restart ID
    if '--MANUAL_RESTART' in sys.argv:
        optional_args['MANUAL_RESTART'] = getResolvedOptions(sys.argv, ['MANUAL_RESTART'])['MANUAL_RESTART']
    else:
        optional_args['MANUAL_RESTART'] = None
    
    # Optional: Estimated data size for partition optimization
    if '--ESTIMATED_SIZE_GB' in sys.argv:
        optional_args['ESTIMATED_SIZE_GB'] = float(getResolvedOptions(sys.argv, ['ESTIMATED_SIZE_GB'])['ESTIMATED_SIZE_GB'])
    else:
        optional_args['ESTIMATED_SIZE_GB'] = None
    
    return args, optional_args

def add_uuid_and_metadata(df):
    """Add unique ID and file path for tracking."""
    df = df.withColumn("uuid", F.expr("uuid()"))
    # remove bucket information and keep only the path after bucket
    prefix_to_remove = f"{INPUT_BASE_DEFAULT}/"
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
    # Source starts with `sangraha_`, then the multiplier should be increased to 1.8 
    # to account for Indic languages which tend to have more tokens per word
    df = df.withColumn("token_count_estimate",
                       F.when(F.col("source").startswith("sangraha_"), (F.col("word_count") * 1.8).cast("int"))
                        .otherwise((F.col("word_count") * 1.3).cast("int"))
                       )
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
    
    # NOTE: capitalization_ratio removed (all-caps rule is disabled, and this wasn't used downstream)
    
    # Batch 3: Single Pass Regex (OPTIMIZED: Using regexp_count instead of regexp_extract_all)
    # Capturing: URL, Boilerplate, Thread Markers
    df = df.withColumn("url_count", F.expr(f"regexp_count(text, '{URL_PATTERN}')"))
    df = df.withColumn("url_ratio", safe_divide(F.col("url_count"), F.col("word_count")))
    
    df = df.withColumn("boilerplate_count", F.expr(f"regexp_count(lower(text), '{BOILERPLATE_PATTERN}')"))
    df = df.withColumn("boilerplate_ratio", safe_divide(F.col("boilerplate_count"), F.col("word_count")))
    
    df = df.withColumn("thread_marker_count", F.expr(f"regexp_count(text, '{THREAD_MARKER_PATTERN}')"))
    df = df.withColumn("thread_fragment_indicator", F.col("thread_marker_count")) # Kept as requested
    
    # OPTIMIZED: Use regexp_count instead of split for sentence counting
    df = df.withColumn("sentence_count_estimate", F.expr("regexp_count(text, '[.!?]+\\\\s+') + 1"))
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
    
    # NOTE: All-caps rule removed (capitalization_ratio computation removed for compute savings)
    
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
    p_python = sql_escape(PYTHON_SYNTAX)
    p_js = sql_escape(JAVASCRIPT_SYNTAX)
    p_java_cpp = sql_escape(JAVA_CPP_SYNTAX)
    p_code_struct = sql_escape(CODE_STRUCTURE)
    p_code_syntax = sql_escape(CODE_SYNTAX_CHARS)
    p_code_keywords = sql_escape(CODE_KEYWORDS_PATTERN)
    p_code_fence = sql_escape(CODE_FENCE_PATTERN)
    p_equation = sql_escape(EQUATION_PATTERN)
    p_latex = sql_escape(LATEX_COMMANDS)
    p_math_term = sql_escape(MATH_TERMINOLOGY)

    word_count_safe = F.greatest(F.col("word_count"), F.lit(1))
    line_count_safe = F.greatest(F.col("line_count"), F.lit(1))
    char_count_safe = F.greatest(F.col("char_length"), F.lit(100))

    # Reuse math symbol count across reasoning + math.
    df = df.withColumn(
        "_math_symbol_count",
        F.length(F.regexp_replace(F.col("text"), f"[^{MATH_SYMBOLS_PATTERN[1:-1]}]", "")),
    )
    df = df.withColumn("_math_symbol_ratio", safe_divide(F.col("_math_symbol_count"), char_count_safe))

    # ------------------------------------------------------------------------
    # 1. Agentic
    # ------------------------------------------------------------------------
    df = df.withColumn("_agentic_struct_hits", F.expr(f"regexp_count(text, '{p_agentic_struct}')"))
    df = df.withColumn("_agentic_vocab_hits", F.expr(f"regexp_count(lower(text), '{p_agentic_vocab}')"))
    df = df.withColumn("_agentic_vocab_ratio", safe_divide(F.col("_agentic_vocab_hits"), word_count_safe))
    df = df.withColumn("_agentic_toolish_hits", F.expr("regexp_count(text, '(?:def|function)\\\\s+\\\\w+_(?:tool|call|agent)')"))

    # Ratio-based scoring helps generalize across long web docs vs short textbook snippets.
    df = df.withColumn(
        "agentic_score",
        F.when(F.col("_agentic_struct_hits") >= 2, F.lit(4)).otherwise(F.lit(0))
        + F.when(F.col("_agentic_toolish_hits") >= 1, F.lit(3)).otherwise(F.lit(0))
        + F.when(F.col("_agentic_vocab_ratio") >= 0.010, F.lit(3))
        .when(F.col("_agentic_vocab_ratio") >= 0.005, F.lit(2))
        .when(F.col("_agentic_vocab_hits") >= 3, F.lit(1))
        .otherwise(F.lit(0)),
    )
    df = df.withColumn("is_agentic", F.when(F.col("agentic_score") >= 7, F.lit(1)).otherwise(F.lit(0)))

    # ------------------------------------------------------------------------
    # 2. Chain of Thought (CoT)
    # ------------------------------------------------------------------------
    df = df.withColumn("_cot_explicit_hits", F.expr(f"regexp_count(text, '{p_cot_explicit}')"))
    df = df.withColumn("_cot_conn_hits", F.expr(f"regexp_count(lower(text), '{p_cot_connectives}')"))
    df = df.withColumn("_cot_conn_ratio", safe_divide(F.col("_cot_conn_hits"), word_count_safe))
    df = df.withColumn("_cot_edu_hits", F.expr(f"regexp_count(text, '{sql_escape(EDUCATIONAL_MARKER_PATTERN)}')"))

    # For CoT, ratios help prevent over-triggering on long prose with many weak connectives.
    df = df.withColumn(
        "cot_score",
        F.when(F.col("_cot_explicit_hits") >= 1, F.lit(7)).otherwise(F.lit(0))
        + F.when((F.col("_cot_conn_hits") >= 3) & (F.col("_cot_conn_ratio") >= 0.003), F.lit(3))
        .when((F.col("_cot_conn_hits") >= 2) & (F.col("_cot_conn_ratio") >= 0.002), F.lit(2))
        .otherwise(F.lit(0))
        + F.when(F.col("_cot_edu_hits") >= 3, F.lit(2)).otherwise(F.lit(0))
        + F.when(
            (F.expr("regexp_count(text, '\\\\?\\\\s+.{30,}\\\\.')") >= 2)
            & (F.expr("regexp_count(text, '(?:So|Therefore|Thus|Hence|Consequently),?\\\\s+')") >= 2),
            F.lit(2),
        ).otherwise(F.lit(0)),
    )
    df = df.withColumn("is_cot", F.when(F.col("cot_score") >= 9, F.lit(1)).otherwise(F.lit(0)))

    # ------------------------------------------------------------------------
    # 3. Formal Reasoning
    # ------------------------------------------------------------------------
    df = df.withColumn("_reason_formal_hits", F.expr(f"regexp_count(text, '{p_formal_reasoning}')"))
    df = df.withColumn("_reason_cond_hits", F.expr("regexp_count(text, '(?:If|Suppose|Assume).{10,}?,\\\\s+then')"))

    df = df.withColumn(
        "reasoning_score",
        F.when(F.col("_reason_formal_hits") >= 2, F.lit(5))
        .when(F.col("_reason_formal_hits") >= 1, F.lit(3))
        .otherwise(F.lit(0))
        + F.when(F.col("_reason_cond_hits") >= 3, F.lit(2)).when(F.col("_reason_cond_hits") >= 2, F.lit(1)).otherwise(F.lit(0))
        + F.when(F.col("_math_symbol_ratio") >= 0.012, F.lit(3))
        .when(F.col("_math_symbol_ratio") >= 0.006, F.lit(2))
        .when(F.col("_math_symbol_count") >= 6, F.lit(1))
        .otherwise(F.lit(0)),
    )
    df = df.withColumn("is_reasoning", F.when(F.col("reasoning_score") >= 6, F.lit(1)).otherwise(F.lit(0)))

    # ------------------------------------------------------------------------
    # 4. Code
    # ------------------------------------------------------------------------
    df = df.withColumn("_code_python_hits", F.expr(f"regexp_count(text, '{p_python}')"))
    df = df.withColumn("_code_js_hits", F.expr(f"regexp_count(text, '{p_js}')"))
    df = df.withColumn("_code_java_cpp_hits", F.expr(f"regexp_count(text, '{p_java_cpp}')"))
    df = df.withColumn("_code_indent_hits", F.expr(f"regexp_count(text, '{p_code_struct}')"))
    df = df.withColumn("_code_syntax_hits", F.expr(f"regexp_count(text, '{p_code_syntax}')"))
    df = df.withColumn("_code_kw_hits", F.expr(f"regexp_count(text, '{p_code_keywords}')"))
    df = df.withColumn("_code_fence_hits", F.expr(f"regexp_count(text, '{p_code_fence}')"))

    # A robust, domain-agnostic indicator: what fraction of tokens look like code keywords/identifiers.
    # Using ~10% as a strong signal works well to separate prose vs code-heavy segments.
    df = df.withColumn("_code_tokenish_ratio", safe_divide(F.col("_code_kw_hits"), word_count_safe))
    df = df.withColumn("_code_line_ratio", safe_divide(F.col("_code_indent_hits"), line_count_safe))

    df = df.withColumn(
        "code_score",
        F.when(F.col("_code_tokenish_ratio") >= 0.10, F.lit(10))
        .when(F.col("_code_tokenish_ratio") >= 0.05, F.lit(7))
        .when(F.col("_code_kw_hits") >= 6, F.lit(5))
        .otherwise(F.lit(0))
        + F.when(F.col("_code_fence_hits") >= 2, F.lit(3)).otherwise(F.lit(0))
        + F.when((F.col("_code_python_hits") >= 1) | (F.col("_code_js_hits") >= 1) | (F.col("_code_java_cpp_hits") >= 1), F.lit(4)).otherwise(F.lit(0))
        + F.when((F.col("_code_syntax_hits") >= 15) & (F.col("_code_line_ratio") >= 0.05), F.lit(2)).otherwise(F.lit(0)),
    )
    df = df.withColumn("is_code", F.when(F.col("code_score") >= 9, F.lit(1)).otherwise(F.lit(0)))

    # ------------------------------------------------------------------------
    # 5. Math
    # ------------------------------------------------------------------------
    # Math symbols count (optimized: removed full unicode range check to generic match in pattern)
    df = df.withColumn("_math_equation_hits", F.expr(f"regexp_count(text, '{p_equation}')"))
    df = df.withColumn("_math_latex_hits", F.expr(f"regexp_count(text, '{p_latex}')"))
    df = df.withColumn("_math_term_hits", F.expr(f"regexp_count(lower(text), '{p_math_term}')"))
    df = df.withColumn(
        "_math_tokenish_ratio",
        safe_divide(F.col("_math_equation_hits") + F.col("_math_latex_hits") + F.col("_math_term_hits"), word_count_safe),
    )

    df = df.withColumn(
        "math_score",
        F.when(F.col("_math_symbol_ratio") >= 0.012, F.lit(5))
        .when(F.col("_math_symbol_ratio") >= 0.006, F.lit(3))
        .when(F.col("_math_symbol_count") >= 5, F.lit(2))
        .otherwise(F.lit(0))
        + F.when(F.col("_math_equation_hits") >= 3, F.lit(4)).when(F.col("_math_equation_hits") >= 1, F.lit(2)).otherwise(F.lit(0))
        + F.when(F.col("_math_latex_hits") >= 2, F.lit(3)).when(F.col("_math_latex_hits") >= 1, F.lit(2)).otherwise(F.lit(0))
        + F.when(F.col("_math_term_hits") >= 3, F.lit(2)).when(F.col("_math_term_hits") >= 2, F.lit(1)).otherwise(F.lit(0))
        + F.when(F.col("_math_tokenish_ratio") >= 0.02, F.lit(1)).otherwise(F.lit(0))
        - F.when(F.expr("regexp_count(text, '\\\\b\\\\d{4}\\\\b|\\\\d{1,2}/\\\\d{1,2}/\\\\d{2,4}')") >= 5, F.lit(2)).otherwise(F.lit(0)),
    )
    df = df.withColumn("is_math", F.when(F.col("math_score") >= 8, F.lit(1)).otherwise(F.lit(0)))

    # Cleanup internal columns (keep public scores/flags)
    df = df.drop(
        "_math_symbol_count",
        "_math_symbol_ratio",
        "_agentic_struct_hits",
        "_agentic_vocab_hits",
        "_agentic_vocab_ratio",
        "_agentic_toolish_hits",
        "_cot_explicit_hits",
        "_cot_conn_hits",
        "_cot_conn_ratio",
        "_cot_edu_hits",
        "_reason_formal_hits",
        "_reason_cond_hits",
        "_code_python_hits",
        "_code_js_hits",
        "_code_java_cpp_hits",
        "_code_indent_hits",
        "_code_syntax_hits",
        "_code_kw_hits",
        "_code_fence_hits",
        "_code_tokenish_ratio",
        "_code_line_ratio",
        "_math_equation_hits",
        "_math_latex_hits",
        "_math_term_hits",
        "_math_tokenish_ratio",
    )

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
    
    # Compute base structural counts needed for difficulty score structure_density.
    # Prefer regexp_count over array materialization (regexp_extract_all) to reduce memory/CPU.
    def sql_escape(pattern):
        return pattern.replace("\\", "\\\\").replace("'", "''")

    df = df.withColumn("code_block_count", F.expr(f"regexp_count(text, '{sql_escape(CODE_FENCE_PATTERN)}')"))
    df = df.withColumn("heading_count", F.expr(f"regexp_count(text, '{sql_escape(HEADING_PATTERN)}')"))

    # Table line estimate: count lines with >=2 pipes (matches prior logic without per-line lambda).
    df = df.withColumn("table_count_estimate", F.expr("regexp_count(text, '(?m)^.*\\\\|.*\\\\|.*$')"))

    # Compute additional flags needed for bands/difficulty
    df = df.withColumn("has_code", F.col("is_code") == 1)
    df = df.withColumn("has_cot", F.col("is_cot") == 1)
    df = df.withColumn("has_reasoning", F.col("is_reasoning") == 1)
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
    
    # Component 5: Rarity proxy (Broadcast keywords) - OPTIMIZED: Using regexp_count
    df = df.withColumn("_high_value_matches", F.expr(f"regexp_count(lower(text), '{keyword_pattern}')"))
    df = df.withColumn("_rarity_proxy", F.least(F.col("_high_value_matches") / 5.0, F.lit(1.0)))

    # Component 6: Readability proxy (Flesch)
    # Higher Flesch => easier text => lower difficulty.
    # Map to [0, 1] hardness via clamp((100 - flesch) / 100).
    df = df.withColumn(
        "_flesch_hardness",
        F.when(
            F.col("flesch_reading_ease").isNotNull(),
            F.greatest(
                F.lit(0.0),
                F.least(F.lit(1.0), (F.lit(100.0) - F.col("flesch_reading_ease")) / F.lit(100.0)),
            ),
        ).otherwise(F.lit(0.0)),
    )
    
    # Component 7: Ground Truth Metadata Nudge
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
                           F.lit(0.15) * F.col("_norm_length") +
                           F.lit(0.20) * F.col("_struct_density") +
                           F.lit(0.25) * F.col("_reason_density") +
                           F.lit(0.15) * F.col("_symbol_density") +
                           F.lit(0.15) * F.col("_rarity_proxy") +
                           F.lit(0.10) * F.col("_flesch_hardness"),
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
                 "_flesch_hardness", "_metadata_base", "_heuristic_score", "flesch_reading_ease")
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

    # create another column called assigned_band which is the same as band, requested by team 3 for easier analysis
    df = df.withColumn("assigned_band", F.col("band"))

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
    core_cols = ["uuid", "id", "file_path", "source", "domain", "hash", "language", "metadata"] # removed text col as requested by team 3
    
    band_cols = ["assigned_band", "band_p_B0", "band_p_B1", "band_p_B2", "band_p_B3", "band_p_B4", "band_p_B5",
                 "band", "difficulty_score"]
                 
    v5_score_cols = ["has_code", "has_cot", "has_reasoning", "has_agentic", "agentic_score", "cot_score", "reasoning_score", "code_score", "math_score"]
    
    stage1_2_cols = ["byte_length", "word_count", "unique_token_ratio", "compression_ratio", "token_count_estimate", "fertility_estimate"]
    
    rejection_cols = ["is_rejected", "rejection_reason", "rejection_level"]
    
    select_cols = core_cols + band_cols + v5_score_cols + stage1_2_cols
    
    if include_rejection:
        select_cols += rejection_cols
        
    existing_cols = [col for col in select_cols if col in df.columns]
    return df.select(*existing_cols)


def clean_dolma_arxiv_spark(df):  # Provided by Team 3
    """
    Optimized ArXiv structural cleaner for AWS Glue.
    Focuses on removing LaTeX artifacts and preventing over-truncation.
    """
    
    # 1. TRUNCATION: Remove Bibliography from the END (Last Match)
    # This regex identifies the LAST instance of a bibliography header to protect the body text.
    # Logic: Search for specific LaTeX section headers or the word References.
    # Group $1 captures everything before the final bibliography section.
    ref_pattern = r"(?s)(.*)\n(\\section\*?\{References\}|\\section\*?\{Bibliography\}|\\begin\{thebibliography\}|\nReferences\n|\nBibliography\n)"
    df = df.withColumn("text", F.regexp_replace(F.col("text"), ref_pattern, "$1"))
    
    # 2. REMOVE FIGURES & TIKZ (Multi-line environments)
    # (?s) is the DOTALL flag, allowing the '.' to match newlines for multi-line blocks.

    # 3. STRIP LATEX COMMANDS (Macros)
    # Removes structural markers like \section or \label while preserving the text inside/around them.

    # 4. REMOVE CITATIONS (\cite{...})
    # Strips citation keys which act as line-noise for natural language training.
    remove_pattern = (
        r"(?s:\\begin\{(figure|tikzpicture)\}.*?\\end\{\1\})|"  # Group 1: Figures (Multi-line)
        r"\\(section|subsection|subsubsection|label|caption)\*?\{.*?\}|"  # Group 2: Structs
        r"\\cite\{.*?\}"  # Cites
    )
    # Single pass to replace ANY of the above with empty string
    df = df.withColumn("text", F.regexp_replace(F.col("text"), remove_pattern, ""))

    return df

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
    estimated_size_in_gb = optional_args.get('ESTIMATED_SIZE_GB', None)

    input_base = f"{input_base}/source={source_filter}" if source_filter else input_base
    rejections_path = f"{output_base}/source={source_filter}/rejections" if source_filter else f"{output_base}/rejections"
    bands_path = f"{output_base}/source={source_filter}/bands" if source_filter else f"{output_base}/bands"
    report_base_path = f"{REPORT_BASE}/source={source_filter}" if source_filter else REPORT_BASE
    
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
    # spark.conf.set("spark.sql.shuffle.partitions", st))
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
    
    if source_filter == "redpajama-arxiv":
        print("Source identified as arxiv, from dolma dataset. Applying structural LaTeX cleaning...")
        # Apply the structural cleaning to the 'text' column before scoring
        df = clean_dolma_arxiv_spark(df)

    
    # ---------------------------------------------------------------------
    # STAGE 1 & 2: REJECTION (Optimized Checkpointing)
    # ---------------------------------------------------------------------
    # Note: We skipped restart logic for simplicity/speed; Saves massive amounts of time and S3 I/O costs
    # If the job fails, just rerun it (it's fast now, without S3 I/O)
    # Breaks Lineage using localCheckpoint(eager=True), 
    #   It clears the memory and the lineage plan
    #   To avoid crashing due to memory issues, stackoverflow issues down in the pipeline
    
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
    
    # Team 3 request
    print(f"Aggregating rejection report saved to {report_base_path}/rejections")
    agg_df = rejected_all.select("source", "token_count_estimate") \
            .groupBy("source") \
            .agg(F.sum("token_count_estimate").alias("total_tokens_estimated"), F.count("*").alias("record_count"))

    agg_df.write.mode("overwrite").csv(f"{report_base_path}/rejections", header=True)


    # Write Rejections (Keep for audit) 
    print(f"Writing Rejections to {rejections_path} ...") 
   
    (rejected_all.write.mode("overwrite")
        # .partitionBy("rejection_level")  # requested by Team 3 to not have levels
        .option("compression", "zstd")
        .parquet(rejections_path))

    # === LINEAGE BREAK ===
    # OPTIMIZATION: Using localCheckpoint for speed (no S3 I/O)
    # Since you're using G.2X workers (not Flex), local checkpoint is safe and much faster
    checkpoint_dir = f"{intermediate_base}/{job_run_id}"
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    print(f"Checkpointing Passed Data to S3 '{checkpoint_dir}' (Breaking Lineage)...")
    passed_stage2 = passed_stage2.checkpoint()
    # print("Using eager local checkpointing to break lineage (fast, in-memory)...")
    # passed_stage2 = passed_stage2.localCheckpoint(eager=True)

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

    # Team 3 request
    print(f"Aggregated banding report saved to {report_base_path}/bands")
    agg_df = bands_out_df.select("assigned_band", "source", "token_count_estimate") \
           .groupBy("assigned_band", "source") \
           .agg(F.sum("token_count_estimate").alias("total_tokens_estimated"), F.count("*").alias("record_count"))
    agg_df.write.mode("overwrite").csv(f"{report_base_path}/bands", header=True)

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
