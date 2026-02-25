"""
T2 Metrics Calculator V5.0 - EMR Optimized Version
===================================================
Converted from AWS Glue to EMR with performance optimizations.

Key Optimizations:
- Dynamic allocation for automatic executor scaling
- Adaptive Query Execution (AQE) for runtime optimization
- Speculative execution for straggler mitigation
- Optimized partition sizing based on data volume
- Memory-efficient broadcast joins
- Progressive checkpointing to break lineage

Usage (EMR Step):
    spark-submit --deploy-mode cluster \
        --conf spark.dynamicAllocation.enabled=true \
        s3://your-bucket/scripts/t2_metrics_calculator_v5_emr.py \
        --SOURCE redpajama-arxiv \
        --INPUT_BASE s3://t1-dataacquisition-datasets/processed_dataset/normalized_data \
        --OUTPUT_BASE s3://t2-datacurriculum-353/processed_dataset/curriculum_data \
        --ESTIMATED_SIZE_GB 20

Author: Converted for EMR
Date: 2026-02-11
"""

import argparse
import logging
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# =========================================================================
# LOGGING SETUP
# =========================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =========================================================================
# CONFIGURATION & CONSTANTS
# =========================================================================

VERSION = "5.0-EMR"
INPUT_BASE_DEFAULT = (
    "s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"
)
OUTPUT_BASE_DEFAULT = "s3://t2-datacurriculum-353/processed_dataset/curriculum_data"
CHECKPOINT_BASE_DEFAULT = "s3://t2-datacurriculum-353/processed_dataset/emr_checkpoints"
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
AGENTIC_STRUCTURAL_PATTERN = r"""(?x)
    (?:(?:Step\s+\d+|Task\s+\d+):\s*(?:Call|Execute|Run|Use|Invoke)\s+\w+)|
    (?:(?:tool|function|api)_(?:use|call|invoke)\s*\()|
    (?:\[(?:PLAN|ACTION|TOOL|STEP)\s*\d*\])|
    (?:Thought\s*\d*:\s*.{10,}Action\s*\d*:)
"""
AGENTIC_VOCAB_PATTERN = r"\b(?:execute|invoke|call|dispatch|orchestrate|coordinate|delegate|subgoal|subtask|decompose|breakdown|workflow|pipeline)\b"

# Pattern 2: CHAIN-OF-THOUGHT
COT_EXPLICIT_PATTERN = r"""(?x)
    (?:Let's\s+think\s+(?:step[- ]by[- ]step|through\s+this|carefully|systematically))|
    (?:\[(?:REASONING|THINKING|ANALYSIS)\])|
    (?:I\s+(?:need\s+to|should|must)\s+(?:think\s+about|consider|analyze))
"""
COT_REASONING_CONNECTIVES = r"\b(?:therefore|thus|hence|consequently|as\s+a\s+result|this\s+implies|which\s+implies|it\s+follows\s+that|we\s+conclude)\b"
EDUCATIONAL_MARKER_PATTERN = (
    r"(?i)###\s*(?:Explanation|Question|Answer|Topic|Metadata|Prerequisites):"
)

# Pattern 3: FORMAL REASONING
FORMAL_REASONING_PATTERN = r"""(?x)
    (?:Proof:|Theorem:|Lemma:|Corollary:)|
    (?:Q\.E\.D\.|∎|□)|
    (?:(?:By|Using)\s+(?:induction|contradiction|construction))|
    (?:It\s+follows\s+that|We\s+can\s+deduce|This\s+implies)
"""
MATH_SYMBOLS_PATTERN = r"[∀∃∈∉⊂⊆∪∩∅⇒⇔∧∨¬→↔⊢⊨≡≠≤≥±∓∞∑∏∫√]"

# Pattern 4: TABLE STRUCTURES
MARKDOWN_TABLE_SEPARATOR = r"\|\s*[-:]{3,}\s*\|\s*[-:]{3,}(?:\s*\|\s*[-:]{3,})*\s*\|"
TABLE_ROW_PATTERN = r"\|(?:[^\n\r\|]+\|){2,}"
TABLE_HEADER_KEYWORDS = (
    r"(?i)\b(?:name|id|value|type|date|count|total|column|field|description)\b"
)

# Pattern 5: CODE WITH COMMENTS
CODE_COMMENT_SYNTAX = r'''(?x)
    (?:^[ \t]*(?://|#)\s+\w+)|
    (?:/\*.*?\*/)|
    (?:(?:"""|\'\'\').{30,}?(?:"""|\'\'\''))
'''
CODE_KEYWORDS_PATTERN = r"\b(?:function|def|class|return|import|from|const|let|var|if|else|for|while|try|catch|public|private)\b"

# Pattern 6: Q&A
QA_PAIR_PATTERN = r"""(?x)
    (?:Q(?:uestion)?|Query)\s*\d*[:.]?\s*.{20,}?\?\s+A(?:nswer)?[:.]?\s*.{30,}|
    (?:^|\n)(?:Q|Question):\s*.{20,}?\?\s+(?:A|Answer):\s*.{30,}
"""
QA_ANSWER_MARKERS = r"\?\s+(?:The\s+answer\s+is|It\s+is\s+because|Yes|No|In\s+summary)"

# Pattern 7: CODE (Multi-language)
PYTHON_SYNTAX = r"(?:^|\n)(?:def|class|import|from\s+\w+\s+import)\s+\w+"
JAVASCRIPT_SYNTAX = r"(?:function\s+\w+\s*\(|const|let|var)\s+\w+\s*=|=>"
JAVA_CPP_SYNTAX = r"(?:public|private|protected|#include|int\s+main)"
CODE_STRUCTURE = r"(?m)^(?:\t| {4,})\S"
CODE_SYNTAX_CHARS = r"[;{}()\[\]]"

# Pattern 8: MATH CONTENT
EQUATION_PATTERN = r"[a-z]\s*[+\-*/=]\s*[a-z0-9]|[a-z]\^[0-9]|\([a-z0-9\s+\-*/]+\)\s*="
LATEX_COMMANDS = r"\\(?:frac|sum|prod|int|lim|infty|sqrt|cdot|times|begin\{equation)"
MATH_TERMINOLOGY = (
    r"\b(?:theorem|lemma|proof|equation|derivative|integral|matrix|vector|polynomial)\b"
)

# Other Patterns
BOILERPLATE_PATTERN = r"(?i)(cookie policy|privacy policy|terms of service|all rights reserved|© copyright|click here|subscribe to|sign up|newsletter|unsubscribe|contact us|about us|follow us on|accept cookies|manage preferences)"
THREAD_MARKER_PATTERN = (
    r"(>>|replied to:|in response to|re:|replying to|quote from|responding to)"
)
URL_PATTERN = r"https?://[^\s]+"
CODE_FENCE_PATTERN = r"```|~~~"
HEADING_PATTERN = r"^#+\s+|^[A-Z][^\n]{5,50}$"

# Metadata Extraction Patterns
METADATA_DIFFICULTY_PATTERN = r"(?i)Difficulty:\s*(\w+)"
METADATA_GRADE_PATTERN = r"(?i)Grade:\s*(\d+)"
METADATA_LEVEL_PATTERN = r"(?i)Student Level:\s*(\w+)"

# High-value keywords for complexity detection
HIGH_VALUE_KEYWORDS = [
    "hypothesis",
    "methodology",
    "empirical",
    "theorem",
    "lemma",
    "corollary",
    "ontology",
    "epistemology",
    "phenomenology",
    "hermeneutics",
    "dialectic",
    "paradigm",
    "heuristic",
    "algorithm",
    "optimization",
    "convergence",
    "heterogeneous",
    "homogeneous",
    "isotropic",
    "anisotropic",
    "stochastic",
    "deterministic",
    "asymptotic",
    "parametric",
    "nonparametric",
    "multivariate",
    "eigenvalue",
    "eigenvector",
    "gradient",
    "jacobian",
    "hessian",
    "pathogenesis",
    "etiology",
    "pharmacokinetics",
    "pharmacodynamics",
    "metabolism",
    "carcinogenesis",
    "immunology",
    "cytology",
    "histology",
    "morphology",
    "jurisprudence",
    "adjudication",
    "litigation",
    "jurisdiction",
    "precedent",
    "appellant",
    "respondent",
    "plaintiff",
    "defendant",
    "indictment",
    "polynomial",
    "exponential",
    "logarithmic",
    "trigonometric",
    "hyperbolic",
    "differential",
    "integral",
    "derivative",
    "convolution",
    "fourier",
    "bayesian",
    "frequentist",
    "likelihood",
    "posterior",
    "prior",
    "polymorphism",
    "encapsulation",
    "inheritance",
    "abstraction",
    "concurrency",
    "parallelism",
    "distributed",
    "synchronization",
    "mutex",
    "semaphore",
    "recursion",
    "memoization",
    "backtracking",
    "hashing",
    "traversal",
    "syllogism",
    "tautology",
    "contradiction",
    "axiom",
    "inference",
    "deduction",
    "induction",
    "abduction",
    "fallacy",
    "proposition",
    "elasticity",
    "equilibrium",
    "arbitrage",
    "volatility",
    "derivative",
    "amortization",
    "depreciation",
    "valuation",
    "liquidity",
    "solvency",
    "thermodynamics",
    "kinetics",
    "dynamics",
    "statics",
    "mechanics",
    "electromagnetic",
    "semiconductor",
    "transistor",
    "amplifier",
    "oscillator",
    "stoichiometry",
    "titration",
    "catalysis",
    "synthesis",
    "hydrolysis",
    "oxidation",
    "reduction",
    "equilibrium",
    "entropy",
    "enthalpy",
    "quantum",
    "relativity",
    "spacetime",
    "superposition",
    "entanglement",
    "hamiltonian",
    "lagrangian",
    "schrodinger",
    "heisenberg",
    "maxwell",
    "dichotomy",
    "juxtaposition",
    "ubiquitous",
    "ephemeral",
    "perpetual",
    "ambiguous",
    "arbitrary",
    "intrinsic",
    "extrinsic",
    "implicit",
    "explicit",
    "analogous",
    "congruent",
    "disparate",
    "heterogeneous",
    "therefore",
    "consequently",
    "nevertheless",
    "notwithstanding",
    "furthermore",
    "moreover",
    "conversely",
    "alternatively",
    "analogously",
    "presumably",
]


# =========================================================================
# ARGUMENT PARSING (EMR Compatible)
# =========================================================================


def parse_args():
    """Parse command line arguments for EMR."""
    parser = argparse.ArgumentParser(description="T2 Metrics Calculator V5 - EMR")
    parser.add_argument("--SOURCE", required=True, help="Data source name (required)")
    parser.add_argument(
        "--INPUT_BASE", default=INPUT_BASE_DEFAULT, help="S3 input path"
    )
    parser.add_argument(
        "--OUTPUT_BASE", default=OUTPUT_BASE_DEFAULT, help="S3 output path"
    )
    parser.add_argument(
        "--CHECKPOINT_BASE", default=CHECKPOINT_BASE_DEFAULT, help="S3 checkpoint path"
    )
    parser.add_argument(
        "--ESTIMATED_SIZE_GB",
        type=float,
        default=None,
        help="Estimated data size in GB",
    )
    return parser.parse_args()


# =========================================================================
# SPARK SESSION WITH OPTIMIZED CONFIGS
# =========================================================================


def create_optimized_spark_session(estimated_size_gb=None):
    """
    Create SparkSession with optimizations for EMR.
    Ensures full cluster utilization.
    """
    logger.info("Creating optimized Spark session...")

    builder = SparkSession.builder.appName(f"T2_Metrics_V5_EMR_{VERSION}")

    # ===== ADAPTIVE QUERY EXECUTION (AQE) =====
    # Automatically optimizes queries at runtime
    builder = (
        builder.config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
    )

    # ===== DYNAMIC ALLOCATION =====
    # Scales executors up/down based on workload
    builder = (
        builder.config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "1")
        .config("spark.dynamicAllocation.maxExecutors", "100")
        .config("spark.dynamicAllocation.initialExecutors", "5")
        .config("spark.dynamicAllocation.executorIdleTimeout", "60s")
        .config("spark.dynamicAllocation.schedulerBacklogTimeout", "5s")
        .config("spark.dynamicAllocation.sustainedSchedulerBacklogTimeout", "5s")
    )

    # ===== SPECULATIVE EXECUTION =====
    # Re-launches slow tasks on other nodes
    builder = (
        builder.config("spark.speculation", "true")
        .config("spark.speculation.multiplier", "1.5")
        .config("spark.speculation.quantile", "0.9")
        .config("spark.speculation.minTaskRuntime", "60s")
    )

    # ===== MEMORY & PARALLELISM =====
    builder = (
        builder.config("spark.sql.files.maxPartitionBytes", "268435456")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.3")
    )

    # ===== BROADCAST & JOIN OPTIMIZATION =====
    builder = builder.config("spark.sql.autoBroadcastJoinThreshold", "50MB").config(
        "spark.sql.broadcastTimeout", "600"
    )

    # ===== OUTPUT OPTIMIZATION =====
    builder = (
        builder.config("spark.sql.parquet.compression.codec", "zstd")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.sql.parquet.mergeSchema", "false")
    )

    # ===== NETWORK & SERIALIZATION =====
    builder = (
        builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "1024m")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "60s")
    )

    spark = builder.getOrCreate()

    # Log configuration
    logger.info(f"Spark version: {spark.version}")
    logger.info(
        f"Dynamic allocation enabled: {spark.conf.get('spark.dynamicAllocation.enabled')}"
    )
    logger.info(f"AQE enabled: {spark.conf.get('spark.sql.adaptive.enabled')}")
    logger.info(f"Speculation enabled: {spark.conf.get('spark.speculation')}")

    return spark


def optimize_partitions(df, spark, estimated_size_gb=None):
    """
    Optimize DataFrame partitions based on data size and cluster resources.
    Ensures even distribution across executors.
    """
    if estimated_size_gb:
        # Target ~128MB per partition for optimal parallelism
        target_partition_mb = 128
        num_partitions = max(int((estimated_size_gb * 1024) / target_partition_mb), 20)

        # Cap at reasonable maximum
        num_partitions = min(num_partitions, 2000)

        logger.info(
            f"Repartitioning to {num_partitions} partitions for {estimated_size_gb}GB data"
        )
        df = df.repartition(num_partitions)
    else:
        # Let AQE handle it, but ensure minimum parallelism
        current_partitions = df.rdd.getNumPartitions()
        if current_partitions < 20:
            logger.info(f"Increasing partitions from {current_partitions} to 100")
            df = df.repartition(100)

    return df


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================


def add_uuid_and_metadata(df, input_base):
    """Add unique ID and file path for tracking."""
    df = df.withColumn("uuid", F.expr("uuid()"))
    prefix_to_remove = f"{input_base}/"
    df = df.withColumn("file_path", F.input_file_name())
    df = df.withColumn(
        "file_path", F.regexp_replace(F.col("file_path"), prefix_to_remove, "")
    )
    return df


def safe_divide(numerator, denominator, default=0.0):
    """Safe division preventing divide-by-zero."""
    from pyspark.sql import Column

    if not isinstance(numerator, Column):
        numerator = F.lit(numerator)
    if not isinstance(denominator, Column):
        denominator = F.lit(denominator)
    if not isinstance(default, Column):
        default = F.lit(default)
    return F.when(denominator > 0, numerator / denominator).otherwise(default)


# =========================================================================
# STAGE 1: PHYSICAL & BASIC CORRUPTION
# =========================================================================


def compute_stage1_metrics(df):
    """Stage 1: Physical properties."""
    logger.info("Computing Stage 1 metrics...")

    df = df.withColumn("byte_length", F.length(F.encode(F.col("text"), "utf-8")))
    df = df.withColumn("char_length", F.length(F.col("text")))
    df = df.withColumn("word_count", F.size(F.split(F.col("text"), r"\s+")) - 1)
    df = df.withColumn("line_count", F.size(F.split(F.col("text"), "\n")))

    # Token count estimate
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


def apply_stage1_rejection(df):
    """Apply Priority 1 rejection rules."""
    logger.info("Applying Stage 1 rejection rules...")

    df = df.withColumn("is_rejected", F.lit(False))
    df = df.withColumn("rejection_reason", F.lit(""))
    df = df.withColumn("rejection_level", F.lit(None).cast(IntegerType()))

    # Rule 1: Byte length < 50
    cond_byte = F.col("byte_length") < 50
    df = df.withColumn(
        "is_rejected", F.when(cond_byte, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond_byte, "too_short_bytes").otherwise(F.col("rejection_reason")),
    )
    df = df.withColumn(
        "rejection_level", F.when(cond_byte, 1).otherwise(F.col("rejection_level"))
    )

    # Rule 2: Char length < 20
    cond_char = F.col("char_length") < 20
    df = df.withColumn(
        "is_rejected", F.when(cond_char, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond_char, "too_short_chars").otherwise(F.col("rejection_reason")),
    )
    df = df.withColumn(
        "rejection_level", F.when(cond_char, 1).otherwise(F.col("rejection_level"))
    )

    # Rule 3: Token count < 10
    cond_token = F.col("token_count_estimate") < 10
    df = df.withColumn(
        "is_rejected", F.when(cond_token, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond_token, "too_short_tokens").otherwise(F.col("rejection_reason")),
    )
    df = df.withColumn(
        "rejection_level", F.when(cond_token, 1).otherwise(F.col("rejection_level"))
    )

    return df.filter(F.col("is_rejected")), df.filter(~F.col("is_rejected"))


# =========================================================================
# STAGE 2: NOISE & SPAM
# =========================================================================


def compute_stage2_metrics(df):
    """Stage 2: Noise detection with batched operations."""
    logger.info("Computing Stage 2 metrics...")

    # Unique token ratio
    df = df.withColumn(
        "unique_token_ratio",
        safe_divide(
            F.size(F.array_distinct(F.split(F.lower(F.col("text")), r"\s+"))),
            F.col("word_count"),
        ),
    )

    # Compression proxy
    df = df.withColumn(
        "compression_ratio", safe_divide(F.col("byte_length"), F.col("char_length"))
    )

    # Whitespace ratio
    text_len = F.length(F.col("text"))
    df = df.withColumn(
        "whitespace_count",
        text_len - F.length(F.regexp_replace(F.col("text"), r"\S", "")),
    )
    df = df.withColumn(
        "whitespace_ratio", safe_divide(F.col("whitespace_count"), F.col("char_length"))
    )

    # URL, boilerplate, thread markers - using regexp_count
    df = df.withColumn("url_count", F.expr(f"regexp_count(text, '{URL_PATTERN}')"))
    df = df.withColumn(
        "url_ratio", safe_divide(F.col("url_count"), F.col("word_count"))
    )

    df = df.withColumn(
        "boilerplate_count",
        F.expr(f"regexp_count(lower(text), '{BOILERPLATE_PATTERN}')"),
    )
    df = df.withColumn(
        "boilerplate_ratio",
        safe_divide(F.col("boilerplate_count"), F.col("word_count")),
    )

    df = df.withColumn(
        "thread_marker_count", F.expr(f"regexp_count(text, '{THREAD_MARKER_PATTERN}')")
    )
    df = df.withColumn("thread_fragment_indicator", F.col("thread_marker_count"))

    # Sentence count estimate
    df = df.withColumn(
        "sentence_count_estimate", F.expr("regexp_count(text, '[.!?]+\\\\s+') + 1")
    )
    df = df.withColumn(
        "sentence_count_estimate",
        F.when(F.col("sentence_count_estimate") < 1, 1).otherwise(
            F.col("sentence_count_estimate")
        ),
    )

    # Rare word estimate
    df = df.withColumn(
        "rare_word_ratio_estimate",
        F.when(
            F.col("unique_token_ratio") > 0.4,
            F.lit(1.0) - (F.lit(2.0) * F.col("unique_token_ratio")),
        ).otherwise(F.lit(0.1)),
    )

    return df


def apply_stage2_rejection(df):
    """Apply Priority 2 rejection rules."""
    logger.info("Applying Stage 2 rejection rules...")

    df = df.withColumn("is_rejected", F.lit(False))
    df = df.withColumn("rejection_reason", F.lit(""))
    df = df.withColumn("rejection_level", F.lit(None).cast(IntegerType()))

    # Rule 1: Repetitive template
    cond_rep = (F.col("unique_token_ratio") < 0.01) & (F.col("word_count") > 200)
    df = df.withColumn(
        "is_rejected", F.when(cond_rep, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond_rep, "repetitive_template").otherwise(F.col("rejection_reason")),
    )
    df = df.withColumn(
        "rejection_level", F.when(cond_rep, 2).otherwise(F.col("rejection_level"))
    )

    # Rule 2: Whitespace
    cond_white = F.col("whitespace_ratio") > 0.95
    df = df.withColumn(
        "is_rejected", F.when(cond_white, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond_white, "excessive_whitespace").otherwise(F.col("rejection_reason")),
    )
    df = df.withColumn(
        "rejection_level", F.when(cond_white, 2).otherwise(F.col("rejection_level"))
    )

    # Rule 3: Link Spam
    cond_link = (F.col("url_ratio") > 0.7) & (F.col("url_count") > 50)
    df = df.withColumn(
        "is_rejected", F.when(cond_link, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond_link, "link_spam").otherwise(F.col("rejection_reason")),
    )
    df = df.withColumn(
        "rejection_level", F.when(cond_link, 2).otherwise(F.col("rejection_level"))
    )

    # Rule 4: Boilerplate
    cond_boil = F.col("boilerplate_ratio") > 0.50
    df = df.withColumn(
        "is_rejected", F.when(cond_boil, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond_boil, "boilerplate_spam").otherwise(F.col("rejection_reason")),
    )
    df = df.withColumn(
        "rejection_level", F.when(cond_boil, 2).otherwise(F.col("rejection_level"))
    )

    # Rule 5: Thread Fragment
    cond_thread = (F.col("thread_fragment_indicator") > 5) & (
        F.col("token_count_estimate") < 200
    )
    df = df.withColumn(
        "is_rejected", F.when(cond_thread, True).otherwise(F.col("is_rejected"))
    )
    df = df.withColumn(
        "rejection_reason",
        F.when(cond_thread, "orphaned_thread_fragment").otherwise(
            F.col("rejection_reason")
        ),
    )
    df = df.withColumn(
        "rejection_level", F.when(cond_thread, 2).otherwise(F.col("rejection_level"))
    )

    return df.filter(F.col("is_rejected")), df.filter(~F.col("is_rejected"))


# =========================================================================
# STAGE 3: MODALITY SCORING (V5)
# =========================================================================


def compute_robust_modality_scores(df):
    """V5 Robust Modality Scoring with multi-signal approach."""
    logger.info("Computing V5 Modality Scores...")

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

    # Math symbol count (reused across reasoning + math)
    df = df.withColumn(
        "_math_symbol_count",
        F.length(
            F.regexp_replace(F.col("text"), f"[^{MATH_SYMBOLS_PATTERN[1:-1]}]", "")
        ),
    )
    df = df.withColumn(
        "_math_symbol_ratio", safe_divide(F.col("_math_symbol_count"), char_count_safe)
    )

    # 1. Agentic
    df = df.withColumn(
        "_agentic_struct_hits", F.expr(f"regexp_count(text, '{p_agentic_struct}')")
    )
    df = df.withColumn(
        "_agentic_vocab_hits", F.expr(f"regexp_count(lower(text), '{p_agentic_vocab}')")
    )
    df = df.withColumn(
        "_agentic_vocab_ratio",
        safe_divide(F.col("_agentic_vocab_hits"), word_count_safe),
    )
    df = df.withColumn(
        "_agentic_toolish_hits",
        F.expr(
            "regexp_count(text, '(?:def|function)\\\\s+\\\\w+_(?:tool|call|agent)')"
        ),
    )

    df = df.withColumn(
        "agentic_score",
        F.when(F.col("_agentic_struct_hits") >= 2, F.lit(4)).otherwise(F.lit(0))
        + F.when(F.col("_agentic_toolish_hits") >= 1, F.lit(3)).otherwise(F.lit(0))
        + F.when(F.col("_agentic_vocab_ratio") >= 0.010, F.lit(3))
        .when(F.col("_agentic_vocab_ratio") >= 0.005, F.lit(2))
        .when(F.col("_agentic_vocab_hits") >= 3, F.lit(1))
        .otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "is_agentic", F.when(F.col("agentic_score") >= 7, F.lit(1)).otherwise(F.lit(0))
    )

    # 2. Chain of Thought (CoT)
    df = df.withColumn(
        "_cot_explicit_hits", F.expr(f"regexp_count(text, '{p_cot_explicit}')")
    )
    df = df.withColumn(
        "_cot_conn_hits", F.expr(f"regexp_count(lower(text), '{p_cot_connectives}')")
    )
    df = df.withColumn(
        "_cot_conn_ratio", safe_divide(F.col("_cot_conn_hits"), word_count_safe)
    )
    df = df.withColumn(
        "_cot_edu_hits",
        F.expr(f"regexp_count(text, '{sql_escape(EDUCATIONAL_MARKER_PATTERN)}')"),
    )

    df = df.withColumn(
        "cot_score",
        F.when(F.col("_cot_explicit_hits") >= 1, F.lit(7)).otherwise(F.lit(0))
        + F.when(
            (F.col("_cot_conn_hits") >= 3) & (F.col("_cot_conn_ratio") >= 0.003),
            F.lit(3),
        )
        .when(
            (F.col("_cot_conn_hits") >= 2) & (F.col("_cot_conn_ratio") >= 0.002),
            F.lit(2),
        )
        .otherwise(F.lit(0))
        + F.when(F.col("_cot_edu_hits") >= 3, F.lit(2)).otherwise(F.lit(0))
        + F.when(
            (F.expr("regexp_count(text, '\\\\?\\\\s+.{30,}\\\\.')") >= 2)
            & (
                F.expr(
                    "regexp_count(text, '(?:So|Therefore|Thus|Hence|Consequently),?\\\\s+')"
                )
                >= 2
            ),
            F.lit(2),
        ).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "is_cot", F.when(F.col("cot_score") >= 9, F.lit(1)).otherwise(F.lit(0))
    )

    # 3. Formal Reasoning
    df = df.withColumn(
        "_reason_formal_hits", F.expr(f"regexp_count(text, '{p_formal_reasoning}')")
    )
    df = df.withColumn(
        "_reason_cond_hits",
        F.expr("regexp_count(text, '(?:If|Suppose|Assume).{10,}?,\\\\s+then')"),
    )

    df = df.withColumn(
        "reasoning_score",
        F.when(F.col("_reason_formal_hits") >= 2, F.lit(5))
        .when(F.col("_reason_formal_hits") >= 1, F.lit(3))
        .otherwise(F.lit(0))
        + F.when(F.col("_reason_cond_hits") >= 3, F.lit(2))
        .when(F.col("_reason_cond_hits") >= 2, F.lit(1))
        .otherwise(F.lit(0))
        + F.when(F.col("_math_symbol_ratio") >= 0.012, F.lit(3))
        .when(F.col("_math_symbol_ratio") >= 0.006, F.lit(2))
        .when(F.col("_math_symbol_count") >= 6, F.lit(1))
        .otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "is_reasoning",
        F.when(F.col("reasoning_score") >= 6, F.lit(1)).otherwise(F.lit(0)),
    )

    # 4. Code
    df = df.withColumn("_code_python_hits", F.expr(f"regexp_count(text, '{p_python}')"))
    df = df.withColumn("_code_js_hits", F.expr(f"regexp_count(text, '{p_js}')"))
    df = df.withColumn(
        "_code_java_cpp_hits", F.expr(f"regexp_count(text, '{p_java_cpp}')")
    )
    df = df.withColumn(
        "_code_indent_hits", F.expr(f"regexp_count(text, '{p_code_struct}')")
    )
    df = df.withColumn(
        "_code_syntax_hits", F.expr(f"regexp_count(text, '{p_code_syntax}')")
    )
    df = df.withColumn(
        "_code_kw_hits", F.expr(f"regexp_count(text, '{p_code_keywords}')")
    )
    df = df.withColumn(
        "_code_fence_hits", F.expr(f"regexp_count(text, '{p_code_fence}')")
    )

    df = df.withColumn(
        "_code_tokenish_ratio", safe_divide(F.col("_code_kw_hits"), word_count_safe)
    )
    df = df.withColumn(
        "_code_line_ratio", safe_divide(F.col("_code_indent_hits"), line_count_safe)
    )

    df = df.withColumn(
        "code_score",
        F.when(F.col("_code_tokenish_ratio") >= 0.10, F.lit(10))
        .when(F.col("_code_tokenish_ratio") >= 0.05, F.lit(7))
        .when(F.col("_code_kw_hits") >= 6, F.lit(5))
        .otherwise(F.lit(0))
        + F.when(F.col("_code_fence_hits") >= 2, F.lit(3)).otherwise(F.lit(0))
        + F.when(
            (F.col("_code_python_hits") >= 1)
            | (F.col("_code_js_hits") >= 1)
            | (F.col("_code_java_cpp_hits") >= 1),
            F.lit(4),
        ).otherwise(F.lit(0))
        + F.when(
            (F.col("_code_syntax_hits") >= 15) & (F.col("_code_line_ratio") >= 0.05),
            F.lit(2),
        ).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "is_code", F.when(F.col("code_score") >= 9, F.lit(1)).otherwise(F.lit(0))
    )

    # 5. Math
    df = df.withColumn(
        "_math_equation_hits", F.expr(f"regexp_count(text, '{p_equation}')")
    )
    df = df.withColumn("_math_latex_hits", F.expr(f"regexp_count(text, '{p_latex}')"))
    df = df.withColumn(
        "_math_term_hits", F.expr(f"regexp_count(lower(text), '{p_math_term}')")
    )
    df = df.withColumn(
        "_math_tokenish_ratio",
        safe_divide(
            F.col("_math_equation_hits")
            + F.col("_math_latex_hits")
            + F.col("_math_term_hits"),
            word_count_safe,
        ),
    )

    df = df.withColumn(
        "math_score",
        F.when(F.col("_math_symbol_ratio") >= 0.012, F.lit(5))
        .when(F.col("_math_symbol_ratio") >= 0.006, F.lit(3))
        .when(F.col("_math_symbol_count") >= 5, F.lit(2))
        .otherwise(F.lit(0))
        + F.when(F.col("_math_equation_hits") >= 3, F.lit(4))
        .when(F.col("_math_equation_hits") >= 1, F.lit(2))
        .otherwise(F.lit(0))
        + F.when(F.col("_math_latex_hits") >= 2, F.lit(3))
        .when(F.col("_math_latex_hits") >= 1, F.lit(2))
        .otherwise(F.lit(0))
        + F.when(F.col("_math_term_hits") >= 3, F.lit(2))
        .when(F.col("_math_term_hits") >= 2, F.lit(1))
        .otherwise(F.lit(0))
        + F.when(F.col("_math_tokenish_ratio") >= 0.02, F.lit(1)).otherwise(F.lit(0))
        - F.when(
            F.expr(
                "regexp_count(text, '\\\\b\\\\d{4}\\\\b|\\\\d{1,2}/\\\\d{1,2}/\\\\d{2,4}')"
            )
            >= 5,
            F.lit(2),
        ).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "is_math", F.when(F.col("math_score") >= 8, F.lit(1)).otherwise(F.lit(0))
    )

    # Cleanup internal columns
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
    """Stage 3: Advanced signals (V5)."""
    logger.info("Computing Stage 3 metrics...")

    df = compute_robust_modality_scores(df)

    def sql_escape(pattern):
        return pattern.replace("\\", "\\\\").replace("'", "''")

    df = df.withColumn(
        "code_block_count",
        F.expr(f"regexp_count(text, '{sql_escape(CODE_FENCE_PATTERN)}')"),
    )
    df = df.withColumn(
        "heading_count", F.expr(f"regexp_count(text, '{sql_escape(HEADING_PATTERN)}')")
    )
    df = df.withColumn(
        "table_count_estimate", F.expr("regexp_count(text, '(?m)^.*\\\\|.*\\\\|.*$')")
    )

    # Flags
    df = df.withColumn("has_code", F.col("is_code") == 1)
    df = df.withColumn("has_cot", F.col("is_cot") == 1)
    df = df.withColumn("has_reasoning", F.col("is_reasoning") == 1)
    df = df.withColumn("has_agentic", F.col("is_agentic") == 1)
    df = df.withColumn(
        "has_research_paper",
        (F.col("reasoning_score") >= 4) & (F.col("math_score") >= 4),
    )

    # Readability
    df = df.withColumn(
        "avg_word_length", safe_divide(F.col("char_length"), F.col("word_count"))
    )
    df = df.withColumn("syllables_per_word_estimate", F.col("avg_word_length") / 3.0)
    df = df.withColumn(
        "words_per_sentence",
        safe_divide(F.col("word_count"), F.col("sentence_count_estimate")),
    )
    df = df.withColumn(
        "flesch_reading_ease",
        F.lit(206.835)
        - (F.lit(1.015) * F.col("words_per_sentence"))
        - (F.lit(84.6) * F.col("syllables_per_word_estimate")),
    )

    # Metadata extraction
    df = df.withColumn(
        "meta_diff_json", F.lower(F.get_json_object(F.col("metadata"), "$.difficulty"))
    )
    df = df.withColumn(
        "meta_grade_json", F.get_json_object(F.col("metadata"), "$.grade")
    )
    df = df.withColumn(
        "meta_diff_text",
        F.lower(F.regexp_extract(F.col("text"), METADATA_DIFFICULTY_PATTERN, 1)),
    )
    df = df.withColumn(
        "meta_grade_text", F.regexp_extract(F.col("text"), METADATA_GRADE_PATTERN, 1)
    )
    df = df.withColumn(
        "meta_level_text",
        F.lower(F.regexp_extract(F.col("text"), METADATA_LEVEL_PATTERN, 1)),
    )
    df = df.withColumn(
        "final_meta_diff", F.coalesce(F.col("meta_diff_json"), F.col("meta_diff_text"))
    )
    df = df.withColumn(
        "final_meta_grade",
        F.coalesce(F.col("meta_grade_json"), F.col("meta_grade_text")),
    )

    return df


# =========================================================================
# DIFFICULTY & BANDING
# =========================================================================


def compute_difficulty_score(df, keyword_pattern):
    """V5 Robust Difficulty Scoring."""
    logger.info("Computing V5 Difficulty Score...")

    # Component 1: Normalized length
    df = df.withColumn(
        "_norm_length", F.least(F.col("token_count_estimate") / 10000.0, F.lit(1.0))
    )

    # Component 2: Structural density
    total_struct = (
        F.col("code_block_count")
        + F.col("heading_count")
        + F.col("table_count_estimate")
    )
    df = df.withColumn(
        "_struct_density",
        F.least(
            total_struct / F.greatest(F.col("line_count"), F.lit(1)) * 10.0, F.lit(1.0)
        ),
    )

    # Component 3: Reasoning difficulty
    df = df.withColumn(
        "_reason_diff",
        F.when(F.col("is_cot") == 1, F.lit(0.15)).otherwise(F.lit(0.0))
        + F.when(F.col("is_reasoning") == 1, F.lit(0.25)).otherwise(F.lit(0.0))
        + F.when(F.col("is_agentic") == 1, F.lit(0.30)).otherwise(F.lit(0.0)),
    )
    df = df.withColumn("_reason_density", F.least(F.col("_reason_diff"), F.lit(0.3)))

    # Component 4: Symbol density
    df = df.withColumn(
        "_symbol_density",
        F.when(F.col("is_math") == 1, F.lit(0.20))
        .when(F.col("is_code") == 1, F.lit(0.15))
        .otherwise(F.lit(0.0)),
    )

    # Component 5: Rarity proxy
    df = df.withColumn(
        "_high_value_matches", F.expr(f"regexp_count(lower(text), '{keyword_pattern}')")
    )
    df = df.withColumn(
        "_rarity_proxy", F.least(F.col("_high_value_matches") / 5.0, F.lit(1.0))
    )

    # Component 6: Readability proxy
    df = df.withColumn(
        "_flesch_hardness",
        F.when(
            F.col("flesch_reading_ease").isNotNull(),
            F.greatest(
                F.lit(0.0),
                F.least(
                    F.lit(1.0),
                    (F.lit(100.0) - F.col("flesch_reading_ease")) / F.lit(100.0),
                ),
            ),
        ).otherwise(F.lit(0.0)),
    )

    # Component 7: Metadata
    df = df.withColumn(
        "_meta_score_diff",
        F.when(F.col("final_meta_diff") == "hard", F.lit(0.8))
        .when(F.col("final_meta_diff").isin("expert", "advanced"), F.lit(0.95))
        .when(F.col("final_meta_diff") == "medium", F.lit(0.45))
        .when(F.col("final_meta_diff") == "easy", F.lit(0.2))
        .otherwise(F.lit(None)),
    )

    df = df.withColumn("_grade_val", F.col("final_meta_grade").cast("int"))
    df = df.withColumn(
        "_meta_score_grade",
        F.when(F.col("_grade_val") >= 11, F.lit(0.8))
        .when(F.col("_grade_val") >= 9, F.lit(0.6))
        .when(F.col("_grade_val") >= 5, F.lit(0.4))
        .when(F.col("_grade_val") > 0, F.lit(0.2))
        .otherwise(F.lit(None)),
    )

    df = df.withColumn(
        "_metadata_base",
        F.greatest(F.col("_meta_score_diff"), F.col("_meta_score_grade")),
    )

    # Weighted Sum
    df = df.withColumn(
        "_heuristic_score",
        F.greatest(
            F.lit(0.0),
            F.least(
                F.lit(0.15) * F.col("_norm_length")
                + F.lit(0.20) * F.col("_struct_density")
                + F.lit(0.25) * F.col("_reason_density")
                + F.lit(0.15) * F.col("_symbol_density")
                + F.lit(0.15) * F.col("_rarity_proxy")
                + F.lit(0.10) * F.col("_flesch_hardness"),
                F.lit(1.0),
            ),
        ),
    )

    # Final: Metadata carries 70% weight if available
    df = df.withColumn(
        "difficulty_score",
        F.when(
            F.col("_metadata_base").isNotNull(),
            (
                F.lit(0.7) * F.col("_metadata_base")
                + F.lit(0.3) * F.col("_heuristic_score")
            ),
        ).otherwise(F.col("_heuristic_score")),
    )

    # Cleanup
    df = df.drop(
        "_norm_length",
        "_struct_density",
        "_reason_diff",
        "_reason_density",
        "_symbol_density",
        "_high_value_matches",
        "_rarity_proxy",
        "_meta_score_diff",
        "_grade_val",
        "_meta_score_grade",
        "_flesch_hardness",
        "_metadata_base",
        "_heuristic_score",
        "flesch_reading_ease",
    )
    return df


def assign_curriculum_band_probabilistic(df):
    """V5 Probabilistic Banding."""
    logger.info("Assigning V5 Probabilistic Bands...")

    # Base Weights
    for band, center in BAND_CENTERS.items():
        df = df.withColumn(
            f"_w_{band}",
            F.greatest(
                F.lit(0.0),
                F.lit(1.0)
                - F.abs(F.col("difficulty_score") - F.lit(center)) / F.lit(WIDTH),
            ),
        )

    # Content Nudges
    df = df.withColumn(
        "_w_B3",
        F.when(F.col("code_score") >= 6, F.col("_w_B3") + 0.05).otherwise(
            F.col("_w_B3")
        ),
    )
    df = df.withColumn(
        "_w_B4",
        F.when(F.col("code_score") >= 10, F.col("_w_B4") + 0.10).otherwise(
            F.col("_w_B4")
        ),
    )
    df = df.withColumn(
        "_w_B4",
        F.when(F.col("agentic_score") >= 5, F.col("_w_B4") + 0.10).otherwise(
            F.col("_w_B4")
        ),
    )
    df = df.withColumn(
        "_w_B5",
        F.when(F.col("agentic_score") >= 7, F.col("_w_B5") + 0.15).otherwise(
            F.col("_w_B5")
        ),
    )
    df = df.withColumn(
        "_w_B4",
        F.when(F.col("reasoning_score") >= 6, F.col("_w_B4") + 0.08).otherwise(
            F.col("_w_B4")
        ),
    )
    df = df.withColumn(
        "_w_B5",
        F.when(F.col("cot_score") >= 5, F.col("_w_B5") + 0.12).otherwise(
            F.col("_w_B5")
        ),
    )
    df = df.withColumn(
        "_w_B3",
        F.when(F.col("math_score") >= 4, F.col("_w_B3") + 0.05).otherwise(
            F.col("_w_B3")
        ),
    )
    df = df.withColumn(
        "_w_B4",
        F.when(F.col("math_score") >= 8, F.col("_w_B4") + 0.08).otherwise(
            F.col("_w_B4")
        ),
    )

    # Normalize
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

    # Final Band
    df = df.withColumn(
        "band",
        F.when(F.col(f"band_p_{BANDS[0]}") >= EPS, BANDS[0])
        .when(F.col(f"band_p_{BANDS[1]}") >= EPS, BANDS[1])
        .when(F.col(f"band_p_{BANDS[2]}") >= EPS, BANDS[2])
        .when(F.col(f"band_p_{BANDS[3]}") >= EPS, BANDS[3])
        .when(F.col(f"band_p_{BANDS[4]}") >= EPS, BANDS[4])
        .otherwise(BANDS[5]),
    )

    df = df.withColumn("assigned_band", F.col("band"))

    # Cleanup
    for band in BANDS:
        df = df.drop(f"_w_{band}")
    df = df.drop("_total_weight")

    return df


# =========================================================================
# OUTPUT PREPARATION
# =========================================================================


def prepare_output_columns(df, include_rejection=False):
    """Prepare columns for output."""
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
    v5_score_cols = [
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
    stage1_2_cols = [
        "byte_length",
        "word_count",
        "unique_token_ratio",
        "compression_ratio",
        "token_count_estimate",
        "fertility_estimate",
    ]
    rejection_cols = ["is_rejected", "rejection_reason", "rejection_level"]

    select_cols = core_cols + band_cols + v5_score_cols + stage1_2_cols
    if include_rejection:
        select_cols += rejection_cols

    existing_cols = [col for col in select_cols if col in df.columns]
    return df.select(*existing_cols)


def clean_dolma_arxiv_spark(df):
    """Optimized ArXiv structural cleaner."""
    logger.info("Applying ArXiv LaTeX cleaning...")

    ref_pattern = r"(?s)(.*)\n(\\section\*?\{References\}|\\section\*?\{Bibliography\}|\\begin\{thebibliography\}|\nReferences\n|\nBibliography\n)"
    df = df.withColumn("text", F.regexp_replace(F.col("text"), ref_pattern, "$1"))

    remove_pattern = (
        r"(?s:\\begin\{(figure|tikzpicture)\}.*?\\end\{\1\})|"
        r"\\(section|subsection|subsubsection|label|caption)\*?\{.*?\}|"
        r"\\cite\{.*?\}"
    )
    df = df.withColumn("text", F.regexp_replace(F.col("text"), remove_pattern, ""))

    return df


# =========================================================================
# MAIN
# =========================================================================


def main():
    args = parse_args()

    source_filter = args.SOURCE
    input_base = args.INPUT_BASE
    output_base = args.OUTPUT_BASE
    checkpoint_base = args.CHECKPOINT_BASE
    estimated_size_gb = args.ESTIMATED_SIZE_GB

    input_path = f"{input_base}/source={source_filter}" if source_filter else input_base
    rejections_path = (
        f"{output_base}/source={source_filter}/rejections"
        if source_filter
        else f"{output_base}/rejections"
    )
    bands_path = (
        f"{output_base}/source={source_filter}/bands"
        if source_filter
        else f"{output_base}/bands"
    )
    report_base_path = (
        f"{REPORT_BASE}/source={source_filter}" if source_filter else REPORT_BASE
    )

    # Track overall timing
    import time

    job_start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"T2 Metrics Calculator V5 - EMR Version {VERSION}")
    logger.info("=" * 60)
    logger.info(f"Job Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Source: {source_filter}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_base}")
    logger.info(f"Estimated Size: {estimated_size_gb} GB")
    logger.info("=" * 60)

    # Create optimized Spark session
    spark = create_optimized_spark_session(estimated_size_gb)

    # Set checkpoint directory
    job_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{checkpoint_base}/{source_filter}/{job_run_id}"
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Keyword pattern for difficulty scoring
    keyword_pattern_str = "\\b(" + "|".join(HIGH_VALUE_KEYWORDS) + ")\\b"

    # =========================================================================
    # READ DATA
    # =========================================================================
    logger.info("=" * 40)
    logger.info("STAGE: DATA LOADING")
    logger.info("=" * 40)
    logger.info(f"Reading from: {input_path}")

    df = spark.read.parquet(input_path).select(
        "id", "text", "source", "domain", "hash", "language", "metadata"
    )

    # Cache and count for progress tracking
    initial_count = df.count()
    logger.info(f"[LOADED] Total records: {initial_count:,}")
    logger.info(f"[LOADED] Input partitions: {df.rdd.getNumPartitions()}")

    # Optimize partitions based on data size
    df = optimize_partitions(df, spark, estimated_size_gb)
    logger.info(
        f"[OPTIMIZED] Partitions after optimization: {df.rdd.getNumPartitions()}"
    )

    # Add tracking metadata
    df = add_uuid_and_metadata(df, input_base)

    # ArXiv cleaning
    if source_filter and "arxiv" in source_filter.lower():
        logger.info("[CLEANING] Applying ArXiv LaTeX cleaning...")
        df = clean_dolma_arxiv_spark(df)

    # =========================================================================
    # STAGE 1 & 2: REJECTION
    # =========================================================================
    logger.info("=" * 40)
    logger.info("STAGE 1: PHYSICAL CORRUPTION CHECK")
    logger.info("=" * 40)

    # Stage 1
    df_s1 = compute_stage1_metrics(df)
    rejected_stage1, passed_stage1 = apply_stage1_rejection(df_s1)

    # Count and log Stage 1 results
    rejected_s1_count = rejected_stage1.count()
    passed_s1_count = passed_stage1.count()
    logger.info("[STAGE 1 COMPLETE]")
    logger.info(f"  - Rejected: {rejected_s1_count:,} records")
    logger.info(f"  - Passed: {passed_s1_count:,} records")
    logger.info(
        f"  - Pass rate: {(passed_s1_count / max(rejected_s1_count + passed_s1_count, 1)) * 100:.2f}%"
    )

    # Show rejection reasons breakdown
    if rejected_s1_count > 0:
        logger.info("[STAGE 1 REJECTION BREAKDOWN]")
        rejection_breakdown = (
            rejected_stage1.groupBy("rejection_reason").count().collect()
        )
        for row in rejection_breakdown:
            logger.info(f"  - {row['rejection_reason']}: {row['count']:,}")

    logger.info("=" * 40)
    logger.info("STAGE 2: NOISE & SPAM DETECTION")
    logger.info("=" * 40)

    # Stage 2
    df_s2 = compute_stage2_metrics(passed_stage1)
    rejected_stage2, passed_stage2 = apply_stage2_rejection(df_s2)

    # Count and log Stage 2 results
    rejected_s2_count = rejected_stage2.count()
    passed_s2_count = passed_stage2.count()
    logger.info("[STAGE 2 COMPLETE]")
    logger.info(f"  - Rejected: {rejected_s2_count:,} records")
    logger.info(f"  - Passed: {passed_s2_count:,} records")
    logger.info(
        f"  - Pass rate: {(passed_s2_count / max(rejected_s2_count + passed_s2_count, 1)) * 100:.2f}%"
    )

    # Show rejection reasons breakdown
    if rejected_s2_count > 0:
        logger.info("[STAGE 2 REJECTION BREAKDOWN]")
        rejection_breakdown = (
            rejected_stage2.groupBy("rejection_reason").count().collect()
        )
        for row in rejection_breakdown:
            logger.info(f"  - {row['rejection_reason']}: {row['count']:,}")

    # Total rejection summary
    total_rejected = rejected_s1_count + rejected_s2_count
    logger.info("=" * 40)
    logger.info("[REJECTION SUMMARY]")
    logger.info(f"  - Total input: {initial_count:,}")
    logger.info(f"  - Total rejected: {total_rejected:,}")
    logger.info(f"  - Proceeding to scoring: {passed_s2_count:,}")
    logger.info(
        f"  - Overall pass rate: {(passed_s2_count / max(initial_count, 1)) * 100:.2f}%"
    )
    logger.info("=" * 40)

    # Consolidate rejections
    rejected_stage1 = prepare_output_columns(rejected_stage1, include_rejection=True)
    rejected_stage2 = prepare_output_columns(rejected_stage2, include_rejection=True)
    rejected_all = rejected_stage1.unionByName(
        rejected_stage2, allowMissingColumns=True
    )

    # Write rejection stats
    logger.info(f"Writing rejection statistics to {report_base_path}/rejections")
    agg_df = (
        rejected_all.select("source", "token_count_estimate")
        .groupBy("source")
        .agg(
            F.sum("token_count_estimate").alias("total_tokens_estimated"),
            F.count("*").alias("record_count"),
        )
    )
    agg_df.write.mode("overwrite").csv(f"{report_base_path}/rejections", header=True)

    # Write rejections
    logger.info(f"Writing rejections to {rejections_path}...")
    rejected_all.write.mode("overwrite").option("compression", "zstd").parquet(
        rejections_path
    )

    # =========================================================================
    # CHECKPOINT (Break Lineage)
    # =========================================================================
    logger.info("Checkpointing passed data (breaking lineage)...")
    passed_stage2 = passed_stage2.checkpoint()

    # Log partition info for monitoring
    num_partitions = passed_stage2.rdd.getNumPartitions()
    logger.info(f"Processing with {num_partitions} partitions")

    # =========================================================================
    # STAGE 3: SCORING
    # =========================================================================
    logger.info("=" * 40)
    logger.info("STAGE 3: MODALITY SCORING & BANDING")
    logger.info("=" * 40)

    logger.info(
        "[SCORING] Computing modality scores (Agentic, CoT, Reasoning, Code, Math)..."
    )
    df_s3 = compute_stage3_metrics(passed_stage2)

    logger.info("[SCORING] Computing difficulty scores...")
    df_s3 = compute_difficulty_score(df_s3, keyword_pattern_str)

    logger.info("[SCORING] Assigning probabilistic bands...")
    df_s3 = assign_curriculum_band_probabilistic(df_s3)

    # Log band distribution
    logger.info("=" * 40)
    logger.info("[BAND DISTRIBUTION]")
    band_counts = (
        df_s3.groupBy("assigned_band").count().orderBy("assigned_band").collect()
    )
    total_banded = sum(row["count"] for row in band_counts)
    for row in band_counts:
        pct = (row["count"] / max(total_banded, 1)) * 100
        bar = "█" * int(pct / 2)  # Visual bar
        logger.info(f"  {row['assigned_band']}: {row['count']:>8,} ({pct:5.1f}%) {bar}")

    # Log modality flags distribution
    logger.info("=" * 40)
    logger.info("[MODALITY FLAGS]")
    modality_stats = df_s3.agg(
        F.sum(F.when(F.col("has_code"), 1).otherwise(0)).alias("has_code"),
        (
            F.sum(F.when(F.col("has_math"), 1).otherwise(0)).alias("has_math")
            if "has_math" in df_s3.columns
            else F.lit(0).alias("has_math")
        ),
        F.sum(F.when(F.col("has_reasoning"), 1).otherwise(0)).alias("has_reasoning"),
        F.sum(F.when(F.col("has_cot"), 1).otherwise(0)).alias("has_cot"),
        F.sum(F.when(F.col("has_agentic"), 1).otherwise(0)).alias("has_agentic"),
    ).collect()[0]
    logger.info(f"  - has_code: {modality_stats['has_code']:,}")
    logger.info(f"  - has_reasoning: {modality_stats['has_reasoning']:,}")
    logger.info(f"  - has_cot: {modality_stats['has_cot']:,}")
    logger.info(f"  - has_agentic: {modality_stats['has_agentic']:,}")
    logger.info("=" * 40)

    # =========================================================================
    # WRITE OUTPUTS
    # =========================================================================
    logger.info("=" * 40)
    logger.info("WRITING OUTPUTS")
    logger.info("=" * 40)

    bands_out_df = prepare_output_columns(df_s3, include_rejection=False)

    # Band statistics
    write_start = time.time()
    logger.info(f"[WRITING] Band statistics to {report_base_path}/bands")
    agg_df = (
        bands_out_df.select("assigned_band", "source", "token_count_estimate")
        .groupBy("assigned_band", "source")
        .agg(
            F.sum("token_count_estimate").alias("total_tokens_estimated"),
            F.count("*").alias("record_count"),
        )
    )
    agg_df.write.mode("overwrite").csv(f"{report_base_path}/bands", header=True)
    logger.info(
        f"[WRITING] Band statistics complete ({time.time() - write_start:.1f}s)"
    )

    # Write bands
    write_start = time.time()
    logger.info(f"[WRITING] Band parquet files to {bands_path}")
    bands_out_df.write.mode("overwrite").partitionBy("band").option(
        "compression", "zstd"
    ).parquet(bands_path)
    logger.info(
        f"[WRITING] Band parquet files complete ({time.time() - write_start:.1f}s)"
    )

    # =========================================================================
    # COMPLETION
    # =========================================================================
    job_end_time = time.time()
    total_runtime = job_end_time - job_start_time
    minutes = int(total_runtime // 60)
    seconds = int(total_runtime % 60)

    logger.info("=" * 60)
    logger.info("JOB COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(
        f"[TIMING] Total runtime: {minutes}m {seconds}s ({total_runtime:.1f} seconds)"
    )
    logger.info(f"[TIMING] Records processed: {initial_count:,}")
    logger.info(
        f"[TIMING] Throughput: {initial_count / max(total_runtime, 1):.1f} records/second"
    )
    logger.info("")
    logger.info("[OUTPUT LOCATIONS]")
    logger.info(f"  - Bands: {bands_path}")
    logger.info(f"  - Rejections: {rejections_path}")
    logger.info(f"  - Statistics: {report_base_path}")
    logger.info("")
    logger.info("[FINAL SUMMARY]")
    logger.info(f"  - Input records: {initial_count:,}")
    logger.info(f"  - Rejected: {total_rejected:,}")
    logger.info(f"  - Banded: {passed_s2_count:,}")
    logger.info("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
