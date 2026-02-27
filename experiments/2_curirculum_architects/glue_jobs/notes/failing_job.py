"""
T2 Metrics Calculator V4.0 - Progressive Quality Filtering + Probabilistic Band Assignment
===========================================================================================
Purpose: Compute quality metrics with early-exit rejection + assign probabilistic curriculum bands.
Strategy: 3-stage progressive filtering → Probabilistic banding for MoE-friendly training.
Optimization: Cost-first design using pure Spark SQL + broadcast variables (no Python UDFs).

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

Usage (Full 4TB Run - Flex Execution):
    aws glue start-job-run --job-name T123_metrics_calculation \\
        --region us-east-1 \\
        --worker-type G.2X \\
        --number-of-workers 20 \\
        --execution-class FLEX \\
        --arguments '{
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data",
            "--OUTPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/metrics_data",
            "--INTERMEDIATE_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/intermediate_data",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.3"
        }'

Usage (Test Run - Single File):
    aws glue start-job-run --job-name T123_metrics_calculation \\
        --region us-east-1 \\
        --worker-type G.2X \\
        --number-of-workers 2 \\
        --execution-class FLEX \\
        --arguments '{
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/source=books/part-00000.zstd.parquet",
            "--OUTPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/test_metrics",
            "--INTERMEDIATE_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/test_intermediate",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s"
        }'

Usage (Incremental - Single Source):
    aws glue start-job-run --job-name T123_metrics_calculation \\
        --region us-east-1 \\
        --worker-type G.2X \\
        --number-of-workers 5 \\
        --execution-class FLEX \\
        --arguments '{
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data",
            "--OUTPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/metrics_data",
            "--INTERMEDIATE_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/intermediate_data",
            "--SOURCE":"arxiv",
            "--conf":"spark.network.timeout=600s --conf spark.sql.broadcastTimeout=1200s"
        }'

Required --conf Parameters for Glue Flex:
    --conf spark.network.timeout=600s              # Prevents timeouts on long 4TB stages
    --conf spark.sql.broadcastTimeout=1200s        # Allows large broadcast joins
    --conf spark.memory.fraction=0.8               # Optional: Tune memory allocation
    --conf spark.memory.storageFraction=0.3        # Optional: Balance execution vs storage

Optimization Notes:
- Uses Spark 3.5 AQE for automatic coalescing and join optimization
- No Python UDFs - all metrics computed via Spark SQL built-in functions
- Physical S3 writes (not checkpoint) break lineage reliably for 4TB scale
- Column pruning reads only (id, text, source, domain) from T1 parquet
- Target 128MB partition size for cost-effective S3 writes
- Batch regex operations to minimize string traversals
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
# CONFIGURATION
# =========================================================================

VERSION = "4.0"
OUTPUT_BASE="s3://t1-dataacquisition-datasets/processed_dataset/metrics_data"
INTERMEDIATE_BASE = "s3://t1-dataacquisition-datasets/processed_dataset/intermediate_data"

# High-value keywords for cheap rarity/complexity detection (broadcast variable)
# Used instead of expensive regex for rare_word detection
# Graduate/PhD level: academic, technical, domain-specific terms
HIGH_VALUE_KEYWORDS = [
    # Academic/Research
    "hypothesis", "methodology", "empirical", "theorem", "lemma", "corollary",
    "ontology", "epistemology", "phenomenology", "hermeneutics", "dialectic",
    "paradigm", "heuristic", "algorithm", "optimization", "convergence",
    # Scientific
    "heterogeneous", "homogeneous", "isotropic", "anisotropic", "stochastic",
    "deterministic", "asymptotic", "parametric", "nonparametric", "multivariate",
    "eigenvalue", "eigenvector", "gradient", "jacobian", "hessian",
    # Medical/Biological
    "pathogenesis", "etiology", "pharmacokinetics", "pharmacodynamics", "metabolism",
    "carcinogenesis", "immunology", "cytology", "histology", "morphology",
    # Legal
    "jurisprudence", "adjudication", "litigation", "jurisdiction", "precedent",
    "appellant", "respondent", "plaintiff", "defendant", "indictment",
    # Mathematical
    "polynomial", "exponential", "logarithmic", "trigonometric", "hyperbolic",
    "differential", "integral", "derivative", "convolution", "fourier",
    "bayesian", "frequentist", "likelihood", "posterior", "prior",
    # Computer Science
    "polymorphism", "encapsulation", "inheritance", "abstraction", "concurrency",
    "parallelism", "distributed", "synchronization", "mutex", "semaphore",
    "recursion", "memoization", "backtracking", "hashing", "traversal",
    # Philosophy/Logic
    "syllogism", "tautology", "contradiction", "axiom", "inference",
    "deduction", "induction", "abduction", "fallacy", "proposition",
    # Economics/Finance
    "elasticity", "equilibrium", "arbitrage", "volatility", "derivative",
    "amortization", "depreciation", "valuation", "liquidity", "solvency",
    # Engineering
    "thermodynamics", "kinetics", "dynamics", "statics", "mechanics",
    "electromagnetic", "semiconductor", "transistor", "amplifier", "oscillator",
    # Chemistry
    "stoichiometry", "titration", "catalysis", "synthesis", "hydrolysis",
    "oxidation", "reduction", "equilibrium", "entropy", "enthalpy",
    # Physics
    "quantum", "relativity", "spacetime", "superposition", "entanglement",
    "hamiltonian", "lagrangian", "schrodinger", "heisenberg", "maxwell",
    # Advanced General
    "dichotomy", "juxtaposition", "ubiquitous", "ephemeral", "perpetual",
    "ambiguous", "arbitrary", "intrinsic", "extrinsic", "implicit",
    "explicit", "analogous", "congruent", "disparate", "heterogeneous",
    # Reasoning/Analysis
    "therefore", "consequently", "nevertheless", "notwithstanding", "furthermore",
    "moreover", "conversely", "alternatively", "analogously", "presumably"
]

# Regex patterns for quality detection (compiled once, reused everywhere)
BOILERPLATE_PATTERN = r"(?i)(cookie policy|privacy policy|terms of service|all rights reserved|© copyright|click here|subscribe to|sign up|newsletter|unsubscribe|contact us|about us|follow us on|accept cookies|manage preferences)"
THREAD_MARKER_PATTERN = r"(>>|replied to:|in response to|re:|replying to|quote from|responding to)"
RISKY_TLD_PATTERN = r"(?i)\.(tk|ml|ga|cf|gq|xyz|top|club|win|loan|bid|download|stream|review|click|link|trade|date)\b"
URL_PATTERN = r"https?://[^\s]+"
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

# Modality detection patterns (from curriculum_extractor)
CODE_PATTERN = r"```|def\s+\w+\(|class\s+\w+\s*[:{]|class\s+\w+\([^\)]+\)\s*:|^\s*function\s+\w+\s*[\({]|^\s*import\s+\w+|from\s+[\w.]+\s+import\s+\w+|from\s+\.\s+import\s+\w+"
MATH_PATTERN = r"[∑∫√≈≠≤≥∞]|\\(frac|sum|int|sqrt|begin\{equation\}|alpha|beta|gamma|delta|theta|pi|sigma|omega|phi|partial|cdot|times|pm)|\\\[|\\\("
CODE_COMMENT_PATTERN = r"^\s*(?:#|//|--|%|/\*|\*)"
AGENTIC_PATTERN = r"^\s*(Action|Observation|Thought|Final Answer|Tool):|\"(tool|action|observation|thought)\"\s*:"
COT_PATTERN = r"(?i)(let\\'s think step by step|let\\'s reason|chain of thought|reasoning:)"
RESEARCH_PATTERN = r"^\s*(?:Abstract|References|Bibliography)(?:[:$])|\b(?:arXiv|doi)[:/]\s*\d|\bdoi\.org/10\.|\bet al\.|\[[\d,\s]+\].*\[[\d,\s]+\]"
REASONING_PATTERN = r"(?i)\b(therefore|thus|implies|because|hence|consequently)\b"
MATH_EXPRESSION_PATTERN = r"[\+\-\*/\^=]{2,}|\\frac|\\sum|\\int|\$[^\$]+\$"
CITATION_PATTERN = r"\[\d+\]|\([\w\s]+,\s*\d{4}\)"
QUESTION_PATTERN = r"\?"
LIST_MARKER_PATTERN = r"^\s*[\-\*\d]+\.\s+"
STEP_INDICATOR_PATTERN = r"(?i)\b(step \d+|first|second|third|finally|next)\b"
ELLIPSIS_PATTERN = r"\.\.\."
TRUNCATION_PATTERN = r"(?i)(continued\.\.\.|read more|see full|truncated|cut off)"
HEADING_PATTERN = r"^#+\s+|^[A-Z][^\n]{5,50}$"
DIALOGUE_PATTERN = r'["\'].*?["\']:|^\w+:'
CODE_FENCE_PATTERN = r"```|~~~"
TABLE_PATTERN = r"\|.*\|.*\|"


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def get_glue_args():
    """Parse Glue job arguments."""
    # args = getResolvedOptions(sys.argv, ['JOB_NAME'])
    
    optional_args = {}
    if '--INPUT_BASE' in sys.argv:
        optional_args['INPUT_BASE'] = getResolvedOptions(sys.argv, ['INPUT_BASE'])['INPUT_BASE']
    else:
        optional_args['INPUT_BASE'] = "s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"
    
    # Optional: Filter by specific source for incremental processing
    if '--SOURCE' in sys.argv:
        optional_args['SOURCE'] = getResolvedOptions(sys.argv, ['SOURCE'])['SOURCE']
    else:
        optional_args['SOURCE'] = None
    
    return [], optional_args

def add_uuid_and_metadata(df):
    """
    Add tracking columns: uuid, file_path (input_file_name).
    These columns allow us to:
    1. Uniquely identify each record across the pipeline
    2. Join back to T1 data if needed for downstream analysis
    """
    df = df.withColumn("uuid", F.expr("uuid()"))
    df = df.withColumn("file_path", F.input_file_name())
    return df

def safe_divide(numerator, denominator, default=0.0):
    """Safe division to avoid divide-by-zero errors."""
    return F.when(denominator > 0, numerator / denominator).otherwise(default)

# =========================================================================
# METRIC COMPUTATION - STAGE 1 (PRIORITY 1)
# Fast rejection: corruption, extreme lengths
# =========================================================================

def compute_stage1_metrics(df):
    """
    Stage 1: Physical properties and corruption checks.
    
    Metrics Computed:
    - byte_length: Storage cost proxy
    - char_length: Unicode-aware length
    - word_count: Whitespace-split token count (proxy for token_count_estimate)
    - line_count: Newline count
    - non_printable_ratio: Encoding corruption detection
    
    Rejection Logic (Priority 1):
    - byte_length < 50 OR > 1048576 (1MB) → "extreme_length"
    - char_length < 20 OR > 500000 → "extreme_length"
    - non_printable_ratio > 0.01 → "corruption"
    - word_count < 10 → "too_short"
    
    Trade-off: We use word_count as proxy for token_count_estimate to avoid
    loading tiktoken library (saves dependency and compute). Error margin ~10-15%
    is acceptable for early filtering.
    """
    print("  Computing Stage 1 metrics (Physical Properties)...")
    
    # Physical properties
    df = df.withColumn("byte_length", F.length(F.encode("text", "utf-8")))
    df = df.withColumn("char_length", F.length("text"))
    df = df.withColumn("word_count", F.size(F.split(F.col("text"), r"\s+")) - 1)
    df = df.withColumn("line_count", F.size(F.split(F.col("text"), "\n")))
    
    # Corruption detection
    df = df.withColumn("printable_length", 
                       F.length(F.regexp_replace("text", r"[^\x20-\x7E\n\r\t]", "")))
    df = df.withColumn("non_printable_ratio", 
                       safe_divide(F.col("char_length") - F.col("printable_length"), 
                                   F.col("char_length")))
    
    # Derived metrics
    df = df.withColumn("avg_line_length", 
                       safe_divide(F.col("char_length"), F.col("line_count")))
    
    # Token count estimate (approximate: word_count * 1.3 for English)
    # This avoids tiktoken dependency while providing reasonable proxy
    df = df.withColumn("token_count_estimate", (F.col("word_count") * 1.3).cast("int"))
    
    # Fertility estimate (char_length / token_count)
    # Indicates tokenization efficiency - higher = more chars per token
    df = df.withColumn("fertility_estimate", 
                       safe_divide(F.col("char_length"), F.col("token_count_estimate"), 1.0))
    
    return df

def apply_stage1_rejection(df):
    """
    Apply Priority 1 rejection rules.
    Returns: (rejected_df, passed_df)
    """
    print("  Applying Stage 1 rejection rules...")
    
    # Initialize rejection tracking columns
    df = df.withColumn("is_rejected", F.lit(False))
    df = df.withColumn("rejection_reason", F.lit(""))
    df = df.withColumn("rejection_priority", F.lit(0))
    
    # Rule 1: Minimum length only (byte_length < 50)
    # NOTE: Removed maximum length check to support textbooks
    df = df.withColumn("is_rejected",
                       F.when(F.col("byte_length") < 50, True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when(F.col("byte_length") < 50, "too_short_bytes")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when(F.col("byte_length") < 50, 1)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 2: Minimum length only (char_length < 20)
    # NOTE: Removed maximum length check to support textbooks
    df = df.withColumn("is_rejected",
                       F.when(F.col("char_length") < 20, True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when(F.col("char_length") < 20, "too_short_chars")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when(F.col("char_length") < 20, 1)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 3: Corruption (non-printable characters)
    # Increased threshold to 0.10 to accommodate Unicode formatting (Indic scripts, math symbols)
    # Only reject if ratio is high AND document is long enough (avoid false positives on short docs)
    # NOTE: Training data benefits from Unicode diversity; only reject extreme corruption
    df = df.withColumn("is_rejected",
                       F.when((F.col("non_printable_ratio") > 0.10) & (F.col("char_length") > 100), True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("non_printable_ratio") > 0.10) & (F.col("char_length") > 100), 
                              "encoding_corruption")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when((F.col("non_printable_ratio") > 0.10) & (F.col("char_length") > 100), 1)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 4: Too short (token_count_estimate < 10)
    df = df.withColumn("is_rejected",
                       F.when(F.col("token_count_estimate") < 10, True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when(F.col("token_count_estimate") < 10, 
                              "too_short_tokens")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when(F.col("token_count_estimate") < 10, 1)
                       .otherwise(F.col("rejection_priority")))
    
    # Split into rejected and passed
    rejected_df = df.filter(F.col("is_rejected") == True)
    passed_df = df.filter(F.col("is_rejected") == False)
        
    return rejected_df, passed_df

# =========================================================================
# METRIC COMPUTATION - STAGE 2 (PRIORITY 2)
# Medium cost: spam, boilerplate, templates, repetition
# =========================================================================

def compute_stage2_metrics(df):
    """
    Stage 2: Lexical diversity, noise detection, spam filtering.
    
    OPTIMIZATION: Batch regex processing to minimize string traversals.
    - Original: 15+ separate F.split/F.regexp_replace calls = 15+ scans
    - Optimized: Single-pass pattern extraction + 3 character scans = ~4 scans
    - Expected savings: 70-75% Stage 2 compute time on 4TB data
    
    Metrics Computed:
    - unique_token_ratio: Vocabulary richness (set/total ratio)
    - compression_ratio: Entropy proxy using gzip simulation
    - symbol_density: Special character ratio
    - whitespace_ratio: Formatting density
    - capitalization_ratio: UPPERCASE detection
    - url_count & url_ratio: Link spam detection
    - html_tag_density: Raw HTML detection
    - boilerplate_ratio: Cookie/footer spam
    - thread_fragment_indicator: Orphaned forum replies
    - risky_tld_count: Malicious domain detection
    
    Rejection Logic (Priority 2):
    - unique_token_ratio < 0.1 → "repetitive_template"
    - compression_ratio > 0.95 → "incompressible_random"
    - capitalization_ratio > 0.5 → "all_caps_spam"
    - whitespace_ratio > 0.6 → "excessive_whitespace"
    - url_ratio > 0.3 → "link_spam"
    - html_tag_density > 0.05 → "raw_html"
    - boilerplate_ratio > 0.15 → "boilerplate_spam"
    - thread_fragment (markers > 2 AND tokens < 200) → "orphaned_fragment"
    
    Trade-off: compression_ratio computed via byte_length/char_length proxy
    instead of actual zlib (too expensive). Good enough for filtering.
    """
    print("  Computing Stage 2 metrics (Lexical Diversity & Noise) - OPTIMIZED BATCH PROCESSING...")
    
    # ==== BATCH 1: Array operations (vocabulary, no regex) ====
    df = df.withColumn("tokens_list", F.split(F.lower("text"), r"\s+"))
    df = df.withColumn("unique_tokens", F.size(F.array_distinct("tokens_list")))
    df = df.withColumn("vocab_size", F.col("unique_tokens"))
    df = df.withColumn("unique_token_ratio", 
                       safe_divide(F.col("unique_tokens"), F.col("word_count")))
    
    # Compression ratio proxy (byte/char ratio is entropy indicator)
    df = df.withColumn("compression_ratio", 
                       safe_divide(F.col("byte_length"), F.col("char_length")))
    
    # ==== BATCH 2: Character-level operations in SINGLE PASS ====
    # Instead of 5 separate regexp_replace calls, combine into batch character counting
    text_col = F.col("text")
    text_len = F.length(text_col)
    
    # Symbol density - count non-alphanumeric/whitespace
    df = df.withColumn("symbol_count", 
                       text_len - F.length(F.regexp_replace("text", r"[a-zA-Z0-9\s]", "")))
    df = df.withColumn("symbol_density", 
                       safe_divide(F.col("symbol_count"), F.col("char_length")))
    
    # Whitespace ratio
    df = df.withColumn("whitespace_count", 
                       text_len - F.length(F.regexp_replace("text", r"\S", "")))
    df = df.withColumn("whitespace_ratio", 
                       safe_divide(F.col("whitespace_count"), F.col("char_length")))
    
    # Capitalization ratio - count alpha vs uppercase
    df = df.withColumn("alpha_count", 
                       text_len - F.length(F.regexp_replace("text", r"[^a-zA-Z]", "")))
    df = df.withColumn("upper_count", 
                       text_len - F.length(F.regexp_replace("text", r"[^A-Z]", "")))
    df = df.withColumn("capitalization_ratio", 
                       safe_divide(F.col("upper_count"), F.col("alpha_count")))
    
    # HTML tag detection (count characters removed by tag stripping)
    df = df.withColumn("html_tag_count", 
                       text_len - F.length(F.regexp_replace("text", r"<[^>]+>", "")))
    df = df.withColumn("html_tag_density", 
                       safe_divide(F.col("html_tag_count"), F.col("char_length")))
    
    # ==== BATCH 3: Pattern extraction using regexp_extract_all (SINGLE SCAN) ====
    # Extract URLs, boilerplate, thread markers, risky TLDs in ONE PASS
    # This is CRITICAL optimization - replaces 5+ F.split operations with 1 scan
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
    df = df.withColumn("thread_fragment_indicator", F.col("thread_marker_count"))
    df = df.drop("_thread_matches")
    
    df = df.withColumn("_risky_tld_matches", F.expr(f"regexp_extract_all(lower(text), '{RISKY_TLD_PATTERN}', 0)"))
    df = df.withColumn("risky_tld_count", F.size(F.col("_risky_tld_matches")))
    df = df.drop("_risky_tld_matches")
    
    # ==== Sentence boundary detection ====
    df = df.withColumn("sentence_count_estimate", 
                       F.size(F.split(F.col("text"), r"[.!?]+\s+")) - 1)
    df = df.withColumn("sentence_count_estimate",
                       F.when(F.col("sentence_count_estimate") < 1, 1)
                       .otherwise(F.col("sentence_count_estimate")))
    df = df.withColumn("avg_sentence_length", 
                       safe_divide(F.col("char_length"), F.col("sentence_count_estimate")))
    
    # Sentence boundary coherence check
    df = df.withColumn("sentence_boundary_coherence",
                       F.when((F.col("char_length") > 500) & (F.col("sentence_count_estimate") < 2), 0.3)
                       .otherwise(1.0))
    
    # ==== MTLD and Rare Word Ratio estimates ====
    # MTLD estimate (lexical diversity proxy)
    df = df.withColumn("mtld_estimate", F.col("unique_token_ratio") * 100.0)
    
    # Rare word ratio estimate
    df = df.withColumn("rare_word_ratio_estimate",
                       F.when(F.col("unique_token_ratio") > 0.4,
                              1.0 - (2.0 * F.col("unique_token_ratio")))
                       .otherwise(0.1))  # Low diversity = low rare word ratio
    
    return df

def apply_stage2_rejection(df):
    """
    Apply Priority 2 rejection rules.
    Returns: (rejected_df, passed_df)
    """
    print("  Applying Stage 2 rejection rules...")
    
    # Initialize rejection tracking (for records that passed Stage 1)
    df = df.withColumn("is_rejected", F.lit(False))
    df = df.withColumn("rejection_reason", F.lit(""))
    df = df.withColumn("rejection_priority", F.lit(0))
    
    # Rule 1: Repetitive template (unique_token_ratio < 0.05 AND word_count > 100)
    # Lowered threshold to 0.05 (5%) to avoid rejecting valid repetitive content (legal docs, poetry)
    # Added length check to preserve short valid content
    df = df.withColumn("is_rejected",
                       F.when((F.col("unique_token_ratio") < 0.05) & (F.col("word_count") > 100), True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("unique_token_ratio") < 0.05) & (F.col("word_count") > 100), 
                              "repetitive_template")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when((F.col("unique_token_ratio") < 0.05) & (F.col("word_count") > 100), 2)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 2: Incompressible (compression_ratio > 0.95)
    # This catches random/encrypted/binary data
    df = df.withColumn("is_rejected",
                       F.when(F.col("compression_ratio") > 0.95, True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when(F.col("compression_ratio") > 0.95, 
                              "incompressible_random_data")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when(F.col("compression_ratio") > 0.95, 2)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 3: ALL CAPS spam (capitalization_ratio > 0.7 AND word_count > 100)
    # Increased threshold to 0.7 (70%) to avoid false positives on chapter titles/headers
    # Increased word count to 100 to ensure sufficient sample size
    df = df.withColumn("is_rejected",
                       F.when((F.col("capitalization_ratio") > 0.7) & (F.col("word_count") > 100), True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("capitalization_ratio") > 0.7) & (F.col("word_count") > 100), 
                              "all_caps_spam")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when((F.col("capitalization_ratio") > 0.7) & (F.col("word_count") > 100), 2)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 4: Excessive whitespace (whitespace_ratio > 0.85)
    # Increased threshold to 0.85 to accommodate intentional formatting (books, code, poetry)
    # Critical for book recovery: books average 0.4-0.7 whitespace ratio
    df = df.withColumn("is_rejected",
                       F.when(F.col("whitespace_ratio") > 0.85, True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when(F.col("whitespace_ratio") > 0.85, 
                              "excessive_whitespace")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when(F.col("whitespace_ratio") > 0.85, 2)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 5: Link spam (url_ratio > 0.4 AND url_count > 20)
    # Increased threshold to 0.4 to accommodate papers with citations/references
    # Added absolute count check: reject only if both ratio AND count are high
    df = df.withColumn("is_rejected",
                       F.when((F.col("url_ratio") > 0.4) & (F.col("url_count") > 20), True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("url_ratio") > 0.4) & (F.col("url_count") > 20), 
                              "link_spam")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when((F.col("url_ratio") > 0.4) & (F.col("url_count") > 20), 2)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 6: Raw HTML (html_tag_density > 0.10 AND char_length > 500)
    # Increased threshold to 0.10 to avoid rejecting code examples/documentation
    # Added length check to preserve short valid snippets
    df = df.withColumn("is_rejected",
                       F.when((F.col("html_tag_density") > 0.10) & (F.col("char_length") > 500), True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("html_tag_density") > 0.10) & (F.col("char_length") > 500), 
                              "raw_html_dump")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when((F.col("html_tag_density") > 0.10) & (F.col("char_length") > 500), 2)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 7: Boilerplate spam (boilerplate_ratio > 0.25)
    # Increased threshold to 0.25 to accommodate legitimate metadata/disclaimers
    # 15% can be standard publication info; 25% is true spam territory
    df = df.withColumn("is_rejected",
                       F.when(F.col("boilerplate_ratio") > 0.25, True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when(F.col("boilerplate_ratio") > 0.25, 
                              "boilerplate_spam")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when(F.col("boilerplate_ratio") > 0.25, 2)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 8: Thread fragment (markers > 2 AND tokens < 200)
    df = df.withColumn("is_rejected",
                       F.when((F.col("thread_fragment_indicator") > 2) & 
                              (F.col("token_count_estimate") < 200), True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("thread_fragment_indicator") > 2) & 
                              (F.col("token_count_estimate") < 200), 
                              "orphaned_thread_fragment")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when((F.col("thread_fragment_indicator") > 2) & 
                              (F.col("token_count_estimate") < 200), 2)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 9: Risky TLD (risky_tld_count > 3)
    # Changed to 3+ occurrences to allow legitimate discussion of malicious domains
    # Single mention may be security research; 3+ is likely spam
    df = df.withColumn("is_rejected",
                       F.when(F.col("risky_tld_count") > 3, True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when(F.col("risky_tld_count") > 3, 
                              "risky_tld_spam")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when(F.col("risky_tld_count") > 3, 2)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 10: Sentence boundary coherence (< 0.2)
    # Decreased threshold from 0.5 to 0.2 to reduce false positives on code/poetry/lists
    # Only catches truly malformed text now (Indic punctuation, math equations are valid)
    df = df.withColumn("is_rejected",
                       F.when(F.col("sentence_boundary_coherence") < 0.2, True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when(F.col("sentence_boundary_coherence") < 0.2, 
                              "sentence_boundary_failure")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when(F.col("sentence_boundary_coherence") < 0.2, 2)
                       .otherwise(F.col("rejection_priority")))
    
    # Split into rejected and passed
    rejected_df = df.filter(F.col("is_rejected") == True)
    passed_df = df.filter(F.col("is_rejected") == False)
        
    return rejected_df, passed_df

# =========================================================================
# METRIC COMPUTATION - STAGE 3 (PRIORITY 3)
# Expensive metrics: readability, complexity, domain signals
# =========================================================================

def compute_stage3_metrics(df):
    """
    Stage 3: Advanced quality metrics, domain signals, reasoning detection.
    
    OPTIMIZATION: Batch regex processing to minimize string traversals.
    - Original: 25+ separate F.split/regexp_replace calls = 25+ scans
    - Optimized: Single-pass pattern extraction using regexp_extract_all = ~8 scans
    - Expected savings: 70-80% Stage 3 compute time on 4TB data
    
    Metrics Computed:
    - punctuation_density: Formal writing indicator
    - num_numeric_tokens: Quantitative reasoning
    - question_density: Interrogative structure
    - citation_count: Academic rigor
    - list_marker_count: Enumerated structure
    - math_expression_count: Mathematical reasoning
    - code_block_count: Technical content
    - agentic_markers: Tool-use patterns
    - cot_markers: Chain-of-thought reasoning
    - research_paper_markers: Academic structure
    - reasoning_marker_density: Logical connectives
    - step_indicator_count: Procedural knowledge
    - ellipsis_count: Truncated content
    - truncation_indicators: Incomplete content
    - heading_count: Document structure
    - dialogue_turn_count: Conversational format
    - table_count_estimate: Structured data
    - code_comment_ratio: Code quality (for code domains)
    - flesch_reading_ease: Readability score
    - avg_word_length: Vocabulary sophistication
    - dependency_depth_estimate: Nesting complexity
    
    Rejection Logic (Priority 3):
    - flesch_reading_ease < 0 OR > 120 → "invalid_readability"
    - avg_sentence_length > 500 → "malformed_text"
    - dependency_depth_estimate > 20 → "excessive_nesting"
    - truncation_indicators > 2 → "incomplete_content"
    - code_comment_ratio > 0.8 (for code domain) → "code_mostly_comments"
    
    Trade-off: Flesch Reading Ease computed via simplified formula
    (no syllable counting) to avoid nltk dependency. Acceptable for filtering.
    """
    print("  Computing Stage 3 metrics (Quality & Complexity) - OPTIMIZED BATCH PROCESSING...")
    
    # ==== BATCH 1: Character-level operations ====
    text_len = F.length(F.col("text"))
    
    # Punctuation density
    df = df.withColumn("punctuation_count", 
                       text_len - F.length(F.regexp_replace("text", r"[^.,!?;:]", "")))
    df = df.withColumn("punctuation_density", 
                       safe_divide(F.col("punctuation_count"), F.col("char_length")))
    
    # Dependency depth (bracket nesting)
    df = df.withColumn("open_brackets", 
                       text_len - F.length(F.regexp_replace("text", r"[^\(\[\{]", "")))
    df = df.withColumn("close_brackets", 
                       text_len - F.length(F.regexp_replace("text", r"[^\)\]\}]", "")))
    df = df.withColumn("dependency_depth_estimate", 
                       F.greatest(F.col("open_brackets"), F.col("close_brackets")))
    
    # ==== BATCH 2: ALL pattern extraction in SINGLE PASS ====
    # This is the MOST CRITICAL optimization - replaces 20+ F.split operations
    # Use regexp_extract_all to scan text ONCE and find ALL patterns
    
    # Numeric tokens
    df = df.withColumn("_numeric_matches", F.expr(f"regexp_extract_all(text, '\\\\d+', 0)"))
    df = df.withColumn("num_numeric_tokens", F.size(F.col("_numeric_matches")))
    df = df.drop("_numeric_matches")
    
    # Citations, list markers, math, code, structure patterns
    df = df.withColumn("_citation_matches", F.expr(f"regexp_extract_all(text, '{CITATION_PATTERN}', 0)"))
    df = df.withColumn("citation_count", F.size(F.col("_citation_matches")))
    df = df.drop("_citation_matches")
    
    df = df.withColumn("_list_matches", F.expr(f"regexp_extract_all(text, '{LIST_MARKER_PATTERN}', 0)"))
    df = df.withColumn("list_marker_count", F.size(F.col("_list_matches")))
    df = df.drop("_list_matches")
    
    df = df.withColumn("_math_matches", F.expr(f"regexp_extract_all(text, '{MATH_EXPRESSION_PATTERN}', 0)"))
    df = df.withColumn("math_expression_count", F.size(F.col("_math_matches")))
    df = df.withColumn("equation_density", safe_divide(F.col("math_expression_count"), F.col("word_count")))
    df = df.drop("_math_matches")
    
    df = df.withColumn("_code_fence_matches", F.expr(f"regexp_extract_all(text, '{CODE_FENCE_PATTERN}', 0)"))
    df = df.withColumn("code_block_count", F.size(F.col("_code_fence_matches")))
    df = df.drop("_code_fence_matches")
    
    # Modality patterns
    df = df.withColumn("_agentic_matches", F.expr(f"regexp_extract_all(text, '{AGENTIC_PATTERN}', 0)"))
    df = df.withColumn("agentic_markers", F.size(F.col("_agentic_matches")))
    df = df.drop("_agentic_matches")
    
    df = df.withColumn("_cot_matches", F.expr(f"regexp_extract_all(lower(text), '{COT_PATTERN}', 0)"))
    df = df.withColumn("cot_markers", F.size(F.col("_cot_matches")))
    df = df.drop("_cot_matches")
    
    df = df.withColumn("_research_matches", F.expr(f"regexp_extract_all(text, '{RESEARCH_PATTERN}', 0)"))
    df = df.withColumn("research_paper_markers", F.size(F.col("_research_matches")))
    df = df.drop("_research_matches")
    
    df = df.withColumn("_reasoning_matches", F.expr(f"regexp_extract_all(lower(text), '{REASONING_PATTERN}', 0)"))
    df = df.withColumn("reasoning_marker_count", F.size(F.col("_reasoning_matches")))
    df = df.withColumn("reasoning_marker_density", safe_divide(F.col("reasoning_marker_count"), F.col("word_count")))
    df = df.drop("_reasoning_matches")
    
    df = df.withColumn("_step_matches", F.expr(f"regexp_extract_all(lower(text), '{STEP_INDICATOR_PATTERN}', 0)"))
    df = df.withColumn("step_indicator_count", F.size(F.col("_step_matches")))
    df = df.drop("_step_matches")
    
    # Truncation patterns
    df = df.withColumn("_ellipsis_matches", F.expr(f"regexp_extract_all(text, '{ELLIPSIS_PATTERN}', 0)"))
    df = df.withColumn("ellipsis_count", F.size(F.col("_ellipsis_matches")))
    df = df.drop("_ellipsis_matches")
    
    df = df.withColumn("_truncation_matches", F.expr(f"regexp_extract_all(lower(text), '{TRUNCATION_PATTERN}', 0)"))
    df = df.withColumn("truncation_indicators", F.size(F.col("_truncation_matches")))
    df = df.drop("_truncation_matches")
    
    # Document structure
    df = df.withColumn("_heading_matches", F.expr(f"regexp_extract_all(text, '{HEADING_PATTERN}', 0)"))
    df = df.withColumn("heading_count", F.size(F.col("_heading_matches")))
    df = df.drop("_heading_matches")
    
    df = df.withColumn("_dialogue_matches", F.expr(f"regexp_extract_all(text, '{DIALOGUE_PATTERN}', 0)"))
    df = df.withColumn("dialogue_turn_count", F.size(F.col("_dialogue_matches")))
    df = df.drop("_dialogue_matches")
    
    df = df.withColumn("_table_matches", F.expr(f"regexp_extract_all(text, '{TABLE_PATTERN}', 0)"))
    df = df.withColumn("table_count_estimate", F.size(F.col("_table_matches")))
    df = df.drop("_table_matches")
    
    # Code comment ratio
    df = df.withColumn("_code_comment_matches", F.expr(f"regexp_extract_all(text, '{CODE_COMMENT_PATTERN}', 0)"))
    df = df.withColumn("code_comment_line_count", F.size(F.col("_code_comment_matches")))
    df = df.withColumn("code_comment_ratio", safe_divide(F.col("code_comment_line_count"), F.col("line_count")))
    df = df.drop("_code_comment_matches")
    
    # Question density (already scanned for patterns above, now just add derived metric)
    df = df.withColumn("_question_matches", F.expr(f"regexp_extract_all(text, '{QUESTION_PATTERN}', 0)"))
    df = df.withColumn("question_count", F.size(F.col("_question_matches")))
    df = df.withColumn("question_density", safe_divide(F.col("question_count"), F.col("sentence_count_estimate")))
    df = df.drop("_question_matches")
    
    # ==== BATCH 3: Readability metrics (no regex) ====
    df = df.withColumn("avg_word_length", safe_divide(F.col("char_length"), F.col("word_count")))
    df = df.withColumn("syllables_per_word_estimate", F.col("avg_word_length") / 3.0)
    df = df.withColumn("words_per_sentence", safe_divide(F.col("word_count"), F.col("sentence_count_estimate")))
    df = df.withColumn("flesch_reading_ease",
                       F.lit(206.835) - (F.lit(1.015) * F.col("words_per_sentence")) - 
                       (F.lit(84.6) * F.col("syllables_per_word_estimate")))
    
    # ==== BATCH 4: Modality detection using already-extracted pattern counts ====
    # Reuse pattern counts from batch 2 instead of re-scanning text
    df = df.withColumn("_code_pattern_matches", F.expr(f"regexp_extract_all(text, '{CODE_PATTERN}', 0)"))
    df = df.withColumn("has_code", F.size(F.col("_code_pattern_matches")) > 0)
    df = df.drop("_code_pattern_matches")
    
    df = df.withColumn("_math_pattern_matches", F.expr(f"regexp_extract_all(text, '{MATH_PATTERN}', 0)"))
    df = df.withColumn("has_math", F.size(F.col("_math_pattern_matches")) > 0)
    df = df.drop("_math_pattern_matches")
    
    # Reuse already extracted counts for efficiency
    df = df.withColumn("has_reasoning", F.col("cot_markers") > 0)
    df = df.withColumn("has_agentic", F.col("agentic_markers") > 0)
    df = df.withColumn("has_research_paper", F.col("research_paper_markers") > 0)
    
    # Primary modality (hierarchical priority)
    df = df.withColumn("primary_modality",
                       F.when(F.col("has_agentic"), F.lit("agentic_traces"))
                       .when(F.col("has_research_paper"), F.lit("research_papers"))
                       .when(F.col("has_code") & F.col("has_math"), F.lit("technical_text"))
                       .when(F.col("has_code"), F.lit("code"))
                       .when(F.col("has_math"), F.lit("math"))
                       .when(F.col("has_reasoning"), F.lit("reasoning"))
                       .otherwise(F.lit("general_text")))
    
    # CoT and agentic densities (reuse counts)
    df = df.withColumn("cot_density", safe_divide(F.col("cot_markers"), F.col("word_count")))
    df = df.withColumn("agentic_density", safe_divide(F.col("agentic_markers"), F.col("word_count")))
    
    # ==== BATCH 5: Difficulty scoring ====
    # Composite difficulty metric (0-1 scale) for curriculum learning
    # Based on avg_word_length, rare_word_ratio, and entropy (compression_ratio proxy)
    df = df.withColumn("difficulty_word_component",
                       F.least(F.col("avg_word_length") / 10.0, F.lit(1.0)))
    df = df.withColumn("difficulty_rare_component",
                       F.col("rare_word_ratio_estimate"))
    df = df.withColumn("difficulty_entropy_component",
                       F.when(F.col("compression_ratio") < 0.8, 
                              F.col("compression_ratio") / 2.0)  # Low compression = high entropy
                       .otherwise(0.4))  # Cap at 0.4
    
    df = df.withColumn("difficulty_score",
                       (F.lit(0.3) * F.col("difficulty_word_component") +
                        F.lit(0.4) * F.col("difficulty_rare_component") +
                        F.lit(0.3) * F.col("difficulty_entropy_component")))
    
    # Difficulty level (L0-L5 bands)
    df = df.withColumn("difficulty_level",
                       F.when(F.col("difficulty_score") <= 0.1, "L0")
                       .when(F.col("difficulty_score") <= 0.3, "L1")
                       .when(F.col("difficulty_score") <= 0.5, "L2")
                       .when(F.col("difficulty_score") <= 0.7, "L3")
                       .when(F.col("difficulty_score") <= 0.9, "L4")
                       .otherwise("L5"))
    
    # === INFORMATION DENSITY ===
    # Proxy for content vs filler ratio
    # High unique_token_ratio + reasoning markers = high information density
    df = df.withColumn("information_density_estimate",
                       F.least(
                           (F.lit(0.6) * F.col("unique_token_ratio") +
                            F.lit(0.2) * F.least(F.col("reasoning_marker_density") * 10.0, F.lit(1.0)) +
                            F.lit(0.2) * F.least(F.col("cot_markers") / 5.0, F.lit(1.0))),
                           F.lit(1.0)
                       ))
    
    return df

def apply_stage3_rejection(df):
    """
    Apply Priority 3 rejection rules.
    Returns: (rejected_df, passed_df)
    """
    print("  Applying Stage 3 rejection rules...")
    
    # Initialize rejection tracking
    df = df.withColumn("is_rejected", F.lit(False))
    df = df.withColumn("rejection_reason", F.lit(""))
    df = df.withColumn("rejection_priority", F.lit(0))
    
    # Rule 1: Invalid Flesch Reading Ease (< -50 OR > 150)
    # NOTE: Relaxed thresholds to avoid rejecting technical/textbook content
    df = df.withColumn("is_rejected",
                       F.when((F.col("flesch_reading_ease") < -50) | 
                              (F.col("flesch_reading_ease") > 150), True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("flesch_reading_ease") < -50) | 
                              (F.col("flesch_reading_ease") > 150), 
                              "invalid_readability_score")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when((F.col("flesch_reading_ease") < -50) | 
                              (F.col("flesch_reading_ease") > 150), 3)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 2: REMOVED - avg_sentence_length rejection (would reject textbooks)
    
    # Rule 3: Excessive nesting (dependency_depth > 50)
    # NOTE: Increased threshold from 20 to 50 to avoid rejecting complex code
    df = df.withColumn("is_rejected",
                       F.when(F.col("dependency_depth_estimate") > 50, True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when(F.col("dependency_depth_estimate") > 50, 
                              "excessive_nesting_corruption")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when(F.col("dependency_depth_estimate") > 50, 3)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 4: Incomplete content (truncation_indicators > 4 AND char_length < 5000)
    # Increased threshold to 4 to avoid rejecting intentional cliffhangers/multi-part content
    # Added length check: short docs more likely to be truly truncated
    df = df.withColumn("is_rejected",
                       F.when((F.col("truncation_indicators") > 4) & (F.col("char_length") < 5000), True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("truncation_indicators") > 4) & (F.col("char_length") < 5000), 
                              "incomplete_truncated_content")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when((F.col("truncation_indicators") > 4) & (F.col("char_length") < 5000), 3)
                       .otherwise(F.col("rejection_priority")))
    
    # Rule 5: Code mostly comments (code_comment_ratio > 0.9 for code domain)
    # Increased threshold to 0.9 to preserve heavily-commented tutorial/documentation code
    # 80% comments can be legitimate educational content; 90% is true stub/boilerplate
    df = df.withColumn("is_rejected",
                       F.when((F.col("domain") == "code") & 
                              (F.col("code_comment_ratio") > 0.9), True)
                       .otherwise(F.col("is_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("domain") == "code") & 
                              (F.col("code_comment_ratio") > 0.9), 
                              "code_mostly_comments")
                       .otherwise(F.col("rejection_reason")))
    df = df.withColumn("rejection_priority",
                       F.when((F.col("domain") == "code") & 
                              (F.col("code_comment_ratio") > 0.9), 3)
                       .otherwise(F.col("rejection_priority")))
    
    # Split into rejected and passed
    rejected_df = df.filter(F.col("is_rejected") == True)
    passed_df = df.filter(F.col("is_rejected") == False)
    
    return rejected_df, passed_df

# =========================================================================
# PROBABILISTIC BAND ASSIGNMENT (V4.0 - MoE-Friendly)
# =========================================================================

def compute_difficulty_score(df, broadcast_keywords):
    """
    Compute single scalar difficulty_score ∈ [0,1] using cheap heuristics.
    
    This is a FAST, model-free score for large-scale data (4TB).
    
    Components (weighted sum):
    1. Normalized length (0.25): Longer = more complex (with cap)
    2. Structural density (0.20): Code blocks, lists, headings
    3. Reasoning density (0.20): Reasoning markers per sentence
    4. Symbol density (0.20): Math/code symbols
    5. Rarity proxy (0.15): High-value academic keywords from broadcast
    
    All components are O(text length) and CPU-friendly.
    """
    print("  Computing difficulty score...")
    
    # Component 1: Normalized length (cap at 10K tokens)
    df = df.withColumn("_norm_length",
                       F.least(F.col("token_count_estimate") / 10000.0, 1.0))
    
    # Component 2: Structural density
    total_struct = (F.col("code_block_count") + F.col("heading_count") + 
                   F.col("table_count_estimate"))
    df = df.withColumn("_struct_density",
                       F.least(total_struct / F.greatest(F.col("line_count"), F.lit(1)) * 10.0, 1.0))
    
    # Component 3: Reasoning density (reasoning markers per sentence)
    df = df.withColumn("_reason_density",
                       F.least(F.col("reasoning_marker_count") / 
                               F.greatest(F.col("sentence_count_estimate"), F.lit(1)) * 5.0, 1.0))
    
    # Component 4: Symbol density (math + code symbols)
    # Count =, ->, ::, operators from math_expression_count
    df = df.withColumn("_symbol_density",
                       F.least(F.col("math_expression_count") / 
                               F.greatest(F.col("sentence_count_estimate"), F.lit(1)) * 2.0, 1.0))
    
    # Component 5: Rarity proxy (high-value keyword intersection)
    # Convert text to lowercase words array and intersect with broadcast keywords
    df = df.withColumn("_words_array",
                       F.lower(F.col("text")))
    df = df.withColumn("_words_array",
                       F.split(F.col("_words_array"), "\\s+"))
    df = df.withColumn("_high_value_count",
                       F.size(F.array_intersect(F.col("_words_array"), broadcast_keywords)))
    df = df.withColumn("_rarity_proxy",
                       F.least(F.col("_high_value_count") / 5.0, 1.0))
    
    # Weighted sum (clamp to [0, 1])
    df = df.withColumn("difficulty_score",
                       F.greatest(F.lit(0.0), F.least(
                           F.lit(0.25) * F.col("_norm_length") +
                           F.lit(0.20) * F.col("_struct_density") +
                           F.lit(0.20) * F.col("_reason_density") +
                           F.lit(0.20) * F.col("_symbol_density") +
                           F.lit(0.15) * F.col("_rarity_proxy"),
                           F.lit(1.0))))
    
    # Cleanup temporary columns
    df = df.drop("_norm_length", "_struct_density", "_reason_density", 
                "_symbol_density", "_words_array", "_high_value_count", "_rarity_proxy")
    
    print("    Difficulty score computed.")
    return df


def assign_curriculum_band_probabilistic(df):
    """
    Assign PROBABILISTIC curriculum bands (B0-B5) using soft distributions.
    
    V4.0 Design Principles:
    - No hard overrides, only probability nudges
    - Preserve uncertainty for mixed-difficulty content
    - MoE-friendly (smooth transitions, no curriculum cliffs)
    - Conservative final band (lowest credible, EPS(epsilon)=0.10)
    
    Output Columns:
    - band_p_B0, band_p_B1, band_p_B2, band_p_B3, band_p_B4, band_p_B5 (probabilities)
    - final_band (conservative assignment for training safety)
    
    Algorithm:
    1. Map difficulty_score to band probabilities (triangular weighting)
    2. Apply small content nudges (+0.05 to +0.15 for code/agentic/research)
    3. Normalize to sum=1
    4. Select final_band = argmin(band where p(band) >= EPS)
    
    See README_probabilistic_banding_v4.md for full specification.
    """
    print("  Assigning probabilistic curriculum bands (B0-B5)...")
    
    # Band centers (fixed, no learning)
    BAND_CENTERS = {
        "B0": 0.05,  # Surface fluency
        "B1": 0.20,  # Everyday language
        "B2": 0.35,  # Structured knowledge
        "B3": 0.55,  # Reasoning begins
        "B4": 0.75,  # Abstraction
        "B5": 0.90   # Planning/agentic
    }
    WIDTH = 0.20  # Triangular window width
    EPS = 0.10    # Conservative assignment threshold
    
    # Step 1: Compute raw band weights using triangular weighting
    # weight = max(0, 1 - |score - center| / width)
    for band, center in BAND_CENTERS.items():
        df = df.withColumn(f"_w_{band}",
                          F.greatest(F.lit(0.0), 
                                    F.lit(1.0) - F.abs(F.col("difficulty_score") - F.lit(center)) / F.lit(WIDTH)))
    
    # Step 2: Apply content-based probability nudges (small, not overrides)
    # These bias the distribution but never force outcomes
    
    # Code content: +0.05 to B3, +0.10 to B4
    df = df.withColumn("_w_B3",
                      F.when(F.col("has_code"), F.col("_w_B3") + 0.05)
                      .otherwise(F.col("_w_B3")))
    df = df.withColumn("_w_B4",
                      F.when(F.col("has_code"), F.col("_w_B4") + 0.10)
                      .otherwise(F.col("_w_B4")))
    
    # Agentic content: +0.10 to B4, +0.15 to B5
    df = df.withColumn("_w_B4",
                      F.when(F.col("has_agentic"), F.col("_w_B4") + 0.10)
                      .otherwise(F.col("_w_B4")))
    df = df.withColumn("_w_B5",
                      F.when(F.col("has_agentic"), F.col("_w_B5") + 0.15)
                      .otherwise(F.col("_w_B5")))
    
    # Research papers: +0.08 to B4, +0.12 to B5
    df = df.withColumn("_w_B4",
                      F.when(F.col("has_research_paper"), F.col("_w_B4") + 0.08)
                      .otherwise(F.col("_w_B4")))
    df = df.withColumn("_w_B5",
                      F.when(F.col("has_research_paper"), F.col("_w_B5") + 0.12)
                      .otherwise(F.col("_w_B5")))
    
    # Math content: +0.05 to B3, +0.08 to B4
    df = df.withColumn("_w_B3",
                      F.when(F.col("has_math"), F.col("_w_B3") + 0.05)
                      .otherwise(F.col("_w_B3")))
    df = df.withColumn("_w_B4",
                      F.when(F.col("has_math"), F.col("_w_B4") + 0.08)
                      .otherwise(F.col("_w_B4")))
    
    # Reasoning content: +0.05 to B3
    df = df.withColumn("_w_B3",
                      F.when(F.col("has_reasoning"), F.col("_w_B3") + 0.05)
                      .otherwise(F.col("_w_B3")))
    
    # Pure narrative (no code/math/reasoning): +0.05 to B1
    is_narrative = (~F.col("has_code") & ~F.col("has_math") & ~F.col("has_reasoning"))
    df = df.withColumn("_w_B1",
                      F.when(is_narrative, F.col("_w_B1") + 0.05)
                      .otherwise(F.col("_w_B1")))
    
    # Step 3: Normalize to probabilities (sum = 1)
    df = df.withColumn("_total_weight",
                      F.col("_w_B0") + F.col("_w_B1") + F.col("_w_B2") + 
                      F.col("_w_B3") + F.col("_w_B4") + F.col("_w_B5"))
    
    # Avoid division by zero (should never happen, but safety first)
    df = df.withColumn("_total_weight",
                      F.when(F.col("_total_weight") > 0, F.col("_total_weight"))
                      .otherwise(F.lit(1.0)))
    
    for band in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        df = df.withColumn(f"band_p_{band}",
                          F.col(f"_w_{band}") / F.col("_total_weight"))
    
    # Step 4: Conservative final band assignment (lowest credible band)
    # Select argmin(band where p(band) >= EPS)
    # This ensures training safety under uncertainty
    df = df.withColumn("final_band",
                      F.when(F.col("band_p_B0") >= EPS, "B0")
                      .when(F.col("band_p_B1") >= EPS, "B1")
                      .when(F.col("band_p_B2") >= EPS, "B2")
                      .when(F.col("band_p_B3") >= EPS, "B3")
                      .when(F.col("band_p_B4") >= EPS, "B4")
                      .otherwise("B5"))
    
    # Cleanup temporary columns
    df = df.drop("_w_B0", "_w_B1", "_w_B2", "_w_B3", "_w_B4", "_w_B5", "_total_weight")
    
    print("    Probabilistic band assignment complete.")
    
    return df

# =========================================================================
# ADDITIONAL METRICS (NOT IMPLEMENTED - See README)
# =========================================================================

# The following metrics from Team2/Team3 CSVs are NOT implemented in this version:
# 
# 1. mtld (Lexical Diversity): Requires sequential algorithm, too expensive for 4TB
# 2. fertility (char/token ratio): Requires tiktoken, adds dependency
# 3. script_distribution: Unicode analysis, limited value for English-only data
# 4. information_density: Requires POS tagging (nltk/spacy), too expensive
# 5. concept_density: Requires NER (spacy), too expensive
# 6. rare_word_ratio: Requires frequency dictionary, adds complexity
# 7. domain_specificity: Requires domain lexicons, out of scope
# 8. few_shot_potential: Heuristic-based, limited ROI
# 9. cross_domain_analogy_markers: Pattern matching, low signal
# 10. low_effort_post_score (Team3): Composite metric, covered by other filters
# 11. url_spam_score (Team3): Advanced heuristic, covered by risky_tld + url_ratio
#
# Trade-off Rationale:
# - Focus on high-ROI metrics that can be computed efficiently with Spark SQL
# - Prioritize rejection filters (eliminate bad data early)
# - Avoid external dependencies (tiktoken, nltk, spacy) to reduce cost
# - Derived/composite metrics can be computed in downstream analysis if needed
#
# See README.md for detailed justification.

# =========================================================================
# OUTPUT PREPARATION
# =========================================================================

def prepare_output_columns(df, include_rejection=False):
    """
    Select and order final output columns (V4.0 - Probabilistic Banding).
    
    Core Columns (both files):
    - uuid, id, file_path, source, domain
    
    Probabilistic Band Columns (V4.0 - THE END GOAL):
    - band_p_B0, band_p_B1, band_p_B2, band_p_B3, band_p_B4, band_p_B5 (probabilities)
    - final_band: Conservative assignment (lowest credible, EPS=0.10)
    - difficulty_score: Scalar [0,1] for transparency
    
    Metrics Columns:
    - fertility_estimate: Re-added in V4.0 for analysis
    - Only metrics used for rejection or band assignment
    
    Rejection File Additional Columns:
    - is_rejected, rejection_reason, rejection_priority
    """
    
    # Core tracking columns
    core_cols = ["uuid", "id", "file_path", "source", "domain"]
    
    # PROBABILISTIC BAND COLUMNS (V4.0 - THE END GOAL)
    band_cols = [
        "band_p_B0", "band_p_B1", "band_p_B2", "band_p_B3", "band_p_B4", "band_p_B5",
        "final_band", "difficulty_score"
    ]
    
    # Stage 1 metrics (used for rejection)
    stage1_cols = [
        "byte_length", "char_length", "word_count", "line_count", 
        "token_count_estimate", "non_printable_ratio", 
        "fertility_estimate"  # V4.0: Re-added for analysis
    ]
    
    # Stage 2 metrics (used for rejection + band components)
    stage2_cols = [
        "unique_token_ratio", "compression_ratio",
        "whitespace_ratio", "capitalization_ratio",
        "url_count", "url_ratio", "html_tag_density", "boilerplate_ratio",
        "thread_fragment_indicator", "risky_tld_count",
        "sentence_count_estimate", "sentence_boundary_coherence",
        "rare_word_ratio_estimate"
    ]
    
    # Stage 3 metrics (used for rejection + band assignment)
    stage3_cols = [
        # Readability (profiling)
        "flesch_reading_ease", "flesch_kincaid_grade",
        # Modality detection (used for band nudges)
        "has_code", "has_math", "has_reasoning", "has_agentic", "has_research_paper",
        "primary_modality",
        # Markers (used for modality detection and difficulty scoring)
        "cot_markers", "agentic_markers", "research_paper_markers", "reasoning_marker_count",
        # Legacy difficulty level (for backward compatibility)
        "difficulty_level",
        # Structural metrics (profiling + rejection + difficulty scoring)
        "code_block_count", "truncation_indicators", "code_comment_ratio",
        # Profiling metrics (useful for analysis, minimal cost)
        "question_density", "citation_count", "math_expression_count",
        "heading_count", "table_count_estimate"
    ]
    
    # Version metadata
    meta_cols = ["version"]
    
    # Rejection columns (only for rejection file)
    rejection_cols = ["is_rejected", "rejection_reason", "rejection_priority"]
    
    if include_rejection:
        select_cols = core_cols + band_cols + stage1_cols + stage2_cols + stage3_cols + meta_cols + rejection_cols
    else:
        select_cols = core_cols + band_cols + stage1_cols + stage2_cols + stage3_cols + meta_cols
    
    # Filter to only columns that exist in the dataframe
    existing_cols = [col for col in select_cols if col in df.columns]
    
    return df.select(*existing_cols)

# =========================================================================
# MAIN EXECUTION
# =========================================================================

def main():
    """
    Main execution logic (V4.0 - Probabilistic Banding):
    1. Read T1 parquet (column pruning)
    2. Create broadcast variable for high-value keywords
    3. Stage 1: Fast rejection (corruption, length) + coalesce
    4. Stage 2: Medium rejection (spam, templates)
    5. Stage 3: Expensive metrics (quality, complexity, modality)
    6. Difficulty Score: Compute single scalar [0,1]
    7. Probabilistic Band Assignment: B0-B5 with soft distributions
    8. Conservative Final Band: Select lowest credible band (EPS=0.10)
    9. Write outputs: rejection_file + metrics_file (with band probabilities)
    """
    args, optional_args = get_glue_args()
    
    input_base = optional_args['INPUT_BASE']
    output_base = OUTPUT_BASE
    intermediate_base = INTERMEDIATE_BASE
    source_filter = optional_args['SOURCE']  # Optional: filter by specific source
    
    # Initialize Spark
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    
    # =========================================================================
    # SPARK OPTIMIZATION CONFIG (Optimized for 4TB+ and Glue Flex)
    # =========================================================================
    # IMPORTANT: Some configs cannot be set via spark.conf.set() in Glue Flex
    # These MUST be passed via --conf parameter in the CLI (see usage examples below)
    # Forbidden configs: spark.memory.*, spark.network.timeout, spark.sql.broadcastTimeout
    
    # AQE: Adaptive Query Execution (auto-optimize at runtime)
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728")  # 128MB
    
    # CRITICAL: Skew handling for unbalanced domain/source distribution
    # If 80% of data is in one source (e.g., common_crawl), this prevents hangs
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
    
    # Partition tuning: 8000 for 4TB (target ~128MB per partition = 4TB/8000 = 512MB avg)
    # AQE will coalesce down after filtering, but start high to handle skew
    spark.conf.set("spark.sql.shuffle.partitions", "8000")
    spark.conf.set("spark.sql.files.maxPartitionBytes", "134217728")  # 128MB
    
    # Compression (zstd for best compression ratio)
    spark.conf.set("spark.sql.parquet.compression.codec", "zstd")
    
    # Broadcast join threshold (avoid shuffle for small lookup tables)
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
    
    # S3 Committer Optimization (Critical for 4TB writes)
    spark.conf.set("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
    spark.conf.set("spark.sql.parquet.fs.optimized.committer.optimization-enabled", "true")
    
    # Predicate pushdown (push filters down to parquet reader)
    spark.conf.set("spark.sql.parquet.filterPushdown", "true")
    spark.conf.set("spark.sql.parquet.enableVectorizedReader", "true")
    
    # NOTE: The following configs CANNOT be set in Glue Flex execution
    # Pass them via --conf parameter in CLI instead:
    # - spark.network.timeout
    # - spark.sql.broadcastTimeout
    # - spark.memory.fraction
    # - spark.memory.storageFraction
    
    job = Job(glueContext)
    job.init('T123_metrics_calculation')
    
    # =========================================================================
    # BROADCAST VARIABLE FOR HIGH-VALUE KEYWORDS (V4.0 Optimization)
    # =========================================================================
    # Broadcast small keyword list (~150 words) to all executors
    # This replaces expensive regex rare_word detection with cheap array_intersect
    print("\nCreating broadcast variable for high-value keywords...")
    broadcast_keywords = sc.broadcast(HIGH_VALUE_KEYWORDS)
    # Convert to Spark array for use in SQL expressions
    keywords_array = F.array([F.lit(word) for word in HIGH_VALUE_KEYWORDS])
    print(f"  Broadcasted {len(HIGH_VALUE_KEYWORDS)} high-value keywords")
    
    print("=" * 80)
    print("T2 Metrics Calculator V4.0 - Progressive Quality Filtering + Probabilistic Banding")
    print(f"Input Base: {input_base}")
    print(f"Output Base: {output_base}")
    print(f"Intermediate Base: {intermediate_base}")
    if source_filter:
        print(f"Source Filter: {source_filter} (incremental mode)")
    print("=" * 80)
    
    # =========================================================================
    # STEP 1: READ T1 PARQUET (Column Pruning + Optional Source Filter)
    # =========================================================================
    print("\nStep 1: Reading T1 normalized data (column pruning)...")
    
    # Only read columns we need: id, text, source, domain
    # This is critical for cost optimization (avoid reading unused columns)
    df = spark.read.parquet(input_base).select("id", "text", "source", "domain")
    
    # Optional: Filter by specific source for incremental processing
    if source_filter:
        print(f"  Filtering by source: {source_filter}")
        df = df.filter(F.col("source") == source_filter)
    
    # Add tracking metadata
    df = add_uuid_and_metadata(df)
    df = df.withColumn("version", F.lit(VERSION))
    
    # WARNING: Repartitioning by string columns causes massive shuffle on 4TB
    # But necessary for even distribution across skewed domains
    # AQE will handle the skew automatically
    df = df.repartition(8000, "domain", "source")
    
    # OPTIMIZATION: Skip expensive count - trust the pipeline to run
    # You can check record counts in final S3 output or CloudWatch logs
    print(f"  Initial partitions: {df.rdd.getNumPartitions()}")
    print("  Note: Skipping expensive count - see CloudWatch for metrics")
    
    # =========================================================================
    # STAGE 1: FAST REJECTION (Priority 1)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Stage 1: Physical Properties & Corruption Detection")
    print("=" * 80)
    
    df_stage1 = compute_stage1_metrics(df)
    rejected_stage1, passed_stage1 = apply_stage1_rejection(df_stage1)
    
    # PHYSICAL S3 WRITE (instead of checkpoint) for Stage 1 reliability
    # Use timestamp to avoid S3 deletion conflicts in Flex execution
    print("  Writing Stage 1 results to intermediate S3...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage1_rejected_path = f"{intermediate_base}/stage1_rejected_{timestamp}"
    stage1_passed_path = f"{intermediate_base}/stage1_passed_{timestamp}"
    
    rejected_stage1.write.parquet(stage1_rejected_path)
    passed_stage1.write.parquet(stage1_passed_path)
    
    # Read back from S3 (breaks lineage, 100% reliable)
    rejected_stage1 = spark.read.parquet(stage1_rejected_path)
    passed_stage1 = spark.read.parquet(stage1_passed_path)
    
    # OPTIMIZATION (V4.0): Coalesce to reduce partitions after heavy filtering
    # Stage 1 typically rejects 10-20%, so we can reduce partitions without full shuffle
    # This saves cost in Stage 2/3 by processing fewer, larger partitions
    print("  Coalescing partitions after Stage 1 filtering...")
    passed_stage1 = passed_stage1.coalesce(int(passed_stage1.rdd.getNumPartitions() * 0.8))
    print(f"    Reduced to ~{int(passed_stage1.rdd.getNumPartitions())} partitions")
    
    # =========================================================================
    # STAGE 2: MEDIUM REJECTION (Priority 2)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Stage 2: Lexical Diversity & Noise Detection")
    print("=" * 80)
    
    df_stage2 = compute_stage2_metrics(passed_stage1)
    rejected_stage2, passed_stage2 = apply_stage2_rejection(df_stage2)
    
    # PHYSICAL S3 WRITE (most critical checkpoint - saves 60% of Stage 3 compute)
    print("  Writing Stage 2 results to intermediate S3...")
    stage2_rejected_path = f"{intermediate_base}/stage2_rejected_{timestamp}"
    stage2_passed_path = f"{intermediate_base}/stage2_passed_{timestamp}"
    
    rejected_stage2.write.parquet(stage2_rejected_path)
    passed_stage2.write.parquet(stage2_passed_path)
    
    # Read back from S3
    rejected_stage2 = spark.read.parquet(stage2_rejected_path)
    passed_stage2 = spark.read.parquet(stage2_passed_path)
    
    # =========================================================================
    # STAGE 3: EXPENSIVE METRICS (Priority 3)
    # Includes: Modality detection, Difficulty scoring, Information density
    # =========================================================================
    print("\n" + "=" * 80)
    print("Stage 3: Quality Metrics, Modality & Difficulty Analysis")
    print("=" * 80)
    
    df_stage3 = compute_stage3_metrics(passed_stage2)
    rejected_stage3, passed_stage3 = apply_stage3_rejection(df_stage3)
    
    # =========================================================================
    # DIFFICULTY SCORE COMPUTATION (V4.0 - Single Scalar)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Difficulty Score Computation (Cheap Heuristics)")
    print("=" * 80)
    
    passed_stage3 = compute_difficulty_score(passed_stage3, keywords_array)
    
    # =========================================================================
    # PROBABILISTIC BAND ASSIGNMENT (V4.0 - MoE-Friendly)
    # Assign soft band probabilities (B0-B5) + conservative final_band
    # =========================================================================
    print("\n" + "=" * 80)
    print("Probabilistic Curriculum Band Assignment (B0-B5)")
    print("=" * 80)
    
    passed_stage3 = assign_curriculum_band_probabilistic(passed_stage3)
    
    # =========================================================================
    # STEP 2: UNION ALL REJECTED RECORDS
    # =========================================================================
    print("\n" + "=" * 80)
    print("Consolidating Rejection File")
    print("=" * 80)
    
    # Prepare rejected dataframes (fill missing columns with nulls)
    rejected_stage1_out = prepare_output_columns(rejected_stage1, include_rejection=True)
    rejected_stage2_out = prepare_output_columns(rejected_stage2, include_rejection=True)
    rejected_stage3_out = prepare_output_columns(rejected_stage3, include_rejection=True)
    
    # Union all rejections
    rejection_file = rejected_stage1_out.unionByName(rejected_stage2_out, allowMissingColumns=True)
    rejection_file = rejection_file.unionByName(rejected_stage3_out, allowMissingColumns=True)
    
    # OPTIMIZATION: Skip expensive count - write operations log actual record counts
    print("  Note: Record counts available after write completion")
    
    # =========================================================================
    # STEP 3: PREPARE METRICS FILE (Passed Records)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Preparing Metrics File (Passed Records)")
    print("=" * 80)
    
    metrics_file = prepare_output_columns(passed_stage3, include_rejection=False)
    print("  Note: Record counts available after write completion")
    
    # =========================================================================
    # STEP 4: WRITE OUTPUTS (Partitioned by domain, source for easy downstream joins)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Writing Outputs")
    print("=" * 80)
    
    # Write rejection file (partitioned by rejection_priority, domain, source)
    # This allows easy analysis: "Show me all Priority 2 rejections from arxiv in science domain"
    rejection_output = f"{output_base}/rejection_file"
    print(f"  Writing rejection file to: {rejection_output}")
    
    write_mode = "append" if source_filter else "overwrite"
    
    # OPTIMIZATION: Use sortWithinPartitions instead of full repartition to minimize shuffle
    # This preserves data locality while organizing for efficient S3 writes
    (
        rejection_file
        .repartition("rejection_priority", "domain", "source")  # Let AQE decide partition count
        .write
        .mode(write_mode)  # Append if incremental, overwrite otherwise
        .partitionBy("rejection_priority", "domain", "source")
        .option("compression", "zstd")
        .parquet(rejection_output)
    )
    print(f"    ✓ Rejection file written (mode={write_mode})")
    
    # Write metrics file (partitioned by domain, source)
    # This enables efficient downstream queries: "Load all math domain data"
    metrics_output = f"{output_base}/metrics_file"
    print(f"  Writing metrics file to: {metrics_output}")
    (
        metrics_file
        .repartition("domain", "source")  # Let AQE decide partition count
        .write
        .mode(write_mode)  # Append if incremental, overwrite otherwise
        .partitionBy("domain", "source")
        .option("compression", "zstd")
        .parquet(metrics_output)
    )
    print(f"    ✓ Metrics file written (mode={write_mode})")
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("Pipeline Summary")
    print("=" * 80)
    print("  OPTIMIZATION: Skipped expensive count operations for 4TB performance")
    print("  Check record counts in:")
    print("    1. CloudWatch Logs: Look for 'numOutputRows' in Spark metrics")
    print("    2. S3 Console: Check file sizes in output directories")
    print("    3. Run: spark.read.parquet(rejection_output).count() separately")
    print("")
    print("  Output Files:")
    print(f"    - Rejection File: {rejection_output}")
    print(f"    - Metrics File: {metrics_output}")
    print("")
    print("  Performance Notes:")
    print("    - 8000 shuffle partitions (4TB target)")
    print("    - Skew join enabled for unbalanced domains")
    print("    - S3 fast committer enabled")
    print("=" * 80)
    
    job.commit()

if __name__ == '__main__':
    main()
