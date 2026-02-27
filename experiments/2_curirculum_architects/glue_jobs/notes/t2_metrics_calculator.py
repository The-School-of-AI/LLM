"""
T2 Metrics Calculator - Curriculum Learning Metrics Pipeline
=============================================================
Purpose: Read T1 Parquet data, compute 60+ text metrics, apply rejection logic.
Data: ~4TB | Output: Partitioned by external_source

Usage:
    aws glue start-job-run --job-name t2-metrics-calculator \\
        --arguments '{
            "--additional-python-modules":"tiktoken,textstat",
            "--INPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/normalized_data",
            "--OUTPUT_BASE":"s3://t1-dataacquisition-datasets/processed_dataset/validated_data"
        }'
"""

import sys
import re
import zlib
import pandas as pd
from collections import Counter

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark import StorageLevel
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType, IntegerType, DoubleType, StructType, 
    StructField, BooleanType
)
from pyspark.sql.functions import pandas_udf

# Import libraries for UDF
try:
    import tiktoken
    import textstat
except ImportError:
    print("WARNING: tiktoken or textstat not available. UDF will fail.")

# -------------------------------------------------------------------------
# REGEX PATTERNS
# -------------------------------------------------------------------------
BOILERPLATE_MARKERS = [
    'cookie policy', 'privacy policy', 'terms of service', 'all rights reserved', 
    '© copyright', 'click here', 'subscribe to', 'sign up', 'newsletter', 
    'unsubscribe', 'contact us', 'about us', 'follow us on', 'accept cookies', 
    'manage preferences'
]
BOILERPLATE_REGEX = "|".join([re.escape(m) for m in BOILERPLATE_MARKERS])

THREAD_MARKERS = [
    '>>', 'replied to:', 'in response to', 're:', 'replying to', 
    'quote from', 'responding to'
]
THREAD_REGEX = "|".join([re.escape(m) for m in THREAD_MARKERS])

RISKY_TLDS = r"(?:\.tk|\.ml|\.ga|\.cf|\.gq|\.xyz|\.top|\.club|\.win)\b"
AGENTIC_MARKERS = r"(Action:|Observation:|Tool:|Thought:|Plan:)"
COT_MARKERS = r"(Let's think step by step|Let's reason|chain of thought|reasoning:)"
RESEARCH_PAPER_MARKERS = r"(Abstract\b|Introduction\b|References\b|Bibliography\b|Conclusion\b)"

# -------------------------------------------------------------------------
# PANDAS UDF FOR TIER 2 METRICS (Python Libraries)
# -------------------------------------------------------------------------
UDF_SCHEMA = StructType([
    StructField("token_count_estimate", IntegerType(), True),
    StructField("unique_token_ratio", DoubleType(), True),
    StructField("vocab_size", IntegerType(), True),
    StructField("fertility", DoubleType(), True),
    StructField("rare_word_ratio", DoubleType(), True),
    StructField("compression_ratio", DoubleType(), True),
    StructField("flesch_reading_ease", DoubleType(), True),
    StructField("avg_word_length", DoubleType(), True),
    StructField("sentence_count_estimate", IntegerType(), True),
    StructField("avg_sentence_length", DoubleType(), True)
])

SAFE_DEFAULT_METRICS = (0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

@pandas_udf(UDF_SCHEMA)
def compute_python_metrics(texts: pd.Series, word_counts: pd.Series) -> pd.DataFrame:
    """
    Vectorized UDF to compute Python-based metrics.
    
    Args:
        texts: Series of text strings (NULL for rejected rows)
        word_counts: Series of word counts from Tier 1
    
    Returns:
        DataFrame with 10 metric columns
    """
    results = []
    
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"ERROR loading tiktoken encoder: {e}")
        # Return defaults for all rows
        return pd.DataFrame([SAFE_DEFAULT_METRICS] * len(texts), columns=[f.name for f in UDF_SCHEMA.fields])
    
    for text, word_count in zip(texts, word_counts):
        # Short-circuit: If text is NULL (rejected by Tier 1), return defaults
        if pd.isna(text) or text is None or len(text) == 0:
            results.append(SAFE_DEFAULT_METRICS)
            continue
        
        try:
            # Tokenization
            tokens = encoder.encode(text, disallowed_special=())
            token_count = len(tokens)
            unique_tokens = len(set(tokens))
            unique_ratio = unique_tokens / token_count if token_count > 0 else 0.0
            
            # Fertility
            fertility = token_count / word_count if word_count > 0 else 0.0
            
            # Rare word ratio
            token_freq = Counter(tokens)
            rare_count = sum(1 for count in token_freq.values() if count == 1)
            rare_ratio = rare_count / token_count if token_count > 0 else 0.0
            
            # Compression
            compressed = zlib.compress(text.encode('utf-8'), level=6)
            compression_ratio = len(compressed) / len(text.encode('utf-8')) if len(text) > 0 else 0.0
            
            # Readability
            flesch = textstat.flesch_reading_ease(text)
            
            # Word length
            words = text.split()
            avg_word_len = sum(len(w) for w in words) / len(words) if words else 0.0
            
            # Sentence count
            sentences = re.findall(r'[.!?]+', text)
            sentence_count = len(sentences)
            avg_sentence_len = len(text) / sentence_count if sentence_count > 0 else 0.0
            
            results.append((
                token_count,
                unique_ratio,
                unique_tokens,
                fertility,
                rare_ratio,
                compression_ratio,
                flesch,
                avg_word_len,
                sentence_count,
                avg_sentence_len
            ))
            
        except Exception as e:
            # On error, return safe defaults for this row
            results.append(SAFE_DEFAULT_METRICS)
    
    return pd.DataFrame(results, columns=[f.name for f in UDF_SCHEMA.fields])

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def get_glue_args():
    """Parse Glue job arguments."""
    args = getResolvedOptions(sys.argv, ['JOB_NAME'])
    
    optional_args = {}
    if '--INPUT_BASE' in sys.argv:
        optional_args['INPUT_BASE'] = getResolvedOptions(sys.argv, ['INPUT_BASE'])['INPUT_BASE']
    else:
        optional_args['INPUT_BASE'] = 's3://t1-dataacquisition-datasets/processed_dataset/raw_data'
    
    if '--OUTPUT_BASE' in sys.argv:
        optional_args['OUTPUT_BASE'] = getResolvedOptions(sys.argv, ['OUTPUT_BASE'])['OUTPUT_BASE']
    else:
        optional_args['OUTPUT_BASE'] = 's3://t1-dataacquisition-datasets/processed_dataset/metrics'
    
    if '--CHECKPOINT_DIR' in sys.argv:
        optional_args['CHECKPOINT_DIR'] = getResolvedOptions(sys.argv, ['CHECKPOINT_DIR'])['CHECKPOINT_DIR']
    else:
        optional_args['CHECKPOINT_DIR'] = 's3://t1-dataacquisition-datasets/processed_dataset/checkpoints/'
    
    return args, optional_args

def main():
    """Main execution logic."""
    args, optional_args = get_glue_args()
    
    input_base = optional_args['INPUT_BASE']
    output_base = optional_args['OUTPUT_BASE']
    checkpoint_dir = optional_args['CHECKPOINT_DIR']
    
    # Initialize Spark
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    
    # -------------------------------------------------------------------------
    # OPTIMIZATION CONFIG (500GB Scale)
    # -------------------------------------------------------------------------
    sc.setCheckpointDir(checkpoint_dir)
    
    spark.conf.set("spark.sql.shuffle.partitions", "2000")
    spark.conf.set("spark.sql.parquet.compression.codec", "zstd")
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728")  # 128MB
    spark.conf.set("spark.sql.files.maxPartitionBytes", "134217728")
    
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)
    
    print("=" * 80)
    print("T2 Metrics Calculator - Starting")
    print(f"Input Base: {input_base}")
    print(f"Output Base: {output_base}")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # STEP 1: READ T1 PARQUET DATA
    # -------------------------------------------------------------------------
    print("\nStep 1: Reading T1 Parquet data...")
    
    df = spark.read.parquet(input_base)
    
    # Repartition for even distribution
    df = df.repartition(2000)
    
    print(f"  Total partitions: {df.rdd.getNumPartitions()}")
    
    # -------------------------------------------------------------------------
    # STEP 2: TIER 1 METRICS (Spark Native - Fast)
    # -------------------------------------------------------------------------
    print("\nStep 2: Computing Tier 1 metrics (Spark Native)...")
    
    # Physical metrics
    df = df.withColumn("byte_length", F.length(F.encode("text", "utf-8")))
    df = df.withColumn("char_length", F.length("text"))
    df = df.withColumn("line_count", F.size(F.split("text", "\n")))
    df = df.withColumn("word_count", F.size(F.split("text", "\\s+")))
    df = df.withColumn("avg_line_length", 
                       F.when(F.col("line_count") > 0, F.col("char_length") / F.col("line_count"))
                       .otherwise(0.0))
    
    # Structure/Quality
    df = df.withColumn("whitespace_ratio", 
                       F.length(F.regexp_replace("text", r"[^\s]", "")) / F.col("char_length"))
    
    df = df.withColumn("alpha_chars", F.length(F.regexp_replace("text", r"[^a-zA-Z]", "")))
    df = df.withColumn("upper_chars", F.length(F.regexp_replace("text", r"[^A-Z]", "")))
    df = df.withColumn("capitalization_ratio", 
                       F.when(F.col("alpha_chars") > 0, F.col("upper_chars") / F.col("alpha_chars"))
                       .otherwise(0.0))
    
    df = df.withColumn("symbol_chars", F.length(F.regexp_replace("text", r"[a-zA-Z0-9\s]", "")))
    df = df.withColumn("symbol_density", F.col("symbol_chars") / F.col("char_length"))
    df = df.withColumn("punctuation_density", F.col("symbol_density"))
    
    # URL/Email/Numbers
    df = df.withColumn("url_count", F.size(F.split("text", r"https?://\S+")) - 1)
    df = df.withColumn("email_count", F.size(F.split("text", r"[\w\.-]+@[\w\.-]+\.\w+")) - 1)
    df = df.withColumn("num_numeric_tokens", F.size(F.split("text", r"[0-9]+")) - 1)
    
    # HTML/Boilerplate/Thread
    df = df.withColumn("no_tags_len", F.length(F.regexp_replace("text", r"<[^>]+>", "")))
    df = df.withColumn("html_tag_density", 
                       (F.col("char_length") - F.col("no_tags_len")) / F.col("char_length"))
    
    df = df.withColumn("boilerplate_count", 
                       F.size(F.split(F.lower("text"), BOILERPLATE_REGEX)) - 1)
    df = df.withColumn("risky_tld_count", 
                       F.size(F.split(F.lower("text"), RISKY_TLDS)) - 1)
    df = df.withColumn("thread_fragment_indicator", 
                       F.size(F.split(F.lower("text"), THREAD_REGEX)) - 1)
    df = df.withColumn("non_printable_ratio", 
                       (F.col("char_length") - F.length(F.regexp_replace("text", r"[^ -~]", ""))) / F.col("char_length"))
    
    # Cognitive/Reasoning
    df = df.withColumn("question_density", 
                       (F.length("text") - F.length(F.regexp_replace("text", r"\?", ""))))
    df = df.withColumn("citation_count", F.size(F.split("text", r"\[\d+\]")) - 1)
    df = df.withColumn("list_marker_count", F.size(F.split("text", r"^\s*[\-\*]\s+")) - 1)
    df = df.withColumn("math_expression_count", F.size(F.split("text", r"[\+\-\*/\^=]{2,}")) - 1)
    df = df.withColumn("equation_density", F.size(F.split("text", r"\$.+?\$")) - 1)
    df = df.withColumn("code_block_count", 
                       (F.length("text") - F.length(F.replace(F.col("text"), F.lit("```"), F.lit("")))) / 3)
    
    # Agentic/CoT/Research
    df = df.withColumn("agentic_markers", F.size(F.split("text", AGENTIC_MARKERS)) - 1)
    df = df.withColumn("cot_markers", F.size(F.split(F.lower("text"), COT_MARKERS)) - 1)
    df = df.withColumn("research_paper_markers", F.size(F.split("text", RESEARCH_PAPER_MARKERS)) - 1)
    df = df.withColumn("reasoning_marker_density", 
                       F.size(F.split(F.lower("text"), r"(\btherefore\b|\bthus\b|\bimplies\b|\bbecause\b)")) - 1)
    
    # Code comment ratio (for code domains)
    df = df.withColumn("code_comment_lines", 
                       F.size(F.split("text", r"^\s*(?://|#|/\*|\*)")) - 1)
    df = df.withColumn("code_comment_ratio", 
                       F.when(F.col("line_count") > 0, F.col("code_comment_lines") / F.col("line_count"))
                       .otherwise(0.0))
    
    print("  ✓ Tier 1 metrics computed")
    
    # -------------------------------------------------------------------------
    # STEP 3: TIER 1 REJECTION LOGIC (Fast Filters)
    # -------------------------------------------------------------------------
    print("\nStep 3: Applying Tier 1 rejection filters...")
    
    df = df.withColumn("tier1_rejected", F.lit(False))
    df = df.withColumn("rejection_reason", F.lit(""))
    
    # Length checks
    df = df.withColumn("tier1_rejected", 
                       F.when((F.col("byte_length") < 50) | (F.col("byte_length") > 1048576), True)
                       .otherwise(F.col("tier1_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("byte_length") < 50) | (F.col("byte_length") > 1048576), "length")
                       .otherwise(F.col("rejection_reason")))
    
    df = df.withColumn("tier1_rejected",
                       F.when((F.col("char_length") < 20) | (F.col("char_length") > 500000), True)
                       .otherwise(F.col("tier1_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("char_length") < 20) | (F.col("char_length") > 500000), "length")
                       .otherwise(F.col("rejection_reason")))
    
    # Corruption
    df = df.withColumn("tier1_rejected",
                       F.when(F.col("non_printable_ratio") > 0.01, True)
                       .otherwise(F.col("tier1_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when(F.col("non_printable_ratio") > 0.01, "corruption")
                       .otherwise(F.col("rejection_reason")))
    
    # Spam/Noise
    df = df.withColumn("tier1_rejected",
                       F.when((F.col("html_tag_density") > 0.05) | 
                              (F.col("boilerplate_count") > 4) | 
                              (F.col("risky_tld_count") > 0), True)
                       .otherwise(F.col("tier1_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("html_tag_density") > 0.05) | 
                              (F.col("boilerplate_count") > 4) | 
                              (F.col("risky_tld_count") > 0), "spam_noise")
                       .otherwise(F.col("rejection_reason")))
    
    # Link farm
    df = df.withColumn("tier1_rejected",
                       F.when((F.col("url_count") / F.greatest(F.col("word_count"), F.lit(1))) > 0.3, True)
                       .otherwise(F.col("tier1_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("url_count") / F.greatest(F.col("word_count"), F.lit(1))) > 0.3, "link_farm")
                       .otherwise(F.col("rejection_reason")))
    
    # Mask text for rejected rows (pass NULL to UDF to skip processing)
    df = df.withColumn("text_for_udf", 
                       F.when(F.col("tier1_rejected") == False, F.col("text"))
                       .otherwise(F.lit(None)))
    
    tier1_reject_count = df.filter(F.col("tier1_rejected") == True).count()
    print(f"  ✓ Tier 1 rejected: {tier1_reject_count} rows")
    
    # -------------------------------------------------------------------------
    # STEP 4: TIER 2 METRICS (Python UDF - Conditional)
    # -------------------------------------------------------------------------
    print("\nStep 4: Computing Tier 2 metrics (Python UDF)...")
    
    df = df.withColumn("python_metrics", 
                       compute_python_metrics(F.col("text_for_udf"), F.col("word_count")))
    
    # Expand struct into columns
    df = df.select("*", "python_metrics.*").drop("python_metrics", "text_for_udf")
    
    print("  ✓ Tier 2 metrics computed")
    
    # -------------------------------------------------------------------------
    # STEP 5: TIER 2 REJECTION LOGIC
    # -------------------------------------------------------------------------
    print("\nStep 5: Applying Tier 2 rejection filters...")
    
    df = df.withColumn("tier2_rejected", F.lit(False))
    
    # Only check Tier 2 if Tier 1 passed
    # Token count
    df = df.withColumn("tier2_rejected",
                       F.when((F.col("tier1_rejected") == False) & 
                              ((F.col("token_count_estimate") < 10) | 
                               (F.col("token_count_estimate") > 131072)), True)
                       .otherwise(F.col("tier2_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("tier1_rejected") == False) & 
                              ((F.col("token_count_estimate") < 10) | 
                               (F.col("token_count_estimate") > 131072)), "token_count")
                       .otherwise(F.col("rejection_reason")))
    
    # Repetition
    df = df.withColumn("tier2_rejected",
                       F.when((F.col("tier1_rejected") == False) & 
                              (F.col("unique_token_ratio") < 0.1), True)
                       .otherwise(F.col("tier2_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("tier1_rejected") == False) & 
                              (F.col("unique_token_ratio") < 0.1), "repetition")
                       .otherwise(F.col("rejection_reason")))
    
    # Entropy
    df = df.withColumn("tier2_rejected",
                       F.when((F.col("tier1_rejected") == False) & 
                              (F.col("compression_ratio") > 0.95), True)
                       .otherwise(F.col("tier2_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("tier1_rejected") == False) & 
                              (F.col("compression_ratio") > 0.95), "entropy")
                       .otherwise(F.col("rejection_reason")))
    
    # Readability
    df = df.withColumn("tier2_rejected",
                       F.when((F.col("tier1_rejected") == False) & 
                              ((F.col("flesch_reading_ease") < 0) | 
                               (F.col("flesch_reading_ease") > 120)), True)
                       .otherwise(F.col("tier2_rejected")))
    df = df.withColumn("rejection_reason",
                       F.when((F.col("tier1_rejected") == False) & 
                              ((F.col("flesch_reading_ease") < 0) | 
                               (F.col("flesch_reading_ease") > 120)), "readability")
                       .otherwise(F.col("rejection_reason")))
    
    # Final rejection flag
    df = df.withColumn("is_rejected", 
                       F.col("tier1_rejected") | F.col("tier2_rejected"))
    
    tier2_reject_count = df.filter((F.col("tier2_rejected") == True) & (F.col("tier1_rejected") == False)).count()
    total_reject_count = df.filter(F.col("is_rejected") == True).count()
    
    print(f"  ✓ Tier 2 rejected: {tier2_reject_count} rows")
    print(f"  ✓ Total rejected: {total_reject_count} rows")
    
    # -------------------------------------------------------------------------
    # STEP 6: DERIVED METRICS & FLAGS
    # -------------------------------------------------------------------------
    print("\nStep 6: Computing derived metrics and flags...")
    
    # Structural complexity
    df = df.withColumn("structural_complexity_score",
                       (F.col("avg_sentence_length") * 0.5) + (F.col("avg_word_length") * 2))
    
    # Noise score
    df = df.withColumn("noise_score",
                       F.col("html_tag_density") + (F.col("boilerplate_count") * 0.1) + F.col("symbol_density"))
    
    # Boolean flags
    df = df.withColumn("has_code", F.col("code_block_count") > 0)
    df = df.withColumn("has_math", 
                       (F.col("equation_density") > 0.001) | (F.col("math_expression_count") > 0))
    df = df.withColumn("has_reasoning", F.col("reasoning_marker_density") > 0.001)
    df = df.withColumn("has_research_paper", 
                       (F.col("research_paper_markers") > 2) & (F.col("citation_count") > 3))
    df = df.withColumn("has_agentic", F.col("agentic_markers") > 0)
    df = df.withColumn("has_cot", F.col("cot_markers") > 0)
    
    # Densities
    df = df.withColumn("agentic_density",
                       F.when(F.col("token_count_estimate") > 0, 
                              F.col("agentic_markers") / F.col("token_count_estimate"))
                       .otherwise(0.0))
    df = df.withColumn("cot_density",
                       F.when(F.col("token_count_estimate") > 0,
                              F.col("cot_markers") / F.col("token_count_estimate"))
                       .otherwise(0.0))
    
    # Primary modality
    df = df.withColumn("primary_modality",
                       F.when(F.col("code_block_count") > 2, "code")
                       .when(F.col("equation_density") > 0.1, "math")
                       .when(F.col("research_paper_markers") > 3, "research")
                       .otherwise("text"))
    
    print("  ✓ Derived metrics computed")
    
    # -------------------------------------------------------------------------
    # STEP 7: CHECKPOINT & PERSIST
    # -------------------------------------------------------------------------
    print("\nStep 7: Checkpointing data...")
    
    df = df.checkpoint()
    df = df.persist(StorageLevel.DISK_ONLY)
    
    # Force materialization
    count = df.count()
    print(f"  ✓ Checkpointed {count} rows")
    
    # -------------------------------------------------------------------------
    # STEP 8: WRITE OUTPUT
    # -------------------------------------------------------------------------
    print("\nStep 8: Writing T2 output partitioned by domain/source...")
    
    # Select relevant columns for T2 output
    output_cols = [
        "id", "hash", "domain", "source", "is_rejected", "rejection_reason",
        # Physical
        "byte_length", "char_length", "line_count", "word_count", "avg_line_length",
        # Structure
        "whitespace_ratio", "capitalization_ratio", "punctuation_density", "symbol_density",
        "non_printable_ratio",
        # Content
        "url_count", "email_count", "num_numeric_tokens", "question_density",
        "citation_count", "list_marker_count", "math_expression_count", "equation_density",
        "code_block_count", "html_tag_density", "boilerplate_count", "risky_tld_count",
        "thread_fragment_indicator", "code_comment_ratio",
        # Agentic/CoT
        "agentic_markers", "agentic_density", "has_agentic",
        "cot_markers", "cot_density", "has_cot",
        "research_paper_markers", "has_research_paper",
        "reasoning_marker_density", "has_reasoning",
        # Tokenization
        "token_count_estimate", "unique_token_ratio", "vocab_size", "fertility", "rare_word_ratio",
        # Readability
        "compression_ratio", "flesch_reading_ease", "avg_word_length",
        "sentence_count_estimate", "avg_sentence_length",
        # Derived
        "structural_complexity_score", "noise_score", "primary_modality",
        "has_code", "has_math"
    ]
    
    df_output = df.select(output_cols)
    
    (
        df_output
        .write
        .mode("overwrite")
        .partitionBy("domain", "source")
        .option("compression", "zstd")
        .parquet(output_base)
    )
    
    print(f"  ✓ Output written to: {output_base}")
    
    # Cleanup
    df.unpersist()
    
    print("\n" + "=" * 80)
    print("T2 Metrics Calculator - Completed")
    print("=" * 80)
    
    job.commit()

if __name__ == '__main__':
    main()
