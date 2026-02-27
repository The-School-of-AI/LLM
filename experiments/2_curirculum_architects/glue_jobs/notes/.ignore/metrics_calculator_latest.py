
import sys
import re
import boto3
import zlib
import math
from collections import Counter
from functools import reduce

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark import StorageLevel
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType, StructType, StructField, BooleanType, ArrayType
from pyspark.sql.functions import pandas_udf
import pandas as pd

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
DATASETS_CONFIG = [
    {"name": "dolma_Pes2o_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_Pes2o_v1_7", "source": "pes2o", "domain": "science"},
    {"name": "dolma_arxiv_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/arxiv_v1_7/", "source": "redpajama-arxiv", "domain": "science"},
    {"name": "dolmas_cc_news_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_news_v1_7", "source": "cc_news", "domain": "news"},
    {"name": "dolmas_starcoder_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_starcoder_v1_7", "source": "Starcoder", "domain": "code"},
    {"name": "dolmas_algebraic-stack_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_algebraic_v1_7", "source": "proof_pile_2-algebraic_stack", "domain": "math"},
    {"name": "dolmas_books_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/books/", "source": "books", "domain": "literature"},
    {"name": "dolmas_open-web-math_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_open-web-math-train_v1_7", "source": "proof_pile_2-open_web_math", "domain": "math"},
    {"name": "dolmas_tulu_flan_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_tulu_flan_v1_7", "source": "flan", "domain": "instruction"},
    {"name": "dolma_C4_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_urls_C4_v1_7", "source": "C4", "domain": "web"},
    {"name": "dolma_RefineWeb_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_RefineWeb_v1_7/", "source": "refinedweb", "domain": "web"},
    {"name": "dolma_megawika_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_megawika_v1_7", "source": "megawika", "domain": "encyclopedia"},
    {"name": "dolma_reddit_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_reddit_v1_7", "source": "reddit", "domain": "social"},
    {"name": "dolma_stackexchange_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_stackexchange_v1_7", "source": "stackexchange", "domain": "qa"},
    {"name": "dolma_cc_en_head_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_en_head_v1_7/", "source": "cc", "domain": "web"},
    {"name": "dolma_cc_en_middle_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_en_middle_v1_7", "source": "cc", "domain": "web"},
    {"name": "dolma_cc_en_tail_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/dolma_cc_en_tail_v1_7", "source": "cc", "domain": "web"},
]

# -------------------------------------------------------------------------
# PATTERNS (Compiled Regex for Spark Native)
# -------------------------------------------------------------------------
BOILERPLATE_MARKERS = [
    'cookie policy', 'privacy policy', 'terms of service', 'all rights reserved', '© copyright', 
    'click here', 'subscribe to', 'sign up', 'newsletter', 'unsubscribe', 'contact us', 
    'about us', 'follow us on', 'accept cookies', 'manage preferences'
]
BOILERPLATE_REGEX = "|".join([re.escape(m) for m in BOILERPLATE_MARKERS])

THREAD_MARKERS = [
    '>>', 'replied to:', 'in response to', 're:', 'replying to', 'quote from', 'responding to'
]
THREAD_REGEX = "|".join([re.escape(m) for m in THREAD_MARKERS])

RISKY_TLDS = r"(?:\.tk|\.ml|\.ga|\.cf|\.gq|\.xyz|\.top|\.club|\.win)\b"

# Agentic / CoT Markers
AGENTIC_MARKERS = r"(Action:|Observation:|Tool:|Thought:|Plan:)"
COT_MARKERS = r"(Let's think step by step|Let's reason|chain of thought|reasoning:)"
RESEARCH_PAPER_MARKERS = r"(Abstract\b|Introduction\b|References\b|Bibliography\b|Conclusion\b)"

# -------------------------------------------------------------------------
# FUSED PANDAS UDF (Tier 2) - Inline Projection
# -------------------------------------------------------------------------

# Output Schema for UDF
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

# Default safe value for when text is masked (rejected)
SAFE_DEFAULT_METRICS = (0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

@pandas_udf(UDF_SCHEMA)
def compute_python_metrics(texts: pd.Series) -> pd.DataFrame:
    import tiktoken
    import textstat
    
    # Initialize tokenizer (once per executor batch)
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except:
        enc = None
        
    results = []
    
    for text in texts:
        # SHORT-CIRCUIT: If text is None (passed from filtering logic), skip compute
        if text is None:
            results.append(SAFE_DEFAULT_METRICS)
            continue
            
        # 1. Tokenization logic
        tokens = enc.encode(text) if enc else []
        n_tokens = len(tokens)
        
        # 2. Word Count (Approx)
        words = text.split()
        n_words = len(words)
        
        # 3. Rare Words & Vocab
        if n_tokens > 0:
            token_counts = Counter(tokens)
            n_unique = len(token_counts)
            # Rare words (appear once)
            n_rare = sum(1 for c in token_counts.values() if c == 1)
            
            unique_ratio = n_unique / n_tokens
            rare_ratio = n_rare / n_tokens
            # Fertility: tokens per word
            fertility = n_tokens / n_words if n_words > 0 else 0.0
        else:
            n_unique = 0
            n_rare = 0
            unique_ratio = 0.0
            rare_ratio = 0.0
            fertility = 0.0
            
        # 4. Compression
        try:
            # Check Zlib standard library speed
            compressed_len = len(zlib.compress(text.encode('utf-8')))
            raw_len = len(text)
            comp_ratio = compressed_len / raw_len if raw_len > 0 else 0.0
        except:
            comp_ratio = 0.0
            
        # 5. Textstat (Flesch)
        try:
            flesch = textstat.flesch_reading_ease(text)
        except:
            flesch = 0.0
            
        # 6. Struct Stats
        avg_word_len = sum(len(w) for w in words) / n_words if n_words > 0 else 0.0
        
        # Sentence Count (Regex approx is faster than NLTK)
        # Using simple split by [.!?]+ logic
        # Optimize: reuse compiled regex if possible, but python re caches.
        sentences = re.split(r'[.!?]+', text)
        n_sentences = len([s for s in sentences if s.strip()])
        n_sentences = max(1, n_sentences)
        
        avg_sent_len = len(text) / n_sentences
        
        results.append((
            n_tokens,
            unique_ratio,
            n_unique,
            fertility,
            rare_ratio,
            comp_ratio,
            float(flesch),
            avg_word_len,
            n_sentences,
            avg_sent_len
        ))
        
    return pd.DataFrame(results, columns=UDF_SCHEMA.names)

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def get_glue_args():
    # Only JOB_NAME is strictly required by Glue; others we can default or parse
    return getResolvedOptions(sys.argv, ['JOB_NAME'])

def main():
    args = get_glue_args()
    
    # Hardcoded base paths as requested user edit previously, or could be args
    output_base_t1 = 's3://t1-dataacquisition-datasets/processed_dataset/raw_data'
    output_base_t2 = 's3://t1-dataacquisition-datasets/processed_dataset/metrics'
    checkpoint_dir = 's3://t1-dataacquisition-datasets/processed_dataset/checkpoints/'
    
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    
    # -------------------------------------------------------------------------
    # OPTIMIZATION: 1TB SCALE CONFIG
    # -------------------------------------------------------------------------
    # Checkpointing to break lineage
    sc.setCheckpointDir(checkpoint_dir)
    
    # Partitioning for 1TB (5000 partitions for shuffle ~200MB tasks)
    spark.conf.set("spark.sql.shuffle.partitions", "5000")
    
    # Compression (Global ZSTD)
    spark.conf.set("spark.sql.parquet.compression.codec", "zstd")
    spark.conf.set("spark.io.compression.zstd.level", "3")
    
    # AQE
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    # Advisory size for output files
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728") # 128MB
    spark.conf.set("spark.sql.files.maxPartitionBytes", "134217728")
    
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)
    
    print("Starting Global Union Job (1TB Scale Optimized)...")
    
    # -------------------------------------------------------------------------
    # STEP 1: READ & UNION (Global Parallel)
    # -------------------------------------------------------------------------
    dfs = []
    
    for dataset in DATASETS_CONFIG:
        name = dataset["name"]
        input_path = dataset["path"]
        domain = dataset["domain"]
        source = dataset["source"]
        
        try:
            # OPTIMIZATION: Use samplingRatio to speed up schema inference (avoid full scan)
            df_temp = spark.read.option("recursiveFileLookup", "true").option("samplingRatio", 0.01).json(input_path)
            
            if "text" not in df_temp.columns:
                print(f"WARN: Skipping {name} (No 'text' column)")
                continue
            
            # Add Lineage Metadata Columns NOW to enable Union
            if "id" not in df_temp.columns:
                df_temp = df_temp.withColumn("id", F.expr("uuid()")) # Fallback
                
            # Normalize schema for Union - Add missing cols if needed
            # We assume a base schema of id, text, added, created, metadata exists or we add nulls
            for col in ["added", "created", "metadata"]:
                if col not in df_temp.columns:
                    df_temp = df_temp.withColumn(col, F.lit(None).cast(StringType()))
            
            # Add Metric metadata
            df_temp = df_temp.withColumn("dataset", F.lit(name)) \
                             .withColumn("domain", F.lit(domain)) \
                             .withColumn("source", F.lit(source)) \
                             .withColumn("file_path", F.input_file_name())
            
            # Select common columns for the Union
            # We keep 'text', 'id', 'file_path', 'dataset', 'domain', 'source', 'metadata', 'added', 'created'
            # And any other metadata we want to preserve.
            # Using casting to string for metadata to be safe
            df_temp = df_temp.select(
                F.col("id").cast("string"),
                F.col("text").cast("string"),
                F.col("file_path").cast("string"),
                F.col("dataset").cast("string"),
                F.col("domain").cast("string"),
                F.col("source").cast("string"),
                F.col("metadata").cast("string"),
                F.col("added").cast("string"),
                F.col("created").cast("string")
            )
            
            dfs.append(df_temp)
            
        except Exception as e:
            print(f"ERROR reading {name}: {e}")
            
    if not dfs:
        raise ValueError("No datasets loaded!")
        
    # UNION ALL
    df_all = reduce(lambda d1, d2: d1.unionByName(d2), dfs)
    
    # -------------------------------------------------------------------------
    # OPTIMIZATION: REPARTITION TO BYPASS GZIP BOTTLENECK
    # -------------------------------------------------------------------------
    # Force shuffle to distribute data evenly across 5000 partitions
    df_all = df_all.repartition(5000)
    
    # -------------------------------------------------------------------------
    # STEP 2: TIER 1 METRICS (Spark Native - Fast)
    # -------------------------------------------------------------------------
    
    # OPTIMIZATION: Use regex count tricks where possible to avoid array materialization overhead
    # Where regex is constant/literal, use len(text) - len(replace(text, literal, ''))
    
    df = df_all.withColumn("byte_length", F.length(F.encode("text", "utf-8")))
    df = df.withColumn("char_length", F.length("text"))
    df = df.withColumn("line_count", F.size(F.split("text", "\n"))) # Newline split is optimized
    df = df.withColumn("word_count", F.size(F.split("text", "\\s+")))
    df = df.withColumn("avg_line_length", F.col("char_length") / F.col("line_count"))
    
    # Structure/Quality
    df = df.withColumn("whitespace_ratio", F.length(F.regexp_replace("text", r"[^\s]", "")) / F.col("char_length"))
    
    # Capitalization
    df = df.withColumn("alpha_chars", F.length(F.regexp_replace("text", r"[^a-zA-Z]", "")))
    df = df.withColumn("upper_chars", F.length(F.regexp_replace("text", r"[^A-Z]", "")))
    df = df.withColumn("capitalization_ratio", F.when(F.col("alpha_chars")>0, F.col("upper_chars")/F.col("alpha_chars")).otherwise(0.0))
    
    # Punctuation/Symbols
    df = df.withColumn("symbol_chars", F.length(F.regexp_replace("text", r"[a-zA-Z0-9\s]", "")))
    df = df.withColumn("symbol_density", F.col("symbol_chars") / F.col("char_length"))
    df = df.withColumn("punctuation_density", F.col("symbol_density")) 
    
    # URL/Email/Numbers
    df = df.withColumn("url_count", F.size(F.split("text", r"https?://\S+")) - 1)
    df = df.withColumn("email_count", F.size(F.split("text", r"[\w\.-]+@[\w\.-]+\.\w+")) - 1)
    df = df.withColumn("num_numeric_tokens", F.size(F.split("text", r"[0-9]+")) - 1)
    
    # HTML / Boilerplate / TLD
    df = df.withColumn("no_tags_len", F.length(F.regexp_replace("text", r"<[^>]+>", "")))
    df = df.withColumn("html_tag_density", (F.col("char_length") - F.col("no_tags_len")) / F.col("char_length"))
    
    df = df.withColumn("boilerplate_count", F.size(F.split(F.lower("text"), BOILERPLATE_REGEX)) - 1)
    # Boilerplate Ratio (approx based on markers vs total lines or just raw count check)
    # User CSV says > 0.15 is reject. Let's use count for now or implement strict ratio if lines available.
    # csv: "boilerplate_ratio" -> Count tokens matching patterns / total tokens? Or count markers.
    # We will use marker count > 3 (from plan) or approximate ratio if we had tokens. 
    # Let's stick to Count > 3 for strictness as per Implementation Plan (Group 1: spam/noise).
    # Update: Plan says "boilerplate_ratio > 0.15". We need tokens for ratio. 
    # Approximating: (boilerplate_count * 5 words) / (char_length/5). Rough.
    # Let's use Count > 4 implies High Boilerplate for Tier 1 check.
    
    df = df.withColumn("risky_tld_count", F.size(F.split(F.lower("text"), RISKY_TLDS)) - 1)
    df = df.withColumn("thread_fragment_indicator", F.size(F.split(F.lower("text"), THREAD_REGEX)) - 1)
    df = df.withColumn("non_printable_ratio", (F.col("char_length") - F.length(F.regexp_replace("text", r"[^ -~]", ""))) / F.col("char_length"))

    # Cognitive/Reasoning/Domain
    # OPTIMIZATION: Use Literal replace for simple counts (?) = (len(text)-len(replace(?, '')))/1
    df = df.withColumn("question_density", F.length("text") - F.length(F.regexp_replace("text", r"\?", "")))
    
    df = df.withColumn("citation_count", F.size(F.split("text", r"\[\d+\]")) - 1)
    df = df.withColumn("list_marker_count", F.size(F.split("text", r"^\s*[\-\*]\s+")) - 1) 
    df = df.withColumn("math_expression_count", F.size(F.split("text", r"[\+\-\*/\^=]{2,}")) - 1) 
    df = df.withColumn("equation_density", F.size(F.split("text", r"\$.+?\$")) - 1)
    
    # OPTIMIZATION: Literal replace for code blocks (```) = (len(text)-len(replace(```, '')))/3
    df = df.withColumn("code_block_count", (F.length("text") - F.length(F.replace(F.col("text"), F.lit("```"), F.lit("")))) / 3)
    
    # Agentic / CoT Markers
    df = df.withColumn("agentic_markers", F.size(F.split("text", AGENTIC_MARKERS)) - 1)
    df = df.withColumn("cot_markers", F.size(F.split(F.lower("text"), COT_MARKERS)) - 1)
    df = df.withColumn("research_paper_markers", F.size(F.split("text", RESEARCH_PAPER_MARKERS)) - 1)
    
    df = df.withColumn("reasoning_marker_density", F.size(F.split(F.lower("text"), r"(\btherefore\b|\bthus\b|\bimplies\b|\bbecause\b)")) - 1)
    
    # -------------------------------------------------------------------------
    # STEP 3: TIER 1 REJECTION (Group 1 - Fast)
    # -------------------------------------------------------------------------
    # 1. Length
    c1 = (F.col("byte_length") < 50) | (F.col("byte_length") > 1000000)
    c2 = (F.col("char_length") < 20) | (F.col("char_length") > 500000)
    # 2. Corruption
    c3 = (F.col("non_printable_ratio") > 0.01)
    # 3. Spam/Noise
    c4 = (F.col("html_tag_density") > 0.05)
    c5 = (F.col("boilerplate_count") > 4) # Proxy for ratio > 0.15
    c6 = (F.col("risky_tld_count") > 0)
    # 4. Quality (Link farm)
    # Need tokens for exact ratio, but can use chars approx. 
    # url chars approx = url_count * 20. 
    c7 = ((F.col("url_count") * 20) / F.col("char_length")) > 0.3 
    
    df = df.withColumn("t1_rejected", c1 | c2 | c3 | c4 | c5 | c6 | c7)
    
    df = df.withColumn("t1_reason",
        F.when(c1 | c2, F.lit("Length Outlier"))
        .when(c3, F.lit("Corruption"))
        .when(c4, F.lit("HTML Spam"))
        .when(c5, F.lit("Boilerplate"))
        .when(c6, F.lit("Risky TLD"))
        .when(c7, F.lit("Link Farm"))
        .otherwise(None)
    )

    # -------------------------------------------------------------------------
    # STEP 4: TIER 2 METRICS (UDF - Slow) - CONDITIONAL EXECUTION
    # -------------------------------------------------------------------------
    # Strategy: If t1_rejected, pass NULL to UDF. UDF returns 0s. 
    # This avoids expensive tiktoken/regex on garbage data.
    
    df = df.withColumn("text_for_udf", F.when(F.col("t1_rejected"), F.lit(None).cast(StringType())).otherwise(F.col("text")))
    
    # Apply UDF using INLINE PROJECTION (No Join)
    df = df.withColumn("py_metrics", compute_python_metrics(F.col("text_for_udf")))
    
    # Flatten py_metrics struct
    # We select all existing columns + expanded py_metrics
    # Note: select statement must cover everything we want to keep.
    
    # -------------------------------------------------------------------------
    # STEP 5: TIER 2 REJECTION (Group 2 - Tokens/Deep)
    # -------------------------------------------------------------------------
    # Access metrics via py_metrics.field
    
    pm = F.col("py_metrics")
    
    # 5. Context
    c8 = (pm.getField("token_count_estimate") < 10) | (pm.getField("token_count_estimate") > 128000)
    # 6. Repetition
    c9 = (pm.getField("unique_token_ratio") < 0.1)
    # 7. Entropy
    c10 = (pm.getField("compression_ratio") > 0.95)
    # 8. Readability
    c11 = (pm.getField("flesch_reading_ease") < 0) | (pm.getField("flesch_reading_ease") > 120)
    # 9. Structure
    c12 = (pm.getField("sentence_count_estimate") < 2) & (pm.getField("token_count_estimate") > 100)
    c13 = (pm.getField("avg_sentence_length") > 500)
    # 10. Formatting
    c14 = (F.col("capitalization_ratio") > 0.5)
    c15 = (F.col("whitespace_ratio") > 0.6)
    # 11. Content (Orphaned thread)
    c17 = (F.col("thread_fragment_indicator") > 2) & (pm.getField("token_count_estimate") < 200)
    
    # Combined Rejection
    df = df.withColumn("t2_rejected", c8 | c9 | c10 | c11 | c12 | c13 | c14 | c15 | c17)
    
    df = df.withColumn("t2_reason",
        F.when(c8, F.lit("Token Outlier"))
        .when(c9, F.lit("Repetitive"))
        .when(c10, F.lit("High Entropy"))
        .when(c11, F.lit("Readability Nonsense"))
        .when(c12 | c13, F.lit("Structure malformed"))
        .when(c14 | c15, F.lit("Bad Formatting"))
        .when(c17, F.lit("Orphaned Thread"))
        .otherwise(None)
    )
    
    # Final is_rejected
    df = df.withColumn("is_rejected", F.col("t1_rejected") | (F.when(F.col("t1_rejected"), F.lit(False)).otherwise(F.col("t2_rejected"))))
    df = df.withColumn("rejection_reason", F.coalesce(F.col("t1_reason"), F.col("t2_reason")))

    # -------------------------------------------------------------------------
    # STEP 6: DERIVED SCORES (Post-compute)
    # -------------------------------------------------------------------------
    df = df.withColumn("structural_complexity_score", 
        (pm.getField("avg_sentence_length") * 0.5) + (pm.getField("avg_word_length") * 2))
        
    df = df.withColumn("noise_score", 
        F.col("html_tag_density") + (F.col("boilerplate_count") * 0.1) + F.col("symbol_density"))
    
    df = df.withColumn("domain_signal", 
        F.when(F.col("code_block_count") > 0, F.lit("code"))
        .when(F.col("equation_density") > 0, F.lit("math"))
        .when(F.col("research_paper_markers") > 1, F.lit("academic"))
        .otherwise(F.lit("general")))

    # Agentic Densities
    df = df.withColumn("t_count", pm.getField("token_count_estimate"))
    df = df.withColumn("agentic_density", F.when(F.col("t_count")>0, F.col("agentic_markers") / F.col("t_count")).otherwise(0.0))
    df = df.withColumn("cot_density", F.when(F.col("t_count")>0, F.col("cot_markers") / F.col("t_count")).otherwise(0.0))
    
    # Booleans
    df = df.withColumn("has_code", F.col("code_block_count") > 0)
    df = df.withColumn("has_math", (F.col("equation_density") > 0) | (F.col("math_expression_count") > 0))
    df = df.withColumn("has_reasoning", F.col("reasoning_marker_density") > 0)
    df = df.withColumn("has_research_paper", (F.col("research_paper_markers") > 0) & (F.col("citation_count") > 2))
    df = df.withColumn("has_agentic", F.col("agentic_markers") > 0)
    df = df.withColumn("has_cot", F.col("cot_markers") > 0)
    df = df.withColumn("primary_modality", F.col("domain_signal"))

    # -------------------------------------------------------------------------
    # OPTIMIZATION: CHECKPOINTING + DISK PERSISTENCE (Lineage Break + Performance)
    # -------------------------------------------------------------------------
    # 1. Break Lineage
    df = df.checkpoint() 
    
    # 2. Persist to Disk to serve both Team1 and Team2 writes from local disk (avoid S3 re-read)
    df.persist(StorageLevel.DISK_ONLY)
    
    # -------------------------------------------------------------------------
    # STEP 7: OUTPUT & WRITE (ZSTD)
    # -------------------------------------------------------------------------
    
    # Prepare Team 1 Output (annotated raw)
    # id, hash, dataset, domain, source, text, language, metadata, added, created, version, is_rejected, rejection_reason
    cols_t1 = [
        "id", "dataset", "domain", "source", "text", "metadata", "added", "created", 
        "is_rejected", "rejection_reason"
    ]
    
    df_t1 = df.select(
        *[F.col(c) for c in cols_t1],
        F.sha2(F.col("text"), 256).alias("hash"),
        F.lit("en").alias("language"),
        F.lit("1.0").alias("version")
    )
    
    print("Writing Team 1 Output (Raw + Rejection info)...")
    # Partition by Source
    df_t1.write.mode("overwrite") \
        .partitionBy("source") \
        .option("compression", "zstd") \
        .parquet(output_base_t1)
        
    # Prepare Team 2 Output (Metrics)
    # Flatten Struct columns for final output
    df_t2 = df.select(
        "id", "file_path", "is_rejected", "rejection_reason",
        "byte_length", "char_length", "line_count", "word_count", "avg_line_length",
        "whitespace_ratio", "capitalization_ratio", "symbol_density", "punctuation_density",
        "url_count", "email_count", "num_numeric_tokens", "question_density", "citation_count",
        "html_tag_density", "boilerplate_count", "risky_tld_count", "thread_fragment_indicator",
        "code_block_count", "equation_density", "math_expression_count", "reasoning_marker_density",
        "structural_complexity_score", "noise_score", "domain_signal",
        "has_code", "has_math", "has_reasoning", "has_research_paper", 
        "has_agentic", "agentic_markers", "agentic_density",
        "has_cot", "cot_markers", "cot_density", "primary_modality",
        "research_paper_markers",
        # Flattened UDF cols
        F.col("py_metrics.token_count_estimate").alias("token_count_estimate"),
        F.col("py_metrics.unique_token_ratio").alias("unique_token_ratio"),
        F.col("py_metrics.vocab_size").alias("vocab_size"),
        F.col("py_metrics.fertility").alias("fertility"),
        F.col("py_metrics.rare_word_ratio").alias("rare_word_ratio"),
        F.col("py_metrics.compression_ratio").alias("compression_ratio"),
        F.col("py_metrics.flesch_reading_ease").alias("flesch_reading_ease"),
        F.col("py_metrics.avg_word_length").alias("avg_word_length"),
        F.col("py_metrics.sentence_count_estimate").alias("sentence_count_estimate"),
        F.col("py_metrics.avg_sentence_length").alias("avg_sentence_length"),
        "domain", "source"
    )
    
    # We need to add domain/source back if we dropped them in select
    # Actually they are in 'df', I missed selecting them in df_t2 above. 
    # But wait, we partition by them, so they must be in the DF. 
    # 'domain' and 'source' are in `df`.
        
    print("Writing Team 2 Output (Metrics)...")
    # Partition by domain, source
    df_t2.write.mode("overwrite") \
        .partitionBy("domain", "source") \
        .option("compression", "zstd") \
        .parquet(output_base_t2)
        
    job.commit()

if __name__ == '__main__':
    main()
