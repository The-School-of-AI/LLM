
import sys
import re
import boto3
import zlib
import math
from collections import Counter

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType, StructType, BooleanType, ArrayType
from pyspark.sql.functions import pandas_udf
import pandas as pd

# -------------------------------------------------------------------------
# CONFIGURATION (Embedded)
# -------------------------------------------------------------------------
DATASETS_CONFIG = [
    {"name": "dolmas_books_v1_7", "path": "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/books/", "source": "books", "domain": "literature"},
]
output_base_t1 = 's3://t1-dataacquisition-datasets/processed_dataset/raw_data'
output_base_t2 = 's3://t1-dataacquisition-datasets/processed_dataset/metrics'


# -------------------------------------------------------------------------
# PATTERNS
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

# Agentic / CoT Markers (Simple keyword presence)
AGENTIC_MARKERS = r"(Action:|Observation:|Tool:|Thought:|Plan:)"
COT_MARKERS = r"(Let's think step by step|Let's reason|chain of thought|reasoning:)"
RESEARCH_PAPER_MARKERS = r"(Abstract\b|Introduction\b|References\b|Bibliography\b|Conclusion\b)"

# -------------------------------------------------------------------------
# FUSED PANDAS UDF (Tier 2)
# -------------------------------------------------------------------------
# Must install: tiktoken, textstat
# We use one strict Struct output to avoid multiple serialization passes.
# Logic: calculate everything in one Python pass per batch.

@pandas_udf(StructType().add("token_count_estimate", IntegerType())
                        .add("unique_token_ratio", DoubleType())
                        .add("vocab_size", IntegerType())
                        .add("fertility", DoubleType())
                        .add("rare_word_ratio", DoubleType())
                        .add("compression_ratio", DoubleType())
                        .add("flesch_reading_ease", DoubleType())
                        .add("avg_word_length", DoubleType())
                        .add("sentence_count_estimate", IntegerType())
                        .add("avg_sentence_length", DoubleType()))
def compute_python_metrics(texts: pd.Series) -> pd.Series:
    import tiktoken
    import textstat
    
    # Initialize tokenizer once
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except:
        enc = None
        
    results = []
    
    for text in texts:
        if not text:
            results.append((0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0))
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
        # Avg Word Length
        avg_word_len = sum(len(w) for w in words) / n_words if n_words > 0 else 0.0
        
        # Sentence Count (Regex approx is faster than NLTK)
        # Simple split by [.!?]+
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
        
    return pd.Series(results)

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def get_glue_args():
    return getResolvedOptions(sys.argv, ['JOB_NAME'])

def main():
    args = get_glue_args()
    
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    
    # Enable AQE as requested
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    # Optimize for ~128MB partitions
    spark.conf.set("spark.sql.files.maxPartitionBytes", "134217728")
    
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)
    
    for dataset in DATASETS_CONFIG:
        name = dataset["name"]
        input_path = dataset["path"]
        domain = dataset["domain"]
        source = dataset["source"]
        
        print(f"Processing {name}...")
        
        try:
            # -----------------------------------------------------------------
            # STEP 1: READ
            # -----------------------------------------------------------------
            # Recursive lookup for nested folders
            df_input = spark.read.option("recursiveFileLookup", "true").json(input_path)
            
            # Ensure required columns
            if "text" not in df_input.columns:
                print(f"Skipping {name}: No text column")
                continue
                
            # Add file path
            df_input = df_input.withColumn("file_path", F.input_file_name())
            
            # -----------------------------------------------------------------
            # STEP 2: TEAM 1 OUTPUT (Raw Convert)
            # -----------------------------------------------------------------
            # Schema: id, hash, dataset, domain, source, text, language, metadata, added, created, version
            # We assume Input has id, text, added, created, metadata. We enrich others.
            
            # Helper to add if missing
            def safe_col(col_name, default=None):
                if col_name in df_input.columns:
                    return F.col(col_name)
                return F.lit(default)

            df_t1 = df_input.select(
                safe_col("id"),
                F.sha2(F.col("text"), 256).alias("hash"),
                F.lit(name).alias("dataset"),
                F.lit(domain).alias("domain"),
                F.lit(source).alias("source"),
                F.col("text"),
                F.lit("en").alias("language"),
                safe_col("metadata").cast("string"),
                safe_col("added"),
                safe_col("created"),
                F.lit("1.0").alias("version") # Default version
            )
            
            # Write Team 1 (One-to-One folder match)
            t1_out = f"{output_base_t1}/{name}/"
            # We rely on AQE to handle partition sizing here, or simple coalesce logic
            # Repartition based on size estimate is safer for 128MB target
            # For now, let AQE handle it via maxPartitionBytes
            df_t1.write.mode("overwrite").parquet(t1_out)
            
            # -----------------------------------------------------------------
            # STEP 3: METRICS (Team 2)
            # -----------------------------------------------------------------
            # Tier 1: Regex & Lengths (Native Spark)
            
            df = df_input.select("id", "text", "file_path")
            
            # Physical
            df = df.withColumn("byte_length", F.length(F.encode("text", "utf-8")))
            df = df.withColumn("char_length", F.length("text"))
            df = df.withColumn("line_count", F.size(F.split("text", "\n")))
            df = df.withColumn("word_count", F.size(F.split("text", "\\s+")))
            df = df.withColumn("avg_line_length", F.col("char_length") / F.col("line_count"))
            
            # Structure/Quality
            df = df.withColumn("whitespace_ratio", F.length(F.regexp_replace("text", r"[^\s]", "")) / F.col("char_length"))
            
            # Capitalization: Remove non-alpha, then count Upper
            df = df.withColumn("alpha_chars", F.length(F.regexp_replace("text", r"[^a-zA-Z]", "")))
            df = df.withColumn("upper_chars", F.length(F.regexp_replace("text", r"[^A-Z]", "")))
            df = df.withColumn("capitalization_ratio", F.when(F.col("alpha_chars")>0, F.col("upper_chars")/F.col("alpha_chars")).otherwise(0.0))
            
            # Punctuation/Symbols
            # Punct: [!,;:."?] etc. 
            # Symbol: Remove alphanum + space.
            df = df.withColumn("symbol_chars", F.length(F.regexp_replace("text", r"[a-zA-Z0-9\s]", "")))
            df = df.withColumn("symbol_density", F.col("symbol_chars") / F.col("char_length"))
            df = df.withColumn("punctuation_density", F.col("symbol_density")) # Approx shared
            
            # URL/Email/Numbers
            df = df.withColumn("url_count", F.size(F.split("text", r"https?://\S+")) - 1)
            df = df.withColumn("email_count", F.size(F.split("text", r"[\w\.-]+@[\w\.-]+\.\w+")) - 1)
            df = df.withColumn("num_numeric_tokens", F.size(F.split("text", r"[0-9]+")) - 1)
            
            # Team 3 Specifics
            # HTML Tag Density
            df = df.withColumn("no_tags_len", F.length(F.regexp_replace("text", r"<[^>]+>", "")))
            df = df.withColumn("html_tag_density", (F.col("char_length") - F.col("no_tags_len")) / F.col("char_length"))
            
            # Boilerplate
            df = df.withColumn("boilerplate_count", F.size(F.split(F.lower("text"), BOILERPLATE_REGEX)) - 1)
            
            # Risky TLDs
            df = df.withColumn("risky_tld_count", F.size(F.split(F.lower("text"), RISKY_TLDS)) - 1)
            
            # Thread Fragments
            df = df.withColumn("thread_fragment_indicator", F.size(F.split(F.lower("text"), THREAD_REGEX)) - 1)

            # Cognitive/Reasoning/Domain
            df = df.withColumn("question_density", F.size(F.split("text", r"\?")) - 1)
            df = df.withColumn("citation_count", F.size(F.split("text", r"\[\d+\]")) - 1)
            df = df.withColumn("list_marker_count", F.size(F.split("text", r"^\s*[\-\*]\s+")) - 1) # Simple bullet check
            df = df.withColumn("math_expression_count", F.size(F.split("text", r"[\+\-\*/\^=]{2,}")) - 1) # Approx math
            df = df.withColumn("equation_density", F.size(F.split("text", r"\$.+?\$")) - 1) # LaTeX style
            df = df.withColumn("code_block_count", F.size(F.split("text", "```")) - 1)
            
            # Agentic / CoT Markers
            df = df.withColumn("agentic_markers", F.size(F.split("text", AGENTIC_MARKERS)) - 1)
            df = df.withColumn("cot_markers", F.size(F.split(F.lower("text"), COT_MARKERS)) - 1)
            df = df.withColumn("research_paper_markers", F.size(F.split("text", RESEARCH_PAPER_MARKERS)) - 1)
            
            # Reasoning Markers
            df = df.withColumn("reasoning_marker_density", F.size(F.split(F.lower("text"), r"(\btherefore\b|\bthus\b|\bimplies\b|\bbecause\b)")) - 1)
            
            # Tier 1 Rejection Logic (Fast)
            # ---------------------------------------------------------
            cond_len = (F.col("byte_length") < 50) | (F.col("byte_length") > 1000000)
            cond_noise = (F.col("html_tag_density") > 0.05) | (F.col("symbol_density") > 0.5)
            
            # We flag early but we must run Tier 2 for everyone who Passes T1 OR if we want metrics for all?
            # User requirement: "if a record is rejected... do not want to compute rest"
            # So we create a flag here.
            df = df.withColumn("t1_rejected", cond_len | cond_noise)
            
            # Tier 2: Python UDF
            # ---------------------------------------------------------
            # We pass 'text' to UDF. 
            # Optimization: Pass null or empty string if t1_rejected to save compute? 
            # Pandas UDF receives batch. We can mask text.
            
            df = df.withColumn("text_for_udf", F.when(F.col("t1_rejected"), F.lit("")).otherwise(F.col("text")))
            
            # Apply UDF
            df_udf = df.select("id", compute_python_metrics("text_for_udf").alias("py_metrics"))
            
            # Expand Struct
            df = df.join(df_udf, on="id", how="inner") # 1-to-1 join
            df = df.select("*", "py_metrics.*").drop("py_metrics", "text_for_udf")
            
            # Tier 2 Rejection Logic (Tokens)
            # ---------------------------------------------------------
            cond_tokens = (F.col("token_count_estimate") < 10) | (F.col("token_count_estimate") > 128000)
            
            # Final Rejection Reason
            df = df.withColumn("rejection_reason",
                F.when(cond_len, F.lit("Length Outlier"))
                .when(cond_noise, F.lit("High Noise/HTML"))
                .when(cond_tokens, F.lit("Token Count Outlier"))
                .when(F.col("risky_tld_count") > 0, F.lit("Spam TLD"))
                .when(F.col("boilerplate_count") > 3, F.lit("Boilerplate"))
                .otherwise(None)
            )
            df = df.withColumn("is_rejected", F.col("rejection_reason").isNotNull())
            
            # Derived Scores
            # ---------------------------------------------------------
            df = df.withColumn("structural_complexity_score", 
                (F.col("avg_sentence_length") * 0.5) + (F.col("avg_word_length") * 2))
                
            df = df.withColumn("noise_score", 
                F.col("html_tag_density") + (F.col("boilerplate_count") * 0.1) + F.col("symbol_density"))
            
            # Domain Signal
            df = df.withColumn("domain_signal", 
                F.when(F.col("code_block_count") > 0, F.lit("code"))
                .when(F.col("equation_density") > 0, F.lit("math"))
                .when(F.col("research_paper_markers") > 1, F.lit("academic"))
                .otherwise(F.lit("general")))
                
            # Boolean Flags & Densities
            df = df.withColumn("has_code", F.col("code_block_count") > 0)
            df = df.withColumn("has_math", (F.col("equation_density") > 0) | (F.col("math_expression_count") > 0))
            df = df.withColumn("has_reasoning", F.col("reasoning_marker_density") > 0)
            df = df.withColumn("has_research_paper", (F.col("research_paper_markers") > 0) & (F.col("citation_count") > 2))
            
            df = df.withColumn("has_agentic", F.col("agentic_markers") > 0)
            df = df.withColumn("agentic_density", F.col("agentic_markers") / F.col("token_count_estimate"))
            
            df = df.withColumn("has_cot", F.col("cot_markers") > 0)
            df = df.withColumn("cot_density", F.col("cot_markers") / F.col("token_count_estimate"))
            
            df = df.withColumn("primary_modality", F.col("domain_signal")) # Simple alias for now

            # Select output columns
            final_cols = [
                "id", "file_path", "is_rejected", "rejection_reason",
                "byte_length", "char_length", "line_count", "word_count", "avg_line_length",
                "token_count_estimate", "unique_token_ratio", "vocab_size", "fertility", "rare_word_ratio",
                "compression_ratio", "flesch_reading_ease", "avg_word_length", "sentence_count_estimate", "avg_sentence_length",
                "whitespace_ratio", "capitalization_ratio", "symbol_density", "punctuation_density",
                "url_count", "email_count", "num_numeric_tokens", "question_density", "citation_count",
                "html_tag_density", "boilerplate_count", "risky_tld_count", "thread_fragment_indicator",
                "code_block_count", "equation_density", "math_expression_count", "reasoning_marker_density",
                "structural_complexity_score", "noise_score", "domain_signal",
                "has_code", "has_math", "has_reasoning", "has_research_paper", 
                "has_agentic", "agentic_markers", "agentic_density",
                "has_cot", "cot_markers", "cot_density", "primary_modality",
                "research_paper_markers"
            ]
            
            df_out = df.select(*[F.col(c) for c in final_cols if c in df.columns])
            
            # Write Team 2
            t2_out = f"{output_base_t2}/domain={domain}/source={source}"
            df_out.write.mode("overwrite").parquet(t2_out)
            
            # Cleanup
            df_input.unpersist()
            df.unpersist()
            
        except Exception as e:
            print(f"FAILED {name}: {e}")
            import traceback
            traceback.print_exc()

    job.commit()

if __name__ == '__main__':
    main()
