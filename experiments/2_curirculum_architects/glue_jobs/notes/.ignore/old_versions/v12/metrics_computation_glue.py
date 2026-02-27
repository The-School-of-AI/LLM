"""
Curriculum Metrics Computation Glue Job

Computes text quality and curriculum metrics on Team 1's parquet data,
with early rejection optimization to skip expensive computations.

Key Features:
- Reads Team 1's parquet files without modifying them
- Computes 59 metrics in rejection priority order
- Early termination when rejection criteria met
- Outputs separate metrics parquet with join keys
- Optimized for 1TB+ datasets
"""

import sys
import re
import zlib
import uuid
from typing import Dict, Tuple, Optional
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    FloatType, BooleanType, ArrayType
)

# ============================================================================
# GLUE JOB SETUP
# ============================================================================

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "TEAM1_INPUT_PATH",      # s3://bucket/parquet/dolma/ (Team 1's output)
        "METRICS_OUTPUT_PATH",    # s3://bucket/metrics/dolma/ (our output)
        "TIKTOKEN_MODEL",         # cl100k_base (default for GPT-4)
        "NUM_PARTITIONS",         # 400 (tune based on cluster)
    ],
)

sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args["JOB_NAME"], args)

# Configuration
TEAM1_INPUT = args["TEAM1_INPUT_PATH"]
METRICS_OUTPUT = args["METRICS_OUTPUT_PATH"]
TIKTOKEN_MODEL = args.get("TIKTOKEN_MODEL", "cl100k_base")
NUM_PARTITIONS = int(args.get("NUM_PARTITIONS", "400"))

# ============================================================================
# METRIC COMPUTATION FUNCTIONS (Optimized for PySpark)
# ============================================================================

# Compiled regex patterns (shared across executors via broadcast)
PATTERNS = {
    'url': re.compile(r'https?://[^\s]+'),
    'sentence': re.compile(r'[.!?]+\s+'),
    'reasoning': re.compile(r'\b(therefore|thus|hence|because|since|consequently)\b', re.IGNORECASE),
    'math_expr': re.compile(r'[\$\^\{\}\\\[\]]|\\[a-zA-Z]+'),
    'step': re.compile(r'\b(step\s+\d+|first|second|third|next|finally)\b', re.IGNORECASE),
    'list_marker': re.compile(r'^\s*[\d\-\*\+]+[\.\)]\s+', re.MULTILINE),
    'truncation': re.compile(r'(\.\.\.|…|\[truncated\]|\[cut\])', re.IGNORECASE),
    'code_fence': re.compile(r'```'),
    'heading': re.compile(r'^#{1,6}\s+|\n={3,}|\n-{3,}', re.MULTILINE),
    'citation': re.compile(r'\[[0-9]+\]|\([A-Za-z]+\s+\d{4}\)'),
}


def generate_uuid() -> str:
    """Generate UUID for metric record"""
    return str(uuid.uuid4())


def compute_basic_metrics(text: str) -> Dict:
    """
    PRIORITY 1 metrics - fastest, most fundamental checks
    These are computed first for early rejection
    """
    if text is None or not isinstance(text, str):
        return {
            'byte_length': 0,
            'char_length': 0,
            'token_count_estimate': 0,
            'non_printable_ratio': 1.0,
            'line_count': 0,
        }
    
    byte_length = len(text.encode('utf-8'))
    char_length = len(text)
    line_count = text.count('\n') + 1
    
    # Non-printable character ratio
    non_printable = sum(1 for c in text if ord(c) < 32 or ord(c) == 127)
    non_printable_ratio = non_printable / max(char_length, 1)
    
    # Token estimate (simple approximation, ~4 chars per token)
    token_count_estimate = max(1, char_length // 4)
    
    return {
        'byte_length': byte_length,
        'char_length': char_length,
        'token_count_estimate': token_count_estimate,
        'non_printable_ratio': round(non_printable_ratio, 6),
        'line_count': line_count,
    }


def check_priority1_rejection(metrics: Dict) -> Tuple[bool, Optional[str]]:
    """Check Priority 1 rejection criteria"""
    
    # byte_length: <50 OR >1,000,000
    if metrics['byte_length'] < 50:
        return True, "byte_length too short (<50): lacks context"
    if metrics['byte_length'] > 1_000_000:
        return True, "byte_length too long (>1M): exceeds processing limits"
    
    # char_length: <20 OR >500,000
    if metrics['char_length'] < 20:
        return True, "char_length too short (<20): insufficient learning signal"
    if metrics['char_length'] > 500_000:
        return True, "char_length too long (>500K): exceeds single-pass processing"
    
    # token_count_estimate: <10 OR >128,000
    if metrics['token_count_estimate'] < 10:
        return True, "token_count too low (<10): noise/meaningless"
    if metrics['token_count_estimate'] > 128_000:
        return True, "token_count too high (>128K): exceeds context window"
    
    # non_printable_ratio: >0.01
    if metrics['non_printable_ratio'] > 0.01:
        return True, "non_printable_ratio too high (>1%): encoding corruption"
    
    return False, None


def compute_lexical_metrics(text: str, char_length: int) -> Dict:
    """
    PRIORITY 2 metrics - lexical diversity and noise detection
    Only computed if Priority 1 checks pass
    """
    if not text or char_length == 0:
        return {
            'unique_token_ratio': 0.0,
            'vocab_size': 0,
            'compression_ratio': 0.0,
            'capitalization_ratio': 0.0,
            'whitespace_ratio': 0.0,
            'symbol_density': 0.0,
        }
    
    # Simple tokenization (whitespace split)
    tokens = text.split()
    token_count = len(tokens)
    unique_tokens = len(set(tokens)) if tokens else 0
    unique_token_ratio = unique_tokens / max(token_count, 1)
    
    # Compression ratio (entropy proxy)
    try:
        compressed = zlib.compress(text.encode('utf-8'), level=6)
        compression_ratio = len(compressed) / max(len(text.encode('utf-8')), 1)
    except:
        compression_ratio = 0.5  # default if compression fails
    
    # Character type ratios
    uppercase = sum(1 for c in text if c.isupper())
    whitespace = sum(1 for c in text if c.isspace())
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    
    capitalization_ratio = uppercase / max(char_length, 1)
    whitespace_ratio = whitespace / max(char_length, 1)
    symbol_density = symbols / max(char_length, 1)
    
    return {
        'unique_token_ratio': round(unique_token_ratio, 6),
        'vocab_size': unique_tokens,
        'compression_ratio': round(compression_ratio, 6),
        'capitalization_ratio': round(capitalization_ratio, 6),
        'whitespace_ratio': round(whitespace_ratio, 6),
        'symbol_density': round(symbol_density, 6),
    }


def check_priority2_rejection(
    metrics: Dict, 
    text: str, 
    token_count: int
) -> Tuple[bool, Optional[str]]:
    """Check Priority 2 rejection criteria"""
    
    # unique_token_ratio: <0.1
    if metrics.get('unique_token_ratio', 1.0) < 0.1:
        return True, "unique_token_ratio too low (<0.1): template/repetitive content"
    
    # compression_ratio: >0.95
    if metrics.get('compression_ratio', 0.0) > 0.95:
        return True, "compression_ratio too high (>0.95): random/encrypted/binary data"
    
    # capitalization_ratio: >0.5
    if metrics.get('capitalization_ratio', 0.0) > 0.5:
        return True, "capitalization_ratio too high (>50%): ALL CAPS spam/shouting"
    
    # whitespace_ratio: >0.6
    if metrics.get('whitespace_ratio', 0.0) > 0.6:
        return True, "whitespace_ratio too high (>60%): mostly empty/formatting artifacts"
    
    # Compute additional metrics needed for compound checks
    truncation_count = len(PATTERNS['truncation'].findall(text))
    if truncation_count > 2:
        return True, f"truncation_indicators too high (>2): incomplete content ({truncation_count} signals)"
    
    # sentence_count_estimate: <2 AND token_count>100 (compound check)
    sentence_count = len(PATTERNS['sentence'].split(text))
    if sentence_count < 2 and token_count > 100:
        return True, "sentence_count low (<2) with high token_count: parsing failure"
    
    # noise_score computation (simplified composite)
    noise_score = (
        metrics.get('capitalization_ratio', 0.0) * 0.3 +
        metrics.get('whitespace_ratio', 0.0) * 0.3 +
        (1.0 - metrics.get('unique_token_ratio', 1.0)) * 0.2 +
        metrics.get('non_printable_ratio', 0.0) * 0.2
    )
    
    if noise_score > 0.6:
        return True, f"noise_score too high (>0.6): low-quality content (score={noise_score:.3f})"
    
    metrics['truncation_indicators'] = truncation_count
    metrics['sentence_count_estimate'] = sentence_count
    metrics['noise_score'] = round(noise_score, 6)
    
    return False, None


def compute_structural_metrics(text: str, sentence_count: int, char_length: int) -> Dict:
    """
    PRIORITY 3 metrics - complex structural analysis
    Only computed if Priority 1 & 2 checks pass
    """
    if not text or char_length == 0:
        return {
            'avg_line_length': 0.0,
            'avg_sentence_length': 0.0,
            'punctuation_density': 0.0,
            'avg_word_length': 0.0,
        }
    
    line_count = text.count('\n') + 1
    avg_line_length = char_length / max(line_count, 1)
    avg_sentence_length = char_length / max(sentence_count, 1)
    
    # Punctuation
    punctuation = sum(1 for c in text if c in '.,;:!?')
    punctuation_density = punctuation / max(char_length, 1)
    
    # Word length
    words = text.split()
    avg_word_length = sum(len(w) for w in words) / max(len(words), 1) if words else 0.0
    
    return {
        'avg_line_length': round(avg_line_length, 2),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'punctuation_density': round(punctuation_density, 6),
        'avg_word_length': round(avg_word_length, 2),
    }


def compute_pattern_metrics(text: str, token_count: int) -> Dict:
    """Pattern-based metrics (URLs, questions, citations, etc.)"""
    if not text:
        return {
            'url_count': 0,
            'question_density': 0.0,
            'citation_count': 0,
            'reasoning_marker_density': 0.0,
            'math_expression_count': 0,
            'step_indicator_count': 0,
            'list_marker_count': 0,
            'code_block_count': 0,
            'heading_count': 0,
        }
    
    url_count = len(PATTERNS['url'].findall(text))
    questions = text.count('?')
    question_density = questions / max(token_count, 1)
    
    return {
        'url_count': url_count,
        'question_density': round(question_density, 6),
        'citation_count': len(PATTERNS['citation'].findall(text)),
        'reasoning_marker_density': round(len(PATTERNS['reasoning'].findall(text)) / max(token_count, 1), 6),
        'math_expression_count': len(PATTERNS['math_expr'].findall(text)),
        'step_indicator_count': len(PATTERNS['step'].findall(text)),
        'list_marker_count': len(PATTERNS['list_marker'].findall(text)),
        'code_block_count': len(PATTERNS['code_fence'].findall(text)),
        'heading_count': len(PATTERNS['heading'].findall(text)),
    }


def check_priority3_rejection(metrics: Dict, text: str, token_count: int) -> Tuple[bool, Optional[str]]:
    """Check Priority 3 rejection criteria"""
    
    # avg_sentence_length: >500
    if metrics.get('avg_sentence_length', 0.0) > 500:
        return True, "avg_sentence_length too high (>500): run-on sentences/parsing errors"
    
    # url_count: url_ratio>0.3
    url_ratio = metrics.get('url_count', 0) / max(token_count, 1)
    if url_ratio > 0.3:
        return True, f"url_ratio too high (>0.3): link spam/scraping artifacts ({url_ratio:.2%})"
    
    # Flesch Reading Ease (simplified calculation)
    sentences = max(metrics.get('sentence_count_estimate', 1), 1)
    words = max(token_count, 1)
    syllables = words * 1.5  # rough approximation
    
    flesch = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    metrics['flesch_reading_ease'] = round(flesch, 2)
    
    if flesch < 0 or flesch > 120:
        return True, f"flesch_reading_ease out of range (0-120): calculation error or extreme outlier (score={flesch:.1f})"
    
    # Dependency depth (bracket nesting)
    max_depth = 0
    current_depth = 0
    for c in text[:10000]:  # sample first 10K chars for performance
        if c in '([{':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif c in ')]}':
            current_depth = max(0, current_depth - 1)
    
    metrics['dependency_depth_estimate'] = max_depth
    
    if max_depth > 20:
        return True, f"dependency_depth too high (>20): malformed code/data corruption (depth={max_depth})"
    
    # Sentence boundary coherence (simplified heuristic)
    # Check if sentences have reasonable endings
    sentences_text = PATTERNS['sentence'].split(text)
    valid_endings = sum(1 for s in sentences_text[:100] if len(s.strip()) > 5)  # sample first 100
    coherence = valid_endings / max(len(sentences_text[:100]), 1)
    metrics['sentence_boundary_coherence'] = round(coherence, 6)
    
    if coherence < 0.5:
        return True, f"sentence_boundary_coherence too low (<0.5): parsing/extraction failures (score={coherence:.2f})"
    
    # Information density (rough approximation: ratio of alphabetic chars to total)
    alpha_chars = sum(1 for c in text if c.isalpha())
    info_density = alpha_chars / max(len(text), 1)
    metrics['information_density'] = round(info_density, 6)
    
    if info_density < 0.2:
        return True, f"information_density too low (<0.2): mostly filler/function words (density={info_density:.2%})"
    
    return False, None


def compute_derived_metrics(all_metrics: Dict) -> Dict:
    """
    Compute derived/composite metrics from already calculated ones
    These are never rejection metrics, only for curriculum ordering
    """
    # Structural complexity score (normalized 0-1)
    structural_score = (
        min(all_metrics.get('sentence_count_estimate', 0) / 100, 1.0) * 0.3 +
        min(all_metrics.get('avg_sentence_length', 0) / 100, 1.0) * 0.2 +
        min(all_metrics.get('dependency_depth_estimate', 0) / 10, 1.0) * 0.3 +
        all_metrics.get('symbol_density', 0.0) * 0.2
    )
    
    # Domain signal (simple heuristic)
    code_score = (
        all_metrics.get('code_block_count', 0) * 0.4 +
        all_metrics.get('symbol_density', 0.0) * 100 * 0.3 +
        all_metrics.get('avg_line_length', 0) / 100 * 0.3
    )
    
    math_score = (
        all_metrics.get('math_expression_count', 0) * 0.6 +
        all_metrics.get('equation_density', 0.0) * 100 * 0.4
    )
    
    dialogue_score = (
        all_metrics.get('question_density', 0.0) * 1000 * 0.5 +
        all_metrics.get('dialogue_turn_count', 0) * 0.5
    )
    
    domain_scores = {
        'code': code_score,
        'math': math_score,
        'dialogue': dialogue_score,
        'general': 1.0  # baseline
    }
    domain_signal = max(domain_scores, key=domain_scores.get)
    
    return {
        'structural_complexity_score': round(structural_score, 6),
        'domain_signal': domain_signal,
    }


def process_record_with_early_rejection(
    record_id: str,
    text: str,
    source_file: str
) -> Dict:
    """
    Main processing function with early rejection optimization
    Returns complete metrics dict with rejection status
    """
    result = {
        'metric_record_uuid': generate_uuid(),
        'source_record_id': record_id,
        'source_file_path': source_file,
        'is_rejected': False,
        'rejection_reason': None,
    }
    
    # ===== PRIORITY 1: Basic Metrics (Early Rejection) =====
    basic_metrics = compute_basic_metrics(text)
    result.update(basic_metrics)
    
    is_rejected, reason = check_priority1_rejection(basic_metrics)
    if is_rejected:
        result['is_rejected'] = True
        result['rejection_reason'] = f"[P1] {reason}"
        return result  # Early exit - skip all remaining metrics
    
    # ===== PRIORITY 2: Lexical & Noise Metrics =====
    lexical_metrics = compute_lexical_metrics(text, basic_metrics['char_length'])
    result.update(lexical_metrics)
    
    is_rejected, reason = check_priority2_rejection(
        result, 
        text, 
        basic_metrics['token_count_estimate']
    )
    if is_rejected:
        result['is_rejected'] = True
        result['rejection_reason'] = f"[P2] {reason}"
        return result  # Early exit
    
    # ===== PRIORITY 3: Structural Metrics =====
    structural_metrics = compute_structural_metrics(
        text,
        result['sentence_count_estimate'],
        basic_metrics['char_length']
    )
    result.update(structural_metrics)
    
    is_rejected, reason = check_priority3_rejection(
        result,
        text,
        basic_metrics['token_count_estimate']
    )
    if is_rejected:
        result['is_rejected'] = True
        result['rejection_reason'] = f"[P3] {reason}"
        return result  # Early exit
    
    # ===== NON-REJECTION METRICS (only if not rejected) =====
    pattern_metrics = compute_pattern_metrics(text, basic_metrics['token_count_estimate'])
    result.update(pattern_metrics)
    
    derived_metrics = compute_derived_metrics(result)
    result.update(derived_metrics)
    
    # Add remaining metrics as null (can be computed later if needed)
    result.update({
        'mtld': None,
        'fertility': None,
        'script_distribution': None,
        'code_language_hint': None,
        'rare_word_ratio': None,
        'num_numeric_tokens': None,
        'num_entities_estimate': None,
        'ellipsis_count': None,
        'table_count_estimate': None,
        'dialogue_turn_count': None,
        'visual_placeholder_count': None,
        'equation_density': None,
        'table_complexity': None,
        'few_shot_potential': None,
        'cross_domain_analogy_markers': None,
        'domain_specificity': None,
        'concept_density': None,
        'example_density': None,
        'prerequisite_density': None,
        'hedging_language_ratio': None,
        'counterargument_presence': None,
        'instruction_complexity': None,
    })
    
    return result


# ============================================================================
# SPARK UDF REGISTRATION
# ============================================================================

# Define return schema for the UDF
metrics_schema = StructType([
    StructField("metric_record_uuid", StringType(), False),
    StructField("source_record_id", StringType(), False),
    StructField("source_file_path", StringType(), False),
    StructField("is_rejected", BooleanType(), False),
    StructField("rejection_reason", StringType(), True),
    # Priority 1
    StructField("byte_length", IntegerType(), True),
    StructField("char_length", IntegerType(), True),
    StructField("token_count_estimate", IntegerType(), True),
    StructField("non_printable_ratio", FloatType(), True),
    StructField("line_count", IntegerType(), True),
    # Priority 2
    StructField("unique_token_ratio", FloatType(), True),
    StructField("vocab_size", IntegerType(), True),
    StructField("compression_ratio", FloatType(), True),
    StructField("capitalization_ratio", FloatType(), True),
    StructField("whitespace_ratio", FloatType(), True),
    StructField("symbol_density", FloatType(), True),
    StructField("truncation_indicators", IntegerType(), True),
    StructField("sentence_count_estimate", IntegerType(), True),
    StructField("noise_score", FloatType(), True),
    # Priority 3
    StructField("avg_line_length", FloatType(), True),
    StructField("avg_sentence_length", FloatType(), True),
    StructField("punctuation_density", FloatType(), True),
    StructField("avg_word_length", FloatType(), True),
    StructField("flesch_reading_ease", FloatType(), True),
    StructField("dependency_depth_estimate", IntegerType(), True),
    StructField("sentence_boundary_coherence", FloatType(), True),
    StructField("information_density", FloatType(), True),
    # Pattern metrics
    StructField("url_count", IntegerType(), True),
    StructField("question_density", FloatType(), True),
    StructField("citation_count", IntegerType(), True),
    StructField("reasoning_marker_density", FloatType(), True),
    StructField("math_expression_count", IntegerType(), True),
    StructField("step_indicator_count", IntegerType(), True),
    StructField("list_marker_count", IntegerType(), True),
    StructField("code_block_count", IntegerType(), True),
    StructField("heading_count", IntegerType(), True),
    # Derived
    StructField("structural_complexity_score", FloatType(), True),
    StructField("domain_signal", StringType(), True),
    # Placeholders for expensive metrics (can add later)
    StructField("mtld", FloatType(), True),
    StructField("fertility", FloatType(), True),
    StructField("script_distribution", StringType(), True),
    StructField("code_language_hint", StringType(), True),
    StructField("rare_word_ratio", FloatType(), True),
    StructField("num_numeric_tokens", IntegerType(), True),
    StructField("num_entities_estimate", IntegerType(), True),
    StructField("ellipsis_count", IntegerType(), True),
    StructField("table_count_estimate", IntegerType(), True),
    StructField("dialogue_turn_count", IntegerType(), True),
    StructField("visual_placeholder_count", IntegerType(), True),
    StructField("equation_density", FloatType(), True),
    StructField("table_complexity", FloatType(), True),
    StructField("few_shot_potential", FloatType(), True),
    StructField("cross_domain_analogy_markers", IntegerType(), True),
    StructField("domain_specificity", FloatType(), True),
    StructField("concept_density", FloatType(), True),
    StructField("example_density", FloatType(), True),
    StructField("prerequisite_density", FloatType(), True),
    StructField("hedging_language_ratio", FloatType(), True),
    StructField("counterargument_presence", BooleanType(), True),
    StructField("instruction_complexity", FloatType(), True),
])


@F.udf(returnType=metrics_schema)
def compute_metrics_udf(record_id: str, text: str, source_file: str):
    """Spark UDF wrapper for metrics computation"""
    return process_record_with_early_rejection(record_id, text, source_file)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main Glue job execution"""
    
    print(f"Reading Team 1 data from: {TEAM1_INPUT}")
    
    # Read Team 1's parquet files
    df_input = spark.read.parquet(TEAM1_INPUT)
    
    # Add source file path column using input_file_name()
    df_with_source = df_input.withColumn(
        "source_file_path",
        F.input_file_name()
    )
    
    print(f"Input record count: {df_with_source.count()}")
    print(f"Schema: {df_with_source.printSchema()}")
    
    # Compute metrics with early rejection
    print("Computing metrics with early rejection optimization...")
    
    df_metrics = df_with_source.select(
        compute_metrics_udf(
            F.col("id"),
            F.col("text"),
            F.col("source_file_path")
        ).alias("metrics")
    ).select("metrics.*")
    
    # Add processing timestamp
    df_metrics = df_metrics.withColumn(
        "processed_at",
        F.current_timestamp()
    )
    
    # Show rejection statistics before writing
    print("\n=== Rejection Statistics ===")
    rejection_stats = df_metrics.groupBy("is_rejected").count()
    rejection_stats.show()
    
    print("\n=== Top Rejection Reasons ===")
    rejection_reasons = (
        df_metrics
        .filter(F.col("is_rejected") == True)
        .groupBy("rejection_reason")
        .count()
        .orderBy(F.desc("count"))
        .limit(20)
    )
    rejection_reasons.show(truncate=False)
    
    # Write metrics to separate parquet file
    print(f"\nWriting metrics to: {METRICS_OUTPUT}")
    
    (
        df_metrics
        .repartition(NUM_PARTITIONS)
        .write
        .mode("overwrite")
        .option("compression", "zstd")
        .parquet(METRICS_OUTPUT)
    )
    
    print(f"✅ Metrics computation complete!")
    print(f"   Output: {METRICS_OUTPUT}")
    print(f"   Records processed: {df_metrics.count()}")
    
    job.commit()


if __name__ == "__main__":
    main()
