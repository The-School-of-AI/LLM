"""
PipelineConfig — typed dataclass tree mirroring pipeline.yaml.
All pipeline code reads from this; zero hardcoded values elsewhere.
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------

@dataclass
class GlobalConfig:
    seed: int = 42
    log_level: str = "INFO"
    work_dir: str = "./run_outputs"
    output_dir: str = "./outputs"
    shard_size_mb: int = 512
    num_proc: int = 4
    arrow_compression: str = "lz4"
    save_intermediates: bool = True
    resume_from_stage: int = 0


# ---------------------------------------------------------------------------
# Stage 0 — Data Mixing
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    name: str = ""
    path: str | None = None
    hf_id: str | None = None
    hf_split: str = "train"
    format: str = "already_conversation"   # alpaca | sharegpt | already_conversation
    weight: float = 1.0
    max_samples: int | None = None
    subsample_seed: int = 42


@dataclass
class Stage0Config:
    enabled: bool = True
    output_file: str = "stage0_mixed.jsonl"
    total_target: int | None = None
    sources: list[SourceConfig] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1 — Schema Validation
# ---------------------------------------------------------------------------

@dataclass
class Stage1Config:
    enabled: bool = True
    input_file: str = "stage0_mixed.jsonl"
    output_file: str = "stage1_validated.jsonl"
    rejected_file: str = "stage1_rejected.jsonl"
    accepted_formats: list[str] = field(default_factory=lambda: ["jsonl", "json", "parquet", "csv"])
    required_roles: list[str] = field(default_factory=lambda: ["user", "assistant"])
    allow_system_turn: bool = True
    allow_multi_turn: bool = True
    require_last_turn_is_assistant: bool = True
    min_turns: int = 2
    max_turns: int = 64
    min_content_chars: int = 1


# ---------------------------------------------------------------------------
# Stage 2 — Cleaning & Filtering (sub-configs per filter)
# ---------------------------------------------------------------------------

@dataclass
class LengthFilterConfig:
    enabled: bool = True
    min_chars: int = 40
    max_chars: int = 65536
    min_tokens: int = 10
    max_tokens: int = 16384
    tokenizer_name_or_path: str | None = None


@dataclass
class LangFilterConfig:
    enabled: bool = True
    backend: str = "langdetect"          # langdetect | fasttext
    fasttext_model_path: str | None = None
    allowed_languages: list[str] = field(default_factory=lambda: ["en", "hi", "ta", "te"])
    sample_from: str = "user"            # user | assistant | all
    min_confidence: float = 0.80
    on_error: str = "keep"              # keep | drop


@dataclass
class ExactDedupConfig:
    enabled: bool = True
    hash_mode: str = "prompt"           # prompt | full


@dataclass
class NearDedupConfig:
    enabled: bool = True
    num_perm: int = 128
    threshold: float = 0.85
    hash_field: str = "user_content"    # user_content | full_text | assistant_content
    ngram_size: int = 5


@dataclass
class ToxicityFilterConfig:
    enabled: bool = True
    backend: str = "keyword"            # keyword | detoxify
    keyword_blocklist_path: str = "config/blocklist.txt"
    detoxify_model: str = "unbiased"
    max_score: float = 0.80
    check_roles: list[str] = field(default_factory=lambda: ["user", "assistant"])
    action: str = "drop"


@dataclass
class PIIFilterConfig:
    enabled: bool = True
    backend: str = "regex"              # regex | presidio
    presidio_entities: list[str] = field(default_factory=lambda: [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"
    ])
    regex_patterns: dict[str, str] = field(default_factory=lambda: {
        "email": r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "ssn":   r"\b\d{3}-\d{2}-\d{4}\b",
    })
    action: str = "drop"               # drop | redact
    check_roles: list[str] = field(default_factory=lambda: ["user", "assistant"])


@dataclass
class RepetitionFilterConfig:
    enabled: bool = True
    max_repetition_ratio: float = 0.40
    ngram_size: int = 10
    check_roles: list[str] = field(default_factory=lambda: ["assistant"])


@dataclass
class SlopFilterConfig:
    enabled: bool = True
    slop_patterns_path: str = "config/slop_patterns.txt"
    max_slop_ratio: float = 0.30
    check_roles: list[str] = field(default_factory=lambda: ["assistant"])


@dataclass
class BenchmarkDecontamConfig:
    enabled: bool = True
    benchmark_hashes_dir: str = "config/benchmark_hashes/"
    benchmark_hash_files: list[str] = field(default_factory=list)
    removed_out: str = "stage2_decontam_removed.jsonl"


@dataclass
class Stage2Config:
    enabled: bool = True
    input_file: str = "stage1_validated.jsonl"
    output_file: str = "stage2_cleaned.jsonl"
    rejected_file: str = "stage2_rejected.jsonl"
    length_filter: LengthFilterConfig = field(default_factory=LengthFilterConfig)
    lang_filter: LangFilterConfig = field(default_factory=LangFilterConfig)
    exact_dedup: ExactDedupConfig = field(default_factory=ExactDedupConfig)
    near_dedup: NearDedupConfig = field(default_factory=NearDedupConfig)
    toxicity_filter: ToxicityFilterConfig = field(default_factory=ToxicityFilterConfig)
    pii_filter: PIIFilterConfig = field(default_factory=PIIFilterConfig)
    repetition_filter: RepetitionFilterConfig = field(default_factory=RepetitionFilterConfig)
    slop_filter: SlopFilterConfig = field(default_factory=SlopFilterConfig)
    benchmark_decontam: BenchmarkDecontamConfig = field(default_factory=BenchmarkDecontamConfig)


# ---------------------------------------------------------------------------
# Stage 3 — Chat Template Application
# ---------------------------------------------------------------------------

@dataclass
class SystemPromptsConfig:
    variation_strategy: str = "random"  # fixed | random | per_source
    prompts: list[str] = field(default_factory=lambda: [
        "You are a helpful assistant.",
        "You are an expert AI assistant.",
    ])


@dataclass
class Stage3Config:
    enabled: bool = True
    input_file: str = "stage2_cleaned.jsonl"
    output_file: str = "stage3_templated.jsonl"
    template: str = "chatml"            # chatml | llama3 | custom | tokenizer_native
    tokenizer_name_or_path: str | None = None
    custom_template_path: str | None = None
    add_generation_prompt: bool = False
    add_eos_token: bool = True
    eos_token: str = "<|im_end|>"
    system_prompts: SystemPromptsConfig = field(default_factory=SystemPromptsConfig)


# ---------------------------------------------------------------------------
# Stage 4 — Tokenization
# ---------------------------------------------------------------------------

@dataclass
class Stage4Config:
    enabled: bool = True
    input_file: str = "stage3_templated.jsonl"
    output_file: str = "stage4_tokenized.arrow"
    tokenizer_name_or_path: str = "meta-llama/Llama-3.1-70B"
    trust_remote_code: bool = False
    max_length: int = 8192
    overflow_strategy: str = "drop"    # drop | truncate | split
    truncate_side: str = "right"
    split_overlap: int = 64


# ---------------------------------------------------------------------------
# Stage 5 — Loss Masking
# ---------------------------------------------------------------------------

@dataclass
class Stage5Config:
    enabled: bool = True
    input_file: str = "stage4_tokenized.arrow"
    output_file: str = "stage5_masked.arrow"
    ignore_index: int = -100
    train_on_roles: list[str] = field(default_factory=lambda: ["assistant"])
    min_unmasked_tokens: int = 4
    mask_eos: bool = False


# ---------------------------------------------------------------------------
# Stage 6 — Quality Validation
# ---------------------------------------------------------------------------

@dataclass
class GateConfig:
    enabled: bool = True
    min_unmasked_ratio: float = 0.05
    max_zero_unmasked_fraction: float = 0.00
    min_p5_length: int = 10
    max_p95_length: int = 8192
    min_total_examples: int = 1000


@dataclass
class ReportConfig:
    compute_length_stats: bool = True
    compute_unmasked_ratio_stats: bool = True
    compute_per_source_counts: bool = True
    compute_per_language_counts: bool = True
    output_json: str = "quality_report.json"


@dataclass
class Stage6Config:
    enabled: bool = True
    input_file: str = "stage5_masked.arrow"
    output_dir_sharded: str = "final_shards/"
    review_sample_size: int = 200
    review_sample_seed: int = 42
    review_output: str = "quality_review_sample.jsonl"
    train_val_split_ratio: float = 0.05
    gate: GateConfig = field(default_factory=GateConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


# ---------------------------------------------------------------------------
# S3 Upload
# ---------------------------------------------------------------------------

@dataclass
class S3Config:
    enabled: bool = False
    bucket: str = "your-bucket"
    prefix: str = "sft_data/v1/"
    aws_profile: str | None = None
    checksum_verify: bool = True
    multipart_threshold_mb: int = 100


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    globals: GlobalConfig = field(default_factory=GlobalConfig)
    stage0: Stage0Config = field(default_factory=Stage0Config)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    stage4: Stage4Config = field(default_factory=Stage4Config)
    stage5: Stage5Config = field(default_factory=Stage5Config)
    stage6: Stage6Config = field(default_factory=Stage6Config)
    s3: S3Config = field(default_factory=S3Config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            raw: dict = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> "PipelineConfig":
        cfg = cls()
        if "global" in d:
            cfg.globals = _populate(GlobalConfig(), d["global"])
        if "stage0" in d:
            s0 = d["stage0"]
            cfg.stage0 = _populate(Stage0Config(), s0)
            cfg.stage0.sources = [_populate(SourceConfig(), s) for s in s0.get("sources", [])]
        if "stage1" in d:
            cfg.stage1 = _populate(Stage1Config(), d["stage1"])
        if "stage2" in d:
            s2 = d["stage2"]
            cfg.stage2 = _populate(Stage2Config(), s2)
            if "length_filter" in s2:
                cfg.stage2.length_filter = _populate(LengthFilterConfig(), s2["length_filter"])
            if "lang_filter" in s2:
                cfg.stage2.lang_filter = _populate(LangFilterConfig(), s2["lang_filter"])
            if "exact_dedup" in s2:
                cfg.stage2.exact_dedup = _populate(ExactDedupConfig(), s2["exact_dedup"])
            if "near_dedup" in s2:
                cfg.stage2.near_dedup = _populate(NearDedupConfig(), s2["near_dedup"])
            if "toxicity_filter" in s2:
                cfg.stage2.toxicity_filter = _populate(ToxicityFilterConfig(), s2["toxicity_filter"])
            if "pii_filter" in s2:
                cfg.stage2.pii_filter = _populate(PIIFilterConfig(), s2["pii_filter"])
            if "repetition_filter" in s2:
                cfg.stage2.repetition_filter = _populate(RepetitionFilterConfig(), s2["repetition_filter"])
            if "slop_filter" in s2:
                cfg.stage2.slop_filter = _populate(SlopFilterConfig(), s2["slop_filter"])
            if "benchmark_decontam" in s2:
                cfg.stage2.benchmark_decontam = _populate(BenchmarkDecontamConfig(), s2["benchmark_decontam"])
        if "stage3" in d:
            s3 = d["stage3"]
            cfg.stage3 = _populate(Stage3Config(), s3)
            if "system_prompts" in s3:
                cfg.stage3.system_prompts = _populate(SystemPromptsConfig(), s3["system_prompts"])
        if "stage4" in d:
            cfg.stage4 = _populate(Stage4Config(), d["stage4"])
        if "stage5" in d:
            cfg.stage5 = _populate(Stage5Config(), d["stage5"])
        if "stage6" in d:
            s6 = d["stage6"]
            cfg.stage6 = _populate(Stage6Config(), s6)
            if "gate" in s6:
                cfg.stage6.gate = _populate(GateConfig(), s6["gate"])
            if "report" in s6:
                cfg.stage6.report = _populate(ReportConfig(), s6["report"])
        if "s3" in d:
            cfg.s3 = _populate(S3Config(), d["s3"])
        return cfg

    def work_path(self, filename: str) -> Path:
        """Resolve a filename relative to work_dir."""
        return Path(self.globals.work_dir) / filename

    def output_path(self, filename: str) -> Path:
        """Resolve a filename relative to output_dir."""
        return Path(self.globals.output_dir) / filename

    def config_path(self, relative: str) -> Path:
        """Resolve a path from the pipeline root (where pipeline.yaml lives)."""
        return Path(relative)


def _populate(obj: Any, data: dict) -> Any:
    """Shallow-populate a dataclass from a dict, ignoring unknown keys."""
    for key, val in data.items():
        if hasattr(obj, key) and not isinstance(val, dict):
            setattr(obj, key, val)
    return obj
