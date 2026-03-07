import os
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class DataConfig:
    """Configuration for data loading."""

    tokenizer_dir: str
    max_length: int = 128
    dataset_name: str | None = None
    dataset_config: str | None = None
    tokenized_dataset_path: str | None = None
    dataset_cache_dir: str | None = None
    local_nvme_cache_dir: str | None = None
    require_local_nvme: bool = False
    pack_into_blocks: bool = False
    block_sizes: list[int] | None = None
    block_size_counts: dict[Any, Any] | None = None
    domain_column: str | None = None
    concat_across_domains: bool = False
    drop_remainder: bool = True
    num_workers: int = 12
    tokenize_num_proc: int | None = None


@dataclass
class ModelConfig:
    # Which model variant to instantiate.
    variant: Literal["1b_reversible"]

    # Embedding type passed to the model constructor.
    embedding_type: Literal["kronecker", "standard"]

    # Optional dict of ModelConfig field overrides applied after construction.
    # Only fields that exist on the model's own ModelConfig class are valid.
    overrides: dict[str, object] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    max_epochs: int

    # Validate (run eval loop) every N optimizer steps.
    eval_interval: int

    # Maximum number of optimizer steps to run per epoch.
    # None = no limit (run the full dataloader each epoch).
    # Useful for quick iteration / smoke tests.
    max_steps_per_epoch: int | None = None

    # Maximum number of steps to run during each validation pass.
    # None = run the full eval dataloader.
    max_val_steps: int | None = None

    # Log training metrics every N optimizer steps.
    log_interval: int = 10

    # Generate sample text every N optimizer steps. 0 = disabled.
    generation_interval: int = 0

    # Path to the DeepSpeed YAML config file.
    deepspeed_config: str = "deepspeed_config.yaml"

    # Override backend-specific communication overlap when supported.
    # None = keep the backend config file's current setting.
    overlap_communication: bool | None = None

    # Override DeepSpeed ZeRO reduce bucket size at runtime.
    # None = keep the backend config file's current setting.
    reduce_bucket_size: int | None = None

    # Seed applied to Python random, numpy, and torch at startup.
    seed: int = 42

    # Hard-fail if Triton fused kernels are not available.
    require_fused_kernels: bool = False

    # Memory budget (GB) for the FusedLinearCE kernel's chunked lm_head matrix
    # multiply. The kernel never materialises the full [B*T, vocab] logit
    # tensor — instead it processes the sequence in chunks sized to fit within
    # this budget. Lower values reduce peak activation memory at the cost of
    # more kernel launches. 16 GB is a safe default for most configurations.
    fused_ce_chunk_gb: float = 16.0

    # Global steps at which to run the step-level CUDA profiler.
    # Empty list = profiling disabled (zero overhead).
    profile_steps: list[int] = field(default_factory=list)

    # Run the DeepSpeed FlopsProfiler at this specific global step.
    # None = disabled. Expensive — only use for one-off profiling runs.
    flops_profiler_step: int | None = None

    check_for_gsa_leak: bool = False


@dataclass
class S3CheckpointConfig:
    """Settings for the S3 checkpoint backend."""

    bucket: str
    prefix: str
    region: str

    # Number of local staging checkpoints to keep before pruning older ones.
    keep_last_n: int = 3

    # Delete the local staging copy after a successful S3 upload.
    cleanup_after_upload: bool = False

    # Maximum concurrent upload threads per node.
    max_concurrency: int = 10

    # Retry attempts per file before giving up.
    max_retries: int = 3


@dataclass
class CheckpointConfig:
    save_interval: int | None = None
    backend: Literal["s3", "none"] = "none"
    resume_from: str | None = None
    resume_step: int = 0
    s3: S3CheckpointConfig | None = None


@dataclass
class P12LoggerConfig:
    """Settings passed to TrainingOps when observability backend = "p12"."""

    log_dir: str = "/tmp/training_logs"
    metrics_port: int = 8000
    system_metrics_interval: float = 5.0

    # Vector systemd service name checked at startup.
    # Set to None to skip the systemd-based check.
    vector_service_name: str | None = "p12-vector.service"

    # Skip the Vector preflight check entirely (useful for local dev).
    skip_vector_check: bool = False

    # Probe ClickHouse directly at startup (warn-only if unreachable).
    check_clickhouse_preflight: bool = False

    # ClickHouse connection details.
    # All fields fall back to environment variables if not set here:
    #   CLICKHOUSE_ENDPOINT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD, CLICKHOUSE_CA_CERT
    clickhouse_url: str | None = None
    clickhouse_user: str | None = None
    clickhouse_password: str | None = None
    clickhouse_ca_cert: str | None = None


@dataclass
class ObservabilityConfig:
    """
    Top-level observability configuration.

    backend = "none"  → no-op (all log/metric/event calls are discarded)
    backend = "p12"   → full P12 stack: JSONLogger → Vector → ClickHouse
                        + MetricsServer + SystemMetricsCollector
    """

    backend: Literal["none", "p12"] = "none"

    p12: P12LoggerConfig | None = None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Configuration for periodic sample generation during training."""

    # Prompt fed to the model at every generation step.
    prompt: str = "The history of artificial intelligence begins with"

    # Maximum number of new tokens to generate.
    max_new_tokens: int = 200

    # Sampling temperature. Lower = more conservative.
    temperature: float = 0.8

    # Top-k sampling. 0 = disabled.
    top_k: int = 50

    # Top-p (nucleus) sampling. 1.0 = disabled.
    top_p: float = 0.92

    # Use greedy decoding instead of sampling (overrides temperature/top_k/top_p).
    greedy: bool = False


# ---------------------------------------------------------------------------
# Top-level run config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """
    Top-level configuration for a training run.

    Directory layout
    ----------------
    All file outputs are rooted at output_dir:

        <output_dir>/
          checkpoints/    ← DeepSpeed checkpoint directories (one per tag)

    Profiles, generated samples, and metrics are published through the
    observability backend (TrainingOps in production, no-op otherwise) and
    appear on the dashboard — they are not written to disk by the trainer.

    run_id
    ------
    Unique identifier for this run. Used in ClickHouse rows, S3 paths, and
    log file names. If not provided, a timestamped ID is auto-generated and
    a warning is printed. Set it explicitly for any run you care about.
    """

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    observability: ObservabilityConfig
    generation: GenerationConfig
    run_id: str | None = None

    # Root directory for all file outputs from this run.
    output_dir: str = "./output"

    @property
    def checkpoints_dir(self) -> str:
        """Absolute path to the checkpoints subdirectory under output_dir."""
        return os.path.join(self.output_dir, "checkpoints")

    def __post_init__(self) -> None:
        if not self.run_id:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:6]
            self.run_id = f"run_{ts}_{uid}"
            warnings.warn(
                f"run_id not set — auto-generated: '{self.run_id}'. "
                "Set run_id explicitly in your config for reproducible naming.",
                stacklevel=2,
            )

    def resolve_paths(self, base_dir: str | None = None) -> None:
        """
        Resolve all relative paths in the config against base_dir.

        Called by the config loader after parsing, with base_dir set to the
        directory containing the config YAML file. This means all relative
        paths in the YAML are interpreted relative to the YAML file itself,
        regardless of where the training script is invoked from.
        """
        if base_dir is None:
            base_dir = os.getcwd()

        def _abs(p: str) -> str:
            if os.path.isabs(p):
                return p
            return os.path.abspath(os.path.join(base_dir, p))

        # Top-level
        self.output_dir = _abs(self.output_dir)

        # Data
        self.data.tokenizer_dir = _abs(self.data.tokenizer_dir)
        if self.data.tokenized_dataset_path:
            self.data.tokenized_dataset_path = _abs(self.data.tokenized_dataset_path)
        if self.data.dataset_cache_dir:
            self.data.dataset_cache_dir = _abs(self.data.dataset_cache_dir)
        if self.data.local_nvme_cache_dir:
            self.data.local_nvme_cache_dir = _abs(self.data.local_nvme_cache_dir)

        # Training
        self.training.deepspeed_config = _abs(self.training.deepspeed_config)

        # Observability
        if self.observability.p12:
            self.observability.p12.log_dir = _abs(self.observability.p12.log_dir)
            if self.observability.p12.clickhouse_ca_cert:
                self.observability.p12.clickhouse_ca_cert = _abs(
                    self.observability.p12.clickhouse_ca_cert
                )
