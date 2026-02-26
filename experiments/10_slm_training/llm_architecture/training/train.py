"""
Training Script for 1B LLM
===========================

Complete training loop with:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Metrics tracking (loss, tokens/sec)
- Checkpointing
- Experiment logging

Supports two configuration modes:
1. Preset mode: --preset 1b-base (uses Python preset configs)
2. YAML mode: --config configs/1b_base.yaml (uses YAML config files)

CLI arguments always override config file values.
"""

import argparse
import inspect
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import PRESET_CONFIGS, ModelConfig, get_preset_config
from models.llm import create_model_from_config


def load_config_from_yaml(config_path: str) -> Tuple[ModelConfig, Dict[str, Any]]:
    """
    Load model and training config from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (ModelConfig, training_config_dict)
    """
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Extract training config (if present)
    training_data = config_data.pop("training", {})

    # Load model config
    model_config = ModelConfig.from_dict(config_data)

    return model_config, training_data


def training_config_from_dict(data: Dict[str, Any]) -> "TrainingConfig":
    """
    Create TrainingConfig from dictionary, ignoring unknown keys.

    Args:
        data: Dictionary with training configuration values

    Returns:
        TrainingConfig instance
    """
    # Get valid field names from TrainingConfig
    valid_fields = {f.name for f in fields(TrainingConfig)}

    # Filter to only valid fields
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}

    return TrainingConfig(**filtered_data)


def align_model_context_to_seq_length(
    model_config: ModelConfig, seq_length: int
) -> None:
    """
    Ensure model positional capacity matches requested training sequence length.

    This prevents out-of-bounds indexing in RoPE/YaRN caches when training with
    seq_length > model_config.max_position_embeddings.
    """
    if seq_length <= model_config.max_position_embeddings:
        return

    old_max = model_config.max_position_embeddings
    model_config.max_position_embeddings = seq_length
    print(
        f"[Context] Increasing max_position_embeddings: {old_max} -> {seq_length} "
        f"to match training seq_length."
    )

    # If YaRN is used, ensure configured scale can cover seq_length.
    pos = getattr(model_config, "position", None)
    if pos is None:
        return

    pos_type = getattr(pos, "position_type", None)
    pos_type_value = pos_type.value if hasattr(pos_type, "value") else str(pos_type)
    if pos_type_value != "yarn":
        return

    original_max = int(getattr(pos, "yarn_original_max_position", old_max) or old_max)
    original_max = max(1, original_max)
    required_scale = seq_length / float(original_max)
    current_scale = float(getattr(pos, "yarn_scale", 1.0))
    if required_scale > current_scale:
        pos.yarn_scale = required_scale
        print(
            f"[Context] Increasing YaRN scale: {current_scale:.4g} -> {required_scale:.4g} "
            f"(original_max={original_max}, target_seq={seq_length})."
        )


def get_best_device(preferred: str = "auto") -> torch.device:
    """
    Get the best available device.

    Args:
        preferred: "auto", "cuda", "mps", or "cpu"

    Returns:
        torch.device for the selected device
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA not available, falling back to CPU")
            return torch.device("cpu")
    elif preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("Warning: MPS not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Training duration
    max_steps: int = 10000
    max_epochs: Optional[int] = None

    # Batch settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    seq_length: int = 1024

    # Optimizer
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # LR Schedule
    warmup_steps: int = 500
    lr_decay_style: str = "cosine"  # cosine, linear, constant

    # Regularization
    gradient_clip: float = 1.0
    dropout: float = 0.0

    # Precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16, float16

    # Device selection
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Checkpointing
    save_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"

    # Logging
    log_interval: int = 10
    eval_interval: int = 500

    # torch.compile (PyTorch 2.0+)
    use_torch_compile: bool = False
    torch_compile_mode: str = (
        "max-autotune-no-cudagraphs"  # default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs
    )
    torch_compile_fullgraph: bool = (
        False  # enforce no graph breaks (stricter but faster)
    )
    torch_compile_dynamic: bool = (
        False  # allow dynamic shapes (slower compile, flexible shapes)
    )
    torch_compile_backend: str = "inductor"  # inductor (default), cudagraphs, etc.

    # Experiment
    experiment_name: str = "1b_base"
    seed: int = 42


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    grad_norm: float = 0.0
    tokens_seen: int = 0
    elapsed_time: float = 0.0

    # Loss components (for MTP)
    main_loss: Optional[float] = None
    aux_loss: Optional[float] = None


class MetricsLogger:
    """Logs training metrics to file and console."""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"
        self.metrics_history: List[Dict] = []

    def log(self, metrics: TrainingMetrics):
        """Log metrics."""
        metrics_dict = asdict(metrics)
        self.metrics_history.append(metrics_dict)

        # Write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics_dict) + "\n")

    def save_summary(self, config: Dict, final_metrics: TrainingMetrics):
        """Save training summary."""
        summary = {
            "config": config,
            "final_metrics": asdict(final_metrics),
            "history_length": len(self.metrics_history),
        }

        summary_file = self.log_file.with_suffix(".summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)


class LRScheduler:
    """Learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float,
        style: str = "cosine",
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.style = style
        self.current_step = 0

    def step(self) -> float:
        """Update learning rate and return current value."""
        self.current_step += 1
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def get_lr(self) -> float:
        """Calculate current learning rate."""
        step = self.current_step

        # Warmup phase
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps

        # Decay phase
        if self.style == "constant":
            return self.max_lr

        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        progress = min(1.0, progress)

        if self.style == "linear":
            return self.min_lr + (self.max_lr - self.min_lr) * (1 - progress)
        elif self.style == "cosine":
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        else:
            return self.max_lr


class TrainingController:
    """
    Controls pause/resume/stop of training with ZERO background thread overhead.

    All input checking happens inline at natural pause points (once per optimizer
    step), so there is no background thread competing for the GIL.

    Controls:
      - Ctrl+C : Pause training (first time) / Stop & save (while paused or second Ctrl+C)
      - While paused, reads stdin:  Enter = resume,  'stop' = save & exit

    The SIGINT handler is managed by the Trainer, not this class.
    """

    def __init__(self):
        self._paused = False
        self._stop_requested = False
        self._pause_time = 0.0
        self._pause_start = None
        self._interactive = True

    def start(self):
        """Print controls info. No background thread is started."""
        # Check if stdin is interactive (not redirected)
        try:
            self._interactive = os.isatty(sys.stdin.fileno())
        except (AttributeError, OSError):
            self._interactive = False

        print(
            "[Controls] Ctrl+C to pause | While paused: Enter=resume, 'stop'=save & exit\n"
        )

    def pause(self):
        """Pause training. Called from SIGINT handler."""
        if not self._paused:
            self._paused = True
            self._pause_start = time.time()
            print(f"\n{'*'*60}")
            print("  TRAINING PAUSED")
            print("  Press Enter to resume | Type 'stop' to save & exit")
            print(f"{'*'*60}")

    def request_stop(self):
        """Request graceful stop. Called from SIGINT handler or user input."""
        self._stop_requested = True
        if self._paused:
            # Account for pause time before unblocking
            if self._pause_start is not None:
                self._pause_time += time.time() - self._pause_start
                self._pause_start = None
            self._paused = False
        print(f"\n{'*'*60}")
        print("  STOP REQUESTED  -  Saving checkpoint and exiting...")
        print(f"{'*'*60}\n")

    def check_and_handle_pause(self):
        """Called once per optimizer step. Blocks while paused, reads user input.

        Returns immediately with zero overhead when not paused.
        """
        if not self._paused:
            return

        # Paused: block and read stdin for resume/stop commands
        while self._paused and not self._stop_requested:
            try:
                line = input("> ").strip().lower()
            except EOFError:
                # Non-interactive, just resume
                self._resume()
                return

            if line == "stop":
                self.request_stop()
                return
            else:
                # Any other input (including empty Enter) resumes
                self._resume()
                return

    def _resume(self):
        """Resume training from paused state."""
        if self._paused:
            if self._pause_start is not None:
                self._pause_time += time.time() - self._pause_start
                self._pause_start = None
            self._paused = False
            print(f"\n{'*'*60}")
            print("  TRAINING RESUMED")
            print(f"{'*'*60}\n")

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def total_pause_time(self) -> float:
        """Total seconds spent paused."""
        extra = 0.0
        if self._paused and self._pause_start is not None:
            extra = time.time() - self._pause_start
        return self._pause_time + extra

    def shutdown(self):
        """Cleanup (no-op since there's no background thread)."""
        pass


class RandomTextDataset(Dataset):
    """
    Random dataset for testing/development.

    In production, replace with real tokenized dataset.
    """

    def __init__(
        self, vocab_size: int, seq_length: int, num_samples: int, seed: int = 42
    ):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

        # Pre-generate for reproducibility
        torch.manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {
            "input_ids": tokens,
            # Model forward already shifts for next-token prediction.
            "labels": tokens.clone(),
        }


class Trainer:
    """
    Main trainer class.

    Handles:
    - Training loop
    - Gradient accumulation
    - Mixed precision
    - Checkpointing
    - Metrics logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = training_config
        self.model_config = model_config

        # Device - supports CUDA, MPS (Apple Silicon), and CPU
        self.device = get_best_device(training_config.device)
        self.model = self.model.to(self.device)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception as e:
                print(f"Warning: gradient checkpointing enable failed: {e}")
        else:
            print(
                "Info: Model does not expose gradient_checkpointing_enable(); continuing."
            )

        # torch.compile - apply AFTER device placement & gradient checkpointing,
        # BEFORE optimizer creation (named_parameters works through compile wrapper)
        self._apply_torch_compile(training_config)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # LR Scheduler
        self.lr_scheduler = LRScheduler(
            optimizer=self.optimizer,
            warmup_steps=training_config.warmup_steps,
            max_steps=training_config.max_steps,
            max_lr=training_config.learning_rate,
            min_lr=training_config.min_learning_rate,
            style=training_config.lr_decay_style,
        )

        # Mixed precision - CUDA supports both float16 and bfloat16, MPS supports float16 only
        self.use_amp = training_config.use_amp and (
            self.device.type == "cuda"
            or (self.device.type == "mps" and training_config.amp_dtype == "float16")
        )
        self.amp_dtype = getattr(torch, training_config.amp_dtype)

        # GradScaler only works with CUDA float16
        scaler_enabled = (
            self.use_amp
            and self.device.type == "cuda"
            and training_config.amp_dtype == "float16"
        )
        self.scaler = GradScaler("cuda", enabled=scaler_enabled)

        # Logging
        self.logger = MetricsLogger(
            log_dir=training_config.checkpoint_dir,
            experiment_name=training_config.experiment_name,
        )

        # Pause controller
        self.pause_controller = TrainingController()

        # State
        self.global_step = 0
        self.epoch = 0
        self.tokens_seen = 0
        self.best_loss = float("inf")
        self.start_time = None
        self._compile_safe_retry_attempted = False
        self._compile_eager_fallback_done = False

    def _apply_torch_compile(self, training_config: TrainingConfig):
        """Apply torch.compile to the model if enabled and available."""
        if not training_config.use_torch_compile:
            return

        if not hasattr(torch, "compile"):
            print(
                "Warning: torch.compile not available (requires PyTorch 2.0+), skipping"
            )
            return

        # torch.compile on MPS has limited support
        if self.device.type == "mps":
            print(
                "Warning: torch.compile has limited MPS support. "
                "If you hit errors, disable with --no-torch-compile or use CUDA."
            )

        mode = training_config.torch_compile_mode
        fullgraph = training_config.torch_compile_fullgraph
        dynamic = (
            training_config.torch_compile_dynamic
            if training_config.torch_compile_dynamic
            else None
        )
        backend = training_config.torch_compile_backend

        # Inductor max-autotune can fail on very long contexts due Triton index
        # integer range limits during autotuning. Prefer "default" mode there.
        if (
            backend == "inductor"
            and self.device.type == "cuda"
            and training_config.seq_length >= 32768
            and mode in {"max-autotune", "max-autotune-no-cudagraphs"}
        ):
            print(
                f"Warning: seq_length={training_config.seq_length} with mode='{mode}' can crash "
                "Inductor/Triton autotuning on large BMM shapes."
            )
            print("  Auto-switching to torch.compile mode='default' for stability.")
            mode = "default"
            training_config.torch_compile_mode = mode

        # CUDAGraphs (used by reduce-overhead AND max-autotune) allocate static
        # output buffers. MTP heads produce multiple large outputs (main_logits +
        # aux_logits) from the same hidden state, causing CUDAGraphs to overwrite
        # earlier buffers. Auto-downgrade to max-autotune-no-cudagraphs.
        cudagraph_modes = {"reduce-overhead", "max-autotune"}
        raw_model = getattr(self.model, "_orig_mod", self.model)
        has_mtp_outputs = (
            getattr(raw_model, "mtp_loss", None) is not None
            or getattr(raw_model, "mtp_block", None) is not None
        )
        if mode in cudagraph_modes and has_mtp_outputs:
            print(
                f"Warning: '{mode}' mode uses CUDAGraphs which is incompatible with "
                "Multi-Token Prediction (buffer reuse overwrites MTP head outputs)."
            )
            print("  Auto-switching to max-autotune-no-cudagraphs mode.")
            mode = "max-autotune-no-cudagraphs"
            training_config.torch_compile_mode = mode

        # Enable TF32 for float32 matmul — better Tensor Core utilization on Ampere+
        # Use only torch.set_float32_matmul_precision (the public API) to avoid
        # mixing legacy and new internal APIs, which crashes Inductor's max-autotune.
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        print(
            f"Applying torch.compile(mode='{mode}', backend='{backend}', "
            f"fullgraph={fullgraph}, dynamic={dynamic})..."
        )

        try:
            self.model = torch.compile(
                self.model,
                mode=mode,
                backend=backend,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            print(
                "Model compiled successfully (first forward pass will be slower due to compilation)"
            )
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
            print("Continuing without compilation.")

    def _is_compile_runtime_error(self, error: Exception) -> bool:
        """Heuristic check for torch.compile/Inductor/Triton runtime failures."""
        if not self.config.use_torch_compile:
            return False
        message = f"{type(error).__name__}: {error}"
        markers = (
            "torch._inductor",
            "torch._dynamo",
            "InductorError",
            "BackendCompilerFailed",
            "triton.compiler.errors.CompilationError",
            "out of range for type int32",
            "torch.compile",
        )
        return any(marker in message for marker in markers)

    def _retry_compile_with_safe_mode(self) -> bool:
        """Try once to recover by recompiling with conservative settings."""
        if not hasattr(torch, "compile"):
            return False

        raw_model = getattr(self.model, "_orig_mod", self.model)
        backend = self.config.torch_compile_backend
        safe_mode = "default"
        print(
            f"Retrying torch.compile with safer settings: mode='{safe_mode}', backend='{backend}', "
            "fullgraph=False, dynamic=None"
        )
        try:
            self.model = torch.compile(
                raw_model,
                mode=safe_mode,
                backend=backend,
                fullgraph=False,
                dynamic=None,
            )
            self.config.torch_compile_mode = safe_mode
            self.config.torch_compile_fullgraph = False
            self.config.torch_compile_dynamic = False
            print("Safe torch.compile retry succeeded.")
            return True
        except Exception as retry_error:
            print(f"Warning: safe torch.compile retry failed: {retry_error}")
            return False

    def _fallback_to_eager(self) -> bool:
        """Disable compile and restore eager model execution."""
        raw_model = getattr(self.model, "_orig_mod", None)
        if raw_model is None:
            return False
        self.model = raw_model
        self.config.use_torch_compile = False
        print("Falling back to eager execution for stability.")
        return True

    def _handle_compile_runtime_failure(self, error: Exception) -> bool:
        """
        Handle runtime compile failures.

        Returns True if recovery action was applied and caller should retry.
        """
        if not self._is_compile_runtime_error(error):
            return False

        if not self._compile_safe_retry_attempted:
            self._compile_safe_retry_attempted = True
            if self._retry_compile_with_safe_mode():
                return True

        if not self._compile_eager_fallback_done:
            self._compile_eager_fallback_done = True
            if self._fallback_to_eager():
                return True

        return False

    def _compute_loss(self, outputs, labels):
        """
        Compute loss outside the compiled graph.

        Used with torch.compile to avoid CUDAGraph buffer reuse errors.
        The model forward returns logits only (labels not passed), and
        loss is computed here in eager mode.

        Returns:
            (loss, loss_dict) tuple
        """
        raw_model = getattr(self.model, "_orig_mod", self.model)
        mtp_loss_fn = getattr(raw_model, "mtp_loss", None)
        logits, aux_logits, aux_residual = self._extract_output_tensors(outputs)

        if logits is None:
            raise ValueError(
                "Model outputs do not contain logits/logits_ntp for loss computation."
            )

        if mtp_loss_fn is not None and aux_logits is not None:
            loss, loss_dict = mtp_loss_fn(logits, aux_logits, labels)
            return loss, loss_dict

        # Main NTP loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        main_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss = main_loss
        loss_dict = {"main_loss": main_loss}

        # Reference MTP output (logits_mtp) path
        if aux_logits is not None and labels.size(1) > 2:
            mtp_shift_logits = aux_logits[..., :-1, :].contiguous()
            mtp_shift_labels = labels[..., 2:].contiguous()
            mtp_min_len = min(mtp_shift_logits.size(1), mtp_shift_labels.size(1))
            if mtp_min_len > 0:
                mtp_loss = F.cross_entropy(
                    mtp_shift_logits[:, :mtp_min_len].reshape(
                        -1, mtp_shift_logits.size(-1)
                    ),
                    mtp_shift_labels[:, :mtp_min_len].reshape(-1),
                    ignore_index=-100,
                )
                mtp_weight = getattr(getattr(raw_model, "config", None), "head", None)
                mtp_weight = getattr(mtp_weight, "mtp_loss_weight", 1.0)
                loss = loss + mtp_weight * mtp_loss
                loss_dict["mtp_loss"] = mtp_loss

        if aux_residual is not None:
            loss = loss + aux_residual
            loss_dict["aux_loss"] = aux_residual

        loss_dict["total_loss"] = loss

        return loss, loss_dict

    @staticmethod
    def _extract_output_tensors(outputs):
        """Extract logits/aux tensors from both LLMOutput and ReferenceLLMOutput/tuple."""
        logits = None
        aux_logits = None
        aux_residual = None

        if isinstance(outputs, tuple):
            if len(outputs) > 0:
                logits = outputs[0]
            if len(outputs) > 1:
                aux_logits = outputs[1]
            return logits, aux_logits, aux_residual

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif hasattr(outputs, "logits_ntp"):
            logits = outputs.logits_ntp

        if hasattr(outputs, "aux_logits"):
            aux_logits = outputs.aux_logits
        elif hasattr(outputs, "logits_mtp"):
            aux_logits = outputs.logits_mtp

        if hasattr(outputs, "aux_loss"):
            aux_residual = outputs.aux_loss

        return logits, aux_logits, aux_residual

    def _build_forward_inputs(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        include_labels: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Build forward kwargs compatible with both LLM and ReferenceLLM.
        """
        forward_inputs: Dict[str, torch.Tensor] = {"input_ids": input_ids}

        raw_model = getattr(self.model, "_orig_mod", self.model)
        try:
            forward_params = inspect.signature(raw_model.forward).parameters
        except (TypeError, ValueError):
            forward_params = {}

        if "next_token_ids" in forward_params and input_ids.size(1) > 1:
            forward_inputs["next_token_ids"] = input_ids[:, 1:].contiguous()

        if include_labels and labels is not None:
            forward_inputs["labels"] = labels

        return forward_inputs

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

    def train(self) -> TrainingMetrics:
        """Run training loop."""
        self.model.train()
        self.start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Starting Training: {self.config.experiment_name}")
        print(f"{'='*60}")
        print(f"Model: {self.model_config.model_name}")
        raw_model = getattr(self.model, "_orig_mod", self.model)
        num_params = getattr(raw_model, "num_parameters", None)
        if num_params is None:
            num_params = sum(p.numel() for p in raw_model.parameters())
        print(f"Parameters: {num_params / 1e9:.2f}B")
        print(f"Device: {self.device}")
        print(f"Max steps: {self.config.max_steps}")
        print(
            f"Batch size: {self.config.batch_size} x {self.config.gradient_accumulation_steps}"
        )
        if self.config.use_torch_compile:
            print(
                f"torch.compile: mode={self.config.torch_compile_mode}, backend={self.config.torch_compile_backend}"
            )
        print(f"{'='*60}\n")

        # Start pause controller
        self.pause_controller.start()

        # Register Ctrl+C handler:
        #   1st Ctrl+C -> pause (if running)
        #   2nd Ctrl+C -> stop & save (if paused)
        #   3rd Ctrl+C -> force exit
        original_sigint = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum, frame):
            ctrl = self.pause_controller
            if ctrl.stop_requested:
                # Already stopping, force exit
                print("\nForce exit requested.")
                signal.signal(signal.SIGINT, original_sigint)
                raise KeyboardInterrupt
            elif ctrl.is_paused:
                # Paused -> stop & save
                ctrl.request_stop()
            else:
                # Running -> pause
                ctrl.pause()

        signal.signal(signal.SIGINT, _sigint_handler)

        accumulation_loss = 0.0
        accumulation_main_loss = 0.0
        accumulation_aux_loss = 0.0
        accumulation_steps = 0
        step_start_time = time.time()
        step_pause_snapshot = self.pause_controller.total_pause_time
        stopped_early = False

        data_iter = iter(self.train_dataloader)

        while self.global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            def _run_forward():
                # Mark step boundary for CUDAGraphs (only needed in CUDAGraph modes).
                _cudagraph_modes = {"reduce-overhead", "max-autotune"}
                _uses_cudagraphs = (
                    self.config.use_torch_compile
                    and self.config.torch_compile_mode in _cudagraph_modes
                )
                if _uses_cudagraphs and hasattr(
                    torch.compiler, "cudagraph_mark_step_begin"
                ):
                    torch.compiler.cudagraph_mark_step_begin()

                # Forward pass with device-aware autocast
                with autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):
                    if _uses_cudagraphs:
                        # CUDAGraph modes — compute loss outside compiled graph
                        # to avoid buffer reuse errors.
                        forward_inputs = self._build_forward_inputs(
                            input_ids=input_ids,
                            include_labels=False,
                        )
                        outputs = self.model(**forward_inputs)
                        micro_loss_local, loss_dict_local = self._compute_loss(
                            outputs, labels
                        )
                    else:
                        # default / max-autotune-no-cudagraphs / non-compiled: standard forward
                        forward_inputs = self._build_forward_inputs(
                            input_ids=input_ids,
                            labels=labels,
                            include_labels=True,
                        )
                        outputs = self.model(**forward_inputs)
                        micro_loss_local = outputs.loss
                        loss_dict_local = outputs.loss_dict
                    loss_local = (
                        micro_loss_local / self.config.gradient_accumulation_steps
                    )

                return loss_local, micro_loss_local, loss_dict_local

            try:
                loss, micro_loss, loss_dict = _run_forward()
            except Exception as e:
                if self._handle_compile_runtime_failure(e):
                    loss, micro_loss, loss_dict = _run_forward()
                else:
                    raise

            # Backward pass (GradScaler only for CUDA float16)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_loss += micro_loss.item()
            if loss_dict is not None:
                accumulation_main_loss += loss_dict.get("main_loss", micro_loss).item()
                aux_key = None
                if "aux_total" in loss_dict:
                    aux_key = "aux_total"
                elif "aux_loss" in loss_dict:
                    aux_key = "aux_loss"
                if aux_key is not None:
                    accumulation_aux_loss += loss_dict[aux_key].item()
            accumulation_steps += 1

            # Gradient accumulation step
            if accumulation_steps >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                ).item()

                # Optimizer step
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # LR update
                current_lr = self.lr_scheduler.step()

                # Update counters
                self.global_step += 1
                tokens_in_step = (
                    self.config.batch_size
                    * self.config.seq_length
                    * self.config.gradient_accumulation_steps
                )
                self.tokens_seen += tokens_in_step

                # Calculate metrics (subtract pause time within this step)
                step_pause_delta = (
                    self.pause_controller.total_pause_time - step_pause_snapshot
                )
                step_time = time.time() - step_start_time - step_pause_delta
                step_time = max(step_time, 1e-6)  # Avoid division by zero
                tokens_per_second = tokens_in_step / step_time
                samples_per_second = (
                    self.config.batch_size * self.config.gradient_accumulation_steps
                ) / step_time

                # Log metrics (exclude pause time from elapsed)
                active_elapsed = (
                    time.time()
                    - self.start_time
                    - self.pause_controller.total_pause_time
                )
                metrics = TrainingMetrics(
                    step=self.global_step,
                    epoch=self.epoch,
                    loss=accumulation_loss / max(1, accumulation_steps),
                    learning_rate=current_lr,
                    tokens_per_second=tokens_per_second,
                    samples_per_second=samples_per_second,
                    grad_norm=grad_norm,
                    tokens_seen=self.tokens_seen,
                    elapsed_time=active_elapsed,
                )

                # Add averaged MTP loss components if available
                if loss_dict is not None:
                    metrics.main_loss = accumulation_main_loss / max(
                        1, accumulation_steps
                    )
                    if "aux_total" in loss_dict or "aux_loss" in loss_dict:
                        metrics.aux_loss = accumulation_aux_loss / max(
                            1, accumulation_steps
                        )

                self.logger.log(metrics)

                # Console logging
                if self.global_step % self.config.log_interval == 0:
                    self._print_progress(metrics)

                # Checkpointing
                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(metrics)

                # Update best loss
                if metrics.loss < self.best_loss:
                    self.best_loss = metrics.loss

                # Reset accumulation
                accumulation_loss = 0.0
                accumulation_main_loss = 0.0
                accumulation_aux_loss = 0.0
                accumulation_steps = 0

                # Pause/stop check (once per optimizer step, zero overhead when running)
                self.pause_controller.check_and_handle_pause()
                if self.pause_controller.stop_requested:
                    stopped_early = True
                    break

                step_start_time = time.time()
                step_pause_snapshot = self.pause_controller.total_pause_time

        # Restore original signal handler and stop pause controller
        signal.signal(signal.SIGINT, original_sigint)
        self.pause_controller.shutdown()

        # Final checkpoint
        total_pause_time = self.pause_controller.total_pause_time
        active_time = time.time() - self.start_time - total_pause_time
        final_metrics = TrainingMetrics(
            step=self.global_step,
            epoch=self.epoch,
            loss=self.best_loss,
            tokens_seen=self.tokens_seen,
            elapsed_time=active_time,
        )

        self._save_checkpoint(final_metrics, is_final=True)
        self.logger.save_summary(
            config=asdict(self.config), final_metrics=final_metrics
        )

        wall_time = time.time() - self.start_time
        status = (
            "Training Stopped (checkpoint saved)"
            if stopped_early
            else "Training Complete!"
        )
        print(f"\n{'='*60}")
        print(status)
        print(f"{'='*60}")
        print(f"Final step: {self.global_step}/{self.config.max_steps}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Tokens seen: {self.tokens_seen:,}")
        print(f"Active training time: {active_time:.1f}s")
        if total_pause_time > 0:
            print(f"Total pause time: {total_pause_time:.1f}s")
        print(f"Wall clock time: {wall_time:.1f}s")
        print(f"{'='*60}\n")

        return final_metrics

    def _print_progress(self, metrics: TrainingMetrics):
        """Print training progress."""
        eta_seconds = (self.config.max_steps - metrics.step) * (
            metrics.elapsed_time / max(1, metrics.step)
        )
        eta_str = (
            f"{eta_seconds/3600:.1f}h"
            if eta_seconds > 3600
            else f"{eta_seconds/60:.1f}m"
        )

        # Build loss string: show model loss and MTP loss separately when available
        if metrics.main_loss is not None and metrics.aux_loss is not None:
            loss_str = (
                f"Model Loss: {metrics.main_loss:.4f} | "
                f"MTP Loss: {metrics.aux_loss:.4f} | "
                f"Total Loss: {metrics.loss:.4f}"
            )
        elif metrics.main_loss is not None:
            loss_str = (
                f"Model Loss: {metrics.main_loss:.4f} | "
                f"Total Loss: {metrics.loss:.4f}"
            )
        else:
            loss_str = f"Loss: {metrics.loss:.4f}"

        print(
            f"Step {metrics.step:>6d}/{self.config.max_steps} | "
            f"{loss_str} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Tok/s: {metrics.tokens_per_second:,.0f} | "
            f"Grad: {metrics.grad_norm:.2f} | "
            f"ETA: {eta_str}"
        )

    def _save_checkpoint(self, metrics: TrainingMetrics, is_final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if is_final:
            checkpoint_path = checkpoint_dir / f"{self.config.experiment_name}_final.pt"
        else:
            checkpoint_path = (
                checkpoint_dir / f"{self.config.experiment_name}_step{metrics.step}.pt"
            )

        # Access underlying model if torch.compiled (state_dict works either way,
        # but saving the unwrapped state_dict is cleaner for portability)
        raw_model = getattr(self.model, "_orig_mod", self.model)

        checkpoint = {
            "step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_step": self.lr_scheduler.current_step,
            "metrics": asdict(metrics),
            "model_config": self.model_config.to_dict(),
            "training_config": asdict(self.config),
            "best_loss": self.best_loss,
            "tokens_seen": self.tokens_seen,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Saved checkpoint: {checkpoint_path}")


def run_training(
    model_preset: str = "1b-base",
    training_config: Optional[TrainingConfig] = None,
    model_config_overrides: Optional[Dict] = None,
) -> Tuple[nn.Module, TrainingMetrics]:
    """
    Run training with specified configuration.

    Args:
        model_preset: Model preset name
        training_config: Training configuration
        model_config_overrides: Overrides for model config

    Returns:
        Trained model and final metrics
    """
    # Set seed
    if training_config is None:
        training_config = TrainingConfig()

    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)

    # Create model config
    model_config = get_preset_config(model_preset)
    if model_config_overrides:
        for key, value in model_config_overrides.items():
            if hasattr(model_config, key):
                # Handle nested dataclass configs (attention, position, ffn, etc.)
                if hasattr(value, "__dataclass_fields__"):
                    setattr(model_config, key, value)
                else:
                    setattr(model_config, key, value)

    # Keep model positional capacity aligned with requested training length.
    align_model_context_to_seq_length(model_config, training_config.seq_length)

    # Create model
    model = create_model_from_config(model_config)

    # Create dataset
    dataset = RandomTextDataset(
        vocab_size=model_config.vocab_size,
        seq_length=training_config.seq_length,
        num_samples=training_config.max_steps * training_config.batch_size * 2,
        seed=training_config.seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        training_config=training_config,
        model_config=model_config,
    )

    # Train
    final_metrics = trainer.train()

    return model, final_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train 1B LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using YAML config file (recommended)
  python train.py --config ../configs/1b_base.yaml
  python train.py --config ../configs/1b_deepseek_gsa.yaml --batch-size 4
  
  # Using preset (legacy mode)
  python train.py --preset 1b-base --max-steps 10000
  
  # YAML config with CLI overrides
  python train.py --config ../configs/1b_gsa.yaml --learning-rate 1e-4 --device cuda

Note: CLI arguments always override config file values.
        """,
    )

    # Configuration source (mutually exclusive conceptually, but --config takes precedence)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/1b_base.yaml). Takes precedence over --preset.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="1b-base",
        choices=list(PRESET_CONFIGS.keys()),
        help="Model preset (used if --config not provided)",
    )

    # Training (can override config file)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use: auto (best available), cuda, mps (Apple Silicon), or cpu",
    )

    # GSA-specific overrides for memory tuning
    parser.add_argument(
        "--gsa-k-base",
        type=int,
        default=None,
        help="Override GSA k_base (default: from preset, try 256-512 for long sequences)",
    )
    parser.add_argument(
        "--gsa-k-max",
        type=int,
        default=None,
        help="Override GSA k_max (default: from preset, try 512-1024 for long sequences)",
    )
    parser.add_argument(
        "--no-triton",
        action="store_true",
        help="Disable Triton kernels (use PyTorch fallback for GSA)",
    )

    # torch.compile
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Enable torch.compile() for graph optimization (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--torch-compile-mode",
        type=str,
        default=None,
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="torch.compile mode: default (safe), max-autotune-no-cudagraphs (recommended), max-autotune (+ CUDAGraphs), reduce-overhead (CUDAGraphs only)",
    )
    parser.add_argument(
        "--torch-compile-fullgraph",
        action="store_true",
        help="Enforce full-graph compilation (no graph breaks). Fails if model has unsupported ops.",
    )
    parser.add_argument(
        "--torch-compile-dynamic",
        action="store_true",
        help="Enable dynamic shapes in torch.compile (slower compile, flexible input shapes)",
    )

    # Experiment
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Logging
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)

    args = parser.parse_args()

    # Load configuration
    if args.config:
        # YAML config mode
        print(f"Loading configuration from: {args.config}")
        model_config, training_dict = load_config_from_yaml(args.config)
        training_config = (
            training_config_from_dict(training_dict)
            if training_dict
            else TrainingConfig()
        )
    else:
        # Preset mode (legacy)
        print(f"Using preset: {args.preset}")
        model_config = get_preset_config(args.preset)
        training_config = TrainingConfig()

    # CLI overrides (only if explicitly provided)
    if args.max_steps is not None:
        training_config.max_steps = args.max_steps
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    if args.gradient_accumulation is not None:
        training_config.gradient_accumulation_steps = args.gradient_accumulation
    if args.seq_length is not None:
        training_config.seq_length = args.seq_length
    if args.learning_rate is not None:
        training_config.learning_rate = args.learning_rate
    if args.warmup_steps is not None:
        training_config.warmup_steps = args.warmup_steps
    if args.device is not None:
        training_config.device = args.device
    if args.experiment_name is not None:
        training_config.experiment_name = args.experiment_name
    if args.checkpoint_dir is not None:
        training_config.checkpoint_dir = args.checkpoint_dir
    if args.seed is not None:
        training_config.seed = args.seed
    if args.log_interval is not None:
        training_config.log_interval = args.log_interval
    if args.save_interval is not None:
        training_config.save_interval = args.save_interval
    if args.use_torch_compile:
        training_config.use_torch_compile = True
    if args.torch_compile_mode is not None:
        training_config.torch_compile_mode = args.torch_compile_mode
    if args.torch_compile_fullgraph:
        training_config.torch_compile_fullgraph = True
    if args.torch_compile_dynamic:
        training_config.torch_compile_dynamic = True

    # Keep model positional capacity aligned with requested training length.
    align_model_context_to_seq_length(model_config, training_config.seq_length)

    # Set seed
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)

    # Create model
    model = create_model_from_config(model_config)

    # Create dataset
    dataset = RandomTextDataset(
        vocab_size=model_config.vocab_size,
        seq_length=training_config.seq_length,
        num_samples=training_config.max_steps * training_config.batch_size * 2,
        seed=training_config.seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        training_config=training_config,
        model_config=model_config,
    )

    # Train
    metrics = trainer.train()

    print(f"\nTraining complete! Final loss: {metrics.loss:.4f}")


if __name__ == "__main__":
    main()
