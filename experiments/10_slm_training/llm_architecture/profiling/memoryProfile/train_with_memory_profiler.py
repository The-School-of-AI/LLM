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
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Add llm_architecture directory to path
# Path: memoryProfile -> profiling -> llm_architecture
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.model_config import PRESET_CONFIGS, ModelConfig, get_preset_config
from memory_profiler import MemoryProfiler, ProfilerConfig
from models.llm import create_model_from_config


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

    # Checkpointing
    save_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"

    # Logging
    log_interval: int = 10
    eval_interval: int = 500

    # Profiling
    enable_profiling: bool = False
    profiling_output_dir: str = "./profiler_logs"
    profiling_active_steps: int = 10
    profiling_wait_steps: int = 5
    profiling_warmup_steps: int = 5
    profiling_repeat: int = 5

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
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length + 1))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


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

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

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

        # Mixed precision
        self.use_amp = training_config.use_amp and torch.cuda.is_available()
        self.amp_dtype = getattr(torch, training_config.amp_dtype)
        self.scaler = GradScaler(
            "cuda", enabled=self.use_amp and training_config.amp_dtype == "float16"
        )

        # Logging
        self.logger = MetricsLogger(
            log_dir=training_config.checkpoint_dir,
            experiment_name=training_config.experiment_name,
        )

        # Profiler
        if training_config.enable_profiling:
            profiler_config = ProfilerConfig(
                output_dir=training_config.profiling_output_dir,
                profile_memory=True,
                profile_cpu=True,
                profile_cuda=True,
                record_shapes=True,
                with_stack=True,
                wait_steps=training_config.profiling_wait_steps,
                warmup_steps=training_config.profiling_warmup_steps,
                active_steps=training_config.profiling_active_steps,
                repeat=training_config.profiling_repeat,
                sort_by="cuda_memory_usage",
                row_limit=20,
            )
            self.profiler = MemoryProfiler(profiler_config)
            print(f"\n📊 Profiling enabled: {profiler_config.output_dir}")
            print(f"   Will profile for {self.profiler.get_total_steps()} steps\n")
        else:
            self.profiler = None

        # State
        self.global_step = 0
        self.epoch = 0
        self.tokens_seen = 0
        self.best_loss = float("inf")
        self.start_time = None

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
        print(f"Parameters: {self.model.num_parameters / 1e9:.2f}B")
        print(f"Device: {self.device}")
        print(f"Max steps: {self.config.max_steps}")
        print(
            f"Batch size: {self.config.batch_size} x {self.config.gradient_accumulation_steps}"
        )
        if self.profiler:
            print(f"Profiling: Enabled ({self.profiler.get_total_steps()} steps)")
        print(f"{'='*60}\n")

        # Start profiler
        if self.profiler:
            self.profiler.start()

        accumulation_loss = 0.0
        accumulation_steps = 0
        step_start_time = time.time()

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

            # Forward pass
            with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.use_amp and self.config.amp_dtype == "float16":
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_loss += loss.item()
            accumulation_steps += 1

            # Gradient accumulation step
            if accumulation_steps >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.use_amp and self.config.amp_dtype == "float16":
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                ).item()

                # Optimizer step
                if self.use_amp and self.config.amp_dtype == "float16":
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

                # Calculate metrics
                step_time = time.time() - step_start_time
                tokens_per_second = tokens_in_step / step_time
                samples_per_second = (
                    self.config.batch_size * self.config.gradient_accumulation_steps
                ) / step_time

                # Log metrics
                metrics = TrainingMetrics(
                    step=self.global_step,
                    epoch=self.epoch,
                    loss=accumulation_loss * self.config.gradient_accumulation_steps,
                    learning_rate=current_lr,
                    tokens_per_second=tokens_per_second,
                    samples_per_second=samples_per_second,
                    grad_norm=grad_norm,
                    tokens_seen=self.tokens_seen,
                    elapsed_time=time.time() - self.start_time,
                )

                # Add MTP loss components if available
                if outputs.loss_dict is not None:
                    metrics.main_loss = outputs.loss_dict.get(
                        "main_loss", outputs.loss
                    ).item()
                    if "aux_total" in outputs.loss_dict:
                        metrics.aux_loss = outputs.loss_dict["aux_total"].item()

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

                # Step profiler
                if self.profiler:
                    self.profiler.step()

                    # Stop profiler after profiling window
                    if self.profiler.should_stop(self.global_step):
                        self.profiler.stop()
                        self.profiler.print_summary()
                        self.profiler.export_chrome_trace()
                        self.profiler.export_stacks()
                        self.profiler.export_memory_timeline()  # NEW: Export memory timeline
                        self.profiler = None  # Disable after profiling

                # Reset accumulation
                accumulation_loss = 0.0
                accumulation_steps = 0
                step_start_time = time.time()

        # Final checkpoint
        final_metrics = TrainingMetrics(
            step=self.global_step,
            epoch=self.epoch,
            loss=self.best_loss,
            tokens_seen=self.tokens_seen,
            elapsed_time=time.time() - self.start_time,
        )

        self._save_checkpoint(final_metrics, is_final=True)
        self.logger.save_summary(
            config=asdict(self.config), final_metrics=final_metrics
        )

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Final step: {self.global_step}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Tokens seen: {self.tokens_seen:,}")
        print(f"Total time: {time.time() - self.start_time:.1f}s")
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

        print(
            f"Step {metrics.step:>6d}/{self.config.max_steps} | "
            f"Loss: {metrics.loss:.4f} | "
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

        checkpoint = {
            "step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
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
                setattr(model_config, key, value)

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
    parser = argparse.ArgumentParser(description="Train 1B LLM")

    # Model
    parser.add_argument(
        "--preset",
        type=str,
        default="1b-base",
        choices=list(PRESET_CONFIGS.keys()),
        help="Model preset",
    )

    # Training
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)

    # Experiment
    parser.add_argument("--experiment-name", type=str, default="1b_training")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)

    # Profiling
    parser.add_argument(
        "--enable-profiling", action="store_true", help="Enable memory profiling"
    )
    parser.add_argument(
        "--profiling-output-dir",
        type=str,
        default="./profiler_logs",
        help="Profiler output directory",
    )
    parser.add_argument(
        "--profiling-active-steps",
        type=int,
        default=10,
        help="Number of active profiling steps",
    )
    parser.add_argument(
        "--profiling-wait-steps",
        type=int,
        default=5,
        help="Number of wait steps before profiling",
    )
    parser.add_argument(
        "--profiling-warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps for profiling",
    )
    parser.add_argument(
        "--profiling-repeat",
        type=int,
        default=5,
        help="Number of times to reapeat profiling",
    )

    args = parser.parse_args()

    # Create training config
    training_config = TrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        seq_length=args.seq_length,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        enable_profiling=args.enable_profiling,
        profiling_output_dir=args.profiling_output_dir,
        profiling_active_steps=args.profiling_active_steps,
        profiling_wait_steps=args.profiling_wait_steps,
        profiling_warmup_steps=args.profiling_warmup_steps,
        profiling_repeat=args.profiling_repeat,
    )

    # Run training
    model, metrics = run_training(
        model_preset=args.preset, training_config=training_config
    )

    print(f"\nTraining complete! Final loss: {metrics.loss:.4f}")


if __name__ == "__main__":
    main()
