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

import os
import sys
import time
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, fields
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig, get_preset_config, PRESET_CONFIGS
from models.llm import LLM, create_model


def load_config_from_yaml(config_path: str) -> Tuple[ModelConfig, Dict[str, Any]]:
    """
    Load model and training config from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Tuple of (ModelConfig, training_config_dict)
    """
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Extract training config (if present)
    training_data = config_data.pop('training', {})
    
    # Load model config
    model_config = ModelConfig.from_dict(config_data)
    
    return model_config, training_data


def training_config_from_dict(data: Dict[str, Any]) -> 'TrainingConfig':
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
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics_dict) + '\n')
    
    def save_summary(self, config: Dict, final_metrics: TrainingMetrics):
        """Save training summary."""
        summary = {
            'config': config,
            'final_metrics': asdict(final_metrics),
            'history_length': len(self.metrics_history)
        }
        
        summary_file = self.log_file.with_suffix('.summary.json')
        with open(summary_file, 'w') as f:
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
        style: str = "cosine"
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
            param_group['lr'] = lr
            
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
        
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        if self.style == "linear":
            return self.min_lr + (self.max_lr - self.min_lr) * (1 - progress)
        elif self.style == "cosine":
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return self.max_lr


class RandomTextDataset(Dataset):
    """
    Random dataset for testing/development.
    
    In production, replace with real tokenized dataset.
    """
    
    def __init__(
        self,
        vocab_size: int,
        seq_length: int,
        num_samples: int,
        seed: int = 42
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
            'input_ids': tokens,
            # Model forward already shifts for next-token prediction.
            'labels': tokens.clone()
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
        model: LLM,
        train_dataloader: DataLoader,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        eval_dataloader: Optional[DataLoader] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = training_config
        self.model_config = model_config

        # Device - supports CUDA, MPS (Apple Silicon), and CPU
        self.device = get_best_device(training_config.device)
        self.model = self.model.to(self.device)
        model.gradient_checkpointing_enable()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # LR Scheduler
        self.lr_scheduler = LRScheduler(
            optimizer=self.optimizer,
            warmup_steps=training_config.warmup_steps,
            max_steps=training_config.max_steps,
            max_lr=training_config.learning_rate,
            min_lr=training_config.min_learning_rate,
            style=training_config.lr_decay_style
        )

        # Mixed precision - CUDA supports both float16 and bfloat16, MPS supports float16 only
        self.use_amp = training_config.use_amp and (
            self.device.type == "cuda" or
            (self.device.type == "mps" and training_config.amp_dtype == "float16")
        )
        self.amp_dtype = getattr(torch, training_config.amp_dtype)

        # GradScaler only works with CUDA float16
        scaler_enabled = (
            self.use_amp and
            self.device.type == "cuda" and
            training_config.amp_dtype == "float16"
        )
        self.scaler = GradScaler("cuda", enabled=scaler_enabled)
        
        # Logging
        self.logger = MetricsLogger(
            log_dir=training_config.checkpoint_dir,
            experiment_name=training_config.experiment_name
        )
        
        # State
        self.global_step = 0
        self.epoch = 0
        self.tokens_seen = 0
        self.best_loss = float('inf')
        self.start_time = None
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps
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
        print(f"Batch size: {self.config.batch_size} x {self.config.gradient_accumulation_steps}")
        print(f"{'='*60}\n")
        
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
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with device-aware autocast
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(input_ids=input_ids, labels=labels)
                micro_loss = outputs.loss
                loss = micro_loss / self.config.gradient_accumulation_steps

            # Backward pass (GradScaler only for CUDA float16)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_loss += micro_loss.item()
            accumulation_steps += 1

            # Gradient accumulation step
            if accumulation_steps >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
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
                    self.config.batch_size *
                    self.config.seq_length *
                    self.config.gradient_accumulation_steps
                )
                self.tokens_seen += tokens_in_step
                
                # Calculate metrics
                step_time = time.time() - step_start_time
                tokens_per_second = tokens_in_step / step_time
                samples_per_second = (
                    self.config.batch_size *
                    self.config.gradient_accumulation_steps
                ) / step_time
                
                # Log metrics
                metrics = TrainingMetrics(
                    step=self.global_step,
                    epoch=self.epoch,
                    loss=accumulation_loss / max(1, accumulation_steps),
                    learning_rate=current_lr,
                    tokens_per_second=tokens_per_second,
                    samples_per_second=samples_per_second,
                    grad_norm=grad_norm,
                    tokens_seen=self.tokens_seen,
                    elapsed_time=time.time() - self.start_time
                )
                
                # Add MTP loss components if available
                if outputs.loss_dict is not None:
                    metrics.main_loss = outputs.loss_dict.get('main_loss', outputs.loss).item()
                    if 'aux_total' in outputs.loss_dict:
                        metrics.aux_loss = outputs.loss_dict['aux_total'].item()
                
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
                accumulation_steps = 0
                step_start_time = time.time()
        
        # Final checkpoint
        final_metrics = TrainingMetrics(
            step=self.global_step,
            epoch=self.epoch,
            loss=self.best_loss,
            tokens_seen=self.tokens_seen,
            elapsed_time=time.time() - self.start_time
        )
        
        self._save_checkpoint(final_metrics, is_final=True)
        self.logger.save_summary(
            config=asdict(self.config),
            final_metrics=final_metrics
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
        eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
        
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
            checkpoint_path = checkpoint_dir / f"{self.config.experiment_name}_step{metrics.step}.pt"
        
        checkpoint = {
            'step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_step': self.lr_scheduler.current_step,
            'metrics': asdict(metrics),
            'model_config': self.model_config.to_dict(),
            'training_config': asdict(self.config),
            'best_loss': self.best_loss,
            'tokens_seen': self.tokens_seen
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Saved checkpoint: {checkpoint_path}")


def run_training(
    model_preset: str = "1b-base",
    training_config: Optional[TrainingConfig] = None,
    model_config_overrides: Optional[Dict] = None
) -> Tuple[LLM, TrainingMetrics]:
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
    model = LLM(model_config)
    
    # Create dataset
    dataset = RandomTextDataset(
        vocab_size=model_config.vocab_size,
        seq_length=training_config.seq_length,
        num_samples=training_config.max_steps * training_config.batch_size * 2,
        seed=training_config.seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        training_config=training_config,
        model_config=model_config
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
        """
    )
    
    # Configuration source (mutually exclusive conceptually, but --config takes precedence)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/1b_base.yaml). Takes precedence over --preset."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="1b-base",
        choices=list(PRESET_CONFIGS.keys()),
        help="Model preset (used if --config not provided)"
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
        help="Device to use: auto (best available), cuda, mps (Apple Silicon), or cpu"
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
        training_config = training_config_from_dict(training_dict) if training_dict else TrainingConfig()
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

    # Set seed
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    
    # Create model
    model = LLM(model_config)
    
    # Create dataset
    dataset = RandomTextDataset(
        vocab_size=model_config.vocab_size,
        seq_length=training_config.seq_length,
        num_samples=training_config.max_steps * training_config.batch_size * 2,
        seed=training_config.seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        training_config=training_config,
        model_config=model_config
    )
    
    # Train
    metrics = trainer.train()
    
    print(f"\nTraining complete! Final loss: {metrics.loss:.4f}")


if __name__ == "__main__":
    main()
