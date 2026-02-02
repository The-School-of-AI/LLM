"""
Training Script for Growth Experiments

Supports:
- Dense model training
- MoE model training
- Checkpoint save/load
- WandB logging
- Gradient accumulation
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import SmolLM2, SmolLM2Config
from src.moe_model import SmolLM2MoE, MoEConfig
from src.dataset import get_dataloader


def get_device(config_device: str = "auto") -> torch.device:
    """Determine the best available device."""
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config_device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    step: int,
    loss: float,
    save_dir: str,
    prefix: str = "checkpoint",
    dataloader_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a training checkpoint with dataloader state for resumable training."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{prefix}_step_{step}.pt")
    
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
        "config": model.config,
        # Dataloader state for resumable training
        "dataloader_state": dataloader_state or {
            "samples_seen": step,
            "rng_state": torch.get_rng_state(),
        },
    }, checkpoint_path)
    
    print(f"  💾 Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler = None,
    restore_dataloader_state: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint and optionally restore dataloader state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Restore RNG state for dataloader reproducibility
    if restore_dataloader_state and "dataloader_state" in checkpoint:
        dataloader_state = checkpoint["dataloader_state"]
        if "rng_state" in dataloader_state:
            torch.set_rng_state(dataloader_state["rng_state"])
            print(f"  🔄 Restored RNG state for dataloader")
    
    print(f"  📂 Loaded checkpoint from step {checkpoint['step']}")
    return checkpoint


def train_phase(
    model: nn.Module,
    dataloader,
    num_steps: int,
    start_step: int = 0,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 4,
    log_every: int = 10,
    checkpoint_every: int = 500,
    save_dir: str = "./checkpoints",
    checkpoint_prefix: str = "checkpoint",
    device: torch.device = torch.device("cpu"),
    wandb_run = None,
    dataset = None,  # NEW: For tracking samples_seen
    initial_samples_seen: int = 0,  # NEW: Starting sample count for resume
) -> float:
    """
    Train a model for a specified number of steps.
    
    Returns:
        Final loss value
    """
    model.to(device)
    model.train()
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=learning_rate * 0.1,
    )
    
    # Training loop
    data_iter = iter(dataloader)
    accumulated_loss = 0.0
    step = start_step
    
    while step < start_step + num_steps:
        optimizer.zero_grad()
        
        # Gradient accumulation
        batch_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] / gradient_accumulation_steps
            loss.backward()
            batch_loss += loss.item()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        step += 1
        accumulated_loss += batch_loss
        
        # Logging
        if step % log_every == 0:
            avg_loss = accumulated_loss / log_every
            lr = scheduler.get_last_lr()[0]
            
            log_msg = f"  Step {step}/{start_step + num_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}"
            
            # Add aux_loss for MoE models
            if "aux_loss" in outputs:
                log_msg += f" | Aux: {outputs['aux_loss']:.4f}"
            
            print(log_msg)
            
            if wandb_run:
                log_dict = {
                    "loss": avg_loss,
                    "learning_rate": lr,
                    "step": step,
                }
                if "aux_loss" in outputs:
                    log_dict["aux_loss"] = outputs["aux_loss"]
                wandb_run.log(log_dict)
            
            accumulated_loss = 0.0
        
        # Checkpointing
        if step % checkpoint_every == 0:
            # Calculate samples seen
            batch_size = dataloader.batch_size or 1
            steps_taken = step - start_step
            samples_seen = initial_samples_seen + (steps_taken * batch_size * gradient_accumulation_steps)
            
            # Get dataset state if available
            dataloader_state = {
                "samples_seen": samples_seen,
                "rng_state": torch.get_rng_state(),
            }
            if dataset is not None and hasattr(dataset, 'state_dict'):
                dataloader_state.update(dataset.state_dict())
            
            save_checkpoint(
                model, optimizer, scheduler, step, batch_loss,
                save_dir, checkpoint_prefix, dataloader_state,
            )
    
    # Final checkpoint with samples_seen (skip if already saved at this step)
    if step % checkpoint_every != 0:
        batch_size = dataloader.batch_size or 1
        steps_taken = step - start_step
        samples_seen = initial_samples_seen + (steps_taken * batch_size * gradient_accumulation_steps)
        
        dataloader_state = {
            "samples_seen": samples_seen,
            "rng_state": torch.get_rng_state(),
        }
        if dataset is not None and hasattr(dataset, 'state_dict'):
            dataloader_state.update(dataset.state_dict())
        
        save_checkpoint(
            model, optimizer, scheduler, step, batch_loss,
            save_dir, checkpoint_prefix, dataloader_state,
        )
    
    return batch_loss


def main():
    parser = argparse.ArgumentParser(description="Train SmolLM2 model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--phase", type=str, choices=["dense", "moe"], default="dense", help="Model type")
    parser.add_argument("--steps", type=int, default=None, help="Override number of steps")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Device
    device = get_device(config.get("device", "auto"))
    print(f"🖥️  Using device: {device}")
    
    # Seed
    torch.manual_seed(config.get("seed", 42))
    
    # Create model
    if args.phase == "dense":
        model_config = SmolLM2Config(**config["model"])
        model = SmolLM2(model_config)
        checkpoint_prefix = "dense"
    else:
        moe_cfg = {**config["model"], **config["moe"]}
        model_config = MoEConfig(**moe_cfg)
        model = SmolLM2MoE(model_config)
        checkpoint_prefix = "moe"
    
    print(f"📦 Model: {args.phase.upper()}")
    print(f"   Parameters: {count_parameters(model):,}")
    
    # Load checkpoint if provided
    start_step = 0
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model)
        start_step = checkpoint["step"]
    
    # Dataloader
    dataloader = get_dataloader(
        dataset_name=config["data"]["dataset_name"],
        batch_size=config["training"]["batch_size"],
        max_length=config["training"]["max_length"],
        vocab_size=config["model"]["vocab_size"],
        num_samples=config["data"].get("num_samples", 10000),
    )
    
    # WandB
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            run_name = config["training"].get("wandb_run_name") or f"{args.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_run = wandb.init(
                project=config["training"].get("wandb_project", "growth-experiment"),
                name=run_name,
                config={**config, "phase": args.phase},
            )
        except ImportError:
            print("⚠️  WandB not installed. Skipping logging.")
    
    # Train
    num_steps = args.steps or config["training"]["phase1_steps"]
    print(f"\n🚀 Starting training for {num_steps} steps...")
    
    final_loss = train_phase(
        model=model,
        dataloader=dataloader,
        num_steps=num_steps,
        start_step=start_step,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        max_grad_norm=config["training"]["max_grad_norm"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_every=config["training"]["log_every"],
        checkpoint_every=config["training"]["checkpoint_every"],
        save_dir=config["training"]["save_dir"],
        checkpoint_prefix=checkpoint_prefix,
        device=device,
        wandb_run=wandb_run,
    )
    
    print(f"\n✅ Training complete! Final loss: {final_loss:.4f}")
    
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
