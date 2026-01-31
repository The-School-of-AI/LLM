"""
Run Growth Experiment End-to-End

This script executes the full 3-phase growth experiment:
1. Phase 1: Train dense model for 1000 steps
2. Phase 2: Convert to MoE, train for 1000 more steps
3. Phase 3: Scale the model, train for 1000 more steps

Goal: Demonstrate that loss does NOT spike at transitions.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import SmolLM2, SmolLM2Config
from src.moe_model import SmolLM2MoE, MoEConfig
from src.growth import dense_to_moe, add_experts, add_layers, scale_hidden_dim
from src.dataset import get_dataloader
from train import train_phase, get_device, save_checkpoint, count_parameters


def run_experiment(config_path: str = "config/config.yaml", use_wandb: bool = False):
    """Run the full 3-phase growth experiment."""
    
    print("=" * 70)
    print("🧪 GROWTH EXPERIMENT")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = get_device(config.get("device", "auto"))
    torch.manual_seed(config.get("seed", 42))
    print(f"\n🖥️  Device: {device}")
    
    # Initialize WandB if requested
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config["training"].get("wandb_project", "growth-experiment"),
                name=f"growth_3phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config,
            )
        except ImportError:
            print("⚠️  WandB not installed. Skipping logging.")
    
    # Create dataloader (shared across all phases)
    dataloader = get_dataloader(
        dataset_name=config["data"]["dataset_name"],
        batch_size=config["training"]["batch_size"],
        max_length=config["training"]["max_length"],
        vocab_size=config["model"]["vocab_size"],
        num_samples=config["data"].get("num_samples", 10000),
    )
    
    total_steps = 0
    
    # =========================================================================
    # PHASE 1: Dense Model Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 1: Dense Model Training")
    print("=" * 70)
    
    # Create dense model
    model_config = SmolLM2Config(**config["model"])
    model = SmolLM2(model_config)
    print(f"✓ Created dense model: {count_parameters(model):,} parameters")
    
    # Train Phase 1
    phase1_steps = config["training"]["phase1_steps"]
    print(f"\n🚀 Training for {phase1_steps} steps...")
    
    phase1_loss = train_phase(
        model=model,
        dataloader=dataloader,
        num_steps=phase1_steps,
        start_step=0,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        max_grad_norm=config["training"]["max_grad_norm"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_every=config["training"]["log_every"],
        checkpoint_every=config["training"]["checkpoint_every"],
        save_dir=config["training"]["save_dir"],
        checkpoint_prefix="phase1_dense",
        device=device,
        wandb_run=wandb_run,
    )
    
    total_steps += phase1_steps
    print(f"\n✅ Phase 1 complete! Loss: {phase1_loss:.4f}")
    
    # Log transition point
    if wandb_run:
        wandb_run.log({"phase": 1, "transition": "phase1_end", "step": total_steps})
    
    # =========================================================================
    # PHASE 2: Dense → MoE Conversion
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 2: Dense → MoE Conversion")
    print("=" * 70)
    
    # Quick forward pass before conversion to get baseline loss
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        model.to(device)
        pre_moe_output = model(input_ids, labels=labels)
        pre_moe_loss = pre_moe_output["loss"].item()
    print(f"📊 Pre-conversion loss: {pre_moe_loss:.4f}")
    
    # Convert to MoE
    moe_config = config["growth"]["dense_to_moe"]
    model = dense_to_moe(
        model,
        num_experts=moe_config["num_experts"],
        num_experts_per_tok=moe_config["num_experts_per_tok"],
    )
    
    # Quick forward pass after conversion
    model.eval()
    with torch.no_grad():
        model.to(device)
        post_moe_output = model(input_ids, labels=labels)
        post_moe_loss = post_moe_output["loss"].item()
    print(f"📊 Post-conversion loss: {post_moe_loss:.4f}")
    
    loss_delta = post_moe_loss - pre_moe_loss
    if abs(loss_delta) < 0.5:
        print(f"✅ Loss delta: {loss_delta:+.4f} (STABLE!)")
    else:
        print(f"⚠️  Loss delta: {loss_delta:+.4f} (SPIKE DETECTED!)")
    
    if wandb_run:
        wandb_run.log({
            "transition": "dense_to_moe",
            "pre_conversion_loss": pre_moe_loss,
            "post_conversion_loss": post_moe_loss,
            "loss_delta": loss_delta,
            "step": total_steps,
        })
    
    # Train Phase 2
    phase2_steps = config["training"]["phase2_steps"]
    print(f"\n🚀 Training MoE for {phase2_steps} steps...")
    
    phase2_loss = train_phase(
        model=model,
        dataloader=dataloader,
        num_steps=phase2_steps,
        start_step=total_steps,
        learning_rate=config["training"]["learning_rate"] * 0.5,  # Lower LR after conversion
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=50,  # Short warmup
        max_grad_norm=config["training"]["max_grad_norm"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_every=config["training"]["log_every"],
        checkpoint_every=config["training"]["checkpoint_every"],
        save_dir=config["training"]["save_dir"],
        checkpoint_prefix="phase2_moe",
        device=device,
        wandb_run=wandb_run,
    )
    
    total_steps += phase2_steps
    print(f"\n✅ Phase 2 complete! Loss: {phase2_loss:.4f}")
    
    if wandb_run:
        wandb_run.log({"phase": 2, "transition": "phase2_end", "step": total_steps})
    
    # =========================================================================
    # PHASE 3: Model Scaling
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 3: Model Scaling")
    print("=" * 70)
    
    # Quick forward pass before scaling
    model.eval()
    with torch.no_grad():
        model.to(device)
        pre_scale_output = model(input_ids, labels=labels)
        pre_scale_loss = pre_scale_output["loss"].item()
    print(f"📊 Pre-scaling loss: {pre_scale_loss:.4f}")
    
    # Apply scaling based on config
    scaling_method = config["growth"]["scaling_method"]
    print(f"🔧 Scaling method: {scaling_method}")
    
    if scaling_method == "add_experts":
        scale_config = config["growth"]["add_experts"]
        model = add_experts(
            model,
            num_new_experts=scale_config["num_new_experts"],
            clone_from=scale_config["clone_from"],
        )
    elif scaling_method == "add_layers":
        scale_config = config["growth"]["add_layers"]
        model = add_layers(
            model,
            num_new_layers=scale_config["num_new_layers"],
            init_mode=scale_config["init_mode"],
        )
    elif scaling_method == "scale_hidden_dim":
        scale_config = config["growth"]["scale_hidden_dim"]
        model = scale_hidden_dim(
            model,
            new_hidden_size=scale_config["new_hidden_size"],
            padding_mode=scale_config["padding_mode"],
        )
    
    # Quick forward pass after scaling (need new input for dimension changes)
    model.eval()
    with torch.no_grad():
        model.to(device)
        # Recreate batch for potential dimension changes
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        post_scale_output = model(input_ids, labels=labels)
        post_scale_loss = post_scale_output["loss"].item()
    print(f"📊 Post-scaling loss: {post_scale_loss:.4f}")
    
    loss_delta = post_scale_loss - pre_scale_loss
    if abs(loss_delta) < 0.5:
        print(f"✅ Loss delta: {loss_delta:+.4f} (STABLE!)")
    else:
        print(f"⚠️  Loss delta: {loss_delta:+.4f} (SPIKE DETECTED!)")
    
    if wandb_run:
        wandb_run.log({
            "transition": f"scale_{scaling_method}",
            "pre_scaling_loss": pre_scale_loss,
            "post_scaling_loss": post_scale_loss,
            "loss_delta": loss_delta,
            "step": total_steps,
        })
    
    # Train Phase 3
    phase3_steps = config["training"]["phase3_steps"]
    print(f"\n🚀 Training scaled model for {phase3_steps} steps...")
    
    phase3_loss = train_phase(
        model=model,
        dataloader=dataloader,
        num_steps=phase3_steps,
        start_step=total_steps,
        learning_rate=config["training"]["learning_rate"] * 0.25,  # Even lower LR
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=50,
        max_grad_norm=config["training"]["max_grad_norm"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_every=config["training"]["log_every"],
        checkpoint_every=config["training"]["checkpoint_every"],
        save_dir=config["training"]["save_dir"],
        checkpoint_prefix="phase3_scaled",
        device=device,
        wandb_run=wandb_run,
    )
    
    total_steps += phase3_steps
    print(f"\n✅ Phase 3 complete! Loss: {phase3_loss:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total steps trained: {total_steps}")
    print(f"Phase 1 (Dense) final loss: {phase1_loss:.4f}")
    print(f"Phase 2 (MoE) final loss: {phase2_loss:.4f}")
    print(f"Phase 3 (Scaled) final loss: {phase3_loss:.4f}")
    print(f"\nFinal model parameters: {count_parameters(model):,}")
    print("=" * 70)
    
    # Save final checkpoint
    save_checkpoint(
        model,
        None,  # No optimizer for final
        None,  # No scheduler for final
        total_steps,
        phase3_loss,
        config["training"]["save_dir"],
        "final",
    )
    
    if wandb_run:
        wandb_run.log({
            "phase1_final_loss": phase1_loss,
            "phase2_final_loss": phase2_loss,
            "phase3_final_loss": phase3_loss,
            "total_steps": total_steps,
            "final_parameters": count_parameters(model),
        })
        wandb_run.finish()
    
    print(f"\n🎉 Experiment complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        "phase1_loss": phase1_loss,
        "phase2_loss": phase2_loss,
        "phase3_loss": phase3_loss,
        "total_steps": total_steps,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Growth Experiment")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()
    
    results = run_experiment(args.config, args.wandb)
