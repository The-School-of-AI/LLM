"""
Run Growth Experiment End-to-End (4-Phase)

This script executes the full 4-phase growth experiment:
1. Phase 1: Train dense model
2. Phase 2: Convert to MoE (keep experts same)
3. Phase 3: Add layers (ghost layers) + scale hidden dim (padding)
4. Phase 4: Multiply experts (expert explosion)

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


def measure_loss(model, dataloader, device):
    """Quick forward pass to measure current loss."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        model.to(device)
        output = model(input_ids, labels=labels)
        return output["loss"].item()


def log_transition(name, pre_loss, post_loss, wandb_run=None, step=0):
    """Log a growth transition."""
    delta = post_loss - pre_loss
    status = "STABLE!" if abs(delta) < 0.5 else "SPIKE DETECTED!"
    symbol = "✅" if abs(delta) < 0.5 else "⚠️"
    
    print(f"📊 Pre-{name} loss: {pre_loss:.4f}")
    print(f"📊 Post-{name} loss: {post_loss:.4f}")
    print(f"{symbol} Loss delta: {delta:+.4f} ({status})")
    
    if wandb_run:
        wandb_run.log({
            f"transition_{name}": True,
            f"pre_{name}_loss": pre_loss,
            f"post_{name}_loss": post_loss,
            f"{name}_loss_delta": delta,
            "step": step,
        })
    
    return delta


def run_experiment(config_path: str = "config/config.yaml", use_wandb: bool = False):
    """Run the full 4-phase growth experiment."""
    
    print("=" * 70)
    print("🧪 GROWTH EXPERIMENT (4 Phases)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGrowth Path:")
    print("  Phase 1: Dense → Train")
    print("  Phase 2: Dense → MoE → Train")
    print("  Phase 3: MoE → +Layers +Dim → Train")
    print("  Phase 4: MoE → ×Experts → Train")
    
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
                name=f"growth_4phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    results = {}
    
    # =========================================================================
    # PHASE 1: Dense Model Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 1: Dense Model Training")
    print("=" * 70)
    
    model_config = SmolLM2Config(**config["model"])
    model = SmolLM2(model_config)
    print(f"✓ Created dense model: {count_parameters(model):,} parameters")
    
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
    results["phase1_loss"] = phase1_loss
    print(f"\n✅ Phase 1 complete! Loss: {phase1_loss:.4f}")
    
    # =========================================================================
    # PHASE 2: Dense → MoE Conversion
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 2: Dense → MoE Conversion")
    print("=" * 70)
    
    pre_moe_loss = measure_loss(model, dataloader, device)
    
    # Convert to MoE
    moe_config = config["growth"]["dense_to_moe"]
    model = dense_to_moe(
        model,
        num_experts=moe_config["num_experts"],
        num_experts_per_tok=moe_config["num_experts_per_tok"],
    )
    
    post_moe_loss = measure_loss(model, dataloader, device)
    results["phase2_delta"] = log_transition("moe_conversion", pre_moe_loss, post_moe_loss, wandb_run, total_steps)
    
    # Train Phase 2
    phase2_steps = config["training"]["phase2_steps"]
    print(f"\n🚀 Training MoE for {phase2_steps} steps...")
    
    phase2_loss = train_phase(
        model=model,
        dataloader=dataloader,
        num_steps=phase2_steps,
        start_step=total_steps,
        learning_rate=config["training"]["learning_rate"] * 0.5,
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=50,
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
    results["phase2_loss"] = phase2_loss
    print(f"\n✅ Phase 2 complete! Loss: {phase2_loss:.4f}")
    
    # =========================================================================
    # PHASE 3: Add Layers + Scale Hidden Dimension
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 3: Add Layers + Scale Hidden Dimension")
    print("=" * 70)
    
    pre_scale_loss = measure_loss(model, dataloader, device)
    
    # Step 3a: Add ghost layers
    layer_config = config["growth"]["add_layers"]
    print(f"\n🔧 Adding {layer_config['num_new_layers']} ghost layers...")
    model = add_layers(
        model,
        num_new_layers=layer_config["num_new_layers"],
        init_mode=layer_config["init_mode"],
    )
    
    # Step 3b: Scale hidden dimension
    dim_config = config["growth"]["scale_hidden_dim"]
    print(f"\n🔧 Scaling hidden dimension to {dim_config['new_hidden_size']}...")
    model = scale_hidden_dim(
        model,
        new_hidden_size=dim_config["new_hidden_size"],
        padding_mode=dim_config["padding_mode"],
    )
    
    post_scale_loss = measure_loss(model, dataloader, device)
    results["phase3_delta"] = log_transition("layers_and_dim", pre_scale_loss, post_scale_loss, wandb_run, total_steps)
    
    # Train Phase 3
    phase3_steps = config["training"]["phase3_steps"]
    print(f"\n🚀 Training scaled model for {phase3_steps} steps...")
    
    phase3_loss = train_phase(
        model=model,
        dataloader=dataloader,
        num_steps=phase3_steps,
        start_step=total_steps,
        learning_rate=config["training"]["learning_rate"] * 0.25,
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
    results["phase3_loss"] = phase3_loss
    print(f"\n✅ Phase 3 complete! Loss: {phase3_loss:.4f}")
    
    # =========================================================================
    # PHASE 4: Expert Explosion (Multiply Experts)
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 4: Expert Explosion (Multiply Experts)")
    print("=" * 70)
    
    pre_expert_loss = measure_loss(model, dataloader, device)
    
    # Multiply experts
    expert_config = config["growth"]["add_experts"]
    print(f"\n🔧 Adding {expert_config['num_new_experts']} new experts per layer...")
    model = add_experts(
        model,
        num_new_experts=expert_config["num_new_experts"],
        clone_from=expert_config["clone_from"],
    )
    
    post_expert_loss = measure_loss(model, dataloader, device)
    results["phase4_delta"] = log_transition("expert_explosion", pre_expert_loss, post_expert_loss, wandb_run, total_steps)
    
    # Train Phase 4
    phase4_steps = config["training"]["phase4_steps"]
    print(f"\n🚀 Training with more experts for {phase4_steps} steps...")
    
    phase4_loss = train_phase(
        model=model,
        dataloader=dataloader,
        num_steps=phase4_steps,
        start_step=total_steps,
        learning_rate=config["training"]["learning_rate"] * 0.125,
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=50,
        max_grad_norm=config["training"]["max_grad_norm"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_every=config["training"]["log_every"],
        checkpoint_every=config["training"]["checkpoint_every"],
        save_dir=config["training"]["save_dir"],
        checkpoint_prefix="phase4_experts",
        device=device,
        wandb_run=wandb_run,
    )
    
    total_steps += phase4_steps
    results["phase4_loss"] = phase4_loss
    print(f"\n✅ Phase 4 complete! Loss: {phase4_loss:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total steps trained: {total_steps}")
    print(f"\n{'Phase':<20} {'Final Loss':<15} {'Transition Δ':<15}")
    print("-" * 50)
    print(f"{'Phase 1 (Dense)':<20} {results['phase1_loss']:<15.4f} {'-':<15}")
    print(f"{'Phase 2 (MoE)':<20} {results['phase2_loss']:<15.4f} {results['phase2_delta']:+.4f}")
    print(f"{'Phase 3 (Layers+Dim)':<20} {results['phase3_loss']:<15.4f} {results['phase3_delta']:+.4f}")
    print(f"{'Phase 4 (Experts)':<20} {results['phase4_loss']:<15.4f} {results['phase4_delta']:+.4f}")
    print(f"\nFinal model parameters: {count_parameters(model):,}")
    print("=" * 70)
    
    # Save final checkpoint
    save_checkpoint(
        model, None, None, total_steps, phase4_loss,
        config["training"]["save_dir"], "final",
    )
    
    if wandb_run:
        wandb_run.log({
            "phase1_final_loss": results["phase1_loss"],
            "phase2_final_loss": results["phase2_loss"],
            "phase3_final_loss": results["phase3_loss"],
            "phase4_final_loss": results["phase4_loss"],
            "total_steps": total_steps,
            "final_parameters": count_parameters(model),
        })
        wandb_run.finish()
    
    print(f"\n🎉 Experiment complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Growth Experiment (4 Phases)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()
    
    results = run_experiment(args.config, args.wandb)
