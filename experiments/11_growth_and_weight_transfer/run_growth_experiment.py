"""
Run Growth Experiment End-to-End (3-Phase MVP)

This script executes the 3-phase growth experiment:
1. Phase 1: Train dense model
2. Phase 2: Convert to MoE + Add ghost layers
3. Phase 3: Add more experts (expert explosion)

NOTE: scale_hidden_dim is skipped for MVP (causes loss spikes, needs more research)
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.model import SmolLM2, SmolLM2Config
from src.moe_model import SmolLM2MoE, MoEConfig
from src.growth import dense_to_moe, add_experts, add_layers
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
    """Run the 3-phase growth experiment."""
    
    print("=" * 70)
    print("🧪 GROWTH EXPERIMENT (3 Phases - MVP)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGrowth Path:")
    print("  Phase 1: Dense → Train")
    print("  Phase 2: Dense → MoE + Layers → Train")
    print("  Phase 3: MoE → ×Experts → Train")
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = get_device(config.get("device", "auto"))
    torch.manual_seed(config.get("seed", 42))
    
    total_planned_steps = (
        config["training"]["phase1_steps"] + 
        config["training"]["phase2_steps"] + 
        config["training"]["phase3_steps"]
    )
    
    print(f"\n🖥️  Device: {device}")
    print(f"📚 Dataset: {config['data']['dataset_name']}")
    print(f"📊 Total steps: {total_planned_steps}")
    
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
    
    # Create dataloader
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
    # PHASE 2: Dense → MoE + Add Ghost Layers
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 2: Dense → MoE + Add Ghost Layers")
    print("=" * 70)
    
    pre_growth_loss = measure_loss(model, dataloader, device)
    
    # Convert to MoE
    moe_config = config["growth"]["dense_to_moe"]
    model = dense_to_moe(
        model,
        num_experts=moe_config["num_experts"],
        num_experts_per_tok=moe_config["num_experts_per_tok"],
    )
    
    # Add ghost layers
    layer_config = config["growth"]["add_layers"]
    print(f"\n🔧 Adding {layer_config['num_new_layers']} ghost layers...")
    model = add_layers(
        model,
        num_new_layers=layer_config["num_new_layers"],
        init_mode=layer_config["init_mode"],
    )
    
    post_growth_loss = measure_loss(model, dataloader, device)
    results["phase2_delta"] = log_transition("moe_and_layers", pre_growth_loss, post_growth_loss, wandb_run, total_steps)
    
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
    
    # Clear some memory before Phase 3
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # =========================================================================
    # PHASE 3: Expert Explosion
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 3: Expert Explosion (×Experts)")
    print("=" * 70)
    
    pre_expert_loss = measure_loss(model, dataloader, device)
    
    expert_config = config["growth"]["add_experts"]
    print(f"\n🔧 Adding {expert_config['num_new_experts']} new experts per layer...")
    model = add_experts(
        model,
        num_new_experts=expert_config["num_new_experts"],
        clone_from=expert_config["clone_from"],
    )
    
    post_expert_loss = measure_loss(model, dataloader, device)
    results["phase3_delta"] = log_transition("expert_explosion", pre_expert_loss, post_expert_loss, wandb_run, total_steps)
    
    phase3_steps = config["training"]["phase3_steps"]
    print(f"\n🚀 Training with more experts for {phase3_steps} steps...")
    
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
        checkpoint_prefix="phase3_experts",
        device=device,
        wandb_run=wandb_run,
    )
    
    total_steps += phase3_steps
    results["phase3_loss"] = phase3_loss
    print(f"\n✅ Phase 3 complete! Loss: {phase3_loss:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total steps trained: {total_steps}")
    print(f"\n{'Phase':<25} {'Final Loss':<15} {'Transition Δ':<15}")
    print("-" * 55)
    print(f"{'Phase 1 (Dense)':<25} {results['phase1_loss']:<15.4f} {'-':<15}")
    print(f"{'Phase 2 (MoE+Layers)':<25} {results['phase2_loss']:<15.4f} {results['phase2_delta']:+.4f}")
    print(f"{'Phase 3 (×Experts)':<25} {results['phase3_loss']:<15.4f} {results['phase3_delta']:+.4f}")
    print(f"\nFinal model parameters: {count_parameters(model):,}")
    print("=" * 70)
    
    # Check if all transitions were stable
    all_stable = abs(results["phase2_delta"]) < 0.5 and abs(results["phase3_delta"]) < 0.5
    if all_stable:
        print("\n🎉 SUCCESS: All transitions were STABLE (delta < 0.5)")
    else:
        print("\n⚠️ WARNING: Some transitions had spikes (delta >= 0.5)")
    
    # Save final checkpoint
    save_checkpoint(
        model, None, None, total_steps, phase3_loss,
        config["training"]["save_dir"], "final",
    )
    
    if wandb_run:
        wandb_run.log({
            "phase1_final_loss": results["phase1_loss"],
            "phase2_final_loss": results["phase2_loss"],
            "phase3_final_loss": results["phase3_loss"],
            "total_steps": total_steps,
            "final_parameters": count_parameters(model),
            "all_transitions_stable": all_stable,
        })
        wandb_run.finish()
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Growth Experiment (3 Phases)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()
    
    results = run_experiment(args.config, args.wandb)
