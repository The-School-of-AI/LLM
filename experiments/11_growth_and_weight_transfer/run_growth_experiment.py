"""
Run Growth Experiment End-to-End (5-Phase)

This script executes the full 5-phase growth experiment:
1. Phase 1: Train dense model
2. Phase 2: Convert to MoE
3. Phase 3: Add ghost layers + Scale hidden dimension
4. Phase 4: Add more experts (expert explosion)
5. Phase 5: YaRN context extension (256 → 1024)

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

sys.path.insert(0, str(Path(__file__).parent))

from src.model import SmolLM2, SmolLM2Config
from src.moe_model import SmolLM2MoE, MoEConfig
from src.growth import dense_to_moe, add_experts, add_layers, scale_hidden_dim, scale_context_length
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
    """Run the full 5-phase growth experiment."""
    
    print("=" * 70)
    print("🧪 GROWTH EXPERIMENT (5 Phases)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGrowth Path:")
    print("  Phase 1: Dense → Train")
    print("  Phase 2: Dense → MoE → Train")
    print("  Phase 3: MoE → +Layers + Scale Dim → Train")
    print("  Phase 4: MoE → ×Experts → Train")
    print("  Phase 5: YaRN Context Extension → Train")
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = get_device(config.get("device", "auto"))
    torch.manual_seed(config.get("seed", 42))
    
    total_planned_steps = (
        config["training"]["phase1_steps"] + 
        config["training"]["phase2_steps"] + 
        config["training"]["phase3_steps"] +
        config["training"]["phase4_steps"] +
        config["training"]["phase5_steps"]
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
                name=f"growth_4phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    # PHASE 2: Dense → MoE
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 2: Dense → MoE")
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
    
    # Clear memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # =========================================================================
    # PHASE 3: Add Ghost Layers + Scale Hidden Dimension
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 3: Add Ghost Layers + Scale Hidden Dimension")
    print("=" * 70)
    
    pre_growth_loss = measure_loss(model, dataloader, device)
    
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
        new_intermediate_size=dim_config.get("new_intermediate_size"),
        padding_mode=dim_config.get("padding_mode", "noise"),
        noise_scale=dim_config.get("noise_scale", 0.01),
    )
    
    post_growth_loss = measure_loss(model, dataloader, device)
    results["phase3_delta"] = log_transition("layers_and_scale", pre_growth_loss, post_growth_loss, wandb_run, total_steps)
    
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
    
    # Clear memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # =========================================================================
    # PHASE 4: Expert Explosion
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 4: Expert Explosion (×Experts)")
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
    results["phase4_delta"] = log_transition("expert_explosion", pre_expert_loss, post_expert_loss, wandb_run, total_steps)
    
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
    # PHASE 5: YaRN Context Extension
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 5: YaRN Context Extension")
    print("=" * 70)
    
    pre_context_loss = measure_loss(model, dataloader, device)
    
    # Apply YaRN context scaling
    context_config = config["growth"]["scale_context"]
    new_max_length = context_config["new_max_length"]
    
    print(f"\n🔧 Extending context with YaRN ({model.config.max_position_embeddings} → {new_max_length})...")
    model = scale_context_length(
        model,
        new_max_length=new_max_length,
        alpha=context_config.get("alpha", 1.0),
        beta=context_config.get("beta", 32.0),
    )
    
    # Create new dataloader with longer sequences
    print(f"\n🔧 Creating new dataloader with max_length={new_max_length}...")
    long_dataloader = get_dataloader(
        dataset_name=config["data"]["dataset_name"],
        batch_size=config["training"]["batch_size"],
        max_length=new_max_length,
        num_samples=config["data"]["num_samples"],
    )
    
    post_context_loss = measure_loss(model, long_dataloader, device)
    results["phase5_delta"] = log_transition("context_extension", pre_context_loss, post_context_loss, wandb_run, total_steps)
    
    phase5_steps = config["training"]["phase5_steps"]
    print(f"\n🚀 Training with extended context for {phase5_steps} steps...")
    
    phase5_loss = train_phase(
        model=model,
        dataloader=long_dataloader,
        num_steps=phase5_steps,
        start_step=total_steps,
        learning_rate=config["training"]["learning_rate"] * 0.1,  # Lower LR for fine-tuning
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=20,
        max_grad_norm=config["training"]["max_grad_norm"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_every=config["training"]["log_every"],
        checkpoint_every=config["training"]["checkpoint_every"],
        save_dir=config["training"]["save_dir"],
        checkpoint_prefix="phase5_yarn",
        device=device,
        wandb_run=wandb_run,
    )
    
    total_steps += phase5_steps
    results["phase5_loss"] = phase5_loss
    print(f"\n✅ Phase 5 complete! Loss: {phase5_loss:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total steps trained: {total_steps}")
    print(f"\n{'Phase':<30} {'Final Loss':<15} {'Transition Δ':<15}")
    print("-" * 60)
    print(f"{'Phase 1 (Dense)':<30} {results['phase1_loss']:<15.4f} {'-':<15}")
    print(f"{'Phase 2 (MoE)':<30} {results['phase2_loss']:<15.4f} {results['phase2_delta']:+.4f}")
    print(f"{'Phase 3 (Layers+Scale)':<30} {results['phase3_loss']:<15.4f} {results['phase3_delta']:+.4f}")
    print(f"{'Phase 4 (×Experts)':<30} {results['phase4_loss']:<15.4f} {results['phase4_delta']:+.4f}")
    print(f"{'Phase 5 (YaRN Context)':<30} {results['phase5_loss']:<15.4f} {results['phase5_delta']:+.4f}")
    print(f"\nFinal model parameters: {count_parameters(model):,}")
    print(f"Final context length: {model.config.max_position_embeddings}")
    print("=" * 70)
    
    # Check if all transitions were stable
    all_stable = (abs(results["phase2_delta"]) < 0.5 and 
                  abs(results["phase3_delta"]) < 0.5 and
                  abs(results["phase4_delta"]) < 0.5 and
                  abs(results["phase5_delta"]) < 0.5)
    if all_stable:
        print("\n🎉 SUCCESS: All transitions were STABLE (delta < 0.5)")
    else:
        print("\n⚠️ WARNING: Some transitions had spikes (delta >= 0.5)")
    
    # Save final checkpoint
    save_checkpoint(
        model, None, None, total_steps, phase5_loss,
        config["training"]["save_dir"], "final",
    )
    
    if wandb_run:
        wandb_run.log({
            "phase1_final_loss": results["phase1_loss"],
            "phase2_final_loss": results["phase2_loss"],
            "phase3_final_loss": results["phase3_loss"],
            "phase4_final_loss": results["phase4_loss"],
            "phase5_final_loss": results["phase5_loss"],
            "total_steps": total_steps,
            "final_parameters": count_parameters(model),
            "final_context_length": model.config.max_position_embeddings,
            "all_transitions_stable": all_stable,
        })
        wandb_run.finish()
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Growth Experiment (5 Phases including YaRN)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()
    
    results = run_experiment(args.config, args.wandb)
