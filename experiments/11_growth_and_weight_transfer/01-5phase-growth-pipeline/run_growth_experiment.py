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
from typing import Optional, Tuple
import glob
import re

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


def find_latest_checkpoint(save_dir: str, config: dict = None) -> Tuple[Optional[str], int, int, bool]:
    """
    Scan checkpoint directory and find the latest checkpoint.
    
    Args:
        save_dir: Directory containing checkpoints
        config: Optional config to determine if phase completed fully
    
    Returns:
        Tuple of (checkpoint_path, phase_number, step_number, phase_complete)
        Returns (None, 0, 0, False) if no checkpoints found.
    """
    if not os.path.exists(save_dir):
        return None, 0, 0, False
    
    checkpoints = glob.glob(os.path.join(save_dir, "*.pt"))
    if not checkpoints:
        return None, 0, 0, False
    
    # Parse checkpoint filenames like "phase3_scaled_step_2500.pt"
    checkpoint_info = []
    for ckpt in checkpoints:
        basename = os.path.basename(ckpt)
        # Skip "final" checkpoints
        if basename.startswith("final"):
            continue
        
        step_match = re.search(r'step_(\d+)', basename)
        phase_match = re.search(r'phase(\d)', basename)
        
        if step_match and phase_match:
            step = int(step_match.group(1))
            phase = int(phase_match.group(1))
            checkpoint_info.append((ckpt, phase, step))
    
    if not checkpoint_info:
        return None, 0, 0, False
    
    # Sort by step number (descending) and return latest
    checkpoint_info.sort(key=lambda x: x[2], reverse=True)
    latest_ckpt, latest_phase, latest_step = checkpoint_info[0]
    
    # Check if phase is complete (if config provided)
    phase_complete = False
    if config:
        phase_end_steps = {
            1: config["training"]["phase1_steps"],
            2: config["training"]["phase1_steps"] + config["training"]["phase2_steps"],
            3: config["training"]["phase1_steps"] + config["training"]["phase2_steps"] + config["training"]["phase3_steps"],
            4: config["training"]["phase1_steps"] + config["training"]["phase2_steps"] + config["training"]["phase3_steps"] + config["training"]["phase4_steps"],
            5: config["training"]["phase1_steps"] + config["training"]["phase2_steps"] + config["training"]["phase3_steps"] + config["training"]["phase4_steps"] + config["training"]["phase5_steps"],
        }
        expected_end = phase_end_steps.get(latest_phase, 0)
        phase_complete = (latest_step >= expected_end)
    
    print(f"🔍 Found {len(checkpoint_info)} checkpoints")
    print(f"   Latest: {os.path.basename(latest_ckpt)} (Phase {latest_phase}, Step {latest_step})")
    if config and not phase_complete:
        print(f"   ⚠️  Phase {latest_phase} is incomplete (mid-phase checkpoint)")
    
    return latest_ckpt, latest_phase, latest_step, phase_complete


def run_experiment(config_path: str = "config/config.yaml", use_wandb: bool = False, resume_phase: int = 0, resume_checkpoint_path: str = None):
    """Run the full 5-phase growth experiment.
    
    Args:
        config_path: Path to config YAML
        use_wandb: Enable WandB logging
        resume_phase: Phase to resume from (1-5). If 0, start fresh.
        resume_checkpoint_path: If provided, load this checkpoint directly (for mid-phase resume).
                               If None, loads end-of-previous-phase checkpoint.
    """
    
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
    
    # Create dataloader (will be recreated with skip_samples if resuming)
    skip_samples = 0  # Will be set from checkpoint if resuming
    dataloader = None
    dataset = None
    
    total_steps = 0
    results = {}
    model = None  # Will be set based on resume_phase
    
    # =========================================================================
    # =========================================================================
    if resume_phase > 0:
        print(f"\n🔄 RESUMING FROM PHASE {resume_phase}")
        print("=" * 70)
        
        # Determine checkpoint path
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            # Mid-phase resume: use provided checkpoint path directly
            checkpoint_path = resume_checkpoint_path
            is_mid_phase = True
        else:
            # Normal resume: look for end-of-previous-phase checkpoint
            checkpoint_map = {
                2: ("phase1_dense", config["training"]["phase1_steps"]),
                3: ("phase2_moe", config["training"]["phase1_steps"] + config["training"]["phase2_steps"]),
                4: ("phase3_scaled", config["training"]["phase1_steps"] + config["training"]["phase2_steps"] + config["training"]["phase3_steps"]),
                5: ("phase4_experts", config["training"]["phase1_steps"] + config["training"]["phase2_steps"] + config["training"]["phase3_steps"] + config["training"]["phase4_steps"]),
            }
            if resume_phase in checkpoint_map:
                prefix, expected_step = checkpoint_map[resume_phase]
                checkpoint_path = os.path.join(config["training"]["save_dir"], f"{prefix}_step_{expected_step}.pt")
            else:
                checkpoint_path = None
            is_mid_phase = False
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            total_steps = checkpoint["step"]
            
            # Extract samples_seen for dataloader skip
            if "dataloader_state" in checkpoint:
                skip_samples = checkpoint["dataloader_state"].get("samples_seen", 0)
                print(f"📊 Will skip {skip_samples:,} previously seen samples")
            
            # Reconstruct the model at the correct architecture stage
            if resume_phase == 1:
                # Mid-phase Phase 1: load dense model
                model_config = SmolLM2Config(**config["model"])
                model = SmolLM2(model_config)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"✓ Loaded dense model from step {total_steps}")
                
            elif resume_phase == 2:
                # Phase 2 has two cases:
                # 1. End of Phase 1 checkpoint (dense model) - will convert to MoE
                # 2. Mid-Phase 2 checkpoint (already MoE) - load directly
                if is_mid_phase:
                    # Mid-phase: checkpoint is already MoE
                    from src.moe_model import SmolLM2MoE, MoEConfig
                    moe_cfg = {**config["model"], 
                               "num_experts": config["growth"]["dense_to_moe"]["num_experts"],
                               "num_experts_per_tok": config["growth"]["dense_to_moe"]["num_experts_per_tok"]}
                    model_config = MoEConfig(**moe_cfg)
                    model = SmolLM2MoE(model_config)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    print(f"✓ Loaded MoE model from step {total_steps} (mid-Phase 2)")
                else:
                    # End of Phase 1: load dense model, will be converted to MoE
                    model_config = SmolLM2Config(**config["model"])
                    model = SmolLM2(model_config)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    print(f"✓ Loaded dense model from step {total_steps}")
                
            elif resume_phase >= 3:
                # Load MoE model (need to recreate MoE structure first)
                from src.moe_model import SmolLM2MoE, MoEConfig
                
                if resume_phase == 3:
                    # Phase 2 checkpoint is base MoE
                    moe_cfg = {**config["model"], 
                               "num_experts": config["growth"]["dense_to_moe"]["num_experts"],
                               "num_experts_per_tok": config["growth"]["dense_to_moe"]["num_experts_per_tok"]}
                    model_config = MoEConfig(**moe_cfg)
                    model = SmolLM2MoE(model_config)
                elif resume_phase == 4:
                    # Phase 3 checkpoint has scaled dimensions
                    moe_cfg = {**config["model"],
                               "hidden_size": config["growth"]["scale_hidden_dim"]["new_hidden_size"],
                               "intermediate_size": config["growth"]["scale_hidden_dim"]["new_intermediate_size"],
                               "num_hidden_layers": config["model"]["num_hidden_layers"] + config["growth"]["add_layers"]["num_new_layers"],
                               "num_attention_heads": config["growth"]["scale_hidden_dim"]["new_hidden_size"] // 64,
                               "num_key_value_heads": (config["growth"]["scale_hidden_dim"]["new_hidden_size"] // 64) // 3,
                               "num_experts": config["growth"]["dense_to_moe"]["num_experts"],
                               "num_experts_per_tok": config["growth"]["dense_to_moe"]["num_experts_per_tok"]}
                    model_config = MoEConfig(**moe_cfg)
                    model = SmolLM2MoE(model_config)
                elif resume_phase == 5:
                    # Phase 4 checkpoint has doubled experts
                    moe_cfg = {**config["model"],
                               "hidden_size": config["growth"]["scale_hidden_dim"]["new_hidden_size"],
                               "intermediate_size": config["growth"]["scale_hidden_dim"]["new_intermediate_size"],
                               "num_hidden_layers": config["model"]["num_hidden_layers"] + config["growth"]["add_layers"]["num_new_layers"],
                               "num_attention_heads": config["growth"]["scale_hidden_dim"]["new_hidden_size"] // 64,
                               "num_key_value_heads": (config["growth"]["scale_hidden_dim"]["new_hidden_size"] // 64) // 3,
                               "num_experts": config["growth"]["dense_to_moe"]["num_experts"] + config["growth"]["add_experts"]["num_new_experts"],
                               "num_experts_per_tok": config["growth"]["dense_to_moe"]["num_experts_per_tok"]}
                    model_config = MoEConfig(**moe_cfg)
                    model = SmolLM2MoE(model_config)
                
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"✓ Loaded MoE model from step {total_steps}")
                print(f"  Parameters: {count_parameters(model):,}")
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("Available checkpoints:")
            for f in os.listdir(config["training"]["save_dir"]):
                if f.endswith(".pt"):
                    print(f"  - {f}")
            raise FileNotFoundError(f"Cannot resume: {checkpoint_path} not found")
        
        # Skip to the appropriate phase
        if resume_phase == 1:
            print("Starting fresh from Phase 1...")
        elif resume_phase > 1:
            # Set dummy results for skipped phases
            for i in range(1, resume_phase):
                results[f"phase{i}_loss"] = 0.0
                if i > 1:
                    results[f"phase{i}_delta"] = 0.0
            print(f"Skipping phases 1-{resume_phase-1}, starting at Phase {resume_phase}\n")
    
    # =========================================================================
    # CREATE DATALOADER (with skip_samples if resuming)
    # =========================================================================
    if skip_samples > 0:
        print(f"\n📊 Creating dataloader with {skip_samples:,} samples to skip...")
    
    dataloader, dataset = get_dataloader(
        dataset_name=config["data"]["dataset_name"],
        batch_size=config["training"]["batch_size"],
        max_length=config["training"]["max_length"],
        vocab_size=config["model"]["vocab_size"],
        num_samples=config["data"].get("num_samples", 10000),
        skip_samples=skip_samples,
        return_dataset=True,
    )
    
    # =========================================================================
    # PHASE 1: Dense Model Training
    # =========================================================================
    if resume_phase <= 1:
        print("\n" + "=" * 70)
        print("📌 PHASE 1: Dense Model Training")
        print("=" * 70)
        
        # Check if we're resuming mid-phase (model already loaded)
        if model is None:
            model_config = SmolLM2Config(**config["model"])
            model = SmolLM2(model_config)
            print(f"✓ Created dense model: {count_parameters(model):,} parameters")
            start_step = 0
        else:
            print(f"✓ Resuming with loaded dense model: {count_parameters(model):,} parameters")
            start_step = total_steps
        
        phase1_steps = config["training"]["phase1_steps"]
        remaining_steps = phase1_steps - start_step
        
        if remaining_steps > 0:
            print(f"\n🚀 Training for {remaining_steps} steps (from step {start_step} to {phase1_steps})...")
            
            phase1_loss = train_phase(
                model=model,
                dataloader=dataloader,
                num_steps=remaining_steps,
                start_step=start_step,
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
            results["phase1_loss"] = phase1_loss
        else:
            print(f"\n⏭️  Phase 1 already complete at step {start_step}")
            phase1_loss = 0.0  # Placeholder
        
        total_steps = phase1_steps  # Phase 1 ends at phase1_steps
        print(f"\n✅ Phase 1 complete! Loss: {phase1_loss:.4f}")
    
    # =========================================================================
    # PHASE 2: Dense → MoE
    # =========================================================================
    if resume_phase <= 2:
        print("\n" + "=" * 70)
        print("📌 PHASE 2: Dense → MoE")
        print("=" * 70)
        
        from src.moe_model import SmolLM2MoE
        
        # Check if model is already MoE (mid-phase resume)
        if isinstance(model, SmolLM2MoE):
            print("✓ Model is already MoE (mid-phase resume)")
            # Calculate remaining steps
            phase2_end_step = config["training"]["phase1_steps"] + config["training"]["phase2_steps"]
            remaining_steps = phase2_end_step - total_steps
            start_step = total_steps
        else:
            # Convert dense to MoE
            pre_moe_loss = measure_loss(model, dataloader, device)
            
            moe_config = config["growth"]["dense_to_moe"]
            model = dense_to_moe(
                model,
                num_experts=moe_config["num_experts"],
                num_experts_per_tok=moe_config["num_experts_per_tok"],
            )
            
            post_moe_loss = measure_loss(model, dataloader, device)
            results["phase2_delta"] = log_transition("moe_conversion", pre_moe_loss, post_moe_loss, wandb_run, total_steps)
            
            remaining_steps = config["training"]["phase2_steps"]
            start_step = total_steps
        
        if remaining_steps > 0:
            print(f"\n🚀 Training MoE for {remaining_steps} steps (from step {start_step})...")
            
            phase2_loss = train_phase(
                model=model,
                dataloader=dataloader,
                num_steps=remaining_steps,
                start_step=start_step,
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
            results["phase2_loss"] = phase2_loss
        else:
            print("\n⏭️  Phase 2 already complete")
            phase2_loss = 0.0
        
        total_steps = config["training"]["phase1_steps"] + config["training"]["phase2_steps"]
        print(f"\n✅ Phase 2 complete! Loss: {phase2_loss:.4f}")
        
        # Clear memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # =========================================================================
    # PHASE 3: Add Ghost Layers + Scale Hidden Dimension
    # =========================================================================
    if resume_phase <= 3:
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
    if resume_phase <= 4:
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
    # PHASE 5: YaRN Context Extension ("Sandwich Protocol")
    # =========================================================================
    print("\n" + "=" * 70)
    print("📌 PHASE 5: YaRN Context Extension")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STEP 1: "Do No Harm" Check - Measure loss on SHORT context before/after YaRN
    # This proves YaRN didn't break the model's existing brain
    # -------------------------------------------------------------------------
    print("\n📋 Step 1: 'Do No Harm' Check (short context)")
    pre_yarn_short_loss = measure_loss(model, dataloader, device)
    print(f"  Pre-YaRN loss (256 context): {pre_yarn_short_loss:.4f}")
    
    # Apply YaRN context scaling
    context_config = config["growth"]["scale_context"]
    new_max_length = context_config["new_max_length"]
    
    print(f"\n🔧 Applying YaRN ({model.config.max_position_embeddings} → {new_max_length})...")
    model = scale_context_length(
        model,
        new_max_length=new_max_length,
        alpha=context_config.get("alpha", 1.0),
        beta=context_config.get("beta", 32.0),
    )
    
    post_yarn_short_loss = measure_loss(model, dataloader, device)
    print(f"  Post-YaRN loss (256 context): {post_yarn_short_loss:.4f}")
    
    yarn_delta = post_yarn_short_loss - pre_yarn_short_loss
    yarn_status = "✅ PRESERVED!" if abs(yarn_delta) < 0.5 else "⚠️ Changed"
    print(f"  YaRN impact: {yarn_delta:+.4f} ({yarn_status})")
    results["phase5_delta"] = yarn_delta
    
    # -------------------------------------------------------------------------
    # STEP 2: Training Loop - Train on LONG context to learn new positions
    # -------------------------------------------------------------------------
    print("\n📋 Step 2: Training on Long Context")
    
    # Create dataloader with longer sequences (reduced batch for memory)
    phase5_batch_size = max(1, config["training"]["batch_size"] // 4)
    print(f"  Creating dataloader: max_length={new_max_length}, batch_size={phase5_batch_size}")
    long_dataloader = get_dataloader(
        dataset_name=config["data"]["dataset_name"],
        batch_size=phase5_batch_size,
        max_length=new_max_length,
        num_samples=config["data"]["num_samples"],
    )
    
    # Measure starting loss on LONG context (expected to be higher initially)
    initial_long_loss = measure_loss(model, long_dataloader, device)
    print(f"  Initial long context loss: {initial_long_loss:.4f} (expected higher)")
    
    phase5_steps = config["training"]["phase5_steps"]
    print(f"\n🚀 Training for {phase5_steps} steps on long context...")
    
    phase5_grad_accum = config["training"]["gradient_accumulation_steps"] * 4
    
    phase5_loss = train_phase(
        model=model,
        dataloader=long_dataloader,
        num_steps=phase5_steps,
        start_step=total_steps,
        learning_rate=config["training"]["learning_rate"] * 0.1,
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=20,
        max_grad_norm=config["training"]["max_grad_norm"],
        gradient_accumulation_steps=phase5_grad_accum,
        log_every=config["training"]["log_every"],
        checkpoint_every=config["training"]["checkpoint_every"],
        save_dir=config["training"]["save_dir"],
        checkpoint_prefix="phase5_yarn",
        device=device,
        wandb_run=wandb_run,
    )
    
    total_steps += phase5_steps
    
    # -------------------------------------------------------------------------
    # STEP 3: "Capability" Check - Prove the model can now handle long context
    # -------------------------------------------------------------------------
    print("\n📋 Step 3: 'Capability' Check (long context)")
    final_long_loss = measure_loss(model, long_dataloader, device)
    print(f"  Final long context loss: {final_long_loss:.4f}")
    
    capability_gain = initial_long_loss - final_long_loss
    capability_status = "✅ LEARNED!" if capability_gain > 0.5 else "⚠️ Needs more training"
    print(f"  Capability gain: {capability_gain:+.4f} ({capability_status})")
    
    results["phase5_loss"] = phase5_loss
    results["phase5_initial_long_loss"] = initial_long_loss
    results["phase5_final_long_loss"] = final_long_loss
    results["phase5_capability_gain"] = capability_gain
    
    print("\n✅ Phase 5 complete!")
    print(f"  Short context preserved: {yarn_delta:+.4f}")
    print(f"  Long context capability: {initial_long_loss:.4f} → {final_long_loss:.4f} ({capability_gain:+.4f})")
    
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
    parser.add_argument("--resume", action="store_true", 
                        help="Auto-resume from latest checkpoint")
    parser.add_argument("--resume-phase", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                        help="Resume from specific phase N (overrides --resume). 0=start fresh")
    args = parser.parse_args()
    
    # Handle auto-resume
    resume_phase = args.resume_phase
    resume_checkpoint_path = None  # Will be set if resuming mid-phase
    
    if args.resume and resume_phase == 0:
        # Load config to get save_dir
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        save_dir = config["training"].get("save_dir", "./checkpoints")
        
        latest_ckpt, detected_phase, detected_step, phase_complete = find_latest_checkpoint(save_dir, config)
        
        if detected_phase > 0:
            if phase_complete:
                # Phase is complete, start next phase
                resume_phase = detected_phase + 1
                print(f"\n🔄 Auto-resume: Phase {detected_phase} complete, will start Phase {resume_phase}")
            else:
                # Mid-phase checkpoint - resume same phase from this checkpoint
                resume_phase = detected_phase
                resume_checkpoint_path = latest_ckpt  # Pass the actual checkpoint path
                print(f"\n🔄 Auto-resume: Will continue Phase {resume_phase} from step {detected_step}")
        else:
            print("\n📝 No checkpoints found. Starting fresh.")
    
    results = run_experiment(args.config, args.wandb, resume_phase, resume_checkpoint_path)
