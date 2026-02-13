#!/usr/bin/env python3
"""
SmolLM Training - Main Entry Point

Fourier vs Standard Embeddings Comparison with SmolLM Architecture

Key Features:
1. Command-line argument: "fourier" or "baseline" to choose embedding type
2. SYNTH dataset with Qwen chat format
3. Pause/resume functionality
4. Generation after specified intervals
5. Checkpoints at regular intervals
6. Unified logging to single file

Usage:
    python main.py fourier                # Train with Fourier embeddings
    python main.py baseline               # Train with baseline embeddings
    python main.py fourier --checkpoint path/to/checkpoint.pt  # Resume from checkpoint
"""

import os
# MPS memory management - critical for Mac M1
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.9"
os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"

import argparse
import time
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

# Import our modular components
from config import Config

# MODEL SELECTION: Uncomment ONE of the following imports
# 1. Multi-Token Prediction (NTP + MTP) with Gated Sparse Attention
# from model_gated_multitoken import SmolLM
# 2. Single-Token Prediction with Gated Sparse Attention
# from model_gated import SmolLM
# 3. Single-Token Prediction with Multi-Head Latent Attention (original)
from model import SmolLM

from data import (
    SYNTHStream,
    SYNTHPromptSampler,
    discover_chars_from_bpe_tokenizer,
    pad_char_vocab_128,
    create_bpe_token_strings,
)
from training import (
    setup_training,
    get_learning_rate,
    update_learning_rate,
    save_checkpoint,
    load_checkpoint,
    setup_tokenizer,
)
from utils import (
    setup_logging,
    get_logger,
    PauseHandler,
    sync_device,
    set_runtime_optimizations,
    sample_generate_single_fast,
)
from fourier_se_decoder import PFConfig, PFCodec


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Train SmolLM with Fourier or Baseline embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start fresh training with Fourier embeddings
  python main.py fourier

  # Start fresh training with Baseline embeddings
  python main.py baseline

  # Resume from a specific checkpoint (bypasses pause state)
  python main.py fourier --checkpoint checkpoints/fourier_checkpoint_step_0006000.pt

  # Resume from latest checkpoint (default behavior)
  python main.py fourier
        """
    )
    parser.add_argument(
        "embedding_type",
        choices=["fourier", "baseline"],
        help="Type of embedding to use: 'fourier' or 'baseline'"
    )
    parser.add_argument(
        "--checkpoint",
        "--load-checkpoint",
        dest="checkpoint_path",
        type=str,
        default=None,
        help="Path to a specific checkpoint file to load"
    )

    return parser.parse_args()


def setup_fourier_codec(config, tokenizer):
    """Setup Fourier codec for Fourier embeddings"""
    # Character discovery from BPE tokenizer
    chars, char_to_id = discover_chars_from_bpe_tokenizer(
        tokenizer, config.model.vocab_size
    )
    chars128, char_to_id = pad_char_vocab_128(chars)

    # Create PF codec
    print("🔧 Setting up Fourier codec...")
    pf_cfg = PFConfig(
        # char_vocab=chars128,
        # char_to_id=char_to_id,
        CHAR_DIM=config.fourier.char_dim,
        POS_DIM=config.fourier.pos_dim,
        D=config.fourier.D,
        length_normalize=config.fourier.length_normalize,
        truncate_long_words=config.fourier.truncate_long_words
    )
    pf_codec = PFCodec(pf_cfg)

    # Convert BPE tokens to strings
    bpe_vocab = create_bpe_token_strings(tokenizer, config.model.vocab_size)

    return pf_codec, bpe_vocab


def create_model(config, pf_codec=None, bpe_vocab=None):
    """Create SmolLM model with specified configuration"""
    print(f"🤖 Creating SmolLM with {config.model.embedding_type} embeddings...")
    model = SmolLM(
        vocab_size=config.model.vocab_size,
        embedding_type=config.model.embedding_type,
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
        hidden_size=config.model.hidden_size,
        num_hidden_layers=config.model.num_hidden_layers,
        num_heads=config.model.num_heads,
        intermediate_size=config.model.intermediate_size,
        max_seq_len=config.model.max_seq_len,
        compression_ratio=config.model.compression_ratio,
        num_experts=config.model.num_experts,
        num_shared_experts=config.model.num_shared_experts,
        top_k=config.model.top_k_experts,
        K=config.model.K
    )
    return model


def run_generation(model, tokenizer, prompt_sampler, device, step, config, logger):
    """Run generation evaluation"""
    model.eval()
    try:
        # Sample prompts (deterministic based on step)
        prompts_t = prompt_sampler.sample_token_ids(
            n=config.training.gen_num_prompts,
            step=step
        )
        prompts_t = [p.to(device) for p in prompts_t]

        if prompts_t:
            logger.info(f"\n{'='*20} GENERATION STEP {step} {'='*20}")

            # Generate from each prompt
            for gi, prompt_t in enumerate(prompts_t):
                gen_t = sample_generate_single_fast(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_t,
                    max_new_tokens=config.training.gen_max_new_tokens,
                    temperature=config.training.gen_temperature,
                    top_p=config.training.gen_top_p,
                    max_seq_len=config.model.max_seq_len,
                )

                # Log prompt and generation
                prompt_text = tokenizer.decode(prompt_t, skip_special_tokens=False)
                new_tokens = gen_t[prompt_t.size(0):]
                new_text = tokenizer.decode(new_tokens, skip_special_tokens=False) if new_tokens.numel() > 0 else ""

                logger.info(f"[GEN {gi+1}]")
                logger.info(f"  Prompt ({prompt_t.size(0)} tokens):")
                prompt_display = prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text
                for line in prompt_display.split('\n'):
                    logger.info(f"    {line}")

                logger.info(f"  Output ({new_tokens.numel()} tokens):")
                output_display = new_text[:1000] + "..." if len(new_text) > 1000 else new_text
                for line in output_display.split('\n'):
                    logger.info(f"    {line}")
                logger.info("")  # Blank line

            logger.info(f"{'='*60}\n")
            del prompts_t

        # Cleanup
        gc.collect()
        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except:
                pass
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        model.train()


def training_loop(model, train_loader, prompt_sampler, device, optimizer,
                 criterion, config, start_step, pause_handler, logger):
    """Main training loop - compatible with single-token and multi-token models"""
    print("🔥 Starting training...")
    model.train()

    step = start_step

    # Detect model type
    is_multitoken = hasattr(model, 'mtp_block') and model.mtp_block is not None

    if is_multitoken:
        logger.info("[MODEL] Multi-Token Prediction model detected (NTP + MTP)")
        running_ntp_loss = 0.0
        running_mtp_loss = 0.0
        running_aux_loss = 0.0
    else:
        logger.info("[MODEL] Single-Token Prediction model detected")
        running_task_loss = 0.0
        running_aux_loss = 0.0

    data_iter = iter(train_loader)
    logger.info(f"[START] Training loop begin at step {step}")

    while step < config.training.total_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        del batch

        # ---- Learning rate schedule (BEFORE forward/backward) ----
        lr = get_learning_rate(step, config)
        update_learning_rate(optimizer, lr)

        # ---- Prepare input/target based on model type ----
        if is_multitoken:
            # Multi-token: need x_input, y_ntp (t+1), y_mtp (t+2)
            # input_ids: [B, seq_len] -> x_input: [B, seq_len-2], y_ntp: [B, seq_len-2], y_mtp: [B, seq_len-2]
            x_input = input_ids[:, :-2].contiguous()
            y_ntp = input_ids[:, 1:-1].contiguous()
            y_mtp = input_ids[:, 2:].contiguous()
            del input_ids
        else:
            # Single-token: standard next-token prediction
            x = input_ids[:, :-1].contiguous()
            y = input_ids[:, 1:].contiguous()
            del input_ids

        # Timing start
        sync_device(device)
        t0 = time.time()

        # Use AMP for MPS
        use_amp = (device.type == "mps")
        amp_dtype = config.training.amp_dtype

        # Forward + loss
        log_emb_stats = (step % 10 == 0)
        emb_stats = None

        if is_multitoken:
            # Multi-token model forward pass
            ntp_loss_t = torch.zeros((), device=device)
            mtp_loss_t = torch.zeros((), device=device)
            aux_loss_t = torch.zeros((), device=device)

            if use_amp:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    if log_emb_stats:
                        logits_ntp, logits_mtp, aux_loss, emb_stats = model(
                            x_input, next_token_ids=y_ntp, return_emb_stats=True
                        )
                    else:
                        logits_ntp, logits_mtp, aux_loss = model(
                            x_input, next_token_ids=y_ntp, return_loss=True
                        )

                    # Compute separate losses for t+1 and t+2
                    loss_ntp = criterion(logits_ntp.view(-1, config.model.vocab_size), y_ntp.view(-1))
                    loss_mtp = criterion(logits_mtp.view(-1, config.model.vocab_size), y_mtp.view(-1))
                    # Weight MTP loss by 0.3 (matches original: 1.0 for NTP primary, 0.3 for MTP auxiliary)
                    loss = loss_ntp + 0.3 * loss_mtp + aux_loss
            else:
                if log_emb_stats:
                    logits_ntp, logits_mtp, aux_loss, emb_stats = model(
                        x_input, next_token_ids=y_ntp, return_emb_stats=True
                    )
                else:
                    logits_ntp, logits_mtp, aux_loss = model(
                        x_input, next_token_ids=y_ntp, return_loss=True
                    )

                # Compute separate losses for t+1 and t+2
                loss_ntp = criterion(logits_ntp.view(-1, config.model.vocab_size), y_ntp.view(-1))
                loss_mtp = criterion(logits_mtp.view(-1, config.model.vocab_size), y_mtp.view(-1))
                # Weight MTP loss by 0.3 (matches original: 1.0 for NTP primary, 0.3 for MTP auxiliary)
                loss = loss_ntp + 0.3 * loss_mtp + aux_loss

            # Backward pass
            loss.backward()
            ntp_loss_t = ntp_loss_t + loss_ntp.detach()
            mtp_loss_t = mtp_loss_t + loss_mtp.detach()
            aux_loss_t = aux_loss_t + aux_loss.detach()
            del loss

        else:
            # Single-token model forward pass
            task_loss_t = torch.zeros((), device=device)
            aux_loss_t = torch.zeros((), device=device)

            if use_amp:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    if log_emb_stats:
                        model_output = model(x, return_emb_stats=True)
                        # Handle both single logits and (logits, aux_loss, emb_stats)
                        if isinstance(model_output, tuple) and len(model_output) >= 3:
                            logits, aux_loss, emb_stats = model_output[0], model_output[-2], model_output[-1]
                        else:
                            logits = model_output if not isinstance(model_output, tuple) else model_output[0]
                            aux_loss = torch.zeros((), device=device)
                    else:
                        model_output = model(x, return_loss=True)
                        # Handle both single logits and (logits, aux_loss)
                        if isinstance(model_output, tuple) and len(model_output) >= 2:
                            logits, aux_loss = model_output[0], model_output[1]
                        else:
                            logits = model_output
                            aux_loss = torch.zeros((), device=device)

                    task_loss = criterion(logits.view(-1, config.model.vocab_size), y.view(-1))
                    loss = task_loss + aux_loss
            else:
                if log_emb_stats:
                    model_output = model(x, return_emb_stats=True)
                    if isinstance(model_output, tuple) and len(model_output) >= 3:
                        logits, aux_loss, emb_stats = model_output[0], model_output[-2], model_output[-1]
                    else:
                        logits = model_output if not isinstance(model_output, tuple) else model_output[0]
                        aux_loss = torch.zeros((), device=device)
                else:
                    model_output = model(x, return_loss=True)
                    if isinstance(model_output, tuple) and len(model_output) >= 2:
                        logits, aux_loss = model_output[0], model_output[1]
                    else:
                        logits = model_output
                        aux_loss = torch.zeros((), device=device)

                task_loss = criterion(logits.view(-1, config.model.vocab_size), y.view(-1))
                loss = task_loss + aux_loss

            # Backward pass
            loss.backward()
            task_loss_t = task_loss_t + task_loss.detach()
            aux_loss_t = aux_loss_t + aux_loss.detach()
            del loss

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.max_grad_norm)

        # Optimizer step
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Timing end
        sync_device(device)
        dt = (time.time() - t0) * 1000.0  # ms

        # Convert to python scalars and calculate tok/sec BEFORE cleanup
        if is_multitoken:
            step_ntp_loss = float(ntp_loss_t.cpu().item())
            step_mtp_loss = float(mtp_loss_t.cpu().item())
            step_aux_loss = float(aux_loss_t.cpu().item())
            tok_sec = x_input.numel() / max(dt / 1000.0, 1e-9)
        else:
            step_task_loss = float(task_loss_t.cpu().item())
            step_aux_loss = float(aux_loss_t.cpu().item())
            tok_sec = x.numel() / max(dt / 1000.0, 1e-9)
        cur_lr = optimizer.param_groups[0]["lr"]

        # Cleanup logits and tensors
        if is_multitoken:
            del logits_ntp, logits_mtp, y_ntp, y_mtp, x_input
        else:
            del logits, x, y

        # Console log
        if step % config.training.log_interval == 0:
            if is_multitoken:
                # Multi-token: log "loss" (t+1) and "loss2" (t+2) separately
                avg_ntp = running_ntp_loss / config.training.log_interval if running_ntp_loss > 0 else step_ntp_loss
                avg_mtp = running_mtp_loss / config.training.log_interval if running_mtp_loss > 0 else step_mtp_loss
                avg_aux = running_aux_loss / config.training.log_interval if running_aux_loss > 0 else step_aux_loss
                logger.info(f"step {step} | loss: {avg_ntp:.4f} | loss2: {avg_mtp:.4f} | r_loss: {avg_aux:.4f} | lr: {cur_lr:.2e} | dt: {dt:7.2f}ms | tok/sec: {tok_sec:9.2f}")
                running_ntp_loss = 0.0
                running_mtp_loss = 0.0
                running_aux_loss = 0.0
            else:
                # Single-token: log "loss" only
                avg_task = running_task_loss / config.training.log_interval if running_task_loss > 0 else step_task_loss
                avg_aux = running_aux_loss / config.training.log_interval if running_aux_loss > 0 else step_aux_loss
                logger.info(f"step {step} | loss: {avg_task:.4f} | r_loss: {avg_aux:.4f} | lr: {cur_lr:.2e} | dt: {dt:7.2f}ms | tok/sec: {tok_sec:9.2f}")
                running_task_loss = 0.0
                running_aux_loss = 0.0
        else:
            if is_multitoken:
                running_ntp_loss += step_ntp_loss
                running_mtp_loss += step_mtp_loss
                running_aux_loss += step_aux_loss
            else:
                running_task_loss += step_task_loss
                running_aux_loss += step_aux_loss

        # Log embedding stats
        if emb_stats is not None:
            logger.info(f"[EMB_STATS] step {step} | emb_std: {emb_stats['emb_std']:.4f} | emb_mean: {emb_stats['emb_mean']:.4f} | proj_std: {emb_stats['proj_std']:.4f} | proj_mean: {emb_stats['proj_mean']:.4f}")

        # Log lambda_e
        if step % 10 == 0:
            if hasattr(model, "use_fourier") and model.use_fourier:
                lambda_e_val = model.lambda_e().item()
                logger.info(f"[step {step}] lambda_e={lambda_e_val:.4f}")

        # Log GSA stats (Gated Sparse Attention)
        if step % 10 == 0:
            gsa_stats = {"k": [], "var": [], "gv": [], "go": []}
            for layer in model.layers:
                if hasattr(layer, 'attn_block') and hasattr(layer.attn_block, 'sublayer'):
                    attn_sublayer = layer.attn_block.sublayer
                    if hasattr(attn_sublayer, 'last_stats'):
                        stats = attn_sublayer.last_stats
                        gsa_stats["k"].append(stats["gsa/k_avg"])
                        gsa_stats["var"].append(stats["gsa/var_score"])
                        gsa_stats["gv"].append(stats["gsa/gate_v"])
                        gsa_stats["go"].append(stats["gsa/gate_o"])

            if gsa_stats["k"]:
                k_avg = sum(gsa_stats['k']) / len(gsa_stats['k'])
                var_avg = sum(gsa_stats['var']) / len(gsa_stats['var'])
                gv_avg = sum(gsa_stats['gv']) / len(gsa_stats['gv'])
                go_avg = sum(gsa_stats['go']) / len(gsa_stats['go'])
                logger.info(
                    f"[GSA] step {step} | k: {k_avg:.1f} | var: {var_avg:.4f} | "
                    f"gate_v: {gv_avg:.2f} | gate_o: {go_avg:.2f}"
                )

        # Log null expert usage
        if step % 10 == 0:
            first_layer = model.layers[0]
            if hasattr(first_layer, 'mlp_block') and hasattr(first_layer.mlp_block.sublayer, 'moe'):
                moe = first_layer.mlp_block.sublayer.moe
                if moe.last_indices is not None and hasattr(moe.gate, 'num_experts'):
                    indices_flat = moe.last_indices.view(-1)
                    total_slots = moe.gate.total_slots
                    counts = torch.bincount(indices_flat, minlength=total_slots).float()
                    total_assignments = counts.sum()

                    null_assignments = counts[moe.gate.num_experts:].sum()
                    null_pct = (null_assignments / total_assignments * 100).item() if total_assignments > 0 else 0.0

                    logger.info(f"[NULL EXPERTS] step {step} | null usage: {null_pct:.1f}% | "
                              f"target: {(1-moe.gate.data_sparsity)*100:.0f}%")

        # Memory cleanup
        if config.training.mem_every > 0 and (step % config.training.mem_every == 0):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        # Generation
        if config.training.gen_every > 0 and step >= config.training.gen_warmup_steps and (step % config.training.gen_every == 0):
            run_generation(model, prompt_sampler.tokenizer, prompt_sampler,
                         device, step, config, logger)

        # Periodic garbage collection
        if config.training.gc_every > 0 and (step % config.training.gc_every == 0):
            gc.collect()
            if device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        # Checkpoint
        if config.training.ckpt_every > 0 and (step % config.training.ckpt_every == 0) and step > 0:
            if is_multitoken:
                ckpt_loss = step_ntp_loss + 0.3 * step_mtp_loss + step_aux_loss
            else:
                ckpt_loss = step_task_loss + step_aux_loss
            save_checkpoint(model, optimizer, None, step, ckpt_loss,
                          config.model.embedding_type, config.training.save_dir)
            logger.info(f"[CKPT] saved checkpoint at step {step}")

        step += 1

        # Check for pause
        if pause_handler.should_pause():
            logger.info(f"[PAUSE] Pausing at step {step}...")

            if pause_handler.should_save or pause_handler.check_pause_flag():
                if is_multitoken:
                    pause_loss = (step_ntp_loss + 0.3 * step_mtp_loss + step_aux_loss) if 'step_ntp_loss' in locals() else 0.0
                else:
                    pause_loss = (step_task_loss + step_aux_loss) if 'step_task_loss' in locals() else 0.0
                save_checkpoint(model, optimizer, None, step, pause_loss,
                              config.model.embedding_type, config.training.save_dir)
                logger.info(f"[PAUSE] Checkpoint saved at step {step}")
                pause_handler.should_save = False

            logger.info(f"[PAUSE] Training paused at step {step}.")
            logger.info(f"[PAUSE] To resume: remove 'checkpoints/.pause' file or restart training")
            logger.info(f"[PAUSE] To exit: Press Ctrl+C again")

            pause_handler.in_pause_wait = True

            try:
                while pause_handler.should_pause():
                    time.sleep(1.0)
                    if not pause_handler.check_pause_flag() and not pause_handler.paused:
                        break
            except KeyboardInterrupt:
                logger.info(f"[EXIT] Exiting at step {step}")
                raise

            pause_handler.in_pause_wait = False
            pause_handler.resume()
            logger.info(f"[RESUME] Resuming training from step {step}...")

    # Final checkpoint
    if is_multitoken:
        final_loss = (step_ntp_loss + 0.3 * step_mtp_loss + step_aux_loss) if 'step_ntp_loss' in locals() else 0.0
    else:
        final_loss = (step_task_loss + step_aux_loss) if 'step_task_loss' in locals() else 0.0
    save_checkpoint(model, optimizer, None, step, final_loss,
                   config.model.embedding_type, config.training.save_dir)
    logger.info("[DONE] Training complete")
    print("🏁 Training completed!")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()

    # Create configuration
    config = Config.from_args(args.embedding_type, args.checkpoint_path)

    print(f"🚀 SMOLLM TRAINING - {config.model.embedding_type.upper()} Embeddings")
    print("=" * 80)

    # Load tokenizer
    print("📚 Loading GPT2 tokenizer with Qwen special tokens...")
    tokenizer = GPT2Tokenizer.from_pretrained(config.data.tokenizer_name)
    tokenizer = setup_tokenizer(tokenizer, config.model.vocab_size)

    # Setup Fourier embeddings if needed
    pf_codec = None
    bpe_vocab = None

    if config.model.embedding_type == "fourier":
        pf_codec, bpe_vocab = setup_fourier_codec(config, tokenizer)

    # Create model
    model = create_model(config, pf_codec, bpe_vocab)

    # Setup training
    device_type = config.auto_detect_device()
    device, optimizer, lr_scheduler, scaler = setup_training(model, device_type=device_type)

    # Set runtime optimizations
    set_runtime_optimizations(device)

    # Setup logging
    logger = setup_logging(config.get_log_filename())

    # Try to resume from checkpoint
    print("🔍 Checking for existing checkpoint...")
    start_step, start_loss = load_checkpoint(
        model, optimizer, lr_scheduler, config.model.embedding_type,
        save_dir=config.training.save_dir,
        checkpoint_path=config.checkpoint_path
    )

    if start_step > 0:
        logger.info(f"[RESUME] Resuming from step {start_step} (loss: {start_loss:.4f})")
        if config.checkpoint_path:
            logger.info(f"[RESUME] Loaded from specified checkpoint: {config.checkpoint_path}")
    else:
        logger.info("[START] Starting fresh training")

    # Load SYNTH dataset
    print("📊 Loading SYNTH dataset...")
    print(f"📊 Dataset will start from step {start_step}")
    dataset = SYNTHStream(
        tokenizer=tokenizer,
        dataset_name=config.data.dataset_name,
        seq_len=config.training.seq_len,
        batch_size=config.training.batch_size,
        shuffle_buffer=config.data.shuffle_buffer,
        seed=config.data.seed,
        include_query=config.data.include_query,
        include_reasoning=config.data.include_reasoning,
        include_answer=config.data.include_answer,
        combine_separator=config.data.combine_separator,
        filter_language=config.data.filter_language,
        start_step=start_step
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Prompt sampler for generation
    prompt_sampler = SYNTHPromptSampler(
        dataset_name=config.data.dataset_name,
        tokenizer=tokenizer,
        seed=config.data.seed
    )

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Pause handler
    pause_handler = PauseHandler(pause_flag_path=f"{config.training.save_dir}/.pause")
    pause_handler.clear_pause_flag()
    logger.info("[PAUSE] Pause/resume enabled. Press Ctrl+C or create '.pause' file to pause.")

    # Run training loop
    training_loop(
        model, train_loader, prompt_sampler, device, optimizer,
        criterion, config, start_step, pause_handler, logger
    )


if __name__ == "__main__":
    main()
