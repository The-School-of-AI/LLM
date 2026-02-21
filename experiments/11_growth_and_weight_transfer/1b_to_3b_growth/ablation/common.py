"""
Shared utilities for Dense-to-MoE ablation experiments.

Centralizes tokenizer loading, model creation, data loading, and logging
to avoid copy-paste across experiment scripts.
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

# Add parent directory (endGame/) to path for imports
ENDGAME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ENDGAME_DIR not in sys.path:
    sys.path.insert(0, ENDGAME_DIR)

from recurrence_model_1b import (
    ModelConfig as Config1B, Model1B,
    KroneckerEmbeddings, KroneckerConfig,
)
from recurrence_model_3b import ModelConfig as Config3B, Model3B
from data_utils import SYNTHStream
from training import save_checkpoint, load_checkpoint, set_moe_freeze_state

# MPS memory management
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.9"
os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"


# ============================================================================
# Device
# ============================================================================

def detect_device():
    """MPS > CUDA > CPU detection."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"Device: CPU")
    return device


# ============================================================================
# Tokenizer & Embeddings
# ============================================================================

def load_tokenizer():
    """Load tokenizer from tokenizer.json in endGame directory."""
    tokenizer_path = os.path.join(ENDGAME_DIR, "tokenizer.json")
    print(f"Loading tokenizer from {tokenizer_path}...")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"  Loaded: {vocab_size:,} tokens")

    bpe_vocab = []
    for token_id in range(vocab_size):
        try:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            bpe_vocab.append(token_text)
        except Exception:
            bpe_vocab.append(f"<TOKEN_{token_id}>")

    return tokenizer, bpe_vocab


def create_kronecker_codec(vocab_size):
    """Create Kronecker embedding codec."""
    pf_cfg = KroneckerConfig(
        CHAR_DIM=256,
        POS_DIM=32,
        D=8192,
        length_normalize=True,
        truncate_long_words=True,
    )
    pf_codec = KroneckerEmbeddings(pf_cfg)
    print(f"  Kronecker: {pf_cfg.CHAR_DIM}x{pf_cfg.POS_DIM} = {pf_cfg.D} dims")
    return pf_codec


# ============================================================================
# Model Creation
# ============================================================================

def create_1b_model(device, bpe_vocab, pf_codec):
    """Create 1B dense model."""
    config = Config1B()
    model = Model1B(config, embedding_type="kronecker", pf_codec=pf_codec, bpe_vocab=bpe_vocab)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  1B Dense model: {total_params:,} params on {device}")
    return model


def create_3b_model(device, bpe_vocab, pf_codec, config_overrides=None):
    """
    Create 3B MoE model with optional config overrides.

    Args:
        config_overrides: dict of {attr: value} to apply to Config3B before construction.
                          e.g. {"expert_intermediate_size": 1024} for Experiment 3.
    """
    config = Config3B()
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"  Config override: {key} = {value}")
            else:
                print(f"  WARNING: Config has no attribute '{key}', skipping")

    model = Model3B(config, embedding_type="kronecker", pf_codec=pf_codec, bpe_vocab=bpe_vocab)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  3B MoE model: {total_params:,} params on {device}")
    return model, config


# ============================================================================
# Data Loading
# ============================================================================

def create_data_loader(tokenizer, seq_len=64, batch_size=1, start_step=0):
    """Create deterministic data loader."""
    dataset = SYNTHStream(
        tokenizer=tokenizer,
        dataset_name="PleIAs/SYNTH",
        local_path="../synth_local_en",
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle_buffer=1000,
        seed=42,
        include_query=True,
        include_reasoning=True,
        include_answer=True,
        combine_separator="\n\n",
        filter_language="en",
        start_step=start_step,
    )
    return DataLoader(dataset, batch_size=batch_size, drop_last=True)


def get_reference_batch(tokenizer, device, seq_len=64):
    """
    Get a single fixed batch for deterministic eval comparisons.
    Uses seed=42, start_step=0 so all experiments compare against the same data.
    """
    loader = create_data_loader(tokenizer, seq_len=seq_len, batch_size=1, start_step=0)
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)

    # Standard input prep: x=input[:-2], y_ntp=input[1:-1], y_mtp=input[2:]
    x_input = input_ids[:, :-2].contiguous()
    y_ntp = input_ids[:, 1:-1].contiguous()
    y_mtp = input_ids[:, 2:].contiguous()
    return x_input, y_ntp, y_mtp


# ============================================================================
# Logging
# ============================================================================

def setup_logging(log_path):
    """
    Setup dual logging: console + file.
    Returns a logger instance.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(os.path.basename(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def log_header(logger):
    """Log column header for training metrics."""
    logger.info("# step | loss_ntp | loss_mtp | total | aux | lr | grad_norm | tok_sec | dt_ms")


def log_step(logger, step, loss_ntp, loss_mtp, total_loss, aux, lr, grad_norm, tok_sec, dt_ms):
    """Log one training step in standardized format."""
    logger.info(
        f"step={step:04d} | loss_ntp={loss_ntp:.4f} | loss_mtp={loss_mtp:.4f} | "
        f"total={total_loss:.4f} | aux={aux:.4f} | lr={lr:.2e} | "
        f"grad_norm={grad_norm:.4f} | tok_sec={tok_sec:.1f} | dt_ms={dt_ms:.1f}"
    )


def log_step_moe(logger, step, loss_ntp, loss_mtp, total_loss, aux, lr, grad_norm,
                 tok_sec, dt_ms, null_rate):
    """Log one training step with MoE-specific metrics."""
    logger.info(
        f"step={step:04d} | loss_ntp={loss_ntp:.4f} | loss_mtp={loss_mtp:.4f} | "
        f"total={total_loss:.4f} | aux={aux:.4f} | null_rate={null_rate:.4f} | "
        f"lr={lr:.2e} | grad_norm={grad_norm:.4f} | tok_sec={tok_sec:.1f} | dt_ms={dt_ms:.1f}"
    )


# ============================================================================
# MoE Metrics
# ============================================================================

def compute_moe_metrics(model):
    """
    Extract null selection rate from model's last_indices.
    Returns dict with 'null_rate'.
    """
    total_null = 0
    total_tokens = 0

    if hasattr(model, 'layers'):
        for layer in model.layers:
            if (hasattr(layer, 'mlp_block')
                    and hasattr(layer.mlp_block, 'sublayer')
                    and hasattr(layer.mlp_block.sublayer, 'moe')):
                moe_layer = layer.mlp_block.sublayer.moe
                if hasattr(moe_layer, 'last_indices') and moe_layer.last_indices is not None:
                    indices = moe_layer.last_indices
                    num_experts = moe_layer.num_experts
                    is_null = (indices >= num_experts)
                    total_null += is_null.float().sum().item()
                    total_tokens += is_null.numel()

    null_rate = total_null / total_tokens if total_tokens > 0 else 0.0
    return {"null_rate": null_rate}


# ============================================================================
# Loss Computation
# ============================================================================

def prepare_inputs(input_ids):
    """Standard input preparation: x=input[:-2], y_ntp=input[1:-1], y_mtp=input[2:]."""
    x_input = input_ids[:, :-2].contiguous()
    y_ntp = input_ids[:, 1:-1].contiguous()
    y_mtp = input_ids[:, 2:].contiguous()
    return x_input, y_ntp, y_mtp


def compute_losses(logits_ntp, logits_mtp, y_ntp, y_mtp, aux_loss):
    """
    Compute NTP, MTP, and total loss.
    Returns dict with loss_ntp, loss_mtp, total, aux.
    """
    criterion = nn.CrossEntropyLoss()
    V = logits_ntp.size(-1)

    loss_ntp = criterion(logits_ntp.view(-1, V), y_ntp.view(-1))

    loss_mtp = torch.tensor(0.0, device=logits_ntp.device)
    if logits_mtp is not None:
        T_mtp = logits_mtp.size(1)
        y_mtp_use = y_mtp[:, :T_mtp] if y_mtp.size(1) > T_mtp else y_mtp
        logits_mtp_use = logits_mtp[:, :y_mtp_use.size(1), :]
        loss_mtp = criterion(logits_mtp_use.reshape(-1, V), y_mtp_use.reshape(-1))

    aux_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else (aux_loss or 0.0)
    aux_tensor = aux_loss if isinstance(aux_loss, torch.Tensor) else torch.tensor(0.0)

    total = loss_ntp + 0.3 * loss_mtp + aux_tensor

    return {
        "loss_ntp": loss_ntp,
        "loss_mtp": loss_mtp,
        "total": total,
        "aux": aux_val,
    }


# ============================================================================
# Router Bias Utilities
# ============================================================================

def force_null_routing(model, logit_bias=-100.0, null_logit=100.0):
    """
    Force all routers to select null experts by setting extreme biases.
    Applies to both backbone layers and MTP block.
    """
    count = 0
    # Backbone layers
    if hasattr(model, 'layers'):
        for layer in model.layers:
            if (hasattr(layer, 'mlp_block')
                    and hasattr(layer.mlp_block, 'sublayer')
                    and hasattr(layer.mlp_block.sublayer, 'moe')):
                moe = layer.mlp_block.sublayer.moe
                if hasattr(moe, 'gate') and moe.gate is not None:
                    if hasattr(moe.gate, 'logit_bias') and moe.gate.logit_bias is not None:
                        nn.init.constant_(moe.gate.logit_bias, logit_bias)
                    if hasattr(moe.gate, 'null_logit') and moe.gate.null_logit is not None:
                        nn.init.constant_(moe.gate.null_logit, null_logit)
                    count += 1

    # MTP block
    if hasattr(model, 'mtp_block') and model.mtp_block is not None:
        if hasattr(model.mtp_block, 'mlp') and hasattr(model.mtp_block.mlp, 'moe'):
            moe = model.mtp_block.mlp.moe
            if hasattr(moe, 'gate') and moe.gate is not None:
                if hasattr(moe.gate, 'logit_bias') and moe.gate.logit_bias is not None:
                    nn.init.constant_(moe.gate.logit_bias, logit_bias)
                if hasattr(moe.gate, 'null_logit') and moe.gate.null_logit is not None:
                    nn.init.constant_(moe.gate.null_logit, null_logit)
                count += 1

    print(f"  Forced null routing on {count} MoE layers (bias={logit_bias}, null={null_logit})")


def set_active_routing_bias(model, logit_bias=-2.65, null_logit=2.65):
    """
    Set router biases for active but null-biased routing.
    prob_null ~ 0.995 at init with gap=5.3.
    """
    count = 0
    if hasattr(model, 'layers'):
        for layer in model.layers:
            if (hasattr(layer, 'mlp_block')
                    and hasattr(layer.mlp_block, 'sublayer')
                    and hasattr(layer.mlp_block.sublayer, 'moe')):
                moe = layer.mlp_block.sublayer.moe
                if hasattr(moe, 'gate') and moe.gate is not None:
                    if hasattr(moe.gate, 'logit_bias') and moe.gate.logit_bias is not None:
                        nn.init.constant_(moe.gate.logit_bias, logit_bias)
                    if hasattr(moe.gate, 'null_logit') and moe.gate.null_logit is not None:
                        nn.init.constant_(moe.gate.null_logit, null_logit)
                    count += 1

    if hasattr(model, 'mtp_block') and model.mtp_block is not None:
        if hasattr(model.mtp_block, 'mlp') and hasattr(model.mtp_block.mlp, 'moe'):
            moe = model.mtp_block.mlp.moe
            if hasattr(moe, 'gate') and moe.gate is not None:
                if hasattr(moe.gate, 'logit_bias') and moe.gate.logit_bias is not None:
                    nn.init.constant_(moe.gate.logit_bias, logit_bias)
                if hasattr(moe.gate, 'null_logit') and moe.gate.null_logit is not None:
                    nn.init.constant_(moe.gate.null_logit, null_logit)
                count += 1

    print(f"  Active routing bias on {count} MoE layers (bias={logit_bias}, null={null_logit})")


# ============================================================================
# Rotation Utility (for Exp3 custom init)
# ============================================================================

def random_small_rotation(dim, eps=0.005, device="cpu"):
    """Generate a small orthogonal rotation matrix via skew-symmetric matrix exponential."""
    A = torch.randn(dim, dim, device=device)
    A = A - A.T  # skew-symmetric
    R = torch.matrix_exp(eps * A)
    return R