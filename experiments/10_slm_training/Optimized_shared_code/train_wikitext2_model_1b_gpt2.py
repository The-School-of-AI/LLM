"""
WikiText-2 training/test runner for model_1b.py using GPT-2 tokenizer.

This script mirrors the workflow of test_wikitext2_reference_gpt2.py but
targets the local Model70B/ModelConfig classes in model_1b.py.
"""

import argparse
import math
import os
import random
import sys
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "Missing dependencies. Install with: pip install datasets transformers"
    ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preferred == "cuda":
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    if preferred == "mps":
        return (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    return torch.device("cpu")


def get_optimal_num_workers() -> int:
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 2
    return max(2, min(16, int(cpu_count * 0.75)))


def get_amp_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float16":
        return torch.float16
    return torch.bfloat16


def ensure_reversible_ops_midpoint() -> bool:
    """
    Injects a local fallback if reversible_ops_midpoint is unavailable.
    Returns True when fallback is injected.
    """
    try:
        __import__("reversible_ops_midpoint")
        return False
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("reversible_ops_midpoint")

    class ReversibleMidpointStack(torch.nn.Module):
        """
        Fallback: plain sequential execution with the same forward contract.
        """

        def __init__(
            self,
            layers: torch.nn.ModuleList,
            step_size: float = 0.25,
            a: float = 0.5,
            noise_eps: float = 0.0,
            bootstrap: str = "euler",
        ):
            super().__init__()
            self.layers = torch.nn.ModuleList(layers)
            self.step_size = step_size
            self.a = a
            self.noise_eps = noise_eps
            self.bootstrap = bootstrap
            self.use_checkpoint = False

        def forward(self, x_stream: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            total_aux = x_stream.new_zeros((), dtype=torch.float32)
            for layer in self.layers:
                x_stream, aux = layer(x_stream, attention_mask=None)
                if aux is not None and torch.isfinite(aux).all():
                    total_aux = total_aux + aux.float()
            return x_stream, total_aux

    module.ReversibleMidpointStack = ReversibleMidpointStack
    sys.modules["reversible_ops_midpoint"] = module
    return True


class TokenBlockDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_length: int, stride: int):
        self.token_ids = token_ids
        self.seq_length = seq_length
        self.stride = stride
        if len(token_ids) < seq_length:
            self.num_blocks = 0
        else:
            self.num_blocks = (len(token_ids) - seq_length) // stride + 1

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_length
        block = self.token_ids[start:end]
        input_ids = torch.tensor(block, dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def build_token_ids(
    split: str,
    tokenizer,
    add_eos: bool = True,
    max_tokens: Optional[int] = None,
) -> List[int]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    token_ids: List[int] = []
    eos_id = tokenizer.eos_token_id

    for text in dataset["text"]:
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        token_ids.extend(ids)
        if add_eos and eos_id is not None:
            token_ids.append(eos_id)

        if max_tokens is not None and len(token_ids) >= max_tokens:
            token_ids = token_ids[:max_tokens]
            break

    return token_ids


def architecture_checks(config: Any) -> Tuple[List[str], List[str]]:
    ok: List[str] = []
    bad: List[str] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        if actual == expected:
            ok.append(f"{name}: {actual}")
        else:
            bad.append(f"{name}: actual={actual}, expected={expected}")

    check("hidden_size", config.hidden_size, 4096)
    check("num_layers(backbone)", config.num_layers, 8)
    check("num_deltanet_layers", config.num_deltanet_layers, 6)
    check("num_gsa_layers", config.num_gsa_layers, 2)
    check("mtp_layers", config.mtp_num_predictions - 1, 1)
    check("delta_gate_dim", config.delta_gate_dim, 384)
    check("ffn_intermediate_size", config.expert_intermediate_size, 2048)
    check("num_real_experts", config.num_real_experts, 0)
    check("top_k", config.top_k, 0)
    check("delta_v_heads", config.delta_v_heads, 32)
    check("delta_qk_heads", config.delta_qk_heads, 16)
    check("delta_head_dim", config.delta_head_dim, 128)
    check("gsa_num_heads", config.gsa_num_heads, 16)
    check("gsa_head_dim", config.gsa_head_dim, 256)
    return ok, bad


def print_architecture_report(config: Any, strict: bool) -> None:
    ok_lines, bad_lines = architecture_checks(config)
    print("\n" + "=" * 72)
    print("Attached Configuration Check")
    print("=" * 72)
    print(f"Backbone Layers: {config.num_layers}")
    print(f"MTP Layers: {config.mtp_num_predictions - 1}")
    print(
        f"Total Computational Layers: {config.num_layers + config.mtp_num_predictions - 1}"
    )
    print(f"Delta/GSA Split: {config.num_deltanet_layers}/{config.num_gsa_layers}")
    if bad_lines:
        print("\nMISMATCHES:")
        for line in bad_lines:
            print(f"  - {line}")
    else:
        print("\nAll attached config checks matched.")
    if ok_lines:
        print("\nMatched checks:")
        for line in ok_lines:
            print(f"  - {line}")
    print("=" * 72 + "\n")
    if strict and bad_lines:
        raise ValueError("Architecture mismatch with --strict-arch enabled.")


def build_kronecker_artifacts(tokenizer, pf_dim: int):
    from model_1b import KroneckerConfig, KroneckerEmbeddings

    vocab_size = len(tokenizer)
    approx_pf_gb = (vocab_size * pf_dim * 2) / (1024**3)  # bf16 table
    print(
        f"[Kronecker] Building PF table for vocab_size={vocab_size}, D={pf_dim}. "
        f"Approx bf16 buffer size: ~{approx_pf_gb:.2f} GB"
    )
    bpe_vocab = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in range(vocab_size)
    ]
    pf_cfg = KroneckerConfig(D=pf_dim)
    pf_codec = KroneckerEmbeddings(pf_cfg)
    return bpe_vocab, pf_codec


@dataclass
class LossBundle:
    total: torch.Tensor
    main: torch.Tensor
    mtp: torch.Tensor
    aux: torch.Tensor


def compute_losses(
    logits_ntp: torch.Tensor,
    logits_mtp: Optional[torch.Tensor],
    labels: torch.Tensor,
    aux_loss: Optional[torch.Tensor],
    mtp_loss_weight: float,
    aux_loss_weight: float,
    zero_expert_mode: bool,
) -> LossBundle:
    vocab = logits_ntp.size(-1)
    main_loss = F.cross_entropy(
        logits_ntp[:, :-1, :].contiguous().view(-1, vocab),
        labels[:, 1:].contiguous().view(-1),
    )

    mtp_loss = main_loss.new_zeros(())
    if logits_mtp is not None and labels.size(1) > 2 and logits_mtp.size(1) > 1:
        mtp_len = min(logits_mtp.size(1) - 1, labels.size(1) - 2)
        if mtp_len > 0:
            mtp_loss = F.cross_entropy(
                logits_mtp[:, :mtp_len, :].contiguous().view(-1, vocab),
                labels[:, 2 : 2 + mtp_len].contiguous().view(-1),
            )

    aux_term = main_loss.new_zeros(())
    if (
        aux_loss is not None
        and isinstance(aux_loss, torch.Tensor)
        and torch.isfinite(aux_loss).all()
        and not zero_expert_mode
        and aux_loss_weight > 0.0
    ):
        aux_term = aux_loss.float() * aux_loss_weight

    total = main_loss + (mtp_loss_weight * mtp_loss) + aux_term
    return LossBundle(total=total, main=main_loss, mtp=mtp_loss, aux=aux_term)


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_steps: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    gradient_clip: float,
    log_interval: int,
    use_amp: bool,
    amp_dtype_str: str,
    mtp_loss_weight: float,
    aux_loss_weight: float,
    fused_adamw: bool,
    non_blocking_transfer: bool,
) -> None:
    model = model.to(device)
    model.train()

    optimizer_kwargs = dict(
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )
    if fused_adamw and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except TypeError:
        # Compatibility fallback for environments without fused AdamW support.
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    amp_dtype = get_amp_dtype(amp_dtype_str)
    scaler_enabled = use_amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = GradScaler("cuda", enabled=scaler_enabled)

    zero_expert_mode = getattr(model.config, "num_real_experts", 0) == 0
    use_cudagraph_step_marker = (
        device.type == "cuda"
        and getattr(model.config, "use_compile", False)
        and hasattr(torch, "compiler")
        and hasattr(torch.compiler, "cudagraph_mark_step_begin")
    )

    def lr_for_step(step: int) -> float:
        if step < warmup_steps:
            return learning_rate * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + (learning_rate - min_learning_rate) * cosine

    global_step = 0
    tokens_seen = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = 0
    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        accum_total = 0.0
        accum_main = 0.0
        accum_mtp = 0.0
        accum_aux = 0.0
        accum_micro = 0
        last_input_ids: Optional[torch.Tensor] = None

        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(
                device, non_blocking=non_blocking_transfer
            )
            labels = batch["labels"].to(device, non_blocking=non_blocking_transfer)
            next_token_ids = input_ids[:, 1:].contiguous()
            last_input_ids = input_ids

            if use_cudagraph_step_marker:
                torch.compiler.cudagraph_mark_step_begin()

            autocast_enabled = use_amp and (
                (device.type == "cuda")
                or (device.type == "mps" and amp_dtype == torch.float16)
            )
            with autocast(
                device_type=device.type, enabled=autocast_enabled, dtype=amp_dtype
            ):
                logits_ntp, logits_mtp, aux_loss = model(
                    input_ids=input_ids,
                    next_token_ids=next_token_ids,
                    return_loss=True,
                )
                losses = compute_losses(
                    logits_ntp=logits_ntp,
                    logits_mtp=logits_mtp,
                    labels=labels,
                    aux_loss=aux_loss,
                    mtp_loss_weight=mtp_loss_weight,
                    aux_loss_weight=aux_loss_weight,
                    zero_expert_mode=zero_expert_mode,
                )
                loss = losses.total / gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_total += losses.total.item()
            accum_main += losses.main.item()
            accum_mtp += losses.mtp.item()
            accum_aux += losses.aux.item()
            accum_micro += 1

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip
        ).item()

        lr = lr_for_step(global_step + 1)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        if last_input_ids is not None:
            tokens_seen += int(last_input_ids.numel() * gradient_accumulation_steps)

        if global_step % log_interval == 0 or global_step == 1:
            now = time.time()
            elapsed = max(1e-6, now - start_time)
            tps = tokens_seen / elapsed
            interval_elapsed = max(1e-6, now - last_log_time)
            interval_tokens = tokens_seen - last_log_tokens
            tps_interval = interval_tokens / interval_elapsed
            print(
                f"step={global_step:5d}/{max_steps} "
                f"loss={accum_total/max(1, accum_micro):.4f} "
                f"main={accum_main/max(1, accum_micro):.4f} "
                f"mtp={accum_mtp/max(1, accum_micro):.4f} "
                f"aux={accum_aux/max(1, accum_micro):.4f} "
                f"lr={lr:.2e} grad_norm={grad_norm:.3f} tok/s(avg)={tps:,.0f} tok/s(cur)={tps_interval:,.0f}"
            )
            last_log_time = now
            last_log_tokens = tokens_seen

    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("Training run complete")
    print("=" * 72)
    print(f"Final step: {global_step}")
    print(f"Tokens seen: {tokens_seen:,}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 72 + "\n")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
    mtp_loss_weight: float,
    aux_loss_weight: float,
) -> None:
    model.eval()
    zero_expert_mode = getattr(model.config, "num_real_experts", 0) == 0

    total = 0.0
    main = 0.0
    mtp = 0.0
    aux = 0.0
    seen = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        next_token_ids = input_ids[:, 1:].contiguous()
        logits_ntp, logits_mtp, aux_loss = model(
            input_ids=input_ids,
            next_token_ids=next_token_ids,
            return_loss=True,
        )
        losses = compute_losses(
            logits_ntp=logits_ntp,
            logits_mtp=logits_mtp,
            labels=labels,
            aux_loss=aux_loss,
            mtp_loss_weight=mtp_loss_weight,
            aux_loss_weight=aux_loss_weight,
            zero_expert_mode=zero_expert_mode,
        )
        total += losses.total.item()
        main += losses.main.item()
        mtp += losses.mtp.item()
        aux += losses.aux.item()
        seen += 1

    if seen == 0:
        print("[Eval] No batches processed.")
        return

    avg_total = total / seen
    avg_main = main / seen
    avg_mtp = mtp / seen
    avg_aux = aux / seen
    ppl = math.exp(min(20.0, avg_main))

    print("\n" + "=" * 72)
    print("Evaluation")
    print("=" * 72)
    print(f"batches={seen}")
    print(
        f"loss={avg_total:.4f} main={avg_main:.4f} mtp={avg_mtp:.4f} aux={avg_aux:.4f}"
    )
    print(f"main_perplexity={ppl:.4f}")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/test model_1b.py on WikiText-2 with GPT-2 tokenizer",
    )
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="standard",
        choices=["standard", "kronecker"],
    )
    parser.add_argument("--strict-arch", action="store_true")

    # Dataset
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-eval-tokens", type=int, default=50000)

    # Training
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--mtp-loss-weight", type=float, default=1.0)
    parser.add_argument("--aux-loss-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Eval
    parser.add_argument("--eval-batches", type=int, default=0, help="0 disables eval")

    # Runtime
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"]
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Enable torch.compile on decoder and MTP attention+MLP sublayers",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode when --use-compile is set",
    )
    parser.add_argument(
        "--compile-target",
        type=str,
        default="deltanet",
        choices=["deltanet", "all"],
        help="Compile only DeltaNet layers (recommended) or all decoder layers including GSA",
    )
    parser.add_argument(
        "--compile-mtp",
        action="store_true",
        help="Also compile MTP block sublayers (often not needed for short-seq throughput)",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 matmul/cudnn on CUDA for throughput",
    )
    parser.add_argument(
        "--fused-adamw",
        action="store_true",
        help="Enable fused AdamW on CUDA if available",
    )
    parser.add_argument(
        "--non-blocking-transfer",
        action="store_true",
        help="Use non_blocking=True for H2D copies",
    )
    parser.add_argument(
        "--delta-recurrence-mode",
        type=str,
        default=None,
        choices=["sequential", "parallel_scan"],
        help="Override DeltaNet recurrence mode",
    )
    parser.add_argument(
        "--delta-chunk-size",
        type=int,
        default=None,
        help="Override DeltaNet chunk size",
    )
    parser.add_argument(
        "--gsa-emit-flash2-note",
        action="store_true",
        help="Emit one-time note that dense additive mask usually bypasses FlashAttention2 kernels",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if args.seq_length < 3:
        raise ValueError("--seq-length must be >= 3 for NTP+MTP loss computation.")

    fallback_used = ensure_reversible_ops_midpoint()
    if fallback_used:
        print(
            "[Compat] reversible_ops_midpoint not found, using sequential fallback stack."
        )

    from model_1b import Model70B, ModelConfig

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = ModelConfig()

    # Attached architecture table settings.
    config.vocab_size = len(tokenizer)
    config.hidden_size = 4096
    config.num_layers = 8
    config.num_deltanet_layers = 6
    config.num_gsa_layers = 2
    config.delta_v_heads = 32
    config.delta_qk_heads = 16
    config.delta_head_dim = 128
    config.gsa_num_heads = 16
    config.gsa_head_dim = 256
    config.delta_gate_dim = 384
    config.num_real_experts = 0
    config.num_null_experts = 0
    config.total_expert_slots = 0
    config.top_k = 0
    config.expert_intermediate_size = 2048
    config.shared_expert_intermediate_size = 2048
    config.enable_mtp = True
    config.mtp_num_predictions = 2
    config.data_sparsity = 0.5
    config.use_compile = args.use_compile
    config.compile_mode = args.compile_mode
    config.compile_target = args.compile_target
    config.compile_mtp = args.compile_mtp
    if args.delta_recurrence_mode is not None:
        config.delta_recurrence_mode = args.delta_recurrence_mode
    if args.delta_chunk_size is not None:
        config.delta_chunk_size = args.delta_chunk_size
    config.gsa_emit_flash2_note = args.gsa_emit_flash2_note
    if args.seq_length > config.max_seq_len:
        config.max_seq_len = args.seq_length

    print_architecture_report(config, strict=args.strict_arch)
    print(f"Tokenizer: {args.tokenizer} | vocab_size={len(tokenizer)}")
    print(f"Embedding mode: {args.embedding_type}")

    train_tokens = build_token_ids(
        split=args.train_split,
        tokenizer=tokenizer,
        add_eos=True,
        max_tokens=args.max_train_tokens,
    )
    stride = args.seq_length if args.stride is None else args.stride
    train_dataset = TokenBlockDataset(
        train_tokens, seq_length=args.seq_length, stride=stride
    )
    if len(train_dataset) == 0:
        raise ValueError("Not enough training tokens for selected seq-length.")

    eval_loader: Optional[DataLoader] = None
    if args.eval_batches > 0:
        eval_tokens = build_token_ids(
            split=args.eval_split,
            tokenizer=tokenizer,
            add_eos=True,
            max_tokens=args.max_eval_tokens,
        )
        eval_dataset = TokenBlockDataset(
            eval_tokens, seq_length=args.seq_length, stride=stride
        )
        if len(eval_dataset) == 0:
            raise ValueError("Not enough eval tokens for selected seq-length.")
    else:
        eval_dataset = None

    device = get_best_device(args.device)
    if device.type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print("[CUDA] TF32 enabled for matmul/cuDNN.")
    use_amp = not args.no_amp
    if device.type == "cpu" and use_amp:
        print("[AMP] CPU autocast disabled for this script.")
        use_amp = False
    if device.type == "mps" and use_amp and args.amp_dtype != "float16":
        print("[AMP] MPS supports float16 autocast only. Overriding amp_dtype=float16.")
        args.amp_dtype = "float16"

    num_workers = (
        args.num_workers if args.num_workers is not None else get_optimal_num_workers()
    )
    pin_memory = device.type == "cuda"
    persistent_workers = args.persistent_workers and num_workers > 0

    print("\n" + "=" * 72)
    print("DataLoader")
    print("=" * 72)
    print(f"train_blocks={len(train_dataset):,}")
    if eval_dataset is not None:
        print(f"eval_blocks={len(eval_dataset):,}")
    print(f"batch_size={args.batch_size}")
    print(f"num_workers={num_workers}")
    print(f"prefetch_factor={args.prefetch_factor}")
    print(f"persistent_workers={persistent_workers}")
    print(f"pin_memory={pin_memory}")
    print(f"device={device}")
    print(f"use_compile={config.use_compile} (mode={config.compile_mode})")
    print(f"compile_target={config.compile_target} compile_mtp={config.compile_mtp}")
    print(
        f"delta_recurrence_mode={config.delta_recurrence_mode} chunk_size={config.delta_chunk_size}"
    )
    print(
        f"fused_adamw={args.fused_adamw} tf32={args.tf32} non_blocking_transfer={args.non_blocking_transfer}"
    )

    if (
        config.use_compile
        and config.compile_target == "all"
        and config.compile_mtp
        and args.seq_length >= 1024
    ):
        print(
            "[Perf hint] compile_target=all + compile_mtp + seq_length>=1024 can raise memory and reduce tok/s. "
            "For better throughput, prefer --compile-target deltanet and omit --compile-mtp."
        )

    if args.seq_length <= 512 and config.delta_recurrence_mode == "parallel_scan":
        print(
            "[Perf hint] seq_length <= 512 with parallel_scan can be slower than sequential; "
            "consider --delta-recurrence-mode sequential for higher tok/s."
        )
    print("=" * 72 + "\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )

    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
        )

    if args.embedding_type == "kronecker":
        bpe_vocab, pf_codec = build_kronecker_artifacts(tokenizer, pf_dim=8192)
        model = Model70B(
            config, embedding_type="kronecker", bpe_vocab=bpe_vocab, pf_codec=pf_codec
        )
    else:
        model = Model70B(config, embedding_type="standard")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {total_params:,} (~{total_params / 1e9:.3f}B)")

    train(
        model=model,
        dataloader=train_loader,
        device=device,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        log_interval=args.log_interval,
        use_amp=use_amp,
        amp_dtype_str=args.amp_dtype,
        mtp_loss_weight=args.mtp_loss_weight,
        aux_loss_weight=args.aux_loss_weight,
        fused_adamw=args.fused_adamw,
        non_blocking_transfer=args.non_blocking_transfer,
    )

    if eval_loader is not None and args.eval_batches > 0:
        evaluate(
            model=model.to(device),
            dataloader=eval_loader,
            device=device,
            max_batches=args.eval_batches,
            mtp_loss_weight=args.mtp_loss_weight,
            aux_loss_weight=args.aux_loss_weight,
        )


if __name__ == "__main__":
    main()
