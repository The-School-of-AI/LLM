"""
WikiText-2 test runner for the Test_Code/model_1b.py reference architecture.

This script is intentionally separate from train_wikitext2_gpt2.py because
ReferenceLLM has:
1) Different forward signature/output object
2) Explicit next_token_ids input for MTP
3) Optional Kronecker embedding initialization path
"""

import argparse
import math
import random

# Add repo root to path
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import (
    PRESET_CONFIGS,
    ConnectionType,
    EmbeddingType,
    ModelConfig,
    get_preset_config,
)
from models.reference_llm import ReferenceLLM, ReferenceLLMOutput

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
    import os

    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 4
    return max(2, min(16, int(cpu_count * 0.75)))


def load_config_from_yaml(config_path: str) -> Tuple[ModelConfig, Dict[str, Any]]:
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    training_data = config_data.pop("training", {})
    model_config = ModelConfig.from_dict(config_data)
    return model_config, training_data


def align_model_context_to_seq_length(
    model_config: ModelConfig, seq_length: int
) -> None:
    if seq_length <= model_config.max_position_embeddings:
        return

    old_max = model_config.max_position_embeddings
    model_config.max_position_embeddings = seq_length
    print(
        f"[Context] Increasing max_position_embeddings: {old_max} -> {seq_length} "
        f"to match seq_length."
    )

    pos = getattr(model_config, "position", None)
    if pos is None:
        return

    pos_type = getattr(pos, "position_type", None)
    pos_type_value = pos_type.value if hasattr(pos_type, "value") else str(pos_type)
    if pos_type_value != "yarn":
        return

    original_max = int(getattr(pos, "yarn_original_max_position", old_max) or old_max)
    original_max = max(1, original_max)
    required_scale = seq_length / float(original_max)
    current_scale = float(getattr(pos, "yarn_scale", 1.0))
    if required_scale > current_scale:
        pos.yarn_scale = required_scale
        print(
            f"[Context] Increasing YaRN scale: {current_scale:.4g} -> {required_scale:.4g} "
            f"(original_max={original_max}, target_seq={seq_length})."
        )


class TokenBlockDataset(Dataset):
    """Simple fixed-length token block dataset."""

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


def architecture_checks(config: ModelConfig) -> Tuple[List[str], List[str]]:
    """
    Validate architecture against expected 1B reference targets from your table.
    Returns (ok_lines, mismatch_lines).
    """
    ok: List[str] = []
    bad: List[str] = []

    def check(name: str, actual: Any, expected: Any):
        if actual == expected:
            ok.append(f"{name}: {actual}")
        else:
            bad.append(f"{name}: actual={actual}, expected={expected}")

    check("hidden_size", config.hidden_size, 4096)
    check("num_hidden_layers(backbone)", config.num_hidden_layers, 8)
    check("num_deltanet_layers", config.num_deltanet_layers, 6)
    check("num_gsa_layers", config.num_gsa_layers, 2)
    check("delta_gate_dim", config.attention.delta_gate_dim, 384)
    check("ffn.intermediate_size", config.ffn.intermediate_size, 2048)
    check("ffn.moe_num_experts", config.ffn.moe_num_experts, 0)
    check("ffn.moe_num_experts_per_tok", config.ffn.moe_num_experts_per_tok, 0)
    check("gsa_indexer_dim(d_idx)", config.attention.gsa_indexer_dim, 32)
    check(
        "connection_type",
        config.connection.connection_type.value,
        ConnectionType.MHC_V2.value,
    )
    check("mhc_alpha_init", config.connection.mhc_alpha_init, 0.1)
    check("integration.use_reversible", config.integration.use_reversible, True)
    check("head.num_predict_tokens", config.head.num_predict_tokens, 2)
    check("head.mtp_block_type", config.head.mtp_block_type, "full_transformer")
    check("attention.delta_v_heads", config.attention.delta_v_heads, 32)
    check("attention.delta_qk_heads", config.attention.delta_qk_heads, 16)
    check("attention.delta_head_dim", config.attention.delta_head_dim, 128)
    check("attention.gsa_num_heads", config.attention.gsa_num_heads, 16)
    check("attention.gsa_head_dim", config.attention.gsa_head_dim, 256)

    return ok, bad


def print_architecture_report(config: ModelConfig, strict: bool) -> None:
    ok_lines, bad_lines = architecture_checks(config)

    print("\n" + "=" * 72)
    print("Reference Architecture Check (Test_Code/model_1b.py targets)")
    print("=" * 72)
    print(f"Backbone Layers: {config.num_hidden_layers}")
    print(f"MTP Layers: {config.head.num_predict_tokens - 1}")
    print(
        f"Total Computational Layers: {config.num_hidden_layers + config.head.num_predict_tokens - 1}"
    )
    print(f"Delta/GSA Split: {config.num_deltanet_layers}/{config.num_gsa_layers}")
    print(f"Estimated Params: {config.num_parameters_billions:.3f}B")

    if bad_lines:
        print("\nMISMATCHES:")
        for line in bad_lines:
            print(f"  - {line}")
    else:
        print("\nAll key architectural checks matched.")

    if ok_lines:
        print("\nMatched checks:")
        for line in ok_lines:
            print(f"  - {line}")

    print("=" * 72 + "\n")

    if strict and bad_lines:
        raise ValueError(
            "Reference architecture mismatch detected with --strict-arch enabled. "
            "Fix config or disable strict mode."
        )


def build_kronecker_artifacts(tokenizer, model_config: ModelConfig):
    from components.embeddings.kronecker_embedding import (
        KroneckerConfig,
        KroneckerEmbeddings,
    )

    vocab_size = model_config.vocab_size
    pf_dim = model_config.embedding.kronecker_pf_dim

    approx_pf_gb = (vocab_size * pf_dim * 2) / (1024**3)  # bf16 table
    print(
        f"[Kronecker] Building PF table for vocab_size={vocab_size}, D={pf_dim}. "
        f"Approx GPU/CPU buffer size (bf16): ~{approx_pf_gb:.2f} GB"
    )

    # Decode each token id into text piece for Kronecker encoding.
    bpe_vocab = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in range(vocab_size)
    ]

    pf_cfg = KroneckerConfig(D=pf_dim)
    pf_codec = KroneckerEmbeddings(pf_cfg)
    return bpe_vocab, pf_codec


def get_amp_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float16":
        return torch.float16
    return torch.bfloat16


def train_reference(
    model: ReferenceLLM,
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
) -> None:
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )

    amp_dtype = get_amp_dtype(amp_dtype_str)

    scaler_enabled = use_amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = GradScaler("cuda", enabled=scaler_enabled)

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
    data_iter = iter(dataloader)

    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        accum_total_loss = 0.0
        accum_main_loss = 0.0
        accum_mtp_loss = 0.0
        accum_aux_loss = 0.0
        accum_micro_steps = 0

        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            next_token_ids = input_ids[:, 1:].contiguous()

            autocast_enabled = use_amp and (
                (device.type == "cuda")
                or (device.type == "mps" and amp_dtype == torch.float16)
            )
            with autocast(
                device_type=device.type,
                enabled=autocast_enabled,
                dtype=amp_dtype,
            ):
                outputs = model(
                    input_ids=input_ids,
                    next_token_ids=next_token_ids,
                    labels=labels,
                )
                if not isinstance(outputs, ReferenceLLMOutput):
                    raise TypeError(
                        f"Expected ReferenceLLMOutput, got {type(outputs)}. "
                        "Please verify model architecture wiring."
                    )
                if outputs.loss is None:
                    raise RuntimeError(
                        "Model returned no loss even though labels were provided."
                    )

                loss = outputs.loss / gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_total_loss += outputs.loss.item()
            if outputs.loss_dict is not None:
                if "main_loss" in outputs.loss_dict:
                    accum_main_loss += outputs.loss_dict["main_loss"].item()
                if "mtp_loss" in outputs.loss_dict:
                    accum_mtp_loss += outputs.loss_dict["mtp_loss"].item()
                if "aux_loss" in outputs.loss_dict:
                    accum_aux_loss += outputs.loss_dict["aux_loss"].item()
            accum_micro_steps += 1

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
        tokens_seen += (
            input_ids.shape[0] * input_ids.shape[1] * gradient_accumulation_steps
        )

        if global_step % log_interval == 0 or global_step == 1:
            elapsed = max(1e-6, time.time() - start_time)
            tps = tokens_seen / elapsed
            avg_total = accum_total_loss / max(1, accum_micro_steps)
            avg_main = (
                accum_main_loss / max(1, accum_micro_steps)
                if accum_main_loss > 0
                else 0.0
            )
            avg_mtp = (
                accum_mtp_loss / max(1, accum_micro_steps)
                if accum_mtp_loss > 0
                else 0.0
            )
            avg_aux = (
                accum_aux_loss / max(1, accum_micro_steps)
                if accum_aux_loss > 0
                else 0.0
            )

            print(
                f"step={global_step:5d}/{max_steps} "
                f"loss={avg_total:.4f} main={avg_main:.4f} mtp={avg_mtp:.4f} aux={avg_aux:.4f} "
                f"lr={lr:.2e} grad_norm={grad_norm:.3f} tok/s={tps:,.0f}"
            )

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("Reference training run complete")
    print("=" * 72)
    print(f"Final step: {global_step}")
    print(f"Tokens seen: {tokens_seen:,}")
    print(f"Elapsed: {total_elapsed:.1f}s")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/Test ReferenceLLM on WikiText-2 using GPT-2 tokenizer",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--preset",
        type=str,
        default="1b-reference",
        choices=list(PRESET_CONFIGS.keys()),
        help="Model preset if --config is not provided",
    )
    parser.add_argument(
        "--strict-arch", action="store_true", help="Fail on architecture mismatch"
    )

    # Tokenizer / dataset
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)

    # Embedding mode
    parser.add_argument(
        "--embedding-type",
        type=str,
        default=None,
        choices=["standard", "kronecker"],
        help="Override embedding type from config",
    )

    # Training hyperparameters
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument(
        "--device", type=str, default=None, choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--amp-dtype", type=str, default=None, choices=["bfloat16", "float16"]
    )
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # DataLoader tuning
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument(
        "--disable-reversible-checkpoint",
        action="store_true",
        help="Disable checkpointing inside ReversibleMidpointStack (higher memory, more stable)",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if args.config:
        print(f"Loading config from: {args.config}")
        model_config, training_dict = load_config_from_yaml(args.config)
    else:
        print(f"Using preset: {args.preset}")
        model_config = get_preset_config(args.preset)
        training_dict = {}

    # Resolve embedding mode
    embedding_type = (
        args.embedding_type
        if args.embedding_type is not None
        else model_config.embedding.embedding_type.value
    )
    model_config.embedding.embedding_type = EmbeddingType(embedding_type)

    # Training defaults, then YAML training values, then CLI overrides.
    train_cfg: Dict[str, Any] = {
        "max_steps": 200,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "seq_length": 256,
        "learning_rate": 3e-4,
        "min_learning_rate": 1e-5,
        "warmup_steps": 20,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "gradient_clip": 1.0,
        "device": "auto",
        "use_amp": True,
        "amp_dtype": "bfloat16",
        "log_interval": 10,
    }
    for key in list(train_cfg.keys()):
        if key in training_dict:
            train_cfg[key] = training_dict[key]

    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.gradient_accumulation is not None:
        train_cfg["gradient_accumulation_steps"] = args.gradient_accumulation
    if args.seq_length is not None:
        train_cfg["seq_length"] = args.seq_length
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
    if args.min_learning_rate is not None:
        train_cfg["min_learning_rate"] = args.min_learning_rate
    if args.warmup_steps is not None:
        train_cfg["warmup_steps"] = args.warmup_steps
    if args.weight_decay is not None:
        train_cfg["weight_decay"] = args.weight_decay
    if args.beta1 is not None:
        train_cfg["beta1"] = args.beta1
    if args.beta2 is not None:
        train_cfg["beta2"] = args.beta2
    if args.eps is not None:
        train_cfg["eps"] = args.eps
    if args.gradient_clip is not None:
        train_cfg["gradient_clip"] = args.gradient_clip
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.no_amp:
        train_cfg["use_amp"] = False
    if args.amp_dtype is not None:
        train_cfg["amp_dtype"] = args.amp_dtype
    if args.log_interval is not None:
        train_cfg["log_interval"] = args.log_interval

    align_model_context_to_seq_length(model_config, train_cfg["seq_length"])
    print_architecture_report(model_config, strict=args.strict_arch)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config.vocab_size = len(tokenizer)
    if model_config.vocab_size != 131072:
        print(
            f"[Note] Using tokenizer vocab_size={model_config.vocab_size} (not 131072). "
            "Total parameter count will differ from the 1B target table."
        )

    print(f"Tokenizer: {args.tokenizer} | vocab_size={model_config.vocab_size}")
    print(f"Embedding mode: {embedding_type}")

    token_ids = build_token_ids(
        split=args.dataset_split,
        tokenizer=tokenizer,
        add_eos=True,
        max_tokens=args.max_tokens,
    )
    stride = train_cfg["seq_length"] if args.stride is None else args.stride
    dataset = TokenBlockDataset(
        token_ids, seq_length=train_cfg["seq_length"], stride=stride
    )
    if len(dataset) == 0:
        raise ValueError(
            "Not enough tokens for selected seq_length. "
            "Reduce --seq-length or increase --max-tokens."
        )

    num_workers = (
        args.num_workers if args.num_workers is not None else get_optimal_num_workers()
    )
    device = get_best_device(train_cfg["device"])
    pin_memory = device.type == "cuda"
    persistent_workers = args.persistent_workers and num_workers > 0

    if (
        device.type == "mps"
        and train_cfg["use_amp"]
        and train_cfg["amp_dtype"] != "float16"
    ):
        print(
            "[AMP] MPS supports float16 autocast only. Overriding amp_dtype to float16."
        )
        train_cfg["amp_dtype"] = "float16"
    if device.type == "cpu" and train_cfg["use_amp"]:
        print("[AMP] CPU autocast disabled for this script.")
        train_cfg["use_amp"] = False

    print("\n" + "=" * 72)
    print("DataLoader")
    print("=" * 72)
    print(f"dataset_blocks={len(dataset):,}")
    print(f"batch_size={train_cfg['batch_size']}")
    print(f"num_workers={num_workers}")
    print(f"prefetch_factor={args.prefetch_factor}")
    print(f"persistent_workers={persistent_workers}")
    print(f"pin_memory={pin_memory}")
    print("=" * 72 + "\n")

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )

    if embedding_type == "kronecker":
        bpe_vocab, pf_codec = build_kronecker_artifacts(tokenizer, model_config)
        model = ReferenceLLM(
            model_config,
            embedding_type="kronecker",
            bpe_vocab=bpe_vocab,
            pf_codec=pf_codec,
        )
    else:
        model = ReferenceLLM(model_config, embedding_type="standard")

    if args.disable_reversible_checkpoint and hasattr(model, "stack"):
        model.stack.use_checkpoint = False
        print("[Reversible] Checkpointing disabled in ReversibleMidpointStack.")

    print(f"Device selected: {device}")

    train_reference(
        model=model,
        dataloader=dataloader,
        device=device,
        max_steps=train_cfg["max_steps"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        min_learning_rate=train_cfg["min_learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        beta1=train_cfg["beta1"],
        beta2=train_cfg["beta2"],
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"],
        gradient_clip=train_cfg["gradient_clip"],
        log_interval=train_cfg["log_interval"],
        use_amp=bool(train_cfg["use_amp"]),
        amp_dtype_str=str(train_cfg["amp_dtype"]),
    )


if __name__ == "__main__":
    main()
