import argparse
import os
from datetime import datetime

# 🔒 HARD OFFLINE LOCK (VERY IMPORTANT)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import deepspeed
import torch
from transformers import GPT2Config

from src.data import get_dataloaders, get_tokenizer
from src.moe_utils import create_moe_param_groups, is_moe_model
from src.train import train_epoch, evaluate, generate_text
from src.utils import set_seed
from src.models.moe_gpt2 import MoEGPT2LMHeadModel

# TensorBoard optional
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def parse_args():
    p = argparse.ArgumentParser()

    # core
    p.add_argument("--model_name", type=str, default="distilgpt2")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # deepspeed
    p.add_argument("--deepspeed_config", type=str, required=True)
    p.add_argument("--local_rank", type=int, default=-1)

    # MoE
    p.add_argument("--num_experts", type=int, default=8)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--moe_layer_idx", type=int, default=0)

    # logging
    p.add_argument("--tb_logdir", type=str, default="tb_logs")
    p.add_argument("--run_name", type=str, default=None)

    # generation
    p.add_argument("--test_generation", action="store_true", default=True)
    p.add_argument("--generation_prompt", type=str, default="Hi")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.tb_logdir, exist_ok=True)

    # TensorBoard
    writer = None
    if SummaryWriter is not None and args.local_rank in (-1, 0):
        writer = SummaryWriter(log_dir=os.path.join(args.tb_logdir, run_name))
    elif args.local_rank in (-1, 0):
        print("[warn] TensorBoard not available (pip install tensorboard)")

    if args.local_rank in (-1, 0):
        print("=" * 80)
        print("DeepSpeed MoE Training")
        print("=" * 80)
        print(f"DeepSpeed: {deepspeed.__version__}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        print(f"Model: {args.model_name}")
        print(f"MoE: experts={args.num_experts}, top_k={args.top_k}, layer={args.moe_layer_idx}")
        print(f"DS config: {args.deepspeed_config}")
        print(f"TensorBoard: {os.path.join(args.tb_logdir, run_name)}")
        print("=" * 80)

    # ------------------
    # DATA (LOCAL ONLY)
    # ------------------
    tokenizer = get_tokenizer(args.model_name)

    train_loader, eval_loader, test_loader, _ = get_dataloaders(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # ------------------
    # MODEL (LOCAL ONLY)
    # ------------------
    config = GPT2Config.from_pretrained(
        args.model_name,
        local_files_only=True,   # ✅ critical fix
    )

    model = MoEGPT2LMHeadModel(
        config=config,
        num_experts=args.num_experts,
        top_k=args.top_k,
        moe_layer_idx=args.moe_layer_idx,
        min_capacity=0,          # ✅ fixes generation crash
    )

    # ------------------
    # DEEPSPEED INIT
    # ------------------
    if is_moe_model(model):
        model_params = create_moe_param_groups(model)
    else:
        model_params = model.parameters()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model_params,
    )

    global_step = 0

    # ------------------
    # TRAIN
    # ------------------
    for epoch in range(args.num_epochs):
        if model_engine.global_rank == 0:
            print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")

        _, global_step = train_epoch(
            model_engine,
            train_loader,
            epoch,
            writer=writer,
            log_interval=args.log_interval,
            global_step_start=global_step,
        )

        evaluate(
            model_engine,
            eval_loader,
            phase="valid",
            writer=writer,
            global_step=global_step,
        )

    evaluate(
        model_engine,
        test_loader,
        phase="test",
        writer=writer,
        global_step=global_step,
    )

    # ------------------
    # GENERATION
    # ------------------
    if args.test_generation and model_engine.global_rank == 0:
        print("\nTesting generation...")
        generate_text(model_engine, tokenizer, prompt=args.generation_prompt)

    if writer:
        writer.flush()
        writer.close()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()