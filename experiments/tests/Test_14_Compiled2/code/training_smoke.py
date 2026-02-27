import argparse
import os
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed

from src.train import train_epoch
from src.utils import print_rank_0


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss)


def build_synthetic_loader(batch_size: int, seq_len: int, vocab_size: int, steps: int):
    batches = []
    for _ in range(steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        labels = input_ids.clone()
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        batches.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

    class _Loader:
        def __iter__(self):
            return iter(batches)

        def __len__(self):
            return len(batches)

    return _Loader()


def parse_args():
    p = argparse.ArgumentParser()
    p = deepspeed.add_config_arguments(p)
    p.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by DeepSpeed launcher)",
    )
    p.set_defaults(deepspeed_config="../deepspeed/smoke_train.json")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--metrics_jsonl_path", type=str, default="../results/run/metrics.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    run_id = os.environ.get("RUN_ID") or f"train_smoke_{int(time.time())}"
    rank = int(os.environ.get("RANK", "0"))

    ops = None
    try:
        from components import TrainingOps

        skip_vector_check = os.environ.get("SKIP_VECTOR_CHECK", "0") == "1"
        vector_service_name = os.environ.get("VECTOR_SERVICE_NAME", "t12-vector.service")

        ops = TrainingOps(
            run_id=run_id,
            rank=rank,
            skip_vector_check=skip_vector_check,
            vector_service_name=vector_service_name,
        )
    except Exception as e:
        print_rank_0(f"[WARN] TrainingOps not available/failed to start: {e}")
        ops = None

    model = TinyLM(vocab_size=args.vocab_size, hidden_size=args.hidden_size)

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        args=args,
    )

    loader = build_synthetic_loader(
        batch_size=1,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        steps=args.steps,
    )

    train_epoch(
        model_engine=engine,
        train_loader=loader,
        epoch=0,
        max_steps=args.steps,
        log_interval=1,
        enable_system_metrics=False,
        checkpoint_interval=None,
        output_dir=None,
        checkpoint_manager=None,
        start_step=0,
        global_step=0,
        metrics_jsonl_path=args.metrics_jsonl_path,
        max_chunk_gb=0.25,
        profiler=None,
        profile_steps=None,
        profile_output_dir=None,
        ops=ops,
    )

    if ops is not None:
        try:
            ops.shutdown()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
