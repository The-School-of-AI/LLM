"""
Staged Cosine LR Scheduler — token-budget-driven, stage-aware.

Replaces DeepSpeed's built-in WarmupCosineLR with a scheduler that:
  1. Reads a single YAML defining all 7 stages (1B → WU_3B → 3B → ... → 70B)
  2. Auto-computes total_steps from token_budget and batch geometry
  3. Chains warmup_from across stages (warmup_from: auto)
  4. Applies cosine decay within each stage

Usage:
    sched = StagedCosineScheduler(
        config_path="configs/lr_schedule.yaml",
        stage_name="1B",
        tokens_per_step=131072,   # global_batch_size * seq_len
    )

    # Inside training loop, AFTER model_engine.step():
    lr = sched.step(model_engine.optimizer, current_step_in_stage)
"""

import math
import os

import yaml

from .utils import print_rank_0


class StagedCosineScheduler:
    """
    Cosine LR scheduler driven by token budgets.

    For a given stage, computes:
        total_steps = token_budget / tokens_per_step
        end_lr = peak_lr / decay_ratio

    LR schedule within a stage:
        [0, warmup_steps):         linear from warmup_from → peak_lr
        [warmup_steps, total_steps): cosine from peak_lr → end_lr
    """

    def __init__(self, config_path: str, stage_name: str, tokens_per_step: int):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"LR schedule config not found: {config_path}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        stages = cfg["stages"]
        stage_names = [s["name"] for s in stages]

        if stage_name not in stage_names:
            raise ValueError(
                f"Stage '{stage_name}' not found in {config_path}. "
                f"Available: {stage_names}"
            )

        stage_idx = stage_names.index(stage_name)
        stage = stages[stage_idx]

        self.stage_name = stage_name
        self.tokens_per_step = tokens_per_step
        self.token_budget = float(stage["token_budget"])
        self.total_steps = int(self.token_budget / tokens_per_step)
        self.peak_lr = float(stage["peak_lr"])
        self.decay_ratio = float(stage.get("decay_ratio", 10))
        self.end_lr = self.peak_lr / self.decay_ratio
        self.warmup_steps = int(stage.get("warmup_steps", 0))
        self.weight_decay = float(stage.get("weight_decay", 0.0))

        # Offset: global_step at which this stage begins.
        # For the 1B stage this is 0.  On stage transitions (e.g. 1B -> WU_3B),
        # main.py sets this to the resumed global_step so that
        # step_in_stage = global_step - stage_step_offset starts from 0.
        self.stage_step_offset: int = 0

        # Resolve warmup_from
        warmup_from = stage.get("warmup_from", 0)
        if warmup_from == "auto":
            if stage_idx == 0:
                raise ValueError(
                    f"Stage '{stage_name}' is the first stage but has warmup_from='auto'. "
                    f"First stage must specify a numeric warmup_from (usually 0)."
                )
            prev = stages[stage_idx - 1]
            prev_decay = float(prev.get("decay_ratio", 10))
            prev_peak = float(prev["peak_lr"])
            self.warmup_from = prev_peak / prev_decay
        else:
            self.warmup_from = float(warmup_from)

        # Cosine phase steps (everything after warmup)
        self.cosine_steps = max(1, self.total_steps - self.warmup_steps)

        # Log the resolved schedule
        print_rank_0(f"\n{'='*70}")
        print_rank_0(f"  LR Schedule: stage={self.stage_name}")
        print_rank_0(f"  token_budget     = {self.token_budget:.3e}")
        print_rank_0(f"  tokens_per_step  = {self.tokens_per_step:,}")
        print_rank_0(f"  total_steps      = {self.total_steps:,}")
        print_rank_0(f"  warmup_steps     = {self.warmup_steps:,}")
        print_rank_0(f"  warmup_from      = {self.warmup_from:.2e}")
        print_rank_0(f"  peak_lr          = {self.peak_lr:.2e}")
        print_rank_0(f"  end_lr           = {self.end_lr:.2e}")
        print_rank_0(f"  decay_ratio      = {self.decay_ratio:.0f}x")
        print_rank_0(f"  weight_decay     = {self.weight_decay}")
        print_rank_0(f"{'='*70}\n")

    def get_lr(self, step: int) -> float:
        """Compute LR for a given step (0-indexed within this stage)."""
        step = max(0, min(step, self.total_steps))

        if step < self.warmup_steps:
            # Linear warmup: warmup_from → peak_lr
            t = step / max(1, self.warmup_steps)
            return self.warmup_from + (self.peak_lr - self.warmup_from) * t
        else:
            # Cosine decay: peak_lr → end_lr
            progress = (step - self.warmup_steps) / self.cosine_steps
            progress = min(progress, 1.0)
            return self.end_lr + 0.5 * (self.peak_lr - self.end_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

    def step(self, optimizer, global_step: int) -> float:
        """Compute LR for global_step (adjusted by stage_step_offset) and apply."""
        step_in_stage = max(0, global_step - self.stage_step_offset)
        lr = self.get_lr(step_in_stage)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def apply_weight_decay(self, optimizer) -> None:
        """
        Set weight_decay on all optimizer param groups to this stage's value.
        Called once at init, not every step.
        """
        for pg in optimizer.param_groups:
            pg["weight_decay"] = self.weight_decay

    def state_dict(self) -> dict:
        """For checkpoint serialization."""
        return {
            "stage_name": self.stage_name,
            "stage_step_offset": self.stage_step_offset,
            "tokens_per_step": self.tokens_per_step,
            "token_budget": self.token_budget,
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "end_lr": self.end_lr,
            "warmup_steps": self.warmup_steps,
            "warmup_from": self.warmup_from,
            "weight_decay": self.weight_decay,
        }
