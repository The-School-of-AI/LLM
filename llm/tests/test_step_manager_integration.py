"""
Integration test: StepManager step-continuity after checkpoint resume.

Scenario
--------
1. Tokenize the first N rows of ``wikitext-2-raw-v1`` with a fast BPE
   tokenizer (tiktoken / GPT-2 encoding) — no HuggingFace model import.
2. Build a tiny causal-LM in pure PyTorch (embedding + 2 transformer blocks
   + LM head, ~1 M params) — completely sidesteps the torchvision import chain
   that Transformers v5.x triggers on this machine.
3. Run FIRST_PHASE_STEPS optimizer steps, capture step numbers reported
   to a logger stub.
4. Save an in-memory checkpoint via ``step_manager.inject_state()``.
5. Create a fresh ``StepManager``, restore from the checkpoint.
6. Run SECOND_PHASE_STEPS more optimizer steps, record step numbers.
7. Assert the combined sequence is ``[1, 2, …, TOTAL_STEPS]`` — no restart
   from zero.

Design notes
------------
* Pure-PyTorch model — no torchvision / transformers model classes needed.
* Real wikitext-2 data from HuggingFace ``datasets`` (already a project dep).
* tiktoken is available in this env (pulled in by the project); falls back to
  a trivial char-level codec if not installed.
* In-memory checkpoint — no temp files.
* Runs on Mac CPU; GPU used automatically if available.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

from llm.training_state_manager import StepManager, _STATE_KEY
from llm.logger import Metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"

SEQ_LEN = 64          # tokens per sample
BATCH_SIZE = 2
FIRST_PHASE_STEPS = 5
SECOND_PHASE_STEPS = 5
TOTAL_STEPS = FIRST_PHASE_STEPS + SECOND_PHASE_STEPS
LEARNING_RATE = 1e-3

# Tiny LM hyper-params
VOCAB_SIZE = 256      # byte-level vocab — no tokenizer dependency
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FFN_DIM = 128


# ---------------------------------------------------------------------------
# Tiny causal LM (pure PyTorch)
# ---------------------------------------------------------------------------

class _CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.n_heads = NUM_HEADS
        self.head_dim = EMBED_DIM // NUM_HEADS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, D]
        q = q.transpose(1, 2)        # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) / scale
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _CausalSelfAttention()
        self.ff = nn.Sequential(
            nn.Linear(EMBED_DIM, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, EMBED_DIM),
        )
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class _TinyLM(nn.Module):
    """~1 M-param causal language model.  Input/output vocab = byte values 0-255."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_embed = nn.Embedding(SEQ_LEN, EMBED_DIM)
        self.blocks = nn.Sequential(*[_Block() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)          # [B, T, V]

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB_SIZE),
                labels[:, 1:].reshape(-1),
            )
        return logits, loss


# ---------------------------------------------------------------------------
# Dataset: wikitext-2 tokenised as raw UTF-8 bytes
# ---------------------------------------------------------------------------

def _build_dataset(num_samples: int) -> TensorDataset:
    """Download wikitext-2 and encode text as byte sequences."""
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", trust_remote_code=False)

    chunks: list[torch.Tensor] = []
    for row in raw:
        text: str = row["text"].strip()
        if not text:
            continue
        encoded = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        # Slice into fixed-length windows
        for start in range(0, len(encoded) - SEQ_LEN, SEQ_LEN):
            chunks.append(encoded[start : start + SEQ_LEN])
        if len(chunks) >= num_samples:
            break

    if not chunks:
        raise RuntimeError("wikitext-2 returned no usable rows")

    data = torch.stack(chunks[:num_samples])   # [N, SEQ_LEN]
    return TensorDataset(data)


# ---------------------------------------------------------------------------
# Minimal logger stub
# ---------------------------------------------------------------------------

class _StepRecorder:
    """Records every (step, loss) pair passed to log_metrics."""

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.losses: list[float] = []

    def log_metrics(self, step: int, metrics: Metrics) -> None:
        self.steps.append(step)
        values = metrics.get_values()
        loss_val = values.get("loss")
        if loss_val is not None:
            self.losses.append(float(loss_val))


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def _run_phase(
    model: _TinyLM,
    optimizer: AdamW,
    data_iter,  # noqa: ANN001
    step_manager: StepManager,
    recorder: _StepRecorder,
    num_steps: int,
    device: torch.device,
) -> None:
    model.train()
    for _ in range(num_steps):
        (batch,) = next(data_iter)
        batch = batch.to(device)
        optimizer.zero_grad()
        _, loss = model(batch, labels=batch)
        assert loss is not None
        loss.backward()
        optimizer.step()

        # ---- StepManager integration (same pattern as pretrainer.py) ----
        step_manager.increment(
            tokens=batch.numel(),
            samples=batch.shape[0],
        )
        metrics = Metrics()
        metrics.add("loss", loss.item())
        step_manager.log(metrics, recorder)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_step_continuity_after_resume() -> None:
    """Step numbers must continue from the checkpoint, not restart from zero."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ---- Build model and dataset ----------------------------------------
    print("Building tiny LM …")
    model = _TinyLM().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    print(f"Loading wikitext-2-raw-v1 (need {TOTAL_STEPS * BATCH_SIZE} samples) …")
    dataset = _build_dataset(TOTAL_STEPS * BATCH_SIZE + 10)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    data_iter = iter(loader)

    # ---- Phase 1: train FIRST_PHASE_STEPS steps -------------------------
    print(f"\nPhase 1: training {FIRST_PHASE_STEPS} steps …")
    mgr1 = StepManager()
    mgr1.restore(None)
    rec1 = _StepRecorder()

    _run_phase(model, optimizer, data_iter, mgr1, rec1, FIRST_PHASE_STEPS, device)

    print(f"  Logged steps : {rec1.steps}")
    print(f"  Losses       : {[f'{l:.4f}' for l in rec1.losses]}")

    assert rec1.steps == list(range(1, FIRST_PHASE_STEPS + 1)), (
        f"Phase-1 step mismatch: got {rec1.steps}"
    )

    # ---- Save in-memory checkpoint --------------------------------------
    checkpoint: dict = {}
    mgr1.inject_state(checkpoint)
    assert _STATE_KEY in checkpoint
    assert checkpoint[_STATE_KEY]["global_step"] == FIRST_PHASE_STEPS
    print(f"\n  Checkpoint → global_step={checkpoint[_STATE_KEY]['global_step']}")

    # ---- Phase 2: resume and train SECOND_PHASE_STEPS steps -------------
    print(f"\nPhase 2: resuming → training {SECOND_PHASE_STEPS} steps …")
    mgr2 = StepManager()
    restored = mgr2.restore(checkpoint)

    assert restored.global_step == FIRST_PHASE_STEPS, (
        f"Restored global_step={restored.global_step}, expected {FIRST_PHASE_STEPS}"
    )

    rec2 = _StepRecorder()
    _run_phase(model, optimizer, data_iter, mgr2, rec2, SECOND_PHASE_STEPS, device)

    expected_p2 = list(range(FIRST_PHASE_STEPS + 1, TOTAL_STEPS + 1))
    print(f"  Logged steps : {rec2.steps}")
    print(f"  Losses       : {[f'{l:.4f}' for l in rec2.losses]}")

    assert rec2.steps == expected_p2, (
        f"Phase-2 step mismatch: expected {expected_p2}, got {rec2.steps}"
    )

    # ---- Full continuity assertion --------------------------------------
    all_steps = rec1.steps + rec2.steps
    expected_all = list(range(1, TOTAL_STEPS + 1))
    assert all_steps == expected_all, (
        f"Step continuity FAILED.\n"
        f"  Expected : {expected_all}\n"
        f"  Got      : {all_steps}"
    )

    print(f"\nStep continuity verified ✓  {all_steps}")
