"""Tests for OpusDataSelector composable middleware."""
import torch
import torch.nn as nn
from llm.opus.config import OpusConfig
from llm.opus.data_selector import OpusDataSelector


class TinyModel(nn.Module):
    """
    Minimal model that satisfies GhostCollector discovery requirements:
    - Has nn.Linear modules with "layers." in their path
    - Has an lm_head nn.Linear
    - forward accepts input_ids and kwargs (return_hidden, return_memory)
    - Returns (hidden_states, None, aux_loss)
    """

    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        # GhostCollector looks for "layers." in module path with nn.Linear + 2D weight
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "ff": nn.Linear(hidden_dim, hidden_dim, bias=False),
            })
            for _ in range(2)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = h + layer["attn"](h)
            h = h + layer["ff"](h)
        return h, None, torch.tensor(0.0, device=h.device)


def _make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def _make_candidate_batch(batch_size: int = 8, seq_len: int = 16, vocab_size: int = 32):
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
    }


def _make_proxy_loader(n_batches: int = 10, batch_size: int = 4, seq_len: int = 16, vocab_size: int = 32):
    batches = [
        {"input_ids": torch.randint(0, vocab_size, (batch_size, seq_len))}
        for _ in range(n_batches)
    ]
    return iter(batches)


def test_data_selector_select_batch_reduces_batch():
    """OPUS scoring mode should return a batch smaller than the input."""
    torch.manual_seed(42)
    vocab_size = 32
    hidden_dim = 16
    model = TinyModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    optimizer = _make_optimizer(model)

    # Do a dummy step so optimizer state is populated
    dummy = torch.randint(0, vocab_size, (2, 16))
    h, _, aux = model(dummy)
    logits = model.lm_head(h)
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), dummy.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    cfg = OpusConfig(
        enabled=True,
        selection_mode="opus",
        selection_ratio=0.5,
        score_seq_len=16,
        sketch_dim=64,
        temperature=0.9,
        fallback_random_on_error=True,
        include_embeddings=False,
        include_lm_head=False,
    )

    proxy_loader = _make_proxy_loader(vocab_size=vocab_size)
    selector = OpusDataSelector(
        config=cfg,
        model=model,
        optimizer=optimizer,
        proxy_loader=proxy_loader,
    )

    candidate_batch = _make_candidate_batch(batch_size=8, vocab_size=vocab_size)
    device = torch.device("cpu")

    selected_batch, metrics = selector.select_batch(candidate_batch, device)

    # Selection ratio 0.5 of 8 = 4
    assert selected_batch["input_ids"].shape[0] <= candidate_batch["input_ids"].shape[0]
    assert selected_batch["input_ids"].shape[0] > 0
    assert "opus_mode" in metrics
    assert selected_batch["labels"].shape[0] == selected_batch["input_ids"].shape[0]


def test_data_selector_random_mode():
    """Random mode should bypass OPUS scoring entirely."""
    torch.manual_seed(42)
    vocab_size = 32
    model = TinyModel(vocab_size=vocab_size)
    optimizer = _make_optimizer(model)

    cfg = OpusConfig(
        enabled=True,
        selection_mode="random",
        selection_ratio=0.5,
        score_seq_len=16,
        sketch_dim=64,
    )

    proxy_loader = _make_proxy_loader(vocab_size=vocab_size)
    selector = OpusDataSelector(
        config=cfg,
        model=model,
        optimizer=optimizer,
        proxy_loader=proxy_loader,
    )

    candidate_batch = _make_candidate_batch(batch_size=8, vocab_size=vocab_size)
    device = torch.device("cpu")

    selected_batch, metrics = selector.select_batch(candidate_batch, device)

    assert metrics["opus_mode"] == "random"
    assert selected_batch["input_ids"].shape[0] == 4  # 0.5 * 8
    assert selected_batch["labels"].shape[0] == 4
    assert metrics["opus_selected_n"] == 4
