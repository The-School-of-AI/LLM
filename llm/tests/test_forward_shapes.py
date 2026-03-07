"""
Test ① — Model Forward Pass Produces Correct Output Shapes.

Verifies that Model1B.forward() returns tensors with the expected shapes
and finite values across all 8 return modes (return_loss × return_memory × return_hidden).

Requirements:
    - CUDA GPU (GatedDeltaNet hard-requires FLA + CUDA)
    - flash-linear-attention package installed
    - triton installed (Linux only)

Run:
    python -m pytest tests/test_forward_shapes.py -v
"""

import pytest
import torch
from llm.models.recurrence_model_1b import Model1B, ModelConfig

# ── Constants ────────────────────────────────────────────────────────────────
B = 2   # batch size
T = 32  # sequence length
V = 256 # vocab size
H = 64  # hidden size


def _mini_config(enable_mtp: bool = True) -> ModelConfig:
    """Shared mini config builder. Tiny model (~1MB) for fast shape testing."""
    return ModelConfig(
        vocab_size=V,
        hidden_size=H,
        num_layers=4,              # minimum for DDDG pattern
        num_deltanet_layers=3,
        num_gsa_layers=1,
        delta_v_heads=2,           # H / head_dim = 64 / 32
        delta_head_dim=32,
        delta_gate_dim=32,
        gsa_num_heads=2,           # H / head_dim = 64 / 32
        gsa_head_dim=32,
        gsa_k_base=8,
        gsa_k_min=4,
        gsa_k_max=16,
        gsa_indexer_heads=2,
        n_streams=2,
        max_seq_len=512,
        enable_mtp=enable_mtp,
        shared_expert_intermediate_size=128,
        require_fused_deltanet_kernel=True,
        require_fused_gsa_kernel=True,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_and_inputs():
    """
    Create a mini Model1B **once** and reuse across all tests in this module.

    embedding_type="standard" uses nn.Embedding — no bpe_vocab/pf_codec needed.
    Kronecker embedding path is tested separately in tokenizer tests, not here.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required — GatedDeltaNet has no CPU fallback")

    device = torch.device("cuda")
    cfg = _mini_config(enable_mtp=True)
    model = Model1B(cfg, embedding_type="standard").to(device).eval()

    input_ids = torch.randint(0, V, (B, T), device=device)
    next_ids = torch.randint(0, V, (B, T), device=device)
    attn_mask = torch.ones(B, T, dtype=torch.long, device=device)

    return model, cfg, input_ids, next_ids, attn_mask


# ── Mode 1: return_loss=False, return_memory=False ───────────────────────────

class TestReturnLogits:
    """Modes 1 & 2: (return_loss=False, return_memory=False)."""

    def test_logits_no_mtp(self, model_and_inputs):
        """NTP logits [B,T,V], MTP is None when next_token_ids omitted."""
        model, cfg, input_ids, _, _ = model_and_inputs
        with torch.no_grad():
            ntp, mtp = model(
                input_ids,
                next_token_ids=None,
                return_loss=False,
                return_memory=False,
                return_hidden=False,
            )
        assert ntp.shape == (B, T, V), f"Expected {(B, T, V)}, got {ntp.shape}"
        assert torch.isfinite(ntp).all(), "NTP logits contain NaN or inf"
        assert mtp is None, "MTP must be None when next_token_ids is None"

    def test_logits_with_mtp(self, model_and_inputs):
        """Both NTP and MTP logits [B,T,V] when next_token_ids provided."""
        model, cfg, input_ids, next_ids, _ = model_and_inputs
        with torch.no_grad():
            ntp, mtp = model(
                input_ids,
                next_token_ids=next_ids,
                return_loss=False,
                return_memory=False,
                return_hidden=False,
            )
        assert ntp.shape == (B, T, V), f"NTP: expected {(B, T, V)}, got {ntp.shape}"
        assert torch.isfinite(ntp).all(), "NTP logits contain NaN or inf"
        assert mtp is not None, "MTP should not be None when next_token_ids provided"
        assert mtp.shape == (B, T, V), f"MTP: expected {(B, T, V)}, got {mtp.shape}"
        assert torch.isfinite(mtp).all(), "MTP logits contain NaN or inf"

    def test_hidden_no_mtp(self, model_and_inputs):
        """NTP hidden [B,T,H] with return_hidden=True, MTP is None."""
        model, cfg, input_ids, _, _ = model_and_inputs
        with torch.no_grad():
            ntp, mtp = model(
                input_ids,
                next_token_ids=None,
                return_loss=False,
                return_memory=False,
                return_hidden=True,
            )
        assert ntp.shape == (B, T, H), f"Expected {(B, T, H)}, got {ntp.shape}"
        assert torch.isfinite(ntp).all(), "NTP hidden contains NaN or inf"
        assert mtp is None, "MTP must be None when next_token_ids is None"

    def test_hidden_with_mtp(self, model_and_inputs):
        """Both NTP and MTP hidden [B,T,H] with return_hidden=True."""
        model, cfg, input_ids, next_ids, _ = model_and_inputs
        with torch.no_grad():
            ntp, mtp = model(
                input_ids,
                next_token_ids=next_ids,
                return_loss=False,
                return_memory=False,
                return_hidden=True,
            )
        assert ntp.shape == (B, T, H), f"NTP: expected {(B, T, H)}, got {ntp.shape}"
        assert torch.isfinite(ntp).all(), "NTP hidden contains NaN or inf"
        assert mtp is not None, "MTP should not be None"
        assert mtp.shape == (B, T, H), f"MTP: expected {(B, T, H)}, got {mtp.shape}"
        assert torch.isfinite(mtp).all(), "MTP hidden contains NaN or inf"


# ── Mode 3/4: return_memory=True ────────────────────────────────────────────

class TestReturnMemory:
    """Modes 3 & 4: (return_loss=False, return_memory=True)."""

    def test_memory_shape_and_detach(self, model_and_inputs):
        """Memory is [B,H], detached (no grad), and finite."""
        model, cfg, input_ids, next_ids, _ = model_and_inputs
        with torch.no_grad():
            # Unpack order: ntp, mtp, memory
            ntp, mtp, mem = model(
                input_ids,
                next_token_ids=next_ids,
                return_loss=False,
                return_memory=True,
                return_hidden=True,
            )
        assert ntp.shape == (B, T, H)
        assert torch.isfinite(ntp).all(), "NTP hidden contains NaN or inf"
        assert mtp is not None and mtp.shape == (B, T, H)
        assert torch.isfinite(mtp).all(), "MTP hidden contains NaN or inf"

        # Memory-specific checks
        assert mem.shape == (B, H), f"Memory: expected {(B, H)}, got {mem.shape}"
        assert not mem.requires_grad, "Memory should be detached (no grad)"
        assert torch.isfinite(mem).all(), "Memory contains NaN or inf"


# ── Mode 5/6: return_loss=True ──────────────────────────────────────────────

class TestReturnLoss:
    """Modes 5 & 6: (return_loss=True, return_memory=False)."""

    def test_aux_loss_is_scalar(self, model_and_inputs):
        """aux_loss is a finite scalar tensor."""
        model, cfg, input_ids, next_ids, _ = model_and_inputs
        with torch.no_grad():
            # Unpack order: ntp, mtp, aux_loss
            ntp, mtp, aux_loss = model(
                input_ids,
                next_token_ids=next_ids,
                return_loss=True,
                return_memory=False,
                return_hidden=True,
            )
        assert ntp.shape == (B, T, H)
        assert torch.isfinite(ntp).all()
        assert mtp is not None and mtp.shape == (B, T, H)
        assert torch.isfinite(mtp).all()

        # Aux loss checks
        assert isinstance(aux_loss, torch.Tensor), "aux_loss must be a tensor"
        assert aux_loss.dim() == 0, f"aux_loss must be scalar, got dim={aux_loss.dim()}"
        assert torch.isfinite(aux_loss), "aux_loss is NaN or inf"


# ── Mode 7/8: return_loss=True + return_memory=True ─────────────────────────

class TestAllFlags:
    """Modes 7 & 8: (return_loss=True, return_memory=True)."""

    def test_all_flags_on(self, model_and_inputs):
        """4-tuple: (h_ntp, h_mtp, aux_loss, memory) — exact unpack order."""
        model, cfg, input_ids, next_ids, attn_mask = model_and_inputs
        with torch.no_grad():
            # EXACT unpack order: ntp, mtp, aux_loss, memory
            # NOT: ntp, mtp, memory, aux_loss ← WRONG
            h_ntp, h_mtp, aux_loss, memory_out = model(
                input_ids,
                next_token_ids=next_ids,
                attention_mask=attn_mask,
                return_loss=True,
                return_memory=True,
                return_hidden=True,
            )

        # NTP
        assert h_ntp.shape == (B, T, H), f"h_ntp: expected {(B, T, H)}, got {h_ntp.shape}"
        assert torch.isfinite(h_ntp).all(), "h_ntp contains NaN or inf"

        # MTP
        assert h_mtp is not None, "h_mtp should not be None"
        assert h_mtp.shape == (B, T, H), f"h_mtp: expected {(B, T, H)}, got {h_mtp.shape}"
        assert torch.isfinite(h_mtp).all(), "h_mtp contains NaN or inf"

        # Aux loss
        assert aux_loss.dim() == 0, f"aux_loss should be scalar, got dim={aux_loss.dim()}"
        assert torch.isfinite(aux_loss), "aux_loss is NaN or inf"

        # Memory
        assert memory_out.shape == (B, H), f"memory: expected {(B, H)}, got {memory_out.shape}"
        assert not memory_out.requires_grad, "memory should be detached"
        assert torch.isfinite(memory_out).all(), "memory contains NaN or inf"


# ── Special cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    """MTP disabled, memory injection, attention mask, mismatched lengths."""

    def test_mtp_disabled(self):
        """
        Separate model with enable_mtp=False.
        Cannot reuse shared fixture — that model has enable_mtp=True.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        device = torch.device("cuda")
        cfg_no_mtp = _mini_config(enable_mtp=False)
        model = Model1B(cfg_no_mtp, embedding_type="standard").to(device).eval()

        input_ids = torch.randint(0, V, (B, T), device=device)
        next_ids = torch.randint(0, V, (B, T), device=device)

        with torch.no_grad():
            ntp, mtp = model(
                input_ids,
                next_token_ids=next_ids,  # provided but should be ignored
                return_loss=False,
                return_memory=False,
                return_hidden=False,
            )

        assert ntp.shape == (B, T, V), f"Expected {(B, T, V)}, got {ntp.shape}"
        assert torch.isfinite(ntp).all(), "NTP contains NaN or inf"
        assert mtp is None, "MTP must be None when enable_mtp=False, even with next_token_ids"

    def test_prev_memory_injection(self, model_and_inputs):
        """Passing prev_memory_stream [B,H] should not change output shapes."""
        model, cfg, input_ids, next_ids, attn_mask = model_and_inputs
        prev_mem = torch.randn(B, H, device=input_ids.device, dtype=torch.bfloat16)

        with torch.no_grad():
            h_ntp, h_mtp, aux_loss, memory_out = model(
                input_ids,
                next_token_ids=next_ids,
                attention_mask=attn_mask,
                prev_memory_stream=prev_mem,
                return_loss=True,
                return_memory=True,
                return_hidden=True,
            )

        assert h_ntp.shape == (B, T, H)
        assert torch.isfinite(h_ntp).all(), "h_ntp NaN/inf with memory injection"
        assert h_mtp is not None and h_mtp.shape == (B, T, H)
        assert torch.isfinite(h_mtp).all(), "h_mtp NaN/inf with memory injection"
        assert aux_loss.dim() == 0
        assert memory_out.shape == (B, H)
        assert torch.isfinite(memory_out).all()

    def test_attention_mask(self, model_and_inputs):
        """Passing attention_mask [B,T] should not change output shapes."""
        model, cfg, input_ids, next_ids, attn_mask = model_and_inputs

        # Mask out second half of sequence
        mask = attn_mask.clone()
        mask[:, T // 2 :] = 0

        with torch.no_grad():
            h_ntp, h_mtp, aux_loss, memory_out = model(
                input_ids,
                next_token_ids=next_ids,
                attention_mask=mask,
                return_loss=True,
                return_memory=True,
                return_hidden=True,
            )

        assert h_ntp.shape == (B, T, H)
        assert torch.isfinite(h_ntp).all(), "h_ntp NaN/inf with partial mask"
        assert h_mtp is not None and h_mtp.shape == (B, T, H)
        assert torch.isfinite(h_mtp).all(), "h_mtp NaN/inf with partial mask"
        assert aux_loss.dim() == 0
        assert memory_out.shape == (B, H)
        assert not memory_out.requires_grad, "memory should be detached"

    def test_mtp_mismatched_length(self, model_and_inputs):
        """
        next_token_ids shorter than input_ids → MTP uses min_len.
        NTP shape should be unchanged; MTP shape should be [B, T//2, V].
        """
        model, cfg, input_ids, next_ids, _ = model_and_inputs
        short_next = next_ids[:, : T // 2]  # half-length

        with torch.no_grad():
            ntp, mtp = model(
                input_ids,
                next_token_ids=short_next,
                return_loss=False,
                return_memory=False,
                return_hidden=False,
            )

        # NTP is full length — model doesn't slice based on next_token_ids
        assert ntp.shape == (B, T, V), f"NTP: expected {(B, T, V)}, got {ntp.shape}"
        assert torch.isfinite(ntp).all(), "NTP contains NaN or inf"

        # MTP is truncated to min(T, next_ids.shape[1]) = T//2
        assert mtp is not None, "MTP should not be None"
        assert mtp.shape == (B, T // 2, V), f"MTP: expected {(B, T // 2, V)}, got {mtp.shape}"
        assert torch.isfinite(mtp).all(), "MTP contains NaN or inf"
