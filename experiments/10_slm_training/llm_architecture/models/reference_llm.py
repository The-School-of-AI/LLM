"""
Reference LLM Model
=====================

Model class matching the Test_Code/model_1b.py (Model70B) architecture.

Forward pass structure:
1. input_ids -> embed (standard or kronecker)
2. Expand to streams (SPARSE - only stream 0)
3. ReversibleMidpointStack (all layers + aux_loss collection)
4. Collapse (mean)
5. norm
6. lm_head -> NTP logits
7. If MTP: embed next_token_ids -> MTPTransformerBlock -> norm -> lm_head -> MTP logits
8. Return (logits_ntp, logits_mtp, total_aux_loss)

Key differences from LLM class:
- Sparse stream initialization (only stream 0, not all streams)
- ReversibleMidpointStack instead of sequential layer list
- Hybrid DeltaNet+GSA layer pattern (75%/25%)
- Full transformer MTP block (not linear projections)
- Single shared lm_head for both NTP and MTP
- No explicit causal mask (DeltaNet is inherently causal, GSA builds its own)
- Returns tuple (logits_ntp, logits_mtp, aux_loss) not LLMOutput dataclass
"""

import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")

from components.connections.mhc_v2 import MHCCoeffsV2, RMSNorm
from components.ffn.moe_ffn import MoEFFN, MoEGate
from components.integration.reversible_midpoint import ReversibleMidpointStack
from config.model_config import ModelConfig
from layers.lightning_decoder import LightningDecoderLayer, MTPTransformerBlock


@dataclass
class ReferenceLLMOutput:
    """Output container for ReferenceLLM forward pass."""

    logits_ntp: torch.Tensor = None
    logits_mtp: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    loss_dict: Optional[Dict[str, torch.Tensor]] = None


class ReferenceLLM(nn.Module):
    """
    Reference LLM matching Test_Code/model_1b.py architecture.

    Supports:
    - Standard (nn.Embedding) or Kronecker product embeddings
    - Hybrid DeltaNet (75%) + GSA (25%) attention
    - mHC V2 connections (norm inside)
    - Reversible Midpoint Integration
    - Full transformer MTP block
    - Single shared lm_head

    Usage:
        config = get_preset_config("1b-reference")
        model = ReferenceLLM(config)

        # Training
        logits_ntp, logits_mtp, aux_loss = model(input_ids, next_token_ids=next_ids, return_loss=True)

        # Inference
        logits_ntp, logits_mtp = model(input_ids)
    """

    def __init__(
        self,
        config: ModelConfig,
        embedding_type: str = "standard",
        bpe_vocab=None,
        pf_codec=None,
    ):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.n_streams = int(config.connection.mhc_expansion_rate)
        self.embedding_type_str = embedding_type.lower()

        # ================================================================
        # Embeddings
        # ================================================================
        if self.embedding_type_str == "kronecker":
            if bpe_vocab is None or pf_codec is None:
                raise ValueError(
                    "bpe_vocab and pf_codec required for Kronecker embeddings"
                )

            from components.embeddings.kronecker_embedding import (
                PureHybridEmbeddingTorch,
            )

            self.kronecker_embeddings = PureHybridEmbeddingTorch(
                bpe_vocab, pf_codec
            ).module()
            D_pf = pf_codec.D
            self.pf_to_model = nn.Linear(D_pf, config.hidden_size, bias=False)
            self.embed_norm = RMSNorm(config.hidden_size)
            self.token_embed = None
            self.use_kronecker = True
            self._D_pf = D_pf
        else:
            self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
            self.kronecker_embeddings = None
            self.pf_to_model = None
            self.embed_norm = None
            self.use_kronecker = False

        # ================================================================
        # Build hybrid layer stack: 75% DeltaNet + 25% GSA
        # ================================================================
        num_deltanet = config.num_deltanet_layers
        layers = []
        layer_types = []

        for i in range(config.num_hidden_layers):
            if i < num_deltanet:
                layer_type = "deltanet"
            else:
                layer_type = "gsa"

            layers.append(LightningDecoderLayer(config, layer_type))
            layer_types.append(layer_type)

        self.layers = nn.ModuleList(layers)
        self.layer_types = layer_types

        # ================================================================
        # Integration: Reversible Midpoint or Sequential
        # ================================================================
        int_config = config.integration
        self.use_reversible = int_config.use_reversible

        if self.use_reversible:
            self.stack = ReversibleMidpointStack(
                self.layers,
                step_size=int_config.step_size,
                a=int_config.a,
                noise_eps=int_config.noise_eps,
                bootstrap=int_config.bootstrap,
            )
        else:
            self.stack = None

        # ================================================================
        # Final normalization
        # ================================================================
        self.norm = RMSNorm(config.hidden_size)

        # ================================================================
        # MTP Block (Full Transformer)
        # ================================================================
        if config.head.use_multi_token_prediction:
            self.mtp_block = MTPTransformerBlock(config)
        else:
            self.mtp_block = None

        # ================================================================
        # Output projection (shared between NTP and MTP)
        # ================================================================
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        # ================================================================
        # Initialize weights
        # ================================================================
        self.apply(self._init_weights)

        # Re-initialize Kronecker projection for scale matching
        if self.use_kronecker and self.pf_to_model is not None:
            pf_to_model_std = 0.02 / math.sqrt(self._D_pf)
            self.pf_to_model.weight.data.normal_(mean=0.0, std=pf_to_model_std)

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print("\nReferenceLLM initialized:")
        print(f"  Vocabulary: {self.vocab_size:,}")
        print(f"  Hidden Size: {config.hidden_size}")
        print(f"  Total Layers: {config.num_hidden_layers}")
        print(
            f"  - DeltaNet: {num_deltanet} layers ({num_deltanet/config.num_hidden_layers*100:.0f}%)"
        )
        print(
            f"  - GSA: {config.num_gsa_layers} layers ({config.num_gsa_layers/config.num_hidden_layers*100:.0f}%)"
        )
        print(f"  Streams: {self.n_streams}")
        print(
            f"  MTP: {'Enabled (Full Transformer)' if self.mtp_block else 'Disabled'}"
        )
        print(f"  Embedding: {'Kronecker' if self.use_kronecker else 'Standard'}")
        if self.use_reversible:
            print(
                f"  Integration: Reversible Midpoint (step={int_config.step_size}, a={int_config.a})"
            )
        else:
            print("  Integration: Sequential (direct forward, no reversible overhead)")
        print(f"  Total Parameters: {total_params:,} (~{total_params/1e9:.2f}B)")

    def _init_weights(self, module):
        """Initialize weights."""
        # Skip modules that handle their own initialization
        if self.use_kronecker and self.kronecker_embeddings is not None:
            for name, param in self.kronecker_embeddings.named_modules():
                if module is param:
                    return

        if isinstance(module, (MoEFFN, MoEGate, MHCCoeffsV2)):
            return

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _embed(self, token_ids):
        """Embed token IDs using the configured embedding type."""
        if self.use_kronecker:
            EMB = self.kronecker_embeddings(token_ids)
            dtype_target = self.pf_to_model.weight.dtype
            x = self.pf_to_model(EMB.to(dtype=dtype_target))
            x = self.embed_norm(x)
        else:
            x = self.token_embed(token_ids)
        return x

    def forward(
        self,
        input_ids,
        next_token_ids=None,
        attention_mask=None,
        labels=None,
        return_loss=False,
    ):
        """
        Forward pass with Multi-Token Prediction.

        Args:
            input_ids: [B, T] - Input token IDs
            next_token_ids: [B, T] - Optional for MTP (t+1 tokens)
            attention_mask: Optional attention mask
            labels: [B, T] - Target token IDs for loss computation
            return_loss: Whether to return auxiliary loss

        Returns:
            If return_loss=True or labels provided:
                ReferenceLLMOutput with logits, loss, etc.
            Otherwise:
                (logits_ntp, logits_mtp) tuple
        """
        batch_size, seq_len = input_ids.size()

        # 1. Embed tokens
        x = self._embed(input_ids)

        # 2. SPARSE stream initialization (only stream 0 gets input)
        B, T, D = x.shape
        x_stream = torch.zeros(B, T, self.n_streams, D, device=x.device, dtype=x.dtype)
        x_stream[:, :, 0, :] = x

        # 3. Pass through layer stack (reversible or sequential)
        if self.use_reversible and self.stack is not None:
            x_stream, total_aux_loss = self.stack(x_stream)
        else:
            # Sequential forward: iterate layers directly (1x forward per layer)
            total_aux_loss = x_stream.new_zeros((), dtype=torch.float32)
            for layer in self.layers:
                x_stream, aux = layer(x_stream, attention_mask=attention_mask)
                if aux is not None:
                    total_aux_loss = total_aux_loss + aux

        # 4. Collapse streams by mean
        h_main = x_stream.mean(dim=2)

        # 5. Final normalization
        h_main = self.norm(h_main)

        # 6. NTP Prediction
        logits_ntp = self.lm_head(h_main)

        # 7. MTP Prediction
        logits_mtp = None
        if self.mtp_block is not None and next_token_ids is not None:
            min_len = min(h_main.size(1), next_token_ids.size(1))
            h_use = h_main[:, :min_len, :]
            next_ids_use = next_token_ids[:, :min_len]

            next_emb = self._embed(next_ids_use)

            h_mtp = self.mtp_block(h_use, next_emb, attention_mask=None)
            logits_mtp = self.lm_head(self.norm(h_mtp))

        # 8. Compute loss if labels provided
        loss = None
        loss_dict = None

        if labels is not None:
            # Main NTP loss
            shift_logits = logits_ntp[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            loss_dict = {"main_loss": main_loss, "aux_loss": total_aux_loss}
            loss = main_loss

            # MTP loss
            if logits_mtp is not None and labels.size(1) > 2:
                mtp_shift_logits = logits_mtp[..., :-1, :].contiguous()
                # MTP predicts t+2, so shift labels by 2
                mtp_shift_labels = labels[..., 2:].contiguous()
                min_len = min(mtp_shift_logits.size(1), mtp_shift_labels.size(1))
                if min_len > 0:
                    mtp_loss = F.cross_entropy(
                        mtp_shift_logits[:, :min_len].reshape(
                            -1, mtp_shift_logits.size(-1)
                        ),
                        mtp_shift_labels[:, :min_len].reshape(-1),
                        ignore_index=-100,
                    )
                    mtp_weight = self.config.head.mtp_loss_weight
                    loss = loss + mtp_weight * mtp_loss
                    loss_dict["mtp_loss"] = mtp_loss

            # Add auxiliary loss
            loss = loss + total_aux_loss
            loss_dict["total_loss"] = loss

            return ReferenceLLMOutput(
                logits_ntp=logits_ntp,
                logits_mtp=logits_mtp,
                aux_loss=total_aux_loss,
                loss=loss,
                loss_dict=loss_dict,
            )

        if return_loss:
            return logits_ntp, logits_mtp, total_aux_loss
        return logits_ntp, logits_mtp

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate text autoregressively.

        Note: KV caching is not supported for DeltaNet (recurrent state).
        Each generation step runs full forward pass.
        """
        generated = input_ids

        for _ in range(max_new_tokens):
            logits_ntp, _ = self.forward(generated)
            logits = logits_ntp[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None:
                if (next_token.squeeze(-1) == eos_token_id).all():
                    break

        return generated

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration info."""
        return {
            "name": self.config.model_name,
            "parameters": self.num_parameters,
            "parameters_billions": self.num_parameters / 1e9,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_hidden_layers,
            "num_deltanet_layers": self.config.num_deltanet_layers,
            "num_gsa_layers": self.config.num_gsa_layers,
            "vocab_size": self.config.vocab_size,
            "max_position": self.config.max_position_embeddings,
            "n_streams": self.n_streams,
            "embedding_type": self.embedding_type_str,
            "mtp_enabled": self.mtp_block is not None,
            "reversible": self.use_reversible,
        }
