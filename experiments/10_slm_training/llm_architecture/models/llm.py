"""
Complete LLM Model
==================

Full language model combining all components:
- Token embeddings
- Position embeddings (RoPE/YaRN)
- Transformer layers (configurable attention, FFN, connections)
- Output heads (standard or multi-token prediction)

Modular, configuration-driven architecture supporting:
- GQA, GSA, DeepSeek Sparse attention
- SwiGLU FFN
- Residual or mHC connections
- YaRN for extended context
- Multi-token prediction
"""

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")


# Components
from components.embeddings.token_embedding import TokenEmbedding
from components.normalization.rms_norm import RMSNorm

# Use Triton-optimized version when available
try:
    from components.kernels.triton_normalization import TritonRMSNorm

    USE_TRITON_NORM = True
except ImportError:
    USE_TRITON_NORM = False
    TritonRMSNorm = RMSNorm  # Fallback
from components.attention.grouped_query_attention import create_causal_mask
from components.heads.multi_token_head import LMHead, MTPLoss, MultiTokenPredictionHead

# Config
from config.model_config import (
    AttentionType,
    ConnectionType,
    EmbeddingType,
    ModelConfig,
    get_preset_config,
)

# Layers
from layers.transformer_block import TransformerBlockList


@dataclass
class LLMOutput:
    """Output container for LLM forward pass."""

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    aux_logits: Optional[List[torch.Tensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    loss_dict: Optional[Dict[str, torch.Tensor]] = None


class LLM(nn.Module):
    """
    Complete Language Model.

    Architecture:
    1. Token embedding
    2. Transformer layers (with configurable components)
    3. Final norm
    4. LM head (standard or MTP)

    Usage:
        config = get_preset_config("1b-base")
        model = LLM(config)

        # Training
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        # Inference
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
        )

        # Transformer layers
        self.layers = TransformerBlockList(config)

        # Final normalization
        self.norm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output head - always untied (following DeepSeek V3)
        # This ensures consistent behavior as model scales (only FFN grows)
        if config.head.use_multi_token_prediction:
            self.lm_head = MultiTokenPredictionHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                num_predict_tokens=config.head.num_predict_tokens,
                init_std=config.initializer_range,
            )
            self.mtp_loss = MTPLoss(
                num_predict_tokens=config.head.num_predict_tokens,
                aux_loss_weight=config.head.mtp_loss_weight,
            )
        else:
            self.lm_head = LMHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                init_std=config.initializer_range,
            )
            self.mtp_loss = None

        # Initialize weights
        self.apply(self._init_weights)

        # Model info
        self._num_parameters = sum(p.numel() for p in self.parameters())

        print(f"Initialized {config.model_name}")
        print(f"  Parameters: {self._num_parameters / 1e9:.2f}B")
        print(f"  Attention: {config.attention.attention_type.value}")
        print(f"  Connection: {config.connection.connection_type.value}")
        print(f"  Position: {config.position.position_type.value}")
        print(f"  MTP: {config.head.use_multi_token_prediction}")

        # HF-style compatibility flag used by training scripts.
        self.gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )

    def get_input_embeddings(self) -> nn.Module:
        """Get input embedding layer."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        """Set input embedding layer."""
        self.embed_tokens = value

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Enable gradient checkpointing on supported submodules.

        Returns:
            Number of submodules toggled.
        """
        enabled = 0

        for module in self.modules():
            if module is self:
                continue

            enable_fn = getattr(module, "gradient_checkpointing_enable", None)
            if callable(enable_fn):
                try:
                    enable_fn(
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                    )
                except TypeError:
                    enable_fn()
                enabled += 1
                continue

            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                enabled += 1

        self.gradient_checkpointing = enabled > 0
        return enabled

    def gradient_checkpointing_disable(self) -> int:
        """
        Disable gradient checkpointing on supported submodules.

        Returns:
            Number of submodules toggled.
        """
        disabled = 0

        for module in self.modules():
            if module is self:
                continue

            disable_fn = getattr(module, "gradient_checkpointing_disable", None)
            if callable(disable_fn):
                disable_fn()
                disabled += 1
                continue

            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
                disabled += 1

        self.gradient_checkpointing = False
        return disabled

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_length: int,
        past_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare combined causal + attention mask."""
        # Create causal mask
        total_length = seq_length + past_length
        causal_mask = create_causal_mask(total_length, device, dtype)

        # If we have past KV, only look at new positions
        if past_length > 0:
            causal_mask = causal_mask[:, :, -seq_length:, :]

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, total_length] -> [batch, 1, 1, total_length]
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            padding_mask = (1.0 - padding_mask.to(dtype)) * torch.finfo(dtype).min
            causal_mask = causal_mask + padding_mask

        return causal_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[LLMOutput, Tuple]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            past_key_values: KV cache for inference
            labels: Target token IDs for loss computation
            use_cache: Return KV cache
            output_attentions: Return attention weights
            output_hidden_states: Return all hidden states
            return_dict: Return LLMOutput dataclass

        Returns:
            LLMOutput with loss, logits, caches, etc.
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Get dtype from embeddings

        # Past length for KV cache
        past_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        # Position IDs
        if position_ids is None:
            position_ids = (
                torch.arange(past_length, past_length + seq_length, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Prepare attention mask
        causal_mask = self._prepare_attention_mask(
            attention_mask,
            batch_size,
            seq_length,
            past_length,
            device,
            hidden_states.dtype,
        )

        # Forward through transformer layers
        layer_outputs = self.layers(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = layer_outputs[0]
        next_cache = layer_outputs[1] if use_cache else None
        all_hidden_states = layer_outputs[2]
        all_attentions = layer_outputs[3]

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head
        # Always return aux_logits when MTP is enabled, so external loss
        # computation (used by torch.compile) can access them.
        if self.config.head.use_multi_token_prediction:
            logits, aux_logits = self.lm_head(hidden_states, return_aux=True)
        else:
            logits = self.lm_head(hidden_states)
            aux_logits = None

        # Compute loss
        loss = None
        loss_dict = None

        if labels is not None:
            if self.mtp_loss is not None:
                loss, loss_dict = self.mtp_loss(logits, aux_logits, labels)
            else:
                # Standard cross-entropy loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss_dict = {"loss": loss}

        if not return_dict:
            outputs = (logits,)
            if loss is not None:
                outputs = (loss,) + outputs
            if use_cache:
                outputs += (next_cache,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            return outputs

        return LLMOutput(
            loss=loss,
            logits=logits,
            aux_logits=aux_logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            loss_dict=loss_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Initial token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Use sampling vs greedy
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID

        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # KV cache for efficiency
        past_key_values = None

        # Track which sequences are done
        unfinished = torch.ones(batch_size, dtype=torch.long, device=device)

        generated = input_ids

        for _ in range(max_new_tokens):
            # Forward pass with cache
            outputs = self.forward(
                input_ids=(
                    generated[:, -1:] if past_key_values is not None else generated
                ),
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # [batch, vocab]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
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

            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Update generated sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if eos_token_id is not None:
                unfinished = (
                    unfinished * (next_token.squeeze(-1) != eos_token_id).long()
                )
                if unfinished.sum() == 0:
                    break

        return generated

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return self._num_parameters

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration info."""
        return {
            "name": self.config.model_name,
            "parameters": self._num_parameters,
            "parameters_billions": self._num_parameters / 1e9,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_hidden_layers,
            "num_heads": self.config.attention.num_attention_heads,
            "num_kv_heads": self.config.attention.num_key_value_heads,
            "vocab_size": self.config.vocab_size,
            "max_position": self.config.max_position_embeddings,
            "attention_type": self.config.attention.attention_type.value,
            "connection_type": self.config.connection.connection_type.value,
            "position_type": self.config.position.position_type.value,
            "mtp_enabled": self.config.head.use_multi_token_prediction,
        }


def _embedding_type_to_str(
    config: ModelConfig,
    embedding_type: Optional[Union[str, EmbeddingType]] = None,
) -> str:
    """Resolve embedding type string from explicit arg or config."""
    if embedding_type is not None:
        if isinstance(embedding_type, EmbeddingType):
            return embedding_type.value
        return str(embedding_type).lower()

    embedding_cfg = getattr(config, "embedding", None)
    if embedding_cfg is None:
        return EmbeddingType.STANDARD.value

    cfg_type = getattr(embedding_cfg, "embedding_type", EmbeddingType.STANDARD)
    if isinstance(cfg_type, EmbeddingType):
        return cfg_type.value
    return str(cfg_type).lower()


def build_kronecker_artifacts_from_tokenizer(
    tokenizer: Any,
    config: ModelConfig,
) -> Tuple[List[str], Any]:
    """
    Build Kronecker embedding artifacts from a tokenizer.

    Returns:
        (bpe_vocab, pf_codec)
    """
    from components.embeddings.kronecker_embedding import (
        KroneckerConfig,
        KroneckerEmbeddings,
    )

    vocab_size = config.vocab_size
    pf_dim = config.embedding.kronecker_pf_dim

    bpe_vocab = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in range(vocab_size)
    ]
    pf_cfg = KroneckerConfig(D=pf_dim)
    pf_codec = KroneckerEmbeddings(pf_cfg)
    return bpe_vocab, pf_codec


def _requires_reference_model(config: ModelConfig) -> bool:
    """Route configs requiring Test_Code reference architecture to ReferenceLLM."""
    use_reversible = bool(
        getattr(getattr(config, "integration", None), "use_reversible", False)
    )
    attn_type = getattr(getattr(config, "attention", None), "attention_type", None)
    conn_type = getattr(getattr(config, "connection", None), "connection_type", None)

    return (
        use_reversible
        or attn_type in {AttentionType.GATED_DELTANET, AttentionType.REFERENCE_GSA}
        or conn_type == ConnectionType.MHC_V2
    )


def create_model_from_config(
    config: ModelConfig,
    embedding_type: Optional[Union[str, EmbeddingType]] = None,
    bpe_vocab: Optional[List[str]] = None,
    pf_codec: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> nn.Module:
    """
    Create model from ModelConfig with architecture-aware routing.

    Routes to ReferenceLLM for reversible/Test_Code-compatible configs.
    """
    resolved_embedding_type = _embedding_type_to_str(config, embedding_type)
    use_reference = _requires_reference_model(config)

    if use_reference:
        from models.reference_llm import ReferenceLLM

        if resolved_embedding_type == EmbeddingType.KRONECKER.value:
            if bpe_vocab is None or pf_codec is None:
                if tokenizer is None:
                    raise ValueError(
                        "Kronecker embedding requires tokenizer or explicit bpe_vocab/pf_codec."
                    )
                bpe_vocab, pf_codec = build_kronecker_artifacts_from_tokenizer(
                    tokenizer, config
                )

        return ReferenceLLM(
            config,
            embedding_type=resolved_embedding_type,
            bpe_vocab=bpe_vocab,
            pf_codec=pf_codec,
        )

    if resolved_embedding_type != EmbeddingType.STANDARD.value:
        raise ValueError(
            f"Embedding type '{resolved_embedding_type}' is only supported by ReferenceLLM."
        )

    return LLM(config)


def create_model(preset: str = "1b-base", **override_kwargs) -> nn.Module:
    """
    Factory function to create model from preset.

    Args:
        preset: Preset name ("1b-base", "1b-gsa", "1b-deepseek", "1b-reference", etc.)
        **override_kwargs: Override specific config values

    Returns:
        Initialized model (LLM or ReferenceLLM)
    """
    config = get_preset_config(preset)

    # Apply overrides
    for key, value in override_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return create_model_from_config(
        config,
        embedding_type=override_kwargs.get("embedding_type"),
        bpe_vocab=override_kwargs.get("bpe_vocab"),
        pf_codec=override_kwargs.get("pf_codec"),
        tokenizer=override_kwargs.get("tokenizer"),
    )
