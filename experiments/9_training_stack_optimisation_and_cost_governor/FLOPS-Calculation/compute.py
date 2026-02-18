import argparse
import dataclasses
import json
import math
import sys
from typing import Optional


def bytes_for_precision(precision: str) -> float:
    prec = str(precision).strip().lower()
    mapping = {
        "fp32": 4,
        "f32": 4,
        "bf16": 2,
        "fp16": 2,
        "f16": 2,
        "fp8": 1,
        "int8": 1,
        "i8": 1,
        "nvfp4": 0.5,
        "int4": 0.5,
        "i4": 0.5,
    }
    return mapping.get(prec, 4)


def parse_attention_type(attention_str: str, num_heads: int = 32) -> dict:
    """
    Parse attention type string into normalized config.

    Supports 5 attention types with optional ratio/parameter notation:

    | Type | Format | Example | Description |
    |------|--------|---------|-------------|
    | mha  | mha | "mha" | Standard Multi-Head Attention |
    | gqa  | gqa[:Q:KV] | "gqa:8:1" | Grouped Query Attention (8 Q heads share 1 KV head) |
    | gsa  | gsa[:k] | "gsa:512" | Gated Sparse Attention (attend to k tokens) |
    | dsa  | dsa[:rank] | "dsa:256" | DeepSeek MLA (KV LoRA rank=256) |
    | hybrid | type1-type2:ratio1:ratio2 | "gqa-gsa:4:1" | Hybrid (4 GQA layers per 1 GSA layer) |

    Args:
        attention_str: Attention type string (e.g., "gqa:4:1", "gsa:512", "gqa-gsa:4:1")
        num_heads: Number of attention heads (for validation)

    Returns:
        dict with keys:
            - type: Normalized type ("mha", "gqa", "gsa", "dsa", "hybrid")
            - kv_ratio: KV heads / Q heads (1.0 for MHA, 0.125 for 8:1 GQA)
            - sparse_k: Sparse attention k tokens (None if not GSA)
            - mla_rank: MLA compression rank (0 if not DSA)
            - hybrid_config: (for hybrid only) dict with type1, type2, ratio1, ratio2

    Examples:
        >>> parse_attention_type("mha")
        {'type': 'mha', 'kv_ratio': 1.0, 'sparse_k': None, 'mla_rank': 0, 'hybrid_config': None}

        >>> parse_attention_type("gqa:8:1")  # 8 Q heads share 1 KV head
        {'type': 'gqa', 'kv_ratio': 0.125, 'sparse_k': None, 'mla_rank': 0, 'hybrid_config': None}

        >>> parse_attention_type("gsa:512")  # Sparse attention with k=512
        {'type': 'gsa', 'kv_ratio': 1.0, 'sparse_k': 512, 'mla_rank': 0, 'hybrid_config': None}

        >>> parse_attention_type("gqa-gsa:4:1")  # 4 GQA layers per 1 GSA layer
        {'type': 'hybrid', 'kv_ratio': 1.0, 'sparse_k': None, 'mla_rank': 0,
         'hybrid_config': {'type1': 'gqa', 'type2': 'gsa', 'ratio1': 4, 'ratio2': 1}}
    """
    attention_str = str(attention_str).strip().lower()
    
    # Check for hybrid format: type1-type2:ratio1:ratio2
    if "-" in attention_str:
        parts = attention_str.split(":")
        type_parts = parts[0].split("-")
        if len(type_parts) == 2:
            type1, type2 = type_parts
            ratio1 = int(parts[1]) if len(parts) > 1 else 1
            ratio2 = int(parts[2]) if len(parts) > 2 else 1
            
            # Parse the sparse_k for GSA if specified (e.g., gqa-gsa:4:1:512)
            sparse_k = int(parts[3]) if len(parts) > 3 else 512  # Default sparse_k
            
            # Normalize type aliases
            type_aliases = {
                "gqa": "gqa", "grouped_query": "gqa",
                "gsa": "gsa", "gated_sparse": "gsa",
                "mha": "mha", "standard": "mha",
                "dsa": "dsa", "mla": "dsa", "deepseek_mla": "dsa",
                "deltanet": "deltanet", "delta": "deltanet", 
                "linear": "deltanet", "gated_linear": "deltanet",
            }
            type1 = type_aliases.get(type1, type1)
            type2 = type_aliases.get(type2, type2)
            
            return {
                "type": "hybrid",
                "kv_ratio": 1.0,  # Will be computed per-layer
                "sparse_k": sparse_k if "gsa" in [type1, type2] else None,
                "mla_rank": 0,
                "is_linear": "deltanet" in [type1, type2],  # Has linear attention component
                "hybrid_config": {
                    "type1": type1,
                    "type2": type2,
                    "ratio1": ratio1,
                    "ratio2": ratio2,
                    "layer_weight_type1": ratio1 / (ratio1 + ratio2),  # e.g., 0.75 for 3:1
                    "layer_weight_type2": ratio2 / (ratio1 + ratio2),  # e.g., 0.25 for 3:1
                },
            }
    
    parts = attention_str.split(":")
    attn_type = parts[0]

    # Normalize aliases to canonical types
    type_aliases = {
        # MHA aliases
        "": "mha",
        "normal": "mha",
        "standard": "mha",
        "mha": "mha",
        "multi_head": "mha",
        "multi_head_attention": "mha",
        # GQA aliases
        "gqa": "gqa",
        "grouped_query": "gqa",
        "grouped_query_attention": "gqa",
        # GSA aliases
        "gsa": "gsa",
        "gated_sparse": "gsa",
        "gated_sparse_attention": "gsa",
        "deepseek_gsa": "gsa",
        # DSA/MLA aliases
        "dsa": "dsa",
        "mla": "dsa",
        "deepseek_mla": "dsa",
        "deepseek_sparse": "dsa",
        "deepseek": "dsa",
        # DeltaNet (linear attention) aliases
        "deltanet": "deltanet",
        "delta": "deltanet",
        "linear": "deltanet",
        "gated_linear": "deltanet",
        "gated_deltanet": "deltanet",
    }
    attn_type = type_aliases.get(attn_type, attn_type)

    result = {
        "type": attn_type,
        "kv_ratio": 1.0,      # KV heads / Q heads (1.0 = MHA, 0.125 = 8:1 GQA)
        "sparse_k": None,     # Sparse attention k tokens
        "mla_rank": 0,        # MLA/DSA compression rank
        "is_linear": attn_type == "deltanet",  # Linear attention (O(Ld²) vs O(L²d))
        "hybrid_config": None,  # For hybrid attention
    }

    if attn_type == "gqa" and len(parts) >= 3:
        # Format: gqa:Q:KV where Q heads share KV heads
        # Example: gqa:8:1 means 8 Q heads share 1 KV head → ratio = 1/8 = 0.125
        try:
            q_per_group = int(parts[1])
            kv_per_group = int(parts[2])
            if q_per_group > 0:
                result["kv_ratio"] = kv_per_group / q_per_group
        except (ValueError, ZeroDivisionError):
            pass  # Keep default ratio

    elif attn_type == "gsa" and len(parts) >= 2:
        # Format: gsa:k means sparse attention with k tokens
        # Example: gsa:512 means attend to 512 tokens
        try:
            result["sparse_k"] = int(parts[1])
        except ValueError:
            pass

    elif attn_type == "dsa" and len(parts) >= 2:
        # Format: dsa:rank means MLA with kv_lora_rank
        # Example: dsa:256 means compress KV to rank 256
        try:
            result["mla_rank"] = int(parts[1])
        except ValueError:
            pass

    return result


@dataclasses.dataclass
class TrainingStage:
    name: str
    total_tokens: float
    architecture: dict
    description: str

    def calculate_params(self) -> dict:
        """
        Calculate total and active parameters for the model architecture.

        Handles both dense and MoE architectures with support for:
        - Null expert routing
        - Shared and routed experts
        - Dynamic expert count calculation
        - Tied embeddings

        Returns:
            dict: Parameter counts including:
                - total_params: All model parameters
                - active_params: Parameters used per forward pass
                - active_non_embed_params: Active params excluding embeddings
                - embedding_params: Token embedding parameters
                - lm_head_params: Language model head parameters
                - num_moe_layers: Number of MoE layers
                - dense_layers: Number of dense FFN layers
                - num_experts: Total number of experts
                - derived_num_experts: Calculated expert count (if solve_for mode)
        """
        arch = self.architecture
        attention_cfg = arch.get("attention") or {}
        router_cfg = arch.get("router") or {}
        expert_cfg = arch.get("expert") or {}
        head_cfg = arch.get("head") or {}

        vocab = arch.get("vocab_size", 50257)
        hidden = arch.get("hidden_size", 2048)
        intermediate = arch.get("intermediate_size", 4 * hidden)
        layers = arch.get("num_layers", arch.get("num_hidden_layers", 24))

        attention_type = str(
            arch.get("attention_type", attention_cfg.get("attention_type", ""))
        ).strip().lower()
        if attention_type in ("grouped_query", "grouped_query_attention"):
            attention_type = "gqa"
        if attention_type in ("deepseek_sparse", "deepseek_mla", "mla", "deepseek"):
            attention_type = "deepseek_mla"

        num_heads = arch.get("num_heads", attention_cfg.get("num_attention_heads"))
        num_kv_heads = arch.get(
            "num_kv_heads", attention_cfg.get("num_key_value_heads", num_heads)
        )

        experts = arch.get(
            "num_experts",
            arch.get("num_routed_experts", router_cfg.get("num_routed_experts", 0)),
        )
        num_shared_experts = arch.get(
            "num_shared_experts", router_cfg.get("num_shared_experts", 0)
        )
        num_null_experts = arch.get(
            "num_null_experts", router_cfg.get("num_null_experts", 0)
        )

        top_k = arch.get(
            "top_k_experts", arch.get("top_k", router_cfg.get("top_k", 1))
        )
        use_adaptive_top_k = bool(router_cfg.get("use_adaptive_top_k", False))
        if use_adaptive_top_k:
            top_k = router_cfg.get("top_k_max", top_k)

        null_prob = arch.get("null_expert_prob", router_cfg.get("null_expert_prob", 0.0))
        if "null_expert_prob" not in arch and "null_expert_prob" not in router_cfg:
            data_sparsity = router_cfg.get("data_sparsity", arch.get("data_sparsity"))
            if data_sparsity is not None:
                null_prob = max(0.0, min(1.0, 1.0 - float(data_sparsity)))

        num_moe_layers = arch.get("num_moe_layers")
        if num_moe_layers is None:
            if experts > 0 or num_shared_experts > 0:
                moe_layer_frequency = arch.get(
                    "moe_layer_frequency", arch.get("moe_frequency", 1)
                )
                num_moe_layers = (
                    int(math.ceil(layers / moe_layer_frequency))
                    if moe_layer_frequency
                    else layers
                )
            else:
                num_moe_layers = 0
        tie_embeddings = arch.get("tie_embeddings", True)
        target_total_params = arch.get("target_total_params")
        target_params_per_expert = arch.get("target_params_per_expert")
        solve_for = str(arch.get("solve_for", "")).strip().lower()

        if experts < 0 or top_k < 0:
            raise ValueError("num_experts and top_k_experts must be >= 0.")
        if experts == 0 and num_shared_experts == 0 and solve_for not in (
            "num_experts",
            "num_experts_from_per_expert",
        ):
            num_moe_layers = 0
        if experts > 0 and top_k > experts:
            raise ValueError("top_k_experts cannot exceed num_experts.")
        if num_moe_layers < 0 or num_moe_layers > layers:
            raise ValueError("num_moe_layers must be between 0 and num_layers.")


        # Embedding parameter calculation
        # Support both standard and Kronecker Product Embeddings
        embedding_type = arch.get("embedding_type", "standard")
        
        if embedding_type == "kronecker":
            # Kronecker Product Embeddings
            # Fixed encoding (no trainable params) + trainable projection
            kronecker_cfg = arch.get("kronecker_config", {})
            D = kronecker_cfg.get("D", 8192)  # Default: 256 bytes × 32 positions
            
            # Trainable parameters:
            # 1. pf_to_model: Linear(D, hidden, bias=False)
            pf_to_model_params = D * hidden
            
            # 2. embed_norm: RMSNorm(hidden) 
            embed_norm_params = hidden
            
            embedding_params = pf_to_model_params + embed_norm_params
            
            # Kronecker embeddings cannot be tied (different dimensions)
            # Force untied to ensure lm_head is counted
            tie_embeddings = False
        else:
            # Standard embeddings: vocab × hidden lookup table
            embedding_params = vocab * hidden

        lm_head_multiplier = arch.get(
            "lm_head_multiplier", head_cfg.get("lm_head_multiplier")
        )
        # Check use_multi_token_prediction at arch root first, then head_cfg
        use_mtp = arch.get("use_multi_token_prediction", 
                          head_cfg.get("use_multi_token_prediction", False))
        if lm_head_multiplier is None and use_mtp:
            lm_head_multiplier = arch.get(
                "num_prediction_heads",
                head_cfg.get("num_prediction_heads", head_cfg.get("mtp_heads", 2))
            )
        if lm_head_multiplier is None:
            lm_head_multiplier = 1
        lm_head_params = 0 if tie_embeddings else vocab * hidden * lm_head_multiplier
        include_lm_head_flops = arch.get("include_lm_head_flops", True)
        lm_head_params_for_flops = (
            vocab * hidden * lm_head_multiplier if include_lm_head_flops else 0
        )

        if num_heads is not None and num_kv_heads is not None:
            if num_kv_heads <= 0:
                raise ValueError("num_kv_heads must be > 0.")
            if num_kv_heads > num_heads:
                raise ValueError("num_kv_heads cannot exceed num_heads.")
            if num_heads % num_kv_heads != 0:
                raise ValueError("num_kv_heads must divide num_heads for GQA/MQA.")
            kv_ratio = num_kv_heads / num_heads
            attn_params_per_layer = hidden * hidden * (2 + 2 * kv_ratio)
        else:
            kv_ratio = 1.0
            attn_params_per_layer = 4 * hidden * hidden

        # Check for GSA (including hybrid with GSA component)
        parsed_attn = parse_attention_type(attention_type, num_heads or 32)
        gsa_in_hybrid = (
            parsed_attn.get("type") == "hybrid" and 
            parsed_attn.get("hybrid_config") and 
            "gsa" in [parsed_attn["hybrid_config"].get("type1"), parsed_attn["hybrid_config"].get("type2")]
        )
        gsa_enabled = attention_type in (
            "gsa",
            "gated_sparse",
            "gated_sparse_attention",
            "deepseek_gsa",
        ) or bool(arch.get("use_gsa", False)) or gsa_in_hybrid
        use_sparse_attn = bool(arch.get("use_sparse_attention", False)) or gsa_enabled
        indexer_heads = arch.get(
            "gsa_num_indexer_heads",
            attention_cfg.get(
                "gsa_num_indexer_heads", arch.get("indexer_heads")
            ),
        )
        indexer_dim = arch.get(
            "gsa_indexer_dim", attention_cfg.get("gsa_indexer_dim", arch.get("indexer_dim"))
        )
        if use_sparse_attn and indexer_heads and indexer_dim:
            indexer_params = (
                hidden * indexer_heads * indexer_dim  # W_Iq: H -> ih*id
                + hidden * indexer_dim                # W_Ik: H -> id
                + hidden * indexer_heads              # W_Iw: H -> ih
                + indexer_heads                       # gate_bias
            )
            attn_params_per_layer += indexer_params

        if gsa_enabled:
            use_value_gate = arch.get(
                "gsa_use_value_gate",
                attention_cfg.get("gsa_use_value_gate", True),
            )
            use_output_gate = arch.get(
                "gsa_use_output_gate",
                attention_cfg.get("gsa_use_output_gate", True),
            )
            if use_value_gate:
                attn_params_per_layer += hidden * hidden  # W_gv: full H*H (not kv_ratio)
            if use_output_gate:
                attn_params_per_layer += hidden * hidden

        # Calculate GSA-specific attention params (for hybrid layer separation)
        # GSA uses: Q, K, V, O projections + dual gating (value gate + output gate) + indexer
        gsa_attn_params_per_layer = 4 * hidden * hidden  # Q, K, V, O
        if gsa_enabled:
            if use_value_gate:
                gsa_attn_params_per_layer += hidden * hidden  # W_gv: full H*H
            if use_output_gate:
                gsa_attn_params_per_layer += hidden * hidden  # W_go
            # GSA indexer params
            if indexer_heads and indexer_dim:
                gsa_indexer_params = (
                    hidden * indexer_heads * indexer_dim  # W_Iq
                    + hidden * indexer_dim                # W_Ik
                    + hidden * indexer_heads              # W_Iw
                    + indexer_heads                       # gate_bias
                )
                gsa_attn_params_per_layer += gsa_indexer_params

        attn_params_per_layer += float(
            arch.get(
                "attn_extra_params_per_layer",
                attention_cfg.get("attn_extra_params_per_layer", 0),
            )
        )

        # =========================================================================
        # DeltaNet Parameters (Gated Linear Attention)
        # DeltaNet has: Q/K/V/G/O projections + beta gate + alpha gate + A_log + D + short convs
        # O(Ld²) complexity vs O(L²d) for standard attention
        # =========================================================================
        deltanet_in_hybrid = (
            parsed_attn.get("type") == "hybrid" and 
            parsed_attn.get("hybrid_config") and 
            "deltanet" in [parsed_attn["hybrid_config"].get("type1"), parsed_attn["hybrid_config"].get("type2")]
        )
        deltanet_enabled = attention_type in (
            "deltanet", "delta", "linear", "gated_linear", "gated_deltanet"
        ) or bool(arch.get("use_deltanet", False)) or deltanet_in_hybrid
        
        # DeltaNet config
        deltanet_cfg = arch.get("deltanet", attention_cfg.get("deltanet", {}))
        delta_num_heads = deltanet_cfg.get("num_heads", num_heads or 32)
        delta_head_dim = deltanet_cfg.get("head_dim", hidden // delta_num_heads if delta_num_heads else 128)
        delta_conv_size = deltanet_cfg.get("conv_size", 4)  # Short convolution kernel size
        
        # Calculate layer distribution for hybrid
        deltanet_layer_weight = 0.0
        gsa_layer_weight = 0.0
        if parsed_attn.get("type") == "hybrid" and parsed_attn.get("hybrid_config"):
            hc = parsed_attn["hybrid_config"]
            if hc.get("type1") == "deltanet":
                deltanet_layer_weight = hc.get("layer_weight_type1", 0.75)
                gsa_layer_weight = hc.get("layer_weight_type2", 0.25)
            elif hc.get("type2") == "deltanet":
                deltanet_layer_weight = hc.get("layer_weight_type2", 0.25)
                gsa_layer_weight = hc.get("layer_weight_type1", 0.75)
        elif deltanet_enabled and not deltanet_in_hybrid:
            deltanet_layer_weight = 1.0
        
        # Read explicit layer counts from config first, then fall back to weight calculation
        num_deltanet_layers = deltanet_cfg.get(
            "num_deltanet_layers", 
            arch.get("num_deltanet_layers", int(layers * deltanet_layer_weight))
        )
        num_gsa_layers_hybrid = deltanet_cfg.get(
            "num_gsa_layers",
            arch.get("num_gsa_layers", int(layers * gsa_layer_weight) if deltanet_in_hybrid else 0)
        )
        
        # DeltaNet-specific parameters per layer
        deltanet_extra_params_per_layer = 0
        if deltanet_enabled:
            # Beta gate: H -> num_heads (weight + bias)
            beta_gate_params = hidden * delta_num_heads + delta_num_heads
            # Alpha (gk) gate: H -> num_heads (weight + bias)
            alpha_gate_params = hidden * delta_num_heads + delta_num_heads
            # A_log: per-head decay parameter
            a_log_params = delta_num_heads
            # D: per-head residual parameter
            d_params = delta_num_heads
            # dt_bias: per-head bias for delta gate
            dt_bias_params = delta_num_heads
            # Short convolutions on Q/K/V (depthwise, kernel=conv_size)
            short_conv_params = 3 * hidden * delta_conv_size
            # Output norm weight (per head_dim)
            o_norm_params = delta_head_dim
            
            deltanet_extra_params_per_layer = (
                beta_gate_params + alpha_gate_params + a_log_params + 
                d_params + dt_bias_params + short_conv_params + o_norm_params
            )
            
            # DeltaNet uses 5 projections: Q, K, V, G (output gate), O
            # Standard attention: Q, K, V, O (4 projections with KV ratio)
            # DeltaNet adds G projection: +H*H params per layer
            deltanet_g_proj_params = hidden * hidden
            deltanet_extra_params_per_layer += deltanet_g_proj_params

        moe_intermediate = arch.get(
            "moe_intermediate_size", expert_cfg.get("intermediate_size", intermediate)
        )
        ffn_params_dense = 3 * hidden * intermediate
        ffn_params_per_expert = 3 * hidden * moe_intermediate

        # Shared expert uses its own intermediate size (may differ from routed experts)
        shared_expert_intermediate = arch.get(
            "shared_expert_intermediate_size",
            expert_cfg.get("shared_expert_intermediate_size", moe_intermediate)
        )
        ffn_params_per_shared_expert = 3 * hidden * shared_expert_intermediate

        total_ffn_params_moe = 0
        active_ffn_params_moe = 0
        router_params = 0
        shared_expert_params = num_shared_experts * ffn_params_per_shared_expert
        null_expert_params_per_expert = arch.get(
            "null_expert_params_per_expert",
            router_cfg.get("null_expert_params_per_expert", 0),
        )
        null_expert_params = num_null_experts * null_expert_params_per_expert
        
        # Data sparsity affects active expert count (e.g. 0.5 = half tokens go to null experts)
        data_sparsity = arch.get(
            "data_sparsity", 
            router_cfg.get("data_sparsity", 1.0)  # Default 1.0 = no null routing
        )
        # Effective active experts: top_k * data_sparsity
        e_k_real = top_k * data_sparsity
        
        router_type = str(router_cfg.get("router_type", "")).strip().lower()
        if (
            experts > 0
            or num_shared_experts > 0
            or solve_for in ("num_experts", "num_experts_from_per_expert")
        ):
            if experts > 0 or solve_for in (
                "num_experts",
                "num_experts_from_per_expert",
            ):
                router_params = hidden * experts
                if router_type in ("gsa", "gsa_router", "gsa_style"):
                    router_heads = int(router_cfg.get("num_router_heads", 4))
                    router_dim = int(router_cfg.get("router_dim", 64))
                    router_keys = (experts + num_null_experts) * router_dim
                    router_params += (
                        hidden * router_heads * router_dim
                        + hidden * router_heads
                        + router_keys
                        + router_heads
                    )
                router_params_multiplier = float(
                    router_cfg.get("router_params_multiplier", 1.0)
                )
                router_params_extra = float(router_cfg.get("router_params_extra", 0))
                router_params = router_params * router_params_multiplier + router_params_extra
            total_ffn_params_moe = (
                experts * ffn_params_per_expert
                + shared_expert_params
                + router_params
                + null_expert_params
            )
            active_ffn_params_moe = (
                e_k_real * ffn_params_per_expert + shared_expert_params + router_params
            )

        dense_layers = layers - num_moe_layers

        derived_experts = None
        if solve_for == "num_experts":
            if target_total_params is None:
                raise ValueError(
                    "target_total_params must be set when solve_for='num_experts'."
                )
            if num_moe_layers <= 0:
                raise ValueError(
                    "num_moe_layers must be > 0 when solve_for='num_experts'."
                )
            base_params = (
                embedding_params
                + lm_head_params
                + layers * attn_params_per_layer
                + dense_layers * ffn_params_dense
                + num_moe_layers * (shared_expert_params + null_expert_params)
            )
            per_expert_per_layer = ffn_params_per_expert + hidden
            derived_experts = (target_total_params - base_params) / (
                num_moe_layers * per_expert_per_layer
            )
            if derived_experts < 1:
                raise ValueError(
                    "target_total_params is too small for the given architecture."
                )
            experts = int(round(derived_experts))
            if experts < 1:
                raise ValueError(
                    "Derived num_experts is < 1; check target_total_params."
                )
            if top_k > experts:
                raise ValueError("top_k_experts cannot exceed derived num_experts.")
            router_params = hidden * experts
            total_ffn_params_moe = (
                experts * ffn_params_per_expert
                + shared_expert_params
                + router_params
                + null_expert_params
            )
            active_ffn_params_moe = (
                e_k_real * ffn_params_per_expert + shared_expert_params + router_params
            )
            num_moe_layers = layers if num_moe_layers == 0 else num_moe_layers
        elif solve_for == "num_experts_from_per_expert":
            if target_total_params is None or target_params_per_expert is None:
                raise ValueError(
                    "target_total_params and target_params_per_expert must be set when solve_for='num_experts_from_per_expert'."
                )
            experts = int(round(target_total_params / target_params_per_expert))
            if experts < 1:
                raise ValueError(
                    "Derived num_experts is < 1; check target_total_params/target_params_per_expert."
                )
            if top_k > experts:
                raise ValueError("top_k_experts cannot exceed derived num_experts.")
            router_params = hidden * experts
            total_ffn_params_moe = (
                experts * ffn_params_per_expert
                + shared_expert_params
                + router_params
                + null_expert_params
            )
            active_ffn_params_moe = (
                e_k_real * ffn_params_per_expert + shared_expert_params + router_params
            )
            derived_experts = experts

        if experts > 0 and top_k > experts:
            raise ValueError("top_k_experts cannot exceed num_experts.")

        # =========================================================================
        # Normalization Parameters (LayerNorm/RMSNorm)
        # 2 norms per layer (pre-attention, pre-FFN) + 1 final output norm
        # RMSNorm: only gamma (scale) = hidden params per norm
        # LayerNorm: gamma + beta = 2 * hidden params per norm
        # =========================================================================
        norm_type = str(arch.get("normalization", arch.get("norm_type", "rmsnorm"))).strip().lower()
        num_norms = layers * 2 + 1  # 2 per layer + 1 final
        if norm_type in ("rmsnorm", "rms", "rms_norm"):
            norm_params = num_norms * hidden  # Only gamma
        else:
            norm_params = num_norms * 2 * hidden  # gamma + beta

        # =========================================================================
        # mHC (Multi-Head Composition) Parameters
        # Used in reversible/stream-based architectures (e.g., LightningLM)
        # Each layer has 2 mHC sublayers (attn + mlp blocks)
        # =========================================================================
        mhc_cfg = arch.get("mhc", {})
        n_streams = mhc_cfg.get("n_streams", arch.get("n_streams", 0))
        mhc_params_per_layer = 0
        if n_streams > 0:
            d_in = n_streams * hidden
            # MHCCoeffs per sublayer
            mhc_phi_pre = d_in * n_streams
            mhc_phi_post = d_in * n_streams
            mhc_phi_res = d_in * (n_streams * n_streams)
            mhc_biases = n_streams + n_streams + (n_streams * n_streams)
            mhc_alphas = 3  # alpha_pre, alpha_post, alpha_res
            mhc_rms = d_in  # RMSNorm per sublayer
            mhc_per_sublayer = mhc_phi_pre + mhc_phi_post + mhc_phi_res + mhc_biases + mhc_alphas + mhc_rms
            mhc_params_per_layer = 2 * mhc_per_sublayer  # 2 sublayers: attn + mlp

        total_mhc_params = layers * mhc_params_per_layer

        # =========================================================================
<<<<<<< HEAD
        # MTP Block Parameters (Multi-Token Prediction)
        # MTP block contains: fusion projection + DeltaNet layer + FFN/MoE + mHC + norms
        # This is a full transformer layer plus fusion projection
=======
        # Memory Stream Recurrence Parameters
        # Used for infinite-length document processing via chunking
        # Components: lambda_r_raw (1), memory_ln LayerNorm(H), memory_gate_proj Linear(H,1)
        # =========================================================================
        recurrence_cfg = arch.get("recurrence", {})
        recurrence_enabled = recurrence_cfg.get("enabled", False)
        recurrence_params = 0
        if recurrence_enabled:
            # lambda_r_raw: 1 scalar parameter (learnable recurrence strength)
            # memory_ln: LayerNorm(hidden) = 2*hidden params (weight + bias)
            # memory_gate_proj: Linear(hidden, 1, bias=True) = hidden + 1 params
            recurrence_params = 1 + (2 * hidden) + (hidden + 1)

        # =========================================================================
        # MTP Block Parameters (Multi-Token Prediction)
        # MTP block contains: fusion projection + attention layer + FFN/MoE + mHC + norms
        # Attention type is configurable: 'gsa' or 'deltanet' (default: deltanet)
>>>>>>> 8073e5d9cfdf80f7531af2f2af5626e91a480bd7
        # =========================================================================
        mtp_cfg = arch.get("mtp", head_cfg)
        mtp_enabled = mtp_cfg.get("enabled", use_mtp)
        mtp_num_predictions = mtp_cfg.get("num_predictions", head_cfg.get("mtp_heads", 2)) if mtp_enabled else 0
        # Calculate attention params per layer type for hybrid models
        # DeltaNet layers: base attention (4*H*H) + G projection + extra params (gates, convs)
        # GSA layers: base attention (4*H*H) + dual gating (W_gv, W_go) + indexer
        base_attn_params = 4 * hidden * hidden
        deltanet_attn_per_layer = base_attn_params + deltanet_extra_params_per_layer
        gsa_attn_per_layer = gsa_attn_params_per_layer if gsa_enabled else base_attn_params

        mtp_block_params = 0
        mtp_block_active_params = 0
        if mtp_enabled and mtp_num_predictions > 0:
            # Fusion projection: (d * 2) * d (combines current and previous hidden states)
            mtp_fusion_params = (hidden * 2) * hidden
<<<<<<< HEAD
            # DeltaNet layer (same as regular DeltaNet layer)
            mtp_deltanet_params = deltanet_attn_per_layer
=======
            # MTP attention type: configurable (model may use GSA for better gradient quality)
            mtp_attn_type = arch.get("mtp_attention_type", "deltanet")
            if mtp_attn_type == "gsa" and gsa_enabled:
                mtp_attn_params = gsa_attn_per_layer
            else:
                mtp_attn_params = deltanet_attn_per_layer
>>>>>>> 8073e5d9cfdf80f7531af2f2af5626e91a480bd7
            # FFN layer (MoE or dense)
            mtp_ffn_total = total_ffn_params_moe if num_moe_layers > 0 else ffn_params_dense
            mtp_ffn_active = active_ffn_params_moe if num_moe_layers > 0 else ffn_params_dense
            # mHC (same as regular layer)
            mtp_mhc_params = mhc_params_per_layer
            # Norms (2 per layer)
            mtp_norms = 2 * hidden
            
<<<<<<< HEAD
            mtp_block_params = mtp_fusion_params + mtp_deltanet_params + mtp_ffn_total + mtp_mhc_params + mtp_norms
            mtp_block_active_params = mtp_fusion_params + mtp_deltanet_params + mtp_ffn_active + mtp_mhc_params + mtp_norms
=======
            mtp_block_params = mtp_fusion_params + mtp_attn_params + mtp_ffn_total + mtp_mhc_params + mtp_norms
            mtp_block_active_params = mtp_fusion_params + mtp_attn_params + mtp_ffn_active + mtp_mhc_params + mtp_norms
>>>>>>> 8073e5d9cfdf80f7531af2f2af5626e91a480bd7
        
        # Number of GSA layers in hybrid mode
        num_gsa_layers_in_model = num_gsa_layers_hybrid if deltanet_in_hybrid else (
            layers if gsa_enabled and not deltanet_enabled else 0
        )
        # Number of pure standard attention layers (neither DeltaNet nor GSA)
        num_std_attn_layers = layers - num_deltanet_layers - num_gsa_layers_in_model
        
        # Total attention params across all layers
        total_attn_params = (
            num_deltanet_layers * deltanet_attn_per_layer +
            num_gsa_layers_in_model * gsa_attn_per_layer +
            num_std_attn_layers * base_attn_params
        )

        total_params = (
            embedding_params
            + lm_head_params
            + total_attn_params  # Layer-type-specific attention params
            + dense_layers * ffn_params_dense
            + num_moe_layers * total_ffn_params_moe
            + norm_params
            + total_mhc_params  # mHC params
            + mtp_block_params  # MTP block (full layer + fusion)
<<<<<<< HEAD
=======
            + recurrence_params  # Memory Stream Recurrence
>>>>>>> 8073e5d9cfdf80f7531af2f2af5626e91a480bd7
        )

        active_non_embed_params = (
            total_attn_params  # Layer-type-specific attention params
            + dense_layers * ffn_params_dense
            + num_moe_layers * active_ffn_params_moe
            + lm_head_params
            + norm_params
            + total_mhc_params  # mHC params
            + mtp_block_active_params  # MTP block active params
<<<<<<< HEAD
=======
            + recurrence_params  # Memory Stream Recurrence
>>>>>>> 8073e5d9cfdf80f7531af2f2af5626e91a480bd7
        )
        active_linear_params = (
            total_attn_params  # Layer-type-specific attention params
            + dense_layers * ffn_params_dense
            + num_moe_layers * active_ffn_params_moe
            + lm_head_params_for_flops
            + norm_params
            + total_mhc_params  # mHC params
            + mtp_block_active_params  # MTP block active params
<<<<<<< HEAD
=======
            + recurrence_params  # Memory Stream Recurrence
>>>>>>> 8073e5d9cfdf80f7531af2f2af5626e91a480bd7
        )
        active_params_base = embedding_params + active_non_embed_params

        params_null_path = (
            embedding_params
            + layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * (router_params + shared_expert_params)
            + lm_head_params
            + norm_params
        )
        params_null_path_non_embed = (
            layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * (router_params + shared_expert_params)
            + lm_head_params
        )
        params_null_path_linear = (
            layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * (router_params + shared_expert_params)
            + lm_head_params_for_flops
        )

        effective_active_params = (
            1 - null_prob
        ) * active_params_base + null_prob * params_null_path
        effective_active_non_embed_params = (
            1 - null_prob
        ) * active_non_embed_params + null_prob * params_null_path_non_embed
        effective_active_linear_params = (
            1 - null_prob
        ) * active_linear_params + null_prob * params_null_path_linear

        # Expert params = routed experts only (for EP sharding)
        routed_expert_params = num_moe_layers * experts * ffn_params_per_expert if experts > 0 else 0
        # Non-expert params = everything except routed experts
        non_expert_params = total_params - routed_expert_params

        return {
            "total_params": total_params,
            "active_params": active_params_base,
            "active_non_embed_params": active_non_embed_params,
            "active_linear_params": active_linear_params,
            "effective_active_params": effective_active_params,
            "effective_active_non_embed_params": effective_active_non_embed_params,
            "effective_active_linear_params": effective_active_linear_params,
            "embedding_params": embedding_params,
            "lm_head_params": lm_head_params,
            "num_moe_layers": num_moe_layers,
            "dense_layers": dense_layers,
            "num_experts": experts,
            "derived_num_experts": derived_experts,
            "routed_expert_params": routed_expert_params,
            "non_expert_params": non_expert_params,
        }

    def calculate_memory_per_gpu(
        self,
        params: Optional[dict] = None,
        num_gpus: int = 8,
        zero_stage: int = 2,
        quantization: str = "bf16",
        cpu_offload: bool = False,
        cpu_offload_config: Optional[dict] = None,
        expert_parallel_size: int = 1,
        checkpoint_factor: Optional[float] = None,
        include_activation_memory: Optional[bool] = None,
        micro_batch_size: Optional[int] = None,
        partition_activations: bool = False,
    ) -> dict:
        """
        Estimate memory footprint per GPU for training this stage.

        Assumptions:
        - Model weights: Depends on quantization (BF16=2, FP8=1, NVFP4=0.5 bytes/param)
        - FP32 optimizer states (Adam): 8 bytes/param (momentum + variance, always FP32)
        - FP32 gradients: 4 bytes/param (gradient accumulation typically FP32)
        - Activations are NOT included (highly batch-size dependent) unless enabled
        - Activation checkpointing reduces activation memory when configured

        ZeRO Stages:
        - ZeRO-0: No sharding (baseline)
        - ZeRO-2: Shard optimizer states + gradients (+ expert sharding with EP)
        - ZeRO-3: Shard everything (params + optimizer + gradients)
        - ZeRO-Infinity (cpu_offload=True): Move selected state to CPU/NVMe, keep a GPU buffer.

        Expert Parallelism:
        - Routed expert weights are sharded across EP group
        - Non-expert params (attention, embeddings, shared experts) remain unsharded in ZeRO-2
        """
        params = params or self.calculate_params()
        total_params = params["total_params"]
        routed_expert_params = params.get("routed_expert_params", 0)
        non_expert_params = params.get("non_expert_params", total_params)
        arch = self.architecture
        precision_cfg = arch.get("precision") or {}
        training_cfg = arch.get("training") or {}

        def _get_prec_value(key: str, default=None):
            if key in arch:
                return arch.get(key)
            return precision_cfg.get(key, default)

        # Model weights precision (defaults to quantization selection)
        weight_bytes_per_param = _get_prec_value("weight_bytes_per_param")
        if weight_bytes_per_param is None:
            weight_precision = _get_prec_value("weight_precision")
            if weight_precision:
                wp = str(weight_precision).strip().lower()
                if wp not in ("auto", "default", ""):
                    weight_bytes_per_param = bytes_for_precision(weight_precision)
            if weight_bytes_per_param is None:
                quantization = str(quantization).lower()
                if quantization == "fp8":
                    weight_bytes_per_param = 1
                elif quantization == "nvfp4":
                    weight_bytes_per_param = 0.5
                else:  # bf16 or default
                    weight_bytes_per_param = 2

        model_bytes = total_params * float(weight_bytes_per_param)

        # Optional FP32 master weights (common for mixed precision training)
        master_weights = bool(_get_prec_value("master_weights", False))
        master_weights_precision = _get_prec_value("master_weights_precision", "fp32")
        master_bytes = (
            total_params * bytes_for_precision(master_weights_precision)
            if master_weights
            else 0
        )

        # Optimizer states (defaults to FP32 Adam: 2 states)
        optimizer_state_bytes_per_param = _get_prec_value(
            "optimizer_state_bytes_per_param"
        )
        if optimizer_state_bytes_per_param is None:
            optimizer_precision = _get_prec_value("optimizer_precision", "fp32")
            optimizer_states_count = int(
                _get_prec_value("optimizer_states_count", 2)
            )
            optimizer_state_bytes_per_param = optimizer_states_count * bytes_for_precision(
                optimizer_precision
            )
        optimizer_state_multiplier = float(
            _get_prec_value("optimizer_state_multiplier", 1.0)
        )
        optimizer_bytes = (
            total_params * float(optimizer_state_bytes_per_param) * optimizer_state_multiplier
        )

        # Gradients (defaults to FP32)
        gradient_bytes_per_param = _get_prec_value("gradient_bytes_per_param")
        if gradient_bytes_per_param is None:
            gradient_precision = _get_prec_value("gradient_precision", "fp32")
            gradient_bytes_per_param = bytes_for_precision(gradient_precision)
        gradient_bytes = total_params * float(gradient_bytes_per_param)

        # Activation memory - ALWAYS calculated (required for accurate GPU memory estimates)
        # Only skip if explicitly set to False in the passed parameter
        include_act_mem = include_activation_memory if include_activation_memory is not None else True
        activation_bytes = 0.0
        # Check for reversible training (midpoint/leapfrog method)
        use_reversible = arch.get("reversible", training_cfg.get("reversible", False))
        if include_act_mem:
            # Use passed micro_batch_size, else fallback to training_cfg
            micro_batch = micro_batch_size
            if micro_batch is None:
                micro_batch = training_cfg.get("micro_batch_size", 1)
            micro_batch = float(micro_batch)
            seq_len = float(
                training_cfg.get(
                    "seq_length",
                    arch.get("sequence_length", arch.get("max_position_embeddings", 4096)),
                )
            )
            hidden = float(arch.get("hidden_size", 2048))
            layers = float(arch.get("num_layers", arch.get("num_hidden_layers", 24)))
            act_bytes_per_elem = training_cfg.get("activation_bytes_per_element")
            if act_bytes_per_elem is None:
                act_prec = training_cfg.get("activation_precision", "bf16")
                act_bytes_per_elem = bytes_for_precision(act_prec)

            if use_reversible:
                # Reversible training (midpoint/leapfrog): only store 2 hidden states
                # (p_prev, p_cur) regardless of model depth. Memory is O(1) in layers.
                activation_bytes = 2.0 * micro_batch * seq_len * hidden * act_bytes_per_elem
            else:
                # Standard training: store activations for all layers
                activation_multiplier = float(training_cfg.get("activation_multiplier", 10.0))
                # Use passed checkpoint_factor, else fallback to training_cfg, else defaults
                ckpt_factor = checkpoint_factor
                if ckpt_factor is None:
                    ckpt_factor = training_cfg.get("activation_checkpointing_factor")
                if ckpt_factor is None:
                    ckpt_factor = 0.5 if training_cfg.get("activation_checkpointing") else 1.0
                ckpt_factor = float(ckpt_factor)
                if ckpt_factor <= 0:
                    raise ValueError("activation_checkpointing_factor must be > 0.")
                activation_bytes = (
                    micro_batch
                    * seq_len
                    * hidden
                    * layers
                    * activation_multiplier
                    * act_bytes_per_elem
                    * ckpt_factor
                )
            # partition_activations: shard activation memory across data parallel GPUs
            if partition_activations and num_gpus > 1:
                activation_bytes = activation_bytes / num_gpus

        cpu_memory_gb = 0.0

        # Per-param bytes for EP calculation
        opt_bytes_per_param = float(optimizer_state_bytes_per_param) * optimizer_state_multiplier
        grad_bytes_per_param = float(gradient_bytes_per_param)

        # Calculate bytes separately for expert vs non-expert params
        expert_model_bytes = routed_expert_params * float(weight_bytes_per_param)
        non_expert_model_bytes = non_expert_params * float(weight_bytes_per_param)
        expert_optimizer_bytes = routed_expert_params * opt_bytes_per_param
        non_expert_optimizer_bytes = non_expert_params * opt_bytes_per_param
        expert_gradient_bytes = routed_expert_params * grad_bytes_per_param
        non_expert_gradient_bytes = non_expert_params * grad_bytes_per_param

        # Per-GPU bytes for each component under the chosen ZeRO stage
        if zero_stage == 0:
            model_gpu_bytes = model_bytes
            master_gpu_bytes = master_bytes
            optimizer_gpu_bytes = optimizer_bytes
            gradient_gpu_bytes = gradient_bytes
        elif zero_stage == 3:
            model_gpu_bytes = model_bytes / num_gpus
            master_gpu_bytes = master_bytes / num_gpus
            optimizer_gpu_bytes = optimizer_bytes / num_gpus
            gradient_gpu_bytes = gradient_bytes / num_gpus
        else:
            # ZeRO-2: shard optimizer + gradients, replicate model
            # With EP: expert weights are sharded across EP group
            ep = max(1, expert_parallel_size)
            
            # Only apply EP if there are actually experts to shard
            if routed_expert_params > 0 and ep > 1:
                # Model weights: experts sharded by EP, non-experts replicated
                model_gpu_bytes = (expert_model_bytes / ep) + non_expert_model_bytes
                # Master weights: sharded like optimizer states in ZeRO-2
                master_gpu_bytes = master_bytes / num_gpus
                
                # Optimizer: non-experts sharded by full num_gpus, experts sharded by DP within EP group
                # DP size within EP group = num_gpus / ep
                dp_size = num_gpus / ep
                optimizer_gpu_bytes = (expert_optimizer_bytes / ep) / dp_size + (non_expert_optimizer_bytes / num_gpus)
                gradient_gpu_bytes = (expert_gradient_bytes / ep) / dp_size + (non_expert_gradient_bytes / num_gpus)
            else:
                # No experts or EP=1: standard ZeRO-2 sharding
                model_gpu_bytes = model_bytes
                # Master weights are sharded in ZeRO-2 (part of optimizer state)
                master_gpu_bytes = master_bytes / num_gpus
                optimizer_gpu_bytes = optimizer_bytes / num_gpus
                gradient_gpu_bytes = gradient_bytes / num_gpus

        if cpu_offload:
            cpu_offload_cfg = cpu_offload_config or {}
            offload_params = bool(cpu_offload_cfg.get("offload_params", True))
            offload_optimizer = bool(cpu_offload_cfg.get("offload_optimizer", True))
            offload_gradients = bool(cpu_offload_cfg.get("offload_gradients", True))

            offloaded_bytes_total = 0.0
            if offload_params:
                offloaded_bytes_total += model_bytes + master_bytes
                model_gpu_bytes = 0.0
                master_gpu_bytes = 0.0
            if offload_optimizer:
                offloaded_bytes_total += optimizer_bytes
                optimizer_gpu_bytes = 0.0
            if offload_gradients:
                offloaded_bytes_total += gradient_bytes
                gradient_gpu_bytes = 0.0

            remaining_gpu_bytes = (
                model_gpu_bytes
                + master_gpu_bytes
                + optimizer_gpu_bytes
                + gradient_gpu_bytes
            )
            # GPU buffer for parameter streaming (DeepSpeed keeps a buffer for overlapping)
            # Default: ~4GB or configurable via gpu_buffer_gb
            gpu_buffer_gb = float(cpu_offload_cfg.get("gpu_buffer_gb", 4.0))
            gpu_buffer_bytes = gpu_buffer_gb * (1024**3)
            memory_bytes = remaining_gpu_bytes + activation_bytes + gpu_buffer_bytes
            cpu_memory_bytes = offloaded_bytes_total / num_gpus
            cpu_memory_gb = cpu_memory_bytes / (1024**3)
        else:
            memory_bytes = (
                model_gpu_bytes
                + master_gpu_bytes
                + optimizer_gpu_bytes
                + gradient_gpu_bytes
                + activation_bytes
            )

        memory_gb = memory_bytes / (1024**3)

        return {
            "memory_per_gpu_gb": memory_gb,
            "cpu_memory_per_gpu_gb": cpu_memory_gb,
            "zero_stage": zero_stage,
            "quantization": quantization,
            "model_gb": model_bytes / (1024**3),
            "master_weights_gb": master_bytes / (1024**3),
            "optimizer_gb": optimizer_bytes / (1024**3),
            "gradient_gb": gradient_bytes / (1024**3),
            "activation_gb": activation_bytes / (1024**3),
        }

    def flops_per_token(self, params: Optional[dict] = None) -> float:
        """
        Calculate per-token FLOPs for this stage.

        Supports both dense and sparse attention mechanisms:
        - Dense: O(L^2) complexity using standard transformer attention
        - Sparse: O(Lk) complexity using DeepSeek Sparse Attention (DSA)
        """
        params = params or self.calculate_params()
        arch = self.architecture

        attention_cfg = arch.get("attention") or {}
        router_cfg = arch.get("router") or {}
        training_cfg = arch.get("training") or {}
        head_cfg = arch.get("head") or {}

        n_linear = params.get("active_linear_params", params["active_non_embed_params"])
        layers = arch.get("num_layers", arch.get("num_hidden_layers", 24))
        h = arch.get("hidden_size", 2048)
        s = arch.get(
            "sequence_length",
            arch.get(
                "seq_length", arch.get("max_position_embeddings", 4096)
            ),
        )

        attention_type = str(
            arch.get("attention_type", attention_cfg.get("attention_type", ""))
        ).strip().lower()
        if attention_type in ("grouped_query", "grouped_query_attention"):
            attention_type = "gqa"
        if attention_type in ("deepseek_sparse", "deepseek_mla", "mla", "deepseek"):
            attention_type = "deepseek_mla"
        
        router_type = str(router_cfg.get("router_type", "")).strip().lower()

        # Parse attention type notation (e.g., "gsa:128" -> base_type="gsa", notation_value=128)
        # Also handle hybrid format (e.g., "gqa-gsa:4:1")
        num_heads_for_parsing = arch.get("num_heads", attention_cfg.get("num_attention_heads", 32))
        parsed_attn = parse_attention_type(attention_type, num_heads_for_parsing or 32)
        base_attention_type = parsed_attn.get("type", "mha")
        
        # Check for GSA (including hybrid with GSA component)
        gsa_in_hybrid = (
            base_attention_type == "hybrid" and 
            parsed_attn.get("hybrid_config") and 
            "gsa" in [parsed_attn["hybrid_config"].get("type1"), parsed_attn["hybrid_config"].get("type2")]
        )
        gsa_enabled = base_attention_type in (
            "gsa",
            "gated_sparse",
            "gated_sparse_attention",
            "deepseek_gsa",
        ) or bool(arch.get("use_gsa", False)) or gsa_in_hybrid

        # Sparse attention configuration - use k from gsa:k notation if present
        use_sparse_attn = bool(arch.get("use_sparse_attention", False)) or gsa_enabled
        sparse_k_tokens = arch.get("sparse_k_tokens", s)

        if gsa_enabled:
            # Priority: gsa:k notation > gsa_k_tokens > sparse_k_tokens > sequence_length
            if ":" in attention_type:
                # Parse k from gsa:k notation
                try:
                    sparse_k_tokens = int(attention_type.split(":")[1])
                except (ValueError, IndexError):
                    pass
            gsa_k_tokens = arch.get("gsa_k_tokens", attention_cfg.get("gsa_k_tokens"))
            if gsa_k_tokens is not None:
                sparse_k_tokens = gsa_k_tokens

        # Indexer defaults (used for FLOPs calculation)
        indexer_heads = 4
        indexer_dim = h // 8

        mla_kv_lora_rank = arch.get(
            "mla_kv_lora_rank",
            attention_cfg.get(
                "mla_kv_lora_rank", arch.get("ds_compressed_dim", 0)
            ),
        )  # MLA compression rank (0 = disabled)

        if s <= 0:
            raise ValueError("sequence_length must be > 0.")

        # =========================================================================
        # Recompute Multiplier (for activation/gradient checkpointing)
        # =========================================================================
        # When checkpointing is enabled, activations are recomputed during backward
        # pass, adding ~33% overhead (extra forward pass for recomputation).
        # Base training FLOPs: Forward (2x) + Backward-grad (2x) + Backward-weight (2x) = 6x
        # With recompute: 6x + 2x (extra forward) = 8x, ratio = 8/6 = 4/3 ≈ 1.33
        # =========================================================================
        gradient_ckpt = training_cfg.get("gradient_checkpointing", False)
        activation_ckpt = training_cfg.get("activation_checkpointing", False)
        # Also check for partition_activations in training config (DeepSpeed-style)
        partition_activations = training_cfg.get("partition_activations", False)
        # Reversible training also recomputes activations during backward
        use_reversible = arch.get("reversible", training_cfg.get("reversible", False))
        
        use_checkpointing = gradient_ckpt or activation_ckpt or partition_activations or use_reversible
        reversible_overhead = float(training_cfg.get("reversible_recompute_overhead", 4/3))
        recompute_multiplier = reversible_overhead if use_checkpointing else 1.0

        # Linear layer FLOPs (includes recompute overhead)
        linear_multiplier = float(arch.get("linear_flops_multiplier", 1.0))
        flops_per_seq_linear = 6 * s * n_linear * linear_multiplier * recompute_multiplier

        # Attention FLOPs calculation
        attention_multiplier = arch.get("attention_flops_multiplier")
        if attention_multiplier is None:
            attention_multiplier = arch.get(
                "attention_kernel_multiplier",
                arch.get("flash_attention_multiplier", 1.0),
            )
        attention_multiplier = float(attention_multiplier)
        if attention_multiplier <= 0:
            raise ValueError("attention_flops_multiplier must be > 0.")
        # Note: YaRN position encoding has zero FLOPs overhead compared to standard RoPE.
        # The cost of longer contexts is captured in the O(S²) attention formula via sequence_length.

        num_heads = arch.get(
            "num_heads", attention_cfg.get("num_attention_heads", 32)
        )
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        head_dim = h / num_heads
        attn_dim = head_dim

        if mla_kv_lora_rank and mla_kv_lora_rank > 0:
            attn_dim = float(mla_kv_lora_rank) / num_heads
            if attn_dim <= 0:
                attn_dim = head_dim

        # =========================================================================
        # DeltaNet FLOPs Calculation (Gated Linear Attention)
        # DeltaNet: O(L × d²) complexity - linear in sequence length!
        # vs Standard Attention: O(L² × d) - quadratic in sequence length
        # =========================================================================
        deltanet_in_hybrid = (
            parsed_attn.get("type") == "hybrid" and 
            parsed_attn.get("hybrid_config") and 
            "deltanet" in [parsed_attn["hybrid_config"].get("type1"), parsed_attn["hybrid_config"].get("type2")]
        )
        deltanet_enabled = attention_type in (
            "deltanet", "delta", "linear", "gated_linear", "gated_deltanet"
        ) or bool(arch.get("use_deltanet", False)) or deltanet_in_hybrid
        
        # DeltaNet config
        deltanet_cfg = arch.get("deltanet", attention_cfg.get("deltanet", {}))
        delta_num_heads = deltanet_cfg.get("num_heads", num_heads)
        delta_head_dim = deltanet_cfg.get("head_dim", h // delta_num_heads if delta_num_heads else 128)
        delta_conv_size = deltanet_cfg.get("conv_size", 4)
        
        # Calculate layer distribution for hybrid
        deltanet_layer_weight = 0.0
        gsa_layer_weight_hybrid = 0.0
        if parsed_attn.get("type") == "hybrid" and parsed_attn.get("hybrid_config"):
            hc = parsed_attn["hybrid_config"]
            if hc.get("type1") == "deltanet":
                deltanet_layer_weight = hc.get("layer_weight_type1", 0.75)
                gsa_layer_weight_hybrid = hc.get("layer_weight_type2", 0.25)
            elif hc.get("type2") == "deltanet":
                deltanet_layer_weight = hc.get("layer_weight_type2", 0.25)
                gsa_layer_weight_hybrid = hc.get("layer_weight_type1", 0.75)
        elif deltanet_enabled and not deltanet_in_hybrid:
            deltanet_layer_weight = 1.0
        
        num_deltanet_layers = int(layers * deltanet_layer_weight)
        num_other_attn_layers = layers - num_deltanet_layers
        
        # DeltaNet FLOPs per layer
        flops_per_seq_deltanet = 0
        if deltanet_enabled and num_deltanet_layers > 0:
            d = delta_head_dim
            
            # 1. Input projections: Q, K, V, G (output gate)
            #    Each: 2 × S × H × H (matmul)
            proj_in_flops = 4 * 2 * s * h * h
            
            # 2. Short convolutions on Q, K, V (depthwise, kernel=conv_size)
            #    Each: S × H × conv_size
            short_conv_flops = 3 * s * h * delta_conv_size
            
            # 3. L2 normalization on Q and K
            #    Each: 2 × S × H (compute norm + divide)
            l2_norm_flops = 2 * 2 * s * h
            
            # 4. Delta rule state updates - THE KEY O(d²) TERM
            #    For each token: query state (d²), update state (d²), outer product (d²)
            #    Total: 6 × d² per token per head
            delta_rule_flops = s * delta_num_heads * d * d * 6
            
            # 5. Output projection: 2 × S × H × H
            proj_out_flops = 2 * s * h * h
            
            # 6. Output normalization and gating (small): ~2 × S × H
            output_ops_flops = 2 * s * h
            
            # Total forward FLOPs per DeltaNet layer
            deltanet_forward_per_layer = (
                proj_in_flops + short_conv_flops + l2_norm_flops + 
                delta_rule_flops + proj_out_flops + output_ops_flops
            )
            
            # Training multiplier (fwd + bwd)
            base_multiplier = 3.0  # Forward (1x) + Backward (2x)
            if use_checkpointing:
                base_multiplier *= recompute_multiplier
            
            flops_per_seq_deltanet = (
                num_deltanet_layers * deltanet_forward_per_layer * base_multiplier * attention_multiplier
            )

        if use_sparse_attn:
            # Sparse attention FLOPs (DSA/GSA)

            # 1. Lightning Indexer: O(L^2) but optimized
            #    - Uses FP8 (2x speedup assumed)
            #    - Fewer heads and smaller dimension
            #    - Formula: 2 * seq_len^2 * indexer_heads * indexer_dim / fp8_speedup
            fp8_speedup = float(
                arch.get(
                    "indexer_fp8_speedup",
                    attention_cfg.get("indexer_fp8_speedup", 2.0),
                )
            )
            
            indexer_flops = (2 * (s**2) * indexer_heads * indexer_dim) / fp8_speedup

            # Sort/TopK selection: O(L × k × log(k))
            sort_flops = s * sparse_k_tokens * math.log2(sparse_k_tokens) * 10  # ~10 ops per comparison
            # Index gathering: O(L × k)
            gather_flops = s * sparse_k_tokens * 2  # Index and gather
            indexer_flops += sort_flops + gather_flops

            # 2. Main Sparse Attention: O(Lk) instead of O(L^2)
            #    - QK^T matmul: 2 * seq_len * k * head_dim
            #    - Softmax: 3 * seq_len * k (approximate)
            #    - Attention-V matmul: 2 * seq_len * k * head_dim
            # If MLA is enabled, KV dimension is compressed
            if mla_kv_lora_rank > 0:
                kv_dim = float(mla_kv_lora_rank)
                # Additional projection FLOPs: compress and decompress
                mla_projection_flops = 2 * s * h * kv_dim * 2  # 2x for K and V
            else:
                mla_projection_flops = 0

            # Sparse attention core operations
            qk_matmul_flops = 2 * s * sparse_k_tokens * attn_dim * num_heads
            softmax_flops = 3 * s * sparse_k_tokens * num_heads
            attn_v_matmul_flops = 2 * s * sparse_k_tokens * attn_dim * num_heads

            sparse_attn_core = qk_matmul_flops + softmax_flops + attn_v_matmul_flops

            # Total sparse attention FLOPs per layer (training = fwd + bwd)
            base_multiplier = 3.0  # Forward (1x) + Backward (2x)
            if bool(training_cfg.get("gradient_checkpointing", False)):
                base_multiplier += 1.0  # Extra forward recompute
            if bool(head_cfg.get("use_multi_token_prediction", False)):
                mtp_heads = int(head_cfg.get("mtp_heads", 2))
                base_multiplier += mtp_heads - 1  # Extra forward passes
            if router_type in ("aux_loss", "load_balance"):
                base_multiplier += 0.1  # ~10% overhead for aux loss backward
            
            sparse_attn_per_layer = (
                indexer_flops + sparse_attn_core + mla_projection_flops
            ) * base_multiplier

            # Total for all non-DeltaNet layers (for hybrid configs like deltanet-gsa)
            attn_layers_for_sparse = num_other_attn_layers if deltanet_enabled else layers
            flops_per_seq_attn_sparse = attn_layers_for_sparse * sparse_attn_per_layer * attention_multiplier

            # Store breakdown for debugging (optional)
            self._attn_flops_breakdown = {
                "indexer_flops": indexer_flops * attn_layers_for_sparse * base_multiplier,
                "sparse_core_flops": sparse_attn_core * attn_layers_for_sparse * base_multiplier,
                "mla_projection_flops": mla_projection_flops
                * attn_layers_for_sparse
                * base_multiplier,
                "deltanet_flops": flops_per_seq_deltanet,
                "total_attn_flops": flops_per_seq_attn_sparse + flops_per_seq_deltanet,
                "sparse_k_tokens": sparse_k_tokens,
                "num_deltanet_layers": num_deltanet_layers,
                "num_other_attn_layers": num_other_attn_layers,
            }
        else:
            # Standard Dense Attention: O(L^2)
            # Forward terms:
            #   - QK^T: 2 * H * S^2
            #   - Softmax: 3 * S^2 * num_heads (approximate)
            #   - Attention-V: 2 * H * S^2
            # Multiply by 3 to approximate training (fwd+bwd).
            qk_matmul_flops = 2 * (s**2) * num_heads * attn_dim
            softmax_flops = 3 * (s**2) * num_heads
            attn_v_matmul_flops = 2 * (s**2) * num_heads * attn_dim
            dense_attn_per_layer = (
                qk_matmul_flops + softmax_flops + attn_v_matmul_flops
            ) * 3.0 * recompute_multiplier
            # For hybrid configs, only apply to non-DeltaNet layers
            attn_layers_for_dense = num_other_attn_layers if deltanet_enabled else layers
            flops_per_seq_attn_dense = attn_layers_for_dense * dense_attn_per_layer * attention_multiplier
        
        # Combine DeltaNet + other attention FLOPs
        if use_sparse_attn:
            flops_per_seq_attn = flops_per_seq_deltanet + flops_per_seq_attn_sparse
        else:
            flops_per_seq_attn = flops_per_seq_deltanet + flops_per_seq_attn_dense

        # =========================================================================
        # MoE Router FLOPs (if MoE layers exist)
        # =========================================================================
        # Each MoE layer requires:
        #   - Router forward: hidden × num_experts matmul
        #   - Softmax: ~3 × num_experts per token
        #   - TopK selection: ~5 × top_k per token
        #   - Dispatch index computation: ~2 × top_k per token
        # =========================================================================
        num_moe_layers = params.get("num_moe_layers", 0)
        num_experts = params.get("num_experts", 0)
        top_k = arch.get("top_k_experts", arch.get("top_k", router_cfg.get("top_k", 1)))
        
        flops_per_seq_router = 0.0
        if num_moe_layers > 0 and num_experts > 0:
            # Router linear projection: 2 * S * hidden * num_experts (forward matmul)
            router_linear_flops = 2 * s * h * num_experts
            # Softmax over experts: ~3 * S * num_experts
            router_softmax_flops = 3 * s * num_experts
            # TopK selection: ~5 * S * top_k (comparison + selection)
            router_topk_flops = 5 * s * top_k
            # Dispatch index computation: ~2 * S * top_k
            router_dispatch_flops = 2 * s * top_k
            
            router_flops_per_layer = (
                router_linear_flops + router_softmax_flops + 
                router_topk_flops + router_dispatch_flops
            )
            # Training multiplier (same as attention)
            router_training_mult = 3.0
            flops_per_seq_router = num_moe_layers * router_flops_per_layer * router_training_mult

        # =========================================================================
        # LayerNorm / RMSNorm FLOPs
        # =========================================================================
        # For each transformer layer:
        #   - Pre-attention norm: LayerNorm=5*S*H, RMSNorm=3*S*H
        #   - Pre-FFN norm: LayerNorm=5*S*H, RMSNorm=3*S*H
        # Plus final output norm (1 layer)
        #
        # LayerNorm (5 ops per element):
        #   1. Compute mean: S ops (sum) + 1 (divide)
        #   2. Compute variance: S ops (squared diff sum) + 1 (divide)
        #   3. Normalize: S ops (subtract mean, divide by std)
        #   4. Scale: S ops (multiply by gamma)
        #   5. Shift: S ops (add beta)
        #
        # RMSNorm (3 ops per element):
        #   1. Compute RMS: S ops (squared sum) + 1 (sqrt)
        #   2. Normalize: S ops (divide by RMS)
        #   3. Scale: S ops (multiply by gamma)
        # =========================================================================
        norm_type = str(arch.get("normalization", arch.get("norm_type", "layernorm"))).strip().lower()
        
        if norm_type in ("rmsnorm", "rms", "rms_norm"):
            # RMSNorm: 3 * S * H per normalization
            norm_flops_per_instance = 3 * s * h
        else:
            # LayerNorm: 5 * S * H per normalization
            norm_flops_per_instance = 5 * s * h
        
        # 2 norms per layer (pre-attention, pre-FFN) + 1 final output norm
        num_norms = layers * 2 + 1
        norm_training_mult = 3.0  # Forward (1x) + Backward (2x)
        flops_per_seq_norm = num_norms * norm_flops_per_instance * norm_training_mult

        # =========================================================================
        # Embedding FLOPs (Kronecker Product Embeddings only)
        # =========================================================================
        # Standard embeddings: Lookup operation (negligible FLOPs)
        # Kronecker embeddings: Matrix multiplication per token
        #   - Forward: s × (D × hidden) = s × D × h
        #   - Backward (gradients): s × (D × hidden) = s × D × h  
        #   - Backward (weight): s × (D × hidden) = s × D × h
        #   - Total: 6 × s × D × h (with recompute overhead)
        # =========================================================================
        flops_per_seq_embedding = 0
        embedding_type = arch.get("embedding_type", "standard")
        
        if embedding_type == "kronecker":
            kronecker_cfg = arch.get("kronecker_config", {})
            D = kronecker_cfg.get("D", 8192)
            
            # Matmul FLOPs: forward (2x) + backward-grad (2x) + backward-weight (2x) = 6x
            # Each matmul: s tokens × D × h
            flops_per_seq_embedding = 6 * s * D * h * recompute_multiplier
        
        # =========================================================================
        # Total FLOPs
        # =========================================================================
        flops_per_seq_total = (
            flops_per_seq_embedding +
            flops_per_seq_linear + 
            flops_per_seq_attn + 
            flops_per_seq_router + 
            flops_per_seq_norm
        )

        return flops_per_seq_total / s

    def calculate_flops(self, params: Optional[dict] = None) -> float:
        """
        Calculate total FLOPs for training this stage.
        """
        params = params or self.calculate_params()
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be > 0.")

        return self.flops_per_token(params) * self.total_tokens

    def calculate_upcycling_flops(self, params: Optional[dict] = None) -> dict:
        """
        Calculate one-time FLOPs for Dense→MoE upcycling.
        
        This is a one-time cost when converting a dense model checkpoint to MoE,
        not a per-training cost.
        
        Upcycling methods:
        - slicing: Zero FLOPs (just memory copy, take first N columns)
        - random_projection: O(H × src × tgt) matmul per weight matrix
        - svd: O(min² × max) per weight matrix (truncated SVD)
        
        Returns:
            dict with upcycling_flops, method, and breakdown
        """
        params = params or self.calculate_params()
        arch = self.architecture
        
        # Get upcycling config
        upcycling_cfg = arch.get("upcycling", {})
        method = upcycling_cfg.get("method", "none")
        
        if method == "none" or not upcycling_cfg:
            return {
                "method": "none",
                "upcycling_flops": 0,
                "upcycling_time_seconds": 0,
                "breakdown": {}
            }
        
        # Source and target dimensions
        hidden = arch.get("hidden_size", 4096)
        source_intermediate = upcycling_cfg.get(
            "source_intermediate_size", 
            arch.get("intermediate_size", arch.get("ffn_intermediate_size", hidden * 4))
        )
        target_intermediate = upcycling_cfg.get(
            "target_intermediate_size",
            arch.get("moe_intermediate_size", arch.get("expert_intermediate_size", 1024))
        )
        num_experts = upcycling_cfg.get(
            "num_experts_to_create",
            arch.get("num_routed_experts", arch.get("num_real_experts", 1))
        )
        layers = arch.get("num_layers", 32)
        
        # SwiGLU has 3 weight matrices per FFN: gate_proj, up_proj, down_proj
        # For upcycling, we need to shrink each matrix
        num_matrices_per_layer = 3
        
        if method == "slicing":
            # Just memory copy, zero compute FLOPs
            upcycling_flops = 0
            breakdown = {
                "method": "slicing",
                "description": "Zero compute (memory copy only)",
                "flops_per_matrix": 0
            }
            
        elif method == "random_projection":
            # FLOPs = 2 × m × n × k for matrix multiply
            # For each weight matrix: W_expert = W_dense @ R
            # gate_proj/up_proj: (H, source) @ (source, target) → 2 × H × source × target
            # down_proj: (source, H) @ (source, target) → project rows
            flops_per_up_proj = 2 * hidden * source_intermediate * target_intermediate
            flops_per_down_proj = 2 * source_intermediate * hidden * target_intermediate
            
            # Total per expert: 2 up-style + 1 down-style (SwiGLU)
            flops_per_expert = 2 * flops_per_up_proj + flops_per_down_proj
            
            # Total across all layers and experts
            upcycling_flops = flops_per_expert * num_experts * layers
            
            breakdown = {
                "method": "random_projection",
                "description": f"Project {source_intermediate}→{target_intermediate} via random matrix",
                "flops_per_up_proj": flops_per_up_proj,
                "flops_per_down_proj": flops_per_down_proj,
                "flops_per_expert": flops_per_expert,
                "num_experts": num_experts,
                "num_layers": layers
            }
            
        elif method == "svd":
            # Truncated SVD: O(min(m,n)² × max(m,n))
            # For up_proj (H × source): min=H, max=source
            m_up, n_up = hidden, source_intermediate
            min_up, max_up = min(m_up, n_up), max(m_up, n_up)
            svd_flops_up = min_up ** 2 * max_up
            
            # For down_proj (source × H): min=H, max=source
            m_down, n_down = source_intermediate, hidden
            min_down, max_down = min(m_down, n_down), max(m_down, n_down)
            svd_flops_down = min_down ** 2 * max_down
            
            # Reconstruction: U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
            # This is much cheaper than SVD itself, so we approximate total as 1.5× SVD
            reconstruction_multiplier = 1.5
            
            # Total per expert: 2 up-style + 1 down-style (SwiGLU)
            flops_per_expert = (2 * svd_flops_up + svd_flops_down) * reconstruction_multiplier
            
            # Total across all layers and experts
            upcycling_flops = flops_per_expert * num_experts * layers
            
            breakdown = {
                "method": "svd",
                "description": f"Truncated SVD {source_intermediate}→{target_intermediate}",
                "svd_flops_up_proj": svd_flops_up,
                "svd_flops_down_proj": svd_flops_down,
                "flops_per_expert": flops_per_expert,
                "num_experts": num_experts,
                "num_layers": layers
            }
        else:
            # Unknown method
            upcycling_flops = 0
            breakdown = {"method": method, "description": "Unknown method"}
        
        # Estimate time assuming H100 at 989 TFLOPs (BF16)
        # Use 50% efficiency for these small operations
        h100_tflops = 989 * 0.5
        upcycling_time_seconds = upcycling_flops / (h100_tflops * 1e12) if upcycling_flops > 0 else 0
        
        return {
            "method": method,
            "upcycling_flops": upcycling_flops,
            "upcycling_time_seconds": upcycling_time_seconds,
            "source_intermediate": source_intermediate,
            "target_intermediate": target_intermediate,
            "num_experts": num_experts,
            "breakdown": breakdown
        }


def load_config(config_path: str):
    """
    Load and parse JSON configuration file.

    Args:
        config_path: Path to JSON config file

    Returns:
        dict: Parsed configuration

    Exits:
        Exits with code 1 if file not found or JSON parsing fails
    """
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON file: {exc}")
        sys.exit(1)


def normalize_deepspeed_config(config: dict) -> dict:
    """
    Normalize DeepSpeed-style config to internal format.

    Supports both formats:
    - DeepSpeed-style: zero_optimization, activation_checkpointing, bf16 at root level
    - Legacy: everything nested under hardware

    DeepSpeed format example:
        {
            "hardware": { "num_gpus": 8, "tflops_per_gpu": {...} },
            "zero_optimization": { "stage": 2, "offload_optimizer": {"device": "cpu"} },
            "activation_checkpointing": { "partition_activations": true },
            "bf16": { "enabled": true },
            "train_micro_batch_size_per_gpu": 1,
            "mfu": 0.3
        }

    Returns config with "hardware" containing all extracted settings.
    """
    hardware = config.get("hardware", {}).copy()

    # 1. Parse zero_optimization (DeepSpeed-style, or nested in hardware)
    zero_opt = config.get("zero_optimization", hardware.get("zero_optimization", {}))
    if zero_opt:
        # Stage
        if "stage" in zero_opt:
            hardware["zero_stage"] = zero_opt["stage"]

        # Offload settings
        offload_optimizer = zero_opt.get("offload_optimizer", {})
        offload_param = zero_opt.get("offload_param", {})

        if isinstance(offload_optimizer, dict):
            opt_device = offload_optimizer.get("device", "none")
        else:
            opt_device = "none"

        if isinstance(offload_param, dict):
            param_device = offload_param.get("device", "none")
        else:
            param_device = "none"

        # Enable cpu_offload if either is set to cpu/nvme
        if opt_device in ("cpu", "nvme") or param_device in ("cpu", "nvme"):
            hardware["cpu_offload"] = True
            hardware["cpu_offload_config"] = {
                "offload_optimizer": opt_device in ("cpu", "nvme"),
                "offload_params": param_device in ("cpu", "nvme"),
                "offload_gradients": zero_opt.get("offload_gradients", True),
                "gpu_buffer_gb": zero_opt.get("pin_memory", 4.0),
            }

    # 2. Parse activation_checkpointing (DeepSpeed-style, or nested in hardware)
    # Always parse and store settings - activation memory is always calculated now
    act_ckpt = config.get("activation_checkpointing", hardware.get("activation_checkpointing", {}))
    partition_activations = bool(act_ckpt.get("partition_activations", False))
    cpu_checkpointing = bool(act_ckpt.get("cpu_checkpointing", False))
    # checkpoint_factor: 1.0 = no checkpointing (full activations), 0.1 = aggressive checkpointing
    checkpoint_factor = float(act_ckpt.get("checkpoint_factor", 1.0))  # Default: no checkpointing
    
    hardware["_partition_activations"] = partition_activations
    hardware["_checkpoint_factor"] = checkpoint_factor
    hardware["_cpu_checkpointing"] = cpu_checkpointing
    # Always include activation memory in calculations
    hardware["_include_activation_memory"] = True

    # 3. Parse precision (bf16/fp16 at root - DeepSpeed-style)
    bf16_cfg = config.get("bf16", {})
    fp16_cfg = config.get("fp16", {})
    if bf16_cfg.get("enabled", False):
        hardware["_precision"] = "bf16"
    elif fp16_cfg.get("enabled", False):
        hardware["_precision"] = "fp16"

    # 4. Parse training settings (root level or nested in hardware)
    if "train_micro_batch_size_per_gpu" in config:
        hardware["_micro_batch_size"] = config["train_micro_batch_size_per_gpu"]
    elif "train_micro_batch_size_per_gpu" in hardware:
        hardware["_micro_batch_size"] = hardware["train_micro_batch_size_per_gpu"]
    if "gradient_accumulation_steps" in config:
        hardware["_gradient_accumulation_steps"] = config["gradient_accumulation_steps"]
    elif "gradient_accumulation_steps" in hardware:
        hardware["_gradient_accumulation_steps"] = hardware["gradient_accumulation_steps"]

    # 5. Parse mfu from root level (our addition)
    if "mfu" in config and "mfu" not in hardware:
        hardware["mfu"] = config["mfu"]

    # 6. Set defaults if not present
    hardware.setdefault("mfu", 0.3)
    hardware.setdefault("zero_stage", 2)
    hardware.setdefault("cpu_offload", False)
    hardware.setdefault("price_per_gpu_hour", 2.5)

    # Update config with normalized hardware
    config["hardware"] = hardware
    return config


def get_zero_efficiency(
    zero_stage: int, cpu_offload: bool, zero_efficiency_cfg: dict
) -> float:
    """
    Get the throughput efficiency multiplier for a ZeRO configuration.

    ZeRO stages have different overheads:
    - ZeRO-0: No sharding, no overhead (baseline)
    - ZeRO-2: Shard optimizer + gradients, minimal overhead
    - ZeRO-3: Shard params too, requires all-gather before each forward/backward
    - ZeRO-Infinity: CPU/NVMe offload, significant PCIe bandwidth bottleneck
    """
    defaults = {
        "zero0": 1.0,  # Baseline (data parallel only)
        "zero2": 0.95,  # ~5% overhead for gradient sharding
        "zero3": 0.70,  # ~30% overhead for param all-gather
        "zero_infinity": 0.25,  # ~75% overhead for CPU offload (PCIe bottleneck)
    }

    if cpu_offload:
        return zero_efficiency_cfg.get("zero_infinity", defaults["zero_infinity"])

    key = f"zero{zero_stage}"
    return zero_efficiency_cfg.get(key, defaults.get(key, 0.70))


def get_scaling_efficiency(num_gpus: int, scaling_cfg: dict) -> float:
    """
    Get the parallel efficiency multiplier for a given GPU count.

    At larger GPU counts, communication overhead increases:
    - All-reduce time scales with ring size
    - Gradient synchronization becomes a bottleneck
    - Network bandwidth limits throughput

    We use log-interpolation between reference points.
    """
    import math

    defaults = {
        "base_gpus": 8,
        "efficiency_at_base": 1.0,
        "efficiency_at_64": 0.90,
        "efficiency_at_256": 0.80,
        "efficiency_at_1024": 0.65,
    }

    base = scaling_cfg.get("base_gpus", defaults["base_gpus"])

    # Reference points: (gpu_count, efficiency)
    points = [
        (base, scaling_cfg.get("efficiency_at_base", defaults["efficiency_at_base"])),
        (64, scaling_cfg.get("efficiency_at_64", defaults["efficiency_at_64"])),
        (256, scaling_cfg.get("efficiency_at_256", defaults["efficiency_at_256"])),
        (1024, scaling_cfg.get("efficiency_at_1024", defaults["efficiency_at_1024"])),
    ]

    if num_gpus <= base:
        return points[0][1]

    # Log-linear interpolation between points
    for i in range(len(points) - 1):
        g1, e1 = points[i]
        g2, e2 = points[i + 1]
        if g1 <= num_gpus <= g2:
            # Log interpolation
            t = math.log(num_gpus / g1) / math.log(g2 / g1)
            return e1 + t * (e2 - e1)

    # Extrapolate beyond 1024 GPUs (continue the trend)
    g1, e1 = points[-2]
    g2, e2 = points[-1]
    t = math.log(num_gpus / g1) / math.log(g2 / g1)
    efficiency = e1 + t * (e2 - e1)
    return max(0.30, efficiency)  # Floor at 30% efficiency


def apply_growth_allocation(stages: list[TrainingStage], growth_cfg: dict) -> None:
    """
    Allocate tokens to stages using growth/expansion mode.

    Implements the "paper" growth strategy where total compute budget equals
    the FLOPs required to train the largest stage from scratch. Smaller stages
    receive proportional token allocations.

    Uses attention-aware FLOPs formula: F = (6*S*N_linear + 12*L*H*S²) * (T/S)

    If a stage has 'actual_tokens' set in architecture, that value is used
    directly (user-defined stabilization point). Otherwise, tokens are allocated
    from the remaining budget.

    Args:
        stages: List of TrainingStage objects to allocate tokens to
        growth_cfg: Growth configuration dict with 'mode' key

    Raises:
        ValueError: If growth mode is not 'paper'
    """
    if not stages:
        return

    mode = str(growth_cfg.get("mode", "none")).strip().lower()
    if mode in ("none", "off", "false", "0", ""):
        return
    if mode != "paper":
        raise ValueError("Only growth mode 'paper' is supported in this script.")

    # Find the largest stage (by active params) to set the budget
    largest_stage = max(stages, key=lambda s: s.calculate_params()["active_params"])

    # Budget = FLOPs to train the largest stage from scratch (using attention-aware formula)
    total_budget_flops = largest_stage.calculate_flops()

    remaining_flops = total_budget_flops
    for stage in stages:
        # Check if user has set actual_tokens (explicit stabilization point)
        actual_tokens = stage.architecture.get("actual_tokens")

        if actual_tokens is not None and actual_tokens > 0:
            # User-defined: use actual_tokens directly
            stage.total_tokens = float(actual_tokens)
            stage_flops = stage.calculate_flops()
            remaining_flops -= stage_flops
        else:
            # Budget allocation: calculate how many tokens this stage can afford
            params = stage.calculate_params()
            flops_per_token = stage.flops_per_token(params)

            # Max tokens this stage can afford with remaining budget
            t = remaining_flops / flops_per_token
            # Cap at max_tokens from config
            t = min(t, stage.total_tokens)
            stage.total_tokens = t

            stage_flops = flops_per_token * t
            remaining_flops -= stage_flops

        if remaining_flops <= 0:
            remaining_flops = 0


def estimate_communication_time(
    stage: TrainingStage,
    params: dict,
    hardware: dict,
    quantization: str,
    num_gpus: int,
    cpu_offload: bool,
) -> dict:
    arch = stage.architecture
    training_cfg = arch.get("training") or {}
    precision_cfg = arch.get("precision") or {}
    comm_cfg = hardware.get("communication", {}) or {}
    parallel_cfg = hardware.get("parallelism", {}) or {}

    micro_batch = float(training_cfg.get("micro_batch_size", 1))
    grad_accum = float(training_cfg.get("gradient_accumulation_steps", 1))
    seq_len = float(
        training_cfg.get(
            "seq_length",
            arch.get("sequence_length", arch.get("max_position_embeddings", 4096)),
        )
    )

    tp = int(parallel_cfg.get("tensor_parallel_size", parallel_cfg.get("tp", 1)))
    pp = int(parallel_cfg.get("pipeline_parallel_size", parallel_cfg.get("pp", 1)))
    ep = int(parallel_cfg.get("expert_parallel_size", parallel_cfg.get("ep", 1)))
    dp = parallel_cfg.get("data_parallel_size")
    if dp is None:
        denom = max(1, tp * pp * ep)
        dp = max(1, int(num_gpus // denom))

    if micro_batch <= 0 or seq_len <= 0 or dp <= 0:
        return {"comm_time_s": 0.0}

    tokens_per_micro_step_per_gpu = micro_batch * seq_len
    micro_steps = stage.total_tokens / (tokens_per_micro_step_per_gpu * dp)
    optimizer_steps = micro_steps / max(1.0, grad_accum)

    # Gradient bytes per param
    grad_bytes = precision_cfg.get("gradient_bytes_per_param")
    if grad_bytes is None:
        grad_precision = precision_cfg.get("gradient_precision", "fp32")
        grad_bytes = bytes_for_precision(grad_precision)
    gradient_bytes_total = params["total_params"] * float(grad_bytes)

    # DP communication (approximate all-reduce)
    dp_comm_multiplier = float(comm_cfg.get("dp_comm_multiplier", 1.0))
    dp_comm_bytes_per_step = 0.0
    if dp > 1:
        dp_comm_bytes_per_step = (
            gradient_bytes_total * (2 * (dp - 1) / dp) * dp_comm_multiplier
        )
    dp_bandwidth = comm_cfg.get("dp_bandwidth_gbps")
    dp_latency = float(comm_cfg.get("dp_latency_ms", 0.0)) / 1000.0
    dp_comm_time = 0.0
    if dp_comm_bytes_per_step > 0 and dp_bandwidth:
        dp_comm_time = optimizer_steps * (
            dp_comm_bytes_per_step / (float(dp_bandwidth) * 1e9) + dp_latency
        )

    # EP communication (approximate dispatch + combine)
    ep_comm_multiplier = float(comm_cfg.get("ep_comm_multiplier", 1.0))
    top_k = arch.get("top_k_experts", arch.get("top_k", 1))
    hidden = float(arch.get("hidden_size", 2048))
    act_bytes = training_cfg.get("activation_bytes_per_element")
    if act_bytes is None:
        act_prec = training_cfg.get(
            "activation_precision", precision_cfg.get("weight_precision", "bf16")
        )
        act_bytes = bytes_for_precision(act_prec)
    ep_comm_bytes_per_micro = 0.0
    if ep > 1 and top_k > 0:
        ep_fraction = (ep - 1) / ep
        ep_comm_bytes_per_micro = (
            2
            * top_k
            * tokens_per_micro_step_per_gpu
            * hidden
            * act_bytes
            * ep_fraction
            * ep_comm_multiplier
        )
    ep_bandwidth = comm_cfg.get("ep_bandwidth_gbps")
    ep_latency = float(comm_cfg.get("ep_latency_ms", 0.0)) / 1000.0
    ep_comm_time = 0.0
    if ep_comm_bytes_per_micro > 0 and ep_bandwidth:
        ep_comm_time = micro_steps * (
            ep_comm_bytes_per_micro / (float(ep_bandwidth) * 1e9) + ep_latency
        )

    # Optional offload communication model (approximate)
    offload_comm_time = 0.0
    offload_bandwidth = comm_cfg.get("offload_bandwidth_gbps")
    offload_latency = float(comm_cfg.get("offload_latency_ms", 0.0)) / 1000.0
    if cpu_offload and offload_bandwidth:
        offload_bytes_per_step = comm_cfg.get("offload_bytes_per_step")
        if offload_bytes_per_step is None:
            weight_bytes = precision_cfg.get("weight_bytes_per_param")
            if weight_bytes is None:
                weight_precision = precision_cfg.get("weight_precision")
                if weight_precision and str(weight_precision).lower() not in (
                    "auto",
                    "default",
                    "",
                ):
                    weight_bytes = bytes_for_precision(weight_precision)
                else:
                    q = str(quantization).lower()
                    weight_bytes = 1 if q == "fp8" else 0.5 if q == "nvfp4" else 2
            opt_state_bytes = precision_cfg.get("optimizer_state_bytes_per_param")
            if opt_state_bytes is None:
                opt_precision = precision_cfg.get("optimizer_precision", "fp32")
                opt_states = int(precision_cfg.get("optimizer_states_count", 2))
                opt_state_bytes = opt_states * bytes_for_precision(opt_precision)
            offload_bytes_per_step = (
                params["total_params"]
                * (float(weight_bytes) + float(opt_state_bytes) + float(grad_bytes))
                / max(1, num_gpus)
            )
        offload_comm_time = optimizer_steps * (
            float(offload_bytes_per_step) / (float(offload_bandwidth) * 1e9)
            + offload_latency
        )

    total_comm_time = dp_comm_time + ep_comm_time + offload_comm_time
    return {
        "comm_time_s": total_comm_time,
        "dp_comm_time_s": dp_comm_time,
        "ep_comm_time_s": ep_comm_time,
        "offload_comm_time_s": offload_comm_time,
    }


def main() -> None:
    """
    Main Function for FLOPS calculator.

    Loads configuration, calculates parameters and FLOPs for all training stages,
    and displays results in a formatted table with:
    - Token counts
    - Parameter counts (total and active)
    - Memory per GPU estimates
    - Total FLOPs (ZFLOPs)
    - Training duration (days)
    - Cost estimates

    Supports multiple precision modes (BF16, FP8, NVFP4) and various ZeRO stages.
    """
    parser = argparse.ArgumentParser(
        description="Calculate training FLOPs and duration (growth/expansion mode)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--upcycling-only",
        action="store_true",
        help="Only show upcycling cost comparison (no training tables)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = normalize_deepspeed_config(config)  # Support DeepSpeed-style configs

    try:
        hardware = config["hardware"]
        num_gpus = hardware["num_gpus"]
        mfu = hardware["mfu"]
        price_per_gpu_hour = hardware["price_per_gpu_hour"]
        precision_specs = hardware["tflops_per_gpu"]
        selected_quantization = hardware.get("quantization", "all").lower()
        tflops_mode = hardware.get("tflops_mode", "dense")
        zero_stage = hardware.get("zero_stage", 2)  # Default ZeRO-2
        cpu_offload = hardware.get("cpu_offload", False)  # ZeRO-Infinity
        expert_parallel_size = int(hardware.get("expert_parallel_size", hardware.get("ep", 1)))

        # Efficiency configs (can be overridden in config.json)
        zero_efficiency_cfg = hardware.get("zero_efficiency", {})
        scaling_cfg = hardware.get("scaling_efficiency", {})

        # Calculate efficiency multipliers
        zero_eff = get_zero_efficiency(zero_stage, cpu_offload, zero_efficiency_cfg)
        scaling_eff = get_scaling_efficiency(num_gpus, scaling_cfg)
        performance_cfg = hardware.get("performance", {})
        use_explicit_comm_model = bool(
            performance_cfg.get("use_explicit_comm_model", False)
        )
        compute_mfu = float(performance_cfg.get("compute_mfu", mfu))

        stages = []
        for stage_conf in config["stages"]:
            stages.append(
                TrainingStage(
                    name=stage_conf["name"],
                    total_tokens=float(stage_conf["total_tokens"]),
                    architecture=stage_conf["architecture"],
                    description=stage_conf.get("description", ""),
                )
            )
    except KeyError as e:
        print(f"Error: Missing required configuration key: {e}")
        print("Please ensure flops_config_growth.json contains all necessary fields.")
        sys.exit(1)

    growth_cfg = config.get("growth", {"mode": "none"})
    apply_growth_allocation(stages, growth_cfg)

    # =========================================================================
    # Upcycling-Only Mode: Show just upcycling cost comparison
    # =========================================================================
    if args.upcycling_only:
        print("=" * 90)
        print("MoE UPCYCLING COST COMPARISON (One-Time Dense→MoE Conversion)")
        print("=" * 90)
        print()
        
        # Check if any stage has upcycling configured
        upcycling_stages = []
        for stage in stages:
            upcycling_info = stage.calculate_upcycling_flops()
            if upcycling_info["method"] != "none":
                upcycling_stages.append((stage, upcycling_info))
        
        if not upcycling_stages:
            print("⚠️  No upcycling configuration found in any stage.")
            print("   Add 'upcycling' block to architecture with 'method': 'slicing|random_projection|svd'")
            print()
            return
        
        # Header
        print(f"{'Stage Name':<35} | {'Method':<18} | {'FLOPs':<18} | {'Time (H100)':<12} | {'Cost':<10}")
        print("-" * 100)
        
        total_upcycling_flops = 0
        total_upcycling_time = 0
        
        for stage, upcycling_info in upcycling_stages:
            method = upcycling_info["method"]
            flops = upcycling_info["upcycling_flops"]
            time_s = upcycling_info["upcycling_time_seconds"]
            src = upcycling_info.get("source_intermediate", "?")
            tgt = upcycling_info.get("target_intermediate", "?")
            n_experts = upcycling_info.get("num_experts", "?")
            
            total_upcycling_flops += flops
            total_upcycling_time += time_s
            
            # Format FLOPs nicely
            if flops >= 1e12:
                flops_str = f"{flops/1e12:.2f} TFLOPs"
            elif flops >= 1e9:
                flops_str = f"{flops/1e9:.2f} GFLOPs"
            elif flops >= 1e6:
                flops_str = f"{flops/1e6:.2f} MFLOPs"
            else:
                flops_str = "0 (copy)" if flops == 0 else f"{flops:.0f} FLOPs"
            
            # Format time
            if time_s >= 60:
                time_str = f"{time_s/60:.1f} min"
            elif time_s >= 1:
                time_str = f"{time_s:.1f} s"
            else:
                time_str = f"{time_s*1000:.1f} ms" if time_s > 0 else "~0"
            
            stage_name = stage.name[:34] if len(stage.name) > 34 else stage.name
            
            # Calculate cost (time × price_per_gpu_hour)
            cost = time_s * price_per_gpu_hour / 3600  # Convert seconds to hours
            cost_str = f"${cost:.4f}" if cost > 0 else "$0"
            
            print(f"{stage_name:<35} | {method:<18} | {flops_str:<18} | {time_str:<12} | {cost_str:<10}")
            print(f"  └─ FFN: {src}→{tgt} × {n_experts} experts, {stage.architecture.get('num_layers', '?')} layers")
        
        print("-" * 100)
        
        # Total
        if total_upcycling_flops >= 1e12:
            total_flops_str = f"{total_upcycling_flops/1e12:.2f} TFLOPs"
        elif total_upcycling_flops >= 1e9:
            total_flops_str = f"{total_upcycling_flops/1e9:.2f} GFLOPs"
        else:
            total_flops_str = f"{total_upcycling_flops/1e6:.2f} MFLOPs" if total_upcycling_flops > 0 else "0"
        
        if total_upcycling_time >= 60:
            total_time_str = f"{total_upcycling_time/60:.1f} min"
        elif total_upcycling_time >= 1:
            total_time_str = f"{total_upcycling_time:.1f} s"
        else:
            total_time_str = f"{total_upcycling_time*1000:.1f} ms" if total_upcycling_time > 0 else "~0"
        
        total_cost = total_upcycling_time * price_per_gpu_hour / 3600
        total_cost_str = f"${total_cost:.4f}" if total_cost > 0 else "$0"
        
        print(f"{'TOTAL':<35} | {'─':<18} | {total_flops_str:<18} | {total_time_str:<12} | {total_cost_str:<10}")
        print("=" * 100)
        print()
        
        # Summary comparison
        print("COST COMPARISON SUMMARY:")
        print("─" * 50)
        slicing_time = sum(u["upcycling_time_seconds"] for _, u in upcycling_stages if u["method"] == "slicing")
        rp_time = sum(u["upcycling_time_seconds"] for _, u in upcycling_stages if u["method"] == "random_projection")
        svd_time = sum(u["upcycling_time_seconds"] for _, u in upcycling_stages if u["method"] == "svd")
        
        methods_found = []
        if any(u["method"] == "slicing" for _, u in upcycling_stages):
            methods_found.append(("Slicing", 0, "~0 (memory copy only)"))
        if any(u["method"] == "random_projection" for _, u in upcycling_stages):
            rp_flops = sum(u["upcycling_flops"] for _, u in upcycling_stages if u["method"] == "random_projection")
            methods_found.append(("Random Projection", rp_flops, f"{rp_time*1000:.1f} ms" if rp_time < 1 else f"{rp_time:.1f} s"))
        if any(u["method"] == "svd" for _, u in upcycling_stages):
            svd_flops = sum(u["upcycling_flops"] for _, u in upcycling_stages if u["method"] == "svd")
            methods_found.append(("SVD", svd_flops, f"{svd_time*1000:.1f} ms" if svd_time < 1 else f"{svd_time:.1f} s"))
        
        for method_name, flops, time_str in methods_found:
            if flops >= 1e12:
                flops_str = f"{flops/1e12:.2f} TFLOPs"
            elif flops >= 1e9:
                flops_str = f"{flops/1e9:.2f} GFLOPs"
            else:
                flops_str = "0 FLOPs" if flops == 0 else f"{flops/1e6:.2f} MFLOPs"
            print(f"  • {method_name:<20}: {flops_str:<18} | {time_str}")
        
        print()
        print("NOTE: All methods produce the SAME MoE architecture.")
        print("      Choose based on quality vs compute tradeoff:")
        print("      - Slicing: Fastest, may lose information")
        print("      - Random Projection: Fast, preserves structure stochastically")  
        print("      - SVD: Slowest, best information preservation")
        print()
        return

    # Effective MFU after all overheads (or compute MFU if explicit comm model)
    effective_mfu = compute_mfu if use_explicit_comm_model else mfu * zero_eff * scaling_eff
    offload_str = " + CPU Offload" if cpu_offload else ""

    print("=" * 120)
    print("ERA V4 Training Compute & Schedule Calculator (Growth/Expansion Mode)")
    print(
        f"Cluster: {num_gpus}x NVIDIA GPUs | Base MFU: {mfu*100:.0f}% | Price/GPU/Hr: ${price_per_gpu_hour}"
    )
    print(
        f"ZeRO Stage: {zero_stage}{offload_str} | ZeRO Eff: {zero_eff*100:.0f}% | Scaling Eff: {scaling_eff*100:.0f}% | Effective MFU: {effective_mfu*100:.1f}%"
    )
    print("=" * 120)
    print("\nNOTE: Parameters are calculated dynamically from architecture specs.")

    for precision, peak_tflops in precision_specs.items():
        if (
            selected_quantization != "all"
            and precision.lower() != selected_quantization
        ):
            continue

        # Apply all efficiency factors to effective throughput
        effective_flops_per_sec = num_gpus * (peak_tflops * 1e12) * effective_mfu

        print("-" * 120)
        print(
            f"--- Precision: {precision.upper()} (Peak/GPU: {peak_tflops} TFLOPS) ---"
        )
        print(
            f"Effective Cluster Performance: {effective_flops_per_sec/1e15:.2f} PFLOPS (after all overheads)"
        )
        print(f"{'-'*120}")
        if cpu_offload:
            print(
                f"{'Stage':<36} | {'Tokens (B)':<10} | {'Total Params':<12} | {'Active Params':<12} | {'Mem/GPU (GB)':<12} | {'Mem/CPU (GB)':<12} | {'ZFLOPs':<7} | {'Days':<6} | {'Cost ($)':<12}"
            )
        else:
            print(
                f"{'Stage':<36} | {'Tokens (B)':<10} | {'Total Params':<12} | {'Active Params':<12} | {'Mem/GPU (GB)':<12} | {'ZFLOPs':<7} | {'Days':<6} | {'Cost ($)':<12}"
            )
        print(f"{'-'*136}")

        total_flops = 0.0
        total_seconds = 0.0

        for stage in stages:
            params = stage.calculate_params()
            stage_flops = stage.calculate_flops(params)
            if use_explicit_comm_model:
                compute_seconds = stage_flops / effective_flops_per_sec
                comm_info = estimate_communication_time(
                    stage, params, hardware, precision, num_gpus, cpu_offload
                )
                stage_seconds = compute_seconds + comm_info["comm_time_s"]
            else:
                stage_seconds = stage_flops / effective_flops_per_sec
            stage_days = stage_seconds / (24 * 3600)

            stage_hours = stage_seconds / 3600
            stage_cost = stage_hours * num_gpus * price_per_gpu_hour

            total_flops += stage_flops
            total_seconds += stage_seconds

            # Calculate memory per GPU (varies by quantization)
            mem_info = stage.calculate_memory_per_gpu(
                params,
                num_gpus,
                zero_stage,
                precision,
                cpu_offload,
                hardware.get("cpu_offload_config"),
                expert_parallel_size,
                hardware.get("_checkpoint_factor"),
                hardware.get("_include_activation_memory"),
                hardware.get("_micro_batch_size"),
                hardware.get("_partition_activations", False),
            )
            mem_per_gpu = mem_info["memory_per_gpu_gb"]

            # Warning indicator if memory exceeds H100 80GB
            mem_warn = "⚠️" if mem_per_gpu > 80 else ""

            # Format memory string(s)
            mem_str_gpu = f"{mem_per_gpu:<6.1f}{mem_warn:<2}"
            if cpu_offload:
                cpu_mem_gb = mem_info.get("cpu_memory_per_gpu_gb", 0)
                mem_str_cpu = f"{cpu_mem_gb:<6.1f}"
                stage_name = stage.name[:35] if len(stage.name) > 35 else stage.name
                print(
                    f"{stage_name:<36} | {stage.total_tokens/1e9:<10.1f} | {params['total_params']/1e9:<5.2f} B       | {params['active_params']/1e9:<5.2f} B       | {mem_str_gpu:<12} | {mem_str_cpu:<12} | {stage_flops/1e21:<7.2f} | {stage_days:<6.2f} | ${stage_cost:,.0f}"
                )
            else:
                stage_name = stage.name[:35] if len(stage.name) > 35 else stage.name
                print(
                    f"{stage_name:<36} | {stage.total_tokens/1e9:<10.1f} | {params['total_params']/1e9:<5.2f} B       | {params['active_params']/1e9:<5.2f} B       | {mem_str_gpu:<12} | {stage_flops/1e21:<7.2f} | {stage_days:<6.2f} | ${stage_cost:,.0f}"
                )

        total_days = total_seconds / (24 * 3600)
        total_hours = total_seconds / 3600
        total_cost = total_hours * num_gpus * price_per_gpu_hour

        print(f"{'-'*136}")
        if cpu_offload:
            print(
                f"{'TOTAL':<36} | {'-':<10} | {'-':<12} | {'-':<12} | {'-':<12} | {'-':<12} | {total_flops/1e21:<7.2f} | {total_days:<6.2f} | ${total_cost:,.0f}"
            )
        else:
            print(
                f"{'TOTAL':<36} | {'-':<10} | {'-':<12} | {'-':<12} | {'-':<12} | {total_flops/1e21:<7.2f} | {total_days:<6.2f} | ${total_cost:,.0f}"
            )
        print(f"{'='*136}")
        
        # =====================================================================
        # Upcycling Cost (One-Time)
        # =====================================================================
        # Check if any stage has upcycling configured
        upcycling_stages = []
        for stage in stages:
            upcycling_info = stage.calculate_upcycling_flops()
            if upcycling_info["method"] != "none":
                upcycling_stages.append((stage, upcycling_info))
        
        if upcycling_stages:
            print(f"\n{'─'*80}")
            print(f"UPCYCLING COST (One-Time Dense→MoE Conversion)")
            print(f"{'─'*80}")
            
            total_upcycling_flops = 0
            total_upcycling_time = 0
            
            for stage, upcycling_info in upcycling_stages:
                method = upcycling_info["method"]
                flops = upcycling_info["upcycling_flops"]
                time_s = upcycling_info["upcycling_time_seconds"]
                src = upcycling_info.get("source_intermediate", "?")
                tgt = upcycling_info.get("target_intermediate", "?")
                n_experts = upcycling_info.get("num_experts", "?")
                
                total_upcycling_flops += flops
                total_upcycling_time += time_s
                
                # Format FLOPs nicely
                if flops >= 1e12:
                    flops_str = f"{flops/1e12:.2f} TFLOPs"
                elif flops >= 1e9:
                    flops_str = f"{flops/1e9:.2f} GFLOPs"
                elif flops >= 1e6:
                    flops_str = f"{flops/1e6:.2f} MFLOPs"
                else:
                    flops_str = f"{flops:.0f} FLOPs" if flops > 0 else "0 (memory copy)"
                
                # Format time
                if time_s >= 60:
                    time_str = f"{time_s/60:.1f} min"
                elif time_s >= 1:
                    time_str = f"{time_s:.1f} s"
                else:
                    time_str = f"{time_s*1000:.1f} ms" if time_s > 0 else "~0"
                
                stage_name = stage.name[:30] if len(stage.name) > 30 else stage.name
                print(f"  {stage_name:<32} | Method: {method:<18} | {flops_str:<18} | Est. Time: {time_str}")
                print(f"  {'':<32} | FFN: {src}→{tgt} × {n_experts} experts")
            
            print(f"{'─'*80}")
            
            # Total
            if total_upcycling_flops >= 1e12:
                total_flops_str = f"{total_upcycling_flops/1e12:.2f} TFLOPs"
            elif total_upcycling_flops >= 1e9:
                total_flops_str = f"{total_upcycling_flops/1e9:.2f} GFLOPs"
            else:
                total_flops_str = f"{total_upcycling_flops/1e6:.2f} MFLOPs" if total_upcycling_flops > 0 else "0"
            
            if total_upcycling_time >= 60:
                total_time_str = f"{total_upcycling_time/60:.1f} min"
            elif total_upcycling_time >= 1:
                total_time_str = f"{total_upcycling_time:.1f} s"
            else:
                total_time_str = f"{total_upcycling_time*1000:.1f} ms" if total_upcycling_time > 0 else "~0"
            
            print(f"  {'TOTAL UPCYCLING':<32} | {total_flops_str:<18} | Est. Time: {total_time_str} (H100 @ 50% util)")
            print(f"{'─'*80}")
        print()


if __name__ == "__main__":
    main()
