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
        position_cfg = arch.get("position") or {}
        position_cfg = arch.get("position") or {}
        expert_cfg = arch.get("expert") or {}
        head_cfg = arch.get("head") or {}
        position_cfg = arch.get("position") or {}

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

        embedding_params = vocab * hidden
        lm_head_multiplier = arch.get(
            "lm_head_multiplier", head_cfg.get("lm_head_multiplier")
        )
        if lm_head_multiplier is None and head_cfg.get(
            "use_multi_token_prediction", False
        ):
            lm_head_multiplier = head_cfg.get(
                "num_prediction_heads", head_cfg.get("mtp_heads", 2)
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

        gsa_enabled = attention_type in (
            "gsa",
            "gated_sparse",
            "gated_sparse_attention",
            "deepseek_gsa",
        ) or bool(arch.get("use_gsa", False))
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
                hidden * indexer_heads * indexer_dim * 2 + hidden * indexer_heads
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
                attn_params_per_layer += hidden * hidden * kv_ratio
            if use_output_gate:
                attn_params_per_layer += hidden * hidden

        attn_params_per_layer += float(
            arch.get(
                "attn_extra_params_per_layer",
                attention_cfg.get("attn_extra_params_per_layer", 0),
            )
        )

        moe_intermediate = arch.get(
            "moe_intermediate_size", expert_cfg.get("intermediate_size", intermediate)
        )
        ffn_params_dense = 3 * hidden * intermediate
        ffn_params_per_expert = 3 * hidden * moe_intermediate

        total_ffn_params_moe = 0
        active_ffn_params_moe = 0
        router_params = 0
        shared_expert_params = num_shared_experts * ffn_params_per_expert
        null_expert_params_per_expert = arch.get(
            "null_expert_params_per_expert",
            router_cfg.get("null_expert_params_per_expert", 0),
        )
        null_expert_params = num_null_experts * null_expert_params_per_expert
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
                top_k * ffn_params_per_expert + shared_expert_params + router_params
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
                top_k * ffn_params_per_expert + shared_expert_params + router_params
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
                top_k * ffn_params_per_expert + shared_expert_params + router_params
            )
            derived_experts = experts

        if experts > 0 and top_k > experts:
            raise ValueError("top_k_experts cannot exceed num_experts.")

        total_params = (
            embedding_params
            + lm_head_params
            + layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * total_ffn_params_moe
        )

        active_non_embed_params = (
            layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * active_ffn_params_moe
            + lm_head_params
        )
        active_linear_params = (
            layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * active_ffn_params_moe
            + lm_head_params_for_flops
        )
        active_params_base = embedding_params + active_non_embed_params

        params_null_path = (
            embedding_params
            + layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * (router_params + shared_expert_params)
            + lm_head_params
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

        return {
            "total_params": total_params,
            "active_params": effective_active_params,
            "active_non_embed_params": effective_active_non_embed_params,
            "active_linear_params": effective_active_linear_params,
            "embedding_params": embedding_params,
            "lm_head_params": lm_head_params,
            "num_moe_layers": num_moe_layers,
            "dense_layers": dense_layers,
            "num_experts": experts,
            "derived_num_experts": derived_experts,
        }

    def calculate_memory_per_gpu(
        self,
        params: Optional[dict] = None,
        num_gpus: int = 8,
        zero_stage: int = 2,
        quantization: str = "bf16",
        cpu_offload: bool = False,
    ) -> dict:
        """
        Estimate memory footprint per GPU for training this stage.

        Assumptions:
        - Model weights: Depends on quantization (BF16=2, FP8=1, NVFP4=0.5 bytes/param)
        - FP32 optimizer states (Adam): 8 bytes/param (momentum + variance, always FP32)
        - FP32 gradients: 4 bytes/param (gradient accumulation typically FP32)
        - Activations are NOT included (highly batch-size dependent)

        ZeRO Stages:
        - ZeRO-0: No sharding (baseline)
        - ZeRO-2: Shard optimizer states + gradients
        - ZeRO-3: Shard everything (params + optimizer + gradients)
        - ZeRO-Infinity (cpu_offload=True): Move everything to CPU/NVMe, keep minimal buffer on GPU.
        """
        params = params or self.calculate_params()
        total_params = params["total_params"]
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

        # Activation memory (optional, depends on batch size)
        include_activation_memory = bool(
            training_cfg.get("include_activation_memory", False)
        )
        activation_bytes = 0.0
        if include_activation_memory:
            micro_batch = float(training_cfg.get("micro_batch_size", 1))
            seq_len = float(
                training_cfg.get(
                    "seq_length",
                    arch.get("sequence_length", arch.get("max_position_embeddings", 4096)),
                )
            )
            hidden = float(arch.get("hidden_size", 2048))
            layers = float(arch.get("num_layers", arch.get("num_hidden_layers", 24)))
            activation_multiplier = float(training_cfg.get("activation_multiplier", 2.0))
            act_bytes = training_cfg.get("activation_bytes_per_element")
            if act_bytes is None:
                act_prec = training_cfg.get("activation_precision", "bf16")
                act_bytes = bytes_for_precision(act_prec)
            activation_bytes = (
                micro_batch * seq_len * hidden * layers * activation_multiplier * act_bytes
            )

        cpu_memory_gb = 0.0

        if cpu_offload:
            # ZeRO-Infinity Logic
            # 1. GPU Memory: Fixed buffer size (e.g., 2GB) + minimal active layer overhead
            #    It does NOT scale with model size in the same way.
            #    We assign a fixed buffer of ~4GB per GPU as a safe "working memory" estimate for large models.
            memory_bytes = 4 * (1024**3)

            # 2. CPU/NVMe Memory: Holds all Parameters + Optimizer States + Gradients
            #    If multiple nodes, this is sharded across nodes.
            #    We assume "num_gpus" implies logical GPUs in the cluster.
            #    Total system memory required = Total Model State
            total_system_mem_bytes = (
                model_bytes + master_bytes + optimizer_bytes + gradient_bytes
            )

            # CPU RAM required per GPU (assuming perfect sharding across available host RAM)
            # In reality, you need enough CPU RAM on the node to hold the shard.
            cpu_memory_bytes = total_system_mem_bytes / num_gpus
            cpu_memory_gb = cpu_memory_bytes / (1024**3)

        elif zero_stage == 0:
            # No sharding - full memory on each GPU
            memory_bytes = (
                model_bytes
                + master_bytes
                + optimizer_bytes
                + gradient_bytes
                + activation_bytes
            )
        elif zero_stage == 2:
            # ZeRO-2: Shard optimizer + gradients, replicate model
            memory_bytes = (
                model_bytes
                + master_bytes
                + (optimizer_bytes + gradient_bytes) / num_gpus
                + activation_bytes
            )
        elif zero_stage == 3:
            # ZeRO-3: Shard everything
            memory_bytes = (
                model_bytes + master_bytes + optimizer_bytes + gradient_bytes
            ) / num_gpus + activation_bytes
        else:
            # Default to ZeRO-2
            memory_bytes = (
                model_bytes
                + master_bytes
                + (optimizer_bytes + gradient_bytes) / num_gpus
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

        gsa_enabled = attention_type in (
            "gsa",
            "gated_sparse",
            "gated_sparse_attention",
            "deepseek_gsa",
        ) or bool(arch.get("use_gsa", False))

        # Sparse attention configuration
        use_sparse_attn = bool(arch.get("use_sparse_attention", False)) or gsa_enabled
        sparse_k_tokens = arch.get("sparse_k_tokens", s)

        if gsa_enabled:
            gsa_k_tokens = arch.get("gsa_k_tokens", attention_cfg.get("gsa_k_tokens"))
            gsa_k_base = arch.get("gsa_k_base", attention_cfg.get("gsa_k_base"))
            gsa_k_min = arch.get("gsa_k_min", attention_cfg.get("gsa_k_min", gsa_k_base))
            gsa_k_max = arch.get("gsa_k_max", attention_cfg.get("gsa_k_max", gsa_k_base))
            if gsa_k_tokens is not None:
                sparse_k_tokens = gsa_k_tokens
            elif gsa_k_base is not None:
                k = gsa_k_base
                if bool(
                    arch.get(
                        "gsa_use_adaptive_k",
                        attention_cfg.get("gsa_use_adaptive_k", False),
                    )
                ):
                    if gsa_k_min is not None:
                        k = max(k, gsa_k_min)
                    if gsa_k_max is not None:
                        k = min(k, gsa_k_max)
                sparse_k_tokens = k

        if gsa_enabled:
            indexer_heads = arch.get(
                "gsa_num_indexer_heads",
                attention_cfg.get("gsa_num_indexer_heads", arch.get("indexer_heads", 4)),
            )
            indexer_dim = arch.get(
                "gsa_indexer_dim",
                attention_cfg.get("gsa_indexer_dim", arch.get("indexer_dim", 64)),
            )
        else:
            indexer_heads = arch.get("indexer_heads", attention_cfg.get("indexer_heads", 4))
            indexer_dim = arch.get(
                "indexer_dim", attention_cfg.get("indexer_dim", h // 8)
            )

        mla_kv_lora_rank = arch.get(
            "mla_kv_lora_rank",
            attention_cfg.get(
                "mla_kv_lora_rank", arch.get("ds_compressed_dim", 0)
            ),
        )  # MLA compression rank (0 = disabled)

        if s <= 0:
            raise ValueError("sequence_length must be > 0.")

        # Linear layer FLOPs
        linear_multiplier = float(arch.get("linear_flops_multiplier", 1.0))
        flops_per_seq_linear = 6 * s * n_linear * linear_multiplier

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

        position_type = str(
            arch.get("position_type", position_cfg.get("position_type", ""))
        ).strip().lower()
        yarn_multiplier = arch.get(
            "yarn_flops_multiplier", position_cfg.get("yarn_flops_multiplier")
        )
        if yarn_multiplier is None:
            yarn_multiplier = 1.0
        yarn_multiplier = float(yarn_multiplier)
        if yarn_multiplier <= 0:
            raise ValueError("yarn_flops_multiplier must be > 0.")
        if position_type in ("yarn", "yarn_rope", "yarn_embedding"):
            attention_multiplier *= yarn_multiplier

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
            training_multiplier = 3.0
            sparse_attn_per_layer = (
                indexer_flops + sparse_attn_core + mla_projection_flops
            ) * training_multiplier

            # Total for all layers
            flops_per_seq_attn = layers * sparse_attn_per_layer * attention_multiplier

            # Store breakdown for debugging (optional)
            self._attn_flops_breakdown = {
                "indexer_flops": indexer_flops * layers * training_multiplier,
                "sparse_core_flops": sparse_attn_core * layers * training_multiplier,
                "mla_projection_flops": mla_projection_flops
                * layers
                * training_multiplier,
                "total_attn_flops": flops_per_seq_attn,
                "sparse_k_tokens": sparse_k_tokens,
                "reduction_vs_dense": (12 * layers * h * (s**2)) / flops_per_seq_attn,
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
            ) * 3.0
            flops_per_seq_attn = layers * dense_attn_per_layer * attention_multiplier

        flops_per_seq_total = flops_per_seq_linear + flops_per_seq_attn

        return flops_per_seq_total / s

    def calculate_flops(self, params: Optional[dict] = None) -> float:
        """
        Calculate total FLOPs for training this stage.
        """
        params = params or self.calculate_params()
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be > 0.")

        return self.flops_per_token(params) * self.total_tokens


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
    args = parser.parse_args()

    config = load_config(args.config)

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
        mem_header = "Mem/CPU(GB)" if cpu_offload else "Mem/GPU"
        print(f"{'-'*120}")
        print(
            f"{'Stage':<20} | {'Tokens (B)':<10} | {'Total Params':<12} | {'Active Params':<12} | {mem_header:<10} | {'ZFLOPs':<7} | {'Days':<6} | {'Cost ($)':<12}"
        )
        print(f"{'-'*120}")

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
                params, num_gpus, zero_stage, precision, cpu_offload
            )
            mem_per_gpu = mem_info["memory_per_gpu_gb"]

            # Warning indicator if memory exceeds H100 80GB
            mem_warn = "⚠️" if mem_per_gpu > 80 else ""

            # Format memory string
            mem_str = f"{mem_per_gpu:<6.1f}{mem_warn:<2}"
            if cpu_offload:
                cpu_mem_gb = mem_info.get("cpu_memory_per_gpu_gb", 0)
                mem_str = f"{mem_per_gpu:<4.1f}|{cpu_mem_gb:<4.0f}{mem_warn:<1}"

            print(
                f"{stage.name:<20} | {stage.total_tokens/1e9:<10.1f} | {params['total_params']/1e9:<5.1f} B       | {params['active_params']/1e9:<5.1f} B       | {mem_str:<10} | {stage_flops/1e21:<7.2f} | {stage_days:<6.2f} | ${stage_cost:,.0f}"
            )

        total_days = total_seconds / (24 * 3600)
        total_hours = total_seconds / 3600
        total_cost = total_hours * num_gpus * price_per_gpu_hour

        print(f"{'-'*120}")
        print(
            f"{'TOTAL':<20} | {'-':<10} | {'-':<12} | {'-':<12} | {'-':<8} | {total_flops/1e21:<7.2f} | {total_days:<6.2f} | ${total_cost:,.0f}"
        )
        print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
