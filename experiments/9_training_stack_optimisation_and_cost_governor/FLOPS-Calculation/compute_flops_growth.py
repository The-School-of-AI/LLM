import argparse
import dataclasses
import json
import sys
from typing import Optional


@dataclasses.dataclass
class TrainingStage:
    name: str
    total_tokens: float
    architecture: dict
    description: str

    def calculate_params(self) -> dict:
        arch = self.architecture
        vocab = arch.get("vocab_size", 50257)
        hidden = arch.get("hidden_size", 2048)
        intermediate = arch.get("intermediate_size", 4 * hidden)
        layers = arch.get("num_layers", 24)
        num_heads = arch.get("num_heads")
        num_kv_heads = arch.get("num_kv_heads", num_heads)
        experts = arch.get("num_experts", 0)
        top_k = arch.get("top_k_experts", 1)
        null_prob = arch.get("null_expert_prob", 0.0)
        num_moe_layers = arch.get("num_moe_layers", layers if experts > 0 else 0)
        tie_embeddings = arch.get("tie_embeddings", True)
        include_lm_head_flops = arch.get("include_lm_head_flops", True)
        moe_capacity_factor = float(arch.get("moe_capacity_factor", 1.0))
        ffn_type = str(arch.get("ffn_type", "swiglu")).strip().lower()
        ffn_multiplier = arch.get("ffn_multiplier")
        target_total_params = arch.get("target_total_params")
        target_params_per_expert = arch.get("target_params_per_expert")
        solve_for = str(arch.get("solve_for", "")).strip().lower()

        if experts < 0 or top_k < 0:
            raise ValueError("num_experts and top_k_experts must be >= 0.")
        if moe_capacity_factor <= 0:
            raise ValueError("moe_capacity_factor must be > 0.")
        if experts == 0 and solve_for not in ("num_experts", "num_experts_from_per_expert"):
            num_moe_layers = 0
        if experts > 0 and top_k > experts:
            raise ValueError("top_k_experts cannot exceed num_experts.")
        if num_moe_layers < 0 or num_moe_layers > layers:
            raise ValueError("num_moe_layers must be between 0 and num_layers.")
        if num_heads is not None and num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads is not None and num_kv_heads is not None:
            if num_kv_heads <= 0:
                raise ValueError("num_kv_heads must be > 0.")
            if num_kv_heads > num_heads:
                raise ValueError("num_kv_heads cannot exceed num_heads.")
            if num_heads % num_kv_heads != 0:
                raise ValueError("num_kv_heads must divide num_heads for GQA/MQA.")

        embedding_params = vocab * hidden
        lm_head_params = 0 if tie_embeddings else vocab * hidden
        # Logits projection compute still happens even if tied.
        lm_head_params_for_flops = vocab * hidden if include_lm_head_flops else 0

        if num_heads is None or num_kv_heads is None:
            attn_params_per_layer = 4 * hidden * hidden
        else:
            kv_ratio = num_kv_heads / num_heads
            attn_params_per_layer = hidden * hidden * (2 + 2 * kv_ratio)

        if ffn_multiplier is None:
            if ffn_type in ("gelu", "relu", "mlp"):
                ffn_multiplier = 2
            elif ffn_type in ("swiglu", "geglu", "glu"):
                ffn_multiplier = 3
            else:
                raise ValueError(
                    f"Unknown ffn_type '{ffn_type}'. Use 'swiglu', 'geglu', 'glu', 'gelu', 'relu', or set ffn_multiplier."
                )
        ffn_params_per_expert = float(ffn_multiplier) * hidden * intermediate
        ffn_params_dense = ffn_params_per_expert

        total_ffn_params_moe = 0
        active_ffn_params_moe = 0
        router_params = 0
        if experts > 0 or solve_for in ("num_experts", "num_experts_from_per_expert"):
            router_params = hidden * experts
            total_ffn_params_moe = experts * ffn_params_per_expert + router_params
            active_ffn_params_moe = top_k * ffn_params_per_expert + router_params

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
            total_ffn_params_moe = experts * ffn_params_per_expert + router_params
            active_ffn_params_moe = top_k * ffn_params_per_expert + router_params
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
            total_ffn_params_moe = experts * ffn_params_per_expert + router_params
            active_ffn_params_moe = top_k * ffn_params_per_expert + router_params
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
        active_params_base = embedding_params + active_non_embed_params
        active_linear_params = (
            layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * (active_ffn_params_moe * moe_capacity_factor)
            + lm_head_params_for_flops
        )

        params_null_path = (
            embedding_params
            + layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * router_params
            + lm_head_params
        )
        params_null_path_non_embed = (
            layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * router_params
            + lm_head_params
        )
        params_null_path_linear = (
            layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * router_params
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
            "ffn_params_per_expert": ffn_params_per_expert,
            "ffn_multiplier": ffn_multiplier,
            "top_k_experts": top_k,
            "moe_capacity_factor": moe_capacity_factor,
            "num_moe_layers": num_moe_layers,
            "dense_layers": dense_layers,
            "num_experts": experts,
            "derived_num_experts": derived_experts,
        }

    def flops_per_token(self, params: Optional[dict] = None) -> float:
        params = params or self.calculate_params()
        arch = self.architecture

        n_linear = params["active_linear_params"]
        num_moe_layers = params["num_moe_layers"]
        ffn_params_per_expert = params["ffn_params_per_expert"]
        top_k = params["top_k_experts"]
        moe_capacity_factor = params["moe_capacity_factor"]
        layers = arch.get("num_layers", 24)
        h = arch.get("hidden_size", 2048)
        s = arch.get("sequence_length", 4096)
        attention_window = arch.get("attention_window")
        attention_sparsity = arch.get("attention_sparsity")
        attention_kernel_multiplier = arch.get("attention_kernel_multiplier")
        if attention_kernel_multiplier is None:
            attention_kernel_multiplier = arch.get("flash_attention_multiplier", 1.0)
        attention_kernel_multiplier = float(attention_kernel_multiplier)
        quantization_flops_multiplier = float(
            arch.get("quantization_flops_multiplier", 1.0)
        )
        moe_routing_overhead_ratio = float(
            arch.get("moe_routing_overhead_ratio", 0.0)
        )
        moe_routing_flops_per_token = arch.get("moe_routing_flops_per_token")
        null_prob = float(arch.get("null_expert_prob", 0.0))
        recompute_multiplier = float(arch.get("recompute_multiplier", 1.0))
        if s <= 0:
            raise ValueError("sequence_length must be > 0.")
        if attention_kernel_multiplier <= 0:
            raise ValueError("attention_kernel_multiplier must be > 0.")
        if quantization_flops_multiplier <= 0:
            raise ValueError("quantization_flops_multiplier must be > 0.")
        if moe_routing_overhead_ratio < 0:
            raise ValueError("moe_routing_overhead_ratio must be >= 0.")
        if null_prob < 0 or null_prob > 1:
            raise ValueError("null_expert_prob must be in [0, 1].")
        if recompute_multiplier <= 0:
            raise ValueError("recompute_multiplier must be > 0.")

        attn_tokens = s
        if attention_window is not None:
            if attention_window <= 0:
                raise ValueError("attention_window must be > 0.")
            attn_tokens = min(s, float(attention_window))
        elif attention_sparsity is not None:
            if attention_sparsity <= 0 or attention_sparsity > 1:
                raise ValueError("attention_sparsity must be in (0, 1].")
            attn_tokens = s * float(attention_sparsity)

        attention_term = 12 * layers * h * attn_tokens
        softmax_term = 0.0
        if arch.get("include_softmax_flops", False):
            heads = arch.get("num_heads")
            if heads is None:
                raise ValueError(
                    "num_heads must be set when include_softmax_flops is True."
                )
            softmax_term = 3 * layers * heads * attn_tokens

        flops_per_token = 6 * n_linear + attention_kernel_multiplier * (
            attention_term + softmax_term
        )

        if num_moe_layers > 0:
            moe_expert_flops_per_token = (
                6 * top_k * ffn_params_per_expert * moe_capacity_factor
            )
            if moe_routing_overhead_ratio > 0:
                flops_per_token += (
                    num_moe_layers
                    * moe_routing_overhead_ratio
                    * moe_expert_flops_per_token
                    * (1 - null_prob)
                )
            if moe_routing_flops_per_token is not None:
                flops_per_token += (
                    num_moe_layers
                    * float(moe_routing_flops_per_token)
                    * (1 - null_prob)
                )

        return flops_per_token * recompute_multiplier * quantization_flops_multiplier

    def calculate_flops(self, params: Optional[dict] = None) -> float:
        params = params or self.calculate_params()
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be > 0.")

        return self.flops_per_token(params) * self.total_tokens


def load_config(config_path: str):
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON file: {exc}")
        sys.exit(1)


def apply_growth_allocation(stages: list[TrainingStage], growth_cfg: dict) -> None:
    """
    Allocate tokens to stages so total compute ~= largest stage from-scratch FLOPs.
    This follows the paper-style compute budget: sum_i F_i = F_max, using the same
    per-token FLOPs formula as calculate_flops.
    """
    if not stages:
        return

    mode = str(growth_cfg.get("mode", "paper")).strip().lower()
    if mode != "paper":
        raise ValueError("Only growth mode 'paper' is supported in this script.")

    largest_stage = max(
        stages,
        key=lambda s: s.flops_per_token(s.calculate_params()) * s.total_tokens,
    )
    total_budget_flops = (
        largest_stage.flops_per_token(largest_stage.calculate_params())
        * largest_stage.total_tokens
    )

    remaining_flops = total_budget_flops
    for stage in stages:
        per_token_flops = stage.flops_per_token(stage.calculate_params())
        # Allocate tokens sequentially from small to large by default order.
        t = remaining_flops / per_token_flops
        # Cap at original stage token budget if provided.
        t = min(t, stage.total_tokens)
        stage.total_tokens = t
        remaining_flops -= per_token_flops * t
        if remaining_flops <= 0:
            remaining_flops = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate training FLOPs and duration (growth/expansion mode)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/9_training_stack_optimisation_and_cost_governor/FLOPS-Calculation/flops_config_growth.json",
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

    growth_cfg = config.get("growth", {"mode": "paper"})
    apply_growth_allocation(stages, growth_cfg)

    print("=" * 112)
    print("ERA V4 Training Compute & Schedule Calculator (Growth/Expansion Mode)")
    print(
        f"Cluster: {num_gpus}x NVIDIA GPUs | MFU: {mfu*100}% | Price/GPU/Hr: ${price_per_gpu_hour}"
    )
    print(
        f"TFLOPs Mode: {tflops_mode} | Growth Mode: {growth_cfg.get('mode', 'paper')}"
    )
    print("=" * 112)
    print("\nNOTE: Parameters are calculated dynamically from architecture specs.")

    for precision, peak_tflops in precision_specs.items():
        if (
            selected_quantization != "all"
            and precision.lower() != selected_quantization
        ):
            continue

        effective_flops_per_sec = num_gpus * (peak_tflops * 1e12) * mfu

        print("-" * 112)
        print(
            f"--- Precision: {precision.upper()} (Peak/GPU: {peak_tflops} TFLOPS) ---"
        )
        print(
            f"Effective Cluster Performance: {effective_flops_per_sec/1e15:.2f} PFLOPS"
        )
        print(f"{'-'*112}")
        print(
            f"{'Stage':<20} | {'Tokens (B)':<12} | {'Active Params':<15} | {'ZettaFLOPs':<12} | {'Days':<10} | {'Cost ($)':<12}"
        )
        print(f"{'-'*112}")

        total_flops = 0.0
        total_seconds = 0.0

        for stage in stages:
            params = stage.calculate_params()
            stage_flops = stage.calculate_flops(params)
            stage_seconds = stage_flops / effective_flops_per_sec
            stage_days = stage_seconds / (24 * 3600)

            stage_hours = stage_seconds / 3600
            stage_cost = stage_hours * num_gpus * price_per_gpu_hour

            total_flops += stage_flops
            total_seconds += stage_seconds

            print(
                f"{stage.name:<20} | {stage.total_tokens/1e9:<12.2f} | {params['active_params']/1e9:<5.2f} B          | {stage_flops/1e21:<10.2f}   | {stage_days:<10.2f} | ${stage_cost:,.2f}"
            )

        total_days = total_seconds / (24 * 3600)
        total_hours = total_seconds / 3600
        total_cost = total_hours * num_gpus * price_per_gpu_hour

        print(f"{'-'*112}")
        print(
            f"{'TOTAL':<20} | {'-':<12} | {'-':<15} | {total_flops/1e21:<10.2f}   | {total_days:<10.2f} | ${total_cost:,.2f}"
        )
        print(f"{'='*112}\n")


if __name__ == "__main__":
    main()
