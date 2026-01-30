import dataclasses
from typing import List, Optional
import json
import argparse
import sys

@dataclasses.dataclass
class TrainingStage:
    name: str
    total_tokens: float
    architecture: dict
    description: str

    def calculate_params(self) -> dict:
        """
        Calculates Total and Active parameters based on architecture.
        """
        arch = self.architecture
        vocab = arch.get("vocab_size", 50257)
        hidden = arch.get("hidden_size", 2048)
        intermediate = arch.get("intermediate_size", 4 * hidden)
        layers = arch.get("num_layers", 24)
        experts = arch.get("num_experts", 0)
        top_k = arch.get("top_k_experts", 1)
        null_prob = arch.get("null_expert_prob", 0.0)
        num_moe_layers = arch.get("num_moe_layers", layers if experts > 0 else 0)
        tie_embeddings = arch.get("tie_embeddings", True)
        target_total_params = arch.get("target_total_params")
        target_params_per_expert = arch.get("target_params_per_expert")
        solve_for = str(arch.get("solve_for", "")).strip().lower()

        if experts < 0 or top_k < 0:
            raise ValueError("num_experts and top_k_experts must be >= 0.")
        if experts == 0:
            num_moe_layers = 0
        if experts > 0 and top_k > experts:
            raise ValueError("top_k_experts cannot exceed num_experts.")
        if num_moe_layers < 0 or num_moe_layers > layers:
            raise ValueError("num_moe_layers must be between 0 and num_layers.")

        # 1. Embeddings (Vocab * Hidden) + Positional (MaxSeq * Hidden - neglected for approx)
        embedding_params = vocab * hidden
        lm_head_params = 0 if tie_embeddings else vocab * hidden

        # 2. Attention Block (per layer)
        # Q, K, V, O projections: 4 * (Hidden * Hidden)
        # We neglect biases for FLOPs approx
        attn_params_per_layer = 4 * hidden * hidden

        # 3. FFN Block (per layer)
        # SwiGLU: 3 matrices (Gate, Up, Down): 3 * (Hidden * Intermediate)
        ffn_params_per_expert = 3 * hidden * intermediate
        ffn_params_dense = ffn_params_per_expert

        total_ffn_params_moe = 0
        active_ffn_params_moe = 0
        if experts > 0 or solve_for in ("num_experts", "num_experts_from_per_expert"):
            # MoE Layer
            # Router: Hidden * Experts
            router_params = hidden * experts
            total_ffn_params_moe = experts * ffn_params_per_expert + router_params
            # Active FFN: Only top_k experts are active
            active_ffn_params_moe = top_k * ffn_params_per_expert + router_params

        dense_layers = layers - num_moe_layers

        derived_experts = None
        if solve_for == "num_experts":
            if target_total_params is None:
                raise ValueError("target_total_params must be set when solve_for='num_experts'.")
            if num_moe_layers <= 0:
                raise ValueError("num_moe_layers must be > 0 when solve_for='num_experts'.")
            base_params = (
                embedding_params
                + lm_head_params
                + layers * attn_params_per_layer
                + dense_layers * ffn_params_dense
            )
            per_expert_per_layer = ffn_params_per_expert + hidden
            derived_experts = (target_total_params - base_params) / (num_moe_layers * per_expert_per_layer)
            if derived_experts < 1:
                raise ValueError("target_total_params is too small for the given architecture.")
            experts = int(round(derived_experts))
            if experts < 1:
                raise ValueError("Derived num_experts is < 1; check target_total_params.")
            if top_k > experts:
                raise ValueError("top_k_experts cannot exceed derived num_experts.")
            router_params = hidden * experts
            total_ffn_params_moe = experts * ffn_params_per_expert + router_params
            active_ffn_params_moe = top_k * ffn_params_per_expert + router_params
            num_moe_layers = layers if num_moe_layers == 0 else num_moe_layers
        elif solve_for == "num_experts_from_per_expert":
            if target_total_params is None or target_params_per_expert is None:
                raise ValueError("target_total_params and target_params_per_expert must be set when solve_for='num_experts_from_per_expert'.")
            experts = int(round(target_total_params / target_params_per_expert))
            if experts < 1:
                raise ValueError("Derived num_experts is < 1; check target_total_params/target_params_per_expert.")
            if top_k > experts:
                raise ValueError("top_k_experts cannot exceed derived num_experts.")
            router_params = hidden * experts
            total_ffn_params_moe = experts * ffn_params_per_expert + router_params
            active_ffn_params_moe = top_k * ffn_params_per_expert + router_params
            derived_experts = experts

        if experts > 0 and top_k > experts:
            raise ValueError("top_k_experts cannot exceed num_experts.")

        # 4. Total Calculation
        total_params = (
            embedding_params
            + lm_head_params
            + layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * total_ffn_params_moe
        )
        
        # 5. Active Calculation (Pre-Null Logic)
        active_non_embed_params = (
            layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + num_moe_layers * active_ffn_params_moe
            + lm_head_params
        )
        active_params_base = embedding_params + active_non_embed_params

        # 6. Apply Null Expert Logic (70B Case)
        # Signal Path (1 - null_prob): uses active_params_base
        # Null Path (null_prob): Skips FFN entirely (only Attn + Embeddings active)
        # Note: Null path still pays for Attention? Usually yes (Mixture of Depths style).
        # If Null expert means "skip FFN", then active is just Attn.
        
        params_null_path = (
            embedding_params
            + layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + lm_head_params
        )
        
        effective_active_params = (1 - null_prob) * active_params_base + null_prob * params_null_path
        effective_active_non_embed_params = (
            (1 - null_prob) * active_non_embed_params
            + null_prob * (layers * attn_params_per_layer + dense_layers * ffn_params_dense + lm_head_params)
        )

        return {
            "total_params": total_params,
            "active_params": effective_active_params,
            "active_non_embed_params": effective_active_non_embed_params,
            "embedding_params": embedding_params,
            "lm_head_params": lm_head_params,
            "num_moe_layers": num_moe_layers,
            "dense_layers": dense_layers,
            "num_experts": experts,
            "derived_num_experts": derived_experts
        }

    def calculate_flops(self, params: Optional[dict] = None) -> float:
        """
        Calculates FLOPs using the precise 'Attention-Aware' formula.
        Ref: (6 * seq_len * num_params) + (12 * num_layers * hidden_size * seq_len^2)
        """
        params = params or self.calculate_params()
        arch = self.architecture
        
        # 1. Fetch Architecture Vars
        N_linear = params["active_non_embed_params"]
        L = arch.get("num_layers", 24)
        H = arch.get("hidden_size", 2048)
        S = arch.get("sequence_length", 4096)
        if S <= 0:
            raise ValueError("sequence_length must be > 0.")
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be > 0.")
        
        # 2. Calculate FLOPs per Sequence (Exact formula from Image)
        # Term 1: Linear Matrix Muls (FFN + Projections)
        # Formula: 6 * seq_len * num_params
        flops_per_seq_linear = 6 * S * N_linear
        
        # Term 2: Attention Mechanism (Quadratic)
        # Formula: 12 * num_layers * hidden_size * seq_len^2
        flops_per_seq_attn = 12 * L * H * (S ** 2)
        
        flops_per_seq_total = flops_per_seq_linear + flops_per_seq_attn
        
        # 3. Scale to Total Training Tokens
        # Number of sequences = Total Tokens / Sequence Length
        num_sequences = self.total_tokens / S
        
        total_flops = flops_per_seq_total * num_sequences
        
        return total_flops



def load_config(config_path: str):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON file: {exc}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Calculate training FLOPs and duration.")
    parser.add_argument("--config", type=str, default="experiments/9_training_stack_optimisation_and_cost_governor/FLOPS-Calculation/flops_config.json", help="Path to JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # --- Configuration ---
    # Strict key access - will raise KeyError if config is missing
    try:
        hardware = config["hardware"]
        NUM_GPUS = hardware["num_gpus"]
        MFU = hardware["mfu"]
        PRICE_PER_GPU_HOUR = hardware["price_per_gpu_hour"]
        precision_specs = hardware["tflops_per_gpu"]
        selected_quantization = hardware.get("quantization", "all").lower()
        tflops_mode = hardware.get("tflops_mode", "dense")
        
        stages = []
        for stage_conf in config["stages"]:
            stages.append(TrainingStage(
                name=stage_conf["name"],
                total_tokens=float(stage_conf["total_tokens"]),
                architecture=stage_conf["architecture"],
                description=stage_conf.get("description", "") # Description is optional
            ))
            
    except KeyError as e:
        print(f"Error: Missing required configuration key: {e}")
        print("Please ensure flops_config.json contains all necessary fields.")
        sys.exit(1)

    print("="*112)
    print("ERA V4 Training Compute & Schedule Calculator (Architecture-Aware)")
    print(f"Cluster: {NUM_GPUS}x NVIDIA GPUs | MFU: {MFU*100}% | Price/GPU/Hr: ${PRICE_PER_GPU_HOUR}")
    print(f"TFLOPs Mode: {tflops_mode}")
    if selected_quantization != "all":
        print(f"Selected Quantization: {selected_quantization.upper()}")
    print("="*112)
    print("\nNOTE: Parameters are calculated dynamically from architecture specs.")
    
    # Iterate through defined precisions (e.g., bf16, fp8, nvfp4) or filter by selection
    for precision, peak_tflops in precision_specs.items():
        if selected_quantization != "all" and precision.lower() != selected_quantization:
            continue
            
        effective_flops_per_sec = NUM_GPUS * (peak_tflops * 1e12) * MFU
        
        print("-"*112)
        print(f"--- Precision: {precision.upper()} (Peak/GPU: {peak_tflops} TFLOPS) ---")
        print(f"Effective Cluster Performance: {effective_flops_per_sec/1e15:.2f} PFLOPS")
        print(f"{'-'*112}")
        print(f"{'Stage':<20} | {'Total Params':<15} | {'Active Params':<15} | {'ZettaFLOPs':<12} | {'Days':<10} | {'Cost ($)':<12}")
        print(f"{'-'*112}")

        total_flops = 0.0
        total_seconds = 0.0

        for stage in stages:
            try:
                params = stage.calculate_params()
                stage_flops = stage.calculate_flops(params)
            except ValueError as exc:
                print(f"Error in stage '{stage.name}': {exc}")
                sys.exit(1)
            stage_seconds = stage_flops / effective_flops_per_sec
            stage_days = stage_seconds / (24 * 3600)
            
            # Cost = Hours * Num_GPUs * Price
            stage_hours = stage_seconds / 3600
            stage_cost = stage_hours * NUM_GPUS * PRICE_PER_GPU_HOUR

            total_flops += stage_flops
            total_seconds += stage_seconds

            if params.get("derived_num_experts") is not None:
                print(f"  (Derived num_experts for {stage.name}: {params['num_experts']} from target_total_params)")
            print(f"{stage.name:<20} | {params['total_params']/1e9:<5.2f} B          | {params['active_params']/1e9:<5.2f} B          | {stage_flops/1e21:<10.2f}   | {stage_days:<10.2f} | ${stage_cost:,.2f}")

        total_days = total_seconds / (24 * 3600)
        total_hours = total_seconds / 3600
        total_cost = total_hours * NUM_GPUS * PRICE_PER_GPU_HOUR
        
        print(f"{'-'*112}")
        print(f"{'TOTAL':<20} | {'-':<15} | {'-':<15} | {total_flops/1e21:<10.2f}   | {total_days:<10.2f} | ${total_cost:,.2f}")
        print(f"{'='*112}\n")

if __name__ == "__main__":
    main()
