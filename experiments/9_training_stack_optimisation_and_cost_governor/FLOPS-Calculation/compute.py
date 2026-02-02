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

        embedding_params = vocab * hidden
        lm_head_params = 0 if tie_embeddings else vocab * hidden

        attn_params_per_layer = 4 * hidden * hidden

        ffn_params_per_expert = 3 * hidden * intermediate
        ffn_params_dense = ffn_params_per_expert

        total_ffn_params_moe = 0
        active_ffn_params_moe = 0
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

        params_null_path = (
            embedding_params
            + layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + lm_head_params
        )

        effective_active_params = (
            1 - null_prob
        ) * active_params_base + null_prob * params_null_path
        effective_active_non_embed_params = (
            1 - null_prob
        ) * active_non_embed_params + null_prob * (
            layers * attn_params_per_layer
            + dense_layers * ffn_params_dense
            + lm_head_params
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
            "derived_num_experts": derived_experts,
        }

    def calculate_memory_per_gpu(
        self, params: Optional[dict] = None, num_gpus: int = 8, zero_stage: int = 2,
        quantization: str = "bf16", cpu_offload: bool = False
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
        
        # Bytes per parameter for model weights depends on quantization
        quantization = quantization.lower()
        if quantization == "fp8":
            weight_bytes_per_param = 1
        elif quantization == "nvfp4":
            weight_bytes_per_param = 0.5
        else:  # bf16 or default
            weight_bytes_per_param = 2
        
        model_bytes = total_params * weight_bytes_per_param
        optimizer_bytes = total_params * 8  # FP32 Adam (m + v) - always FP32
        gradient_bytes = total_params * 4  # FP32 gradients
        
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
            total_system_mem_bytes = model_bytes + optimizer_bytes + gradient_bytes
            
            # CPU RAM required per GPU (assuming perfect sharding across available host RAM)
            # In reality, you need enough CPU RAM on the node to hold the shard.
            cpu_memory_bytes = total_system_mem_bytes / num_gpus
            cpu_memory_gb = cpu_memory_bytes / (1024**3)
            
        elif zero_stage == 0:
            # No sharding - full memory on each GPU
            memory_bytes = model_bytes + optimizer_bytes + gradient_bytes
        elif zero_stage == 2:
            # ZeRO-2: Shard optimizer + gradients, replicate model
            memory_bytes = model_bytes + (optimizer_bytes + gradient_bytes) / num_gpus
        elif zero_stage == 3:
            # ZeRO-3: Shard everything
            memory_bytes = (model_bytes + optimizer_bytes + gradient_bytes) / num_gpus
        else:
            # Default to ZeRO-2
            memory_bytes = model_bytes + (optimizer_bytes + gradient_bytes) / num_gpus
        
        memory_gb = memory_bytes / (1024**3)
        
        return {
            "memory_per_gpu_gb": memory_gb,
            "cpu_memory_per_gpu_gb": cpu_memory_gb,
            "zero_stage": zero_stage,
            "quantization": quantization,
            "model_gb": model_bytes / (1024**3),
            "optimizer_gb": optimizer_bytes / (1024**3),
            "gradient_gb": gradient_bytes / (1024**3),
        }

    def calculate_flops(self, params: Optional[dict] = None) -> float:
        params = params or self.calculate_params()
        arch = self.architecture

        n_linear = params["active_non_embed_params"]
        layers = arch.get("num_layers", 24)
        h = arch.get("hidden_size", 2048)
        s = arch.get("sequence_length", 4096)
        if s <= 0:
            raise ValueError("sequence_length must be > 0.")
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be > 0.")

        flops_per_seq_linear = 6 * s * n_linear
        flops_per_seq_attn = 12 * layers * h * (s**2)

        flops_per_seq_total = flops_per_seq_linear + flops_per_seq_attn

        num_sequences = self.total_tokens / s

        return flops_per_seq_total * num_sequences


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


def get_zero_efficiency(zero_stage: int, cpu_offload: bool, zero_efficiency_cfg: dict) -> float:
    """
    Get the throughput efficiency multiplier for a ZeRO configuration.
    
    ZeRO stages have different overheads:
    - ZeRO-0: No sharding, no overhead (baseline)
    - ZeRO-2: Shard optimizer + gradients, minimal overhead
    - ZeRO-3: Shard params too, requires all-gather before each forward/backward
    - ZeRO-Infinity: CPU/NVMe offload, significant PCIe bandwidth bottleneck
    
    Sources for default values:
    - DeepSpeed ZeRO paper: https://arxiv.org/abs/1910.02054
    - Microsoft benchmarks show ZeRO-3 is ~15-30% slower than ZeRO-2
    - ZeRO-Infinity paper reports 2-5x slowdown for CPU offload
    """
    defaults = {
        "zero0": 1.0,      # Baseline (data parallel only)
        "zero2": 0.95,     # ~5% overhead for gradient sharding
        "zero3": 0.70,     # ~30% overhead for param all-gather
        "zero_infinity": 0.25  # ~75% overhead for CPU offload (PCIe bottleneck)
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
    
    Sources for default values:
    - NVIDIA Megatron-LM paper: reports ~90% efficiency at 64 GPUs
    - Google PaLM paper: ~80% efficiency at 6144 TPUs
    - Meta OPT-175B: reported ~70% MFU at 992 GPUs
    
    We use log-interpolation between reference points.
    """
    import math
    
    defaults = {
        "base_gpus": 8,
        "efficiency_at_base": 1.0,
        "efficiency_at_64": 0.90,
        "efficiency_at_256": 0.80,
        "efficiency_at_1024": 0.65
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
    Allocate tokens to stages so total compute ~= largest stage from-scratch FLOPs.
    Uses attention-aware FLOPs: F = (6*S*N_linear + 12*L*H*S^2) * (T/S)
    
    If a stage has 'actual_tokens' set, that value is used directly (user-defined
    stabilization point). Otherwise, tokens are allocated from the budget.
    """
    if not stages:
        return

    mode = str(growth_cfg.get("mode", "paper")).strip().lower()
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
            # Using attention-aware formula is complex to invert, so we iterate
            params = stage.calculate_params()
            n_linear = params["active_non_embed_params"]
            arch = stage.architecture
            L = arch.get("num_layers", 24)
            H = arch.get("hidden_size", 2048)
            S = arch.get("sequence_length", 4096)
            
            # FLOPs per token = (6*N_linear + 12*L*H*S) for attention-aware
            flops_per_token = 6 * n_linear + 12 * L * H * S
            
            # Max tokens this stage can afford with remaining budget
            t = remaining_flops / flops_per_token
            # Cap at max_tokens from config
            t = min(t, stage.total_tokens)
            stage.total_tokens = t
            
            stage_flops = flops_per_token * t
            remaining_flops -= stage_flops
        
        if remaining_flops <= 0:
            remaining_flops = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate training FLOPs and duration (growth/expansion mode)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/9_training_stack_optimisation_and_cost_governor/FLOPS-Calculation/config.json",
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

    # Effective MFU after all overheads
    effective_mfu = mfu * zero_eff * scaling_eff
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
            stage_seconds = stage_flops / effective_flops_per_sec
            stage_days = stage_seconds / (24 * 3600)

            stage_hours = stage_seconds / 3600
            stage_cost = stage_hours * num_gpus * price_per_gpu_hour

            total_flops += stage_flops
            total_seconds += stage_seconds

            # Calculate memory per GPU (varies by quantization)
            mem_info = stage.calculate_memory_per_gpu(params, num_gpus, zero_stage, precision, cpu_offload)
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
