import dataclasses
from typing import List, Optional

@dataclasses.dataclass
class TrainingStage:
    name: str
    total_tokens: float  # In Billions (e.g., 20e9)
    total_params: float  # In Billions (e.g., 1e9)
    active_params: float # In Billions (e.g., 1.0e9)
    description: str

    def calculate_flops(self) -> float:
        """
        Calculates total FLOPs for this stage using C = 6 * ActiveParams * Tokens.
        Returns:
            float: Total FLOPs
        """
        return 6 * self.active_params * self.total_tokens

def get_h100_specs(precision: str = "bf16") -> float:
    """
    Returns peak TFLOPS per H100 GPU based on precision.
    Source: NVIDIA H100 Datasheet
    """
    specs = {
        "bf16": 989.0,  # Tensor Core BF16
        "fp8": 1979.0   # Tensor Core FP8 with sparsity (effective) or dense doubled
    }
    return specs.get(precision.lower(), 989.0)

def main():
    # --- Configuration ---
    # Hardware Config
    NUM_GPUS = 8
    MFU = 0.30  # Model Flops Utilization
    
    # Define Stages based on z.md and user requirements
    stages = []

    # Stage 1: 1B Dense Seed
    # 20B tokens, 1B params, 1B active
    stages.append(TrainingStage(
        name="1B Dense Seed",
        total_tokens=20e9,
        total_params=1e9,
        active_params=1.0e9,
        description="Dense Decoder-Only"
    ))

    # Stage 2: 3B MoE (Small)
    # 40B tokens, 3B params, ~1.2B active (Shared backbone + top-k)
    stages.append(TrainingStage(
        name="3B MoE (Small)",
        total_tokens=40e9,
        total_params=3e9,
        active_params=1.2e9,
        description="Sparse MoE (Small)"
    ))

    # Stage 3: 8B MoE (Mid-Scale)
    # 100B tokens, 8B params, ~2.5B active (~30% active density)
    stages.append(TrainingStage(
        name="8B MoE (Mid)",
        total_tokens=100e9,
        total_params=8e9,
        active_params=2.5e9,
        description="Sparse MoE (Mid-Scale)"
    ))

    # Stage 4: 70B MoE (Optimized with Null Experts)
    # 240B tokens
    # Active Params Logic:
    # - Signal Path (30% tokens): 16B Active
    # - Null Path (70% tokens): 2B Active
    # Effective Active = 0.3 * 16 + 0.7 * 2 = 4.8 + 1.4 = 6.2B
    active_params_70b = (0.3 * 16e9) + (0.7 * 2e9)
    
    stages.append(TrainingStage(
        name="70B MoE (Opt)",
        total_tokens=240e9,
        total_params=70e9,
        active_params=active_params_70b,
        description="MoE with Null Expert Strategy"
    ))

    # --- Calculations ---
    print(f"{'='*80}")
    print(f"ERA V4 Training Compute & Schedule Calculator")
    print(f"Cluster: {NUM_GPUS}x NVIDIA H100 | MFU: {MFU*100}%")
    print(f"{'='*80}\n")

    precisions = ["bf16", "fp8"]
    
    for precision in precisions:
        peak_tflops = get_h100_specs(precision)
        effective_flops_per_sec = NUM_GPUS * (peak_tflops * 1e12) * MFU
        
        print(f"--- Precision: {precision.upper()} (Peak/GPU: {peak_tflops} TFLOPS) ---")
        print(f"Effective Cluster Performance: {effective_flops_per_sec/1e15:.2f} PFLOPS")
        print(f"{'-'*80}")
        print(f"{'Stage':<20} | {'Active Params':<15} | {'ZettaFLOPs':<12} | {'Days':<10}")
        print(f"{'-'*80}")

        total_flops = 0.0
        total_seconds = 0.0

        for stage in stages:
            stage_flops = stage.calculate_flops()
            stage_seconds = stage_flops / effective_flops_per_sec
            stage_days = stage_seconds / (24 * 3600)

            total_flops += stage_flops
            total_seconds += stage_seconds

            print(f"{stage.name:<20} | {stage.active_params/1e9:<5.2f} B          | {stage_flops/1e21:<10.2f}   | {stage_days:<10.2f}")

        total_days = total_seconds / (24 * 3600)
        print(f"{'-'*80}")
        print(f"{'TOTAL':<20} | {'-':<15} | {total_flops/1e21:<10.2f}   | {total_days:<10.2f} Days")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
