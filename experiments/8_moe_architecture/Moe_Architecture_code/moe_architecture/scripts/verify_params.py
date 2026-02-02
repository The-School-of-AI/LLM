
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import config_1b_dense, config_3b_moe, config_8b_moe, config_70b_moe
from model.config import MoEModelConfig

def print_params(name, config: MoEModelConfig):
    total = config.estimated_total_params / 1e9
    active = config.estimated_active_params / 1e9
    print(f"{name:<20} | Total: {total:>6.2f}B | Active: {active:>6.2f}B | Layers: {config.num_layers} | Experts: {config.num_routed_experts}+{config.num_shared_experts}")

print("-" * 80)
print(f"{'Model':<20} | {'Total':>8} | {'Active':>8} | {'Structure':<20}")
print("-" * 80)

print_params("1B Dense", config_1b_dense.get_config())
print_params("3B MoE", config_3b_moe.get_config())
print_params("8B MoE", config_8b_moe.get_config())
print_params("70B MoE", config_70b_moe.get_config())

print("-" * 80)
