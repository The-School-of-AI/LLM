import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from deepspeed.moe.layer import MoE


class MoEWrapper(nn.Module):
    """
    DeepSpeed MoE sometimes returns:
      - output Tensor
      - OR (output, aux_loss/metadata)
    GPT2 block expects a Tensor only.
    """
    def __init__(self, moe_layer: nn.Module):
        super().__init__()
        self.moe = moe_layer

    def forward(self, x):
        out = self.moe(x)
        if isinstance(out, tuple):
            return out[0]
        return out


class MoEGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(
        self,
        config: GPT2Config,
        num_experts: int = 8,
        top_k: int = 1,
        moe_layer_idx: int = 0,
        expert_hidden_size: int = None,
        min_capacity: int = 0,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
    ):
        super().__init__(config)

        hidden_size = config.n_embd
        expert_hidden_size = expert_hidden_size or hidden_size * 4

        moe = MoE(
            hidden_size=hidden_size,
            expert=nn.Sequential(
                nn.Linear(hidden_size, expert_hidden_size),
                nn.GELU(),
                nn.Linear(expert_hidden_size, hidden_size),
            ),
            num_experts=num_experts,
            k=top_k,
            min_capacity=min_capacity,              # ✅ fixes “tokens < min_capacity”
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
        )

        # Replace one transformer block MLP with MoE (keep it simple)
        self.transformer.h[moe_layer_idx].mlp = MoEWrapper(moe)

        self.post_init()