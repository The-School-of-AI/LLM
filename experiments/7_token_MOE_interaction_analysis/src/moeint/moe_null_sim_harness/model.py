import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_heads=9,
        compression_ratio=8,
        max_position_embeddings=2048,
        rope_theta=100000.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.latent_dim = hidden_size // compression_ratio

        # compression: projection to low-rank latent space
        self.q_down = nn.Linear(hidden_size, self.latent_dim, bias=False)
        self.kv_down = nn.Linear(hidden_size, self.latent_dim, bias=False)

        # decompression: project latent back to full size
        self.q_up = nn.Linear(self.latent_dim, hidden_size, bias=False)
        self.k_up = nn.Linear(self.latent_dim, hidden_size, bias=False)
        self.v_up = nn.Linear(self.latent_dim, hidden_size, bias=False)

        # output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # RoPE
        self.rotary_emb = LlamaRotaryEmbedding(
            LlamaConfig(
                rope_theta=rope_theta,
                head_dim=self.head_dim,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=max_position_embeddings,
            )
        )

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # compress to latent
        q_latent = self.q_down(x)  # (batch_size, seq_len, latent_dim)
        kv_latent = self.kv_down(x)  # (batch_size, seq_len, latent_dim)

        # decompress
        q = self.q_up(q_latent)  # (batch_size, seq_len, hidden_size)
        k = self.k_up(kv_latent)  # (batch_size, seq_len, hidden_size)
        v = self.v_up(kv_latent)  # (batch_size, seq_len, hidden_size)

        # reshape for multi head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # apply RoPE
        position_ids = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        cos, sin = self.rotary_emb(x, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # transpose for attention
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # scaled dot product
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=True
        )

        # reshape and output projection
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        return self.o_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=1536):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=1536,
        num_routed_experts=7,
        num_shared_experts=1,
        num_null_experts=2,
        num_experts_per_tok=2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.num_null_experts = num_null_experts
        self.top_k = num_experts_per_tok

        # shared experts - always active
        self.shared_experts = nn.ModuleList(
            [MLP(hidden_size, intermediate_size) for _ in range(num_shared_experts)]
        )

        # routed experts (top-k selection)
        self.routed_experts = nn.ModuleList(
            [MLP(hidden_size, intermediate_size) for _ in range(num_routed_experts)]
        )

        # router network - routes to all non shared experts (real + null)
        self.router = nn.Linear(
            hidden_size, num_routed_experts + num_null_experts, bias=False
        )

        # routing bias for load balancing (learned, but manually updated)
        self.routing_bias = nn.Parameter(
            torch.zeros(num_routed_experts + num_null_experts)
        )

        # track expert load from last forward pass
        self.last_expert_load: Tensor | None = None

        # tracking routing decisions for monitoring
        self.last_routing_logits: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, hidden_dim = x.shape

        # flatten to (batch * seq_len, hidden_size) for easier processing
        x_flat = x.view(-1, hidden_dim)

        # shared experts
        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output += expert(x_flat)
        shared_output = shared_output / self.num_shared_experts

        # routing scores
        routing_logits = self.router(x_flat) + self.routing_bias

        # get top-k experts per token
        routing_probs = torch.sigmoid(routing_logits)
        scores, indices = torch.topk(routing_probs, self.top_k, dim=-1)

        # normalize scores over selected real experts only, null experts don't contribute to output.
        real_expert_mask = indices < self.num_routed_experts
        scores = scores * real_expert_mask.float()
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # routed experts
        combined_output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = indices[..., k]  # which expert for this k-th position
            expert_scores = scores[..., k : k + 1]  # score for this k-th position

            # process each expert
            for i in range(self.num_routed_experts):
                mask = expert_indices == i
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.routed_experts[i](expert_input)
                    combined_output[mask] += expert_output * expert_scores[mask]

            # null experts are the ones from index >= num_routed_experts. since
            # they dont contribute to output, no code required.

        # final output
        final_output = shared_output + combined_output

        # reshape back to (batch_size, seq_len, hidden_size)
        final_output = final_output.view(batch_size, seq_len, hidden_dim)

        # track expert load for bias updates (done in training loop)
        self.last_expert_load = self._compute_expert_load(indices)

        # track routing decisions for monitoring
        self.last_routing_logits = routing_logits

        return final_output

    def update_bias_terms(self, expert_load: Tensor):
        # update routing bias to balance expert load (loss-less balancing) across all experts (real + null)
        # Uses adaptive learning rate based on load imbalance

        target_load = 1.0 / (self.num_routed_experts + self.num_null_experts)
        load_diff = expert_load - target_load

        # adaptive update rate: bigger imbalance -> bigger correction
        update_rate = 0.1 * torch.abs(load_diff)

        # update bias (outside autograd to avoid affecting gradients)
        with torch.no_grad():
            self.routing_bias -= update_rate * torch.sign(load_diff)

    def _compute_expert_load(self, indices: Tensor) -> Tensor:
        # Computes what fraction of tokens were routed to each expert.
        # indices shape: (num_tokens, top_k) - expert indices selected per token

        num_tokens = indices.shape[0]
        total_experts = self.num_routed_experts + self.num_null_experts
        expert_load = torch.zeros(total_experts, device=indices.device)

        for i in range(total_experts):
            expert_load[i] = (indices == i).sum().float() / num_tokens

        return expert_load


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_heads=9,
        compression_ratio=8,
        intermediate_size=1536,
        num_routed_experts=7,
        num_shared_experts=1,
        num_null_experts=2,
        num_experts_per_tok=2,
        rms_norm_eps=1e-5,
        max_position_embeddings=2048,
        rope_theta=100000.0,
    ):
        super().__init__()

        # pre-attention norm
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

        # multi head latent attention
        self.self_attn = MultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            compression_ratio=compression_ratio,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

        # pre-moe norm
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

        # DeepSeek MOE
        self.moe = DeepSeekMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_routed_experts=num_routed_experts,
            num_shared_experts=num_shared_experts,
            num_null_experts=num_null_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.moe(x)
        x = residual + x

        return x


class DeepSeekIsh(nn.Module):
    def __init__(
        self,
        vocab_size=49152,
        hidden_size=768,
        num_hidden_layers=30,
        num_attention_heads=9,
        compression_ratio=8,
        intermediate_size=1536,
        num_routed_experts=7,
        num_shared_experts=1,
        num_null_experts=2,
        num_experts_per_tok=2,
        rms_norm_eps=1e-5,
        max_position_embeddings=2048,
        rope_theta=100000.0,
        tie_word_embeddings=True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    compression_ratio=compression_ratio,
                    intermediate_size=intermediate_size,
                    num_routed_experts=num_routed_experts,
                    num_shared_experts=num_shared_experts,
                    num_null_experts=num_null_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    rms_norm_eps=rms_norm_eps,
                    max_position_embeddings=max_position_embeddings,
                    rope_theta=rope_theta,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        # final layer norm
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

        # optional separate output projection
        self.lm_head = None
        if not tie_word_embeddings:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.apply(self._init_weights)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        # input_ids shape: (batch_size, seq_len)

        x = self.embed_tokens(input_ids)  # (batch, seq_len, hidden_size)

        # pass through the transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)

        # final norm
        x = self.norm(x)

        # project to vocabulary
        logits: Tensor
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        return logits

    def _init_weights(self, module):
        std = 1.0 / 24.0  # 0.041666666666666664
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
