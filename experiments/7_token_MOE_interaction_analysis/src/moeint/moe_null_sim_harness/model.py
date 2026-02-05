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


class MoERouter(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_experts: int = 7,
        topk: int = 2,
        data_sparsity: float = 0.5,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.data_sparsity = data_sparsity

        # Calculate number of null expert copies: M = N · (1-ρ)/ρ
        # For ρ=0.5, N=8: M = 8 · 0.5/0.5 = 8 null copies
        self.num_null_experts = int(num_experts * (1 - data_sparsity) / data_sparsity)
        self.num_total_experts = num_experts + self.num_null_experts

        # gate for real experts only
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.router.weight.data.normal_(mean=0.0, std=0.02)
        self.router_bias = nn.Parameter(torch.zeros(num_experts))

        # single expert logit (will be duplicated self.num_null_expert times)
        self.null_expert_logit = nn.Parameter(torch.tensor(0.0))

        self.last_router_logits: Tensor | None = None

    def forward(self, x: Tensor):
        batch_size, seq_len, hidden_dim = x.shape

        # router logits for real experts
        real_logits = (
            self.router(x) + self.router_bias
        )  # shape: (batch_size, seq_len, num_experts)

        # duplicate null expert logic num_null_expert times
        null_logits = self.null_expert_logit.view(1, 1, 1).expand(
            batch_size, seq_len, self.num_null_experts
        )  # shape: (batch_size, seq_len, num_null_experts)

        # creating router logits by concantenating real and null expert router logits
        router_logits = torch.cat(
            [real_logits, null_logits], dim=-1
        )  # shape: (batch_size, seq_len, num_total_experts)
        self.last_router_logits = router_logits.detach()

        # softmax routing
        router_probs = F.softmax(router_logits, dim=-1)

        # top-k calculation
        topk_weights, topk_indices = torch.topk(router_probs, self.topk, dim=-1)

        # null expert selections
        is_null = topk_indices >= self.num_experts

        # renormalize weights over only the real experts
        # zero out the null weights, and renormalize
        real_weights = topk_weights * (~is_null).float()
        topk_weight = real_weights / real_weights.sum(dim=-1, keepdim=True).clamp(1e-6)

        ## compute auxiliary loss

        # P_i: average routing probability of expert i
        P = router_probs.mean(dim=(0, 1))  # shape: (num_total_experts,)

        # f_i: fraction of tokens routed to expert i
        f = torch.bincount(
            topk_indices.flatten(), minlength=self.num_total_experts
        ).float() / (batch_size * seq_len)

        L_bal = self.num_total_experts * torch.sum(f * P)

        # Z-Loss
        # log^2(sum(exp(logits))) -> (log_sum_exp(logits))^2
        L_z = (torch.logsumexp(router_logits, dim=-1) ** 2).mean()

        # final aux loss
        aux_loss = 2e-2 * L_bal + 1e-3 * L_z

        return topk_indices, topk_weight, is_null, aux_loss


class MoEBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 1536,
        num_experts: int = 7,
        topk=2,
        data_sparsity: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk
        self.dropout = dropout

        self.router = MoERouter(hidden_size, num_experts, topk, data_sparsity)

        # expert weights (real experts only)
        self.w_gate = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )
        self.w_up = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )
        self.w_down = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )

        # shared expert (1 shared expert, always active)
        self.shared_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.shared_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.shared_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        for module in [self.shared_gate, self.shared_up, self.shared_down]:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: Tensor):
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        device, dtype = x.device, x.dtype

        # shared expert
        shared_hidden = F.silu(self.shared_gate(x)) * self.shared_up(x)
        if self.training and self.dropout > 0:
            shared_hidden = F.dropout(shared_hidden, p=self.dropout)
        shared_out = self.shared_down(shared_hidden)

        # routed experts with null expert handling
        topk_indices, topk_weights, is_null, aux_loss = self.router(x)

        # filter out null expert routes
        # create a mask for real expert routes
        real_mask = ~(is_null.view(num_tokens, self.topk))

        # flatten and filter
        token_indices = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1)
            .expand(num_tokens, self.topk)
        )

        # keep only real experts routes
        real_token_indices = token_indices[real_mask]
        real_expert_indices = topk_indices.view(num_tokens, self.topk)[real_mask]
        real_expert_weights = topk_weights.view(num_tokens, self.topk)[real_mask]

        # sort by expert for vectorized computation
        sorted_indices = real_expert_indices.argsort()
        sorted_token_indices = real_token_indices[sorted_indices]
        sorted_expert_weights = real_expert_weights[sorted_indices]
        sorted_x = x.view(num_tokens, hidden_size)[sorted_token_indices]

        expert_counts = torch.bincount(real_expert_indices, minlength=self.num_experts)
        expert_offsets = expert_counts.cumsum(0)

        # process each real expert's chunk
        num_real_assignments = sorted_token_indices.size(0)
        sorted_out = torch.empty(
            num_real_assignments, hidden_size, device=device, dtype=dtype
        )

        start = 0
        for e in range(self.num_experts):
            end = expert_offsets[e].item()
            if end > start:
                chunk_x = sorted_x[start:end]
                h = F.silu(chunk_x @ self.w_gate[e]) * (chunk_x @ self.w_up[e])
                if self.training and self.dropout > 0:
                    h = F.dropout(h, p=self.dropout)
                sorted_out[start:end] = h @ self.w_down[e]
            start = end

        # scatter back (only real experts, null contributes nothing)
        weighted_out = sorted_out * sorted_expert_weights.unsqueeze(-1)
        routed_out = torch.zeros(num_tokens, hidden_size, device=device, dtype=dtype)
        routed_out.scatter_add_(
            0, sorted_token_indices.unsqueeze(-1).expand(-1, hidden_size), weighted_out
        )

        y = shared_out + routed_out.view(batch_size, seq_len, hidden_size)
        return y, aux_loss


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_heads=9,
        compression_ratio=8,
        intermediate_size=1536,
        num_experts=7,
        topk=2,
        data_sparsity=0.5,
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

        # MoE with null experts
        self.moe = MoEBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            topk=topk,
            data_sparsity=data_sparsity,
        )

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x, aux_loss = self.moe(x)
        x = residual + x

        return x, aux_loss


class DeepSeekIsh(nn.Module):
    def __init__(
        self,
        vocab_size=49152,
        hidden_size=768,
        num_hidden_layers=30,
        num_attention_heads=9,
        compression_ratio=8,
        intermediate_size=1536,
        num_experts=7,
        topk=2,
        data_sparsity=0.5,
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
                    num_experts=num_experts,
                    topk=topk,
                    data_sparsity=data_sparsity,
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
    ) -> tuple[Tensor, Tensor]:
        # input_ids shape: (batch_size, seq_len)

        x = self.embed_tokens(input_ids)  # (batch, seq_len, hidden_size)

        # accumulate aux losses from all moe layers
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # pass through the transformer blocks
        for layer in self.layers:
            x, aux_loss = layer(x, attention_mask)
            total_aux_loss = total_aux_loss + aux_loss

        # final norm
        x = self.norm(x)

        # project to vocabulary
        logits: Tensor
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        return logits, total_aux_loss

    def _init_weights(self, module):
        std = 1.0 / 24.0  # 0.041666666666666664
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
